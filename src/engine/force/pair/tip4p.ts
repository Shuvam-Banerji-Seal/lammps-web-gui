import { StyleError, type BondedEquilibria, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairLJCut } from './lj_cut';
import { inumeric, parseNum } from '../util';
import { erfcFast, erfcPoly, EWALD_F, ErfcTableCache, type ErfcTable } from '../erfc';
import { buildOwnedSites, tip4pAlpha, type OwnedSites, type Tip4pModel } from './tip4p_sites';
import { libm } from '../../libm';

/*
 * pair_style lj/cut/tip4p/cut, lj/cut/tip4p/long, tip4p/cut and tip4p/long.
 *
 * docs.lammps.org/pair_lj_cut_tip4p.html: "pair_style style args" with
 * "*lj/cut/tip4p/cut* args = otype htype btype atype qdist cutoff (cutoff2)"
 * and "otype,htype = atom types (numeric or type label) for TIP4P O and H",
 * "btype,atype = bond and angle types (numeric or type label) for TIP4P waters",
 * "qdist = distance from O atom to massless charge (distance units)",
 * "cutoff = global cutoff for LJ (and Coulombic if only 1 arg) (distance
 * units)", "cutoff2 = global cutoff for Coulombic (optional) (distance units)".
 * docs.lammps.org/pair_coul.html gives "pair_style tip4p/cut otype htype btype atype qdist cutoff"
 * for the Coulomb-only styles (no cutoff2, no LJ coefficients).
 * "If one cutoff is specified in the pair_style command, it is used for
 * both the LJ and Coulombic terms.  If two cutoffs are specified, they are
 * used as cutoffs for the LJ and Coulombic terms respectively."
 * "Note that the neighbor list cutoff for Coulomb interactions is
 * effectively extended by a distance" of 2 qdist "when using the TIP4P pair
 * style, to account for the offset distance of the fictitious charges on O
 * atoms in water molecules."
 * "For *lj/cut/tip4p/cut* and *lj/cut/tip4p/long* only the LJ cutoff can be
 * specified since a Coulombic cutoff cannot be specified for an individual I,J
 * type pair."
 *
 * Coulomb: the charge of O sits on M (site geometry in tip4p_sites.ts); all
 * Coulomb pairs use the site positions, the LJ pairs use the atom positions.
 * Measured with native LAMMPS (black box): the Coulomb cutoff is tested on the
 * M-to-site distance (an ion 5.1 A from O, 4.95 A from M, with cutoff2 5.0 gave
 * the Coulomb energy C qO qI / 4.95; an ion at 5.2 A gave 0), and the 1-2
 * (O-H) and 1-3 (H-H) special_bonds weights apply to the M-H1, M-H2 and H-H
 * pairs of a water (special_bonds coul 0.2 0.3 0.5 gave -64.44349 kcal/mol,
 * as the weights on M-H and H-H pairs predict).
 * The long-range Coulomb term is the coul/long damped form: docs.lammps.org/pair_coul.html
 * "Styles *coul/long* and *coul/msm* compute the same Coulombic interactions as
 * style *coul/cut* except that an additional damping factor is applied so it
 * can be used in conjunction with the :doc:`kspace_style <kspace_style>` command
 * and its *ewald* or *pppm* option." The real-space kernel is copied from coul_long.ts (its helper
 * is not exported): erfc(g r) / r with the special-bond correction, and the
 * pair_modify table N erfc tables (erfc.ts).
 * Forces on M are projected on O and the two H atoms as tip4p_sites.ts
 * describes; the virial is tallied from the site displacements (the atom
 * dot-product shortcut would use the atom positions, not the sites).
 */


/** Real-space long-range Coulomb kernel, as coul_long.ts coulLongPair (qq = C qi qj). */
const coulLong = (rsq: number, qq: number, g: number, fc: number, poly: boolean, table: ErfcTable | null): { e: number; f: number } => {
  if (table && rsq > table.inner) {
    let forcecoul = qq * table.force(rsq);
    let e = qq * table.energy(rsq);
    if (fc < 1) {
      const bare = qq * table.coul(rsq);
      forcecoul -= (1 - fc) * bare;
      e -= (1 - fc) * bare;
    }
    return { e, f: forcecoul / rsq };
  }
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const ex = libm.exp(-grij * grij);
  const erfc = poly || table ? erfcPoly(grij, ex) : erfcFast(grij, ex);
  const pre = qq / r;
  let forcecoul = pre * (erfc + EWALD_F * grij * ex);
  let e = pre * erfc;
  if (fc < 1) {
    forcecoul -= (1 - fc) * pre;
    e -= (1 - fc) * pre;
  }
  return { e, f: forcecoul / rsq };
};

/**
 * The O and H atom types and the bond and angle types of the otype htype btype atype words of pair_style, read at each
 * pair_coeff as native LAMMPS does (src/MOLECULE/pair_lj_cut_tip4p_cut.cpp coeff at b332f28, "set atom types from
 * pair_style command"; utils::expand_type_int with verify, src/utils.cpp): a type label becomes its number, then the
 * integer check and the range check ("Atom type 5 is out of bounds (1-2)"). The pair_style command keeps the words.
 */
export const tip4pTypes = (words: readonly string[], ctx: StyleContext | undefined): [number, number, number, number] => {
  const s = ctx?.s;
  const kinds = ['atom', 'atom', 'bond', 'angle'] as const;
  const label = { atom: 'Atom', bond: 'Bond', angle: 'Angle' };
  const nmax = { atom: s?.ntypes ?? 0, bond: s?.topo.nbondtypes ?? 0, angle: s?.topo.nangletypes ?? 0 };
  const out = kinds.map((kind, i) => {
    const t = inumeric(ctx?.typeWord ? ctx.typeWord(kind, words[i]) : words[i]);
    if (t <= 0 || t > nmax[kind]) throw new StyleError(`${label[kind]} type ${t} is out of bounds (1-${nmax[kind]})`);
    return t;
  });
  return [out[0], out[1], out[2], out[3]];
};

export class PairTIP4PBase extends PairLJCut {
  /** Native single_enable = 0 (src/KSPACE/pair_tip4p_long.cpp and the other tip4p styles at b332f28): compute pair/local refuses it. */
  protected override singleOff = true;
  /** Set by the subclasses: LJ term present, long-range (kspace) Coulomb. */
  hasLJ = true;
  coulLong = false;
  virialFdotr = false;
  readonly name: string = 'lj/cut/tip4p/cut';
  otype = 0;
  htype = 0;
  btype = 0;
  atype = 0;
  /** The otype htype btype atype words of pair_style, read at pair_coeff (tip4pTypes). */
  typeWords: string[] = [];
  qdist = 0;
  cutCoul = 0;
  /**
   * cut_coulsq and cut_coulsqplus = (cut_coul + 2 qdist)^2, the Coulomb cutoff of the M sites and of the atoms. As native
   * LAMMPS does (src/MOLECULE/pair_lj_cut_tip4p_cut.cpp and pair_tip4p_cut.cpp settings, src/KSPACE/pair_coul_long.cpp
   * init_style, pair_lj_cut_tip4p_long.cpp and pair_tip4p_long.cpp compute at b332f28): the cut styles set both in
   * settings; the long ones set cut_coulsq at init and cut_coulsqplus in each force computation from cut_coul and qdist
   * as they are then. fix adapt may change cut_coul and qdist in between (alpha follows qdist at the next init).
   */
  protected coulSq = 0;
  protected coulSqPlus = 0;
  private alpha = Number.NaN;
  private bonded: BondedEquilibria | null = null;
  private readonly erfcTables = new ErfcTableCache();

  /** Force-field hook (forcefield.ts init): equilibrium bond and angle lookup. */
  linkBonded(link: BondedEquilibria): void { this.bonded = link; }

  /** The M-site model for kspace_style pppm/tip4p (valid after init). */
  siteModel(): Tip4pModel | null {
    if (!Number.isFinite(this.alpha)) return null;
    return { otype: this.otype, htype: this.htype, alpha: this.alpha };
  }

  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.coulSq, this.tabinnerRsq); }

  /** Parses otype htype btype atype qdist cutoff [cutoff2]; the Coulomb-only styles take no cutoff2. */
  settings(args: string[]): void {
    const counts = this.hasLJ ? [6, 7] : [6];
    if (!counts.includes(args.length)) throw new StyleError(`usage: pair_style ${this.name} otype htype btype atype qdist cutoff${this.hasLJ ? ' (cutoff2)' : ''}`);
    this.typeWords = args.slice(0, 4);
    this.qdist = parseNum(args[4], 'qdist');
    const cut = parseNum(args[5], 'cutoff');
    if (!(this.qdist > 0)) throw new StyleError('qdist must be > 0');
    if (!(cut > 0)) throw new StyleError('cutoffs must be > 0');
    if (args.length === 7) {
      this.cutCoul = parseNum(args[6], 'Coulomb cutoff');
      this.cutGlobal = cut;
    } else {
      this.cutCoul = cut;
      this.cutGlobal = cut;
    }
    if (!(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
    if (!this.coulLong) {
      this.coulSq = this.cutCoul * this.cutCoul;
      this.coulSqPlus = (this.cutCoul + 2 * this.qdist) ** 2;
    }
  }

  /** The atom, bond and angle types of the pair_style words (tip4pTypes), at each pair_coeff. */
  protected setTypes(ctx: StyleContext | undefined): void {
    // a style read from a native restart file has its types and no words (native: "unless we were restarted")
    if (this.typeWords.length === 0) return;
    [this.otype, this.htype, this.btype, this.atype] = tip4pTypes(this.typeWords, ctx);
    if (this.otype === this.htype) throw new StyleError(`pair_style ${this.name}: O and H need different atom types`);
  }

  /**
   * pair_coeff I J epsilon sigma [cutoff] (LJ); the Coulomb-only styles take "pair_coeff * *". As native LAMMPS does
   * (src/MOLECULE/pair_tip4p_cut.cpp coeff at b332f28), tip4p/cut checks the word count before it reads the types.
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (!this.hasLJ) {
      const usage = () => pairCoeffArgs(`usage: pair_coeff I J (${this.name} takes no coefficients)`);
      if (!this.coulLong && args.length !== 2) throw usage();
      this.setTypes(ctx);
      if (args.length !== 2) throw usage();
      return;
    }
    this.setTypes(ctx);
    super.coeff(args);
  }

  /** Equilibrium OH length and HOH angle come from the bond and angle styles (forcefield.ts hook). */
  initStyle(): void {
    if (!this.bonded) throw new StyleError(`pair style ${this.name}: the bond and angle equilibria are not available (force field hook)`);
    const r0 = this.bonded.bond(this.btype), th0 = this.bonded.angle(this.atype);
    if (!Number.isFinite(r0)) throw new StyleError(`pair style ${this.name}: bond type ${this.btype} has no equilibrium length (bond_style and bond_coeff needed)`);
    if (!Number.isFinite(th0)) throw new StyleError(`pair style ${this.name}: angle type ${this.atype} has no equilibrium angle (angle_style and angle_coeff needed)`);
    this.alpha = tip4pAlpha(this.qdist, r0, th0);
    if (this.coulLong) this.coulSq = this.cutCoul * this.cutCoul;
  }

  initOne(i: number, j: number): number {
    // the neighbor cutoff covers the M sites: |x_i - x_j| <= cutCoul + 2 qdist for an M-M pair
    const coulCut = this.cutCoul + 2 * this.qdist;
    if (!this.hasLJ) return coulCut;
    return Math.max(super.initOne(i, j), coulCut);
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }

  compute(pc: PairCompute): void {
    const model = this.siteModel();
    if (!model) throw new StyleError(`pair style ${this.name} is not initialized`);
    const list = pc.half!;
    const { x, f, type, q, geom, nb } = pc;
    const nlocal = pc.nlocal;
    const sites: OwnedSites = buildOwnedSites(nlocal, x, type, pc.s.id, geom, model);
    const M = sites.M, h1 = sites.h1, h2 = sites.h2;
    const owner = nb.owner;
    const { otype, alpha } = model;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const g = this.gEwald;
    const tab = this.coulLong ? this.erfcTable(g) : null;
    const cutCsq = this.coulSq;
    const cutCsqPlus = this.coulLong ? (this.cutCoul + 2 * this.qdist) ** 2 : this.coulSqPlus;
    const cutljsq = new Float64Array(nt * nt);
    if (this.hasLJ) for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const v = pc.acc.virial;
    const isO = (k: number) => type[k] === otype;
    // owned index an atom (owned or ghost) is an image of
    const own = (k: number) => (k < nlocal ? k : owner[k]);
    // position of the Coulomb site of atom k (M for O, the atom otherwise); ghosts shift their owner's site
    const site = (k: number, out: number[]) => {
      if (k < nlocal) {
        for (let c = 0; c < 3; c++) out[c] = isO(k) ? M[3 * k + c] : x[3 * k + c];
      } else if (isO(k)) {
        const o = owner[k];
        for (let c = 0; c < 3; c++) out[c] = M[3 * o + c] + (x[3 * k + c] - x[3 * o + c]);
      } else {
        for (let c = 0; c < 3; c++) out[c] = x[3 * k + c];
      }
      return out;
    };
    // Coulomb force F on the site of atom k: distributed to the atom, or to O, H1, H2 for an M site
    const apply = (k: number, Fx: number, Fy: number, Fz: number) => {
      if (isO(k)) {
        const o = own(k), a = h1[o], b = h2[o];
        const wo = 1 - alpha, wh = 0.5 * alpha;
        f[3 * o] += wo * Fx; f[3 * o + 1] += wo * Fy; f[3 * o + 2] += wo * Fz;
        f[3 * a] += wh * Fx; f[3 * a + 1] += wh * Fy; f[3 * a + 2] += wh * Fz;
        f[3 * b] += wh * Fx; f[3 * b + 1] += wh * Fy; f[3 * b + 2] += wh * Fz;
      } else {
        const t = own(k);
        f[3 * t] += Fx; f[3 * t + 1] += Fy; f[3 * t + 2] += Fz;
      }
    };
    // special_bonds apply only to the bonded pair itself: a pair whose raw separation exceeds half the box
    // along a periodic axis has weight 1. Measured with native LAMMPS (black box): two waters in a 9.3 A
    // periodic box gave ecoul -11.8998750 kcal/mol only when the water's own periodic image pairs (M to
    // H images 8.4 A apart) counted with weight 1; excluding them gave 10.867027.
    const per = pc.s.box.periodic;
    const halfL = [geom.lx / 2, geom.ly / 2, geom.lz / 2];
    const imageSep = (dx: number, dy: number, dz: number) =>
      (per[0] && Math.abs(dx) > halfL[0]) || (per[1] && Math.abs(dy) > halfL[1]) || (per[2] && Math.abs(dz) > halfL[2]);
    // per-atom energy and virial, as native LAMMPS does (src/pair.cpp at b332f28): the LJ term through ev_tally, half to
    // each atom; the Coulomb term through ev_tally_tip4p, half to each side, an M side's half split (1 - alpha) to the O
    // and alpha / 2 to each H. Ghost entries are folded onto their owners here, as native's reverse communication does.
    const { eatom, vatom } = pc;
    const tallyAtom = (k: number, w: number, e: number, w6: number[]) => {
      if (eatom) eatom[k] += w * e;
      if (vatom) for (let c = 0; c < 6; c++) vatom[6 * k + c] += w * w6[c];
    };
    const tallySide = (k: number, e: number, w6: number[]) => {
      if (isO(k)) {
        const o = own(k);
        tallyAtom(o, 0.5 * (1 - alpha), e, w6);
        tallyAtom(h1[o], 0.25 * alpha, e, w6);
        tallyAtom(h2[o], 0.25 * alpha, e, w6);
      } else tallyAtom(own(k), 0.5, e, w6);
    };
    const peratom = eatom !== null || vatom !== null;
    const w6 = [0, 0, 0, 0, 0, 0];
    const si = [0, 0, 0], sj = [0, 0, 0];
    let evdwl = 0, ecoul = 0;
    const nb0 = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      site(i, si);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb0[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const sb = imageSep(dx, dy, dz) ? 0 : jj >>> SBBITS;
        const rsq = dx * dx + dy * dy + dz * dz;
        if (this.hasLJ) {
          const t = ti + type[j];
          if (rsq < cutljsq[t]) {
            const factor = sLJ[sb];
            const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
            const fpair = factor * r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv;
            fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
            f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
            const elj = factor * (r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t]);
            evdwl += elj;
            v[0] += dx * dx * fpair; v[1] += dy * dy * fpair; v[2] += dz * dz * fpair;
            v[3] += dx * dy * fpair; v[4] += dx * dz * fpair; v[5] += dy * dz * fpair;
            if (peratom) {
              w6[0] = dx * dx * fpair; w6[1] = dy * dy * fpair; w6[2] = dz * dz * fpair;
              w6[3] = dx * dy * fpair; w6[4] = dx * dz * fpair; w6[5] = dy * dz * fpair;
              tallyAtom(i, 0.5, elj, w6);
              tallyAtom(own(j), 0.5, elj, w6);
            }
          }
        }
        const qj = q[j];
        if (qi === 0 || qj === 0 || rsq >= cutCsqPlus) continue;
        site(j, sj);
        const ex = si[0] - sj[0], ey = si[1] - sj[1], ez = si[2] - sj[2];
        const rc = ex * ex + ey * ey + ez * ez;
        if (rc >= cutCsq) continue;
        const qq = qqrd2e * qi * qj;
        let r: { e: number; f: number };
        if (this.coulLong) r = coulLong(rc, qq, g, sC[sb], this.table === 0, tab);
        else {
          const fc = sC[sb];
          const e = (fc * qq) / Math.sqrt(rc);
          r = { e, f: e / rc };
        }
        ecoul += r.e;
        const Fx = ex * r.f, Fy = ey * r.f, Fz = ez * r.f;
        apply(i, Fx, Fy, Fz);
        apply(j, -Fx, -Fy, -Fz);
        v[0] += ex * Fx; v[1] += ey * Fy; v[2] += ez * Fz;
        v[3] += ex * Fy; v[4] += ex * Fz; v[5] += ey * Fz;
        if (peratom) {
          w6[0] = ex * Fx; w6[1] = ey * Fy; w6[2] = ez * Fz; w6[3] = ex * Fy; w6[4] = ex * Fz; w6[5] = ey * Fz;
          tallySide(i, r.e, w6);
          tallySide(j, r.e, w6);
        }
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}

/** lj/cut/tip4p/cut and lj/cut/tip4p/long (LJ on the O atoms, Coulomb on the M sites). */
export class PairLJCutTIP4PCut extends PairTIP4PBase {
  readonly name: string = 'lj/cut/tip4p/cut';

  /**
   * As native LAMMPS does (src/MOLECULE/pair_lj_cut_tip4p_cut.cpp init_one at b332f28): "Water H epsilon must be
   * 0.0 for pair style lj/cut/tip4p/cut" for a pair involving the water H type while its H-H epsilon is nonzero, also
   * when fix adapt or compute fep re-initializes the style.
   */
  initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const h = this.htype;
    if ((i === h || j === h) && this.p.get('epsilon', h, h) !== 0) {
      throw new StyleError('Water H epsilon must be 0.0 for pair style lj/cut/tip4p/cut');
    }
    return cut;
  }
}

export class PairLJCutTIP4PLong extends PairTIP4PBase {
  readonly name: string = 'lj/cut/tip4p/long';
  coulLong = true;

  /**
   * As native LAMMPS does (src/KSPACE/pair_lj_cut_tip4p_long.cpp init_one at b332f28): a pair that
   * involves the water H type is an error when this style's H-H epsilon is nonzero, "Water H epsilon
   * must be 0.0 for pair style lj/cut/tip4p/long". Measured with native LAMMPS (black box, 27 TIP4P
   * waters): with pair_coeff 2 2 0.1 1.0 on this style the run fails at init; with H-H on a lj/cut
   * sub-style of a hybrid (the H-H coefficient not on this style) it runs.
   */
  initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    const h = this.htype;
    if ((i === h || j === h) && this.p.get('epsilon', h, h) !== 0) {
      throw new StyleError('Water H epsilon must be 0.0 for pair style lj/cut/tip4p/long');
    }
    return cut;
  }
}

/** tip4p/cut and tip4p/long: Coulomb only (no LJ term, no pair coefficients). */
export class PairTIP4PCut extends PairTIP4PBase {
  readonly name: string = 'tip4p/cut';
  hasLJ = false;
}

export class PairTIP4PLong extends PairTIP4PBase {
  readonly name: string = 'tip4p/long';
  hasLJ = false;
  coulLong = true;
}
