/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/pair_lj_cut_tip4p_long_soft.cpp, pair_lj_cut_tip4p_long_soft.h, pair_tip4p_long_soft.cpp and
 * pair_tip4p_long_soft.h (classes PairLJCutTIP4PLongSoft and PairTIP4PLongSoft, styles lj/cut/tip4p/long/soft and
 * tip4p/long/soft) at commit b332f28, by the LAMMPS developers (FEP package, Copyright Sandia Corporation).
 * GNU General Public License. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style lj/cut/tip4p/long/soft otype htype btype atype qdist nlambda alpha_LJ alpha_C cut_LJ (cut_coul)
 * pair_style tip4p/long/soft otype htype btype atype qdist nlambda alpha_C cut_coul
 * pair_coeff I J epsilon sigma lambda [cut_LJ] (lj/cut/tip4p/long/soft) and pair_coeff I J lambda (tip4p/long/soft)
 * (docs.lammps.org/pair_lj_cut_tip4p_long.html and pair_fep_soft.html).
 *
 * The soft-core forms are those of fep_soft.ts (the same LJ and real-space Ewald functional forms, scaled by
 * lambda^nlambda); the TIP4P geometry is the one of pair_tip4p.ts: the Coulomb charge of O sits on the M site,
 * all Coulomb pairs use the sites, the LJ pairs use the atom positions, the M force is projected on O, H1 and H2
 * (tip4p_sites.ts). Source notes kept: the H atoms have zero LJ epsilon and no LJ term ("Water H epsilon must be
 * 0.0"); the H-containing pairs have no LJ cutoff; the Coulomb neighbor cutoff is cut_coul + 2 qdist.
 *
 * Per-atom energy and virial follow native ev_tally and ev_tally_tip4p (the Coulomb x (x) f virial split by side and site).
 */
import { PairParams, StyleError, mixDistance, mixEpsilon, type BondedEquilibria, type PairCompute, type StyleContext, pairCoeffArgs } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { erfcPoly, EWALD_F } from '../../engine/force/erfc';
import { PairTIP4PBase } from '../../engine/force/pair/tip4p';
import { buildOwnedSites, tip4pAlpha, type OwnedSites, type Tip4pModel } from '../../engine/force/pair/tip4p_sites';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

const VALUES = ['epsilon', 'sigma', 'lambda', 'cut'];

/*
 * The soft TIP4P styles derive from PairTIP4PBase so that pppm/tip4p accepts them as the M-site model (kspace
 * pppm_tip4p.ts checks instanceof). Their settings, coefficients and compute are their own.
 */
export class PairTIP4PSoftBase extends PairTIP4PBase {
  nlambda = 0;
  alphaLJ = 0;
  alphaC = 0;
  cutLJGlobal = 0;
  p!: PairParams;
  private siteAlpha = Number.NaN;
  private bondedLink: BondedEquilibria | null = null;
  // per type pair after init (ntypes+1)^2, both orders
  private epsT = new Float64Array(0);
  private sinv6 = new Float64Array(0);
  private lamN = new Float64Array(0);
  private aL = new Float64Array(0);
  private aC = new Float64Array(0);
  private offT = new Float64Array(0);
  private cutLJsq = new Float64Array(0);
  private cutCsq = new Float64Array(0);

  /** Force-field hook (forcefield.ts init): equilibrium bond and angle lookup. */
  linkBonded(link: BondedEquilibria): void {
    super.linkBonded(link);
    this.bondedLink = link;
  }

  /** The M-site model for kspace_style pppm/tip4p (valid after init), as the core TIP4P styles give. */
  siteModel(): Tip4pModel | null {
    if (!Number.isFinite(this.siteAlpha)) return null;
    return { otype: this.otype, htype: this.htype, alpha: this.siteAlpha };
  }

  settings(args: string[]): void {
    const usage = this.hasLJ
      ? 'usage: pair_style lj/cut/tip4p/long/soft otype htype btype atype qdist nlambda alpha_LJ alpha_C cut_LJ (cut_coul)'
      : 'usage: pair_style tip4p/long/soft otype htype btype atype qdist nlambda alpha_C cut_coul';
    if (this.hasLJ ? args.length !== 9 && args.length !== 10 : args.length !== 8) throw new StyleError(usage);
    // the type words are read at pair_coeff (coeff: "set atom types from pair_style command")
    this.typeWords = args.slice(0, 4);
    this.qdist = parseNum(args[4], 'qdist');
    this.nlambda = parseNum(args[5], 'nlambda');
    if (this.hasLJ) {
      this.alphaLJ = parseNum(args[6], 'alpha_LJ');
      this.alphaC = parseNum(args[7], 'alpha_C');
      this.cutLJGlobal = parseNum(args[8], 'cut_LJ');
      this.cutCoul = args.length === 10 ? parseNum(args[9], 'cut_coul') : this.cutLJGlobal;
    } else {
      this.alphaC = parseNum(args[6], 'alpha_C');
      this.cutCoul = parseNum(args[7], 'cut_coul');
    }
    if (!(this.qdist > 0)) throw new StyleError('qdist must be > 0');
    if (!(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, VALUES);
  }

  /** The cut_coul the kspace style reads (pair extract). */
  extract(name: string): unknown {
    // pair_lj_cut_tip4p_long_soft.cpp extract: "epsilon", "sigma", "lambda" (dim 2) besides "cut_coul"
    if (name === 'epsilon' || name === 'sigma' || name === 'lambda') return this.p.p(name);
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }

  coeff(args: string[], ctx?: StyleContext): void {
    this.setTypes(ctx);
    if (this.hasLJ) {
      if (args.length < 5 || args.length > 6) throw pairCoeffArgs(`usage: pair_coeff I J epsilon sigma lambda [cutoff] for pair style ${this.name}`);
      const cut = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutLJGlobal;
      this.p.setRange(args[0], args[1], [parseNum(args[2], 'epsilon'), parseNum(args[3], 'sigma'), parseNum(args[4], 'lambda'), cut]);
      return;
    }
    if (args.length !== 3) throw pairCoeffArgs(`usage: pair_coeff I J lambda for pair style ${this.name}`);
    this.p.setRange(args[0], args[1], [0, 0, parseNum(args[2], 'lambda'), this.cutCoul]);
  }

  initStyle(): void {
    if (!this.bondedLink) throw new StyleError(`pair style ${this.name}: the bond and angle equilibria are not available (force field hook)`);
    const r0 = this.bondedLink.bond(this.btype), th0 = this.bondedLink.angle(this.atype);
    if (!Number.isFinite(r0)) throw new StyleError(`pair style ${this.name}: bond type ${this.btype} has no equilibrium length (bond_style and bond_coeff needed)`);
    if (!Number.isFinite(th0)) throw new StyleError(`pair style ${this.name}: angle type ${this.atype} has no equilibrium angle (angle_style and angle_coeff needed)`);
    this.siteAlpha = tip4pAlpha(this.qdist, r0, th0);
    // cut_coulsq of this init (src/FEP/pair_coul_long_soft.cpp init_style); cut_coulsqplus is the compute's (see there)
    this.coulSq = this.cutCoul * this.cutCoul;
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    const coulCut = this.cutCoul + 2 * this.qdist;
    if (this.hasLJ && (i === this.htype || j === this.htype) && p.get('epsilon', this.htype, this.htype) !== 0) {
      throw new StyleError(`Water H epsilon must be 0.0 for pair style ${this.name}`);
    }
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}, pair style ${this.name})`);
      if (p.get('lambda', i, i) !== p.get('lambda', j, j)) throw new StyleError(`pair ${this.name}: types ${i} and ${j} have different lambda values, so their cross term cannot be mixed (set pair_coeff ${i} ${j} explicitly)`);
      if (this.hasLJ) {
        p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
        p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
        p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
      }
      p.setMixed(i, j, 'lambda', p.get('lambda', i, i));
    }
    const nt = this.ntypes + 1;
    if (this.lamN.length !== nt * nt) {
      const z = () => new Float64Array(nt * nt);
      this.epsT = z(); this.sinv6 = z(); this.lamN = z(); this.aL = z(); this.aC = z(); this.offT = z(); this.cutLJsq = z(); this.cutCsq = z();
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), lam = p.get('lambda', i, j);
    // "Water H epsilon must be 0.0 ... set LJ cutoff to 0.0 for any interaction involving water H"
    const isH = i === this.htype || j === this.htype;
    const cutLJ = this.hasLJ && !isH ? p.get('cut', i, j) : 0;
    if (this.hasLJ && !isH && !(cutLJ > 0)) throw new StyleError(`pair ${this.name}: LJ cutoff for types ${i} ${j} must be > 0`);
    const pl = libmPow(lam, this.nlambda);
    const oml2 = (1 - lam) * (1 - lam);
    const aL = this.alphaLJ * oml2, aC = this.alphaC * oml2;
    const sinv6 = sig > 0 ? 1 / libmPow(sig, 6) : 0; // the tip4p/long/soft style has no LJ term (sigma unset)
    const k1 = i * nt + j, k2 = j * nt + i;
    let off = 0;
    if (this.hasLJ && this.shift && cutLJ > 0) {
      const x = libmPow(cutLJ, 6) * sinv6, D = aL + x, iD = 1 / D;
      off = pl * 4 * eps * (iD * iD - iD);
    }
    for (const k of [k1, k2]) {
      this.epsT[k] = eps; this.sinv6[k] = sinv6; this.lamN[k] = pl; this.aL[k] = aL; this.aC[k] = aC;
      this.offT[k] = off; this.cutLJsq[k] = cutLJ * cutLJ; this.cutCsq[k] = this.coulSq;
    }
    return this.hasLJ ? Math.max(cutLJ, coulCut) : coulCut;
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
    const gv = pc.acc.virial;
    let v = gv;
    const isO = (k: number) => type[k] === otype;
    const own = (k: number) => (k < nlocal ? k : owner[k]);
    // Coulomb site of atom k: M for O, the atom otherwise; ghosts shift their owner's site
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
    // the Coulomb force on the site of atom k goes to the atom, or to O, H1 and H2 for an M site
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
    // Virial of the Coulomb forces, as the source tallies it: x_p (x) f_p over the atoms that carry the force
    // (the atom positions, not the site displacement; for an M site the O and the two H atoms of its water, the H
    // images moved with a ghost O). Measured with native LAMMPS (black box): the pressure of the lj/cut/tip4p/long/soft
    // fixture differs from the site-displacement form at 1e-8.
    const addVirial = (k: number, Fx: number, Fy: number, Fz: number) => {
      if (!isO(k)) {
        const xk = x[3 * k], yk = x[3 * k + 1], zk = x[3 * k + 2];
        v[0] += xk * Fx; v[1] += yk * Fy; v[2] += zk * Fz;
        v[3] += xk * Fy; v[4] += xk * Fz; v[5] += yk * Fz;
        return;
      }
      const o = own(k), a = h1[o], b = h2[o];
      // a ghost O brings its H atoms as images with the same displacement
      const sh = k < nlocal ? [0, 0, 0] : [x[3 * k] - x[3 * o], x[3 * k + 1] - x[3 * o + 1], x[3 * k + 2] - x[3 * o + 2]];
      const wo = 1 - alpha, wh = 0.5 * alpha;
      const pts: Array<[number, number, number, number]> = [
        [x[3 * k], x[3 * k + 1], x[3 * k + 2], wo],
        [x[3 * a] + sh[0], x[3 * a + 1] + sh[1], x[3 * a + 2] + sh[2], wh],
        [x[3 * b] + sh[0], x[3 * b + 1] + sh[1], x[3 * b + 2] + sh[2], wh],
      ];
      for (const [px, py, pz, w] of pts) {
        const fx = w * Fx, fy = w * Fy, fz = w * Fz;
        v[0] += px * fx; v[1] += py * fy; v[2] += pz * fz;
        v[3] += px * fy; v[4] += px * fz; v[5] += py * fz;
      }
    };
    // special_bonds apply only to the bonded pair itself: a pair whose raw separation exceeds half the box
    // along a periodic axis has weight 1 (as pair_tip4p.ts, measured with native LAMMPS)
    const per = pc.s.box.periodic;
    const halfL = [geom.lx / 2, geom.ly / 2, geom.lz / 2];
    const imageSep = (dx: number, dy: number, dz: number) =>
      (per[0] && Math.abs(dx) > halfL[0]) || (per[1] && Math.abs(dy) > halfL[1]) || (per[2] && Math.abs(dz) > halfL[2]);
    // per-atom energy and virial (pair.cpp ev_tally and ev_tally_tip4p at b332f28): the LJ term half to each atom, the
    // Coulomb term and its x (x) f virial half to each side, an M side's half split (1 - alpha) to the O and alpha / 2
    // to each H; ghost entries fold onto their owners
    const { eatom, vatom } = pc;
    const peratom = eatom !== null || vatom !== null;
    const tallyAtom = (k: number, w: number, e: number, w6: ArrayLike<number>) => {
      if (eatom) eatom[k] += w * e;
      if (vatom) for (let c = 0; c < 6; c++) vatom[6 * k + c] += w * w6[c];
    };
    const tallySide = (k: number, e: number, w6: ArrayLike<number>) => {
      if (isO(k)) {
        const o = own(k);
        tallyAtom(o, 0.5 * (1 - alpha), e, w6);
        tallyAtom(h1[o], 0.25 * alpha, e, w6);
        tallyAtom(h2[o], 0.25 * alpha, e, w6);
      } else tallyAtom(own(k), 0.5, e, w6);
    };
    const w6 = new Float64Array(6);
    const si = [0, 0, 0], sj = [0, 0, 0];
    let evdwl = 0, ecoul = 0;
    // src/FEP/pair_lj_cut_tip4p_long_soft.cpp and pair_tip4p_long_soft.cpp compute: the atoms' Coulomb cutoff
    // cut_coulsqplus from cut_coul and qdist as they are now (fix adapt), the M sites' cut_coulsq from the run's init
    const cutCsqPlus = (this.cutCoul + 2 * this.qdist) ** 2;
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
        const t = ti + type[j];
        if (this.hasLJ && rsq < this.cutLJsq[t]) {
          // soft LJ (fep_soft.ts form): D = aL + (r/sigma)^6, E = lambda^n 4 eps (1/D^2 - 1/D)
          const fl = sLJ[sb];
          const xr = rsq * rsq * rsq * this.sinv6[t];
          const D = this.aL[t] + xr, iD = 1 / D;
          const pl = this.lamN[t], eps = this.epsT[t];
          const fpair = fl * pl * 24 * eps * xr * (2 * iD * iD * iD - iD * iD) / rsq;
          const e = fl * (pl * 4 * eps * (iD * iD - iD) - this.offT[t]);
          evdwl += e;
          fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
          f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
          v[0] += dx * dx * fpair; v[1] += dy * dy * fpair; v[2] += dz * dz * fpair;
          v[3] += dx * dy * fpair; v[4] += dx * dz * fpair; v[5] += dy * dz * fpair;
          if (peratom) {
            const l6 = [dx * dx * fpair, dy * dy * fpair, dz * dz * fpair, dx * dy * fpair, dx * dz * fpair, dy * dz * fpair];
            tallyAtom(i, 0.5, e, l6);
            tallyAtom(own(j), 0.5, e, l6);
          }
        }
        const qj = q[j];
        if (qi === 0 || qj === 0 || rsq >= cutCsqPlus) continue;
        site(j, sj);
        const ex = si[0] - sj[0], ey = si[1] - sj[1], ez = si[2] - sj[2];
        const rc = ex * ex + ey * ey + ez * ez;
        if (rc >= this.cutCsq[t]) continue;
        // soft real-space Ewald term: P erfc(g r) / r_s with r_s^2 = aC + r^2 (fep_soft.ts coulPair, long)
        const fc = sC[sb];
        const P = this.lamN[t] * qqrd2e * qi * qj;
        const rs2 = this.aC[t] + rc, rs = Math.sqrt(rs2);
        const r = Math.sqrt(rc);
        const gr = g * r;
        const expg = libmExp(-gr * gr);
        const corr = erfcPoly(gr, expg) - (1 - fc);
        const e = (P * corr) / rs;
        const fr = P * (EWALD_F * g * expg / (r * rs) + corr / (rs2 * rs));
        ecoul += e;
        const Fx = ex * fr, Fy = ey * fr, Fz = ez * fr;
        apply(i, Fx, Fy, Fz);
        apply(j, -Fx, -Fy, -Fz);
        if (peratom) {
          // the pair's virial alone, then added to the global one
          w6.fill(0);
          v = w6;
          addVirial(i, Fx, Fy, Fz);
          addVirial(j, -Fx, -Fy, -Fz);
          v = gv;
          for (let c = 0; c < 6; c++) gv[c] += w6[c];
          tallySide(i, e, w6);
          tallySide(j, e, w6);
        } else {
          addVirial(i, Fx, Fy, Fz);
          addVirial(j, -Fx, -Fy, -Fz);
        }
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}

/** lj/cut/tip4p/long/soft: soft LJ on the O atoms, soft real-space Coulomb on the M sites. */
export class PairLJCutTIP4PSoft extends PairTIP4PSoftBase {
  readonly name: string = 'lj/cut/tip4p/long/soft';
  hasLJ = true;
  coulLong = true;
}

/** tip4p/long/soft: soft Coulomb on the M sites only (no LJ term, no pair coefficients beyond lambda). */
export class PairTIP4PSoft extends PairTIP4PSoftBase {
  readonly name: string = 'tip4p/long/soft';
  hasLJ = false;
  coulLong = true;
}
