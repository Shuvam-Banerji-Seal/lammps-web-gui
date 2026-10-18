/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pair_lj_long_tip4p_long.cpp
 * and pair_lj_long_tip4p_long.h (PairLJLongTIP4PLong::settings, coeff, init_style, init_one, compute_newsite and the
 * non-rRESPA compute of the LJ and Coulomb real-space terms). Contributing authors of the LAMMPS file: Rolf Isele-Holder
 * (Aachen University), Axel Kohlmeyer (Temple University) and Paul Crozier (SNL). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too.
 *
 * docs.lammps.org/pair_lj_long.html: "pair_style lj/long/tip4p/long flag_lj flag_coul otype htype btype atype qdist
 * cutoff (cutoff2)" is the TIP4P variant of lj/long/coul/long: the Coulomb charge of each O sits on its M site and the
 * forces on M are projected on O (1 - alpha) and on each H (alpha / 2), as the native compute does. The Coulomb term
 * must be long (native: "Coulomb cut not supported in pair_style lj/long/tip4p/long"). The LJ term is either the
 * long-range dispersion real space (flag_lj long, with pair_modify table/disp) or a cut term (flag_lj cut); the M
 * sites take no LJ, the H types have no LJ (cut_ljsq = 0). The k-space side is kspace_style pppm/disp/tip4p
 * (src/engine-gpl/tip4pdisp/pppm_disp_tip4p.ts), which reads the M-site model of this style (siteModel).
 *
 * The C++ compute keeps the atom virial as the sum over atoms of x_k F_k with the forces on O, H1 and H2 (no
 * F-dot-r shortcut: native sets no_virial_fdotr_compute), so the virial is tallied here pair by pair from those
 * forces, as the native code does. Ghost atoms: the M site of a ghost O is its owner's M site shifted by the image
 * (native compute_newsite of the ghost's own H images gives the same), and the force on a ghost is applied to its owner.
 * Per-atom energy and virial are refused by name.
 */
import { StyleError, type BondedEquilibria, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairLJLongCoulLong, type DispTable } from '../../engine/force/pair/lj_long';
import { buildOwnedSites, tip4pAlpha, type Tip4pModel } from '../../engine/force/pair/tip4p_sites';
import { erfcPoly, EWALD_F, ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { exp } from '../libm/exp';
import { makeDispTable } from '../pair_disp/dispTable';

/** Coulomb real space of one pair at squared distance rsq (native compute, the direct and the table branches). */
const coulTerm = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, tab: ErfcTable | null,
): { e: number; cforce: number } => {
  const r2inv = 1.0 / rsq;
  if (tab !== null && rsq > tab.inner) {
    const qq = qqrd2e * qi * qj;
    let forcecoul = qq * tab.force(rsq);
    let ecoul = qq * tab.energy(rsq);
    if (fc < 1.0) {
      const bare = qq * tab.coul(rsq);
      forcecoul -= (1.0 - fc) * bare;
      ecoul -= (1.0 - fc) * bare;
    }
    return { e: ecoul, cforce: forcecoul * r2inv };
  }
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const expm2 = exp(-grij * grij);
  const erfc = erfcPoly(grij, expm2);
  const prefactor = qqrd2e * qi * qj / r;
  let forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
  let ecoul = prefactor * erfc;
  if (fc < 1.0) {
    forcecoul -= (1.0 - fc) * prefactor;
    ecoul -= (1.0 - fc) * prefactor;
  }
  return { e: ecoul, cforce: forcecoul * r2inv };
};

/**
 * The H atoms of each owned O (atom IDs O+1 and O+2), with the native errors of compute (pair_lj_long_tip4p_long.cpp at
 * b332f28: "TIP4P hydrogen is missing" and "TIP4P hydrogen has incorrect atom type"). Called before the M sites are built.
 */
export const checkTip4pHydrogens = (n: number, type: Int32Array, ids: Int32Array, otype: number, htype: number): void => {
  const byId = new Map<number, number>();
  for (let i = 0; i < n; i++) byId.set(ids[i], i);
  for (let o = 0; o < n; o++) {
    if (type[o] !== otype) continue;
    const a = byId.get(ids[o] + 1), b = byId.get(ids[o] + 2);
    if (a === undefined || b === undefined) throw new StyleError('TIP4P hydrogen is missing');
    if (type[a] !== htype || type[b] !== htype) throw new StyleError('TIP4P hydrogen has incorrect atom type');
  }
};

export class PairLJLongTIP4PLong extends PairLJLongCoulLong {
  override readonly name: string = 'lj/long/tip4p/long';
  /** The virial is tallied here pair by pair (PairLJCut sets fdotr; the TIP4P sites need this tally, as tip4p.ts does). */
  override virialFdotr = false;
  typeO = 0;
  typeH = 0;
  typeB = 0;
  typeA = 0;
  qdist = 0;
  private alpha = Number.NaN;
  private bonded: BondedEquilibria | null = null;
  private readonly coulTables = new ErfcTableCache();
  private ownDispTab: DispTable | null = null;
  private ownDispKey: number[] = [];

  /** Force-field hook (forcefield.ts init): equilibrium bond and angle lookup. */
  linkBonded(link: BondedEquilibria): void { this.bonded = link; }

  /** The M-site model for kspace_style pppm/disp/tip4p (valid after init_style). */
  siteModel(): Tip4pModel | null {
    if (!Number.isFinite(this.alpha)) return null;
    return { otype: this.typeO, htype: this.typeH, alpha: this.alpha };
  }

  /** pair_style lj/long/tip4p/long flag_lj flag_coul otype htype btype atype qdist cutoff (cutoff2). */
  settings(args: string[]): void {
    if (args.length !== 8 && args.length !== 9) throw new StyleError('Illegal pair_style command');
    if (args[1] !== 'long') throw new StyleError('Coulomb cut not supported in pair_style lj/long/tip4p/long');
    this.typeO = parseNum(args[2], 'otype');
    this.typeH = parseNum(args[3], 'htype');
    this.typeB = parseNum(args[4], 'btype');
    this.typeA = parseNum(args[5], 'atype');
    this.qdist = parseNum(args[6], 'qdist');
    super.settings([args[0], args[1], args[7], ...(args.length === 9 ? [args[8]] : [])]);
  }

  /** Equilibrium OH length and HOH angle come from the bond and angle styles (forcefield.ts hook). */
  initStyle(): void {
    // As native LAMMPS does (init_style at b332f28): the bond and angle styles are required.
    if (!this.bonded) throw new StyleError('Must use a bond style with TIP4P potential');
    const r0 = this.bonded.bond(this.typeB), th0 = this.bonded.angle(this.typeA);
    if (!Number.isFinite(r0)) throw new StyleError(`pair style ${this.name}: bond type ${this.typeB} has no equilibrium length (bond_style and bond_coeff needed)`);
    if (!Number.isFinite(th0)) throw new StyleError(`pair style ${this.name}: angle type ${this.typeA} has no equilibrium angle (angle_style and angle_coeff needed)`);
    this.alpha = tip4pAlpha(this.qdist, r0, th0);
  }

  initOne(i: number, j: number): number {
    const cut = super.initOne(i, j);
    // As native LAMMPS does (init_one at b332f28): the H type takes no LJ term
    if (i === this.typeH && this.p.get('epsilon', i, i) !== 0) {
      throw new StyleError('Water H epsilon must be 0.0 for pair style lj/long/tip4p/long');
    }
    // the neighbor cutoff covers the Coulomb M-site pairs: |x_i - x_j| <= cut_coul + 2 qdist
    return Math.max(cut, this.cutCoul + 2 * this.qdist);
  }

  extract(name: string): unknown {
    if (name === 'qdist') return this.qdist;
    return super.extract(name);
  }

  /** The Coulomb table of pair_modify table N (null for N = 0: the direct erfc polynomial). */
  private coulTable(g: number): ErfcTable | null {
    return this.coulTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq);
  }

  /** The dispersion table of pair_modify table/disp N > 0 (lj/long/coul/long helper, src/engine-gpl/pair_disp). */
  private dispTableFor(gd: number): DispTable | null {
    if (!this.ljLong || this.tableDisp <= 0) return null;
    const key = [this.tableDisp, gd, this.cutGlobal, this.tabinnerDisp];
    if (this.ownDispTab === null || key.some((v, i) => v !== this.ownDispKey[i])) {
      this.ownDispTab = makeDispTable(this.tableDisp, gd, this.cutGlobal, this.tabinnerDisp);
      this.ownDispKey = key;
    }
    return this.ownDispTab;
  }

  compute(pc: PairCompute): void {
    if (pc.eatom || pc.vatom) throw new StyleError(`per-atom energy and virial are not supported by pair style ${this.name}`);
    const model = this.siteModel();
    if (!model) throw new StyleError(`pair style ${this.name} is not initialized`);
    const list = pc.half!;
    const { x, f, type, q, geom, nb } = pc;
    const nlocal = pc.nlocal;
    checkTip4pHydrogens(nlocal, type, pc.s.id, model.otype, model.htype);
    const sites = buildOwnedSites(nlocal, x, type, pc.s.id, geom, model);
    const M = sites.M, h1 = sites.h1, h2 = sites.h2;
    const owner = nb.owner;
    const otype = model.otype, htype = model.htype, alpha = model.alpha;
    // H minus O for each owned O (minimum image): the closest H image of native closest_image
    const dH1 = new Float64Array(3 * nlocal), dH2 = new Float64Array(3 * nlocal);
    const d1 = [0, 0, 0], d2 = [0, 0, 0];
    for (let o = 0; o < nlocal; o++) {
      if (type[o] !== otype) continue;
      for (let c = 0; c < 3; c++) {
        d1[c] = x[3 * h1[o] + c] - x[3 * o + c];
        d2[c] = x[3 * h2[o] + c] - x[3 * o + c];
      }
      geom.minimumImage(d1);
      geom.minimumImage(d2);
      for (let c = 0; c < 3; c++) { dH1[3 * o + c] = d1[c]; dH2[3 * o + c] = d2[c]; }
    }
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const g = this.gEwald, gd = this.gEwaldDisp;
    const g2 = gd * gd, g6 = g2 * g2 * g2, g8 = g6 * g2;
    const ljLong = this.ljLong;
    const dtab = this.dispTableFor(gd);
    const tab = this.coulTable(g);
    const cutCoulsq = this.cutCoul * this.cutCoul;
    const cutCoulsqplus = (this.cutCoul + 2.0 * this.qdist) * (this.cutCoul + 2.0 * this.qdist);
    // cut_ljsq: the pair cutoffs squared, 0 for pairs with the H type (native init_one)
    const cutljsq = new Float64Array(nt * nt);
    for (let a = 1; a <= this.ntypes; a++) {
      for (let b = 1; b <= this.ntypes; b++) {
        if (a === htype || b === htype) continue;
        const c = this.p.get('cut', a, b);
        cutljsq[a * nt + b] = c * c;
      }
    }
    const lj1 = this.lj1, lj2 = this.lj2, lj3 = this.lj3, lj4 = this.lj4, offset = this.offset;
    const v = pc.acc.virial;
    const isO = (k: number) => type[k] === otype;
    const own = (k: number) => (k < nlocal ? k : owner[k]);
    // the Coulomb site of atom k (M for O, the atom otherwise); a ghost O is its owner's M shifted by the image
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
    // the virial of an O site: x_O fO + x_H1 fH + x_H2 fH per component (native ev tally of the O branch)
    const siteVirial = (xo: number[], xh1: number[], xh2: number[], fO: number[], fH: number[]): number[] => [
      xo[0] * fO[0] + xh1[0] * fH[0] + xh2[0] * fH[0],
      xo[1] * fO[1] + xh1[1] * fH[1] + xh2[1] * fH[1],
      xo[2] * fO[2] + xh1[2] * fH[2] + xh2[2] * fH[2],
      xo[0] * fO[1] + xh1[0] * fH[1] + xh2[0] * fH[1],
      xo[0] * fO[2] + xh1[0] * fH[2] + xh2[0] * fH[2],
      xo[1] * fO[2] + xh1[1] * fH[2] + xh2[1] * fH[2],
    ];
    // special_bonds apply only to the bonded pair itself: a pair whose raw separation exceeds half the box
    // along a periodic axis has weight 1 (the measured rule of tip4p.ts, shared with lj/cut/tip4p/long).
    const per = pc.s.box.periodic;
    const halfL = [geom.lx / 2, geom.ly / 2, geom.lz / 2];
    const imageSep = (dx: number, dy: number, dz: number) =>
      (per[0] && Math.abs(dx) > halfL[0]) || (per[1] && Math.abs(dy) > halfL[1]) || (per[2] && Math.abs(dz) > halfL[2]);
    const si = [0, 0, 0], sj = [0, 0, 0];
    const xo = [0, 0, 0], xh1 = [0, 0, 0], xh2 = [0, 0, 0];
    const fO = [0, 0, 0], fH = [0, 0, 0], fd = [0, 0, 0];
    let evdwl = 0, ecoul = 0;
    const nb0 = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      const iO = isO(i);
      let fxi = 0, fyi = 0, fzi = 0;
      site(i, si);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb0[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const sb = imageSep(dx, dy, dz) ? 0 : jj >>> SBBITS;
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        // the LJ term on the atoms (cut_ljsq as native: 0 for the H type)
        if (rsq < cutljsq[t]) {
          const r2inv = 1.0 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          let forcelj: number, ev: number;
          if (ljLong) {
            const r12inv = r6inv * r6inv;
            if (dtab === null || rsq <= dtab.inner) {
              const gr2 = g2 * rsq, a2 = 1.0 / gr2;
              const expterm = a2 * exp(-gr2) * lj4[t];
              const g6term = g6 * ((a2 + 1.0) * a2 + 0.5) * expterm;
              const g8term = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * expterm * rsq;
              if (sb === 0) {
                forcelj = r12inv * lj1[t] - g8term;
                ev = r12inv * lj3[t] - g6term;
              } else {
                const factor = sLJ[sb], tt = r6inv * (1.0 - factor);
                forcelj = factor * r12inv * lj1[t] - g8term + tt * lj2[t];
                ev = factor * r12inv * lj3[t] - g6term + tt * lj4[t];
              }
            } else {
              const ftable = dtab.force(rsq), etable = dtab.energy(rsq);
              if (sb === 0) {
                forcelj = r12inv * lj1[t] - ftable * lj4[t];
                ev = r12inv * lj3[t] - etable * lj4[t];
              } else {
                const factor = sLJ[sb], tt = r6inv * (1.0 - factor);
                forcelj = factor * r12inv * lj1[t] - ftable * lj4[t] + tt * lj2[t];
                ev = factor * r12inv * lj3[t] - etable * lj4[t] + tt * lj4[t];
              }
            }
          } else if (sb === 0) {
            forcelj = r6inv * (r6inv * lj1[t] - lj2[t]);
            ev = r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t];
          } else {
            const factor = sLJ[sb];
            forcelj = factor * r6inv * (r6inv * lj1[t] - lj2[t]);
            ev = factor * (r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t]);
          }
          forcelj *= r2inv;
          fxi += dx * forcelj; fyi += dy * forcelj; fzi += dz * forcelj;
          f[3 * j] -= dx * forcelj; f[3 * j + 1] -= dy * forcelj; f[3 * j + 2] -= dz * forcelj;
          evdwl += ev;
          v[0] += dx * dx * forcelj; v[1] += dy * dy * forcelj; v[2] += dz * dz * forcelj;
          v[3] += dx * dy * forcelj; v[4] += dx * dz * forcelj; v[5] += dy * dz * forcelj;
        }
        // the Coulomb term on the sites (the atom distance gates it, as native: rsq < cut_coulsqplus)
        if (rsq >= cutCoulsqplus) continue;
        const jO = isO(j);
        let rc = rsq;
        if (iO || jO) {
          site(j, sj);
          const ex = si[0] - sj[0], ey = si[1] - sj[1], ez = si[2] - sj[2];
          rc = ex * ex + ey * ey + ez * ez;
        }
        if (rc >= cutCoulsq) continue;
        // delx, dely, delz of native compute: the site difference (the atom difference when neither atom is O)
        const delx = iO || jO ? si[0] - sj[0] : dx;
        const dely = iO || jO ? si[1] - sj[1] : dy;
        const delz = iO || jO ? si[2] - sj[2] : dz;
        const fc = sC[sb];
        const r = coulTerm(rc, qi, q[j], g, qqrd2e, fc, tab);
        ecoul += r.e;
        const cforce = r.cforce;
        // the virial of the pair, tallied as native ev_tally_tip4p: the i part assigns, the j part adds or subtracts
        let vloc: number[];
        if (!iO) {
          vloc = [
            x[3 * i] * delx * cforce, x[3 * i + 1] * dely * cforce, x[3 * i + 2] * delz * cforce,
            x[3 * i] * dely * cforce, x[3 * i] * delz * cforce, x[3 * i + 1] * delz * cforce,
          ];
          fxi += delx * cforce; fyi += dely * cforce; fzi += delz * cforce;
        } else {
          fd[0] = delx * cforce; fd[1] = dely * cforce; fd[2] = delz * cforce;
          for (let c = 0; c < 3; c++) { fO[c] = fd[c] * (1 - alpha); fH[c] = 0.5 * alpha * fd[c]; }
          fxi += fO[0]; fyi += fO[1]; fzi += fO[2];
          const a = h1[i], b = h2[i];
          for (let c = 0; c < 3; c++) { f[3 * a + c] += fH[c]; f[3 * b + c] += fH[c]; }
          for (let c = 0; c < 3; c++) {
            xo[c] = x[3 * i + c];
            xh1[c] = xo[c] + dH1[3 * i + c];
            xh2[c] = xo[c] + dH2[3 * i + c];
          }
          vloc = siteVirial(xo, xh1, xh2, fO, fH);
        }
        if (!jO) {
          const tj = own(j);
          f[3 * tj] -= delx * cforce; f[3 * tj + 1] -= dely * cforce; f[3 * tj + 2] -= delz * cforce;
          vloc[0] -= x[3 * j] * delx * cforce;
          vloc[1] -= x[3 * j + 1] * dely * cforce;
          vloc[2] -= x[3 * j + 2] * delz * cforce;
          vloc[3] -= x[3 * j] * dely * cforce;
          vloc[4] -= x[3 * j] * delz * cforce;
          vloc[5] -= x[3 * j + 1] * delz * cforce;
        } else {
          const oj = own(j), aj = h1[oj], bj = h2[oj];
          fd[0] = -delx * cforce; fd[1] = -dely * cforce; fd[2] = -delz * cforce;
          for (let c = 0; c < 3; c++) { fO[c] = fd[c] * (1 - alpha); fH[c] = 0.5 * alpha * fd[c]; }
          for (let c = 0; c < 3; c++) {
            f[3 * oj + c] += fO[c];
            f[3 * aj + c] += fH[c];
            f[3 * bj + c] += fH[c];
          }
          for (let c = 0; c < 3; c++) {
            xo[c] = x[3 * j + c];
            xh1[c] = xo[c] + dH1[3 * oj + c];
            xh2[c] = xo[c] + dH2[3 * oj + c];
          }
          const vj = siteVirial(xo, xh1, xh2, fO, fH);
          for (let c = 0; c < 6; c++) vloc[c] += vj[c];
        }
        for (let c = 0; c < 6; c++) v[c] += vloc[c];
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }
}
