import { Pair, PairParams, StyleError, type PairCompute, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairLJCut, tallyAtom } from './lj_cut';
import { parseNum } from '../util';
import { erfcFast, erfcPoly, EWALD_F, ErfcTableCache, type ErfcTable } from '../erfc';
import { libm } from '../../libm';

/*
 * Real-space parts of Ewald/PPPM Coulombics.
 *
 * docs.lammps.org/pair_coul.html: "Styles coul/long and coul/msm compute the
 * same Coulombic interactions as style coul/cut except that an additional
 * damping factor is applied so it can be used in conjunction with the
 * kspace_style command and its ewald or pppm option. The Coulombic cutoff
 * specified for this style means that pairwise interactions within this
 * distance are computed directly; interactions outside that distance are
 * computed in reciprocal space." "For coul/cut/global, coul/long and coul/msm
 * no cutoff can be specified for an individual I,J type pair".
 * docs.lammps.org/pair_lj_cut_coul.html: "lj/cut/coul/long args = cutoff
 * (cutoff2)"; "cutoff = global cutoff for LJ (and Coulombic if only 1 arg)
 * (distance units)"; "cutoff2 = global cutoff for Coulombic (optional)
 * (distance units)"; "For lj/cut/coul/long
 * ... only the LJ cutoff can be specified".
 * The damped pair term is the standard Ewald real-space sum: E = C q_i q_j
 * erfc(g r)/r with C = qqr2e/dielectric and g the kspace G-ewald parameter.
 * special_bonds.html: excluded pairs are kept in the neighbor list for
 * kspace styles; their Coulomb weight w removes (1 - w) of the bare C q_i q_j
 * / r term, which the reciprocal sum includes.
 * pair_modify table N: with N = 0 the polynomial erfc fit LAMMPS documents
 * is used (erfc.ts erfcPoly; native LAMMPS agrees to ~1e-12); with N > 0
 * (default 12) the engine interpolates the same tables native builds
 * (erfc.ts makeErfcTable) for r^2 >= 2, which agrees with native to ~1e-14.
 */

/** Coulomb long-range real-space loop shared by both styles. */
export const coulLongPair = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, poly: boolean, table: ErfcTable | null = null,
  scale = 1,
): { e: number; f: number } => {
  if (table && rsq > table.inner) {
    // pair_modify table N > 0 (erfc.ts makeErfcTable): native interpolates erfc/r, the force kernel
    // and the bare 1/r of the special-bond correction from its tables
    const qq = qqrd2e * scale * qi * qj;
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
  // native's direct branch (table 0, and table N > 0 below r^2 = 2) is the polynomial fit; measured
  // with native LAMMPS (black box): table 0 and table 12 give the same energy at r = 1.2
  const erfc = poly || table ? erfcPoly(grij, ex) : erfcFast(grij, ex);
  // native: prefactor = qqrd2e * scale[itype][jtype] * qtmp * q[j] / r (scale 1 for the styles without one)
  const pre = qqrd2e * scale * qi * qj / r;
  let forcecoul = pre * (erfc + EWALD_F * grij * ex);
  let e = pre * erfc;
  if (fc < 1) {
    forcecoul -= (1 - fc) * pre;
    e -= (1 - fc) * pre;
  }
  return { e, f: forcecoul / rsq };
};

export class PairCoulLong extends Pair {
  readonly name: string = 'coul/long';
  virialFdotr = true;
  coulLong = true;
  cutCoul = 0;
  private readonly erfcTables = new ErfcTableCache();
  /** pair_modify table N (null for 0): erfc.ts makeErfcTable. */
  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq); }
  /** The pairs pair_coeff set, holding fix adapt's scale (native scale[i][j]; coeff sets 1, init_one mirrors it). */
  protected set!: PairParams;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style coul/long cutoff');
    this.cutCoul = parseNum(args[0], 'cutoff');
    if (!(this.cutCoul > 0)) throw new StyleError('coul/long cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.set = new PairParams(ntypes, ['scale']);
  }

  coeff(args: string[]): void {
    if (args.length !== 2) throw pairCoeffArgs('usage: pair_coeff I J (coul/long takes no coefficients)');
    this.set.setRange(args[0], args[1], [1]);
  }

  initOne(i: number, j: number): number {
    if (!this.set.isSet(i, j) && !(this.set.isSet(i, i) && this.set.isSet(j, j))) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    // As native LAMMPS does (src/KSPACE/pair_coul_long.cpp init_one at b332f28): fix adapt's scale is 1 for a pair
    // pair_coeff did not set (coeff sets 1 for its pairs), then mirrored
    const sc = this.set.p('scale');
    if (!this.set.isSet(i, j)) sc[this.set.idx(i, j)] = 1;
    sc[this.set.idx(j, i)] = sc[this.set.idx(i, j)];
    return this.cutCoul;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, q, type } = pc;
    const nt = this.ntypes + 1;
    const scale = this.set.p('scale');
    const cutsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTable(g);
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      if (qi === 0) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const qj = q[j];
        if (qj === 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq) continue;
        const r = coulLongPair(rsq, qi, qj, g, pc.qqrd2e, sC[jj >>> SBBITS], this.table === 0, tab, scale[type[i] * nt + type[j]]);
        fxi += dx * r.f; fyi += dy * r.f; fzi += dz * r.f;
        f[3 * j] -= dx * r.f; f[3 * j + 1] -= dy * r.f; f[3 * j + 2] -= dz * r.f;
        ecoul += r.e;
        if (tally) tallyAtom(pc, i, j, r.e, r.f, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

  // Measured with native LAMMPS (black box): write_data writes no Pair Coeffs or PairIJ Coeffs
  // section for coul/long (with or without pair ij).
  dataCoeffs(): string[] | null { return null; }

  extract(name: string): unknown {
    return name === 'cut_coul' ? this.cutCoul : undefined;
  }
}

export class PairLJCutCoulLong extends PairLJCut {
  readonly name: string = 'lj/cut/coul/long';
  coulLong = true;
  cutCoul = 0;
  /** Coulomb prefactor of the last force evaluation (single() reads it). */
  private qqrd2eLast = 1;
  private readonly erfcTables = new ErfcTableCache();
  /** pair_modify table N (null for 0): erfc.ts makeErfcTable. */
  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq); }
  /** The table values of one pair (ErfcTable.all): erfc/r, the force kernel and 1/r. */
  private readonly tabVals = new Float64Array(3);

  settings(args: string[]): void {
    if (args.length !== 1 && args.length !== 2) throw new StyleError('usage: pair_style lj/cut/coul/long cutoff (cutoff2)');
    this.cutGlobal = parseNum(args[0], 'LJ cutoff');
    this.cutCoul = args.length === 2 ? parseNum(args[1], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError('cutoffs must be > 0');
  }

  initOne(i: number, j: number): number {
    const cutLJ = super.initOne(i, j);
    return Math.max(cutLJ, this.cutCoul);
  }

  /**
   * Pair energy and force-over-r of one pair (single(), used by compute pair/local and hbond/local): the
   * LJ term inside its cutoff plus the real-space Coulomb term inside the Coulomb cutoff, with the given
   * special weights (the caller's; 1 for an unweighted pair).
   */
  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    let eng = 0, fforce = 0;
    if (rsq < this.p.get('cut', itype, jtype) ** 2) {
      const b = super.single(_i, _j, itype, jtype, rsq, factorCoul, factorLJ);
      eng = b.eng;
      fforce = b.fforce;
    }
    if (rsq < this.cutCoul * this.cutCoul && qi !== 0 && qj !== 0) {
      const r = coulLongPair(rsq, qi, qj, this.gEwald, this.qqrd2eLast, factorCoul, this.table === 0, this.erfcTable(this.gEwald));
      eng += r.e;
      fforce += r.f;
    }
    return { eng, fforce };
  }

  compute(pc: PairCompute): void {
    this.qqrd2eLast = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const cutljsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.erfcTable(g);
    const tabInner = tab ? tab.inner : Infinity;
    const tv = this.tabVals;
    const poly = this.table === 0;
    const qqrd2e = pc.qqrd2e;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        const qj = q[j];
        if (rsq < cutcsq && qi !== 0 && qj !== 0) {
          // coulLongPair written inline (the same arithmetic in the same order, without a result object per pair),
          // with the three table values from one lookup
          const fc = sC[sb];
          let cf: number, ce: number;
          if (rsq > tabInner && tab!.all(rsq, tv)) {
            const qq = qqrd2e * qi * qj;
            let forcecoul = qq * tv[1];
            ce = qq * tv[0];
            if (fc < 1) {
              const bare = qq * tv[2];
              forcecoul -= (1 - fc) * bare;
              ce -= (1 - fc) * bare;
            }
            cf = forcecoul / rsq;
          } else if (rsq > tabInner) {
            // past the table's last point: its own direct evaluation
            const r = coulLongPair(rsq, qi, qj, g, qqrd2e, fc, poly, tab);
            cf = r.f; ce = r.e;
          } else {
            const r = Math.sqrt(rsq);
            const grij = g * r;
            const ex = libm.exp(-grij * grij);
            const erfc = poly || tab ? erfcPoly(grij, ex) : erfcFast(grij, ex);
            const pre = qqrd2e * qi * qj / r;
            let forcecoul = pre * (erfc + EWALD_F * grij * ex);
            ce = pre * erfc;
            if (fc < 1) {
              forcecoul -= (1 - fc) * pre;
              ce -= (1 - fc) * pre;
            }
            cf = forcecoul / rsq;
          }
          fpair += cf;
          ecoul += ce;
          e += ce;
        }
        if (rsq < cutljsq[t]) {
          const factor = sLJ[sb];
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          fpair += factor * r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
          const ev = factor * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}
