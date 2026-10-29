/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DRUDE/pair_lj_cut_thole_long.cpp and
 * src/DRUDE/pair_lj_cut_thole_long.h (class PairLJCutTholeLong, pair_style lj/cut/thole/long). Copyright (2003) Sandia
 * Corporation; GPL-2.0 (src/engine-gpl/LICENSE). DRUDE package authors: Alain Dequidt, Julien Devemy, Agilio Padua;
 * contributors Kateryna Goloviznina and Zheng Gong.
 *
 * pair_style lj/cut/thole/long thole cutoff_lj [cutoff_coul] (docs.lammps.org/pair_lj_cut_thole_long.html): Lennard-Jones
 * plus the real-space Ewald Coulomb term, and the Thole damped interaction of the induced dipoles of core-Drude pairs
 * (pair_thole.ts). "pair_coeff I J epsilon sigma polar [thole] [cutoff]". The direct (non-table) Coulomb term uses
 * the erf form for a special-bond pair (special weight below 1), the B-polynomial otherwise, and the table of
 * pair_modify table above the inner radius. The erf branch and the polynomial carry the constants of the native
 * branches at full precision (measured: with the 8-digit EWALD_F in the erf branch the forces of the bonded core-Drude
 * pairs were off by 9e-5, with the erfc fit of coul_long off by the fit's own error).
 */
import { PairLJCutCoulLong, coulLongPair } from '../../engine/force/pair/coul_long';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../../engine/force/types';
import { EWALD_F, ErfcTableCache, TABLE_INNER_RSQ, type ErfcTable } from '../../engine/force/erfc';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { CORE_TYPE, NOPOL_TYPE, drudeModelOf, type DrudeModel } from './drude_model';
import { bounds } from './pair_thole';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { erfc as libmErfc } from '../libm/erf';

/** LAMMPS EPSILON: added to rsq so that r = 0 between a Drude pair stays finite (the special bond removes it). */
const EPSILON = 1.0e-20;

/** 2/sqrt(pi) at full precision: the erf branch of the native direct term uses it (EWALD_F is the 8-digit value). */
const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/**
 * erf(x) for x >= 0 at full precision (the native branch calls the C library erf). Taylor series below 2, where the
 * terms do not cancel badly, and 1 - erfc above it, where erfc is small.
 */
const erfNative = (x: number): number => {
  if (x >= 2) return 1 - libmErfc(x);
  let term = x, sum = x;
  const x2 = x * x;
  for (let n = 1; n < 200; n++) {
    term *= -x2 / n;
    const t = term / (2 * n + 1);
    sum += t;
    if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
  }
  return TWO_OVER_SQRTPI * sum;
};

/** Ewald polynomial of the direct (non-table) branch: native's EWALD_P and B0..B5 fit. */
const EWALD_P = 9.95473818e-1;
const B0 = -0.1335096380159268, B1 = -2.57839507e-1, B2 = -1.37203639e-1, B3 = -8.88822059e-3;
const B4 = -5.80844129e-3, B5 = 1.14652755e-1;

/**
 * Real-space Coulomb term of one pair (the direct branch of native compute: the table, when on, is used only above
 * the inner radius). Direct, factor < 1: erfc = factor - erf(g r) with the force constant 2/sqrt(pi) at full precision;
 * direct, factor >= 1: the B-polynomial erfc and the 8-digit EWALD_F. Returns the energy and the force over r^2.
 */
const coulKernel = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, tab: ErfcTable | null,
): { e: number; f: number } => {
  if (tab !== null && rsq > TABLE_INNER_RSQ) return coulLongPair(rsq, qi, qj, g, qqrd2e, fc, false, tab);
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const expm2 = libmExp(-grij * grij);
  const pre = qqrd2e * qi * qj / r;
  if (fc < 1) {
    const erfc = fc - erfNative(grij);
    return { e: pre * erfc, f: pre * (erfc + TWO_OVER_SQRTPI * grij * expm2) / rsq };
  }
  const t = 1 / (1 + EWALD_P * grij);
  const u = 1 - t;
  const erfc = t * (1 + u * (B0 + u * (B1 + u * (B2 + u * (B3 + u * (B4 + u * B5)))))) * expm2;
  return { e: pre * erfc, f: pre * (erfc + EWALD_F * grij * expm2) / rsq };
};

export class PairLJCutTholeLong extends PairLJCutCoulLong {
  /** Native single_enable = 0 (src/DRUDE/pair_lj_cut_thole_long.cpp at b332f28): compute pair/local refuses it. */
  protected override singleOff = true;
  override readonly name: string = 'lj/cut/thole/long';
  /** Global Thole screening and per-type-pair values (polar, thole, ascreen = thole / polar^(1/3)). */
  private tholeGlobal = 0;
  private polar = new Float64Array(0);
  private thole = new Float64Array(0);
  private ascreen = new Float64Array(0);
  private tset = new Uint8Array(0);
  private model: DrudeModel | null = null;
  private readonly tables = new ErfcTableCache();

  /** The bonded core-Drude pairs stay in the list, with their special weights (special_bonds.html). */
  override keepExcluded = true;
  /** The pair forces are summed into f, so the virial is the dot product x . f (as pair_thole). */
  override virialFdotr = true;

  override settings(args: string[]): void {
    if (args.length < 2 || args.length > 3) throw new StyleError('Illegal pair_style command');
    this.tholeGlobal = parseNum(args[0], 'thole');
    super.settings(args.slice(1));
    const nt = this.ntypes + 1;
    if (this.tset.length === nt * nt) {
      for (let i = 1; i < nt; i++) {
        for (let j = i; j < nt; j++) {
          if (!this.tset[i * nt + j]) continue;
          this.thole[i * nt + j] = this.tholeGlobal;
        }
      }
    }
  }

  override allocate(ntypes: number): void {
    const nt = ntypes + 1;
    // the base allocate resets the pair parameters: it runs once (or again only when ntypes changes)
    if (this.tset.length === nt * nt) return;
    super.allocate(ntypes);
    this.tset = new Uint8Array(nt * nt);
    this.polar = new Float64Array(nt * nt);
    this.thole = new Float64Array(nt * nt);
    this.ascreen = new Float64Array(nt * nt);
  }

  /** pair_coeff I J epsilon sigma polar [thole] [cutoff]: the Lennard-Jones part goes to the lj/cut coefficients. */
  override coeff(args: string[]): void {
    if (args.length < 5 || args.length > 7) throw new StyleError('Incorrect args for pair coefficients');
    const ljArgs = args.slice(0, 4).concat(args.length === 7 ? [args[6]] : []);
    if (this.tset.length === 0) this.allocate(this.ntypes);
    super.coeff(ljArgs);
    const polarOne = parseNum(args[4], 'polar');
    const tholeOne = args.length >= 6 ? parseNum(args[5], 'thole') : this.tholeGlobal;
    const [ilo, ihi] = bounds(args[0], this.ntypes);
    const [jlo, jhi] = bounds(args[1], this.ntypes);
    const nt = this.ntypes + 1;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const k = i * nt + j;
        this.polar[k] = polarOne;
        this.thole[k] = tholeOne;
        this.ascreen[k] = tholeOne / libmPow(polarOne, 1 / 3);
        this.tset[k] = 1;
      }
    }
  }

  /**
   * fix adapt polar, thole and ascreen (native extract(), dim 2): the I <= J entry; init_one mirrors it and, for a pair
   * pair_coeff did not set, mixes polar and thole again and derives ascreen from them (as initOne does).
   */
  override adaptParam(name: string): AdaptAccess | null {
    const arr = name === 'polar' ? this.polar : name === 'thole' ? this.thole : name === 'ascreen' ? this.ascreen : null;
    if (!arr) return super.adaptParam(name);
    const nt = this.ntypes + 1;
    return { dim: 2, get: (i, j) => arr[i * nt + j], set: (i, j, v) => { arr[i * nt + j] = v; } };
  }

  override initStyle(ctx: StyleContext): void {
    this.model = drudeModelOf(ctx.s, 'Pair lj/cut/thole/long');
    super.initStyle(ctx);
  }

  override initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    const cut = super.initOne(i, j);
    if (!this.tset[k]) {
      const kii = i * nt + i, kjj = j * nt + j;
      this.polar[k] = Math.sqrt(this.polar[kii] * this.polar[kjj]);
      this.thole[k] = 0.5 * (this.thole[kii] + this.thole[kjj]);
      this.ascreen[k] = this.thole[k] / libmPow(this.polar[k], 1 / 3);
    }
    this.polar[kr] = this.polar[k];
    this.thole[kr] = this.thole[k];
    this.ascreen[kr] = this.ascreen[k];
    return cut;
  }

  override compute(pc: PairCompute): void {
    const model = this.model ?? drudeModelOf(pc.s, 'Pair lj/cut/thole/long');
    const s = pc.s;
    const partner = model.partnerIndex(s);
    const kindOf = new Uint8Array(s.n);
    for (let i = 0; i < s.n; i++) kindOf[i] = model.kind[s.type[i]];
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset, cutsq } = this;
    const cutljsq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) cutljsq[i * nt + j] = this.p.get('cut', i, j) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const tab = this.tables.get(this.table, g, cutcsq);
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const qqrd2e = pc.qqrd2e;
    const owner = pc.nb.owner;
    const list = pc.half!;
    const nb = list.neighbors;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const ki = kindOf[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      // the Drude partner of a polarizable atom: the core carries the charge -q of its Drude particle
      let di = -1, dqi = qi;
      if (ki !== NOPOL_TYPE) {
        di = partner[i];
        if (di < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[i]} is not defined`);
        dqi = ki === CORE_TYPE ? -q[di] : qi;
      }
      const pidi = di >= 0 ? s.id[di] : -1;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const factorLJ = sLJ[sb], factorCoul = sC[sb];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        let rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        rsq += EPSILON;
        const r2inv = 1 / rsq;
        let fpair = 0, e = 0;
        if (rsq < cutcsq) {
          const c = coulKernel(rsq, qi, q[j], g, qqrd2e, factorCoul, tab);
          fpair += c.f;
          e += c.e;
          ecoul += c.e;
          const oj = owner[j];
          const kj = kindOf[oj];
          if (ki !== NOPOL_TYPE && kj !== NOPOL_TYPE && s.id[oj] !== pidi) {
            let dqj = q[j];
            if (kj === CORE_TYPE) {
              const dj = partner[oj];
              if (dj < 0) throw new StyleError(`Drude atom for core atom ID ${s.id[oj]} is not defined`);
              dqj = -q[dj];
            }
            const r = Math.sqrt(rsq);
            const asr = this.ascreen[t] * r;
            const expAsr = libmExp(-asr);
            const dcoul = qqrd2e * dqi * dqj / r;
            const factorF = 0.5 * (2.0 + expAsr * (-2.0 - asr * (2.0 + asr))) - factorCoul;
            const factorE = 0.5 * (2.0 - expAsr * (2.0 + asr)) - factorCoul;
            fpair += factorF * dcoul * r2inv;
            e += factorE * dcoul;
            ecoul += factorE * dcoul;
          }
        }
        if (rsq < cutljsq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          fpair += factorLJ * r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
          const ev = factorLJ * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
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
}
