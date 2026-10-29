import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { erfAcc } from './coul_cs';
import { fmtCoeff, parseNum } from '../util';
import { libm } from '../../libm';

/*
 * pair_style buck6d/coul/gauss/dsf and buck6d/coul/gauss/long (MOF-FF), docs.lammps.org/pair_buck6d_coul_gauss.html
 * (text: plans/lammps-docs-live/pair_buck6d_coul_gauss.rst). Written from the doc and from black-box runs of
 * native LAMMPS; the doc's own formulas are not used where a measurement disagrees (see the notes below).
 *
 * Syntax of the doc: "buck6d/coul/gauss/dsf args = smooth cutoff (cutoff2)" and
 * "buck6d/coul/gauss/long args = smooth smooth2 cutoff (cutoff2)"; "cutoff = global cutoff for Buckingham (and
 * Coulombic if only 1 arg)". Coefficients: "A (energy units)", "rho (distance^-1 units)", "C (energy-distance^6
 * units)", "D (distance^14 units)", "alpha (distance^-1 units)", "cutoff (distance units)"; "The second
 * coefficient, rho, must be greater than zero. The latter coefficient is optional."
 *
 * vdW term. The doc writes E = A e^{-kappa r} - C/r^6 * 1/(1 + D r^14). Measured with native LAMMPS (black box):
 * the damping factor is 1/(1 + D / r^14), i.e. D divides r^14 (dimers at r = 1.5, 2.5, 3.0, 4.0, 6.0, 10.0 agree
 * with A e^{-rho r} - C/r^6 / (1 + D/r^14) to 1e-15 relative for the first MOF-FF pair of in.hkust1, 10304 3.0612245 457.17971 4.5218516).
 *
 * Smoothing. Measured with native LAMMPS (black box): inside rs = smooth * rc (rc = the pair's Buckingham
 * cutoff) the whole vdW energy is multiplied by S(t) = 1 - 10 t^3 + 15 t^4 - 6 t^5 with t = (r - rs)/(rc - rs). The
 * native/formula ratio equals S(t) to 1e-9 relative for t < 0.6 and departs by 3e-8 at t = 0.83 and 2e-6 at
 * t = 0.96 (not explained; the switch is otherwise exact). The force is the exact derivative of the switched energy
 * (native force agrees with the analytic derivative to 1e-10 relative at r = 11.5). gauss/long applies the same
 * switch to the real-space Coulomb with rs2 = smooth2 * cutoff2 (measured ratio to S(t) about 1e-7, the largest
 * error of the Coulomb check; the g-ewald value used in the measurement was rounded to 7 digits).
 *
 * Coulomb, dsf. Measured with native LAMMPS (black box): with erf(alpha r)/r as the damped kernel g(r) (alpha is
 * the pair's 5th coefficient), the energy is C q_i q_j [g(r) - g(rc) - (r - rc) g'(rc)] for r < rc (C = qqrd2e,
 * i.e. 332.06371 for units real; the Fennell-Gezelter shifted-force form). Measured: the dimer energy with
 * q_i = 1, q_j = -0.5 at r = 2, 5, 11 agrees to 1e-14 relative. No self-energy term is present (one charged atom
 * gives 0.0). The Coulomb has no smoothing in the dsf style (the doc says so).
 *
 * Coulomb, long. Measured with native LAMMPS (black box): the real-space term is C q_i q_j
 * (erf(alpha r) - erf(g r)) / r with g the kspace G-ewald parameter (the kspace solver supplies the rest); the
 * pair term is zero for a charge-free atom. The style refuses to run without a kspace style, as native does
 * ("Pair style requires a KSpace style").
 *
 * Mixing: the doc says "These pair styles do not support mixing", so every I,J pair must be set (native: "All pair
 * coeffs are not set"). pair_modify shift and tail: the doc says both are not supported; native accepts them and
 * leaves the energy unchanged (measured), this engine refuses them instead. Special bonds: the Lennard-Jones
 * weight multiplies the vdW term. A special Coulomb weight other than 1.0 is refused (its native form is not
 * measured here).
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Switching function S(t) of the smoothing and its derivative with respect to t (see the header). */
const switchS = (t: number): number => 1 - t * t * t * (10 - t * (15 - 6 * t));
const switchDs = (t: number): number => -30 * t * t * (1 - t) * (1 - t);

/** erf(alpha r)/r kernel g(r) and its derivative, for the shifted-force form. */
const gKernel = (a: number, r: number): number => erfAcc(a * r) / r;
const gDeriv = (a: number, r: number): number => -erfAcc(a * r) / (r * r) + TWO_OVER_SQRTPI * a * libm.exp(-a * a * r * r) / r;

/** Buckingham term with the D/r^14 damping: energy and dE/dr (before smoothing). */
const vdwRaw = (A: number, rho: number, C: number, D: number, r: number): { e: number; de: number } => {
  const inv = 1 / r;
  const inv2 = inv * inv;
  const inv6 = inv2 * inv2 * inv2;
  const inv14 = inv6 * inv6 * inv2;
  const g = 1 / (1 + D * inv14);
  const ex = A * libm.exp(-rho * r);
  const h = C * inv6 * g;
  // d/dr [C r^-6 g] = -6 C r^-7 g + C r^-6 * 14 D r^-15 g^2
  const dh = -6 * C * inv6 * inv * g + C * inv6 * 14 * D * inv14 * inv * g * g;
  return { e: ex - h, de: -rho * ex - dh };
};

/** Smoothed vdW energy and dE/dr for a pair with Buckingham cutoff rc and smoothing onset rs. */
const vdwTerm = (A: number, rho: number, C: number, D: number, rs: number, rc: number, r: number): { e: number; de: number } => {
  const raw = vdwRaw(A, rho, C, D, r);
  if (r <= rs) return raw;
  const t = (r - rs) / (rc - rs);
  const S = switchS(t), dS = switchDs(t) / (rc - rs);
  return { e: raw.e * S, de: raw.de * S + raw.e * dS };
};

/** Real-space long-range Coulomb with a Gaussian-smeared charge: qq (erf(alpha r) - erf(g r)) / r, dE/dr. */
const coulLongRaw = (alpha: number, g: number, qq: number, r: number): { e: number; de: number } => {
  const a = erfAcc(alpha * r), b = erfAcc(g * r);
  const k = (a - b) / r;
  const dk = (TWO_OVER_SQRTPI * (alpha * libm.exp(-alpha * alpha * r * r) - g * libm.exp(-g * g * r * r))) / r - (a - b) / (r * r);
  return { e: qq * k, de: qq * dk };
};

export class PairBuck6dCoulGauss extends Pair {
  readonly name: string;
  virialFdotr = true;
  /** true for the long-range variant (needs a kspace style). */
  readonly longRange: boolean;
  smooth = 1;
  smooth2 = 1;
  cutGlobal = 0;
  cutCoul = 0;
  p!: PairParams;
  private aA = new Float64Array(0);
  private aRho = new Float64Array(0);
  private aC = new Float64Array(0);
  private aD = new Float64Array(0);
  private aAlpha = new Float64Array(0);
  private cutVdw = new Float64Array(0);
  private cutSmooth = new Float64Array(0);
  /** dsf: g(rc) and g'(rc) of the shifted-force form for the pair's alpha, set in initOne. */
  private dsfG = new Float64Array(0);
  private dsfGp = new Float64Array(0);
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  private qqrd2e = 1;

  constructor(longRange: boolean) {
    super();
    this.longRange = longRange;
    this.coulLong = longRange;
    this.name = longRange ? 'buck6d/coul/gauss/long' : 'buck6d/coul/gauss/dsf';
  }

  settings(args: string[], _ctx: StyleContext): void {
    const style = this.name;
    if (this.longRange) {
      if (args.length !== 3 && args.length !== 4) throw new StyleError(`Illegal pair_style command: usage: pair_style ${style} smooth smooth2 cutoff (cutoff2)`);
      this.smooth = parseNum(args[0], 'smooth');
      this.smooth2 = parseNum(args[1], 'smooth2');
      this.cutGlobal = parseNum(args[2], 'cutoff');
      this.cutCoul = args.length === 4 ? parseNum(args[3], 'cutoff2') : this.cutGlobal;
      if (!(this.smooth2 > 0 && this.smooth2 <= 1)) throw new StyleError(`${style}: smooth2 must be in (0, 1]`);
    } else {
      if (args.length !== 2 && args.length !== 3) throw new StyleError(`Illegal pair_style command: usage: pair_style ${style} smooth cutoff (cutoff2)`);
      this.smooth = parseNum(args[0], 'smooth');
      this.cutGlobal = parseNum(args[1], 'cutoff');
      this.cutCoul = args.length === 3 ? parseNum(args[2], 'cutoff2') : this.cutGlobal;
    }
    if (!(this.smooth > 0 && this.smooth <= 1)) throw new StyleError(`${style}: smooth must be in (0, 1]`);
    if (!(this.cutGlobal > 0) || !(this.cutCoul > 0)) throw new StyleError(`${style}: cutoffs must be > 0`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['A', 'rho', 'C', 'D', 'alpha', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    // As native LAMMPS does (src/MOFFF/pair_buck6d_coul_gauss_dsf.cpp and _long.cpp coeff at b332f28): a word count other than
    // seven or eight (I, J, the five coefficients and an optional cutoff) gives "Incorrect args for pair coefficients".
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const names = ['A', 'rho', 'C', 'D', 'alpha'];
    const vals = args.slice(2, 7).map((w, k) => parseNum(w, names[k]));
    if (!(vals[1] > 0)) throw new StyleError(`${this.name}: the second coefficient rho must be greater than zero`);
    const cut = args[7] !== undefined ? parseNum(args[7], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [...vals, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    // pair_modify shift / tail: refused (the doc says they are not supported; native ignores them, measured).
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set. pair ${i} ${j} is not set (${this.name} does not mix)`);
    const nt = this.ntypes + 1;
    if (this.aA.length !== nt * nt) {
      this.aA = new Float64Array(nt * nt);
      this.aRho = new Float64Array(nt * nt);
      this.aC = new Float64Array(nt * nt);
      this.aD = new Float64Array(nt * nt);
      this.aAlpha = new Float64Array(nt * nt);
      this.cutVdw = new Float64Array(nt * nt);
      this.cutSmooth = new Float64Array(nt * nt);
      this.dsfG = new Float64Array(nt * nt);
      this.dsfGp = new Float64Array(nt * nt);
    }
    const cut = p.get('cut', i, j);
    const rcv = Number.isNaN(cut) ? this.cutGlobal : cut;
    const sm = this.smooth;
    for (const k of [i * nt + j, j * nt + i]) {
      this.aA[k] = p.get('A', i, j);
      this.aRho[k] = p.get('rho', i, j);
      this.aC[k] = p.get('C', i, j);
      this.aD[k] = p.get('D', i, j);
      this.aAlpha[k] = p.get('alpha', i, j);
      this.cutVdw[k] = rcv;
      this.cutSmooth[k] = sm * rcv;
      if (!this.longRange) {
        this.dsfG[k] = gKernel(this.aAlpha[k], this.cutCoul);
        this.dsfGp[k] = gDeriv(this.aAlpha[k], this.cutCoul);
      }
    }
    return Math.max(rcv, this.cutCoul);
  }

  /**
   * Smoothed vdW energy and dE/dr of the pair of types ti, tj at distance r, or null beyond the pair's
   * Buckingham cutoff. angle_style cosine/buck6d adds this term between the end atoms of each angle.
   */
  vdwEnergy(ti: number, tj: number, r: number): { e: number; de: number } | null {
    const k = ti * (this.ntypes + 1) + tj;
    const rc = this.cutVdw[k];
    if (!(r < rc)) return null;
    return vdwTerm(this.aA[k], this.aRho[k], this.aC[k], this.aD[k], this.cutSmooth[k], rc, r);
  }

  /** Coulomb energy and dE/dr at distance r for pair type t (qq = qqrd2e qi qj). */
  private coulTerm(t: number, r: number, qq: number, g: number): { e: number; de: number } {
    const alpha = this.aAlpha[t];
    if (this.longRange) {
      const raw = coulLongRaw(alpha, g, qq, r);
      const rc = this.cutCoul, rs = this.smooth2 * rc;
      if (r <= rs) return raw;
      const tt = (r - rs) / (rc - rs);
      const S = switchS(tt), dS = switchDs(tt) / (rc - rs);
      return { e: raw.e * S, de: raw.de * S + raw.e * dS };
    }
    // shifted-force form: E = qq [g(r) - g(rc) - (r - rc) g'(rc)], dE/dr = qq [g'(r) - g'(rc)]
    const gpRc = this.dsfGp[t];
    const gr = gKernel(alpha, r), gpr = gDeriv(alpha, r);
    return { e: qq * (gr - this.dsfG[t] - (r - this.cutCoul) * gpRc), de: qq * (gpr - gpRc) };
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const g = this.gEwald;
    const cutc2 = this.cutCoul * this.cutCoul;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    if (this.longRange && !(g > 0)) throw new StyleError(`Pair style ${this.name} requires a KSpace style`);
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
        if (rsq >= cutsq[t]) continue;
        if (sb !== 0 && sC[sb] !== 1) throw new StyleError(`${this.name}: special_bonds coul factor ${sC[sb]} is not supported (only 1.0)`);
        const r = Math.sqrt(rsq);
        const factorLJ = sLJ[sb];
        let fpair = 0, eij = 0;
        if (r < this.cutVdw[t]) {
          const v = vdwTerm(this.aA[t], this.aRho[t], this.aC[t], this.aD[t], this.cutSmooth[t], this.cutVdw[t], r);
          fpair += -factorLJ * v.de / r;
          eij += factorLJ * v.e;
          evdwl += factorLJ * v.e;
        }
        const qq = this.qqrd2e * qi * q[j];
        if (rsq < cutc2 && qq !== 0) {
          const c = this.coulTerm(t, r, qq, g);
          fpair += -c.de / r;
          eij += c.e;
          ecoul += c.e;
        }
        if (fpair === 0 && eij === 0) continue;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (tally) tallyAtom(pc, i, j, eij, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number): { eng: number; fforce: number } {
    if (factorCoul !== 1) throw new StyleError(`${this.name}: special_bonds coul factor ${factorCoul} is not supported (only 1.0)`);
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    let fforce = 0, eng = 0;
    if (r < this.cutVdw[t]) {
      const v = vdwTerm(this.aA[t], this.aRho[t], this.aC[t], this.aD[t], this.cutSmooth[t], this.cutVdw[t], r);
      fforce += -factorLJ * v.de / r;
      eng += factorLJ * v.e;
    }
    const qq = this.qqrd2e * qi * qj;
    if (rsq < this.cutCoul * this.cutCoul && qq !== 0) {
      const c = this.coulTerm(t, r, qq, this.gEwald);
      fforce += -c.de / r;
      eng += c.e;
    }
    return { eng, fforce };
  }

  extract(name: string): unknown {
    return name === 'cut_coul' && this.longRange ? this.cutCoul : undefined;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('A', i, i))} ${fmtCoeff(this.p.get('rho', i, i))} ${fmtCoeff(this.p.get('C', i, i))} ${fmtCoeff(this.p.get('D', i, i))} ${fmtCoeff(this.p.get('alpha', i, i))}`);
    }
    return out;
  }
}

export class PairBuck6dCoulGaussDsf extends PairBuck6dCoulGauss {
  constructor() { super(false); }
}

export class PairBuck6dCoulGaussLong extends PairBuck6dCoulGauss {
  constructor() { super(true); }
}
