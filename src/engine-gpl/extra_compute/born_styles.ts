/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the born_matrix() methods (first and second derivative of a term's energy, for compute born/matrix) of
 * src/pair_lj_cut.cpp, src/pair_lj_cut_coul_cut.cpp, src/pair_coul_cut.cpp, src/pair_coul_debye.cpp, src/pair_buck.cpp,
 * src/pair_buck_coul_cut.cpp, src/pair_born.cpp, src/pair_hybrid.cpp, src/pair_hybrid_scaled.cpp,
 * src/CLASS2/pair_lj_class2.cpp, src/CLASS2/pair_lj_class2_coul_cut.cpp, src/EXTRA-PAIR/pair_lj96_cut.cpp,
 * src/EXTRA-PAIR/pair_harmonic_cut.cpp, src/MOLECULE/bond_harmonic.cpp, bond_fene.cpp, bond_gromos.cpp, bond_morse.cpp,
 * src/CLASS2/bond_class2.cpp, src/EXTRA-MOLECULE/bond_gaussian.cpp, bond_harmonic_shift.cpp, bond_harmonic_shift_cut.cpp,
 * bond_nonlinear.cpp, src/MOLECULE/angle_harmonic.cpp, angle_cosine.cpp, angle_cosine_squared.cpp,
 * src/EXTRA-MOLECULE/angle_cosine_periodic.cpp, angle_cosine_delta.cpp, angle_cosine_squared_restricted.cpp,
 * angle_fourier.cpp, angle_fourier_simple.cpp, angle_quartic.cpp, src/YAFF/angle_mm3.cpp,
 * src/MOLECULE/dihedral_opls.cpp, dihedral_multi_harmonic.cpp, src/EXTRA-MOLECULE/dihedral_nharmonic.cpp,
 * dihedral_quadratic.cpp, dihedral_helix.cpp, dihedral_cosine_squared_restricted.cpp, and the born_matrix_enable flags
 * of those styles, at commit b332f28 (the authors are named in each file; most born_matrix methods are by Germain
 * Clavier (TUe) and Aidan Thompson (Sandia)). Copyright Sandia Corporation (GNU General Public License); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Each function writes du (dU/dr for a pair or bond, dU/dcos for an angle or dihedral) and du2 (the second derivative)
 * into out[0], out[1], from the coefficients the engine's style stores. The formulas are native's as written, including
 * their quirks: lj96/cut's du2 has r^-14 where the derivative of its r^-9 term would give r^-11; cosine/periodic's du
 * multiplies by sin(theta) where the derivative divides; lj/cut/coul/cut and lj/cut/coul/debye add both terms for every
 * pair within the larger cutoff, and lj/cut/coul/debye (a subclass of lj/cut/coul/cut without its own born_matrix) takes
 * the unscreened Coulomb term. One native behaviour is not reproduced: PairBuck, PairBuckCoulCut and PairBorn init_one
 * never set rho[j][i], so native's born_matrix divides by an unset value for an unlike pair visited with the larger type
 * first (measured with native LAMMPS: NaN with two atom types); here rho is symmetric.
 */
import type { Bonded, Pair, PairParams } from '../../engine/force/types';
import type { PairHybrid } from '../../engine/force/pair/hybrid';
import { StyleError } from '../../engine/force/types';

export type Out = [number, number];

/** What a pair born_matrix call sees: the pair, its types and charges, the special factors and qqrd2e. */
export interface PairArgs {
  ti: number;
  tj: number;
  rsq: number;
  factorCoul: number;
  factorLJ: number;
  qi: number;
  qj: number;
  qqrd2e: number;
}

type PairFn = (p: Pair, a: PairArgs, out: Out) => void;

const params = (p: Pair): PairParams => (p as unknown as { p: PairParams }).p;

// --- pair styles -----------------------------------------------------------------------------------------------

/** PairCoulCut::born_matrix: du = -qqrd2e qi qj / r^2, du2 = 2 qqrd2e qi qj / r^3. */
const coulCut = (a: PairArgs): Out => {
  const r2inv = 1.0 / a.rsq;
  const rinv = Math.sqrt(r2inv);
  const r3inv = r2inv * rinv;
  return [-a.qqrd2e * a.qi * a.qj * r2inv, 2.0 * a.qqrd2e * a.qi * a.qj * r3inv];
};

/** PairLJCut::born_matrix with lj1 = 48 eps sigma^12, lj2 = 24 eps sigma^6 (PairLJCut::init_one). */
const ljCut = (p: Pair, a: PairArgs): Out => {
  const pp = params(p);
  const eps = pp.get('epsilon', a.ti, a.tj), sig = pp.get('sigma', a.ti, a.tj);
  const lj1 = 48.0 * eps * Math.pow(sig, 12.0), lj2 = 24.0 * eps * Math.pow(sig, 6.0);
  const r2inv = 1.0 / a.rsq;
  const rinv = Math.sqrt(r2inv);
  const r6inv = r2inv * r2inv * r2inv;
  return [r6inv * rinv * (lj2 - lj1 * r6inv), r6inv * r2inv * (13 * lj1 * r6inv - 7 * lj2)];
};

/** PairLJClass2::born_matrix with lj1 = 18 eps sigma^9, lj2 = 18 eps sigma^6 (PairLJClass2::init_one). */
const ljClass2 = (p: Pair, a: PairArgs): Out => {
  const pp = params(p);
  const eps = pp.get('epsilon', a.ti, a.tj), sig = pp.get('sigma', a.ti, a.tj);
  const lj1 = 18.0 * eps * Math.pow(sig, 9.0), lj2 = 18.0 * eps * Math.pow(sig, 6.0);
  const r2inv = 1.0 / a.rsq;
  const rinv = Math.sqrt(r2inv);
  const r3inv = r2inv * rinv;
  const r7inv = r3inv * r3inv * rinv;
  const r8inv = r7inv * rinv;
  return [r7inv * (lj2 - lj1 * r3inv), r8inv * (10 * lj1 * r3inv - 7 * lj2)];
};

/** PairBuck::born_matrix with buck1 = A / rho, buck2 = 6 C, rhoinv = 1 / rho (PairBuck::init_one). */
const buck = (p: Pair, a: PairArgs): Out => {
  const pp = params(p);
  const A = pp.get('A', a.ti, a.tj), rho = pp.get('rho', a.ti, a.tj), C = pp.get('C', a.ti, a.tj);
  const rhoinv = 1.0 / rho, buck1 = A / rho, buck2 = 6.0 * C;
  const r = Math.sqrt(a.rsq);
  const rexp = Math.exp(-r * rhoinv);
  const r2inv = 1.0 / a.rsq;
  const rinv = Math.sqrt(r2inv);
  const r6inv = r2inv * r2inv * r2inv;
  const r7inv = r6inv * rinv;
  const r8inv = r6inv * r2inv;
  return [buck2 * r7inv - buck1 * rexp, (buck1 / rho) * rexp - 7 * buck2 * r8inv];
};

const withCoul = (lj: (p: Pair, a: PairArgs) => Out): PairFn => (p, a, out) => {
  const [dl, d2l] = lj(p, a);
  const [dc, d2c] = coulCut(a);
  out[0] = a.factorLJ * dl + a.factorCoul * dc;
  out[1] = a.factorLJ * d2l + a.factorCoul * d2c;
};

const ljOnly = (lj: (p: Pair, a: PairArgs) => Out): PairFn => (p, a, out) => {
  const [d, d2] = lj(p, a);
  out[0] = a.factorLJ * d;
  out[1] = a.factorLJ * d2;
};

const PAIRS: Record<string, PairFn> = {
  'lj/cut': ljOnly(ljCut),
  'lj/cut/coul/cut': withCoul(ljCut),
  'lj/cut/coul/debye': withCoul(ljCut),
  'coul/cut': (_p, a, out) => {
    const [d, d2] = coulCut(a);
    out[0] = a.factorCoul * d;
    out[1] = a.factorCoul * d2;
  },
  'coul/debye': (p, a, out) => {
    const kappa = (p as unknown as { kappa: number }).kappa;
    const r = Math.sqrt(a.rsq);
    const r2inv = 1.0 / a.rsq;
    const rinv = Math.sqrt(r2inv);
    const r3inv = r2inv * rinv;
    const screening = Math.exp(-kappa * r);
    const qq = a.qqrd2e * a.qi * a.qj;
    out[0] = a.factorCoul * (-qq * r2inv * (1 + kappa * r) * screening);
    out[1] = a.factorCoul * (qq * r3inv * (2 + 2 * kappa * r + kappa * kappa * a.rsq) * screening);
  },
  buck: ljOnly(buck),
  'buck/coul/cut': withCoul(buck),
  born: ljOnly((p, a) => {
    const pp = params(p);
    const A = pp.get('A', a.ti, a.tj), rho = pp.get('rho', a.ti, a.tj), sigma = pp.get('sigma', a.ti, a.tj);
    const C = pp.get('C', a.ti, a.tj), D = pp.get('D', a.ti, a.tj);
    const rhoinv = 1.0 / rho, born1 = A / rho, born2 = 6.0 * C, born3 = 8.0 * D;
    const r = Math.sqrt(a.rsq);
    const rexp = Math.exp((sigma - r) * rhoinv);
    const r2inv = 1.0 / a.rsq;
    const rinv = Math.sqrt(r2inv);
    const r7inv = r2inv * r2inv * r2inv * rinv;
    const r8inv = r7inv * rinv;
    const r9inv = r7inv * r2inv;
    const r10inv = r9inv * rinv;
    return [born2 * r7inv - born1 * rexp - born3 * r9inv, (born1 / rho) * rexp - 7 * born2 * r8inv + 9 * born3 * r10inv];
  }),
  'lj/class2': ljOnly(ljClass2),
  'lj/class2/coul/cut': withCoul(ljClass2),
  'lj96/cut': ljOnly((p, a) => {
    const pp = params(p);
    const eps = pp.get('epsilon', a.ti, a.tj), sig = pp.get('sigma', a.ti, a.tj);
    const lj1 = 36.0 * eps * Math.pow(sig, 9.0), lj2 = 24.0 * eps * Math.pow(sig, 6.0);
    const r2inv = 1.0 / a.rsq;
    const rinv = Math.sqrt(r2inv);
    const r6inv = r2inv * r2inv * r2inv;
    const r3inv = r2inv * rinv;
    return [r6inv * rinv * (lj2 - lj1 * r3inv), r6inv * r2inv * (10 * lj1 * r6inv - 7 * lj2)];
  }),
  'harmonic/cut': ljOnly((p, a) => {
    const pp = params(p);
    const r = Math.sqrt(a.rsq);
    const dr = r - pp.get('cut', a.ti, a.tj);
    const du2 = 2 * pp.get('k', a.ti, a.tj);
    return [r > 0 ? du2 * dr : 0, du2];
  }),
};

const HYBRIDS = new Set(['hybrid', 'hybrid/overlay', 'hybrid/scaled', 'hybrid/molecular']);

/** born_matrix_enable of a pair style: a hybrid style has it when every sub-style has (PairHybrid::flags). */
export const pairBornEnabled = (p: Pair): boolean => {
  if (HYBRIDS.has(p.name)) return (p as PairHybrid).subs.every((s) => s.name in PAIRS);
  return p.name in PAIRS;
};

/** pair->born_matrix(...): out = [du, du2]; a style without one (switched to after the compute was made) gives 0, 0. */
export const pairBorn = (p: Pair, a: PairArgs, out: Out): void => {
  out[0] = out[1] = 0;
  if (!HYBRIDS.has(p.name)) {
    PAIRS[p.name]?.(p, a, out);
    return;
  }
  const h = p as PairHybrid;
  const subs = h.subsOf(a.ti, a.tj);
  if (subs.length === 0) throw new StyleError('Invoked pair born_matrix on pair style none');
  const scaled = h.mode === 'hybrid/scaled';
  const t = a.ti * (p.ntypes + 1) + a.tj;
  const one: Out = [0, 0];
  for (const k of subs) {
    const sub = h.subs[k];
    if (!(a.rsq < sub.style.cutsq[t])) continue;
    const fn = PAIRS[sub.name];
    // PairHybridScaled checks single_enable here, PairHybrid born_matrix_enable
    if (!scaled && !fn) throw new StyleError('Pair hybrid sub-style does not support born_matrix call');
    if (sub.specialLJ || sub.specialCoul) {
      throw new StyleError(scaled
        ? 'Pair hybrid single() does not support per sub-style special_bond'
        : 'Pair hybrid born_matrix calls do not support per sub-style special bond values');
    }
    one[0] = one[1] = 0;
    fn?.(sub.style, a, one);
    const scale = scaled ? h.scaleOf(sub) : 1;
    out[0] += scale * one[0];
    out[1] += scale * one[1];
  }
};

// --- bond styles -----------------------------------------------------------------------------------------------

type BondFn = (b: Bonded, type: number, rsq: number, boltz: number) => Out;

const bp = (b: Bonded, name: string): Float64Array => (b as unknown as { params: { p(n: string): Float64Array } }).params.p(name);

const MY_CUBEROOT2 = 1.25992104989487316476;
const MY_PI2 = 1.57079632679489661923;
const MY_PI4 = 0.78539816339744830962;
const MY_PI = Math.PI;

const BONDS: Record<string, BondFn> = {
  harmonic: (b, t, rsq) => {
    const r = Math.sqrt(rsq);
    const dr = r - bp(b, 'r0')[t];
    const du2 = 2 * bp(b, 'K')[t];
    return [r > 0.0 ? du2 * dr : 0.0, du2];
  },
  fene: (b, t, rsq) => {
    const k = bp(b, 'K')[t], r0 = bp(b, 'R0')[t], epsilon = bp(b, 'epsilon')[t], sigma = bp(b, 'sigma')[t];
    const r = Math.sqrt(rsq);
    const r0sq = r0 * r0;
    const rlogarg = 1.0 - rsq / r0sq;
    let du = k * r / rlogarg;
    let du2 = k * (1.0 + rsq / r0sq) / (rlogarg * rlogarg);
    if (rsq < MY_CUBEROOT2 * sigma * sigma) {
      const sr2 = sigma * sigma / rsq;
      const sr6 = sr2 * sr2 * sr2;
      du += 48.0 * epsilon * sr6 * (0.5 - sr6) / r;
      du2 += 48.0 * epsilon * sr6 * (13.0 * sr6 - 3.5) / rsq;
    }
    return [du, du2];
  },
  gromos: (b, t, rsq) => {
    const k = bp(b, 'K')[t], r0 = bp(b, 'r0')[t];
    const r = Math.sqrt(rsq);
    return [r > 0.0 ? 4 * k * r * (rsq - r0 * r0) : 0.0, 4 * k * (3 * rsq - r0 * r0)];
  },
  morse: (b, t, rsq) => {
    const d0 = bp(b, 'D')[t], alpha = bp(b, 'alpha')[t], r0 = bp(b, 'r0')[t];
    const r = Math.sqrt(rsq);
    const ralpha = Math.exp(-alpha * (r - r0));
    return [2.0 * d0 * alpha * (1.0 - ralpha) * ralpha, -2.0 * d0 * alpha * alpha * (1.0 - 2.0 * ralpha) * ralpha];
  },
  class2: (b, t, rsq) => {
    const r0 = bp(b, 'r0')[t], k2 = bp(b, 'K2')[t], k3 = bp(b, 'K3')[t], k4 = bp(b, 'K4')[t];
    const r = Math.sqrt(rsq);
    const dr = r - r0;
    const du2 = 2 * k2 + 6 * k3 * dr + 12 * k4 * dr * dr;
    return [r > 0.0 ? 2 * k2 * dr + 3 * k3 * dr * dr + 4 * k4 * dr * dr * dr : 0.0, du2];
  },
  gaussian: (b, t, rsq, boltz) => {
    const g = b as unknown as { T: Float64Array; terms: { a: number; w: number; r0: number }[][] };
    const SMALL = 2.3e-308;
    const r = Math.sqrt(rsq);
    const kT = boltz * g.T[t];
    let sumG = 0.0, sumNum = 0.0;
    for (const { a, w, r0 } of g.terms[t]) {
      const dr = r - r0;
      const gi = (a / (w * Math.sqrt(MY_PI2))) * Math.exp(-2 * dr * dr / (w * w));
      sumG += gi;
      sumNum += gi * dr / (w * w);
    }
    if (sumG < SMALL) sumG = SMALL;
    const du = 4.0 * kT * (sumNum / sumG);
    sumG = 0.0;
    let sumDg = 0.0, sumD2g = 0.0;
    for (const { a, w, r0 } of g.terms[t]) {
      const dr = r - r0;
      const gi = (a / (w * Math.sqrt(MY_PI2))) * Math.exp(-2 * dr * dr / (w * w));
      sumG += gi;
      sumDg -= 4.0 * gi * dr / Math.pow(w, 2);
      sumD2g += 4.0 * gi * (4.0 * Math.pow(r0, 2) - 8.0 * r0 * r - Math.pow(w, 2) + 4.0 * r * r) / Math.pow(w, 4);
    }
    if (sumG < SMALL) sumG = SMALL;
    return [du, -kT * (sumD2g * sumG - sumDg * sumDg) / (sumG * sumG)];
  },
  // BondHarmonicShift::coeff: k = Umin / (r0 - r1)^2, r1 the rc coefficient
  'harmonic/shift': (b, t, rsq) => {
    const r0 = bp(b, 'r0')[t], r1 = bp(b, 'rc')[t];
    const k = bp(b, 'Umin')[t] / ((r0 - r1) * (r0 - r1));
    const r = Math.sqrt(rsq);
    const du2 = 2 * k;
    return [r > 0.0 ? du2 * (r - r0) : 0.0, du2];
  },
  'harmonic/shift/cut': (b, t, rsq) => {
    const r0 = bp(b, 'r0')[t], r1 = bp(b, 'rc')[t];
    const k = bp(b, 'Umin')[t] / ((r0 - r1) * (r0 - r1));
    const r = Math.sqrt(rsq);
    if (r > r1) return [0.0, 0.0];
    const du2 = 2 * k;
    return [r > 0.0 ? du2 * (r - r0) : 0.0, du2];
  },
  nonlinear: (b, t, rsq) => {
    const epsilon = bp(b, 'epsilon')[t], r0 = bp(b, 'r0')[t], lamda = bp(b, 'lam')[t];
    const r = Math.sqrt(rsq);
    const dr = r - r0;
    const drsq = dr * dr;
    const lamdasq = lamda * lamda;
    const denom = lamdasq - drsq;
    const denomsq = denom * denom;
    return [2.0 * epsilon * lamdasq * dr / denomsq, 2.0 * epsilon * lamdasq * (lamdasq + 3.0 * drsq) / (denomsq * denom)];
  },
};

export const bondBornEnabled = (b: Bonded): boolean => b.name in BONDS;

/** bond->born_matrix(type, rsq, ...): [du, du2]; 0, 0 for a style without one. */
export const bondBorn = (b: Bonded, type: number, rsq: number, boltz: number): Out => BONDS[b.name]?.(b, type, rsq, boltz) ?? [0.0, 0.0];

// --- angle styles ----------------------------------------------------------------------------------------------

/** c = cos(theta) as each style's born_matrix computes it (minimum-image bond vectors, clamped to [-1, 1]). */
type AngleFn = (b: Bonded, type: number, c: number) => Out;

const ANGLES: Record<string, AngleFn> = {
  harmonic: (b, t, c) => {
    const theta = Math.acos(c);
    const k = bp(b, 'K')[t];
    const dtheta = theta - bp(b, 'theta0')[t];
    return [-2 * k * dtheta / Math.sin(theta), 2 * k * (Math.sin(theta) - dtheta * Math.cos(theta)) / Math.pow(Math.sin(theta), 3)];
  },
  cosine: (b, t) => [bp(b, 'K')[t], 0],
  'cosine/squared': (b, t, c) => {
    const du2 = 2 * bp(b, 'K')[t];
    return [du2 * (c - Math.cos(bp(b, 'theta0')[t])), du2];
  },
  // AngleCosinePeriodic::coeff: k = C / n^2, b = B, multiplicity = n
  'cosine/periodic': (b, t, c) => {
    const SMALL = 0.001;
    const n = bp(b, 'n')[t];
    const k = bp(b, 'C')[t] / (n * n), bb = bp(b, 'B')[t];
    const theta = Math.acos(c);
    let s = Math.sqrt(1.0 - c * c);
    if (s < SMALL) s = SMALL;
    s = 1.0 / s;
    const mAngle = n * theta;
    const powsign = n & 1 ? -1.0 : 1.0;
    const prefactor = -2.0 * k * bb * powsign * n;
    return [prefactor * Math.sin(mAngle) / s, prefactor * (c * Math.sin(mAngle) - s * Math.cos(mAngle) * n) / (s * s * s)];
  },
  'cosine/delta': (b, t, c) => {
    const k = bp(b, 'K')[t];
    const theta = Math.acos(c);
    const dtheta = theta - bp(b, 'theta0')[t];
    return [-k * Math.sin(dtheta) / Math.sin(theta),
      k * (Math.cos(dtheta) * Math.sin(theta) - Math.sin(dtheta) * Math.cos(theta)) / Math.pow(Math.sin(theta), 3)];
  },
  'cosine/squared/restricted': (b, t, c) => {
    const k = bp(b, 'K')[t];
    const c0 = Math.cos(bp(b, 'theta0')[t]);
    const sqSin = 1.0 - c * c;
    const numerator = 2.0 * c0 * c * c * c - 3.0 * (c0 * c0 + 1) * c * c + 6 * c0 * c - c0 * c0 - 1;
    return [2 * k * (c - c0) * (1.0 - c * c0) / (sqSin * sqSin), 2 * k * numerator / (sqSin * sqSin * sqSin)];
  },
  fourier: (b, t, c) => {
    const k = bp(b, 'K')[t], C1 = bp(b, 'C1')[t], C2 = bp(b, 'C2')[t];
    return [k * (C1 + 4 * C2 * c), 4 * k * C2];
  },
  'fourier/simple': (b, t, c) => {
    const k = bp(b, 'K')[t], C = bp(b, 'c')[t], N = bp(b, 'n')[t];
    const theta = Math.acos(c);
    return [k * C * N * Math.sin(N * theta) / Math.sin(theta),
      k * C * N * (Math.cos(theta) * Math.sin(N * theta) - N * Math.sin(theta) * Math.cos(N * theta)) / Math.pow(Math.sin(theta), 3)];
  },
  quartic: (b, t, c) => {
    const SMALL = 0.001;
    const k2 = bp(b, 'K2')[t], k3 = bp(b, 'K3')[t], k4 = bp(b, 'K4')[t];
    const theta = Math.acos(c);
    let s = Math.sqrt(1.0 - c * c);
    if (s < SMALL) s = SMALL;
    const dtheta = theta - bp(b, 'theta0')[t];
    const dtheta2 = dtheta * dtheta;
    const dtheta3 = dtheta2 * dtheta;
    return [-(2.0 * k2 * dtheta + 3.0 * k3 * dtheta2 + 4.0 * k4 * dtheta3) / s,
      (2.0 * k2 + 6.0 * k3 * dtheta + 12.0 * k4 * dtheta2) / (s * s) -
        (2.0 * k2 * dtheta + 3.0 * k3 * dtheta2 + 4.0 * k4 * dtheta3) * c / (s * s * s)];
  },
  mm3: (b, t, c) => {
    const SMALL = 0.001;
    const k2 = bp(b, 'K')[t];
    const theta = Math.acos(c);
    let s = Math.sqrt(1.0 - c * c);
    if (s < SMALL) s = SMALL;
    s = 1.0 / s;
    const dtheta = theta - bp(b, 'theta0')[t];
    const dtheta2 = dtheta * dtheta;
    const dtheta3 = dtheta2 * dtheta;
    const dtheta4 = dtheta3 * dtheta;
    const dtheta5 = dtheta4 * dtheta;
    const df = 2.0 * dtheta - 2.406423 * dtheta2 + 0.735348 * dtheta3 - 0.65832 * dtheta4 + 1.42254 * dtheta5;
    const d2f = 2.0 - 4.812846 * dtheta + 2.206044 * dtheta2 - 2.63328 * dtheta3 + 7.1127 * dtheta4;
    return [-k2 * df / s, k2 * (d2f - df * c / s) / (s * s)];
  },
};

export const angleBornEnabled = (b: Bonded): boolean => b.name in ANGLES;

/** angle->born_matrix(type, i1, i2, i3, ...): [du, du2] with respect to cos(theta); 0, 0 for a style without one. */
export const angleBorn = (b: Bonded, type: number, c: number): Out => ANGLES[b.name]?.(b, type, c) ?? [0.0, 0.0];

// --- dihedral styles -------------------------------------------------------------------------------------------

/**
 * cos(phi) of the dihedral styles' compute, from vb1 = x1 - x2, vb2 = x3 - x2, vb3 = x4 - x3: the c0 / c1mag / c2mag
 * form with the SMALL guards, clamped to [-1, 1], and dx = (vb1 x vb2) . vb3 / (|vb1 x vb2| |vb3|) for the sign of phi.
 */
const dihedralCos = (vb1: readonly number[], vb2: readonly number[], vb3: readonly number[]): { c: number; dx: number } => {
  const SMALL = 0.001;
  const [vb1x, vb1y, vb1z] = vb1, [vb2x, vb2y, vb2z] = vb2, [vb3x, vb3y, vb3z] = vb3;
  const vb2xm = -vb2x, vb2ym = -vb2y, vb2zm = -vb2z;
  const sb1 = 1.0 / (vb1x * vb1x + vb1y * vb1y + vb1z * vb1z);
  const sb3 = 1.0 / (vb3x * vb3x + vb3y * vb3y + vb3z * vb3z);
  const rb1 = Math.sqrt(sb1);
  const rb3 = Math.sqrt(sb3);
  const c0 = (vb1x * vb3x + vb1y * vb3y + vb1z * vb3z) * rb1 * rb3;
  const b1mag = Math.sqrt(vb1x * vb1x + vb1y * vb1y + vb1z * vb1z);
  const b2mag = Math.sqrt(vb2x * vb2x + vb2y * vb2y + vb2z * vb2z);
  const b3mag = Math.sqrt(vb3x * vb3x + vb3y * vb3y + vb3z * vb3z);
  let ctmp = vb1x * vb2x + vb1y * vb2y + vb1z * vb2z;
  const c1mag = ctmp * (1.0 / (b1mag * b2mag));
  ctmp = vb2xm * vb3x + vb2ym * vb3y + vb2zm * vb3z;
  const c2mag = ctmp * (1.0 / (b2mag * b3mag));
  let sc1 = Math.sqrt(Math.max(1.0 - c1mag * c1mag, 0.0));
  if (sc1 < SMALL) sc1 = SMALL;
  sc1 = 1.0 / sc1;
  let sc2 = Math.sqrt(Math.max(1.0 - c2mag * c2mag, 0.0));
  if (sc2 < SMALL) sc2 = SMALL;
  sc2 = 1.0 / sc2;
  let c = (c0 + c1mag * c2mag) * (sc1 * sc2);
  const cx = vb1y * vb2z - vb1z * vb2y;
  const cy = vb1z * vb2x - vb1x * vb2z;
  const cz = vb1x * vb2y - vb1y * vb2x;
  const cmag = Math.sqrt(cx * cx + cy * cy + cz * cz);
  const dx = (cx * vb3x + cy * vb3y + cz * vb3z) / cmag / b3mag;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;
  return { c, dx };
};

type DihedralFn = (b: Bonded, type: number, vb1: readonly number[], vb2: readonly number[], vb3: readonly number[]) => Out;

const SMALLER = 0.00001;

const DIHEDRALS: Record<string, DihedralFn> = {
  // DihedralOPLS::coeff stores half of each K
  opls: (b, t, vb1, vb2, vb3) => {
    const k1 = 0.5 * bp(b, 'K1')[t], k2 = 0.5 * bp(b, 'K2')[t], k3 = 0.5 * bp(b, 'K3')[t], k4 = 0.5 * bp(b, 'K4')[t];
    const { c, dx } = dihedralCos(vb1, vb2, vb3);
    let phi = Math.acos(c);
    if (dx < 0.0) phi *= -1.0;
    let si = Math.sin(phi);
    if (Math.abs(si) < SMALLER) si = SMALLER;
    const du = k1 - 2.0 * k2 * Math.sin(2.0 * phi) / si + 3.0 * k3 * Math.sin(3.0 * phi) / si - 4.0 * k4 * Math.sin(4.0 * phi) / si;
    const du2 = (4.0 * k2 * si * Math.cos(2.0 * phi) - 2.0 * k2 * Math.sin(2.0 * phi) -
      9.0 * k3 * si * Math.cos(3.0 * phi) + 3.0 * k3 * Math.sin(3.0 * phi) +
      16.0 * k4 * si * Math.cos(4.0 * phi) - 4.0 * k4 * Math.sin(4.0 * phi)) / (si * si * si);
    return [du, du2];
  },
  'multi/harmonic': (b, t, vb1, vb2, vb3) => {
    const a2 = bp(b, 'A2')[t], a3 = bp(b, 'A3')[t], a4 = bp(b, 'A4')[t], a5 = bp(b, 'A5')[t];
    const { c } = dihedralCos(vb1, vb2, vb3);
    return [a2 + c * (2.0 * a3 + c * (3.0 * a4 + c * 4.0 * a5)), 2.0 * a3 + 6.0 * c * (a4 + 2.0 * a5 * c)];
  },
  nharmonic: (b, t, vb1, vb2, vb3) => {
    const a = (b as unknown as { vals: (Float64Array | null)[] }).vals[t] ?? new Float64Array(0);
    const [vb1x, vb1y, vb1z] = vb1, [vb2x, vb2y, vb2z] = vb2, [vb3x, vb3y, vb3z] = vb3;
    const vb2xm = -vb2x, vb2ym = -vb2y, vb2zm = -vb2z;
    const ax = vb1y * vb2zm - vb1z * vb2ym;
    const ay = vb1z * vb2xm - vb1x * vb2zm;
    const az = vb1x * vb2ym - vb1y * vb2xm;
    const bx = vb3y * vb2zm - vb3z * vb2ym;
    const by = vb3z * vb2xm - vb3x * vb2zm;
    const bz = vb3x * vb2ym - vb3y * vb2xm;
    const rasq = ax * ax + ay * ay + az * az;
    const rbsq = bx * bx + by * by + bz * bz;
    const ra2inv = rasq > 0 ? 1.0 / rasq : 0.0;
    const rb2inv = rbsq > 0 ? 1.0 / rbsq : 0.0;
    const c = (ax * bx + ay * by + az * bz) * Math.sqrt(ra2inv * rb2inv);
    let du = 0.0, du2 = 0.0;
    for (let i = 1; i < a.length; i++) du += a[i] * i * Math.pow(c, i - 1);
    for (let i = 2; i < a.length; i++) du2 += a[i] * i * (i - 1) * Math.pow(c, i - 2);
    return [du, du2];
  },
  quadratic: (b, t, vb1, vb2, vb3) => {
    const k = bp(b, 'K')[t];
    const { c, dx } = dihedralCos(vb1, vb2, vb3);
    let phi = Math.acos(c);
    if (dx > 0.0) phi *= -1.0;
    let si = Math.sin(phi);
    if (Math.abs(si) < SMALLER) si = SMALLER;
    const siinv = 1.0 / si;
    let dphi = phi - bp(b, 'phi0')[t];
    if (dphi > MY_PI) dphi -= 2 * MY_PI;
    else if (dphi < -MY_PI) dphi += 2 * MY_PI;
    return [-2.0 * k * dphi * siinv, 2.0 * k * siinv * siinv * (1.0 - dphi * c * siinv)];
  },
  helix: (b, t, vb1, vb2, vb3) => {
    const aphi = bp(b, 'A')[t], bphi = bp(b, 'B')[t], cphi = bp(b, 'C')[t];
    const { c, dx } = dihedralCos(vb1, vb2, vb3);
    let phi = Math.acos(c);
    if (dx > 0.0) phi *= -1.0;
    let si = Math.sin(phi);
    if (Math.abs(si) < SMALLER) si = SMALLER;
    const siinv = 1.0 / si;
    return [-aphi + 3.0 * bphi * Math.sin(3.0 * phi) * siinv + cphi * Math.sin(phi + MY_PI4) * siinv,
      -(9.0 * bphi * Math.cos(3.0 * phi) + cphi * Math.cos(phi + MY_PI4)) * siinv * siinv +
        (3.0 * bphi * Math.sin(3.0 * phi) + cphi * Math.sin(phi + MY_PI4)) * c * siinv * siinv * siinv];
  },
  'cosine/squared/restricted': (b, t, vb1, vb2, vb3) => {
    const k = bp(b, 'K')[t];
    const { c } = dihedralCos(vb1, vb2, vb3);
    const p0 = Math.cos(bp(b, 'phi0')[t]);
    const sqSin = 1.0 - c * c;
    const numerator = 2.0 * p0 * c * c * c - 3.0 * (p0 * p0 + 1) * c * c + 6 * p0 * c - p0 * p0 - 1;
    return [2 * k * (c - p0) * (1.0 - c * p0) / (sqSin * sqSin), 2 * k * numerator / (sqSin * sqSin * sqSin)];
  },
};

export const dihedralBornEnabled = (b: Bonded): boolean => b.name in DIHEDRALS;

/** dihedral->born_matrix(...): [du, du2] with respect to cos(phi); 0, 0 for a style without one. */
export const dihedralBorn = (b: Bonded, type: number, vb1: readonly number[], vb2: readonly number[], vb3: readonly number[]): Out =>
  DIHEDRALS[b.name]?.(b, type, vb1, vb2, vb3) ?? [0.0, 0.0];
