/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/math_pswf.cpp at b332f28
 * (the prolate spheroidal wave function eigen-solver prolmatr, prolql1, prolfact, prolsolv, prolfun0, prolps0i, prol0ini,
 * prol0eva, prol0int0r and the Prolate0Fun wrappers). Copyright of the LAMMPS sources: Sandia Corporation, GPL-2.0.
 * This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 *
 * The C++ routines take pointers into one workspace array w; here each pointer is a Float64Array view
 * (subarray) of the same buffer, so the arithmetic and the array contents are those of the C++ code.
 */
import { legeexev, legeFDER, legerts, prosinin } from './pswf_base';

const sq = Math.sqrt;

/** prolcoef: recurrence coefficients of the prolate matrix for order k (math_pswf.cpp prolcoef). */
const prolcoef = (rlam: number, k: number, c: number): { alpha: number; beta: number; gamma: number } => {
  let d = k * (k - 1);
  d = d / (2 * k + 1) / (2 * k - 1);
  const uk = d;
  d = (k + 1) * (k + 1);
  d = d / (2 * k + 3);
  let d2 = k * k;
  d2 = d2 / (2 * k - 1);
  const vk = (d + d2) / (2 * k + 1);
  d = (k + 1) * (k + 2);
  d = d / (2 * k + 1) / (2 * k + 3);
  const wk = d;
  return { alpha: -c * c * uk, beta: rlam - k * (k + 1) - c * c * vk, gamma: -c * c * wk };
};

/** prolmatr: the (symmetrised) tridiagonal matrix of the prolate problem (math_pswf.cpp prolmatr). */
const prolmatr = (as: Float64Array, bs: Float64Array, cs: Float64Array, n: number, c: number, rlam: number,
  ifsymm: number, ifodd: number): void => {
  const done = 1.0;
  const half = done / 2.0;
  let k = 0;
  if (ifodd > 0) {
    for (let k0 = 1; k0 <= n + 2; k0 += 2) {
      k++;
      const r = prolcoef(rlam, k0, c);
      as[k - 1] = r.alpha;
      bs[k - 1] = r.beta;
      cs[k - 1] = r.gamma;
      if (ifsymm !== 0) {
        if (k0 > 1) as[k - 1] = as[k - 1] / sq(k0 - 2 + half) * sq(k0 + half);
        cs[k - 1] = cs[k - 1] * sq(k0 + half) / sq(k0 + half + 2);
      }
    }
  } else {
    for (let k0 = 0; k0 <= n + 2; k0 += 2) {
      k++;
      const r = prolcoef(rlam, k0, c);
      as[k - 1] = r.alpha;
      bs[k - 1] = r.beta;
      cs[k - 1] = r.gamma;
      if (ifsymm !== 0) {
        if (k0 !== 0) as[k - 1] = as[k - 1] / sq(k0 - 2 + half) * sq(k0 + half);
        cs[k - 1] = cs[k - 1] * sq(k0 + half) / sq(k0 + half + 2);
      }
    }
  }
};

/** prolql1: eigenvalues of a symmetric tridiagonal matrix (d = diagonal, e = off-diagonal); returns ierr. */
const prolql1 = (n: number, d: Float64Array, e: Float64Array): number => {
  if (n === 1) return 0;
  for (let i = 1; i < n; ++i) e[i - 1] = e[i];
  e[n - 1] = 0.0;
  for (let l = 0; l < n; ++l) {
    let j = 0;
    for (;;) {
      let m: number;
      for (m = l; m < n - 1; ++m) {
        const tst1 = Math.abs(d[m]) + Math.abs(d[m + 1]);
        const tst2 = tst1 + Math.abs(e[m]);
        if (tst2 === tst1) break;
      }
      if (m === l) break;
      if (j === 30) return l + 1;
      ++j;
      let g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      let r = sq(g * g + 1.0);
      g = d[m] - d[l] + e[l] / (g + (g < 0 || Object.is(g, -0) ? -Math.abs(r) : Math.abs(r)));
      let s = 1.0;
      let c = 1.0;
      let p = 0.0;
      for (let i = m - 1; i >= l; --i) {
        const f = s * e[i];
        const b = c * e[i];
        r = sq(f * f + g * g);
        e[i + 1] = r;
        if (r === 0.0) {
          d[i + 1] -= p;
          e[m] = 0.0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;
      }
      if (r === 0.0) break;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }
    if (l === 0) continue;
    for (let i = l; i > 0; --i) {
      if (d[i] >= d[i - 1]) break;
      const t = d[i];
      d[i] = d[i - 1];
      d[i - 1] = t;
    }
  }
  return 0;
};

/** prolfact: LU-type factorisation of a tridiagonal matrix (math_pswf.cpp prolfact). */
const prolfact = (a: Float64Array, b: Float64Array, c: Float64Array, n: number, u: Float64Array,
  v: Float64Array, w: Float64Array): void => {
  for (let i = 0; i < n - 1; ++i) {
    const d = c[i + 1] / a[i];
    a[i + 1] -= b[i] * d;
    u[i] = d;
  }
  for (let i = n - 1; i > 0; --i) {
    const d = b[i - 1] / a[i];
    v[i] = d;
  }
  const done = 1.0;
  for (let i = 0; i < n; ++i) w[i] = done / a[i];
};

/** prolsolv: solve with the factorisation of prolfact, in place on rhs (math_pswf.cpp prolsolv). */
const prolsolv = (u: Float64Array, v: Float64Array, w: Float64Array, n: number, rhs: Float64Array): void => {
  for (let i = 0; i < n - 1; ++i) rhs[i + 1] -= u[i] * rhs[i];
  for (let i = n - 1; i > 0; --i) rhs[i - 1] -= rhs[i] * v[i];
  for (let i = 0; i < n; ++i) rhs[i] *= w[i];
};

/**
 * prolfun0: Legendre coefficients of the prolate function psi_0^c (math_pswf.cpp prolfun0). Writes the coefficient
 * array into xk (interleaved with zeros, length 2*nterms+2) and returns { ier, nterms, rkhi }.
 */
const prolfun0 = (n: number, c: number, as: Float64Array, bs: Float64Array, cs: Float64Array,
  xk: Float64Array, u: Float64Array, v: Float64Array, w: Float64Array, eps: number):
  { ier: number; nterms: number; rkhi: number } => {
  let ier = 0;
  const delta = 1.0e-8;
  const ifsymm = 1;
  const numit = 4;
  let rlam = 0;
  const ifodd = -1;
  prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);
  ier = prolql1(Math.trunc(n / 2), bs, as);
  if (ier !== 0) return { ier: 2048, nterms: 0, rkhi: 0 };
  const rkhi = -bs[Math.trunc(n / 2) - 1];
  rlam = -bs[Math.trunc(n / 2) - 1] + delta;
  xk.fill(1.0, 0, n);
  prolmatr(as, bs, cs, n, c, rlam, ifsymm, ifodd);
  prolfact(bs, cs, as, Math.trunc(n / 2), u, v, w);
  const half2 = Math.trunc(n / 2);
  for (let ijk = 0; ijk < numit; ++ijk) {
    prolsolv(u, v, w, half2, xk);
    let d = 0;
    for (let j = 0; j < half2; ++j) d += xk[j] * xk[j];
    d = sq(d);
    for (let j = 0; j < half2; ++j) xk[j] /= d;
    for (let j = 0; j < half2; ++j) as[j] = xk[j];
  }
  const half = 0.5;
  let nterms = 0;
  for (let i = 0; i < half2; ++i) {
    if (Math.abs(xk[i]) > eps) nterms = i + 1;
    xk[i] *= sq(i * 2 + half);
    cs[i] = xk[i];
  }
  let j = 0;
  for (let i = 0; i <= nterms; ++i) {
    xk[j++] = cs[i];
    xk[j++] = 0;
  }
  nterms *= 2;
  return { ier, nterms, rkhi };
};

const NS = [48, 64, 80, 92, 106, 120, 130, 144, 156, 168, 178, 190, 202, 214, 224, 236, 248, 258, 268, 280];

/** prolps0i: sizes the workspace and computes the Legendre expansion (math_pswf.cpp prolps0i). */
const prolps0i = (c: number, w: Float64Array, lenw: number): { ier: number; nterms: number; ltot: number; rkhi: number } => {
  const eps = 1.0e-16;
  let n = Math.trunc(c * 3);
  n = Math.trunc(n / 2);
  const i = Math.trunc(c / 10);
  if (i <= 19) n = NS[i];
  const ixk = 1;
  const lxk = n + 2;
  const ias = ixk + lxk;
  const las = n + 2;
  const ibs = ias + las;
  const lbs = n + 2;
  const ics = ibs + lbs;
  const lcs = n + 2;
  const iu = ics + lcs;
  const lu = n + 2;
  const iv = iu + lu;
  const lv = n + 2;
  const iw = iv + lv;
  const lw = n + 2;
  const ltot = iw + lw;
  if (ltot >= lenw) return { ier: 512, nterms: 0, ltot, rkhi: 0 };
  // C pointers w + k - 1 become views starting at index k - 1
  const r = prolfun0(n, c, w.subarray(ias - 1), w.subarray(ibs - 1), w.subarray(ics - 1),
    w.subarray(ixk - 1), w.subarray(iu - 1), w.subarray(iv - 1), w.subarray(iw - 1), eps);
  return { ier: r.ier, nterms: r.nterms, ltot, rkhi: r.rkhi };
};

/** prol0ini: prepares psi_0^c on [-1, 1] (and the quadrature data outside), math_pswf.cpp prol0ini. */
export const prol0ini = (c: number, w: Float64Array, lenw: number):
  { ier: number; rlam20: number; rkhi: number; keep: number; ltot: number } => {
  let ier = 0;
  const thresh = 45;
  const iw = 11;
  w[0] = iw + 0.1;
  w[8] = thresh;
  let ltot = 0;
  const ps = prolps0i(c, w.subarray(iw - 1), lenw);
  ier = ps.ier;
  ltot = ps.ltot;
  const nterms = ps.nterms;
  const rkhi = ps.rkhi;
  if (ier !== 0) return { ier, rlam20: 0, rkhi, keep: 0, ltot };
  if (c >= thresh) {
    w[7] = c;
    w[4] = nterms + 0.1;
    return { ier, rlam20: 0, rkhi, keep: nterms + 3, ltot };
  }
  const ngauss = nterms * 2;
  const lw = nterms + 2;
  const its = iw + lw;
  const lts = ngauss + 2;
  const iwhts = its + lts;
  const lwhts = ngauss + 2;
  const ifs = iwhts + lwhts;
  const lfs = ngauss + 2;
  const keep = ifs + lfs;
  if (keep > ltot) ltot = keep;
  if (keep >= lenw) return { ier: 1024, rlam20: 0, rkhi, keep, ltot };
  w[1] = its + 0.1;
  w[2] = iwhts + 0.1;
  w[3] = ifs + 0.1;
  const ts = w.subarray(its - 1);
  const whts = w.subarray(iwhts - 1);
  const fs = w.subarray(ifs - 1);
  const pexp = w.subarray(iw - 1);
  legerts(1, ngauss, ts, whts);
  for (let i = 0; i < ngauss; ++i) {
    fs[i] = legeexev(ts[i], pexp, nterms - 1);
  }
  const x0 = 0;
  const f0 = legeexev(x0, pexp, nterms - 1);
  const pr = prosinin(c, ts, whts, fs, x0, ngauss);
  let rlam = pr.rint;
  rlam = rlam / f0;
  const rlam20 = rlam;
  w[4] = nterms + 0.1;
  w[5] = ngauss + 0.1;
  w[6] = rlam;
  w[7] = c;
  return { ier, rlam20, rkhi, keep, ltot };
};

/** prol0eva: psi_0^c(x) and its derivative (math_pswf.cpp prol0eva). */
export const prol0eva = (x: number, w: Float64Array): { psi0: number; derpsi0: number } => {
  const iw = Math.trunc(w[0]);
  const its = Math.trunc(w[1]);
  const iwhts = Math.trunc(w[2]);
  const ifs = Math.trunc(w[3]);
  const nterms = Math.trunc(w[4]);
  const ngauss = Math.trunc(w[5]);
  const rlam = w[6];
  const c = w[7];
  const thresh = w[8];
  if (Math.abs(x) > 1) {
    if (c >= thresh - 1.0e-10) return { psi0: 0, derpsi0: 0 };
    const pr = prosinin(c, w.subarray(its - 1), w.subarray(iwhts - 1), w.subarray(ifs - 1), x, ngauss);
    return { psi0: pr.rint / rlam, derpsi0: pr.derrint / rlam };
  }
  const f = legeFDER(x, w.subarray(iw - 1), nterms - 2);
  // to match chebfun psi0, needs a factor of sqrt(2)
  return { psi0: sq(2.0) * f.val, derpsi0: sq(2.0) * f.der };
};

let xs200: Float64Array | null = null;
let ws200: Float64Array | null = null;

/** prol0int0r: int_0^r psi_0^c(x) dx by 200-point Gauss-Legendre (math_pswf.cpp prol0int0r). */
export const prol0int0r = (w: Float64Array, r: number): number => {
  const npts = 200;
  if (!xs200 || !ws200) {
    xs200 = new Float64Array(npts);
    ws200 = new Float64Array(npts);
    legerts(1, npts, xs200, ws200);
  }
  const fvals = new Float64Array(npts);
  for (let i = 0; i < npts; ++i) {
    const xs_r = (xs200[i] + 1) * r / 2;
    fvals[i] = prol0eva(xs_r, w).psi0;
  }
  let val = 0;
  for (let i = 0; i < npts; ++i) val += ws200[i] * r / 2 * fvals[i];
  return val;
};

/** Prolate0Fun of math_pswf.cpp: the prolate function data for one c (workspace of length 10000). */
export interface Prolate0Fun {
  c: number;
  w: Float64Array;
}
const funCache = new Map<number, Prolate0Fun>();
export const prolate0Fun = (c: number): Prolate0Fun => {
  let f = funCache.get(c);
  if (!f) {
    const w = new Float64Array(10000);
    prol0ini(c, w, 10000);
    f = { c, w };
    funCache.set(c, f);
  }
  return f;
};
/** prolate0_eval_derivative(c, x) */
export const prolate0EvalDerivative = (c: number, x: number): number => prol0eva(x, prolate0Fun(c).w).derpsi0;
/** prolate0_eval(c, x) */
export const prolate0Eval = (c: number, x: number): number => prol0eva(x, prolate0Fun(c).w).psi0;
/** prolate0_int_eval(c, r) */
export const prolate0IntEval = (c: number, r: number): number => prol0int0r(prolate0Fun(c).w, r);
