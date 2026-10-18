/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/math_pswf.cpp at b332f28
 * (the Legendre, Gauss-Legendre quadrature, Chebyshev and monomial interpolation helpers used by the prolate spheroidal
 * wave function tables of kspace_style esp). Copyright of the LAMMPS sources: Sandia Corporation, GPL-2.0.
 * This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 */
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { cosFma as libmCosFma } from '../libm/sincos_fma';

export const MY_PI = 3.141592653589793;   // math_const.h: MY_PI (4 atan(1))
export const MAX_CHEB_ORDER = 40;

/** legepol: P_n(x) and P_n'(x) by the three-term recurrence (math_pswf.cpp legepol). */
export const legepol = (x: number, n: number): { pol: number; der: number } => {
  if (n === 0) return { pol: 1.0, der: 0.0 };
  if (n === 1) return { pol: x, der: 1.0 };
  let pkm1: number;
  let pk = 1.0;
  let pkp1 = x;
  for (let k = 1; k < n; ++k) {
    pkm1 = pk;
    pk = pkp1;
    pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
  }
  const pol = pkp1;
  const der = n * (x * pkp1 - pk) / (x * x - 1);
  return { pol, der };
};

/** legetayl: Taylor sum of P_n about x (math_pswf.cpp legetayl). */
export const legetayl = (pol: number, der: number, x: number, h: number, n: number, k: number):
  { sum: number; sumder: number } => {
  const done = 1.0;
  const q0 = pol;
  const q1 = der * h;
  let q2 = (2 * x * der - n * (n + done) * pol) / (1 - x * x);
  q2 = q2 * h * h / 2;
  let sum = q0 + q1 + q2;
  let sumder = q1 / h + q2 * 2 / h;
  if (k <= 2) return { sum, sumder };
  let qi = q1;
  let qip1 = q2;
  for (let i = 1; i <= k - 2; ++i) {
    let d = 2 * x * (i + 1) * (i + 1) / h * qip1 - (n * (n + done) - i * (i + 1)) * qi;
    d = d / (i + 1) / (i + 2) * h * h / (1 - x * x);
    const qip2 = d;
    sum += qip2;
    sumder += d * (i + 2) / h;
    qi = qip1;
    qip1 = qip2;
  }
  return { sum, sumder };
};

/**
 * legerts: nodes (ts) and weights (whts) of the n-point Gauss-Legendre rule (math_pswf.cpp legerts).
 * itype > 0 also fills whts. Arrays are length n.
 */
export const legerts = (itype: number, n: number, ts: Float64Array, whts: Float64Array): void => {
  let k = 30;
  const d = 1.0;
  const d2 = d + 1.0e-24;
  if (d2 !== d) k = 54;
  const half = Math.trunc(n / 2);
  const ifodd = n - 2 * half;
  const pi_val = 4.0 * Math.atan(1.0);
  const h = pi_val / (2.0 * n);
  let ii = 0;
  for (let i = 1; i <= n; i++) {
    if (i < (Math.trunc(n / 2) + 1)) continue;
    ii++;
    const t = (2.0 * i - 1.0) * h;
    ts[ii - 1] = -libmCosFma(t);
  }
  let { pol, der } = legepol(0.0, n);
  let x0 = 0.0;
  let x1 = ts[0];
  const n2 = Math.trunc((n + 1) / 2);
  let pol3 = pol;
  let der3 = der;
  for (let kk = 1; kk <= n2; kk++) {
    if (ifodd === 1 && kk === 1) {
      ts[kk - 1] = x0;
      if (itype > 0) whts[kk - 1] = der;
      x0 = x1;
      x1 = ts[kk];
      pol3 = pol;
      der3 = der;
      continue;
    }
    let ifstop = 0;
    for (let i = 1; i <= 10; i++) {
      const hh = x1 - x0;
      const r = legetayl(pol3, der3, x0, hh, n, k);
      pol = r.sum;
      der = r.sumder;
      x1 = x1 - pol / der;
      if (Math.abs(pol) < 1.0e-12) ifstop++;
      if (ifstop === 3) break;
    }
    ts[kk - 1] = x1;
    if (itype > 0) whts[kk - 1] = der;
    x0 = x1;
    x1 = ts[kk];
    pol3 = pol;
    der3 = der;
  }
  for (let i = n2; i >= 1; i--) ts[i - 1 + half] = ts[i - 1];
  for (let i = 1; i <= half; i++) ts[i - 1] = -ts[n - i];
  if (itype <= 0) return;
  for (let i = n2; i >= 1; i--) whts[i - 1 + half] = whts[i - 1];
  for (let i = 1; i <= half; i++) whts[i - 1] = whts[n - i];
  for (let i = 0; i < n; i++) {
    const tmp = 1.0 - ts[i] * ts[i];
    whts[i] = 2.0 / tmp / (whts[i] * whts[i]);
  }
};

/** legepols: P_0..P_n at x (math_pswf.cpp legepols), written to pols[0..n]. */
export const legepols = (x: number, n: number, pols: Float64Array): void => {
  let pkm1 = 1.0;
  let pk = x;
  if (n === 0) { pols[0] = 1.0; return; }
  if (n === 1) { pols[0] = 1.0; pols[1] = x; return; }
  pols[0] = 1.0;
  pols[1] = x;
  for (let k = 1; k < n; ++k) {
    const pkp1 = ((2 * k + 1) * x * pk - k * pkm1) / (k + 1);
    pols[k + 1] = pkp1;
    pkm1 = pk;
    pk = pkp1;
  }
};

/** legeexev: Legendre series sum_j pexp[j] P_j(x) (math_pswf.cpp legeexev). */
export const legeexev = (x: number, pexp: Float64Array, n: number): number => {
  let pjm2 = 1.0;
  let pjm1 = x;
  let val = pexp[0] * pjm2 + pexp[1] * pjm1;
  for (let j = 2; j <= n; ++j) {
    const pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
    val += pexp[j] * pj;
    pjm2 = pjm1;
    pjm1 = pj;
  }
  return val;
};

/** legeFDER: Legendre series value and derivative (math_pswf.cpp legeFDER). */
export const legeFDER = (x: number, pexp: Float64Array, n: number): { val: number; der: number } => {
  let pjm2 = 1.0;
  let pjm1 = x;
  let derjm2 = 0.0;
  let derjm1 = 1.0;
  let val = pexp[0] * pjm2 + pexp[1] * pjm1;
  let der = pexp[1];
  for (let j = 2; j <= n; ++j) {
    const pj = ((2 * j - 1) * x * pjm1 - (j - 1) * pjm2) / j;
    val += pexp[j] * pj;
    let derj = (2 * j - 1) * (pjm1 + x * derjm1) - (j - 1) * derjm2;
    derj /= j;
    der += pexp[j] * derj;
    pjm2 = pjm1;
    pjm1 = pj;
    derjm2 = derjm1;
    derjm1 = derj;
  }
  return { val, der };
};

/** legendre: P_n(x) and derivative (math_pswf.cpp legendre, used only by its disabled branch). */
export const legendre = (n: number, x: number): { pn: number; pn_prime: number } => {
  if (n === 0) return { pn: 1.0, pn_prime: 0.0 };
  if (n === 1) return { pn: x, pn_prime: 1.0 };
  let pn_minus1 = 1.0;
  let pn_minus2 = 0.0;
  let pn = x;
  for (let k = 2; k <= n; ++k) {
    pn_minus2 = pn_minus1;
    pn_minus1 = pn;
    pn = ((2.0 * k - 1.0) * x * pn_minus1 - (k - 1.0) * pn_minus2) / k;
  }
  const pn_prime = n * (x * pn - pn_minus1) / (x * x - 1.0);
  return { pn, pn_prime };
};

const gaussCache = new Map<number, { nodes: Float64Array; weights: Float64Array }>();

/** gaussian_quadrature: n-point nodes and weights, computed once per n (legerts with itype 1). */
export const gaussianQuadrature = (n: number): { nodes: Float64Array; weights: Float64Array } => {
  const hit = gaussCache.get(n);
  if (hit) return hit;
  const nodes = new Float64Array(n);
  const weights = new Float64Array(n);
  legerts(1, n, nodes, weights);
  const r = { nodes, weights };
  gaussCache.set(n, r);
  return r;
};

/** cheb_nodes_1d: Chebyshev nodes of the given order mapped to [a, b]. */
export const chebNodes1d = (order: number, a: number, b: number): number[] => {
  const nodes = new Array<number>(order);
  for (let i = 0; i < order; i++) {
    nodes[i] = -libmCosFma((i + 0.5) * MY_PI / order) * 0.5 + 0.5;
    nodes[i] = nodes[i] * (b - a) + a;
  }
  return nodes;
};

/**
 * monomial_interp_1d: coefficients (monomial basis) of the interpolant through (nodes_i, fn_v_i) with Chebyshev nodes
 * on [a, b] (Newton divided differences, then expansion in the monomial basis).
 */
export const monomialInterp1d = (nnodes: number, fn_v: number[], a: number, b: number): number[] => {
  const x = chebNodes1d(nnodes, a, b);
  const multiplyX = (p: number[], xx: number): number[] => {
    const r = new Array<number>(p.length + 1).fill(0);
    for (let i = 0; i < p.length; ++i) {
      r[i] += -xx * p[i];
      r[i + 1] += p[i];
    }
    return r;
  };
  const dof = Math.trunc(fn_v.length / nnodes);
  const newton = fn_v.slice();
  const coeff = new Array<number>(dof * nnodes).fill(0);
  for (let idof = 0; idof < dof; ++idof) {
    const offset = idof * nnodes;
    for (let j = 1; j < nnodes; ++j)
      for (let i = nnodes - 1; i >= j; --i)
        newton[offset + i] = (newton[offset + i] - newton[offset + i - 1]) / (x[i] - x[i - j]);
    coeff[offset] = newton[offset];
  }
  let basis: number[] = [1.0];
  for (let j = 1; j < nnodes; ++j) {
    basis = multiplyX(basis, x[j - 1]);
    for (let idof = 0; idof < dof; idof++) {
      const offset = idof * nnodes;
      for (let m = 0; m < basis.length; ++m) coeff[offset + m] += newton[offset + j] * basis[m];
    }
  }
  return coeff;
};

/** cheb_basis_1d: Chebyshev basis T_0..T_{order-1} at the points x on [a, b], y[i*n + j]. */
export const chebBasis1d = (order: number, x: number[], a: number, b: number): number[] => {
  const n = x.length;
  const y = new Array<number>(order * n).fill(0);
  if (order > 0) for (let i = 0; i < n; i++) y[i] = 1.0;
  if (order > 1) for (let i = 0; i < n; i++) y[i + n] = x[i] * 2 / (b - a) - 2 * a / (b - a) - 1;
  for (let i = 2; i < order; i++) {
    for (let j = 0; j < n; j++) {
      y[i * n + j] = 2 * y[n + j] * y[i * n - 1 * n + j] - y[i * n - 2 * n + j];
    }
  }
  return y;
};

const chebCache = new Map<number, Float64Array>();

/** cheb_interp_1d: Chebyshev coefficients of the interpolant of fn_v at the order Chebyshev nodes on [-1, 1]. */
export const chebInterp1d = (order: number, fn_v: number[]): number[] => {
  let p = chebCache.get(order);
  if (!p) {
    const x = chebNodes1d(order, -1, 1);
    p = Float64Array.from(chebBasis1d(order, x, -1, 1));
    chebCache.set(order, p);
  }
  const dof = Math.trunc(fn_v.length / order);
  const coeff = new Array<number>(dof * order).fill(0);
  const invOrder = 1.0 / order;
  const twoInvOrder = 2.0 * invOrder;
  for (let id = 0; id < dof; ++id) {
    const base = id * order;
    for (let k = 0; k < order; ++k) {
      let sum = 0.0;
      const off = k * order;
      for (let j = 0; j < order; ++j) sum += fn_v[base + j] * p[off + j];
      coeff[base + k] = k === 0 ? sum * invOrder : sum * twoInvOrder;
    }
  }
  return coeff;
};

/** prosinin: the sin/cos quadrature sums of prosinin (math_pswf.cpp prosinin). */
export const prosinin = (c: number, ts: Float64Array, whts: Float64Array, fs: Float64Array, x: number,
  n: number): { rint: number; derrint: number } => {
  let rint = 0.0;
  let derrint = 0.0;
  for (let i = 0; i < n; ++i) {
    const diff = x - ts[i];
    const sin_term = libmSin(c * diff);
    const cos_term = libmCos(c * diff);
    rint += whts[i] * fs[i] * sin_term / diff;
    derrint += whts[i] * fs[i] / (diff * diff) * (c * diff * cos_term - sin_term);
  }
  return { rint, derrint };
};
