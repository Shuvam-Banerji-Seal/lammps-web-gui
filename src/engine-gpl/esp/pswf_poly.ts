/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/math_pswf.cpp at b332f28
 * (prolc180, force_poly, energy_poly, fourier_poly, spread_fourier_poly and spread_real_poly: the polynomial fits of
 * the splitting, force, energy, spreading and Fourier kernels used by kspace_style esp). Copyright of the LAMMPS
 * sources: Sandia Corporation, GPL-2.0. This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 */
import { MAX_CHEB_ORDER, chebNodes1d, chebInterp1d, monomialInterp1d, gaussianQuadrature } from './pswf_base';
import { prolate0Eval, prolate0IntEval } from './pswf_prolate';
import { cosFma as libmCos } from '../libm/sincos_fma';
import { log as libmLog } from '../libm/log';

/** The 180 table values of math_pswf.cpp prolc180 (c for accuracy 10^-(i/10)). */
const CS: number[] = [
  0.43368E-16, 0.10048E+01, 0.17298E+01, 0.22271E+01, 0.26382E+01, 0.30035E+01, 0.33409E+01, 0.36598E+01,
  0.39658E+01, 0.42621E+01, 0.45513E+01, 0.48347E+01, 0.51136E+01, 0.53887E+01, 0.56606E+01, 0.59299E+01,
  0.61968E+01, 0.64616E+01, 0.67247E+01, 0.69862E+01, 0.72462E+01, 0.75049E+01, 0.77625E+01, 0.80189E+01,
  0.82744E+01, 0.85289E+01, 0.87826E+01, 0.90355E+01, 0.92877E+01, 0.95392E+01, 0.97900E+01, 0.10040E+02,
  0.10290E+02, 0.10539E+02, 0.10788E+02, 0.11036E+02, 0.11284E+02, 0.11531E+02, 0.11778E+02, 0.12024E+02,
  0.12270E+02, 0.12516E+02, 0.12762E+02, 0.13007E+02, 0.13251E+02, 0.13496E+02, 0.13740E+02, 0.13984E+02,
  0.14228E+02, 0.14471E+02, 0.14714E+02, 0.14957E+02, 0.15200E+02, 0.15443E+02, 0.15685E+02, 0.15927E+02,
  0.16169E+02, 0.16411E+02, 0.16652E+02, 0.16894E+02, 0.17135E+02, 0.17376E+02, 0.17617E+02, 0.17858E+02,
  0.18098E+02, 0.18339E+02, 0.18579E+02, 0.18819E+02, 0.19059E+02, 0.19299E+02, 0.19539E+02, 0.19778E+02,
  0.20018E+02, 0.20257E+02, 0.20496E+02, 0.20736E+02, 0.20975E+02, 0.21214E+02, 0.21452E+02, 0.21691E+02,
  0.21930E+02, 0.22168E+02, 0.22407E+02, 0.22645E+02, 0.22884E+02, 0.23122E+02, 0.23360E+02, 0.23598E+02,
  0.23836E+02, 0.24074E+02, 0.24311E+02, 0.24549E+02, 0.24787E+02, 0.25024E+02, 0.25262E+02, 0.25499E+02,
  0.25737E+02, 0.25974E+02, 0.26211E+02, 0.26448E+02, 0.26685E+02, 0.26922E+02, 0.27159E+02, 0.27396E+02,
  0.27633E+02, 0.27870E+02, 0.28106E+02, 0.28343E+02, 0.28580E+02, 0.28816E+02, 0.29053E+02, 0.29289E+02,
  0.29526E+02, 0.29762E+02, 0.29998E+02, 0.30234E+02, 0.30471E+02, 0.30707E+02, 0.30943E+02, 0.31179E+02,
  0.31415E+02, 0.31651E+02, 0.31887E+02, 0.32123E+02, 0.32358E+02, 0.32594E+02, 0.32830E+02, 0.33066E+02,
  0.33301E+02, 0.33537E+02, 0.33773E+02, 0.34008E+02, 0.34244E+02, 0.34479E+02, 0.34714E+02, 0.34950E+02,
  0.35185E+02, 0.35421E+02, 0.35656E+02, 0.35891E+02, 0.36126E+02, 0.36362E+02, 0.36597E+02, 0.36832E+02,
  0.37067E+02, 0.37302E+02, 0.37537E+02, 0.37772E+02, 0.38007E+02, 0.38242E+02, 0.38477E+02, 0.38712E+02,
  0.38947E+02, 0.39181E+02, 0.39416E+02, 0.39651E+02, 0.39886E+02, 0.40120E+02, 0.40355E+02, 0.40590E+02,
  0.40824E+02, 0.41059E+02, 0.41294E+02, 0.41528E+02, 0.41763E+02, 0.41997E+02, 0.42232E+02, 0.42466E+02,
  0.42700E+02, 0.42935E+02, 0.43169E+02, 0.43404E+02, 0.43638E+02, 0.43872E+02, 0.44107E+02, 0.44341E+02,
  0.44575E+02, 0.44809E+02, 0.45044E+02, 0.45278E+02,
];


/** Chebyshev filter + monomial re-interpolation shared by the fits (the same steps in every math_pswf.cpp fit). */
const fitPoly = (tol: number, f: (x: number) => number, nTol: number): number[] => {
  const order = MAX_CHEB_ORDER;
  let nodes = chebNodes1d(order, -1.0, 1.0);
  const fnV: number[] = nodes.map((x) => f(x));
  const cheb = chebInterp1d(order, fnV);
  let maxOrder = -1;
  let maxCoeffs = 0.0;
  for (let i = 0; i < order; i++) maxCoeffs = Math.max(maxCoeffs, Math.abs(cheb[i]));
  for (let i = 0; i < order; i++) {
    if (Math.abs(cheb[i]) > tol * maxCoeffs) maxOrder = Math.max(maxOrder, i + 1);
  }
  void nTol;
  const nnodes = maxOrder;
  nodes = chebNodes1d(nnodes, -1.0, 1.0);
  const fn2 = nodes.map((x) => f(x));
  const coeffs = monomialInterp1d(nnodes, fn2, -1.0, 1.0);
  return coeffs.slice(0, maxOrder);
};

/** prolc180(eps, c): the splitting parameter c for accuracy eps (table lookup, math_pswf.cpp prolc180). */
export const prolc180 = (eps0: number): number => {
  let eps = eps0;
  if (eps < 1.0e-18) eps = 1.0e-18;
  const d = -libmLog(eps) / libmLog(10.0);
  const i = Math.trunc(d * 10 + 0.1);
  return CS[i - 1];
};

/** force_poly(tol, c): Chebyshev-filtered polynomial of the real-space force splitting kernel. */
export const forcePoly = (tol: number, c: number): number[] => {
  const c0 = prolate0IntEval(c, 1.0);
  return fitPoly(tol, (x0) => {
    const x = (x0 + 1.0) / 2.0;
    let val = prolate0IntEval(c, x) / c0 - x / c0 * prolate0Eval(c, x);
    val = 1 - val;
    return val;
  }, 0);
};

/** energy_poly(tol, c): polynomial of the real-space energy splitting kernel. */
export const energyPoly = (tol: number, c: number): number[] => {
  const c0 = prolate0IntEval(c, 1.0);
  return fitPoly(tol, (x0) => {
    const x = (x0 + 1.0) / 2.0;
    let val = prolate0IntEval(c, x) / c0;
    val = 1 - val;
    return val;
  }, 0);
};

/** fourier_poly(tol, c): { lambda, coeffs } of the Fourier-space splitting kernel. */
export const fourierPoly = (tol: number, c: number): { lambda: number; coeffs: number[] } => {
  const c0 = prolate0IntEval(c, 1.0);
  const q = gaussianQuadrature(200);
  let lambda = 0.0;
  for (let i = 0; i < 200; i++) {
    lambda += q.weights[i] * prolate0Eval(c, q.nodes[i]) * libmCos(c * q.nodes[i] * 0.5);
  }
  lambda /= prolate0Eval(c, 0.5);
  const coeffs = fitPoly(tol, (x0) => {
    const x = (x0 + 1.0) / 2.0;
    return lambda * prolate0Eval(c, x) / c0;
  }, 0);
  return { lambda, coeffs };
};

/** spread_fourier_poly(tol, c): { lambda, coeffs } of the Fourier-space spreading kernel. */
export const spreadFourierPoly = (tol: number, c: number): { lambda: number; coeffs: number[] } => {
  const q = gaussianQuadrature(200);
  let lambda = 0.0;
  for (let i = 0; i < 200; i++) {
    lambda += q.weights[i] * prolate0Eval(c, q.nodes[i]) * libmCos(c * q.nodes[i] * 0.5);
  }
  lambda /= prolate0Eval(c, 0.5);
  const coeffs = fitPoly(tol, (x0) => {
    const x = (x0 + 1.0) / 2.0;
    return lambda * prolate0Eval(c, x);
  }, 0);
  return { lambda, coeffs };
};

/**
 * spread_real_poly(P, tol, c): coefficients of the P piecewise polynomials of the real-space spreading kernel,
 * laid out coeffs[j * P + i] (math_pswf.cpp spread_real_poly).
 */
export const spreadRealPoly = (P: number, tol: number, c: number): number[] => {
  const order = MAX_CHEB_ORDER;
  const nodes0 = chebNodes1d(order, -0.5, 0.5);
  const f = (iP: number, x: number): number => {
    let arg = x - P / 2.0 + iP + 0.5;
    arg /= P / 2.0;
    return prolate0Eval(c, arg);
  };
  const fnV: number[] = new Array<number>(P * order);
  for (let idof = 0; idof < P; idof++) {
    for (let i = 0; i < order; i++) fnV[idof * order + i] = f(idof, nodes0[i]);
  }
  const cheb = chebInterp1d(order, fnV);
  let maxOrder = -1;
  for (let idof = 0; idof < P; idof++) {
    let maxCoeffs = 0.0;
    for (let i = 0; i < order; i++) maxCoeffs = Math.max(maxCoeffs, Math.abs(cheb[idof * order + i]));
    for (let i = 0; i < order; i++) {
      if (Math.abs(cheb[idof * order + i]) > tol * maxCoeffs) maxOrder = Math.max(maxOrder, i + 1);
    }
  }
  const nnodes = maxOrder;
  const nodes = chebNodes1d(nnodes, -0.5, 0.5);
  const fn2 = new Array<number>(P * nnodes);
  for (let iP = 0; iP < P; iP++) {
    for (let i = 0; i < nnodes; i++) fn2[iP * nnodes + i] = f(iP, nodes[i]);
  }
  const tmp = monomialInterp1d(nnodes, fn2, -0.5, 0.5);
  const coeffs = new Array<number>(P * maxOrder).fill(0);
  for (let i = 0; i < P; i++) {
    for (let j = 0; j < maxOrder; j++) coeffs[j * P + i] = tmp[i * maxOrder + j];
  }
  return coeffs;
};
