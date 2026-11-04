/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/ASPHERE/math_extra_superellipsoids.cpp and
 * math_extra_superellipsoids.h, GPL-2.0 (src/engine-gpl/LICENSE). Contributing authors: Jacopo Bilotto (EPFL), Jibril B.
 * Coulibaly. The contact point (midway Newton method with line search), the surface derivatives, the overlap distance,
 * the mean and gaussian curvatures and the oriented bounding box test are those of the C++ file, in the same order of
 * operations where it matters for the iteration.
 *
 * Block types (the C++ BlockType): 0 = ellipsoid (block 2 2), 1 = n1 equal n2 (not 2), 2 = general superquadric. The engine
 * chooses the type from the block exponents (superellipsoid.ts); the formulas of 0 and 1 are the same surfaces, so only the
 * rounding of the iteration depends on the choice.
 *
 * Vectors and matrices are Float64Arrays (a rotation or Hessian row-major, M[3 i + j]); the contact iteration works in
 * module-level scratch arrays, so a Newton or line-search step allocates nothing. The arithmetic, and its order, is that
 * of the nested-array form this replaced.
 */
import { pow as libmPow } from '../libm/pow';

export const FORMULATION_ALGEBRAIC = 0;
export const FORMULATION_GEOMETRIC = 1;

const TIKHONOV_SCALE = 1e-14;
const ITERMAX_NR = 100;
const TOL_NR_RES = 1e-10 * 1e-10;
const TOL_NR_POS = 1e-6 * 1e-6;
const ITERMAX_LS = 10;
const PARAMETER_LS = 1e-4;
const CUTBACK_LS = 0.5;
const TOL_OVERLAP = 1e-8;
const ITERMAX_OVERLAP = 20;
const MINSLOPE_OVERLAP = 1e-12;
const REGULARIZATION_EPSILON = 1e-12;
const MAX_B_FAST = 1e30;

/** A particle of the contact computations: centre x, rotation R (row-major), half-diameters, exponents and block type. */
export interface ContactParticle {
  x: Float64Array;
  R: Float64Array;
  shape: Float64Array;
  block: Float64Array;
  flag: number;
}

export const contactParticle = (): ContactParticle => ({
  x: new Float64Array(3), R: new Float64Array(9), shape: new Float64Array(3), block: new Float64Array(2), flag: 0,
});

/** The value, global gradient and global Hessian (row-major) of a shape function (derivGlobal). */
interface Deriv {
  f: number;
  g: Float64Array;
  h: Float64Array;
}

const deriv = (): Deriv => ({ f: 0, g: new Float64Array(3), h: new Float64Array(9) });

/** derivLocal's gradient and Hessian in the particle frame; surfLocal's gradient; H R^T of the Hessian rotation. */
const GL = new Float64Array(3);
const HL = new Float64Array(9);
const SG = new Float64Array(3);
const TT = new Float64Array(9);
/** derivGlobal of particles i and j at the current trial point. */
const DI = deriv();
const DJ = deriv();
/** determineContactPoint: shape values, residual, Jacobian, the solves' copies, the step and the trial point. */
const SHAPEF = new Float64Array(2);
const RES = new Float64Array(4);
const JAC = new Float64Array(16);
const AS = new Float64Array(16);
const BF = new Float64Array(4);
const A2 = new Float64Array(16);
const B2 = new Float64Array(4);
const RHS = new Float64Array(4);
const XL = new Float64Array(4);
/** checkOrientedBoundingBoxes: R1^T R2, its absolute values and the centre offset in both frames. */
const OC = new Float64Array(9);
const OA = new Float64Array(9);
const OC1 = new Float64Array(3);
const OC2 = new Float64Array(3);

const norm2 = (a: Float64Array): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

/** Determinant of a 3x3 matrix given by rows (the det3 of the C++ det4_M44_zero). */
const det3 = (r00: number, r01: number, r02: number, r10: number, r11: number, r12: number, r20: number, r21: number, r22: number): number =>
  r00 * (r11 * r22 - r12 * r21) -
  r01 * (r10 * r22 - r12 * r20) +
  r02 * (r10 * r21 - r11 * r20);

/**
 * Solves the 4x4 system A x = b in place (A row-major, 16 entries; b 4 entries) with partial pivoting, as the C++
 * solve_4x4_robust_unrolled. Returns false on a zero pivot.
 */
const solve4x4 = (A: Float64Array, b: Float64Array): boolean => {
  let p = 0;
  let maxVal = Math.abs(A[0]);
  let val = Math.abs(A[4]);
  if (val > maxVal) { maxVal = val; p = 1; }
  val = Math.abs(A[8]);
  if (val > maxVal) { maxVal = val; p = 2; }
  val = Math.abs(A[12]);
  if (val > maxVal) { maxVal = val; p = 3; }
  if (maxVal <= 0) return false;
  if (p !== 0) {
    const ro = p * 4;
    let t = b[0]; b[0] = b[p]; b[p] = t;
    for (let k = 0; k < 4; k++) { t = A[k]; A[k] = A[ro + k]; A[ro + k] = t; }
  }
  {
    const inv = 1 / A[0];
    const f1 = A[4] * inv;
    A[5] -= f1 * A[1]; A[6] -= f1 * A[2]; A[7] -= f1 * A[3]; b[1] -= f1 * b[0];
    const f2 = A[8] * inv;
    A[9] -= f2 * A[1]; A[10] -= f2 * A[2]; A[11] -= f2 * A[3]; b[2] -= f2 * b[0];
    const f3 = A[12] * inv;
    A[13] -= f3 * A[1]; A[14] -= f3 * A[2]; A[15] -= f3 * A[3]; b[3] -= f3 * b[0];
  }
  p = 1;
  maxVal = Math.abs(A[5]);
  val = Math.abs(A[9]);
  if (val > maxVal) { maxVal = val; p = 2; }
  val = Math.abs(A[13]);
  if (val > maxVal) { maxVal = val; p = 3; }
  if (maxVal <= 0) return false;
  if (p !== 1) {
    const ro = p * 4;
    let t = b[1]; b[1] = b[p]; b[p] = t;
    for (let k = 1; k < 4; k++) { t = A[4 + k]; A[4 + k] = A[ro + k]; A[ro + k] = t; }
  }
  {
    const inv = 1 / A[5];
    const f2 = A[9] * inv;
    A[10] -= f2 * A[6]; A[11] -= f2 * A[7]; b[2] -= f2 * b[1];
    const f3 = A[13] * inv;
    A[14] -= f3 * A[6]; A[15] -= f3 * A[7]; b[3] -= f3 * b[1];
  }
  p = 2;
  maxVal = Math.abs(A[10]);
  val = Math.abs(A[14]);
  if (val > maxVal) { maxVal = val; p = 3; }
  if (maxVal <= 0) return false;
  if (p !== 2) {
    let t = b[2]; b[2] = b[3]; b[3] = t;
    t = A[10]; A[10] = A[14]; A[14] = t;
    t = A[11]; A[11] = A[15]; A[15] = t;
  }
  {
    const inv = 1 / A[10];
    const f3 = A[14] * inv;
    A[15] -= f3 * A[11];
    b[3] -= f3 * b[2];
  }
  if (Math.abs(A[15]) <= 0) return false;
  b[3] *= 1 / A[15];
  b[2] = (b[2] - A[11] * b[3]) * (1 / A[10]);
  b[1] = (b[1] - A[6] * b[2] - A[7] * b[3]) * (1 / A[5]);
  b[0] = (b[0] - A[1] * b[1] - A[2] * b[2] - A[3] * b[3]) * (1 / A[0]);
  return true;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Shape functions in the local frame of a particle (shape = half-diameters a, b, c; block = n1, n2): the value is
 * returned, the gradient and Hessian go to GL and HL.
 * ------------------------------------------------------------------------------------------------------------------- */

const signOf = (v: number): number => (v > 0 ? 1 : -1);

/** Superquadric (block types 2): n1 along z, n2 in the x-y plane. */
const derivSuperquad = (x0: number, x1: number, x2: number, shape: Float64Array, block: Float64Array): number => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x0 * aInv), yb = Math.abs(x1 * bInv), zc = Math.abs(x2 * cInv);
  const n1 = block[0], n2 = block[1];
  const xaP2m2 = libmPow(xa, n2 - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = libmPow(yb, n2 - 2), ybP2m1 = ybP2m2 * yb;
  const nu = xaP2m1 * xa + ybP2m1 * yb;
  const nuP2m2 = libmPow(nu, n1 / n2 - 2), nuP2m1 = nuP2m2 * nu;
  const zcP1m2 = libmPow(zc, n1 - 2), zcP1m1 = zcP1m2 * zc;
  const sx = signOf(x0), sy = signOf(x1), sz = signOf(x2);
  GL[0] = n1 * aInv * xaP2m1 * nuP2m1 * sx;
  GL[1] = n1 * bInv * ybP2m1 * nuP2m1 * sy;
  GL[2] = n1 * cInv * zcP1m1 * sz;
  HL[0] = aInv * aInv * (n1 * (n2 - 1) * xaP2m2 * nuP2m1 + (n1 - n2) * n1 * (xaP2m1 * xaP2m1) * nuP2m2);
  HL[4] = bInv * bInv * (n1 * (n2 - 1) * ybP2m2 * nuP2m1 + (n1 - n2) * n1 * (ybP2m1 * ybP2m1) * nuP2m2);
  HL[1] = HL[3] = aInv * bInv * (n1 - n2) * n1 * xaP2m1 * ybP2m1 * nuP2m2 * (sx * sy);
  HL[8] = cInv * cInv * n1 * (n1 - 1) * zcP1m2;
  HL[2] = HL[5] = HL[6] = HL[7] = 0;
  return nuP2m1 * nu + zcP1m1 * zc - 1;
};
/** Exponent n1 = n2 = n (block type 1). */
const derivEqual = (x0: number, x1: number, x2: number, shape: Float64Array, n: number): number => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x0 * aInv), yb = Math.abs(x1 * bInv), zc = Math.abs(x2 * cInv);
  const xaP2m2 = libmPow(xa, n - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = libmPow(yb, n - 2), ybP2m1 = ybP2m2 * yb;
  const zcP2m2 = libmPow(zc, n - 2), zcP2m1 = zcP2m2 * zc;
  GL[0] = n * aInv * xaP2m1 * signOf(x0);
  GL[1] = n * bInv * ybP2m1 * signOf(x1);
  GL[2] = n * cInv * zcP2m1 * signOf(x2);
  HL[0] = aInv * aInv * n * (n - 1) * xaP2m2;
  HL[4] = bInv * bInv * n * (n - 1) * ybP2m2;
  HL[8] = cInv * cInv * n * (n - 1) * zcP2m2;
  HL[1] = HL[2] = HL[3] = HL[5] = HL[6] = HL[7] = 0;
  return xaP2m1 * xa + ybP2m1 * yb + zcP2m1 * zc - 1;
};

/** Ellipsoid (block type 0). */
const derivEllipsoid = (x0: number, x1: number, x2: number, shape: Float64Array): number => {
  const a = 2 / (shape[0] * shape[0]), b = 2 / (shape[1] * shape[1]), c = 2 / (shape[2] * shape[2]);
  GL[0] = a * x0; GL[1] = b * x1; GL[2] = c * x2;
  HL[0] = a; HL[4] = b; HL[8] = c;
  HL[1] = HL[2] = HL[3] = HL[5] = HL[6] = HL[7] = 0;
  return 0.5 * (GL[0] * x0 + GL[1] * x1 + GL[2] * x2) - 1;
};

/** Local shape function and derivatives for block type flag. */
const derivLocal = (x0: number, x1: number, x2: number, shape: Float64Array, block: Float64Array, flag: number): number => {
  if (flag === 0) return derivEllipsoid(x0, x1, x2, shape);
  if (flag === 1) return derivEqual(x0, x1, x2, shape, block[0]);
  return derivSuperquad(x0, x1, x2, shape, block);
};

/** Value and gradient (into SG) only, for the surface search (the C++ shape_and_gradient_*_surfacesearch). */
const surfSuperquad = (x0: number, x1: number, x2: number, shape: Float64Array, block: Float64Array): number => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x0 * aInv), yb = Math.abs(x1 * bInv), zc = Math.abs(x2 * cInv);
  const n1 = block[0], n2 = block[1];
  const xaP2m2 = libmPow(xa, n2 - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = libmPow(yb, n2 - 2), ybP2m1 = ybP2m2 * yb;
  const nu = xaP2m1 * xa + ybP2m1 * yb;
  const nuP2m2 = libmPow(nu, n1 / n2 - 2), nuP2m1 = nuP2m2 * nu;
  const zcP1m2 = libmPow(zc, n1 - 2), zcP1m1 = zcP1m2 * zc;
  const sx = signOf(x0), sy = signOf(x1), sz = signOf(x2);
  const g0 = n1 * aInv * xaP2m1 * nuP2m1 * sx, g1 = n1 * bInv * ybP2m1 * nuP2m1 * sy, g2 = n1 * cInv * zcP1m1 * sz;
  const F = nuP2m1 * nu + zcP1m1 * zc;
  const scale = libmPow(F, 1 / n1 - 1) / n1;
  const f = libmPow(F, 1 / n1) - 1;
  SG[0] = g0 * scale; SG[1] = g1 * scale; SG[2] = g2 * scale;
  return f;
};

const surfEqual = (x0: number, x1: number, x2: number, shape: Float64Array, n: number): number => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x0 * aInv), yb = Math.abs(x1 * bInv), zc = Math.abs(x2 * cInv);
  const xaP2m2 = libmPow(xa, n - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = libmPow(yb, n - 2), ybP2m1 = ybP2m2 * yb;
  const zcP2m2 = libmPow(zc, n - 2), zcP2m1 = zcP2m2 * zc;
  const g0 = n * aInv * xaP2m1 * signOf(x0), g1 = n * bInv * ybP2m1 * signOf(x1), g2 = n * cInv * zcP2m1 * signOf(x2);
  const F = xaP2m1 * xa + ybP2m1 * yb + zcP2m1 * zc;
  const scale = libmPow(F, 1 / n - 1) / n;
  const f = libmPow(F, 1 / n) - 1;
  SG[0] = g0 * scale; SG[1] = g1 * scale; SG[2] = g2 * scale;
  return f;
};

/** Surface search derivative for flags 0 and 1 (the C++ uses the n1 = n2 form there). */
const surfLocal = (x0: number, x1: number, x2: number, shape: Float64Array, block: Float64Array, flag: number): number =>
  flag <= 1 ? surfEqual(x0, x1, x2, shape, block[0]) : surfSuperquad(x0, x1, x2, shape, block);

/**
 * Generalised shape function G = (F + 1)^(1/n1) - 1 scaled by the average radius (geometric formulation). Modifies the
 * gradient and Hessian in place and returns the new value.
 */
const regularize = (n1: number, avgRadius: number, value: number, grad: Float64Array, hess: Float64Array): number => {
  const base = Math.max(value + 1, REGULARIZATION_EPSILON);
  const invF = 1 / base;
  const invN1 = 1 / n1;
  const Fp = libmPow(base, invN1);
  const scaleGrad = avgRadius * invN1 * Fp * invF;
  const scaleHessAdd = scaleGrad * (invN1 - 1) * invF;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) hess[3 * i + j] = hess[3 * i + j] * scaleGrad + scaleHessAdd * (grad[i] * grad[j]);
  }
  for (let i = 0; i < 3; i++) grad[i] *= scaleGrad;
  return avgRadius * (Fp - 1);
};

/** Shape function (local value, regularized in the geometric formulation) of particle p at X0, gradient and Hessian in GL, HL. */
const derivAt = (p: ContactParticle, X0: Float64Array, formulation: number, avgRadius: number): number => {
  const R = p.R, xc = p.x;
  // R^T (X0 - xc)
  const d0 = X0[0] - xc[0], d1 = X0[1] - xc[1], d2 = X0[2] - xc[2];
  const f = derivLocal(R[0] * d0 + R[3] * d1 + R[6] * d2, R[1] * d0 + R[4] * d1 + R[7] * d2, R[2] * d0 + R[5] * d1 + R[8] * d2,
    p.shape, p.block, p.flag);
  return formulation === FORMULATION_GEOMETRIC ? regularize(p.block[0], avgRadius, f, GL, HL) : f;
};

/** Shape function, global gradient R g and global Hessian R H R^T (the C++ times3_transpose then times3) of particle p at X0. */
const derivGlobal = (p: ContactParticle, X0: Float64Array, formulation: number, avgRadius: number, out: Deriv): void => {
  out.f = derivAt(p, X0, formulation, avgRadius);
  const R = p.R, g = out.g, h = out.h;
  g[0] = R[0] * GL[0] + R[1] * GL[1] + R[2] * GL[2];
  g[1] = R[3] * GL[0] + R[4] * GL[1] + R[5] * GL[2];
  g[2] = R[6] * GL[0] + R[7] * GL[1] + R[8] * GL[2];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) TT[3 * i + j] = HL[3 * i] * R[3 * j] + HL[3 * i + 1] * R[3 * j + 1] + HL[3 * i + 2] * R[3 * j + 2];
  }
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) h[3 * i + j] = R[3 * i] * TT[j] + R[3 * i + 1] * TT[3 + j] + R[3 * i + 2] * TT[6 + j];
  }
};

/** Residual (4 components) and its norm. */
const computeResidual = (fi: number, gi: Float64Array, fj: number, gj: Float64Array, mu2: number, residual: Float64Array, formulation: number, radiusScale: number): number => {
  for (let k = 0; k < 3; k++) residual[k] = mu2 * gj[k] + gi[k];
  residual[3] = fi - fj;
  const giMag = 0.5 * (norm2(gi) + norm2(gj));
  const spatial = (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]) / giMag;
  let denom = formulation === FORMULATION_GEOMETRIC ? radiusScale : fi + 1;
  if (Math.abs(denom) < 1e-12) denom = 1;
  return spatial + (residual[3] * residual[3]) / (denom * denom);
};

/** Jacobian of the residual, row-major 4x4. */
const computeJacobian = (gi: Float64Array, hi: Float64Array, gj: Float64Array, hj: Float64Array, mu2: number, J: Float64Array): void => {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) J[r * 4 + c] = hi[3 * r + c] + mu2 * hj[3 * r + c];
    J[r * 4 + 3] = gj[r];
  }
  for (let c = 0; c < 3; c++) J[12 + c] = gi[c] - gj[c];
  J[15] = 0;
};

/** Residual, Jacobian and shape values at the trial point X = (x, y, z, mu^2). */
const residualAndJacobian = (pi: ContactParticle, pj: ContactParticle, X: Float64Array, formulation: number, avgI: number, avgJ: number): number => {
  derivGlobal(pi, X, formulation, avgI, DI);
  derivGlobal(pj, X, formulation, avgJ, DJ);
  SHAPEF[0] = DI.f;
  SHAPEF[1] = DJ.f;
  computeJacobian(DI.g, DI.h, DJ.g, DJ.h, X[3], JAC);
  return computeResidual(DI.f, DI.g, DJ.f, DJ.g, X[3], RES, formulation, (avgI + avgJ) * 0.5);
};

/** out = R v / |R v| (the C++ normalize3 of a global gradient). */
const normalizeRotated = (R: Float64Array, v: Float64Array, out: Float64Array): void => {
  const a0 = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  const a1 = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  const a2 = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
  const n = Math.sqrt(a0 * a0 + a1 * a1 + a2 * a2);
  out[0] = a0 / n; out[1] = a1 / n; out[2] = a2 / n;
};

/** Normal at the contact point from particle i's gradient (the C++ surface-search gradient in the global frame). */
const normalFrom = (pi: ContactParticle, X0: Float64Array, nij: Float64Array): void => {
  const R = pi.R, xc = pi.x;
  const d0 = X0[0] - xc[0], d1 = X0[1] - xc[1], d2 = X0[2] - xc[2];
  surfLocal(R[0] * d0 + R[3] * d1 + R[6] * d2, R[1] * d0 + R[4] * d1 + R[7] * d2, R[2] * d0 + R[5] * d1 + R[8] * d2,
    pi.shape, pi.block, pi.flag);
  normalizeRotated(R, SG, nij);
};

/**
 * Midway contact point X0 (x, y, z, mu^2; updated in place) and normal nij of particles i and j. Returns 0 when the particles
 * overlap (converged, both shape functions <= 0), 1 when they are separated (converged, a shape function > 0), 2 when the
 * Newton iteration did not converge and no sign was found (the C++ return codes).
 */
export const determineContactPoint = (pi: ContactParticle, pj: ContactParticle, X0: Float64Array, nij: Float64Array, formulation: number): number => {
  const l0 = pj.x[0] - pi.x[0], l1 = pj.x[1] - pi.x[1], l2 = pj.x[2] - pi.x[2];
  const lsq = l0 * l0 + l1 * l1 + l2 * l2;
  let converged = false;
  let avgI = 1, avgJ = 1;
  const maxStep = Math.sqrt(lsq) * 0.2;
  if (formulation === FORMULATION_GEOMETRIC) {
    avgI = (pi.shape[0] + pi.shape[1] + pi.shape[2]) * (1 / 3);
    avgJ = (pj.shape[0] + pj.shape[1] + pj.shape[2]) * (1 / 3);
  }
  const shapefunc = SHAPEF, residual = RES, J = JAC, rhs = RHS;
  let norm = residualAndJacobian(pi, pj, X0, formulation, avgI, avgJ);
  if (norm < TOL_NR_RES) {
    normalFrom(pi, X0, nij);
    return shapefunc[0] > 0 || shapefunc[1] > 0 ? 1 : 0;
  }
  for (let iter = 0; iter < ITERMAX_NR; iter++) {
    const normOld = norm;
    AS.set(J);
    for (let k = 0; k < 4; k++) BF[k] = -residual[k];
    let solved = solve4x4(AS, BF);
    for (let k = 0; k < 4; k++) if (!Number.isFinite(BF[k]) || Math.abs(BF[k]) > MAX_B_FAST) { solved = false; break; }
    rhs.set(BF);
    if (!solved) {
      A2.set(J);
      for (let k = 0; k < 4; k++) B2[k] = -residual[k];
      const trace = J[0] + J[5] + J[10];
      const dw = Math.max(TIKHONOV_SCALE * trace, TIKHONOV_SCALE);
      A2[0] += dw; A2[5] += dw; A2[10] += dw;
      if (solve4x4(A2, B2)) rhs.set(B2);
    }
    // backtracking line search
    let a = 1;
    const srn = Math.sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
    if (srn > maxStep) {
      const s = maxStep / srn;
      rhs[0] *= s; rhs[1] *= s; rhs[2] *= s;
    }
    let iterLs = 0;
    for (iterLs = 0; iterLs < ITERMAX_LS; iterLs++) {
      for (let k = 0; k < 4; k++) XL[k] = X0[k] + a * rhs[k];
      derivGlobal(pi, XL, formulation, avgI, DI);
      derivGlobal(pj, XL, formulation, avgJ, DJ);
      shapefunc[0] = DI.f;
      shapefunc[1] = DJ.f;
      norm = computeResidual(DI.f, DI.g, DJ.f, DJ.g, XL[3], residual, formulation, (avgI + avgJ) / 2);
      if (norm <= TOL_NR_RES && norm2(rhs) * a * a <= TOL_NR_POS * lsq) {
        converged = true;
        const g = DI.g;
        const n = Math.sqrt(norm2(g));
        nij[0] = g[0] / n; nij[1] = g[1] / n; nij[2] = g[2] / n;
        break;
      } else if (norm > normOld - PARAMETER_LS * a * normOld) {
        a *= CUTBACK_LS;
      } else {
        computeJacobian(DI.g, DI.h, DJ.g, DJ.h, XL[3], J);
        break;
      }
    }
    if (iterLs === ITERMAX_LS) {
      for (let k = 0; k < 4; k++) X0[k] += rhs[k];
      norm = residualAndJacobian(pi, pj, X0, formulation, avgI, avgJ);
      if (norm < TOL_NR_RES) {
        converged = true;
        const f = derivAt(pi, X0, formulation, avgI);
        if (formulation === FORMULATION_GEOMETRIC) shapefunc[0] = f;
        normalizeRotated(pi.R, GL, nij);
      }
    } else {
      for (let k = 0; k < 4; k++) X0[k] = X0[k] + a * rhs[k];
    }
    if (converged) break;
  }
  if (!converged) return shapefunc[0] > 0 || shapefunc[1] > 0 ? 1 : 2;
  return shapefunc[0] > 0 || shapefunc[1] > 0 ? 1 : 0;
};

/**
 * Distance from the contact point g along the normal n to the particle surface (positive, outward): analytic for the
 * ellipsoid, Newton iteration on the distance estimator otherwise (the C++ compute_overlap_distance).
 */
export const overlapDistance = (shape: Float64Array, block: Float64Array, R: Float64Array, flag: number,
  g0: number, g1: number, g2: number, n0: number, n1: number, n2: number, center: Float64Array): number => {
  const d0 = g0 - center[0], d1 = g1 - center[1], d2 = g2 - center[2];
  const p0 = R[0] * d0 + R[3] * d1 + R[6] * d2, p1 = R[1] * d0 + R[4] * d1 + R[7] * d2, p2 = R[2] * d0 + R[5] * d1 + R[8] * d2;
  const m0 = R[0] * n0 + R[3] * n1 + R[6] * n2, m1 = R[1] * n0 + R[4] * n1 + R[7] * n2, m2 = R[2] * n0 + R[5] * n1 + R[8] * n2;
  if (flag === 0) {
    const ai2 = 1 / (shape[0] * shape[0]), bi2 = 1 / (shape[1] * shape[1]), ci2 = 1 / (shape[2] * shape[2]);
    const A = m0 * m0 * ai2 + m1 * m1 * bi2 + m2 * m2 * ci2;
    const B = 2 * (p0 * m0 * ai2 + p1 * m1 * bi2 + p2 * m2 * ci2);
    const C = p0 * p0 * ai2 + p1 * p1 * bi2 + p2 * p2 * ci2 - 1;
    let delta = B * B - 4 * A * C;
    if (delta < 0) delta = 0;
    return (-B + Math.sqrt(delta)) / (2 * A);
  }
  let overlap = 0;
  for (let iter = 0; iter < ITERMAX_OVERLAP; iter++) {
    const f = surfLocal(p0 + overlap * m0, p1 + overlap * m1, p2 + overlap * m2, shape, block, flag);
    if (Math.abs(f) < TOL_OVERLAP) break;
    const slope = SG[0] * m0 + SG[1] * m1 + SG[2] * m2;
    if (Math.abs(slope) < MINSLOPE_OVERLAP) break;
    overlap -= f / slope;
  }
  return overlap;
};

/** derivLocal at the global surface point s of a particle at xc with rotation R (gradient and Hessian in GL, HL). */
const derivSurface = (shape: Float64Array, block: Float64Array, flag: number, R: Float64Array, s0: number, s1: number, s2: number, xc: Float64Array): void => {
  const d0 = s0 - xc[0], d1 = s1 - xc[1], d2 = s2 - xc[2];
  derivLocal(R[0] * d0 + R[3] * d1 + R[6] * d2, R[1] * d0 + R[4] * d1 + R[7] * d2, R[2] * d0 + R[5] * d1 + R[8] * d2, shape, block, flag);
};

/** Mean curvature of the surface at a global surface point (the C++ mean_curvature_superellipsoid). */
export const meanCurvature = (shape: Float64Array, block: Float64Array, flag: number, R: Float64Array, s0: number, s1: number, s2: number, xc: Float64Array): number => {
  derivSurface(shape, block, flag, R, s0, s1, s2, xc);
  const g = GL, h = HL;
  const nn = Math.sqrt(norm2(g));
  const u0 = g[0] / nn, u1 = g[1] / nn, u2 = g[2] / nn;
  const t0 = h[0] * u0 + h[1] * u1 + h[2] * u2;
  const t1 = h[3] * u0 + h[4] * u1 + h[5] * u2;
  const t2 = h[6] * u0 + h[7] * u1 + h[8] * u2;
  const Fmag = Math.sqrt(norm2(g));
  return Math.abs((u0 * t0 + u1 * t1 + u2 * t2) - (h[0] + h[4] + h[8])) / (2 * Fmag);
};

/** Gaussian curvature coefficient at a global surface point (the C++ gaussian_curvature_superellipsoid). */
export const gaussianCurvature = (shape: Float64Array, block: Float64Array, flag: number, R: Float64Array, s0: number, s1: number, s2: number, xc: Float64Array): number => {
  derivSurface(shape, block, flag, R, s0, s1, s2, xc);
  const h = HL;
  const Fmag = Math.sqrt(norm2(GL));
  const fx = GL[0], fy = GL[1], fz = GL[2];
  // det4_M44_zero of [[h00, h01, h02, fx], [h01, h11, h12, fy], [h02, h12, h22, fz], [fx, fy, fz, 0]]
  const det = -fx * det3(h[1], h[2], fx, h[4], h[5], fy, h[5], h[8], fz)
    + fy * det3(h[0], h[2], fx, h[1], h[5], fy, h[2], h[8], fz)
    - fz * det3(h[0], h[1], fx, h[1], h[4], fy, h[2], h[5], fz);
  const K = -det / (Fmag * Fmag * Fmag * Fmag);
  return Math.sqrt(Math.abs(K));
};

/**
 * Oriented bounding box test (the C++ check_oriented_bounding_boxes): returns -1 when no separating axis was found (the
 * boxes may overlap), else the separating axis index (0..14), which the caller caches.
 */
export const checkOrientedBoundingBoxes = (p1: ContactParticle, p2: ContactParticle, axis: number): number => {
  const R1 = p1.R, R2 = p2.R, C = OC, A = OA;
  // R1^T R2 and its absolute values
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) C[3 * i + j] = R1[i] * R2[j] + R1[3 + i] * R2[3 + j] + R1[6 + i] * R2[6 + j];
  }
  for (let k = 0; k < 9; k++) A[k] = Math.abs(C[k]);
  const d0 = p2.x[0] - p1.x[0], d1 = p2.x[1] - p1.x[1], d2 = p2.x[2] - p1.x[2];
  OC1[0] = R1[0] * d0 + R1[3] * d1 + R1[6] * d2; OC1[1] = R1[1] * d0 + R1[4] * d1 + R1[7] * d2; OC1[2] = R1[2] * d0 + R1[5] * d1 + R1[8] * d2;
  OC2[0] = R2[0] * d0 + R2[3] * d1 + R2[6] * d2; OC2[1] = R2[1] * d0 + R2[4] * d1 + R2[7] * d2; OC2[2] = R2[2] * d0 + R2[5] * d1 + R2[8] * d2;
  const s1 = p1.shape, s2 = p2.shape;
  if (axisSeparated(axis, C, A, OC1, OC2, s1, s2)) return axis;
  for (let id = 0; id < 15; id++) {
    if (id === axis) continue;
    if (axisSeparated(id, C, A, OC1, OC2, s1, s2)) return id;
  }
  return -1;
};

/** Separation along one of the 15 axes (faces of both boxes, then edge cross products). */
const axisSeparated = (id: number, C: Float64Array, A: Float64Array, c1: Float64Array, c2: Float64Array, a: Float64Array, b: Float64Array): boolean => {
  let R1: number, R2: number, R: number;
  switch (id) {
    case 0: R1 = a[0]; R2 = b[0] * A[0] + b[1] * A[1] + b[2] * A[2]; R = Math.abs(c1[0]); break;
    case 1: R1 = a[1]; R2 = b[0] * A[3] + b[1] * A[4] + b[2] * A[5]; R = Math.abs(c1[1]); break;
    case 2: R1 = a[2]; R2 = b[0] * A[6] + b[1] * A[7] + b[2] * A[8]; R = Math.abs(c1[2]); break;
    case 3: R1 = a[0] * A[0] + a[1] * A[3] + a[2] * A[6]; R2 = b[0]; R = Math.abs(c2[0]); break;
    case 4: R1 = a[0] * A[1] + a[1] * A[4] + a[2] * A[7]; R2 = b[1]; R = Math.abs(c2[1]); break;
    case 5: R1 = a[0] * A[2] + a[1] * A[5] + a[2] * A[8]; R2 = b[2]; R = Math.abs(c2[2]); break;
    case 6: R1 = a[1] * A[6] + a[2] * A[3]; R2 = b[1] * A[2] + b[2] * A[1]; R = Math.abs(c1[2] * C[3] - c1[1] * C[6]); break;
    case 7: R1 = a[1] * A[7] + a[2] * A[4]; R2 = b[0] * A[2] + b[2] * A[0]; R = Math.abs(c1[2] * C[4] - c1[1] * C[7]); break;
    case 8: R1 = a[1] * A[8] + a[2] * A[5]; R2 = b[0] * A[1] + b[1] * A[0]; R = Math.abs(c1[2] * C[5] - c1[1] * C[8]); break;
    case 9: R1 = a[0] * A[6] + a[2] * A[0]; R2 = b[1] * A[5] + b[2] * A[4]; R = Math.abs(c1[0] * C[6] - c1[2] * C[0]); break;
    case 10: R1 = a[0] * A[7] + a[2] * A[1]; R2 = b[0] * A[5] + b[2] * A[3]; R = Math.abs(c1[0] * C[7] - c1[2] * C[1]); break;
    case 11: R1 = a[0] * A[8] + a[2] * A[2]; R2 = b[0] * A[4] + b[1] * A[3]; R = Math.abs(c1[0] * C[8] - c1[2] * C[2]); break;
    case 12: R1 = a[0] * A[3] + a[1] * A[0]; R2 = b[1] * A[8] + b[2] * A[7]; R = Math.abs(c1[1] * C[0] - c1[0] * C[3]); break;
    case 13: R1 = a[0] * A[4] + a[1] * A[1]; R2 = b[0] * A[8] + b[2] * A[6]; R = Math.abs(c1[1] * C[1] - c1[0] * C[4]); break;
    default: R1 = a[0] * A[5] + a[1] * A[2]; R2 = b[0] * A[7] + b[1] * A[6]; R = Math.abs(c1[1] * C[2] - c1[0] * C[5]); break;
  }
  return R > R1 + R2;
};

