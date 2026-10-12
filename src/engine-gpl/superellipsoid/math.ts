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
 */

export type V3 = [number, number, number];
export type M3 = [V3, V3, V3];

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

export interface Deriv {
  f: number;
  g: V3;
  h: M3;
}

const zero3 = (): M3 => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];

/** R^T v. */
export const transposeMatvec = (R: M3, v: V3): V3 => [
  R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
  R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
  R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
];
/** R v. */
export const matvec = (R: M3, v: V3): V3 => [
  R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
  R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
  R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
];
/** R^T S (3x3). */
export const transposeTimes = (A: M3, B: M3): M3 => {
  const C = zero3();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[0][i] * B[0][j] + A[1][i] * B[1][j] + A[2][i] * B[2][j];
  return C;
};
/** A B^T (3x3). */
const timesTranspose = (A: M3, B: M3): M3 => {
  const C = zero3();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[j][0] + A[i][1] * B[j][1] + A[i][2] * B[j][2];
  return C;
};
/** A B (3x3). */
const times = (A: M3, B: M3): M3 => {
  const C = zero3();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
};
/** Global Hessian R H R^T (the C++ times3_transpose then times3). */
const rotateHess = (R: M3, H: M3): M3 => times(R, timesTranspose(H, R));

const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm2 = (a: V3): number => a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
const normalize = (a: V3): V3 => {
  const n = Math.sqrt(norm2(a));
  return [a[0] / n, a[1] / n, a[2] / n];
};

/** Determinant of the 4x4 matrix with M[3][3] = 0 (the C++ det4_M44_zero). */
export const det4M44Zero = (m: number[][]): number => {
  const det3 = (r: number[][]): number =>
    r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) -
    r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
    r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
  const m41 = [[m[0][1], m[0][2], m[0][3]], [m[1][1], m[1][2], m[1][3]], [m[2][1], m[2][2], m[2][3]]];
  const m42 = [[m[0][0], m[0][2], m[0][3]], [m[1][0], m[1][2], m[1][3]], [m[2][0], m[2][2], m[2][3]]];
  const m43 = [[m[0][0], m[0][1], m[0][3]], [m[1][0], m[1][1], m[1][3]], [m[2][0], m[2][1], m[2][3]]];
  return -m[3][0] * det3(m41) + m[3][1] * det3(m42) - m[3][2] * det3(m43);
};

/**
 * Solves the 4x4 system A x = b in place (A row-major, 16 entries; b 4 entries) with partial pivoting, as the C++
 * solve_4x4_robust_unrolled. Returns false on a zero pivot.
 */
export const solve4x4 = (A: number[], b: number[]): boolean => {
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
    [b[0], b[p]] = [b[p], b[0]];
    for (let k = 0; k < 4; k++) [A[k], A[ro + k]] = [A[ro + k], A[k]];
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
    [b[1], b[p]] = [b[p], b[1]];
    for (let k = 1; k < 4; k++) [A[4 + k], A[ro + k]] = [A[ro + k], A[4 + k]];
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
    [b[2], b[3]] = [b[3], b[2]];
    [A[10], A[14]] = [A[14], A[10]];
    [A[11], A[15]] = [A[15], A[11]];
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
 * Shape functions in the local frame of a particle (shape = half-diameters a, b, c; block = n1, n2)
 * ------------------------------------------------------------------------------------------------------------------- */

const signOf = (v: number): number => (v > 0 ? 1 : -1);

/** Superquadric (block types 2): n1 along z, n2 in the x-y plane. */
export const derivSuperquad = (x: V3, shape: V3, block: [number, number]): Deriv => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x[0] * aInv), yb = Math.abs(x[1] * bInv), zc = Math.abs(x[2] * cInv);
  const n1 = block[0], n2 = block[1];
  const xaP2m2 = Math.pow(xa, n2 - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = Math.pow(yb, n2 - 2), ybP2m1 = ybP2m2 * yb;
  const nu = xaP2m1 * xa + ybP2m1 * yb;
  const nuP2m2 = Math.pow(nu, n1 / n2 - 2), nuP2m1 = nuP2m2 * nu;
  const zcP1m2 = Math.pow(zc, n1 - 2), zcP1m1 = zcP1m2 * zc;
  const sx = signOf(x[0]), sy = signOf(x[1]), sz = signOf(x[2]);
  const g: V3 = [n1 * aInv * xaP2m1 * nuP2m1 * sx, n1 * bInv * ybP2m1 * nuP2m1 * sy, n1 * cInv * zcP1m1 * sz];
  const h = zero3();
  h[0][0] = aInv * aInv * (n1 * (n2 - 1) * xaP2m2 * nuP2m1 + (n1 - n2) * n1 * (xaP2m1 * xaP2m1) * nuP2m2);
  h[1][1] = bInv * bInv * (n1 * (n2 - 1) * ybP2m2 * nuP2m1 + (n1 - n2) * n1 * (ybP2m1 * ybP2m1) * nuP2m2);
  h[0][1] = h[1][0] = aInv * bInv * (n1 - n2) * n1 * xaP2m1 * ybP2m1 * nuP2m2 * (sx * sy);
  h[2][2] = cInv * cInv * n1 * (n1 - 1) * zcP1m2;
  return { f: nuP2m1 * nu + zcP1m1 * zc - 1, g, h };
};
/** Exponent n1 = n2 = n (block type 1). */
export const derivEqual = (x: V3, shape: V3, n: number): Deriv => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x[0] * aInv), yb = Math.abs(x[1] * bInv), zc = Math.abs(x[2] * cInv);
  const xaP2m2 = Math.pow(xa, n - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = Math.pow(yb, n - 2), ybP2m1 = ybP2m2 * yb;
  const zcP2m2 = Math.pow(zc, n - 2), zcP2m1 = zcP2m2 * zc;
  const g: V3 = [n * aInv * xaP2m1 * signOf(x[0]), n * bInv * ybP2m1 * signOf(x[1]), n * cInv * zcP2m1 * signOf(x[2])];
  const h = zero3();
  h[0][0] = aInv * aInv * n * (n - 1) * xaP2m2;
  h[1][1] = bInv * bInv * n * (n - 1) * ybP2m2;
  h[2][2] = cInv * cInv * n * (n - 1) * zcP2m2;
  return { f: xaP2m1 * xa + ybP2m1 * yb + zcP2m1 * zc - 1, g, h };
};

/** Ellipsoid (block type 0). */
export const derivEllipsoid = (x: V3, shape: V3): Deriv => {
  const a = 2 / (shape[0] * shape[0]), b = 2 / (shape[1] * shape[1]), c = 2 / (shape[2] * shape[2]);
  const g: V3 = [a * x[0], b * x[1], c * x[2]];
  const h = zero3();
  h[0][0] = a; h[1][1] = b; h[2][2] = c;
  return { f: 0.5 * (g[0] * x[0] + g[1] * x[1] + g[2] * x[2]) - 1, g, h };
};

/** Local shape function and derivatives for block type flag. */
export const derivLocal = (x: V3, shape: V3, block: [number, number], flag: number): Deriv => {
  if (flag === 0) return derivEllipsoid(x, shape);
  if (flag === 1) return derivEqual(x, shape, block[0]);
  return derivSuperquad(x, shape, block);
};

/** Value and gradient only, for the surface search (the C++ shape_and_gradient_*_surfacesearch). */
export const surfSuperquad = (x: V3, shape: V3, block: [number, number]): { f: number; g: V3 } => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x[0] * aInv), yb = Math.abs(x[1] * bInv), zc = Math.abs(x[2] * cInv);
  const n1 = block[0], n2 = block[1];
  const xaP2m2 = Math.pow(xa, n2 - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = Math.pow(yb, n2 - 2), ybP2m1 = ybP2m2 * yb;
  const nu = xaP2m1 * xa + ybP2m1 * yb;
  const nuP2m2 = Math.pow(nu, n1 / n2 - 2), nuP2m1 = nuP2m2 * nu;
  const zcP1m2 = Math.pow(zc, n1 - 2), zcP1m1 = zcP1m2 * zc;
  const sx = signOf(x[0]), sy = signOf(x[1]), sz = signOf(x[2]);
  const g: V3 = [n1 * aInv * xaP2m1 * nuP2m1 * sx, n1 * bInv * ybP2m1 * nuP2m1 * sy, n1 * cInv * zcP1m1 * sz];
  const F = nuP2m1 * nu + zcP1m1 * zc;
  const scale = Math.pow(F, 1 / n1 - 1) / n1;
  return { f: Math.pow(F, 1 / n1) - 1, g: [g[0] * scale, g[1] * scale, g[2] * scale] };
};

export const surfEqual = (x: V3, shape: V3, n: number): { f: number; g: V3 } => {
  const aInv = 1 / shape[0], bInv = 1 / shape[1], cInv = 1 / shape[2];
  const xa = Math.abs(x[0] * aInv), yb = Math.abs(x[1] * bInv), zc = Math.abs(x[2] * cInv);
  const xaP2m2 = Math.pow(xa, n - 2), xaP2m1 = xaP2m2 * xa;
  const ybP2m2 = Math.pow(yb, n - 2), ybP2m1 = ybP2m2 * yb;
  const zcP2m2 = Math.pow(zc, n - 2), zcP2m1 = zcP2m2 * zc;
  const g: V3 = [n * aInv * xaP2m1 * signOf(x[0]), n * bInv * ybP2m1 * signOf(x[1]), n * cInv * zcP2m1 * signOf(x[2])];
  const F = xaP2m1 * xa + ybP2m1 * yb + zcP2m1 * zc;
  const scale = Math.pow(F, 1 / n - 1) / n;
  return { f: Math.pow(F, 1 / n) - 1, g: [g[0] * scale, g[1] * scale, g[2] * scale] };
};

/** Surface search derivative for flags 0 and 1 (the C++ uses the n1 = n2 form there). */
const surfLocal = (x: V3, shape: V3, block: [number, number], flag: number): { f: number; g: V3 } =>
  flag <= 1 ? surfEqual(x, shape, block[0]) : surfSuperquad(x, shape, block);

/**
 * Generalised shape function G = (F + 1)^(1/n1) - 1 scaled by the average radius (geometric formulation). Modifies the
 * gradient and Hessian in place and returns the new value.
 */
const regularize = (n1: number, avgRadius: number, value: number, grad: V3, hess: M3): number => {
  const base = Math.max(value + 1, REGULARIZATION_EPSILON);
  const invF = 1 / base;
  const invN1 = 1 / n1;
  const Fp = Math.pow(base, invN1);
  const scaleGrad = avgRadius * invN1 * Fp * invF;
  const scaleHessAdd = scaleGrad * (invN1 - 1) * invF;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) hess[i][j] = hess[i][j] * scaleGrad + scaleHessAdd * (grad[i] * grad[j]);
  }
  for (let i = 0; i < 3; i++) grad[i] *= scaleGrad;
  return avgRadius * (Fp - 1);
};

/** Shape function, global gradient and global Hessian of particle at centre xc with rotation R. */
export const derivGlobal = (xc: V3, R: M3, shape: V3, block: [number, number], flag: number, X0: V3 | number[], formulation: number, avgRadius: number): Deriv => {
  const xl = transposeMatvec(R, sub([X0[0], X0[1], X0[2]], xc));
  const d = derivLocal(xl, shape, block, flag);
  let f = d.f;
  let gl = d.g;
  let hl = d.h;
  if (formulation === FORMULATION_GEOMETRIC) {
    gl = [...gl] as V3;
    hl = hl.map((r) => [...r]) as M3;
    f = regularize(block[0], avgRadius, f, gl, hl);
  }
  return { f, g: matvec(R, gl), h: rotateHess(R, hl) };
};

/** Residual (4 components) and its norm. */
const computeResidual = (fi: number, gi: V3, fj: number, gj: V3, mu2: number, residual: number[], formulation: number, radiusScale: number): number => {
  for (let k = 0; k < 3; k++) residual[k] = mu2 * gj[k] + gi[k];
  residual[3] = fi - fj;
  const giMag = 0.5 * (norm2(gi) + norm2(gj));
  const spatial = (residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]) / giMag;
  let denom = formulation === FORMULATION_GEOMETRIC ? radiusScale : fi + 1;
  if (Math.abs(denom) < 1e-12) denom = 1;
  return spatial + (residual[3] * residual[3]) / (denom * denom);
};

/** Jacobian of the residual, row-major 4x4. */
const computeJacobian = (gi: V3, hi: M3, gj: V3, hj: M3, mu2: number, J: number[]): void => {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) J[r * 4 + c] = hi[r][c] + mu2 * hj[r][c];
    J[r * 4 + 3] = gj[r];
  }
  for (let c = 0; c < 3; c++) J[12 + c] = gi[c] - gj[c];
  J[15] = 0;
};

export interface ContactParticle {
  x: V3;
  R: M3;
  shape: V3;
  block: [number, number];
  flag: number;
}

/** Residual, Jacobian and shape values at the trial point X = (x, y, z, mu^2). */
const residualAndJacobian = (pi: ContactParticle, pj: ContactParticle, X: number[], shapefunc: number[], residual: number[], J: number[], formulation: number, avgI: number, avgJ: number): number => {
  const di = derivGlobal(pi.x, pi.R, pi.shape, pi.block, pi.flag, X, formulation, avgI);
  const dj = derivGlobal(pj.x, pj.R, pj.shape, pj.block, pj.flag, X, formulation, avgJ);
  shapefunc[0] = di.f;
  shapefunc[1] = dj.f;
  computeJacobian(di.g, di.h, dj.g, dj.h, X[3], J);
  return computeResidual(di.f, di.g, dj.f, dj.g, X[3], residual, formulation, (avgI + avgJ) * 0.5);
};

/** Normal at the contact point from particle i's gradient (the C++ surface-search gradient in the global frame). */
const normalFrom = (pi: ContactParticle, X0: number[]): V3 => {
  const xl = transposeMatvec(pi.R, sub([X0[0], X0[1], X0[2]], pi.x));
  const sg = surfLocal(xl, pi.shape, pi.block, pi.flag).g;
  return normalize(matvec(pi.R, sg));
};

/**
 * Midway contact point X0 (x, y, z, mu^2; updated in place) and normal nij of particles i and j. Returns 0 when the particles
 * overlap (converged, both shape functions <= 0), 1 when they are separated (converged, a shape function > 0), 2 when the
 * Newton iteration did not converge and no sign was found (the C++ return codes).
 */
export const determineContactPoint = (pi: ContactParticle, pj: ContactParticle, X0: number[], nij: V3, formulation: number): number => {
  const lsq = norm2(sub(pj.x, pi.x));
  let converged = false;
  let avgI = 1, avgJ = 1;
  const maxStep = Math.sqrt(lsq) * 0.2;
  if (formulation === FORMULATION_GEOMETRIC) {
    avgI = (pi.shape[0] + pi.shape[1] + pi.shape[2]) * (1 / 3);
    avgJ = (pj.shape[0] + pj.shape[1] + pj.shape[2]) * (1 / 3);
  }
  const shapefunc = [0, 0];
  const residual = [0, 0, 0, 0];
  const J = new Array<number>(16).fill(0);
  let norm = residualAndJacobian(pi, pj, X0, shapefunc, residual, J, formulation, avgI, avgJ);
  if (norm < TOL_NR_RES) {
    const n = normalFrom(pi, X0);
    for (let k = 0; k < 3; k++) nij[k] = n[k];
    return shapefunc[0] > 0 || shapefunc[1] > 0 ? 1 : 0;
  }
  for (let iter = 0; iter < ITERMAX_NR; iter++) {
    const normOld = norm;
    const A = J.slice();
    const bf = [-residual[0], -residual[1], -residual[2], -residual[3]];
    let solved = solve4x4(A, bf);
    if (bf.some((v) => !Number.isFinite(v) || Math.abs(v) > MAX_B_FAST)) solved = false;
    const rhs = bf.slice();
    if (!solved) {
      const A2 = J.slice();
      const b2 = [-residual[0], -residual[1], -residual[2], -residual[3]];
      const trace = J[0] + J[5] + J[10];
      const dw = Math.max(TIKHONOV_SCALE * trace, TIKHONOV_SCALE);
      A2[0] += dw; A2[5] += dw; A2[10] += dw;
      if (solve4x4(A2, b2)) {
        for (let k = 0; k < 4; k++) rhs[k] = b2[k];
      }
    }
    // backtracking line search
    let a = 1;
    const srn = Math.sqrt(rhs[0] * rhs[0] + rhs[1] * rhs[1] + rhs[2] * rhs[2]);
    if (srn > maxStep) {
      const s = maxStep / srn;
      rhs[0] *= s; rhs[1] *= s; rhs[2] *= s;
    }
    let iterLs = 0;
    let gi: V3, gj: V3;
    let hi: M3, hj: M3;
    for (iterLs = 0; iterLs < ITERMAX_LS; iterLs++) {
      const XL = [X0[0] + a * rhs[0], X0[1] + a * rhs[1], X0[2] + a * rhs[2], X0[3] + a * rhs[3]];
      const di = derivGlobal(pi.x, pi.R, pi.shape, pi.block, pi.flag, XL, formulation, avgI);
      const dj = derivGlobal(pj.x, pj.R, pj.shape, pj.block, pj.flag, XL, formulation, avgJ);
      shapefunc[0] = di.f;
      shapefunc[1] = dj.f;
      gi = di.g; gj = dj.g;
      norm = computeResidual(di.f, gi, dj.f, gj, XL[3], residual, formulation, (avgI + avgJ) / 2);
      if (norm <= TOL_NR_RES && norm2(rhs as V3) * a * a <= TOL_NR_POS * lsq) {
        converged = true;
        const n = normalize(gi);
        for (let k = 0; k < 3; k++) nij[k] = n[k];
        break;
      } else if (norm > normOld - PARAMETER_LS * a * normOld) {
        a *= CUTBACK_LS;
      } else {
        hi = di.h; hj = dj.h;
        computeJacobian(gi, hi, gj, hj, XL[3], J);
        break;
      }
    }
    if (iterLs === ITERMAX_LS) {
      for (let k = 0; k < 4; k++) X0[k] += rhs[k];
      norm = residualAndJacobian(pi, pj, X0, shapefunc, residual, J, formulation, avgI, avgJ);
      if (norm < TOL_NR_RES) {
        converged = true;
        const xl = transposeMatvec(pi.R, sub([X0[0], X0[1], X0[2]], pi.x));
        const d = derivLocal(xl, pi.shape, pi.block, pi.flag);
        let gl = d.g;
        if (formulation === FORMULATION_GEOMETRIC) {
          gl = [...gl] as V3;
          const hd = d.h.map((r) => [...r]) as M3;
          shapefunc[0] = regularize(pi.block[0], avgI, d.f, gl, hd);
        }
        const n = normalize(matvec(pi.R, gl));
        for (let k = 0; k < 3; k++) nij[k] = n[k];
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
 * Distance from the contact point along the normal to the particle surface (positive, outward): analytic for the ellipsoid,
 * Newton iteration on the distance estimator otherwise (the C++ compute_overlap_distance).
 */
export const overlapDistance = (shape: V3, block: [number, number], R: M3, flag: number, globalPoint: V3, globalNormal: V3, center: V3): number => {
  const localPoint = transposeMatvec(R, sub(globalPoint, center));
  const localNormal = transposeMatvec(R, globalNormal);
  if (flag === 0) {
    const ai2 = 1 / (shape[0] * shape[0]), bi2 = 1 / (shape[1] * shape[1]), ci2 = 1 / (shape[2] * shape[2]);
    const A = localNormal[0] * localNormal[0] * ai2 + localNormal[1] * localNormal[1] * bi2 + localNormal[2] * localNormal[2] * ci2;
    const B = 2 * (localPoint[0] * localNormal[0] * ai2 + localPoint[1] * localNormal[1] * bi2 + localPoint[2] * localNormal[2] * ci2);
    const C = localPoint[0] * localPoint[0] * ai2 + localPoint[1] * localPoint[1] * bi2 + localPoint[2] * localPoint[2] * ci2 - 1;
    let delta = B * B - 4 * A * C;
    if (delta < 0) delta = 0;
    return (-B + Math.sqrt(delta)) / (2 * A);
  }
  let overlap = 0;
  for (let iter = 0; iter < ITERMAX_OVERLAP; iter++) {
    const cp: V3 = [
      localPoint[0] + overlap * localNormal[0],
      localPoint[1] + overlap * localNormal[1],
      localPoint[2] + overlap * localNormal[2],
    ];
    const s = surfLocal(cp, shape, block, flag);
    if (Math.abs(s.f) < TOL_OVERLAP) break;
    const slope = s.g[0] * localNormal[0] + s.g[1] * localNormal[1] + s.g[2] * localNormal[2];
    if (Math.abs(slope) < MINSLOPE_OVERLAP) break;
    overlap -= s.f / slope;
  }
  return overlap;
};

/** Mean curvature of the surface at a global surface point (the C++ mean_curvature_superellipsoid). */
export const meanCurvature = (shape: V3, block: [number, number], flag: number, R: M3, surfPoint: V3, xc: V3): number => {
  const xl = transposeMatvec(R, sub(surfPoint, xc));
  const d = derivLocal(xl, shape, block, flag);
  const normal = normalize(d.g);
  const tv = matvec(d.h, normal);
  const Fmag = Math.sqrt(norm2(d.g));
  return Math.abs(dot(normal, tv) - (d.h[0][0] + d.h[1][1] + d.h[2][2])) / (2 * Fmag);
};

/** Gaussian curvature coefficient at a global surface point (the C++ gaussian_curvature_superellipsoid). */
export const gaussianCurvature = (shape: V3, block: [number, number], flag: number, R: M3, surfPoint: V3, xc: V3): number => {
  const xl = transposeMatvec(R, sub(surfPoint, xc));
  const d = derivLocal(xl, shape, block, flag);
  const Fmag = Math.sqrt(norm2(d.g));
  const [fx, fy, fz] = d.g;
  const h = d.h;
  const mat = [
    [h[0][0], h[0][1], h[0][2], fx],
    [h[0][1], h[1][1], h[1][2], fy],
    [h[0][2], h[1][2], h[2][2], fz],
    [fx, fy, fz, 0],
  ];
  const K = -det4M44Zero(mat) / (Fmag * Fmag * Fmag * Fmag);
  return Math.sqrt(Math.abs(K));
};

/**
 * Oriented bounding box test (the C++ check_oriented_bounding_boxes): returns -1 when no separating axis was found (the
 * boxes may overlap), else the separating axis index (0..14), which the caller caches.
 */
export const checkOrientedBoundingBoxes = (xc1: V3, R1: M3, shape1: V3, xc2: V3, R2: M3, shape2: V3, axis: number): number => {
  const C = transposeTimes(R1, R2);
  const A = C.map((r) => r.map((v) => Math.abs(v))) as M3;
  const cd = sub(xc2, xc1);
  const cd1 = transposeMatvec(R1, cd);
  const cd2 = transposeMatvec(R2, cd);
  if (axisSeparated(axis, C, A, cd1, cd2, shape1, shape2)) return axis;
  for (let id = 0; id < 15; id++) {
    if (id === axis) continue;
    if (axisSeparated(id, C, A, cd1, cd2, shape1, shape2)) return id;
  }
  return -1;
};

/** Separation along one of the 15 axes (faces of both boxes, then edge cross products). */
const axisSeparated = (id: number, C: M3, A: M3, c1: V3, c2: V3, a: V3, b: V3): boolean => {
  let R1: number, R2: number, R: number;
  switch (id) {
    case 0: R1 = a[0]; R2 = b[0] * A[0][0] + b[1] * A[0][1] + b[2] * A[0][2]; R = Math.abs(c1[0]); break;
    case 1: R1 = a[1]; R2 = b[0] * A[1][0] + b[1] * A[1][1] + b[2] * A[1][2]; R = Math.abs(c1[1]); break;
    case 2: R1 = a[2]; R2 = b[0] * A[2][0] + b[1] * A[2][1] + b[2] * A[2][2]; R = Math.abs(c1[2]); break;
    case 3: R1 = a[0] * A[0][0] + a[1] * A[1][0] + a[2] * A[2][0]; R2 = b[0]; R = Math.abs(c2[0]); break;
    case 4: R1 = a[0] * A[0][1] + a[1] * A[1][1] + a[2] * A[2][1]; R2 = b[1]; R = Math.abs(c2[1]); break;
    case 5: R1 = a[0] * A[0][2] + a[1] * A[1][2] + a[2] * A[2][2]; R2 = b[2]; R = Math.abs(c2[2]); break;
    case 6: R1 = a[1] * A[2][0] + a[2] * A[1][0]; R2 = b[1] * A[0][2] + b[2] * A[0][1]; R = Math.abs(c1[2] * C[1][0] - c1[1] * C[2][0]); break;
    case 7: R1 = a[1] * A[2][1] + a[2] * A[1][1]; R2 = b[0] * A[0][2] + b[2] * A[0][0]; R = Math.abs(c1[2] * C[1][1] - c1[1] * C[2][1]); break;
    case 8: R1 = a[1] * A[2][2] + a[2] * A[1][2]; R2 = b[0] * A[0][1] + b[1] * A[0][0]; R = Math.abs(c1[2] * C[1][2] - c1[1] * C[2][2]); break;
    case 9: R1 = a[0] * A[2][0] + a[2] * A[0][0]; R2 = b[1] * A[1][2] + b[2] * A[1][1]; R = Math.abs(c1[0] * C[2][0] - c1[2] * C[0][0]); break;
    case 10: R1 = a[0] * A[2][1] + a[2] * A[0][1]; R2 = b[0] * A[1][2] + b[2] * A[1][0]; R = Math.abs(c1[0] * C[2][1] - c1[2] * C[0][1]); break;
    case 11: R1 = a[0] * A[2][2] + a[2] * A[0][2]; R2 = b[0] * A[1][1] + b[1] * A[1][0]; R = Math.abs(c1[0] * C[2][2] - c1[2] * C[0][2]); break;
    case 12: R1 = a[0] * A[1][0] + a[1] * A[0][0]; R2 = b[1] * A[2][2] + b[2] * A[2][1]; R = Math.abs(c1[1] * C[0][0] - c1[0] * C[1][0]); break;
    case 13: R1 = a[0] * A[1][1] + a[1] * A[0][1]; R2 = b[0] * A[2][2] + b[2] * A[2][0]; R = Math.abs(c1[1] * C[0][1] - c1[0] * C[1][1]); break;
    default: R1 = a[0] * A[1][2] + a[1] * A[0][2]; R2 = b[0] * A[2][1] + b[1] * A[2][0]; R = Math.abs(c1[1] * C[0][2] - c1[0] * C[1][2]); break;
  }
  return R > R1 + R2;
};

