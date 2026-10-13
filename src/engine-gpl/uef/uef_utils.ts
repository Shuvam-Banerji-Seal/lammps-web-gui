/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/UEF/uef_utils.cpp and src/UEF/uef_utils.h at commit b332f28 (the UEFBox class, the lattice
 * reduction routines greedy, col_sort, red12, red3, greedy_recurse, make_unique, and rotation_matrix).
 * Copyright (2003) Sandia Corporation (GNU General Public License); contributing author David Nicholson (MIT).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_uef); the core does not import it.
 *
 * Operation order is kept as in the C++ source: the box and the lattice reduction are compared with native
 * LAMMPS to about 1e-8 over a run, so a reordered sum would change the last bits and grow over the run.
 */

/** A 3x3 matrix stored row-major as number[row][column]. Integer matrices hold whole numbers. */
export type M3 = number[][];

const zero3 = (): M3 => [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
/** A new zero 3x3 matrix (rotation buffers are filled by the callers). */
export const zeroRot = (): M3 => zero3();
const copy3 = (m: M3): M3 => m.map((row) => row.slice());
const setFrom = (dst: M3, src: M3): void => { for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) dst[k][j] = src[k][j]; };

/** C round(): half away from zero (JS Math.round rounds halves up). */
export const cRound = (x: number): number => (x < 0 ? -Math.round(-x) : Math.round(x));

const colProd = (x: M3, c1: number, c2: number): number =>
  x[0][c1] * x[0][c2] + x[1][c1] * x[1][c2] + x[2][c1] * x[2][c2];

const colSwap = (x: M3, c1: number, c2: number): void => {
  for (let k = 0; k < 3; k++) {
    const t = x[k][c2];
    x[k][c2] = x[k][c1];
    x[k][c1] = t;
  }
};

const negCol = (x: M3, c1: number): void => {
  x[0][c1] = -x[0][c1];
  x[1][c1] = -x[1][c1];
  x[2][c1] = -x[2][c1];
};

const negRow = (x: M3, c1: number): void => {
  x[c1][0] = -x[c1][0];
  x[c1][1] = -x[c1][1];
  x[c1][2] = -x[c1][2];
};

const det = (x: M3): number => {
  let val = x[0][0] * (x[1][1] * x[2][2] - x[1][2] * x[2][1]);
  val -= x[0][1] * (x[1][0] * x[2][2] - x[1][2] * x[2][0]);
  val += x[0][2] * (x[1][0] * x[2][1] - x[1][1] * x[2][0]);
  return val;
};

const matSame = (x1: M3, x2: M3): boolean => {
  let v = true;
  for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) v = v && x1[k][j] === x2[k][j];
  return v;
};

const transpose = (m: M3): void => {
  for (let k = 0; k < 3; k++) {
    for (let j = k + 1; j < 3; j++) {
      const x = m[k][j];
      m[k][j] = m[j][k];
      m[j][k] = x;
    }
  }
};

/** m1 = m1 * m2 */
const mulM1 = (m1: M3, m2: M3): void => {
  const t = copy3(m1);
  for (let k = 0; k < 3; k++) {
    for (let j = 0; j < 3; j++) m1[k][j] = t[k][0] * m2[0][j] + t[k][1] * m2[1][j] + t[k][2] * m2[2][j];
  }
};

/** m2 = m1 * m2 */
const mulM2 = (m1: M3, m2: M3): void => {
  const t = copy3(m2);
  for (let k = 0; k < 3; k++) {
    for (let j = 0; j < 3; j++) m2[k][j] = m1[k][0] * t[0][j] + m1[k][1] * t[1][j] + m1[k][2] * t[2][j];
  }
};

/**
 * QR reduction by Householder reflections (uef_utils.cpp rotation_matrix): q * m = r, q orthogonal, r upper
 * triangular with positive diagonal. Writes q and r.
 */
export const rotationMatrix = (q: M3, r: M3, m: M3): void => {
  for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) r[k][j] = m[k][j];
  let a = -Math.sqrt(colProd(r, 0, 0)) * r[0][0] / Math.abs(r[0][0]);
  const v = [0, 0, 0];
  v[0] = r[0][0] - a;
  v[1] = r[1][0];
  v[2] = r[2][0];
  a = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  v[0] /= a; v[1] /= a; v[2] /= a;
  const qt = zero3();
  for (let k = 0; k < 3; k++) {
    for (let j = 0; j < 3; j++) {
      qt[k][j] = (k === j ? 1 : 0) - 2 * v[k] * v[j];
      q[k][j] = qt[k][j];
    }
  }
  mulM2(qt, r);
  a = -Math.sqrt(r[1][1] * r[1][1] + r[2][1] * r[2][1]) * r[1][1] / Math.abs(r[1][1]);
  v[0] = 0;
  v[1] = r[1][1] - a;
  v[2] = r[2][1];
  a = Math.sqrt(v[1] * v[1] + v[2] * v[2]);
  v[1] /= a;
  v[2] /= a;
  for (let k = 0; k < 3; k++) {
    for (let j = 0; j < 3; j++) qt[k][j] = (k === j ? 1 : 0) - 2 * v[k] * v[j];
  }
  mulM2(qt, r);
  mulM2(qt, q);
  // this makes r have positive diagonals: q*m = r <==> (-q)*m = (-r) holds row-wise
  if (r[0][0] < 0) { negRow(q, 0); negRow(r, 0); }
  if (r[1][1] < 0) { negRow(q, 1); negRow(r, 1); }
  if (r[2][2] < 0) { negRow(q, 2); negRow(r, 2); }
};

/** Sort the columns of b by increasing length, mimicking the column operations on r and ri. */
const colSort = (b: M3, r: M3, ri: M3): void => {
  if (colProd(b, 0, 0) > colProd(b, 1, 1)) {
    colSwap(b, 0, 1); colSwap(r, 0, 1); colSwap(ri, 0, 1);
  }
  if (colProd(b, 0, 0) > colProd(b, 2, 2)) {
    colSwap(b, 0, 2); colSwap(r, 0, 2); colSwap(ri, 0, 2);
  }
  if (colProd(b, 1, 1) > colProd(b, 2, 2)) {
    colSwap(b, 1, 2); colSwap(r, 1, 2); colSwap(ri, 1, 2);
  }
};

/** 1-2 reduction (Gram-Schmidt on the first two columns). */
const red12 = (b: M3, r: M3, ri: M3): void => {
  const y = cRound(colProd(b, 0, 1) / colProd(b, 0, 0));
  b[0][1] -= y * b[0][0];
  b[1][1] -= y * b[1][0];
  b[2][1] -= y * b[2][0];
  r[0][1] -= y * r[0][0];
  r[1][1] -= y * r[1][0];
  r[2][1] -= y * r[2][0];
  ri[0][0] += y * ri[0][1];
  ri[1][0] += y * ri[1][1];
  ri[2][0] += y * ri[2][1];
  if (colProd(b, 1, 1) < colProd(b, 0, 0)) {
    colSwap(b, 0, 1); colSwap(r, 0, 1); colSwap(ri, 0, 1);
    red12(b, r, ri);
  }
};

/** Semaev's condition for a 3-reduced basis (may recurse through greedyRecurse). */
const red3 = (b: M3, r: M3, ri: M3): void => {
  const b11 = colProd(b, 0, 0);
  const b22 = colProd(b, 1, 1);
  const b12 = colProd(b, 0, 1);
  const b13 = colProd(b, 0, 2);
  const b23 = colProd(b, 1, 2);
  const y2 = -(b23 / b22 - b12 / b22 * b13 / b11) / (1 - b12 / b11 * b12 / b22);
  const y1 = -(b13 / b11 - b12 / b11 * b23 / b22) / (1 - b12 / b11 * b12 / b22);
  let x1 = 0;
  let x2 = 0;
  let min = colProd(b, 2, 2);
  const x1v = [Math.floor(y1), 0];
  x1v[1] = x1v[0] + 1;
  const x2v = [Math.floor(y2), 0];
  x2v[1] = x2v[0] + 1;
  for (let k = 0; k < 2; k++) {
    for (let j = 0; j < 2; j++) {
      const a = [0, 0, 0];
      a[0] = b[0][2] + x1v[k] * b[0][0] + x2v[j] * b[0][1];
      a[1] = b[1][2] + x1v[k] * b[1][0] + x2v[j] * b[1][1];
      a[2] = b[2][2] + x1v[k] * b[2][0] + x2v[j] * b[2][1];
      const val = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
      if (val < min) {
        min = val;
        x1 = x1v[k];
        x2 = x2v[j];
      }
    }
  }
  if (x1 || x2) {
    b[0][2] += x1 * b[0][0] + x2 * b[0][1];
    b[1][2] += x1 * b[1][0] + x2 * b[1][1];
    b[2][2] += x1 * b[2][0] + x2 * b[2][1];
    r[0][2] += x1 * r[0][0] + x2 * r[0][1];
    r[1][2] += x1 * r[1][0] + x2 * r[1][1];
    r[2][2] += x1 * r[2][0] + x2 * r[2][1];
    ri[0][0] += -x1 * ri[0][2];
    ri[1][0] += -x1 * ri[1][2];
    ri[2][0] += -x1 * ri[2][2];
    ri[0][1] += -x2 * ri[0][2];
    ri[1][1] += -x2 * ri[1][2];
    ri[2][1] += -x2 * ri[2][2];
    greedyRecurse(b, r, ri); // the recursion step is here
  }
};

/** The meat of the greedy reduction. */
const greedyRecurse = (b: M3, r: M3, ri: M3): void => {
  colSort(b, r, ri);
  red12(b, r, ri);
  red3(b, r, ri);
};

/** A reduced basis is not unique; make it more so (degenerate cases are unlikely in floating point). */
const makeUnique = (b: M3, r: M3, ri: M3): void => {
  if (Math.abs(b[0][0]) < Math.abs(b[0][1])) {
    colSwap(b, 0, 1); colSwap(r, 0, 1); colSwap(ri, 0, 1);
  }
  if (Math.abs(b[0][0]) < Math.abs(b[0][2])) {
    colSwap(b, 0, 2); colSwap(r, 0, 2); colSwap(ri, 0, 2);
  }
  if (Math.abs(b[1][1]) < Math.abs(b[1][2])) {
    colSwap(b, 1, 2); colSwap(r, 1, 2); colSwap(ri, 1, 2);
  }
  if (b[0][0] < 0) { negCol(b, 0); negCol(r, 0); negCol(ri, 0); }
  if (b[1][1] < 0) { negCol(b, 1); negCol(r, 1); negCol(ri, 1); }
  if (det(b) < 0) { negCol(b, 2); negCol(r, 2); negCol(ri, 2); }
};

/**
 * Reduces the basis b in place; r receives the change of basis and ri its inverse (both integer matrices).
 */
export const greedy = (b: M3, r: M3, ri: M3): void => {
  r[0][1] = r[0][2] = r[1][0] = r[1][2] = r[2][0] = r[2][1] = 0;
  r[0][0] = r[1][1] = r[2][2] = 1;
  ri[0][1] = ri[0][2] = ri[1][0] = ri[1][2] = ri[2][0] = ri[2][1] = 0;
  ri[0][0] = ri[1][1] = ri[2][2] = 1;
  greedyRecurse(b, r, ri);
  makeUnique(b, r, ri);
  transpose(ri);
};

/** The box of a deforming cell, carried as a basis plus the lattice-reduction state (uef_utils.h UEFBox). */
export class UEFBox {
  private readonly l0: M3;
  private readonly w1: number[];
  private readonly w2: number[];
  private readonly winv: M3;
  private readonly theta: number[];
  private readonly l: M3;
  private readonly rot: M3;
  private readonly lrot: M3;
  private readonly r: M3;
  private readonly ri: M3;
  private readonly a1: M3;
  private readonly a2: M3;
  private readonly a1i: M3;
  private readonly a2i: M3;

  constructor() {
    // initial basis (also an inverse eigenvector matrix of automorphisms)
    const x = 0.327985277605681;
    const y = 0.591009048506103;
    const z = 0.736976229099578;
    this.l0 = [[z, y, x], [-x, z, -y], [-y, x, z]];
    // spectra of the two automorphisms (log of eigenvalues)
    this.w1 = [-1.177725211523360, -0.441448620566067, 1.619173832089425];
    this.w2 = [this.w1[1], this.w1[2], this.w1[0]];
    // strain = w1 * theta1 + w2 * theta2
    this.theta = [0, 0];
    this.l = copy3(this.l0);
    this.rot = zero3();
    this.lrot = zero3();
    this.r = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    this.ri = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    rotationMatrix(this.rot, this.lrot, this.l);
    // the automorphisms themselves (only called once)
    const t1 = zero3(), t1i = zero3(), t2 = zero3(), t2i = zero3(), l0t = zero3();
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 3; j++) {
        t1[k][j] = Math.exp(this.w1[k]) * this.l0[k][j];
        t1i[k][j] = Math.exp(-this.w1[k]) * this.l0[k][j];
        t2[k][j] = Math.exp(this.w2[k]) * this.l0[k][j];
        t2i[k][j] = Math.exp(-this.w2[k]) * this.l0[k][j];
        l0t[k][j] = this.l0[j][k];
      }
    }
    mulM2(l0t, t1);
    mulM2(l0t, t1i);
    mulM2(l0t, t2);
    mulM2(l0t, t2i);
    this.a1 = zero3(); this.a1i = zero3(); this.a2 = zero3(); this.a2i = zero3();
    for (let k = 0; k < 3; k++) {
      for (let j = 0; j < 3; j++) {
        this.a1[k][j] = cRound(t1[k][j]);
        this.a1i[k][j] = cRound(t1i[k][j]);
        this.a2[k][j] = cRound(t2[k][j]);
        this.a2i[k][j] = cRound(t2i[k][j]);
      }
    }
    // winv maps strain increments to theta increments
    this.winv = zero3();
    this.winv[0][0] = this.w2[1];
    this.winv[0][1] = -this.w2[0];
    this.winv[1][0] = -this.w1[1];
    this.winv[1][1] = this.w1[0];
    const d = this.w1[0] * this.w2[1] - this.w2[0] * this.w1[1];
    for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) this.winv[k][j] /= d;
  }

  /** The box for volume v: basis * cbrt(v) (get_box). */
  getBox(out: M3, v: number): void {
    const s = Math.cbrt(v);
    for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) out[k][j] = this.lrot[k][j] * s;
  }

  /** The rotation matrix q of the lab frame (get_rot). */
  getRot(out: M3): void {
    setFrom(out, this.rot);
  }

  /** The inverse change of basis (get_inverse_cob). */
  getInverseCob(out: M3): void {
    setFrom(out, this.ri);
  }

  /** Sets the strain to (ex, ey) directly (set_strain). */
  setStrain(ex: number, ey: number): void {
    this.theta[0] = this.winv[0][0] * ex + this.winv[0][1] * ey;
    this.theta[1] = this.winv[1][0] * ex + this.winv[1][1] * ey;
    this.theta[0] -= cRound(this.theta[0]);
    this.theta[1] -= cRound(this.theta[1]);
    for (let k = 0; k < 3; k++) {
      const eps = Math.exp(this.theta[0] * this.w1[k] + this.theta[1] * this.w2[k]);
      this.l[k][0] = eps * this.l0[k][0];
      this.l[k][1] = eps * this.l0[k][1];
      this.l[k][2] = eps * this.l0[k][2];
    }
    greedy(this.l, this.r, this.ri);
    rotationMatrix(this.rot, this.lrot, this.l);
  }

  /** Applies the diagonal, incompressible deformation (ex, ey, -ex-ey) for one step (step_deform). */
  stepDeform(ex: number, ey: number): void {
    this.theta[0] += this.winv[0][0] * ex + this.winv[0][1] * ey;
    this.theta[1] += this.winv[1][0] * ex + this.winv[1][1] * ey;
    const eps = [ex, ey, -ex - ey];
    for (let k = 0; k < 3; k++) {
      eps[k] = Math.exp(eps[k]);
      this.l[k][0] = eps[k] * this.l[k][0];
      this.l[k][1] = eps[k] * this.l[k][1];
      this.l[k][2] = eps[k] * this.l[k][2];
    }
    rotationMatrix(this.rot, this.lrot, this.l);
  }

  /**
   * Reduces the current basis (reduce). Returns true when the change of basis changed, i.e. the atoms and the
   * image flags must be mapped to the new cell.
   */
  reduce(): boolean {
    // how many times the automorphisms apply, and the remaining theta values
    const f1 = cRound(this.theta[0]);
    const f2 = cRound(this.theta[1]);
    this.theta[0] -= f1;
    this.theta[1] -= f2;
    // the old change of basis, adjusted for the automorphisms applied (r0 equals r when the reduced basis is unchanged)
    const r0 = copy3(this.r);
    if (f1 > 0) for (let k = 0; k < f1; k++) mulM2(this.a1, r0);
    if (f1 < 0) for (let k = 0; k < -f1; k++) mulM2(this.a1i, r0);
    if (f2 > 0) for (let k = 0; k < f2; k++) mulM2(this.a2, r0);
    if (f2 < 0) for (let k = 0; k < -f2; k++) mulM2(this.a2i, r0);
    // robust reduction to the box defined by Dobson
    for (let k = 0; k < 3; k++) {
      const eps = Math.exp(this.theta[0] * this.w1[k] + this.theta[1] * this.w2[k]);
      this.l[k][0] = eps * this.l0[k][0];
      this.l[k][1] = eps * this.l0[k][1];
      this.l[k][2] = eps * this.l0[k][2];
    }
    // further greedy reduction; ri times the old r gives the transformation itself (image flags)
    greedy(this.l, this.r, this.ri);
    mulM1(this.ri, r0);
    rotationMatrix(this.rot, this.lrot, this.l);
    return !matSame(this.r, r0);
  }
}

/**
 * Rotates a symmetric tensor from the upper-triangular frame into the lab frame, [P]rot = Q [P] Q^t, in place.
 * x is in Voigt order xx yy zz xy xz yz (compute_temp_uef.cpp / compute_pressure_uef.cpp virial_rot).
 */
export const virialRot = (x: Float64Array | number[], r: M3): void => {
  const t = zero3();
  for (let k = 0; k < 3; k++) {
    t[0][k] = x[0] * r[0][k] + x[3] * r[1][k] + x[4] * r[2][k];
    t[1][k] = x[3] * r[0][k] + x[1] * r[1][k] + x[5] * r[2][k];
    t[2][k] = x[4] * r[0][k] + x[5] * r[1][k] + x[2] * r[2][k];
  }
  x[0] = r[0][0] * t[0][0] + r[1][0] * t[1][0] + r[2][0] * t[2][0];
  x[3] = r[0][0] * t[0][1] + r[1][0] * t[1][1] + r[2][0] * t[2][1];
  x[4] = r[0][0] * t[0][2] + r[1][0] * t[1][2] + r[2][0] * t[2][2];
  x[1] = r[0][1] * t[0][1] + r[1][1] * t[1][1] + r[2][1] * t[2][1];
  x[5] = r[0][1] * t[0][2] + r[1][1] * t[1][2] + r[2][1] * t[2][2];
  x[2] = r[0][2] * t[0][2] + r[1][2] * t[1][2] + r[2][2] * t[2][2];
};
