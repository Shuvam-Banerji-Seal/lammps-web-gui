/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_gayberne.cpp at commit b332f28 (compute, settings, coeff, init_style, init_one, single,
 * gayberne_analytic, gayberne_lj, compute_eta_torque), with the MathExtra helpers in ./math_extra.ts (src/math_extra.h
 * and src/math_extra.cpp at b332f28). Copyright (2003) Sandia Corporation, distributed under the GNU General Public
 * License (the LAMMPS file headers). Contributing author: Mike Brown (SNL). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The operation order follows the C++ line by line, so the energies and forces match native LAMMPS bit for bit
 * (the port exists because a re-derived form differs in the last bits and chaotic dynamics amplify that).
 *
 * pair_style gayberne gamma upsilon mu cutoff; pair_coeff I J epsilon sigma epsilon_ia epsilon_ib epsilon_ic
 * epsilon_ja epsilon_jb epsilon_jc [cutoff] (docs.lammps.org/pair_gayberne.html). As native: upsilon is the second
 * setting divided by 2, the per-type well is epsilon a,b,c ^ (-1/mu), and a point particle (shape x = 0) uses
 * radius 1 in each direction.
 *
 * Engine conventions kept from the previous implementation: torques are accumulated in an owned array and folded
 * from ghosts by nb.reverseSum; per-atom energy and virial are split half to each atom.
 *
 * Refusals (StyleError, kept from the previous implementation, not native errors): an atom style without
 * ellipsoids is refused by the engine's atom check; a point particle with anisotropic epsilon; an LJ sphere paired
 * with an ellipsoid when mu is not 1; a special neighbour with a factor other than 1 (the page describes no
 * special-bond factor for this style). Native errors kept: the shape rule per type, the missing epsilon a,b,c
 * coeffs, and "Bad matrix inversion in mldivide3".
 */
import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, typeBounds, type PairCompute, type PairSep, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { NativePairOrder, NativeReverseComm } from '../neighbor/nativeOrder';
import { parseNum } from '../../engine/force/util';
import { type Mat3, newMat3, quatToMatTrans, diagTimes3, transposeTimes3, plus3, det3, normalize3, dot3, cross3, vecmat, mldivide3, computeEtaTorque } from './math_extra';
import { pow as libmPow } from '../libm/pow';

/** Form per type pair (native enum): the sphere and ellipsoid forms of pair_gayberne.cpp init_one. */
const SPHERE_SPHERE = 0;
const SPHERE_ELLIPSE = 1;
const ELLIPSE_SPHERE = 2;
const ELLIPSE_ELLIPSE = 3;

const BAD_INVERSION = 'Bad matrix inversion in mldivide3';

const v3 = (): Float64Array => new Float64Array(3);

/** The tables the Gay-Berne functions read (the class fields, flat with index ti * nt + tj for pairs). */
export interface GbTables {
  gamma: number;
  upsilon: number;
  mu: number;
  nt: number;
  eps: Float64Array;
  sig: Float64Array;
  lshape: Float64Array;
  /** Per type: shape2 (3 entries per type). */
  shape2: Float64Array;
  /** Per type: well = epsilon a,b,c ^ (-1/mu) (3 entries per type). */
  well: Float64Array;
  /** Per type: point particle flag (shape x = 0 before the replacement). */
  ptFlag: Uint8Array;
}

/**
 * PairGayBerne::gayberne_analytic: energy, force on i (fforce), torque on i (ttor) and torque on j (rtor) of an
 * ellipsoid pair; a1..g2 are the matrices of i and j, r12 = x_j - x_i. Returns the energy (before factor_lj).
 */
function gayberneAnalytic(g: GbTables, ti: number, tj: number, a1: Mat3, a2: Mat3, b1: Mat3, b2: Mat3, g1: Mat3, g2: Mat3,
  r12: Float64Array, rsq: number, fforce: Float64Array, ttor: Float64Array, rtor: Float64Array): number {
  const tempv = v3(), tempv2 = v3();
  const temp = newMat3();
  const r12hat = v3();
  normalize3(r12, r12hat);
  const r = Math.sqrt(rsq);

  // compute distance of closest approach
  const g12 = newMat3();
  plus3(g1, g2, g12);
  const kappa = v3();
  if (mldivide3(g12, r12, kappa)) throw new StyleError(BAD_INVERSION);
  // tempv = G12^-1*r12hat
  tempv[0] = kappa[0] / r;
  tempv[1] = kappa[1] / r;
  tempv[2] = kappa[2] / r;
  let sigma12 = dot3(r12hat, tempv);
  sigma12 = libmPow(0.5 * sigma12, -0.5);
  const h12 = r - sigma12;

  // energy
  // compute u_r
  const k = ti * g.nt + tj;
  const sg = g.sig[k];
  const varrho = sg / (h12 + g.gamma * sg);
  const varrho6 = libmPow(varrho, 6.0);
  const varrho12 = varrho6 * varrho6;
  const u_r = 4.0 * g.eps[k] * (varrho12 - varrho6);

  // compute eta_12
  let eta = 2.0 * g.lshape[ti] * g.lshape[tj];
  const det_g12 = det3(g12);
  eta = libmPow(eta / det_g12, g.upsilon);

  // compute chi_12
  const b12 = newMat3();
  plus3(b1, b2, b12);
  const iota = v3();
  if (mldivide3(b12, r12, iota)) throw new StyleError(BAD_INVERSION);
  tempv[0] = iota[0] / r;
  tempv[1] = iota[1] / r;
  tempv[2] = iota[2] / r;
  let chi = dot3(r12hat, tempv);
  chi = libmPow(chi * 2.0, g.mu);

  // force
  // compute dUr/dr
  let temp1 = (2.0 * varrho12 * varrho - varrho6 * varrho) / sg;
  temp1 = temp1 * 24.0 * g.eps[k];
  const u_slj = temp1 * libmPow(sigma12, 3.0) / 2.0;
  const dUr = v3();
  let temp2 = dot3(kappa, r12hat);
  const uslj_rsq = u_slj / rsq;
  dUr[0] = temp1 * r12hat[0] + uslj_rsq * (kappa[0] - temp2 * r12hat[0]);
  dUr[1] = temp1 * r12hat[1] + uslj_rsq * (kappa[1] - temp2 * r12hat[1]);
  dUr[2] = temp1 * r12hat[2] + uslj_rsq * (kappa[2] - temp2 * r12hat[2]);

  // compute dChi_12/dr
  const dchi = v3();
  temp1 = dot3(iota, r12hat);
  temp2 = -4.0 / rsq * g.mu * libmPow(chi, (g.mu - 1.0) / g.mu);
  dchi[0] = temp2 * (iota[0] - temp1 * r12hat[0]);
  dchi[1] = temp2 * (iota[1] - temp1 * r12hat[1]);
  dchi[2] = temp2 * (iota[2] - temp1 * r12hat[2]);

  temp1 = -eta * u_r;
  let temp3 = eta * chi;
  fforce[0] = temp1 * dchi[0] - temp3 * dUr[0];
  fforce[1] = temp1 * dchi[1] - temp3 * dUr[1];
  fforce[2] = temp1 * dchi[2] - temp3 * dUr[2];

  // torque for particle 1 and 2
  // compute dUr
  tempv[0] = -uslj_rsq * kappa[0];
  tempv[1] = -uslj_rsq * kappa[1];
  tempv[2] = -uslj_rsq * kappa[2];
  vecmat(kappa, g1, tempv2);
  cross3(tempv, tempv2, dUr);
  const dUr2 = v3();
  vecmat(kappa, g2, tempv2);
  cross3(tempv, tempv2, dUr2);

  // compute d_chi
  vecmat(iota, b1, tempv);
  cross3(tempv, iota, dchi);
  dchi[0] *= temp2;
  dchi[1] *= temp2;
  dchi[2] *= temp2;
  const dchi2 = v3();
  vecmat(iota, b2, tempv);
  cross3(tempv, iota, dchi2);
  dchi2[0] *= temp2;
  dchi2[1] *= temp2;
  dchi2[2] *= temp2;

  // compute d_eta
  const deta = v3();
  computeEtaTorque(g12, a1, g.shape2.subarray(3 * ti, 3 * ti + 3), temp);
  temp1 = -eta * g.upsilon;
  for (let m = 0; m < 3; m++) {
    for (let y = 0; y < 3; y++) tempv[y] = temp1 * temp[m][y];
    cross3(a1[m], tempv, tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // compute d_eta for particle 2
  const deta2 = v3();
  computeEtaTorque(g12, a2, g.shape2.subarray(3 * tj, 3 * tj + 3), temp);
  for (let m = 0; m < 3; m++) {
    for (let y = 0; y < 3; y++) tempv[y] = temp1 * temp[m][y];
    cross3(a2[m], tempv, tempv2);
    deta2[0] += tempv2[0];
    deta2[1] += tempv2[1];
    deta2[2] += tempv2[2];
  }

  // torque
  temp1 = u_r * eta;
  temp2 = u_r * chi;
  temp3 = chi * eta;
  ttor[0] = (temp1 * dchi[0] + temp2 * deta[0] + temp3 * dUr[0]) * -1.0;
  ttor[1] = (temp1 * dchi[1] + temp2 * deta[1] + temp3 * dUr[1]) * -1.0;
  ttor[2] = (temp1 * dchi[2] + temp2 * deta[2] + temp3 * dUr[2]) * -1.0;
  rtor[0] = (temp1 * dchi2[0] + temp2 * deta2[0] + temp3 * dUr2[0]) * -1.0;
  rtor[1] = (temp1 * dchi2[1] + temp2 * deta2[1] + temp3 * dUr2[1]) * -1.0;
  rtor[2] = (temp1 * dchi2[2] + temp2 * deta2[2] + temp3 * dUr2[2]) * -1.0;
  return temp1 * chi;
}

/**
 * PairGayBerne::gayberne_lj: an ellipsoid (type ti, matrices a1, b1, g1) with a spherical partner (type tj, shape2
 * of its x entry). Writes the force on the ellipsoid (fforce) and its torque (ttor). Returns the energy.
 */
function gayberneLj(g: GbTables, ti: number, tj: number, a1: Mat3, b1: Mat3, g1: Mat3, r12: Float64Array, rsq: number,
  fforce: Float64Array, ttor: Float64Array): number {
  const tempv = v3(), tempv2 = v3();
  const temp = newMat3();
  const r12hat = v3();
  normalize3(r12, r12hat);
  const r = Math.sqrt(rsq);

  // compute distance of closest approach
  const g12 = newMat3();
  const s2 = g.shape2[3 * tj];
  g12[0][0] = g1[0][0] + s2;
  g12[1][1] = g1[1][1] + s2;
  g12[2][2] = g1[2][2] + s2;
  g12[0][1] = g1[0][1]; g12[1][0] = g1[1][0];
  g12[0][2] = g1[0][2]; g12[2][0] = g1[2][0];
  g12[1][2] = g1[1][2]; g12[2][1] = g1[2][1];
  const kappa = v3();
  if (mldivide3(g12, r12, kappa)) throw new StyleError(BAD_INVERSION);
  tempv[0] = kappa[0] / r;
  tempv[1] = kappa[1] / r;
  tempv[2] = kappa[2] / r;
  let sigma12 = dot3(r12hat, tempv);
  sigma12 = libmPow(0.5 * sigma12, -0.5);
  const h12 = r - sigma12;

  // energy
  // compute u_r
  const k = ti * g.nt + tj;
  const sg = g.sig[k];
  const varrho = sg / (h12 + g.gamma * sg);
  const varrho6 = libmPow(varrho, 6.0);
  const varrho12 = varrho6 * varrho6;
  const u_r = 4.0 * g.eps[k] * (varrho12 - varrho6);

  // compute eta_12
  let eta = 2.0 * g.lshape[ti] * g.lshape[tj];
  const det_g12 = det3(g12);
  eta = libmPow(eta / det_g12, g.upsilon);

  // compute chi_12
  const b12 = newMat3();
  const wl = g.well[3 * tj];
  b12[0][0] = b1[0][0] + wl;
  b12[1][1] = b1[1][1] + wl;
  b12[2][2] = b1[2][2] + wl;
  b12[0][1] = b1[0][1]; b12[1][0] = b1[1][0];
  b12[0][2] = b1[0][2]; b12[2][0] = b1[2][0];
  b12[1][2] = b1[1][2]; b12[2][1] = b1[2][1];
  const iota = v3();
  if (mldivide3(b12, r12, iota)) throw new StyleError(BAD_INVERSION);
  tempv[0] = iota[0] / r;
  tempv[1] = iota[1] / r;
  tempv[2] = iota[2] / r;
  let chi = dot3(r12hat, tempv);
  chi = libmPow(chi * 2.0, g.mu);

  // force
  // compute dUr/dr
  let temp1 = (2.0 * varrho12 * varrho - varrho6 * varrho) / sg;
  temp1 = temp1 * 24.0 * g.eps[k];
  const u_slj = temp1 * libmPow(sigma12, 3.0) / 2.0;
  const dUr = v3();
  const temp2a = dot3(kappa, r12hat);
  const uslj_rsq = u_slj / rsq;
  dUr[0] = temp1 * r12hat[0] + uslj_rsq * (kappa[0] - temp2a * r12hat[0]);
  dUr[1] = temp1 * r12hat[1] + uslj_rsq * (kappa[1] - temp2a * r12hat[1]);
  dUr[2] = temp1 * r12hat[2] + uslj_rsq * (kappa[2] - temp2a * r12hat[2]);

  // compute dChi_12/dr
  const dchi = v3();
  temp1 = dot3(iota, r12hat);
  const temp2 = -4.0 / rsq * g.mu * libmPow(chi, (g.mu - 1.0) / g.mu);
  dchi[0] = temp2 * (iota[0] - temp1 * r12hat[0]);
  dchi[1] = temp2 * (iota[1] - temp1 * r12hat[1]);
  dchi[2] = temp2 * (iota[2] - temp1 * r12hat[2]);

  temp1 = -eta * u_r;
  const temp2b = eta * chi;
  fforce[0] = temp1 * dchi[0] - temp2b * dUr[0];
  fforce[1] = temp1 * dchi[1] - temp2b * dUr[1];
  fforce[2] = temp1 * dchi[2] - temp2b * dUr[2];

  // torque for particle 1
  // compute dUr
  tempv[0] = -uslj_rsq * kappa[0];
  tempv[1] = -uslj_rsq * kappa[1];
  tempv[2] = -uslj_rsq * kappa[2];
  vecmat(kappa, g1, tempv2);
  cross3(tempv, tempv2, dUr);

  // compute d_chi
  vecmat(iota, b1, tempv);
  cross3(tempv, iota, dchi);
  const temp1b = -4.0 / rsq;
  dchi[0] *= temp1b;
  dchi[1] *= temp1b;
  dchi[2] *= temp1b;

  // compute d_eta
  const deta = v3();
  computeEtaTorque(g12, a1, g.shape2.subarray(3 * ti, 3 * ti + 3), temp);
  temp1 = -eta * g.upsilon;
  for (let m = 0; m < 3; m++) {
    for (let y = 0; y < 3; y++) tempv[y] = temp1 * temp[m][y];
    cross3(a1[m], tempv, tempv2);
    deta[0] += tempv2[0];
    deta[1] += tempv2[1];
    deta[2] += tempv2[2];
  }

  // torque
  temp1 = u_r * eta;
  const temp2c = u_r * chi;
  const temp3 = chi * eta;
  ttor[0] = (temp1 * dchi[0] + temp2c * deta[0] + temp3 * dUr[0]) * -1.0;
  ttor[1] = (temp1 * dchi[1] + temp2c * deta[1] + temp3 * dUr[1]) * -1.0;
  ttor[2] = (temp1 * dchi[2] + temp2c * deta[2] + temp3 * dUr[2]) * -1.0;
  return temp1 * chi;
}

/** Matrices of one ellipsoid atom: a = quat_to_mat_trans, b = a^T (well a), g = a^T (shape2 a). */
interface AtomMats { a: Mat3; b: Mat3; g: Mat3 }

const atomMats = (g: GbTables, quat: Float64Array, o: number, t: number): AtomMats => {
  const a = newMat3();
  quatToMatTrans(quat.subarray(4 * o, 4 * o + 4), a);
  const tmp = newMat3();
  const b = newMat3();
  const gm = newMat3();
  const well = g.well.subarray(3 * t, 3 * t + 3);
  const s2 = g.shape2.subarray(3 * t, 3 * t + 3);
  diagTimes3(well, a, tmp);
  transposeTimes3(a, tmp, b);
  diagTimes3(s2, a, tmp);
  transposeTimes3(a, tmp, gm);
  return { a, b, g: gm };
};

/** Result of one pair evaluation (unscaled by factor_lj). */
interface PairResult { one: number; radial: number }

/**
 * The form dispatch of PairGayBerne::compute / single for one pair: fi = force on i, ti and tj = torques on i and j.
 * r12 = x_j - x_i. radial is the native single() fforce before factor_lj (sphere pairs: forcelj * r2inv; ellipsoid
 * pairs: -(fvec . r12) / rsq).
 */
function evalPair(g: GbTables, form: Uint8Array, lj: { lj1: Float64Array; lj2: Float64Array; lj3: Float64Array; lj4: Float64Array; offset: Float64Array },
  ti: number, tj: number, mi: AtomMats | null, mj: AtomMats | null, r12: Float64Array, rsq: number,
  fi: Float64Array, tqi: Float64Array, tqj: Float64Array, out: PairResult): number {
  const nt = g.nt;
  const f = form[ti * nt + tj];
  const k = ti * nt + tj;
  if (f === SPHERE_SPHERE) {
    const r2inv = 1.0 / rsq;
    const r6inv = r2inv * r2inv * r2inv;
    let forcelj = r6inv * (lj.lj1[k] * r6inv - lj.lj2[k]);
    out.radial = forcelj * r2inv;
    forcelj *= -r2inv;
    const one = r6inv * (r6inv * lj.lj3[k] - lj.lj4[k]) - lj.offset[k];
    fi[0] = r12[0] * forcelj;
    fi[1] = r12[1] * forcelj;
    fi[2] = r12[2] * forcelj;
    tqi.fill(0);
    tqj.fill(0);
    out.one = one;
    return one;
  }
  // Measured with native LAMMPS (black box): a point particle (shape 0 0 0) whose epsilon a,b,c are not equal gives
  // an energy that differs from the Gay-Berne form with radius 1.0. Refused rather than reproduced.
  const sI = form[ti * nt + ti] === SPHERE_SPHERE, sJ = form[tj * nt + tj] === SPHERE_SPHERE;
  if ((g.ptFlag[ti] && !sI) || (g.ptFlag[tj] && !sJ)) {
    throw new StyleError('pair_style gayberne: point particles need equal epsilon a,b,c (anisotropic point particles are not supported)');
  }
  // Refused: an LJ sphere paired with an ellipsoid when mu is not 1 (kept from the previous implementation).
  if (sI !== sJ && g.mu !== 1) {
    throw new StyleError('pair_style gayberne: an LJ sphere paired with an ellipsoid needs mu = 1 (the native torques differ otherwise)');
  }
  let one: number;
  if (f === SPHERE_ELLIPSE) {
    one = gayberneLj(g, tj, ti, mj!.a, mj!.b, mj!.g, r12, rsq, fi, tqj);
    tqi.fill(0);
  } else if (f === ELLIPSE_SPHERE) {
    one = gayberneLj(g, ti, tj, mi!.a, mi!.b, mi!.g, r12, rsq, fi, tqi);
    tqj.fill(0);
  } else {
    one = gayberneAnalytic(g, ti, tj, mi!.a, mj!.a, mi!.b, mj!.b, mi!.g, mj!.g, r12, rsq, fi, tqi, tqj);
  }
  // single(): fforce = -dot3(fvec, r12) / rsq
  out.radial = -dot3(fi, r12) / rsq;
  out.one = one;
  return one;
}

export class PairGayBerne extends Pair {
  readonly name = 'gayberne';
  virialFdotr = true;
  private readonly order = new NativePairOrder();
  private readonly comm = new NativeReverseComm();
  private newtonPair = true;
  /** pair_style gamma (native gamma). */
  gamma = 1;
  /** Native upsilon: the second setting divided by 2 (pair_gayberne.cpp settings). */
  upsilon = 0.5;
  mu = 1;
  /** ntypes + 1 (set in initStyle). */
  nt = 1;
  cutGlobal = 0;
  p!: PairParams;
  /** Per type: epsilon a,b,c (index 3 t + d), and whether any were set. */
  eT = new Float64Array(0);
  eSet = new Uint8Array(0);
  /** Per type: 0 unset, 1 anisotropic well, 2 isotropic well (native setwell). */
  setwell = new Uint8Array(0);
  /** Per type after init_style: shape (zero x shape -> 1), shape2 and lshape (native shape1, shape2, lshape). */
  shape1 = new Float64Array(0);
  shape2 = new Float64Array(0);
  lshape = new Float64Array(0);
  /** Per type: well = epsilon a,b,c ^ (-1/mu) (native well). */
  well = new Float64Array(0);
  /** Per type: point particle (shape x = 0), native shape1[t][0] == 0 before the replacement. */
  ptFlag = new Uint8Array(0);
  /** Per type pair (flat ti * nt + tj): epsilon, sigma, form, lj1..lj4 and offset (native arrays, both orders set). */
  eps = new Float64Array(0);
  sig = new Float64Array(0);
  form = new Uint8Array(0);
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  offset = new Float64Array(0);
  /** The system of the last init (single() reads the shapes and quaternions of its two atoms). */
  private state: SimState | null = null;
  /** Native svector of the last single() call: force and torque on i (pN 1..6). */
  private readonly svec = new Float64Array(6);

  settings(args: string[]): void {
    if (args.length !== 4) throw new StyleError('Illegal pair_style command');
    this.gamma = parseNum(args[0], 'gamma');
    this.upsilon = parseNum(args[1], 'upsilon') / 2.0;
    this.mu = parseNum(args[2], 'mu');
    this.cutGlobal = parseNum(args[3], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style gayberne cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
    this.eT = new Float64Array(3 * (ntypes + 1));
    this.eSet = new Uint8Array(ntypes + 1);
  }

  coeff(args: string[]): void {
    if (args.length < 10 || args.length > 11) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const ei = [4, 5, 6].map((k) => parseNum(args[k], 'epsilon_i'));
    const ej = [7, 8, 9].map((k) => parseNum(args[k], 'epsilon_j'));
    const cut = args.length === 11 ? parseNum(args[10], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
    // Per-type epsilon a,b,c: non-zero triples are assigned to the type (the three values are assigned to atom type I).
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const assign = (lo: number, hi: number, e: number[]) => {
      if (!e.some((v) => v !== 0)) return;
      for (let t = lo; t <= hi; t++) {
        for (let d = 0; d < 3; d++) this.eT[3 * t + d] = e[d];
        this.eSet[t] = 1;
      }
    };
    for (let i = ilo; i <= ihi; i++) for (let j = jlo; j <= jhi; j++) {
      assign(i, i, ei);
      assign(j, j, ej);
    }
  }

  /** Native init_style: per-type shape (identical within a type), shape2, lshape, well and setwell. */
  initStyle(ctx: StyleContext): void {
    const s = ctx.s;
    if (!s || !s.shape || !s.quat || !s.torque) throw new StyleError('Pair gayberne requires atom style ellipsoid');
    const nt = this.ntypes + 1;
    this.nt = nt;
    this.state = s;
    this.eps = new Float64Array(nt * nt);
    this.sig = new Float64Array(nt * nt);
    this.form = new Uint8Array(nt * nt);
    this.lj1 = new Float64Array(nt * nt);
    this.lj2 = new Float64Array(nt * nt);
    this.lj3 = new Float64Array(nt * nt);
    this.lj4 = new Float64Array(nt * nt);
    this.offset = new Float64Array(nt * nt);
    this.shape1 = new Float64Array(3 * nt);
    this.shape2 = new Float64Array(3 * nt);
    this.lshape = new Float64Array(nt);
    this.well = new Float64Array(3 * nt);
    this.ptFlag = new Uint8Array(nt);
    this.setwell = new Uint8Array(nt);
    // require that atom shapes are identical within each type (atom.shape_consistency)
    const seen = new Uint8Array(nt);
    const shape = s.shape;
    for (let o = 0; o < s.n; o++) {
      const t = s.type[o];
      if (!seen[t]) {
        seen[t] = 1;
        for (let d = 0; d < 3; d++) this.shape1[3 * t + d] = shape[3 * o + d];
      } else if (shape[3 * o] !== this.shape1[3 * t] || shape[3 * o + 1] !== this.shape1[3 * t + 1] || shape[3 * o + 2] !== this.shape1[3 * t + 2]) {
        throw new StyleError('Pair gayberne requires atoms with same type have same shape');
      }
    }
    for (let t = 1; t < nt; t++) {
      // if shape = 0 for point particle, set shape = 1 as required by Gay-Berne
      if (this.shape1[3 * t] === 0.0) {
        this.ptFlag[t] = 1;
        this.shape1[3 * t] = this.shape1[3 * t + 1] = this.shape1[3 * t + 2] = 1.0;
      }
      for (let d = 0; d < 3; d++) this.shape2[3 * t + d] = this.shape1[3 * t + d] * this.shape1[3 * t + d];
      const s0 = this.shape1[3 * t], s1 = this.shape1[3 * t + 1], s2 = this.shape1[3 * t + 2];
      this.lshape[t] = (s0 * s1 + s2 * s2) * Math.sqrt(s0 * s1);
      if (this.eSet[t]) {
        for (let d = 0; d < 3; d++) this.well[3 * t + d] = libmPow(this.eT[3 * t + d], -1.0 / this.mu);
        const e = [this.eT[3 * t], this.eT[3 * t + 1], this.eT[3 * t + 2]];
        this.setwell[t] = e[0] === e[1] && e[1] === e[2] ? 2 : 1;
      }
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    for (const t of [i, j]) {
      // Measured with native LAMMPS (black box): init fails when a type has no epsilon a,b,c.
      if (this.setwell[t] === 0) throw new StyleError('Pair gayberne epsilon a,b,c coeffs are not all set');
    }
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    const set = (arr: Float64Array, v: number) => { arr[k1] = v; arr[k2] = v; };
    set(this.eps, eps);
    set(this.sig, sig);
    set(this.lj1, 48.0 * eps * libmPow(sig, 12.0));
    set(this.lj2, 24.0 * eps * libmPow(sig, 6.0));
    set(this.lj3, 4.0 * eps * libmPow(sig, 12.0));
    set(this.lj4, 4.0 * eps * libmPow(sig, 6.0));
    // pair_modify shift: the LJ offset is used by sphere-sphere (LJ sphere) pairs only.
    if (this.shift && cut > 0.0) {
      const ratio = sig / cut;
      set(this.offset, 4.0 * eps * (libmPow(ratio, 12.0) - libmPow(ratio, 6.0)));
    } else set(this.offset, 0);

    // ellipsoidal shape: non-equal radii or an anisotropic well
    const ishape = this.isEllipsoidType(i);
    const jshape = this.isEllipsoidType(j);
    const F = this.form;
    if (!ishape && !jshape) {
      F[i * nt + i] = F[j * nt + j] = F[i * nt + j] = F[j * nt + i] = SPHERE_SPHERE;
    } else if (!ishape) {
      F[i * nt + i] = SPHERE_SPHERE; F[j * nt + j] = ELLIPSE_ELLIPSE;
      F[i * nt + j] = SPHERE_ELLIPSE; F[j * nt + i] = ELLIPSE_SPHERE;
    } else if (!jshape) {
      F[j * nt + j] = SPHERE_SPHERE; F[i * nt + i] = ELLIPSE_ELLIPSE;
      F[j * nt + i] = SPHERE_ELLIPSE; F[i * nt + j] = ELLIPSE_SPHERE;
    } else {
      F[i * nt + i] = F[j * nt + j] = F[i * nt + j] = F[j * nt + i] = ELLIPSE_ELLIPSE;
    }
    return cut;
  }

  /** Native ishape of init_one: an ellipsoid when the radii differ or the well is anisotropic. */
  private isEllipsoidType(t: number): boolean {
    const s = this.shape1;
    const unequal = s[3 * t] !== s[3 * t + 1] || s[3 * t] !== s[3 * t + 2] || s[3 * t + 1] !== s[3 * t + 2];
    return unequal || this.setwell[t] === 1;
  }

  init(ctx: StyleContext): void {
    // The page says that this style does not support the pair_modify tail option. Native LAMMPS ignores tail yes for
    // this style (measured, black box: the pair energy is unchanged); the engine refuses it instead.
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair_style gayberne');
    this.newtonPair = ctx.newtonPair !== false;
    super.init(ctx);
  }

  /** Native single_extra (pair_gayberne.cpp: single_extra = 6): the force and torque on i (pN 1..6). */
  singleExtra(): number { return 6; }

  singleNeedsSeparation(): boolean { return true; }

  /** Matrices of owned atom o of type t, or null for a sphere type. */
  private matsOf(o: number, t: number): AtomMats | null {
    if (this.form[t * (this.ntypes + 1) + t] !== ELLIPSE_ELLIPSE) return null;
    return atomMats(this, this.state!.quat!, o, t);
  }

  /**
   * Native single() (pair_gayberne.cpp): fforce is the force on i projected on the centre axis, the svector holds the
   * full force on i (1-3) and the torque on i (4-6), both times factor_lj, and the energy is factor_lj times the pair
   * energy. i and j are owned atoms; del = x_i - x_j.
   */
  single(i: number, j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, del?: PairSep): { eng: number; fforce: number; svector: Float64Array } {
    const s = this.state;
    if (!s || !s.shape || !s.quat || !s.torque) throw new StyleError('Pair gayberne requires atom style ellipsoid');
    if (!del) throw new StyleError('pair gayberne single() needs the separation vector x_i - x_j (compute pair/local passes it)');
    // native r12 = x_j - x_i (exact negation of del)
    const r12 = Float64Array.of(-del.dx, -del.dy, -del.dz);
    const fi = v3(), tqi = v3(), tqj = v3();
    const out: PairResult = { one: 0, radial: 0 };
    const mi = this.matsOf(i, itype), mj = this.matsOf(j, jtype);
    const one = evalPair(this, this.form, this, itype, jtype, mi, mj, r12, rsq, fi, tqi, tqj, out);
    const svector = this.svec;
    svector[0] = factorLJ * fi[0]; svector[1] = factorLJ * fi[1]; svector[2] = factorLJ * fi[2];
    svector[3] = factorLJ * tqi[0]; svector[4] = factorLJ * tqi[1]; svector[5] = factorLJ * tqi[2];
    return { eng: factorLJ * one, fforce: factorLJ * out.radial, svector };
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.shape || !s.quat || !s.torque) throw new StyleError('Pair gayberne requires atom style ellipsoid');
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const nlocal = pc.nlocal;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);
    const sC = pc.specialCoul, sL = pc.specialLJ;
    const cutsq = this.cutsq;

    // Per owned ellipsoid atom: a1, b1, g1 (native, computed when form[itype][itype] == ELLIPSE_ELLIPSE).
    const mats: (AtomMats | null)[] = new Array(nlocal);
    for (let o = 0; o < nlocal; o++) mats[o] = this.matsOf(o, type[o]);

    let evdwl = 0;
    const fi = v3(), tqi = v3(), tqj = v3();
    const out: PairResult = { one: 0, radial: 0 };
    const r12 = v3();
    // the pairs in native order (ilist, then the half/bin/newton enumeration), so the energy sum and the force and
    // torque accumulation run in native's order (../neighbor/nativeOrder.ts; triclinic boxes keep the engine order)
    const pairs = this.order.pairs(s, nb, pc.geom, list);
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const oi = owner[i];
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      {
        const jj = pairs[p + 1];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        // r12 = center to center vector (native: x[j] - x[i])
        r12[0] = x[3 * j] - xi;
        r12[1] = x[3 * j + 1] - yi;
        r12[2] = x[3 * j + 2] - zi;
        const rsq = dot3(r12, r12);
        const tj = type[j];
        const t = ti * nt + tj;
        // compute if less than cutoff
        if (!(rsq < cutsq[t])) continue;
        const factorLJ = sL[jj >>> SBBITS];
        const factorC = sC[jj >>> SBBITS];
        if (jj >>> SBBITS !== 0 && (factorLJ !== 1 || factorC !== 1)) {
          throw new StyleError('pair_style gayberne does not support special_bonds factors other than 1');
        }
        const one = evalPair(this, this.form, this, ti, tj, mats[oi], mats[oj], r12, rsq, fi, tqi, tqj, out);
        // scale by factor_lj (native fforce, ttor, rtor, evdwl)
        for (let c = 0; c < 3; c++) {
          fi[c] *= factorLJ;
          tqi[c] *= factorLJ;
          tqj[c] *= factorLJ;
        }
        f[3 * i] += fi[0]; f[3 * i + 1] += fi[1]; f[3 * i + 2] += fi[2];
        f[3 * j] -= fi[0]; f[3 * j + 1] -= fi[1]; f[3 * j + 2] -= fi[2];
        for (let c = 0; c < 3; c++) { tq[3 * i + c] += tqi[c]; tq[3 * j + c] += tqj[c]; }
        const e = factorLJ * one;
        evdwl += e;

        if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
        if (pc.vatom) {
          const va = pc.vatom;
          // ev_tally_xyz: del = x_i - x_j = -r12, force on i
          const dx = -r12[0], dy = -r12[1], dz = -r12[2];
          const fx = fi[0], fy = fi[1], fz = fi[2];
          const wv = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * wv[c]; va[6 * j + c] += 0.5 * wv[c]; }
        }
      }
    }
    // As native: the fdotr virial over the pair forces of owned atoms and ghosts (pair.cpp virial_fdotr_compute at the
    // end of compute), then reverse_comm folds the ghost forces and torques into their owners, in native order. This
    // holds when gayberne is the only pair style (a hybrid sub-style shares the forces with the other sub-styles).
    if (!pc.geom.triclinic && this.newtonPair && pc.f === nb.fall && list === nb.half) {
      this.comm.fdotrFold(s, nb, pc.geom, x, f, pc.acc.virial);
      pc.fdotrDone = true;
      this.comm.foldInto(s, nb, pc.geom, tq, s.torque);
    } else {
      nb.reverseSum(tq, 3, s.torque);
    }
    pc.acc.evdwl += evdwl;
  }
}
