/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BPM/bond_bpm_rotational.cpp and src/BPM/bond_bpm_rotational.h at commit b332f28 (the average and
 * particle reference-frame force routines, the Alkuino et al. and Wang et al. DEM damping, compute, coeff,
 * settings and init_style). Authors credited in the BPM sources: Joel Clemmer (SNL), with the frame and damping
 * formulations of Y. Wang (Acta Geotechnica 2009), P. Mora and Y. Wang (2009), Y. Wang, F. Alonso-Marroquin and
 * W. Guo (2015) and Alkuino et al. (2026). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * bond_style bpm/rotational: docs.lammps.org/bond_bpm_rotational.html. The doc says "The bpm/rotational bond
 * style computes forces and torques based on deviations from the initial reference state of the two atoms" and
 * "Bonds will break at a strain of epsilon_c" is replaced by the critical forces and torques fcr, fcs, tct, tcb.
 * Browser-engine differences: the reference state of a bond (r0, the unit vector, and the derivative-damping
 * priors gamma, theta, psi) is a map keyed by the two atom IDs (LAMMPS: fix BOND_HISTORY); the atom quaternions
 * and angular velocities are the SimState arrays (atom_style bpm/sphere). The engine has no ghost atoms, so
 * displacements are minimum-image displacements.
 */
import { delta } from '../../engine/force/bonded_util';
import { StyleError, typeBounds, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import { bondKey, BondBPMBase, yesno } from './bond_bpm';
import { add3, cross3, dot3, len3, normalize3, quatquat, qconjugate, qnormalize, quatrotvec, scale3, sub3, type Quat, type Vec3 } from './quat';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { pow as libmPow } from '../libm/pow';
import { acos as libmAcos } from '../libm/acos';

// LAMMPS: static constexpr double EPSILON = 1e-10 (bond_bpm_rotational.cpp)
const EPSILON = 1e-10;
const SQRT2 = Math.SQRT2;
const PI = Math.PI;

const acosLimit = (c: number): number => libmAcos(Math.min(1, Math.max(-1, c)));

/** Wraps an angle difference into (-pi, pi] as the LAMMPS wrap_2pi. */
const wrap2pi = (t: number): number => {
  let u = t;
  if (u > PI) u -= 2 * PI;
  if (u < -PI) u += 2 * PI;
  return u;
};

/** Average of two quaternions by the shortest-path SLERP midpoint (quat_average in the C++ file). */
const quatAverage = (qA: Quat, qB: Quat): Quat => {
  let qBuse = qB;
  let dot = qA[0] * qB[0] + qA[1] * qB[1] + qA[2] * qB[2] + qA[3] * qB[3];
  if (dot < 0) {
    qBuse = [-qB[0], -qB[1], -qB[2], -qB[3]];
    dot = -dot;
  }
  const s = 1 / Math.sqrt(2 * (1 + dot));
  return qnormalize([(qA[0] + qBuse[0]) * s, (qA[1] + qBuse[1]) * s, (qA[2] + qBuse[2]) * s, (qA[3] + qBuse[3]) * s]);
};

/** Swing-twist decomposition of a quaternion by Wang's formula (decompose_swing_twist). */
const decomposeSwingTwist = (q: Quat): { swing: Quat; twist: Quat } => {
  const tmp = Math.sqrt(q[0] * q[0] + q[3] * q[3]);
  let psi = tmp < EPSILON ? 0 : 2 * acosLimit(q[0] / tmp);
  if (q[3] < 0) psi = -psi;
  const twist: Quat = [libmCos(psi / 2), 0, 0, libmSin(psi / 2)];
  const swing = qnormalize(quatquat(qconjugate(twist), q));
  return { swing, twist };
};

/** quatrotvec's result (rotv). */
const R3 = new Float64Array(3);
/** quatrotvec (quat.ts, MathExtra::quatrotvec) of (b0, b1, b2) by (a0, a1, a2, a3), into R3. */
const rotv = (a0: number, a1: number, a2: number, a3: number, b0: number, b1: number, b2: number): void => {
  const t0 = -a1 * b0 - a2 * b1 - a3 * b2;
  const t1 = a0 * b0 + a2 * b2 - a3 * b1;
  const t2 = a0 * b1 + a3 * b0 - a1 * b2;
  const t3 = a0 * b2 + a1 * b1 - a2 * b0;
  R3[0] = -a1 * t0 + a0 * t1 - a3 * t2 + a2 * t3;
  R3[1] = -a2 * t0 + a3 * t1 + a0 * t2 - a1 * t3;
  R3[2] = -a3 * t0 - a2 * t1 + a1 * t2 + a0 * t3;
};
/** quatquat's result (qmul). */
const Q4 = new Float64Array(4);
/** quatquat (quat.ts, MathExtra::quatquat) a b, into Q4. */
const qmul = (a0: number, a1: number, a2: number, a3: number, b0: number, b1: number, b2: number, b3: number): void => {
  const r0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3;
  const r1 = a0 * b1 + b0 * a1 + a2 * b3 - a3 * b2;
  const r2 = a0 * b2 + b0 * a2 + a3 * b1 - a1 * b3;
  const r3 = a0 * b3 + b0 * a3 + a1 * b2 - a2 * b1;
  Q4[0] = r0; Q4[1] = r1; Q4[2] = r2; Q4[3] = r3;
};
/** decomposeSwingTwist's twist (0..3) and swing (4..7) (swingTwist). */
const ST = new Float64Array(8);
/** decomposeSwingTwist of (q0, q1, q2, q3) into ST, in its operation order. */
const swingTwist = (q0: number, q1: number, q2: number, q3: number): void => {
  const tmp = Math.sqrt(q0 * q0 + q3 * q3);
  let psi = tmp < EPSILON ? 0 : 2 * acosLimit(q0 / tmp);
  if (q3 < 0) psi = -psi;
  const tw0 = libmCos(psi / 2), tw3 = libmSin(psi / 2);
  // swing = qnormalize(quatquat(qconjugate(twist), q)), twist = (tw0, 0, 0, tw3)
  qmul(tw0, -0, -0, -tw3, q0, q1, q2, q3);
  const n = 1 / Math.sqrt(Q4[0] * Q4[0] + Q4[1] * Q4[1] + Q4[2] * Q4[2] + Q4[3] * Q4[3]);
  ST[0] = tw0; ST[1] = 0; ST[2] = 0; ST[3] = tw3;
  ST[4] = Q4[0] * n; ST[5] = Q4[1] * n; ST[6] = Q4[2] * n; ST[7] = Q4[3] * n;
};

/** The loads on a bond: force and torques on atom i2 (force1on2, torque1on2) and on atom i1 (torque2on1). */
interface Loads {
  f1on2: Vec3;
  t1on2: Vec3;
  t2on1: Vec3;
  breaking: number;
  ebond: number;
}

/** Per-bond reference state (LAMMPS bondstore): r0, unit vector ri/r0, gamma, theta, psi priors. */
type Hist = Float64Array;

export class BondBpmRotational extends BondBPMBase {
  readonly name = 'bpm/rotational';
  protected readonly localStoreOk = true;
  readonly kind = 'bond' as const;
  readonly paramNames = ['kr', 'ks', 'kt', 'kb', 'fcr', 'fcs', 'tct', 'tcb', 'gr', 'gs', 'gt', 'gb'];
  smooth = true;
  normalize = false;
  frame: 'average' | 'particle' = 'average';
  damping: 'derivative' | 'dem' = 'derivative';
  private readonly hist = new Map<number, Hist>();
  /**
   * Per-bond damping priors (gamma, theta, psi) as the atoms hold them: native LAMMPS copies bondstore into the
   * per-atom fix bond/history storage at each reneighbouring (FixBondHistory::pre_exchange), and single() reads that
   * copy. Taken at each neighbor-list build (bc.nbuild), before the forces of that step update hist.
   */
  private readonly synced = new Map<number, Float64Array>();
  private syncBuild = -1;

  settings(args: string[], _ctx?: StyleContext): void {
    const rest = this.takeCommon(args);
    for (let k = 0; k < rest.length; k += 2) {
      const key = rest[k], val = rest[k + 1];
      // bond_bpm_rotational.cpp settings: the keyword, then its value
      if (!['smooth', 'normalize', 'frame', 'damping'].includes(key)) throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
      if (val === undefined) throw missingArgs(`bond_style bpm/rotational ${key}`);
      if (key === 'smooth') this.smooth = yesno(val, 'bond_style bpm/rotational', key);
      else if (key === 'normalize') this.normalize = yesno(val, 'bond_style bpm/rotational', key);
      else if (key === 'frame') {
        if (val !== 'average' && val !== 'particle') throw new StyleError(`Unknown frame style ${val}`);
        this.frame = val;
      } else if (key === 'damping') {
        if (val !== 'derivative' && val !== 'dem') throw new StyleError(`Unknown damping style ${val}`);
        this.damping = val;
      } else throw new StyleError(`Illegal bond bpm command, invalid argument ${key}`);
    }
    if (!this.breakOn && this.smooth) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  /** bond_coeff N kr ks kt kb fcr fcs tct tcb gr gs gt gb (the doc's order of the coefficient list). */
  coeff(args: string[], _ctx?: StyleContext): void {
    // As native LAMMPS does (src/BPM/bond_bpm_rotational.cpp BondBPMRotational::coeff at b332f28): the argument count
    // (type and 12 values) is checked first, then the type range, then utils::numeric reads each value.
    if (args.length !== this.paramNames.length + 1) {
      throw new StyleError('Incorrect args for bond coefficients: expected Kr Ks Kt Kb Fcr Fcs Tct Tcb gr gs gt gb');
    }
    typeBounds(args[0], this.ntypes);
    const v = this.parse(args.slice(1));
    // no range check on fcr, fcs, tct, tcb: native BondBPMRotational::coeff (bond_bpm_rotational.cpp at b332f28)
    // takes them as given; zero or negative values are used in the breaking sum (see the oracle cases)
    this.params.setRange(args[0], v);
  }

  init(ctx: StyleContext): void {
    super.init(ctx);
    const s = ctx.s;
    if (s && (!s.quat || !s.radius || !s.omega)) throw new StyleError('Bond bpm/rotational requires atom style bpm/sphere');
  }

  /** Stores the reference state of a bond (store_data / store_bond): r0 and the unit vector from the smaller to the larger ID. */
  private reference(bc: BondedCompute, i1: number, i2: number, key: number): Hist {
    let h = this.hist.get(key);
    if (!h) {
      const d = [0, 0, 0];
      delta(bc, i1, i2, d);
      const r = len3(d);
      h = new Float64Array(7);
      h[0] = r;
      h[1] = d[0] / r; h[2] = d[1] / r; h[3] = d[2] / r;
      this.hist.set(key, h);
    }
    return h;
  }

  /** The loads of a bond of type t between atoms i1 and i2 with reference state h (the frame and damping of the style). */
  private loads(bc: BondedCompute, i1: number, i2: number, t: number, h: Hist): Loads {
    // the coefficient arrays in paramNames order (kr ks kt kb fcr fcs tct tcb gr gs gt gb)
    const D = this.params.data;
    const r0 = h[0];
    const ri: Vec3 = [h[1] * r0, h[2] * r0, h[3] * r0];
    const d = [0, 0, 0];
    delta(bc, i1, i2, d);
    const rf: Vec3 = [d[0], d[1], d[2]];
    const params = {
      Kr: D[0][t], Ks: D[1][t], Kt: D[2][t], Kb: D[3][t],
      Fcr: D[4][t], Fcs: D[5][t], Tct: D[6][t], Tcb: D[7][t],
      gr: D[8][t], gs: D[9][t], gt: D[10][t], gb: D[11][t],
    };
    const L = this.frame === 'average' ? this.averageFrame(bc, i1, i2, params, ri, rf, h) : this.particleFrame(bc, i1, i2, params, ri, rf, h);
    if (this.damping === 'dem') this.demDamping(bc, i1, i2, params, rf, L);
    return L;
  }

  /**
   * compute bond/local bN (native BondBPMRotational::single at b332f28, svector): [ri_norm, ri (3), the smoothed force
   * on the second atom (3)] of the bond with atom IDs ia, ib (ia first as stored), each vector negated when ia has the
   * larger ID. The reference state is a copy (native single() copies bondstore), so the history is not changed.
   */
  bondExtra(bc: BondedCompute, ia: number, ib: number, t: number): number[] {
    this.syncHistory(bc);
    const { i1, i2, a1, a2 } = this.pairOf(bc, ia, ib);
    const key = bondKey(a1, a2);
    const h = Float64Array.from(this.reference(bc, i1, i2, key));
    // the damping priors are the per-atom copy (zero before the first copy: store_bond sets them to 0)
    const snap = this.synced.get(key);
    h[4] = snap ? snap[0] : 0; h[5] = snap ? snap[1] : 0; h[6] = snap ? snap[2] : 0;
    const L = this.loads(bc, i1, i2, t, h);
    const smooth = this.smooth ? 1 - libmPow(L.breaking, 4) : 1;
    const F = scale3(smooth, L.f1on2);
    const sg = ia > ib ? -1 : 1;
    const r0 = h[0];
    return [r0, sg * h[1] * r0, sg * h[2] * r0, sg * h[3] * r0, sg * F[0], sg * F[1], sg * F[2]];
  }

  /** Copies the damping priors of every bond at a new neighbor-list build (native pre_exchange). */
  private syncHistory(bc: BondedCompute): void {
    if (bc.nbuild === undefined || bc.nbuild === this.syncBuild) return;
    this.syncBuild = bc.nbuild;
    for (const [key, h] of this.hist) this.synced.set(key, Float64Array.of(h[4], h[5], h[6]));
  }

  compute(bc: BondedCompute): void {
    this.syncHistory(bc);
    const b = bc.s.topo.bonds;
    const s = bc.s;
    const d = [0, 0, 0];
    const order = this.visitOrder(bc);
    const nVisit = order ? order.length : b.n;
    for (let q = 0; q < nVisit; q++) {
      const k = order ? order[q] : q;
      const t = b.type[k];
      if (t === 0) continue; // broken bond: bond type 0
      const ia = b.atoms[2 * k], ib = b.atoms[2 * k + 1];
      const { i1, i2, a1, a2 } = this.pairOf(bc, ia, ib);
      const h = this.reference(bc, i1, i2, bondKey(a1, a2));
      delta(bc, i1, i2, d);
      const rf: Vec3 = [d[0], d[1], d[2]];
      const L = this.loads(bc, i1, i2, t, h);
      if (L.breaking >= 1 && this.breakOn) {
        this.breakBond(b.type, k, b.atoms);
        this.emitBreak(bc, ia, ib);
        continue;
      }
      const smooth = this.smooth ? 1 - libmPow(L.breaking, 4) : 1;
      const F = scale3(smooth, L.f1on2);
      const T1 = scale3(smooth, L.t1on2);
      const T2 = scale3(smooth, L.t2on1);
      for (let c = 0; c < 3; c++) {
        s.f[3 * i1 + c] -= F[c];
        s.f[3 * i2 + c] += F[c];
        s.torque![3 * i1 + c] += T2[c];
        s.torque![3 * i2 + c] += T1[c];
      }
      bc.acc.ebond += L.ebond;
      // ev_tally_xyz(i1, i2, ..., fx = -F (force on i1), del = rf): v = del (x) f, as Bond::ev_tally_xyz
      const fi1: Vec3 = scale3(-1, F);
      const v = bc.virial;
      v[0] += rf[0] * fi1[0]; v[1] += rf[1] * fi1[1]; v[2] += rf[2] * fi1[2];
      v[3] += rf[0] * fi1[1]; v[4] += rf[0] * fi1[2]; v[5] += rf[1] * fi1[2];
      if (bc.eatom) { bc.eatom[i1] += 0.5 * L.ebond; bc.eatom[i2] += 0.5 * L.ebond; }
      // per-atom: half of v to each atom (Bond::ev_tally_xyz, newton_bond on)
      if (bc.vatom) { const va = bc.vatom; const w = [rf[0] * fi1[0], rf[1] * fi1[1], rf[2] * fi1[2], rf[0] * fi1[1], rf[0] * fi1[2], rf[1] * fi1[2]]; for (let c = 0; c < 6; c++) { va[6 * i1 + c] += 0.5 * w[c]; va[6 * i2 + c] += 0.5 * w[c]; } }
    }
  }

  /**
   * The average-frame elastic and derivative-damped loads (average_frame_forces), in scalars: every value and operation
   * order is that of the vector form (quat.ts helpers: quatAverage, quatrotvec, cross3, normalize3, quatquat,
   * qnormalize, decomposeSwingTwist), the quaternion products and rotations through rotv and qmul.
   */
  private averageFrame(bc: BondedCompute, i1: number, i2: number, p: Record<string, number>, ri: Vec3, rf: Vec3, h: Hist): Loads {
    const quat = bc.s.quat!, v = bc.s.v;
    const ri0 = ri[0], ri1 = ri[1], ri2 = ri[2];
    const rf0 = rf[0], rf1 = rf[1], rf2 = rf[2];
    const riNorm = Math.sqrt(ri0 * ri0 + ri1 * ri1 + ri2 * ri2);
    let Kr = p.Kr, Ks = p.Ks;
    if (this.normalize) { Kr *= 1 / riNorm; Ks *= 1 / riNorm; }
    const a0 = quat[4 * i1], a1 = quat[4 * i1 + 1], a2 = quat[4 * i1 + 2], a3 = quat[4 * i1 + 3];
    const b0 = quat[4 * i2], b1 = quat[4 * i2 + 1], b2 = quat[4 * i2 + 2], b3 = quat[4 * i2 + 3];
    // qcf = quatAverage(q1f, q2f)
    let dot = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
    let u0 = b0, u1 = b1, u2 = b2, u3 = b3;
    if (dot < 0) { u0 = -b0; u1 = -b1; u2 = -b2; u3 = -b3; dot = -dot; }
    const sAvg = 1 / Math.sqrt(2 * (1 + dot));
    const m0 = (a0 + u0) * sAvg, m1 = (a1 + u1) * sAvg, m2 = (a2 + u2) * sAvg, m3 = (a3 + u3) * sAvg;
    const nAvg = 1 / Math.sqrt(m0 * m0 + m1 * m1 + m2 * m2 + m3 * m3);
    const c0 = m0 * nAvg, c1 = m1 * nAvg, c2 = m2 * nAvg, c3 = m3 * nAvg;
    // qcfInv = qconjugate(qcf); rc = quatrotvec(qcfInv, rf)
    const ci1 = -c1, ci2 = -c2, ci3 = -c3;
    rotv(c0, ci1, ci2, ci3, rf0, rf1, rf2);
    const rc0 = R3[0], rc1 = R3[1], rc2 = R3[2];
    const rcNorm = Math.sqrt(rc0 * rc0 + rc1 * rc1 + rc2 * rc2);
    const FrElastic = Kr * (rcNorm - riNorm);
    let Fr = FrElastic;
    const riDotRc = ri0 * rc0 + ri1 * rc1 + ri2 * rc2;
    const gamma = acosLimit(riDotRc / (riNorm * rcNorm));
    const x0 = ri1 * rc2 - ri2 * rc1, x1 = ri2 * rc0 - ri0 * rc2, x2 = ri0 * rc1 - ri1 * rc0;
    const rixrcMag = Math.sqrt(x0 * x0 + x1 * x1 + x2 * x2);
    // rcHat = normalize3(rc)
    let rh0 = 0, rh1 = 0, rh2 = 0;
    if (rcNorm > 0) { const iv = 1 / rcNorm; rh0 = iv * rc0; rh1 = iv * rc1; rh2 = iv * rc2; }
    let FsElastic = 0, Fs = 0;
    let sh0 = 0, sh1 = 0, sh2 = 0, th0 = 0, th1 = 0, th2 = 0;
    if (rixrcMag > EPSILON) {
      const it = 1 / rixrcMag;
      th0 = it * x0; th1 = it * x1; th2 = it * x2;
      const y0 = th1 * rh2 - th2 * rh1, y1 = th2 * rh0 - th0 * rh2, y2 = th0 * rh1 - th1 * rh0;
      const ny = Math.sqrt(y0 * y0 + y1 * y1 + y2 * y2);
      if (ny > 0) { const iy = 1 / ny; sh0 = iy * y0; sh1 = iy * y1; sh2 = iy * y2; }
      FsElastic = Ks * rcNorm * gamma;
      Fs = FsElastic;
    }
    // qu0 = qnormalize(quatquat(qcfInv, q2f))
    qmul(c0, ci1, ci2, ci3, b0, b1, b2, b3);
    let n = 1 / Math.sqrt(Q4[0] * Q4[0] + Q4[1] * Q4[1] + Q4[2] * Q4[2] + Q4[3] * Q4[3]);
    const p0 = Q4[0] * n, p1 = Q4[1] * n, p2 = Q4[2] * n, p3 = Q4[3] * n;
    // qm: the rotation that takes the z axis onto rc
    let temp = Math.max(0, rcNorm + rc2);
    const qm0 = SQRT2 * 0.5 * Math.sqrt(temp / rcNorm);
    let qm1 = 0, qm2 = 0;
    temp = Math.sqrt(rc0 * rc0 + rc1 * rc1);
    if (temp > EPSILON) {
      const temp2 = Math.max(0, rcNorm - rc2);
      const factor = (-SQRT2 * 0.5 * Math.sqrt(temp2 / rcNorm)) / temp;
      qm1 = factor * rc1;
      qm2 = -factor * rc0;
    }
    const qm3 = 0;
    if (qm0 === 0 && qm1 === 0 && qm2 === 0) qm2 = 1;
    n = 1 / Math.sqrt(qm0 * qm0 + qm1 * qm1 + qm2 * qm2 + qm3 * qm3);
    const e0 = qm0 * n, e1 = qm1 * n, e2 = qm2 * n, e3 = qm3 * n;
    // qu = qnormalize(quatquat(quatquat(qconjugate(qmN), qu0), qmN))
    qmul(e0, -e1, -e2, -e3, p0, p1, p2, p3);
    qmul(Q4[0], Q4[1], Q4[2], Q4[3], e0, e1, e2, e3);
    n = 1 / Math.sqrt(Q4[0] * Q4[0] + Q4[1] * Q4[1] + Q4[2] * Q4[2] + Q4[3] * Q4[3]);
    const g0 = Q4[0] * n, g1 = Q4[1] * n, g2 = Q4[2] * n, g3 = Q4[3] * n;
    // sw2 = decomposeSwingTwist(qu), sw1 = decomposeSwingTwist(qconjugate(qu))
    swingTwist(g0, g1, g2, g3);
    const tw20 = ST[0], tw23 = ST[3], sw20 = ST[4], sw21 = ST[5], sw22 = ST[6], sw23 = ST[7];
    swingTwist(g0, -g1, -g2, -g3);
    const tw10 = ST[0], tw13 = ST[3], sw10 = ST[4], sw11 = ST[5], sw12 = ST[6], sw13 = ST[7];
    // twist = quatquat(sw2.twist, qconjugate(sw1.twist)) (the twists' i and j are 0, conjugated -0)
    qmul(tw20, 0, 0, tw23, tw10, -0, -0, -tw13);
    const k0 = Q4[0], k1 = Q4[1], k2 = Q4[2], k3 = Q4[3];
    const psi = 2 * acosLimit(k0);
    let ta0 = 0, ta1 = 0, ta2 = 1;
    if (Math.abs(psi) > EPSILON) {
      const nt = Math.sqrt(k1 * k1 + k2 * k2 + k3 * k3);
      if (nt > 0) { const iv = 1 / nt; ta0 = iv * k1; ta1 = iv * k2; ta2 = iv * k3; } else { ta2 = 0; }
    }
    // swing = quatquat(sw2.swing, qconjugate(sw1.swing))
    qmul(sw20, sw21, sw22, sw23, sw10, -sw11, -sw12, -sw13);
    const w0 = Q4[0], w1 = Q4[1], w2 = Q4[2], w3 = Q4[3];
    const theta = 2 * acosLimit(w0);
    let ba0 = 0, ba1 = 0, ba2 = 1;
    if (Math.abs(theta) > EPSILON) {
      const nb = Math.sqrt(w1 * w1 + w2 * w2 + w3 * w3);
      if (nb > 0) { const iv = 1 / nb; ba0 = iv * w1; ba1 = iv * w2; ba2 = iv * w3; } else { ba2 = 0; }
    }
    const TtElastic = p.Kt * psi;
    let Tt = TtElastic;
    const TbElastic = p.Kb * theta;
    let Tb = TbElastic;
    if (this.damping === 'derivative') {
      const dtInv = 1 / bc.s.dt;
      rotv(c0, ci1, ci2, ci3, v[3 * i2] - v[3 * i1], v[3 * i2 + 1] - v[3 * i1 + 1], v[3 * i2 + 2] - v[3 * i1 + 2]);
      Fr += p.gr * (R3[0] * rh0 + R3[1] * rh1 + R3[2] * rh2);
      if (rixrcMag > EPSILON) Fs += p.gs * rcNorm * (wrap2pi(gamma - h[4]) * dtInv);
      Tt += p.gt * (wrap2pi(psi - h[6]) * dtInv);
      Tb += p.gb * (wrap2pi(theta - h[5]) * dtInv);
      h[4] = gamma;
      h[5] = theta;
      h[6] = psi;
    }
    const FrC0 = Fr * rh0, FrC1 = Fr * rh1, FrC2 = Fr * rh2;
    let Fs0 = 0, Fs1 = 0, Fs2 = 0, Ts0 = 0, Ts1 = 0, Ts2 = 0;
    if (rixrcMag > EPSILON) {
      Fs0 = Fs * sh0; Fs1 = Fs * sh1; Fs2 = Fs * sh2;
      const st = 0.5 * rcNorm * Fs;
      Ts0 = st * th0; Ts1 = st * th1; Ts2 = st * th2;
    }
    // TbC = quatrotvec(qmN, TbP), TtC = quatrotvec(qmN, TtP)
    rotv(e0, e1, e2, e3, Tb * ba0, Tb * ba1, Tb * ba2);
    const tb0 = R3[0], tb1 = R3[1], tb2 = R3[2];
    rotv(e0, e1, e2, e3, Tt * ta0, Tt * ta1, Tt * ta2);
    const tt0 = tb0 + R3[0], tt1 = tb1 + R3[1], tt2 = tb2 + R3[2];
    const L = this.scratch;
    // force2on1 = quatrotvec(qcf, F1C); f1on2 = -force2on1
    rotv(c0, c1, c2, c3, FrC0 + Fs0, FrC1 + Fs1, FrC2 + Fs2);
    L.f1on2[0] = -1 * R3[0]; L.f1on2[1] = -1 * R3[1]; L.f1on2[2] = -1 * R3[2];
    rotv(c0, c1, c2, c3, Ts0 - tt0, Ts1 - tt1, Ts2 - tt2);
    L.t1on2[0] = R3[0]; L.t1on2[1] = R3[1]; L.t1on2[2] = R3[2];
    rotv(c0, c1, c2, c3, Ts0 + tt0, Ts1 + tt1, Ts2 + tt2);
    L.t2on1[0] = R3[0]; L.t2on1[1] = R3[1]; L.t2on1[2] = R3[2];
    const breaking0 = Math.abs(FrElastic) / p.Fcr + FsElastic / p.Fcs + TbElastic / p.Tcb + TtElastic / p.Tct;
    L.breaking = Math.max(0, breaking0);
    L.ebond = 0.5 * (Kr * (rcNorm - riNorm) * (rcNorm - riNorm) + Ks * rcNorm * rcNorm * gamma * gamma + p.Kt * psi * psi + p.Kb * theta * theta);
    return L;
  }

  /** averageFrame's result (one bond at a time; read before the next call). */
  private readonly scratch: Loads = { f1on2: [0, 0, 0], t1on2: [0, 0, 0], t2on1: [0, 0, 0], breaking: 0, ebond: 0 };

  /** The particle-frame loads (particle_frame_forces). */
  private particleFrame(bc: BondedCompute, i1: number, i2: number, p: Record<string, number>, r0neg: Vec3, r: Vec3, h: Hist): Loads {
    const quat = bc.s.quat!, v = bc.s.v;
    const rMag = len3(r);
    const rMagInv = 1 / rMag;
    const r0Mag = len3(r0neg);
    const r0MagInv = 1 / r0Mag;
    let Kr = p.Kr, Ks = p.Ks;
    if (this.normalize) { Kr *= r0MagInv; Ks *= r0MagInv; }
    const q1: Quat = [quat[4 * i1], quat[4 * i1 + 1], quat[4 * i1 + 2], quat[4 * i1 + 3]];
    const q2: Quat = [quat[4 * i2], quat[4 * i2 + 1], quat[4 * i2 + 2], quat[4 * i2 + 3]];
    const q2inv = qconjugate(q2);
    const rb = scale3(-1, quatrotvec(q2inv, r));
    const r0 = scale3(-1, r0neg);
    let Fr = Kr * (rMag - r0Mag);
    const r0Dotrb = dot3(r0, rb);
    const gamma = acosLimit(r0Dotrb * rMagInv * r0MagInv);
    const rbXr0 = cross3(rb, r0);
    const sVec = normalize3(cross3(rb, rbXr0));
    let Fs = Ks * rMag * gamma;
    const tVec = normalize3(cross3(r0, rb));
    let temp = Math.max(0, rMag + rb[2]);
    const mq: Quat = [0, 0, 0, 0];
    mq[0] = SQRT2 * 0.5 * Math.sqrt(temp * rMagInv);
    temp = Math.sqrt(rb[0] * rb[0] + rb[1] * rb[1]);
    if (temp !== 0) {
      mq[1] = -SQRT2 * 0.5 / temp;
      mq[1] *= Math.sqrt(Math.max(0, rMag - rb[2]) * rMagInv);
      mq[2] = -mq[1];
      mq[1] *= rb[1];
      mq[2] *= rb[0];
    }
    mq[3] = 0;
    const qp21 = quatquat(q2inv, q1);
    const q21 = quatquat(quatquat(qconjugate(mq), qp21), mq);
    temp = Math.sqrt(q21[0] * q21[0] + q21[3] * q21[3]);
    let psi = temp !== 0 ? 2 * acosLimit(q21[0] / temp) : 0;
    if (q21[3] < 0) psi = -psi;
    if (q21[3] === 0) psi = 0;
    const c2 = q21[0] * q21[0] - q21[1] * q21[1] - q21[2] * q21[2] + q21[3] * q21[3];
    const theta = acosLimit(c2);
    const magOut = q21[0] * q21[0] + q21[3] * q21[3];
    const magIn = q21[1] * q21[1] + q21[2] * q21[2];
    let cosPhi = 0, sinPhi = 0;
    if (magIn === 0) {
      cosPhi = 0; sinPhi = 0;
    } else if (magOut === 0) {
      cosPhi = q21[2] / Math.sqrt(magIn);
      sinPhi = -q21[1] / Math.sqrt(magIn);
    } else {
      cosPhi = (q21[1] * q21[3] + q21[0] * q21[2]) / Math.sqrt(magOut * magIn);
      sinPhi = (q21[2] * q21[3] - q21[0] * q21[1]) / Math.sqrt(magOut * magIn);
    }
    const Tbp: Vec3 = [-p.Kb * theta * sinPhi, p.Kb * theta * cosPhi, 0];
    let Ttp: Vec3 = [0, 0, p.Kt * psi];
    const Fsrp: Vec3 = [-0.5 * Ks * rMag * theta * cosPhi, -0.5 * Ks * rMag * theta * sinPhi, 0];
    const Tsrp: Vec3 = [0.25 * Ks * rMag * rMag * theta * sinPhi, -0.25 * Ks * rMag * rMag * theta * cosPhi, 0];
    if (this.damping === 'derivative') {
      const dtInv = 1 / bc.s.dt;
      const dv = sub3([v[3 * i2], v[3 * i2 + 1], v[3 * i2 + 2]], [v[3 * i1], v[3 * i1 + 1], v[3 * i1 + 2]]);
      const rhat = scale3(rMagInv, r);
      Fr += p.gr * dot3(dv, rhat);
      Fs += p.gs * rMag * (wrap2pi(gamma - h[4]) * dtInv);
      Ttp = [0, 0, Ttp[2] + p.gt * (wrap2pi(psi - h[6]) * dtInv)];
      const dtheta = wrap2pi(theta - h[5]) * dtInv;
      Tbp[0] += -p.gb * dtheta * sinPhi;
      Tbp[1] += p.gb * dtheta * cosPhi;
      Fsrp[0] += -0.5 * p.gs * rMag * dtheta * cosPhi;
      Fsrp[1] += -0.5 * p.gs * rMag * dtheta * sinPhi;
      Tsrp[0] += 0.25 * p.gs * rMag * rMag * dtheta * sinPhi;
      Tsrp[1] += -0.25 * p.gs * rMag * rMag * dtheta * cosPhi;
      h[4] = gamma;
      h[5] = theta;
      h[6] = psi;
    }
    const Fn = scale3(Fr * rMagInv, rb);
    const FrotS = scale3(Fs, sVec);
    const Tst = scale3(0.5 * rMag * Fs, tVec);
    const Fsr = quatrotvec(mq, Fsrp);
    const Tsr = quatrotvec(mq, Tsrp);
    const Tb = quatrotvec(mq, Tbp);
    const Tt = quatrotvec(mq, Ttp);
    const Frot: Vec3 = [Fn[0] + FrotS[0] + Fsr[0], Fn[1] + FrotS[1] + Fsr[1], Fn[2] + FrotS[2] + Fsr[2]];
    const f1on2 = quatrotvec(q2, Frot);
    const Trot1: Vec3 = [Tst[0] + Tsr[0] + Tt[0] + Tb[0], Tst[1] + Tsr[1] + Tt[1] + Tb[1], Tst[2] + Tsr[2] + Tt[2] + Tb[2]];
    const Trot2: Vec3 = [Tst[0] + Tsr[0] - Tt[0] - Tb[0], Tst[1] + Tsr[1] - Tt[1] - Tb[1], Tst[2] + Tsr[2] - Tt[2] - Tb[2]];
    const FsTot = add3(FrotS, Fsr);
    // the doc's breaking sum uses the damped radial force Fr and the total shear magnitude (as particle_frame_forces)
    const breaking0 = Fr / p.Fcr + len3(FsTot) / p.Fcs + len3(Tb) / p.Tcb + len3(Tt) / p.Tct;
    const ebond = 0.5 * (Kr * (rMag - r0Mag) * (rMag - r0Mag) + Ks * rMag * rMag * gamma * gamma + p.Kt * psi * psi + p.Kb * theta * theta);
    return {
      f1on2,
      t1on2: quatrotvec(q2, Trot1),
      t2on1: quatrotvec(q2, Trot2),
      breaking: Math.max(0, breaking0),
      ebond,
    };
  }

  /** DEM damping (dem_damping_forces): adds the velocity and spin damping terms to the loads of one bond. */
  private demDamping(bc: BondedCompute, i1: number, i2: number, p: Record<string, number>, r: Vec3, L: Loads): void {
    const s = bc.s;
    const rhat = normalize3(r);
    const vel = (i: number): Vec3 => [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    const om = (i: number): Vec3 => [s.omega![3 * i], s.omega![3 * i + 1], s.omega![3 * i + 2]];
    const v1 = vel(i1), v2 = vel(i2), w1 = om(i1), w2 = om(i2);
    const vn1 = scale3(dot3(v1, rhat), rhat);
    const vn2 = scale3(dot3(v2, rhat), rhat);
    const vt1 = sub3(v1, vn1), vt2 = sub3(v2, vn2);
    const half = scale3(0.5, sub3(vt2, vt1));
    // Eqs. 12 and 13: the surface velocities of the two spheres
    const s1 = sub3(scale3(0.5, cross3(w1, r)), half);
    const s2 = add3(scale3(-0.5, cross3(w2, r)), half);
    // normal, shear and torque damping forces
    const fdamp = add3(scale3(p.gr, sub3(vn1, vn2)), scale3(p.gs, sub3(s1, s2)));
    const tShear = scale3(-0.5, cross3(r, scale3(p.gs, sub3(s1, s2))));
    // Eq. 31: rolling (bending) damping
    const vroll = sub3(cross3(w1, rhat), cross3(w2, rhat));
    const tBend = scale3(0.5 * p.gb, cross3(r, vroll));
    // Eq. 38: twisting damping
    const tTwist = scale3(0.5 * p.gt, sub3(scale3(dot3(w1, rhat), rhat), scale3(dot3(w2, rhat), rhat)));
    L.f1on2 = add3(L.f1on2, fdamp);
    L.t1on2 = add3(add3(L.t1on2, tShear), add3(tBend, tTwist));
    L.t2on1 = sub3(add3(L.t2on1, tShear), add3(tBend, tTwist));
  }
}

