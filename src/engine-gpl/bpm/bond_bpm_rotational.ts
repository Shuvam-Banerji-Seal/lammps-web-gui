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
import { StyleError, type BondedCompute, type StyleContext } from '../../engine/force/types';
import { bondKey, BondBPMBase, yesno } from './bond_bpm';
import { add3, cross3, dot3, len3, normalize3, quatquat, qconjugate, qnormalize, quatrotvec, scale3, sub3, type Quat, type Vec3 } from './quat';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

// LAMMPS: static constexpr double EPSILON = 1e-10 (bond_bpm_rotational.cpp)
const EPSILON = 1e-10;
const SQRT2 = Math.SQRT2;
const PI = Math.PI;

const acosLimit = (c: number): number => Math.acos(Math.min(1, Math.max(-1, c)));

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
      if (val === undefined) throw new StyleError(`bond_style bpm/rotational: keyword '${key}' needs a value`);
      if (key === 'smooth') this.smooth = yesno(val, 'bond_style bpm/rotational', key);
      else if (key === 'normalize') this.normalize = yesno(val, 'bond_style bpm/rotational', key);
      else if (key === 'frame') {
        if (val !== 'average' && val !== 'particle') throw new StyleError(`bond_style bpm/rotational frame must be average or particle (got '${val}')`);
        this.frame = val;
      } else if (key === 'damping') {
        if (val !== 'derivative' && val !== 'dem') throw new StyleError(`bond_style bpm/rotational damping must be derivative or dem (got '${val}')`);
        this.damping = val;
      } else throw new StyleError(`illegal bond_style bpm/rotational keyword '${key}'`);
    }
    if (!this.breakOn && this.smooth) throw new StyleError('Illegal bond bpm command, must turn off smoothing with break no option');
  }

  /** bond_coeff N kr ks kt kb fcr fcs tct tcb gr gs gt gb (the doc's order of the coefficient list). */
  coeff(args: string[], _ctx?: StyleContext): void {
    if (!args.length) throw new StyleError('usage: bond_coeff N coefficients');
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
    const P = (name: string): number => this.params.p(name)[t];
    const r0 = h[0];
    const ri: Vec3 = [h[1] * r0, h[2] * r0, h[3] * r0];
    const d = [0, 0, 0];
    delta(bc, i1, i2, d);
    const rf: Vec3 = [d[0], d[1], d[2]];
    const params = {
      Kr: P('kr'), Ks: P('ks'), Kt: P('kt'), Kb: P('kb'),
      Fcr: P('fcr'), Fcs: P('fcs'), Tct: P('tct'), Tcb: P('tcb'),
      gr: P('gr'), gs: P('gs'), gt: P('gt'), gb: P('gb'),
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
    const smooth = this.smooth ? 1 - Math.pow(L.breaking, 4) : 1;
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
        this.breakBond(b.type, k);
        this.emitBreak(bc, ia, ib);
        continue;
      }
      const smooth = this.smooth ? 1 - Math.pow(L.breaking, 4) : 1;
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
    }
  }

  /** The average-frame elastic and derivative-damped loads (average_frame_forces). */
  private averageFrame(bc: BondedCompute, i1: number, i2: number, p: Record<string, number>, ri: Vec3, rf: Vec3, h: Hist): Loads {
    const quat = bc.s.quat!, v = bc.s.v;
    const riNorm = len3(ri);
    let Kr = p.Kr, Ks = p.Ks;
    if (this.normalize) { Kr *= 1 / riNorm; Ks *= 1 / riNorm; }
    const q1f: Quat = [quat[4 * i1], quat[4 * i1 + 1], quat[4 * i1 + 2], quat[4 * i1 + 3]];
    const q2f: Quat = [quat[4 * i2], quat[4 * i2 + 1], quat[4 * i2 + 2], quat[4 * i2 + 3]];
    const qcf = quatAverage(q1f, q2f);
    const qcfInv = qconjugate(qcf);
    const rc = quatrotvec(qcfInv, rf);
    const rcNorm = len3(rc);
    const FrElastic = Kr * (rcNorm - riNorm);
    let Fr = FrElastic;
    const riDotRc = dot3(ri, rc);
    const gamma = acosLimit(riDotRc / (riNorm * rcNorm));
    const rixrc = cross3(ri, rc);
    const rixrcMag = len3(rixrc);
    let FstC: Vec3 = [0, 0, 0];
    let TstC: Vec3 = [0, 0, 0];
    const rcHat = normalize3(rc);
    let FsElastic = 0, Fs = 0;
    let sHat: Vec3 = [0, 0, 0], tHat: Vec3 = [0, 0, 0];
    if (rixrcMag > EPSILON) {
      tHat = normalize3(rixrc);
      sHat = normalize3(cross3(tHat, rcHat));
      FsElastic = Ks * rcNorm * gamma;
      Fs = FsElastic;
    }
    const qu0 = qnormalize(quatquat(qcfInv, q2f));
    // qm: the rotation that takes the z axis onto rc (MathExtra-style minimal rotation), built from rc
    let temp = Math.max(0, rcNorm + rc[2]);
    const qm: Quat = [0, 0, 0, 0];
    qm[0] = SQRT2 * 0.5 * Math.sqrt(temp / rcNorm);
    temp = Math.sqrt(rc[0] * rc[0] + rc[1] * rc[1]);
    if (temp > EPSILON) {
      const temp2 = Math.max(0, rcNorm - rc[2]);
      const factor = (-SQRT2 * 0.5 * Math.sqrt(temp2 / rcNorm)) / temp;
      qm[1] = factor * rc[1];
      qm[2] = -factor * rc[0];
    } else {
      qm[1] = 0;
      qm[2] = 0;
    }
    qm[3] = 0;
    if (qm[0] === 0 && qm[1] === 0 && qm[2] === 0) qm[2] = 1;
    const qmN = qnormalize(qm);
    const qu = qnormalize(quatquat(quatquat(qconjugate(qmN), qu0), qmN));
    const sw2 = decomposeSwingTwist(qu);
    const sw1 = decomposeSwingTwist(qconjugate(qu));
    const twist = quatquat(sw2.twist, qconjugate(sw1.twist));
    const psi = 2 * acosLimit(twist[0]);
    const taxis: Vec3 = Math.abs(psi) > EPSILON ? normalize3([twist[1], twist[2], twist[3]]) : [0, 0, 1];
    const swing = quatquat(sw2.swing, qconjugate(sw1.swing));
    const theta = 2 * acosLimit(swing[0]);
    const baxis: Vec3 = Math.abs(theta) > EPSILON ? normalize3([swing[1], swing[2], swing[3]]) : [0, 0, 1];
    const TtElastic = p.Kt * psi;
    let Tt = TtElastic;
    const TbElastic = p.Kb * theta;
    let Tb = TbElastic;
    if (this.damping === 'derivative') {
      const dtInv = 1 / bc.s.dt;
      const dvc = quatrotvec(qcfInv, sub3([v[3 * i2], v[3 * i2 + 1], v[3 * i2 + 2]], [v[3 * i1], v[3 * i1 + 1], v[3 * i1 + 2]]));
      Fr += p.gr * dot3(dvc, rcHat);
      if (rixrcMag > EPSILON) Fs += p.gs * rcNorm * (wrap2pi(gamma - h[4]) * dtInv);
      Tt += p.gt * (wrap2pi(psi - h[6]) * dtInv);
      Tb += p.gb * (wrap2pi(theta - h[5]) * dtInv);
      h[4] = gamma;
      h[5] = theta;
      h[6] = psi;
    }
    const FrC = scale3(Fr, rcHat);
    if (rixrcMag > EPSILON) {
      FstC = scale3(Fs, sHat);
      TstC = scale3(0.5 * rcNorm * Fs, tHat);
    }
    const TtP = scale3(Tt, taxis);
    const TbP = scale3(Tb, baxis);
    const TbC = quatrotvec(qmN, TbP);
    const TtC = quatrotvec(qmN, TtP);
    const F1C = add3(FrC, FstC);
    const Ttmp = add3(TbC, TtC);
    const T1C = add3(TstC, Ttmp);
    const T2C = sub3(TstC, Ttmp);
    const force2on1 = quatrotvec(qcf, F1C);
    const breaking0 = Math.abs(FrElastic) / p.Fcr + FsElastic / p.Fcs + TbElastic / p.Tcb + TtElastic / p.Tct;
    const ebond = 0.5 * (Kr * (rcNorm - riNorm) * (rcNorm - riNorm) + Ks * rcNorm * rcNorm * gamma * gamma + p.Kt * psi * psi + p.Kb * theta * theta);
    return {
      f1on2: scale3(-1, force2on1),
      t1on2: quatrotvec(qcf, T2C),
      t2on1: quatrotvec(qcf, T1C),
      breaking: Math.max(0, breaking0),
      ebond,
    };
  }

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

