/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/fix_nve_dotc_langevin.cpp and
 * fix_nve_dotc_langevin.h (class FixNVEDotcLangevin, fix nve/dotc/langevin): the nve/dot integration of
 * ellipsoidal oxDNA nucleotides with a Langevin thermostat on the translation (damping gamma = 1/t_period, noise
 * from a Marsaglia RanMars stream seeded with the seed of the command) and on the rotation (damping
 * Gamma = gamma * angmom factor, or none with "angmom no"). Original author (from the header): Oliver Henrich
 * (University of Strathclyde, Glasgow). LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { FixNVE } from '../../engine/fix/nve';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isEllipsoid, massOf } from '../../engine/atoms';
import { parseInt_, parseNum } from '../../engine/force/util';
import { RanMars } from '../random_mars';
import { noSquishRotate, vec3ToVec4, vec4ToVec3 } from './fix_nve_dot';
import { exp as libmExp } from '../libm/exp';

/** fix_nve_dotc_langevin.cpp: "static constexpr double INERTIA = 0.2;" */
const INERTIA = 0.2;

export class FixNVEDotcLangevin extends FixNVE {
  override readonly style: string = 'nve/dotc/langevin';
  private tStart = 0;
  private tStop = 0;
  private tTarget = 0;
  private tsqrt = 0;
  private gamma = 0;
  private Gamma = 0;
  private gfactor1 = 0;
  private readonly random: RanMars;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_nve_dotc_langevin.cpp: "if (narg != 9) error->all(FLERR,"Illegal fix nve/dotc/langevin command");"
    if (args.length !== 6) throw new StyleError('Illegal fix nve/dotc/langevin command');
    if (args[4] !== 'angmom') throw new StyleError('Illegal fix nve/dotc/langevin command');
    this.tStart = parseNum(args[0], 'Tstart');
    this.tTarget = this.tStart;
    this.tStop = parseNum(args[1], 'Tstop');
    const period = parseNum(args[2], 'period');
    if (period <= 0.0) throw new StyleError('Fix nve/dotc/langevin period must be > 0.0');
    this.gamma = 1.0 / period;
    const seed = parseInt_(args[3], 'seed');
    if (seed <= 0) throw new StyleError('Illegal fix nve/dotc/langevin command');
    // "angmom no" zeroes the rotational damping; "angmom value" gives Gamma = gamma value
    if (args[5] === 'no') this.Gamma = 0.0;
    else this.Gamma = this.gamma * parseNum(args[5], 'angmom');
    this.random = new RanMars(seed);
  }

  /** fix_nve_dotc_langevin.cpp compute_target(): the linear ramp of the target temperature over the run. */
  private computeTarget(): void {
    const s = this.sys.state;
    const run = this.sys.run;
    let delta = s.step - run.beginStep;
    if (delta !== 0.0) delta /= run.endStep - run.beginStep;
    this.tTarget = this.tStart + delta * (this.tStop - this.tStart);
    this.tsqrt = Math.sqrt(this.tTarget);
  }

  override init(): void {
    const s = this.sys.state;
    if (!s.quat || !s.shape || !s.torque || !s.angmom) throw new StyleError('Fix nve/dotc/langevin requires atom style ellipsoid');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix nve/dotc/langevin requires extended particles');
    }
    this.gfactor1 = libmExp(-this.gamma * s.dt);
    this.computeTarget();
    super.init();
  }

  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const quat = s.quat!, shape = s.shape!, torque = s.torque!, angmom = s.angmom!;
    const bit = this.groupBit;
    const dt = s.dt;
    const dthlf = 0.5 * dt;
    const dtqrt = 0.25 * dt;
    this.computeTarget();
    const conj = [0, 0, 0, 0], fquat = [0, 0, 0, 0], l3 = [0, 0, 0], slq = [0, 0, 0], g5 = [0, 0, 0];
    const inertia = [0, 0, 0], gf3 = [0, 0, 0], gf4 = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const m0 = massOf(s, i);
      const dthlfm = dthlf / m0;
      const qo = 4 * i;
      v[3 * i] += dthlfm * f[3 * i]; v[3 * i + 1] += dthlfm * f[3 * i + 1]; v[3 * i + 2] += dthlfm * f[3 * i + 2];
      x[3 * i] += dthlf * v[3 * i]; x[3 * i + 1] += dthlf * v[3 * i + 1]; x[3 * i + 2] += dthlf * v[3 * i + 2];
      vec3ToVec4(quat, qo, [angmom[3 * i], angmom[3 * i + 1], angmom[3 * i + 2]], conj);
      for (let m = 0; m < 4; m++) conj[m] *= 2.0;
      vec3ToVec4(quat, qo, [torque[3 * i], torque[3 * i + 1], torque[3 * i + 2]], fquat);
      for (let m = 0; m < 4; m++) conj[m] += dt * fquat[m];
      const sh0 = shape[3 * i], sh1 = shape[3 * i + 1], sh2 = shape[3 * i + 2];
      inertia[0] = INERTIA * m0 * (sh1 * sh1 + sh2 * sh2);
      inertia[1] = INERTIA * m0 * (sh0 * sh0 + sh2 * sh2);
      inertia[2] = INERTIA * m0 * (sh0 * sh0 + sh1 * sh1);
      const M = (inertia[0] * inertia[1] * inertia[2]) / (inertia[1] * inertia[2] + inertia[0] * inertia[2] + inertia[0] * inertia[1]);
      const gfactor2 = Math.sqrt(12.0 * (1.0 - this.gfactor1 * this.gfactor1) / m0) * this.tsqrt;
      for (let c = 0; c < 3; c++) {
        gf3[c] = libmExp(-this.Gamma * M * dt / inertia[c]);
        gf4[c] = Math.sqrt(48.0 * inertia[c] * (1.0 - gf3[c] * gf3[c])) * this.tsqrt;
      }
      noSquishRotate(3, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(2, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(1, conj, quat, qo, inertia, dthlf);
      noSquishRotate(2, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(3, conj, quat, qo, inertia, dtqrt);
      // translational Langevin kick: v = v gfactor1 + gfactor2 (u - 1/2), one draw per component
      for (let c = 0; c < 3; c++) v[3 * i + c] = v[3 * i + c] * this.gfactor1 + gfactor2 * (this.random.uniform() - 0.5);
      for (let c = 0; c < 3; c++) x[3 * i + c] += dthlf * v[3 * i + c];
      // rotational Langevin kick in the body frame
      const q0 = quat[qo], q1 = quat[qo + 1], q2 = quat[qo + 2], q3 = quat[qo + 3];
      slq[0] = -q1 * conj[0] + q0 * conj[1] + q3 * conj[2] - q2 * conj[3];
      slq[1] = -q2 * conj[0] - q3 * conj[1] + q0 * conj[2] + q1 * conj[3];
      slq[2] = -q3 * conj[0] + q2 * conj[1] - q1 * conj[2] + q0 * conj[3];
      for (let c = 0; c < 3; c++) g5[c] = gf3[c] * slq[c] + gf4[c] * (this.random.uniform() - 0.5);
      conj[0] = -q1 * g5[0] - q2 * g5[1] - q3 * g5[2];
      conj[1] = q0 * g5[0] - q3 * g5[1] + q2 * g5[2];
      conj[2] = q3 * g5[0] + q0 * g5[1] - q1 * g5[2];
      conj[3] = -q2 * g5[0] + q1 * g5[1] + q0 * g5[2];
      noSquishRotate(3, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(2, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(1, conj, quat, qo, inertia, dthlf);
      noSquishRotate(2, conj, quat, qo, inertia, dtqrt);
      noSquishRotate(3, conj, quat, qo, inertia, dtqrt);
      const n = Math.sqrt(quat[qo] * quat[qo] + quat[qo + 1] * quat[qo + 1] + quat[qo + 2] * quat[qo + 2] + quat[qo + 3] * quat[qo + 3]);
      for (let m = 0; m < 4; m++) quat[qo + m] /= n;
      vec4ToVec3(quat, qo, conj, l3);
      for (let c = 0; c < 3; c++) angmom[3 * i + c] = 0.5 * l3[c];
    }
  }

  /** The final half-kick is that of nve/dot (fix_nve_dotc_langevin.cpp final_integrate). */
  override finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask } = s;
    const quat = s.quat!, torque = s.torque!, angmom = s.angmom!;
    const bit = this.groupBit;
    const dt = s.dt;
    const conj = [0, 0, 0, 0], fquat = [0, 0, 0, 0], l3 = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dthlfm = (0.5 * dt) / massOf(s, i);
      v[3 * i] += dthlfm * f[3 * i]; v[3 * i + 1] += dthlfm * f[3 * i + 1]; v[3 * i + 2] += dthlfm * f[3 * i + 2];
      const qo = 4 * i;
      vec3ToVec4(quat, qo, [angmom[3 * i], angmom[3 * i + 1], angmom[3 * i + 2]], conj);
      for (let m = 0; m < 4; m++) conj[m] *= 2.0;
      vec3ToVec4(quat, qo, [torque[3 * i], torque[3 * i + 1], torque[3 * i + 2]], fquat);
      for (let m = 0; m < 4; m++) conj[m] += dt * fquat[m];
      const dot = conj[0] * quat[qo] + conj[1] * quat[qo + 1] + conj[2] * quat[qo + 2] + conj[3] * quat[qo + 3];
      for (let m = 0; m < 4; m++) conj[m] -= dot * quat[qo + m];
      vec4ToVec3(quat, qo, conj, l3);
      for (let c = 0; c < 3; c++) angmom[3 * i + c] = 0.5 * l3[c];
    }
  }
}
