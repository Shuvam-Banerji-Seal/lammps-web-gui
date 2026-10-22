/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/fix_nve_dot.cpp and
 * fix_nve_dot.h (class FixNVEDot, fix nve/dot): constant-NVE integration of the translation and of the
 * orientation of ellipsoidal oxDNA nucleotides, with the no-squish rotation (no_squish_rotate of
 * src/math_extra.cpp) and the moment of inertia prefactor 0.2 m (b^2 + c^2) of an ellipsoid. The vector
 * conversions vec3_to_vec4 and vec4_to_vec3 are those of fix_nve_dot.h. Original author (from the header):
 * Oliver Henrich (University of Strathclyde, Glasgow). LAMMPS is Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import { FixNVE } from '../../engine/fix/nve';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isEllipsoid, massOf } from '../../engine/atoms';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** fix_nve_dot.cpp: "static constexpr double INERTIA = 0.2;  // moment of inertia prefactor for ellipsoid" */
const INERTIA = 0.2;

/** fix_nve_dot.h vec3_to_vec4: 4-vector of a space-frame 3-vector with the quaternion q. */
export const vec3ToVec4 = (q: Float64Array, qo: number, v3: readonly number[], out: number[]): void => {
  const q0 = q[qo], q1 = q[qo + 1], q2 = q[qo + 2], q3 = q[qo + 3];
  out[0] = -q1 * v3[0] - q2 * v3[1] - q3 * v3[2];
  out[1] = q0 * v3[0] + q3 * v3[1] - q2 * v3[2];
  out[2] = -q3 * v3[0] + q0 * v3[1] + q1 * v3[2];
  out[3] = q2 * v3[0] - q1 * v3[1] + q0 * v3[2];
};

/** fix_nve_dot.h vec4_to_vec3: the 3-vector of a 4-vector with the quaternion q. */
export const vec4ToVec3 = (q: Float64Array, qo: number, v4: readonly number[], out: number[]): void => {
  const q0 = q[qo], q1 = q[qo + 1], q2 = q[qo + 2], q3 = q[qo + 3];
  out[0] = -q1 * v4[0] + q0 * v4[1] - q3 * v4[2] + q2 * v4[3];
  out[1] = -q2 * v4[0] + q3 * v4[1] + q0 * v4[2] - q1 * v4[3];
  out[2] = -q3 * v4[0] - q2 * v4[1] + q1 * v4[2] + q0 * v4[3];
};

/**
 * math_extra.cpp no_squish_rotate(k, p, q, inertia, dt): rotation of the quaternion q and its conjugate momentum
 * p about body axis k (1, 2 or 3) for the time dt, in place.
 */
export const noSquishRotate = (k: number, p: number[], q: Float64Array, qo: number, inertia: readonly number[], dt: number): void => {
  const kq = [0, 0, 0, 0], kp = [0, 0, 0, 0];
  const q0 = q[qo], q1 = q[qo + 1], q2 = q[qo + 2], q3 = q[qo + 3];
  if (k === 1) {
    kq[0] = -q1; kp[0] = -p[1];
    kq[1] = q0; kp[1] = p[0];
    kq[2] = q3; kp[2] = p[3];
    kq[3] = -q2; kp[3] = -p[2];
  } else if (k === 2) {
    kq[0] = -q2; kp[0] = -p[2];
    kq[1] = -q3; kp[1] = -p[3];
    kq[2] = q0; kp[2] = p[0];
    kq[3] = q1; kp[3] = p[1];
  } else {
    kq[0] = -q3; kp[0] = -p[3];
    kq[1] = q2; kp[1] = p[2];
    kq[2] = -q1; kp[2] = -p[1];
    kq[3] = q0; kp[3] = p[0];
  }
  let phi = p[0] * kq[0] + p[1] * kq[1] + p[2] * kq[2] + p[3] * kq[3];
  if (inertia[k - 1] === 0.0) phi = 0.0;
  else phi /= 4.0 * inertia[k - 1];
  const c = libmCos(dt * phi), sn = libmSin(dt * phi);
  for (let m = 0; m < 4; m++) p[m] = c * p[m] + sn * kp[m];
  for (let m = 0; m < 4; m++) q[qo + m] = c * q[qo + m] + sn * kq[m];
};

export class FixNVEDot extends FixNVE {
  override readonly style: string = 'nve/dot';

  constructor(sys: System, id: string, group: string, args: string[]) {
    // fix_nve_dot.cpp (a FixNVE): further words are not read
    super(sys, id, group, args);
  }

  override init(): void {
    const s = this.sys.state;
    if (!s.quat || !s.shape || !s.torque || !s.angmom) throw new StyleError('Compute nve/dot requires atom style ellipsoid');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix nve/dot requires extended particles');
    }
    super.init();
  }

  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const quat = s.quat!, shape = s.shape!, torque = s.torque!, angmom = s.angmom!;
    const bit = this.groupBit;
    const dt = s.dt;
    const dthlf = 0.5 * dt;
    const conj = [0, 0, 0, 0], fquat = [0, 0, 0, 0], l3 = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dthlfm = dthlf / massOf(s, i);
      v[3 * i] += dthlfm * f[3 * i]; v[3 * i + 1] += dthlfm * f[3 * i + 1]; v[3 * i + 2] += dthlfm * f[3 * i + 2];
      x[3 * i] += dt * v[3 * i]; x[3 * i + 1] += dt * v[3 * i + 1]; x[3 * i + 2] += dt * v[3 * i + 2];
      const qo = 4 * i;
      vec3ToVec4(quat, qo, [angmom[3 * i], angmom[3 * i + 1], angmom[3 * i + 2]], conj);
      for (let m = 0; m < 4; m++) conj[m] *= 2.0;
      vec3ToVec4(quat, qo, [torque[3 * i], torque[3 * i + 1], torque[3 * i + 2]], fquat);
      for (let m = 0; m < 4; m++) conj[m] += dt * fquat[m];
      const m0 = massOf(s, i);
      const sh0 = shape[3 * i], sh1 = shape[3 * i + 1], sh2 = shape[3 * i + 2];
      const inertia = [INERTIA * m0 * (sh1 * sh1 + sh2 * sh2), INERTIA * m0 * (sh0 * sh0 + sh2 * sh2), INERTIA * m0 * (sh0 * sh0 + sh1 * sh1)];
      noSquishRotate(3, conj, quat, qo, inertia, dthlf);
      noSquishRotate(2, conj, quat, qo, inertia, dthlf);
      noSquishRotate(1, conj, quat, qo, inertia, dt);
      noSquishRotate(2, conj, quat, qo, inertia, dthlf);
      noSquishRotate(3, conj, quat, qo, inertia, dthlf);
      const n = Math.sqrt(quat[qo] * quat[qo] + quat[qo + 1] * quat[qo + 1] + quat[qo + 2] * quat[qo + 2] + quat[qo + 3] * quat[qo + 3]);
      for (let m = 0; m < 4; m++) quat[qo + m] /= n;
      vec4ToVec3(quat, qo, conj, l3);
      for (let c = 0; c < 3; c++) angmom[3 * i + c] = 0.5 * l3[c];
    }
  }

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
