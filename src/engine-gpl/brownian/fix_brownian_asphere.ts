/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_brownian_asphere.cpp and
 * fix_brownian_asphere.h (class FixBrownianAsphere, fix brownian/asphere: Brownian translation and rotation of
 * ellipsoids with anisotropic damping gamma_t_eigen and gamma_r_eigen, the orientation quaternion advanced with
 * the rotational velocity, and an optional body-fixed dipole). Original authors: Sam Cameron (University of
 * Bristol) and Arthur Straube (Zuse Institute Berlin). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { isEllipsoid } from '../../engine/atoms';
import { FixBrownianBase } from './fix_brownian_base';
import { matvec, qnormalize, quatToMat, quatToMatTrans, quatvec, transposeMatvec } from './mathextra';

export class FixBrownianAsphere extends FixBrownianBase {
  private g4 = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'brownian/asphere');
    if (this.gammaTFlag || this.gammaRFlag) throw new StyleError('Keywords gamma_t or gamma_r are not compatible with fix brownian/asphere');
    if (!this.gammaTEigenFlag || !this.gammaREigenFlag)
      throw new StyleError('Keywords gamma_t_eigen and gamma_r_eigen are required for fix brownian/asphere');
    if (this.dipoleFlag && !sys.state.mu) throw new StyleError('Fix brownian/asphere dipole requires atom attribute mu');
    if (!sys.state.quat || !sys.state.shape) throw new StyleError('Fix brownian/asphere requires atom style ellipsoid');
  }

  /** FixBrownianAsphere::init(): the dipole is set from the body frame, then the prefactors g4 (noise) and g2. */
  override init(): void {
    const s = this.sys.state;
    if (!s.quat) throw new StyleError('Fix brownian/asphere requires atom style ellipsoid');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !isEllipsoid(s, i)) throw new StyleError('Fix brownian/asphere requires extended particles');
    }
    if (this.dipoleFlag) {
      const q = [0, 0, 0, 0];
      const Q = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ];
      const fRot = [0, 0, 0];
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        for (let k = 0; k < 4; k++) q[k] = s.quat[4 * i + k];
        quatToMat(q, Q);
        matvec(Q, this.dipoleBody, fRot);
        for (let k = 0; k < 3; k++) s.mu![4 * i + k] = fRot[k];
      }
    }
    super.init();
    this.g4 = this.g2 * Math.sqrt(this.rotTemp);
    this.g2 *= Math.sqrt(this.temp);
  }

  /**
   * FixBrownianAsphere::initial_integrate_templated: the body-frame angular velocity (gamma_r_eigen) advances the
   * quaternion, the body-frame force (gamma_t_eigen) gives the velocity, and the dipole follows the quaternion.
   */
  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask, torque, quat } = s;
    const bit = this.groupBit;
    const dt = this.dt;
    const dim2 = this.sys.dimension === 2;
    const rot2D = dim2 || this.planarRotFlag;
    const noisy = this.noisy;
    const q = [0, 0, 0, 0];
    const qw = [0, 0, 0, 0];
    const RT = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
    ];
    const tmp = [0, 0, 0];
    const dv = [0, 0, 0];
    const fRot = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      for (let k = 0; k < 4; k++) q[k] = quat![4 * i + k];
      quatToMatTrans(q, RT);
      matvec(RT, [torque![3 * i], torque![3 * i + 1], torque![3 * i + 2]], tmp);
      const gri = this.gammaRInv;
      const grs = this.gammaRInvsqrt;
      const g1 = this.g1;
      const g4 = this.g4;
      if (rot2D) {
        tmp[0] = 0.0;
        tmp[1] = 0.0;
        tmp[2] = noisy ? g1 * tmp[2] * gri[2] + grs[2] * this.noise() * g4 : g1 * tmp[2] * gri[2];
      } else {
        for (let k = 0; k < 3; k++) tmp[k] = noisy ? g1 * tmp[k] * gri[k] + grs[k] * this.noise() * g4 : g1 * tmp[k] * gri[k];
      }
      quatvec(q, tmp, qw);
      for (let k = 0; k < 4; k++) q[k] = q[k] + 0.5 * dt * qw[k];
      qnormalize(q);
      for (let k = 0; k < 4; k++) quat![4 * i + k] = q[k];

      matvec(RT, [f[3 * i], f[3 * i + 1], f[3 * i + 2]], tmp);
      const gti = this.gammaTInv;
      const gts = this.gammaTInvsqrt;
      const g2 = this.g2;
      if (dim2) {
        tmp[2] = 0.0;
        for (let k = 0; k < 2; k++) tmp[k] = noisy ? g1 * tmp[k] * gti[k] + gts[k] * this.noise() * g2 : g1 * tmp[k] * gti[k];
      } else {
        for (let k = 0; k < 3; k++) tmp[k] = noisy ? g1 * tmp[k] * gti[k] + gts[k] * this.noise() * g2 : g1 * tmp[k] * gti[k];
      }
      transposeMatvec(RT, tmp, dv);
      v[3 * i] = dv[0];
      v[3 * i + 1] = dv[1];
      v[3 * i + 2] = dv[2];
      x[3 * i] += dv[0] * dt;
      x[3 * i + 1] += dv[1] * dt;
      x[3 * i + 2] += dv[2] * dt;
      if (this.dipoleFlag) {
        quatToMatTrans(q, RT);
        transposeMatvec(RT, this.dipoleBody, fRot);
        for (let k = 0; k < 3; k++) s.mu![4 * i + k] = fRot[k];
      }
    }
  }
}
