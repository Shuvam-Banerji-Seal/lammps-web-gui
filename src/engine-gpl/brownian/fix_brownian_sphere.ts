/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_brownian_sphere.cpp and
 * fix_brownian_sphere.h (class FixBrownianSphere, fix brownian/sphere: Brownian translation and rotation of
 * spheres, with the dipole orientation rotated by the rotational noise and the torque). Original authors: Sam
 * Cameron (University of Bristol) and Arthur Straube (Zuse Institute Berlin). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { FixBrownianBase } from './fix_brownian_base';
import { cross3, dot3, len3, scale3, scaleadd3, scaleadd3Two, snormalize3 } from './mathextra';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

export class FixBrownianSphere extends FixBrownianBase {
  private g3 = 0;
  private g4 = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'brownian/sphere');
    if (this.gammaTEigenFlag || this.gammaREigenFlag)
      throw new StyleError('Keywords gamma_t_eigen or gamma_r_eigen are not compatible with fix brownian/sphere');
    if (!this.gammaTFlag || !this.gammaRFlag) throw new StyleError('Keywords gamma_t and gamma_r are required for fix brownian/sphere');
    if (!sys.state.mu) throw new StyleError('Fix brownian/sphere requires atom attribute mu');
  }

  /** FixBrownianSphere::init(): the rotational prefactors g3 (torque) and g4 (noise) come before the translational ones. */
  override init(): void {
    super.init();
    this.g3 = this.g1 / this.gammaR;
    this.g4 = this.g2 * Math.sqrt(this.rotTemp / this.gammaR);
    this.g1 /= this.gammaT;
    this.g2 *= Math.sqrt(this.temp / this.gammaT);
  }

  /**
   * FixBrownianSphere::initial_integrate_templated: translation (x, v), then the angular velocity
   * w = noise + g3 torque rotates the dipole mu (geometric: exact rotation about the axis; projection: first order).
   */
  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask, torque } = s;
    const mu = s.mu!;
    const bit = this.groupBit;
    const dt = this.dt;
    const dim2 = this.sys.dimension === 2;
    const planar = this.planarRotFlag;
    const fullRot = !dim2 && !planar;
    const geometric = this.rotStyle === 'geometric';
    const noisy = this.noisy;
    const u = [0, 0, 0];
    const w = [0, 0, 0];
    const wperp = [0, 0, 0];
    const n = [0, 0, 0];
    const uxn = [0, 0, 0];
    const wxu = [0, 0, 0];
    const muNew = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      let dx: number;
      let dy: number;
      let dz = 0;
      let wx = 0;
      let wy = 0;
      let wz = 0;
      if (noisy) {
        dx = dt * (this.g1 * f[3 * i] + this.g2 * this.noise());
        dy = dt * (this.g1 * f[3 * i + 1] + this.g2 * this.noise());
        if (!dim2) dz = dt * (this.g1 * f[3 * i + 2] + this.g2 * this.noise());
        if (fullRot) {
          wx = this.noise() * this.g4;
          wy = this.noise() * this.g4;
          wz = this.noise() * this.g4;
        } else wz = this.noise() * this.g4;
      } else {
        dx = dt * this.g1 * f[3 * i];
        dy = dt * this.g1 * f[3 * i + 1];
        if (!dim2) dz = dt * this.g1 * f[3 * i + 2];
      }
      x[3 * i] += dx;
      v[3 * i] = dx / dt;
      x[3 * i + 1] += dy;
      v[3 * i + 1] = dy / dt;
      x[3 * i + 2] += dz;
      v[3 * i + 2] = dz / dt;
      w[0] = wx + this.g3 * torque![3 * i];
      w[1] = wy + this.g3 * torque![3 * i + 1];
      w[2] = wz + this.g3 * torque![3 * i + 2];
      const m0 = 4 * i;
      const mulen = len3(mu.subarray(m0, m0 + 3));
      if (!(mulen > 0)) throw new StyleError('Fix brownian/sphere dipole moment must have positive length');
      scale3(1 / mulen, mu.subarray(m0, m0 + 3), u);
      muNew[0] = mu[m0];
      muNew[1] = mu[m0 + 1];
      muNew[2] = mu[m0 + 2];
      if (geometric) {
        if (dim2 || planar) {
          const theta = dt * w[2];
          const c = libmCos(theta);
          const sn = libmSin(theta);
          const ux = u[0];
          const uy = u[1];
          u[0] = c * ux - sn * uy;
          u[1] = sn * ux + c * uy;
          snormalize3(mulen, u, muNew);
        } else {
          const dotWu = dot3(w, u);
          scaleadd3(-dotWu, u, w, wperp);
          const wperplen = len3(wperp);
          if (wperplen > 0) {
            scale3(1 / wperplen, wperp, n);
            cross3(u, n, uxn);
            const theta = dt * wperplen;
            const c = libmCos(theta);
            const sn = libmSin(theta);
            scaleadd3Two(c, u, -sn, uxn, u);
            snormalize3(mulen, u, muNew);
          }
        }
      } else {
        cross3(w, u, wxu);
        scaleadd3(dt, wxu, u, u);
        snormalize3(mulen, u, muNew);
      }
      mu[m0] = muNew[0];
      mu[m0 + 1] = muNew[1];
      mu[m0 + 2] = muNew[2];
    }
  }
}
