/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BROWNIAN/fix_brownian.cpp and
 * fix_brownian.h (class FixBrownian, fix brownian: overdamped Brownian translation of point particles with
 * isotropic damping gamma_t). Original authors: Sam Cameron (University of Bristol) and Arthur Straube
 * (Zuse Institute Berlin). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';
import { FixBrownianBase } from './fix_brownian_base';

export class FixBrownian extends FixBrownianBase {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, 'brownian');
    if (this.dipoleFlag || this.gammaTEigenFlag || this.gammaREigenFlag || this.gammaRFlag || this.rotTempFlag || this.planarRotFlag)
      throw new StyleError('Using keyword(s) not compatible with fix brownian');
    if (!this.gammaTFlag) throw new StyleError('Keyword gamma_t is required for fix brownian');
  }

  /** FixBrownian::init(): the mobility 1/gamma_t scales the force, sqrt(T / gamma_t) the noise. */
  override init(): void {
    super.init();
    this.g1 /= this.gammaT;
    this.g2 *= Math.sqrt(this.temp / this.gammaT);
  }

  /** FixBrownian::initial_integrate: x += dt (g1 f + g2 noise), v = dx / dt; z is untouched in 2d. */
  override initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const dt = this.dt;
    const dim2 = this.sys.dimension === 2;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      let dx: number;
      let dy: number;
      let dz = 0;
      if (this.noisy) {
        dx = dt * (this.g1 * f[3 * i] + this.g2 * this.noise());
        dy = dt * (this.g1 * f[3 * i + 1] + this.g2 * this.noise());
        if (!dim2) dz = dt * (this.g1 * f[3 * i + 2] + this.g2 * this.noise());
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
    }
  }
}
