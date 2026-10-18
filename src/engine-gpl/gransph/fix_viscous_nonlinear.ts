/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRANULAR/fix_viscous_nonlinear.cpp
 * and fix_viscous_nonlinear.h (FixViscousNonlinear: fix viscous/nonlinear). LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Syntax (from the source): "fix ID group-ID viscous/nonlinear rho_f mu_f [velocity vx vy vz]". The drag on a finite-size
 * sphere relative to a uniform fluid velocity is F = -1/2 C_d rho_f (pi r^2) |v_rel| v_rel with Re = rho_f |v_rel| 2r / mu_f
 * and C_d = 24/Re (1 + 0.15 Re^0.687). The fix needs atom attribute radius. Its post_force applies the force in the
 * same loop order as the source.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { pow } from '../libm/pow';

/** MY_PI of the LAMMPS math constants. */
const MY_PI = 3.14159265358979323846;

export class FixViscousNonlinear extends Fix {
  readonly style = 'viscous/nonlinear';
  private readonly rhoFluid: number;
  private readonly muFluid: number;
  private readonly vFluid = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 2) throw new StyleError('Illegal fix viscous/nonlinear command');
    this.rhoFluid = parseNum(args[0], 'fluid density');
    this.muFluid = parseNum(args[1], 'fluid viscosity');
    if (this.rhoFluid <= 0.0 || this.muFluid <= 0.0) {
      throw new StyleError('Fix viscous/nonlinear fluid density and viscosity must be > 0');
    }
    for (let k = 2; k < args.length; ) {
      if (args[k] === 'velocity') {
        if (k + 4 > args.length) throw new StyleError('Illegal fix viscous/nonlinear velocity command: missing argument(s)');
        this.vFluid[0] = parseNum(args[k + 1], 'velocity');
        this.vFluid[1] = parseNum(args[k + 2], 'velocity');
        this.vFluid[2] = parseNum(args[k + 3], 'velocity');
        k += 4;
      } else {
        throw new StyleError('Illegal fix viscous/nonlinear command');
      }
    }
  }

  init(): void {
    if (!this.sys.state.radius) throw new StyleError('Fix viscous/nonlinear requires atom attribute radius');
  }

  postForce(): void {
    const s = this.sys.state;
    const v = s.v, f = s.f, radius = s.radius;
    if (!radius) throw new StyleError('Fix viscous/nonlinear requires atom attribute radius');
    const vrel = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      vrel[0] = v[3 * i] - this.vFluid[0];
      vrel[1] = v[3 * i + 1] - this.vFluid[1];
      vrel[2] = v[3 * i + 2] - this.vFluid[2];
      const vmag = Math.sqrt(vrel[0] * vrel[0] + vrel[1] * vrel[1] + vrel[2] * vrel[2]);
      if (vmag === 0.0) continue;
      const r = radius[i];
      const re = this.rhoFluid * vmag * (2.0 * r) / this.muFluid;
      const cd = (24.0 / re) * (1.0 + 0.15 * pow(re, 0.687));
      // F = -1/2 Cd rho_g (pi r^2) |v_rel| v_rel
      const pref = 0.5 * cd * this.rhoFluid * MY_PI * r * r * vmag;
      f[3 * i] -= pref * vrel[0];
      f[3 * i + 1] -= pref * vrel[1];
      f[3 * i + 2] -= pref * vrel[2];
    }
  }

  minPostForce(): void {
    this.postForce();
  }
}
