/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/fix_sph_stationary.cpp and
 * fix_sph_stationary.h (FixSPHStationary: fix sph/stationary). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Time integration of the internal energy and the density only (docs.lammps.org/fix_sph_stationary.html): the atoms of
 * the group do not move, so only esph and rho advance with their rates, half a step at each end of the step.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { clearSphRates, requireSph, sphArrays } from './common';

export class FixSPHStationary extends Fix {
  readonly style: string = 'sph/stationary';
  private dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_sph_stationary.cpp constructor: the atom attributes, then the number of words
    requireSph(sys.state, 'Fix sph/stationary requires atom attributes energy and density, e.g. in atom_style sph');
    if (args.length) throw new StyleError('Illegal number of arguments for fix sph/stationary command');
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  setupPreForce(): void { clearSphRates(this.sys.state); }

  preForce(): void { clearSphRates(this.sys.state); }

  initialIntegrate(): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Fix sph/stationary');
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      a.esph[i] += this.dtf * a.desph[i];
      a.rho[i] += this.dtf * a.drho[i];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Fix sph/stationary');
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      a.esph[i] += this.dtf * a.desph[i];
      a.rho[i] += this.dtf * a.drho[i];
    }
  }
}
