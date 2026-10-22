/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/fix_sph.cpp and fix_sph.h (FixSPH:
 * fix sph). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version
 * 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg
 * Ganzenmuller (Fraunhofer EMI).
 *
 * Velocity-Verlet time integration of SPH particles (docs.lammps.org/fix_sph.html): positions and velocities advance as
 * in fix nve; the internal energy esph and the density rho advance with their rates desph and drho (a half step at
 * each end of the step); the velocity estimate vest = v + 2 dtfm F is set for the pair styles of the force evaluation.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf } from '../../engine/atoms';
import { clearSphRates, dropSphGhostVest, refreshSphGhostVest, requireSph, sphArrays } from './common';

export class FixSPH extends Fix {
  readonly style: string = 'sph';
  protected dtv = 0;
  protected dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_sph.cpp constructor: the atom attributes, then the number of words
    requireSph(sys.state, 'Fix sph requires atom attributes energy, density, and velocity estimates, e.g. in atom_style sph');
    if (args.length) throw new StyleError('Illegal number of arguments for fix sph command');
    this.timeIntegrate = true;
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  /** fix_sph.cpp setup_pre_force: the velocity estimate of the group is the velocity. */
  setupPreForce(): void {
    const s = this.sys.state;
    const { vest } = sphArrays(s, 'Fix sph');
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      vest[3 * i] = s.v[3 * i];
      vest[3 * i + 1] = s.v[3 * i + 1];
      vest[3 * i + 2] = s.v[3 * i + 2];
    }
    // fix_sph.cpp setup_pre_force: forward_comm of every atom's vest
    refreshSphGhostVest(s, vest);
    clearSphRates(s);
  }

  /** The rates are cleared before each force evaluation (force_clear); the step's pair styles then sum into them. */
  preForce(): void { clearSphRates(this.sys.state); }

  /** fix_sph.cpp initial_integrate: half steps of esph and rho, the velocity estimate, then v and x. */
  initialIntegrate(): void {
    const s = this.sys.state;
    dropSphGhostVest(s);
    const a = sphArrays(s, 'Fix sph');
    const { x, v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      a.esph[i] += dtf * a.desph[i];
      a.rho[i] += dtf * a.drho[i];
      for (let c = 0; c < 3; c++) a.vest[3 * i + c] = v[3 * i + c] + 2.0 * dtfm * f[3 * i + c];
      for (let c = 0; c < 3; c++) v[3 * i + c] += dtfm * f[3 * i + c];
      for (let c = 0; c < 3; c++) x[3 * i + c] += dtv * v[3 * i + c];
    }
  }

  /** fix_sph.cpp final_integrate: the second half kick of v, and of esph and rho. */
  finalIntegrate(): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Fix sph');
    const { v, f, mask } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const dtfm = dtf / massOf(s, i);
      for (let c = 0; c < 3; c++) v[3 * i + c] += dtfm * f[3 * i + c];
      a.esph[i] += dtf * a.desph[i];
      a.rho[i] += dtf * a.drho[i];
    }
  }
}
