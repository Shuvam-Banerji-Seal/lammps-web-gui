/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/fix_langevin_spin.cpp and
 * fix_langevin_spin.h (FixLangevinSpin: fix langevin/spin, the random torque for a temperature above zero), with
 * the RanMars stream of src/random_mars.cpp (engine-gpl/random_mars.ts). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing authors (from fix_langevin_spin.cpp): Julien Tranchida (SNL) and Stan Moore
 * (SNL).
 *
 * fix ID group-ID langevin/spin T Tdamp seed (docs.lammps.org/fix_langevin_spin.html). The random torque of one spin
 * update is three Gaussian deviates from a RanMars(seed) stream, scaled by sigma = sqrt(2 D) with
 * D = alpha (1 + alpha^2) k_B T / (hbar dts), dts = dt/4, and the whole precession vector (damping and random
 * torque) is then multiplied by 1/(1 + alpha^2). The stream is drawn in the order of the spin updates of fix nve/spin
 * (src/engine/fix/nve_spin.ts), once per group atom per update, as in the source.
 */
import { FixLangevinSpin } from '../../engine/fix/langevin_spin';
import type { System } from '../../engine/system';
import { RanMars } from '../random_mars';

export class FixLangevinSpinT extends FixLangevinSpin {
  private readonly random: RanMars;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.random = new RanMars(this.seed);
  }

  protected supportsTemperature(): boolean { return true; }

  /** add_temperature of the source: w = (w + sigma (g1, g2, g3)) / (1 + alpha^2). */
  addTemperature(w: number[]): void {
    const s = this.sys.state;
    const alpha = this.damp;
    const dts = 0.25 * s.dt;
    const hbar = s.units.hplanck / (2 * Math.PI);
    const kb = s.units.boltz;
    const gil = 1.0 / (1.0 + alpha * alpha);
    let D = alpha * (1.0 + alpha * alpha) * kb * this.temp;
    D /= hbar * dts;
    const sigma = Math.sqrt(2.0 * D);
    const rx = sigma * this.random.gaussian();
    const ry = sigma * this.random.gaussian();
    const rz = sigma * this.random.gaussian();
    w[0] = (w[0] + rx) * gil;
    w[1] = (w[1] + ry) * gil;
    w[2] = (w[2] + rz) * gil;
  }
}
