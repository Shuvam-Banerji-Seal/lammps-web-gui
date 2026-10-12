/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_temp_csld.cpp and fix_temp_csld.h (FixTempCSLD: end_of_step) at commit b332f28, with
 * the RanMars generator of src/random_mars.cpp (src/engine-gpl/random_mars.ts). Based on code by Paolo Raiteri
 * (Curtin U) and Giovanni Bussi (SISSA); contributing author of the source: Axel Kohlmeyer (Temple U). Copyright
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Bussi and Parrinello, Phys. Rev. E 75, 056707 (2007): the velocities become a linear combination of the old
 * velocities and new gaussian ones at the target temperature. The draws are x, y, z per atom of the group, in
 * atom order, z included in 2d (as the source does).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { RescaleFix } from '../../engine/fix/thermostats';
import { parseNumOrVar, ramp, valueOf } from '../../engine/fix/util';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';

export class FixTempCSLD extends RescaleFix {
  readonly style = 'temp/csld';
  private readonly tPeriod: number;
  private readonly rng: RanMars;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('usage: fix ID group temp/csld Tstart Tstop Tdamp seed');
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = Number(args[1]);
    this.tPeriod = Number(args[2]);
    const seed = Number(args[3]);
    if (!Number.isFinite(this.tStop) || !(this.tPeriod > 0)) throw new StyleError('Illegal fix temp/csld command period value');
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError('Illegal fix temp/csld command seed value');
    this.rng = new RanMars(seed);
  }

  /** Target temperature now: the Tstart..Tstop ramp, or an equal-style variable. */
  private target(): number {
    return typeof this.tStart === 'number' ? ramp(this.sys, this.tStart, this.tStop) : valueOf(this.sys, this.tStart);
  }

  endOfStep(): void {
    const sys = this.sys;
    const s = sys.state;
    const comp = this.temp;
    const tTarget = this.target();
    const tCur = this.current();
    const efactor = 0.5 * comp.dof * s.units.boltz;
    const ekinOld = tCur * efactor;
    // nothing to do without degrees of freedom: no draw is made
    if (comp.dof < 1) return;

    // the draws follow the native storage order of the atoms (the random numbers go to atoms in that order)
    const atoms: number[] = [];
    for (const i of nativeOrder(s)) if (s.mask[i] & this.groupBit) atoms.push(i);
    const bias = comp.hasBias();
    // the old velocities without the bias (the bias is added back at the end, as the source does)
    if (bias) { comp.computeBias(); comp.removeBiasAll(); }
    const vhold = new Float64Array(3 * atoms.length);
    atoms.forEach((i, k) => {
      for (let d = 0; d < 3; d++) vhold[3 * k + d] = s.v[3 * i + d];
    });

    // new random velocities: x, y and z are drawn for every atom of the group
    for (const i of atoms) {
      const factor = 1.0 / Math.sqrt(massOf(s, i));
      s.v[3 * i] = this.rng.gaussian() * factor;
      s.v[3 * i + 1] = this.rng.gaussian() * factor;
      s.v[3 * i + 2] = this.rng.gaussian() * factor;
    }
    sys.refreshComputes();
    // the source takes the temperature of the random velocities for the mixing factor
    const c1 = Math.exp(-s.dt / this.tPeriod);
    const c2 = Math.sqrt(((1.0 - c1 * c1) * tTarget) / comp.scalarValue());
    atoms.forEach((i, k) => {
      for (let d = 0; d < 3; d++) s.v[3 * i + d] = vhold[3 * k + d] * c1 + s.v[3 * i + d] * c2;
    });
    if (bias) comp.restoreBiasAll();
    sys.refreshComputes();

    // tally the kinetic energy transferred between heat bath and system
    const ekinNew = comp.scalarValue() * efactor;
    this.exchanged += ekinOld - ekinNew;
  }
}
