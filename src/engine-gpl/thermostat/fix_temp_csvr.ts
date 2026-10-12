/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_temp_csvr.cpp and fix_temp_csvr.h (FixTempCSVR: end_of_step, resamplekin, sumnoises,
 * gamdev) at commit b332f28, with the RanMars generator of src/random_mars.cpp (src/engine-gpl/random_mars.ts).
 * Based on code by Paolo Raiteri (Curtin U) and Giovanni Bussi (SISSA); contributing author of the source:
 * Axel Kohlmeyer (Temple U). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Bussi, Donadio, Parrinello, J Chem Phys 126, 014101 (2007): the scaling factor is drawn from the canonical
 * distribution of the kinetic energy; the random draws are the same as the source's (gaussian for r1, then the
 * sum of tdof-1 squared gaussians through gamdev), so a seed gives the same trajectory as native LAMMPS.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { RescaleFix } from '../../engine/fix/thermostats';
import { parseNumOrVar, ramp, valueOf } from '../../engine/fix/util';
import { RanMars } from '../random_mars';

export class FixTempCSVR extends RescaleFix {
  readonly style = 'temp/csvr';
  private readonly tPeriod: number;
  private readonly rng: RanMars;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4) throw new StyleError('usage: fix ID group temp/csvr Tstart Tstop Tdamp seed');
    this.tStart = parseNumOrVar(args[0], 'Tstart');
    this.tStop = Number(args[1]);
    this.tPeriod = Number(args[2]);
    const seed = Number(args[3]);
    if (!Number.isFinite(this.tStop) || !(this.tPeriod > 0)) throw new StyleError('Illegal fix temp/csvr command period value');
    if (!Number.isInteger(seed) || seed <= 0) throw new StyleError('Illegal fix temp/csvr command seed value');
    this.rng = new RanMars(seed);
  }

  /** Target temperature now: the Tstart..Tstop ramp, or an equal-style variable. */
  private target(): number {
    return typeof this.tStart === 'number' ? ramp(this.sys, this.tStart, this.tStop) : valueOf(this.sys, this.tStart);
  }

  /** gamdev(ia): gamma deviate of integer order ia (Numerical Recipes style; the source's branches). */
  private gamdev(ia: number): number {
    const rng = this.rng;
    if (ia < 1) return 0;
    if (ia < 6) {
      let x = 1;
      for (let j = 1; j <= ia; j++) x *= rng.uniform();
      // make certain that -log() does not overflow
      return x < 2.2250759805e-308 ? 708.4 : -Math.log(x);
    }
    const am = ia - 1;
    const s = Math.sqrt(2.0 * am + 1.0);
    restart: for (;;) {
      let x: number, y: number, v1: number;
      let e: number;
      do {
        do {
          let v2: number;
          do {
            v1 = rng.uniform();
            v2 = 2.0 * rng.uniform() - 1.0;
          } while (v1 * v1 + v2 * v2 > 1.0);
          y = v2 / v1;
          x = s * y + am;
        } while (x <= 0.0);
        if (am * Math.log(x / am) - s * y < -700 || v1 < 0.00001) continue restart;
        e = (1.0 + y * y) * Math.exp(am * Math.log(x / am) - s * y);
      } while (rng.uniform() > e);
      return x;
    }
  }

  /** sumnoises(nn): the sum of nn squared standard normal deviates (a chi-squared variate). */
  private sumnoises(nn: number): number {
    if (nn === 0) return 0;
    if (nn === 1) {
      const rr = this.rng.gaussian();
      return rr * rr;
    }
    if (nn % 2 === 0) return 2.0 * this.gamdev(nn / 2);
    const rr = this.rng.gaussian();
    return 2.0 * this.gamdev((nn - 1) / 2) + rr * rr;
  }

  /** resamplekin: the velocity scaling factor that samples the canonical kinetic energy. */
  private resamplekin(ekinOld: number, ekinNew: number): number {
    const tdof = this.temp.dof;
    const c1 = Math.exp(-this.sys.state.dt / this.tPeriod);
    const c2 = ((1.0 - c1) * ekinNew) / ekinOld / tdof;
    const r1 = this.rng.gaussian();
    const r2 = this.sumnoises(Math.trunc(tdof - 1));
    const scale = c1 + c2 * (r1 * r1 + r2) + 2.0 * r1 * Math.sqrt(c1 * c2);
    return Math.sqrt(scale);
  }

  endOfStep(): void {
    const tTarget = this.target();
    const tCur = this.current();
    const efactor = 0.5 * this.temp.dof * this.sys.state.units.boltz;
    const ekinOld = tCur * efactor;
    const ekinNew = tTarget * efactor;
    // nothing to do without degrees of freedom: no draw is made
    if (this.temp.dof < 1) return;
    // The engine keeps its guard for a zero temperature: the factor would divide by ekinOld = 0.
    if (tCur === 0) return;
    const lambda = this.resamplekin(ekinOld, ekinNew);
    this.rescale(lambda, tCur);
  }
}
