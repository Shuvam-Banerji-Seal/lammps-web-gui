/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/fix_temp_rescale_eff.cpp and
 * fix_temp_rescale_eff.h (class FixTempRescaleEff, fix temp/rescale/eff): periodic rescaling of the nuclear and
 * electron radial velocities toward a ramped target temperature (compute temp/eff), applied only when the
 * temperature is outside the window. The energy removed by the rescaling is the scalar of the fix. Contributing
 * author (from the source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { ownCompute, ramp, removeCompute, temperatureOf } from '../../engine/fix/util';

export class FixTempRescaleEff extends Fix {
  readonly style: string = 'temp/rescale/eff';
  private readonly tStart: number;
  private readonly tStop: number;
  private readonly tWindow: number;
  private readonly fraction: number;
  private readonly tempId: string;
  private removed = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 5) throw new StyleError('Illegal fix temp/rescale/eff command');
    const nevery = Number(args[0]);
    if (!Number.isInteger(nevery) || nevery <= 0) throw new StyleError('Illegal fix temp/rescale/eff command');
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Fix temp/rescale/eff requires atom style electron');
    this.nevery = nevery;
    // fix_temp_rescale_eff.cpp: global_freq = nevery
    this.globalFreq = nevery;
    this.scalarFlag = true;
    this.extscalar = 1;
    this.ecouple = () => this.removed;
    this.tStart = Number(args[1]);
    this.tStop = Number(args[2]);
    this.tWindow = Number(args[3]);
    this.fraction = Number(args[4]);
    // fix temp/rescale/eff creates its compute temp/eff for the fix group (id = fix-ID + "_temp")
    this.tempId = `${id}_temp`;
    ownCompute(sys, this.tempId, group, 'temp/eff', []);
  }

  override destroy(): void {
    removeCompute(this.sys, this.tempId);
  }

  /** fix_temp_rescale_eff.cpp end_of_step: rescale the velocities (and ervel of electrons) outside the window. */
  endOfStep(): void {
    const temp = this.sys.compute(this.tempId);
    const tCurrent = temperatureOf(this.sys, temp);
    if (tCurrent === 0.0) throw new StyleError('Computed temperature for fix temp/rescale/eff cannot be 0.0');
    let tTarget = ramp(this.sys, this.tStart, this.tStop);
    if (Math.abs(tCurrent - tTarget) > this.tWindow) {
      tTarget = tCurrent - this.fraction * (tCurrent - tTarget);
      const factor = Math.sqrt(tTarget / tCurrent);
      const s = this.sys.state;
      const efactor = 0.5 * s.units.boltz * temp.dof;
      this.removed += (tCurrent - tTarget) * efactor;
      const spin = s.custom.get(ELECTRON_SPIN)!.data;
      const ervel = s.custom.get(ELECTRON_RVEL)!.data;
      const { v, mask } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(mask[i] & this.groupBit)) continue;
        v[3 * i] *= factor;
        v[3 * i + 1] *= factor;
        v[3 * i + 2] *= factor;
        if (Math.abs(spin[i]) === 1) ervel[i] *= factor;
      }
    }
  }

  computeScalar(): number {
    return this.removed;
  }
}
