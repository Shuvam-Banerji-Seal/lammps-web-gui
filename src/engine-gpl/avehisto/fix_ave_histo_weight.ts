/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/fix_ave_histo_weight.cpp and
 * fix_ave_histo_weight.h (FixAveHistoWeight: fix ave/histo/weight, the first value is binned and the second is its
 * weight). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version
 * 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from the source): Shawn Coleman
 * (ARL).
 *
 * The binning, the min/max and the missing count are the core histogram's (src/engine/fix/ave_atom.ts), with a weight
 * per sample. The weighted variant takes exactly two values, of the same length, and has no local kind (native
 * fix_ave_histo_weight.cpp: "must have two data arguments"; "value and weight vector lengths do not match").
 */
import { FixAveHisto } from '../../engine/fix/ave_atom';
import { StyleError } from '../../engine/force/types';
import { globalScalar, globalVector, peratomValues } from '../../engine/refs';
import type { System } from '../../engine/system';

export class FixAveHistoWeight extends FixAveHisto {
  /** The style the core constructor names in its messages. */
  static readonly histoStyle = 'ave/histo/weight';
  readonly style: string = 'ave/histo/weight';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (this.inputs.length !== 2) {
      throw new StyleError(`Illegal fix ave/histo/weight command: must have two data arguments`);
    }
    if (this.inputs.some((inp) => inp.local)) {
      throw new StyleError(`fix ${id} (ave/histo/weight): local values are not supported by the browser engine`);
    }
  }

  /** Bins the first value of each repeat with the second as its weight (native: bin_one_weights and friends). */
  protected sampleInputs(): void {
    const [valIn, wtIn] = this.inputs;
    const sys = this.sys;
    const n = sys.state.n;
    if (valIn.peratom) {
      // "For per-atom input values, only atoms in the group contribute to the histogram"
      const vals = peratomValues(sys, valIn.ref);
      const wts = peratomValues(sys, wtIn.ref);
      for (let i = 0; i < n; i++) {
        if (this.inGroup(i)) this.tally(vals[i], wts[i]);
      }
    } else if (this.mode === 'scalar') {
      this.tally(globalScalar(sys, valIn.ref), globalScalar(sys, wtIn.ref));
    } else {
      const vals = globalVector(sys, valIn.ref);
      const wts = globalVector(sys, wtIn.ref);
      if (vals.length !== wts.length) {
        throw new StyleError(`Fix ave/histo/weight value and weight vector lengths do not match`);
      }
      for (let k = 0; k < vals.length; k++) this.tally(vals[k], wts[k]);
    }
  }
}
