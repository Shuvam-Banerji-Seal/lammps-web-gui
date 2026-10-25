/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-MESO/compute_edpd_temp_atom.cpp and
 * compute_edpd_temp_atom.h (ComputeEDPDTempAtom: compute edpd/temp/atom, the temperature of each atom) and
 * src/DPD-MESO/compute_tdpd_cc_atom.cpp and compute_tdpd_cc_atom.h (ComputeTDPDCCAtom: compute tdpd/cc/atom, the chemical
 * concentration of one species). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_edpd_temp_atom.html: "Define a computation that calculates the temperature of each eDPD particle
 * in a group. ... This compute calculates a per-atom vector", and docs.lammps.org/compute_tdpd_cc_atom.html: "compute ID group
 * tdpd/cc/atom index"; "index = index of chemical species (1 to Nspecies)". Atoms outside the group have value 0.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { parseInt_ } from '../../engine/force/util';
import { EDPD_TEMP, TDPD_CC, isEdpdStyle, isTdpdStyle, tdpdSpecies } from '../../engine/atoms';
import type { System } from '../../engine/system';

export class ComputeEDPDTempAtom extends Compute {
  readonly style = 'edpd/temp/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Number of arguments for compute edpd/temp/atom command != 3');
    if (!isEdpdStyle(sys.state.atomStyle)) throw new StyleError('compute edpd/temp/atom command requires atom_style edpd');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const T = s.custom.get(EDPD_TEMP)!.data;
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = T[i];
  }
}

export class ComputeTDPDCCAtom extends Compute {
  readonly style = 'tdpd/cc/atom';
  private readonly index: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Number of arguments for compute tdpd/cc/atom command != 4');
    if (!isTdpdStyle(sys.state.atomStyle)) throw new StyleError('compute tdpd/cc/atom command requires atom_style tdpd');
    // the index is read by utils::inumeric with no range check (as the constructor does); native reads cc[i][index-1]
    // outside the species when it is out of range, which the engine refuses when the compute is evaluated
    this.index = parseInt_(args[0], 'index');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const nsp = tdpdSpecies(s.atomStyle);
    if (this.index < 1 || this.index > nsp) throw new StyleError(`compute tdpd/cc/atom: index ${this.index} is not a species 1..${nsp} (native LAMMPS reads outside its array)`);
    const cc = s.custom.get(TDPD_CC)!.data;
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = cc[nsp * i + this.index - 1];
  }
}
