/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/compute_ke_atom_eff.cpp and
 * compute_ke_atom_eff.h (class ComputeKEAtomEff, compute ke/atom/eff): the kinetic energy of each atom of an electron
 * force field system, with the radial kinetic energy of each electron (|espin| = 1) added. Contributing author (from
 * the source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';

export class ComputeKEAtomEff extends Compute {
  readonly style: string = 'ke/atom/eff';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute ke/atom/eff command');
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Compute ke/atom/eff requires atom style electron');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  /** compute_ke_atom_eff.cpp compute_peratom: 0.5 mvv2e m v^2 for atoms of the group, plus the electron radial term. */
  protected computePeratom(): void {
    const s = this.sys.state;
    const v = s.v;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    const mvv2e = s.units.mvv2e;
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      let ke = 0.5 * mvv2e * m * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
      if (Math.abs(spin[i]) === 1) ke += 0.5 * mvv2e * mefactor * m * ervel[i] * ervel[i];
      this.vectorAtom[i] = ke;
    }
  }
}
