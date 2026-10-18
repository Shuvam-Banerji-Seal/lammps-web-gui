/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/SPH/compute_sph_t_atom.cpp and
 * compute_sph_t_atom.h (ComputeSPHTAtom: compute sph/t/atom). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * docs.lammps.org/compute_sph_t_atom.html: "compute ID group-ID sph/t/atom" gives the per-atom temperature of an SPH
 * particle, the internal energy divided by the specific heat. The source keeps the previous value for an atom of the
 * group whose specific heat is not positive (its tvector entry is not written); this port reports 0 there, since a
 * freshly allocated per-atom array is zero.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { sphArrays } from '../sph/common';

export class ComputeSPHTAtom extends Compute {
  readonly style = 'sph/t/atom';
  peratomFlag = true;
  sizePeratomCols = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Number of arguments for compute sph/t/atom command != 3');
    sphArrays(sys.state, 'Compute sph/t/atom requires atom attributes energy and specific heat, e.g. in atom_style sph');
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const a = sphArrays(s, 'Compute sph/t/atom');
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (a.cv[i] > 0.0) this.vectorAtom[i] = a.esph[i] / a.cv[i];
    }
  }
}
