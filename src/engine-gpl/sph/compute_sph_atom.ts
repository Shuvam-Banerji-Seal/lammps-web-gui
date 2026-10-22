/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/compute_sph_rho_atom.cpp and
 * compute_sph_e_atom.cpp (ComputeSPHRhoAtom: compute sph/rho/atom; ComputeSPHEAtom: compute sph/e/atom). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Per-atom density (docs.lammps.org/compute_sph_rho_atom.html) and internal energy (compute_sph_e_atom.html) of the
 * atoms of the group; atoms outside the group give 0.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { requireSph, sphArrays } from './common';

abstract class ComputeSPHAtom extends Compute {
  peratomFlag = true;
  sizePeratomCols = 0;
  /** The per-atom property of the state this compute reports (rho or esph). */
  protected abstract pick(a: ReturnType<typeof sphArrays>): Float64Array;

  constructor(sys: System, id: string, group: string, args: string[], private readonly what: string, attribute: string) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(what === 'sph/e/atom' ? 'Number of arguments for compute sph/e/atom command != 3' : `Illegal compute ${what} command`);
    requireSph(sys.state, `Compute ${what} requires atom attribute ${attribute}, e.g. in atom_style sph`);
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const src = this.pick(sphArrays(s, `Compute ${this.what}`));
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = src[i];
  }
}

export class ComputeSPHRhoAtom extends ComputeSPHAtom {
  readonly style: string = 'sph/rho/atom';
  protected pick(a: ReturnType<typeof sphArrays>): Float64Array { return a.rho; }
  constructor(sys: System, id: string, group: string, args: string[]) { super(sys, id, group, args, 'sph/rho/atom', 'density'); }
}

export class ComputeSPHEAtom extends ComputeSPHAtom {
  readonly style: string = 'sph/e/atom';
  protected pick(a: ReturnType<typeof sphArrays>): Float64Array { return a.esph; }
  constructor(sys: System, id: string, group: string, args: string[]) { super(sys, id, group, args, 'sph/e/atom', 'energy'); }
}
