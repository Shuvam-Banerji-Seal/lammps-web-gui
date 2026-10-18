/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PERI/compute_plasticity_atom.cpp and compute_plasticity_atom.h (ComputePlasticityAtom: compute
 * plasticity/atom) at commit b332f28, with the lambdaValue array of src/PERI/fix_peri_neigh.cpp that pair_style peri/eps
 * fills. Contributing authors: Rezwanur Rahman, John Foster (UTSA). Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_plasticity_atom.html: "compute ID group-ID plasticity/atom" and "Define a computation that
 * calculates the plasticity parameter for each atom" (pair_style peri/eps). Atoms outside the group give 0.
 * The value is the cumulative plastic multiplier kept by PairPeriEPS (see pair_peri_eps.ts).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { PairPeriEPS } from './pair_peri_eps';

/** compute ID group-ID plasticity/atom: takes no arguments and needs pair_style peri/eps (native constructor checks). */
export class ComputePlasticityAtom extends Compute {
  readonly style = 'plasticity/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute plasticity/atom command');
    if (!(sys.ff.pair instanceof PairPeriEPS)) throw new StyleError('Compute plasticity/atom cannot be used with this pair style');
    this.peratomFlag = true;
    this.sizePeratomCols = 0;
  }

  protected computePeratom(): void {
    const s = this.sys.state;
    const pair = this.sys.ff.pair;
    if (!(pair instanceof PairPeriEPS)) throw new StyleError('Compute plasticity/atom cannot be used with this pair style');
    // the native fix PERI_NEIGH array exists once the family has been set up by the first evaluation
    if (!pair.isBuilt()) throw new StyleError('Compute plasticity/atom: pair_style peri/eps has not been evaluated yet (run a timestep first)');
    const lambda = pair.lambdaValue(s.n);
    this.vectorAtom = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.vectorAtom[i] = lambda[i];
  }
}
