/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/TALLY/compute_pe_mol_tally.cpp and
 * compute_pe_mol_tally.h (ComputePEMolTally: compute pe/mol/tally). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_tally.html: "Define a computation that calculates properties between two groups of atoms by
 * accumulating them from pairwise non-bonded computations." The pair energy of every pair with one atom in group1 and the
 * other in group2 is summed in four parts: the intra-molecular and the inter-molecular pair energy, each split into the
 * van der Waals (evdwl) and the Coulomb (ecoul) part. The pair style's energy of a pair is shared between its two atoms
 * (0.5 each, as the native pair_tally_callback does) and summed back to the same total.
 *
 * The pair energies come through the engine's pairwise tally walk (walkTallyPairs, the single() call of the pair style),
 * which is the mechanism the other tally computes use in this engine; the native ev_tally callbacks are not reproduced.
 * Molecule identity uses the molecule ID of the owner of each atom (ghosts carry their owner's molecule).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { walkTallyPairs } from '../../engine/compute/tally';
import { hasMolecule } from '../../engine/atoms';
import type { System } from '../../engine/system';

export class ComputePEMolTally extends Compute {
  readonly style = 'pe/mol/tally';
  private readonly group2Bit: number;
  private cacheEpoch = -1;
  private readonly total = new Float64Array(4);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal compute pe/mol/tally command: missing argument(s)');
    if (sys.groups.find(args[0]) < 0) throw new StyleError(`Could not find compute pe/mol/tally second group ID ${args[0]}`);
    this.group2Bit = sys.groups.bit(args[0]);
    this.vectorFlag = true;
    this.sizeVector = 4;
    this.extvector = 1;
    this.vector = new Float64Array(4);
    if (!hasMolecule(sys.state)) throw new StyleError('Compute pe/mol/tally requires molecule IDs');
  }

  private evaluate(): void {
    if (this.cacheEpoch === this.sys.epoch) return;
    const s = this.sys.state;
    const owner = this.sys.nb.owner;
    const t = this.total;
    t.fill(0);
    walkTallyPairs(this.sys, this.id, this.groupBit, this.group2Bit, 'off', (i, j, evdwl, ecoul) => {
      const same = s.molecule[i] === s.molecule[owner[j]];
      // each of the two atoms of the pair takes 0.5 of its energy (native pair_tally_callback), which sums to the pair energy
      const h = 0.5 * evdwl, hc = 0.5 * ecoul;
      const k = same ? 0 : 2;
      t[k] += h; t[k + 1] += hc;
      t[k] += h; t[k + 1] += hc;
    });
    this.vector.set(t);
    this.cacheEpoch = this.sys.epoch;
  }

  protected computeVector(): void {
    this.evaluate();
  }
}
