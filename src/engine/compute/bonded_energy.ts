import { Compute } from './compute';
import { StyleError } from '../force/types';
import { BondedHybrid } from '../force/bonded_hybrid';
import type { System } from '../system';

/*
 * compute ID group-ID bond | angle | dihedral | improper —
 * docs.lammps.org/compute_bond.html: "Define a computation that extracts the bond energy calculated by each"
 * (the rest of that sentence names the bond_style hybrid command). compute_angle.html, compute_dihedral.html and
 * compute_improper.html say the same for their kind. The Output info says the vector has one entry per sub-style:
 * "is the number of sub_styles defined by the" hybrid command, indices 1 through N; the values are extensive energies.
 * The group specified for this command is ignored (the group must still exist,
 * as for every compute).
 *
 * Measured with native LAMMPS (black box): a compute bond, angle, dihedral or
 * improper is refused unless that kind's style is hybrid, both when the style is
 * still none and when it is a plain style (Bond style for compute bond command
 * is not hybrid, and the same pattern for the other kinds). A compute defined
 * under a hybrid style that is later replaced by a plain one stops the next
 * evaluation with the same refusal. The vector entry k is the energy of the k-th
 * listed sub-style (0 for a sub-style that no type uses, and for a type set to none).
 */

export type BondedKind = 'bond' | 'angle' | 'dihedral' | 'improper';

const LABEL: Record<BondedKind, string> = { bond: 'Bond', angle: 'Angle', dihedral: 'Dihedral', improper: 'Improper' };

const hybridOf = (sys: System, kind: BondedKind): BondedHybrid => {
  const st = sys.ff[kind];
  if (!(st instanceof BondedHybrid)) throw new StyleError(`${LABEL[kind]} style for compute ${kind} command is not hybrid`);
  return st;
};

export class ComputeBondedEnergy extends Compute {
  readonly style: BondedKind;

  constructor(sys: System, id: string, group: string, args: string[], kind: BondedKind) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Illegal compute ${kind} command`);
    this.style = kind;
    const st = hybridOf(sys, kind);
    const n = st.subNames.length;
    this.vectorFlag = true;
    this.extvector = 1;
    this.sizeVector = n;
    this.vector = new Float64Array(n);
  }

  protected computeVector(): void {
    const st = hybridOf(this.sys, this.style);
    if (st.subNames.length !== this.vector.length) {
      throw new StyleError(`compute ${this.id}: ${this.style}_style hybrid now has ${st.subNames.length} sub-styles, not ${this.vector.length}; define the compute again`);
    }
    // the sub-style energies come from the last force evaluation (compute pe uses the same accumulator)
    this.sys.forces();
    this.vector.set(st.subEnergies());
  }
}
