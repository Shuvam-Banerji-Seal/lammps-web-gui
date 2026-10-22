/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/compute_pressure_alchemy.cpp and src/REPLICA/compute_pressure_alchemy.h (ComputePressureAlchemy:
 * the constructor, init, compute_scalar and compute_vector) at commit b332f28. Authors of the source: the LAMMPS
 * REPLICA package (Axel Kohlmeyer, Temple University). Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * compute ID group-ID pressure/alchemy fix-ID (docs.lammps.org/compute_pressure_alchemy.html): the combined pressure
 * tensor of the two partitions, taken from fix alchemy. The scalar is the average of Pxx, Pyy and Pzz (in 2d, of Pxx
 * and Pyy), as native compute_scalar does.
 */
import { StyleError } from '../../engine/force/types';
import { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { FixAlchemy } from './fix_alchemy';

export class ComputePressureAlchemy extends Compute {
  readonly style = 'pressure/alchemy';
  private readonly fixId: string;
  pressFlag = true;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // compute_pressure_alchemy.cpp constructor: the count, group all, then the fix must exist
    if (args.length !== 1) throw new StyleError('Illegal compute pressure/alchemy command');
    if (group !== 'all') throw new StyleError('Compute pressure/alchemy must use group all');
    this.fixId = args[0];
    if (!sys.fixes.some((f) => f.id === this.fixId)) throw new StyleError(`Could not find compute pressure/alchemy fix ID ${this.fixId} for fix alchemy`);
    this.scalarFlag = true;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extscalar = 0;
    this.extvector = 0;
    this.vector = new Float64Array(6);
  }

  /** The fix alchemy this compute reads (checked at use, as native init does). */
  private fix(): FixAlchemy {
    const f = this.sys.fixes.find((x) => x.id === this.fixId);
    if (!(f instanceof FixAlchemy)) throw new StyleError(`Could not find compute pressure/alchemy fix ID ${this.fixId} for fix alchemy`);
    return f;
  }

  protected computeScalar(): number {
    const w = this.fix().pressureTensor();
    return this.sys.dimension === 3 ? (w[0] + w[1] + w[2]) / 3.0 : (w[0] + w[1]) / 2.0;
  }

  protected computeVector(): void {
    this.vector.set(this.fix().pressureTensor());
  }
}
