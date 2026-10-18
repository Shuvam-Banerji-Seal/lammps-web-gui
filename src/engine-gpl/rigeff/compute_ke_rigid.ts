/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/RIGID/compute_ke_rigid.cpp and
 * compute_ke_rigid.h (class ComputeKERigid, compute ke/rigid): the total translational kinetic energy of the rigid
 * bodies of a fix rigid. Contributing author (from the sources): Trung Dac Nguyen (ORNL). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { extractKe, rigidBodiesOf } from './rigid_bodies';

export class ComputeKERigid extends Compute {
  readonly style: string = 'ke/rigid';
  private readonly rfix: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal compute ke/rigid command');
    this.rfix = args[0];
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /** compute_ke_rigid.cpp init: the fix must exist and be a rigid fix. */
  override init(): void {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    if (!fix) throw new StyleError(`Fix ID ${this.rfix} for compute ke/rigid does not exist`);
    if (!fix.style.startsWith('rigid')) throw new StyleError(`Compute ke/rigid with non-rigid fix ${this.rfix}`);
  }

  /** compute_ke_rigid.cpp compute_scalar: extract_ke (0 for a rigid fix without bodies), times mvv2e. */
  protected computeScalar(): number {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    const bodies = fix ? rigidBodiesOf(this.sys, fix) : null;
    const e = bodies ? extractKe(bodies) : 0.0;
    return e * this.sys.state.units.mvv2e;
  }
}
