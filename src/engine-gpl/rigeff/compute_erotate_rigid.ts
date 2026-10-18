/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/RIGID/compute_erotate_rigid.cpp
 * and compute_erotate_rigid.h (class ComputeERotateRigid, compute erotate/rigid): the total rotational kinetic energy
 * of the rigid bodies of a fix rigid. Contributing author (from the sources): Trung Dac Nguyen (ORNL). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { extractErotational, rigidBodiesOf } from './rigid_bodies';

export class ComputeERotateRigid extends Compute {
  readonly style: string = 'erotate/rigid';
  private readonly rfix: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError('Illegal compute erotate/rigid command');
    this.rfix = args[0];
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  /** compute_erotate_rigid.cpp init: the fix must exist and be a rigid fix. */
  override init(): void {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    if (!fix) throw new StyleError(`Fix ID ${this.rfix} for compute erotate/rigid does not exist`);
    if (!fix.style.startsWith('rigid')) throw new StyleError(`Compute erotate/rigid with non-rigid fix ${this.rfix}`);
  }

  /** compute_erotate_rigid.cpp compute_scalar: extract_erotational (0 for a rigid fix without bodies), times mvv2e. */
  protected computeScalar(): number {
    const fix = this.sys.fixes.find((f) => f.id === this.rfix);
    const bodies = fix ? rigidBodiesOf(this.sys, fix) : null;
    const e = bodies ? extractErotational(bodies) : 0.0;
    return e * this.sys.state.units.mvv2e;
  }
}
