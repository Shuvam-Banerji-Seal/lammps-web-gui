import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group ke — docs.lammps.org/compute_ke.html: "Define a
 * computation that calculates the translational kinetic energy of a group
 * of particles. The kinetic energy of each particle is computed as 1/2 m v^2,
 * where m and v are the mass and velocity of the particle." The value is
 * extensive.
 */

export class ComputeKE extends Compute {
  readonly style = 'ke';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute ke takes no arguments');
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    let k = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      k += m * (s.v[3 * i] ** 2 + s.v[3 * i + 1] ** 2 + s.v[3 * i + 2] ** 2);
    }
    return 0.5 * s.units.mvv2e * k;
  }
}
