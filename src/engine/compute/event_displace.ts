import type { System } from '../system';
import { Compute } from './compute';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';

/*
 * compute event/displace — implemented from docs.lammps.org/compute_event_displace.html
 * (plans/lammps-docs/compute_event_displace.rst). Quoted sentences are copied
 * character for character from that page:
 *
 *   "Define a computation that flags an "event" if any particle in the group
 *   has moved a distance greater than the specified threshold distance when
 *   compared to a previously stored reference state (i.e., the previous event)."
 *   "This value calculated by the compute is equal to 0 if no particle has
 *   moved far enough, and equal to 1 if one or more particles have moved
 *   further than the threshold distance."
 *   "The scalar value calculated by this compute is "intensive"."
 *   "This command can only be used if LAMMPS was built with the REPLICA
 *   package."
 *
 * The threshold must be a positive distance (measured with native LAMMPS, black
 * box: a threshold of -1 stops with the error Distance must be > 0 for compute
 * event/displace).
 *
 * The reference state belongs to the replica-dynamics drivers (prd, tad), which
 * the browser engine does not run. Measured with native LAMMPS (black box): in
 * ordinary runs the flag stayed 0 in every case tried, including displacements
 * of 1 length unit per step, a single 5 length unit jump between runs, a
 * threshold of 1e-6, a thermal LJ liquid with thermo every step, and a minimize
 * between runs. The mechanism that stores the reference was not identified, so
 * this compute returns 0 and makes no comparison, which matches those runs.
 * Only the threshold argument is checked.
 */
export class ComputeEventDisplace extends Compute {
  readonly style = 'event/displace';
  readonly threshold: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} event/displace: expected one argument (threshold), got ${args.length}`);
    this.threshold = parseNum(args[0], `compute ${id} event/displace threshold`);
    if (!(this.threshold > 0)) throw new StyleError('Distance must be > 0 for compute event/displace');
    this.scalarFlag = true;
    this.extscalar = 0;
  }

  protected computeScalar(): number {
    return 0;
  }
}
