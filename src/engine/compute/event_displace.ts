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
 * The reference state is stored by the hyper command. hyper.html: "Compute event/displace checks whether any atom
 * in the compute group has moved further than a specified threshold distance." Measured with native LAMMPS (black
 * box): in ordinary runs, with no reference stored, the flag stayed 0 in every case tried, including
 * displacements of 1 length unit per step, a single 5 length unit jump between runs, a threshold of 1e-6, a
 * thermal LJ liquid with thermo every step, and a minimize between runs.
 */
export class ComputeEventDisplace extends Compute {
  readonly style = 'event/displace';
  readonly threshold: number;
  /** Unwrapped positions of the reference state (the previous event), 3 per atom; null until one is stored. */
  private reference: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 1) throw new StyleError(`compute ${id} event/displace: expected one argument (threshold), got ${args.length}`);
    this.threshold = parseNum(args[0], `compute ${id} event/displace threshold`);
    if (!(this.threshold > 0)) throw new StyleError('Distance must be > 0 for compute event/displace');
    this.scalarFlag = true;
    this.extscalar = 0;
  }

  /**
   * Set by the hyper command while it owns the reference (the event check of the hyper command is the only user of
   * the flag: see computeScalar).
   */
  hyperOwned = false;

  /** The reference state is the quenched state of the last event (or of the start of a hyper run). */
  storeReference(): void {
    const s = this.sys.state;
    const out = new Float64Array(3 * s.n);
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      this.sys.geom.unwrap(s.x, s.image, i, u);
      out[3 * i] = u[0]; out[3 * i + 1] = u[1]; out[3 * i + 2] = u[2];
    }
    this.reference = out;
  }

  /**
   * Compares the current positions with the reference: flags the group atoms that moved more than the threshold.
   * maxMove is the largest move of any group atom (the drift of fix hyper/global).
   */
  displacement(): { count: number; maxMove: number; flags: Uint8Array } {
    const s = this.sys.state;
    const flags = new Uint8Array(s.n);
    const out = { count: 0, maxMove: 0, flags };
    // no reference stored yet: the reference is zero (measured with native LAMMPS, black box: the first quench of a
    // hyper command flags atoms far from the origin)
    const ref = this.reference && this.reference.length === 3 * s.n ? this.reference : new Float64Array(3 * s.n);
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - ref[3 * i], dy = u[1] - ref[3 * i + 1], dz = u[2] - ref[3 * i + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d > out.maxMove) out.maxMove = d;
      if (d > this.threshold) { flags[i] = 1; out.count++; }
    }
    return out;
  }

  /**
   * 1 when any group atom has moved further than the threshold from the reference (the quenched state of the last
   * event or bond reset, stored by the hyper command), else 0. Measured with native LAMMPS (black box): outside a
   * hyper command the flag stayed 0 in every case tried: a single 0.8 length unit displacement of every atom
   * between two runs, an atom displaced by 2 units before a run or a minimize, a single atom given a velocity of 40
   * (lattice units) during a run, and a run after a hyper command.
   */
  protected computeScalar(): number {
    if (!this.hyperOwned) return 0;
    return this.displacement().count > 0 ? 1 : 0;
  }
}
