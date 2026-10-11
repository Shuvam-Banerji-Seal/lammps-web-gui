import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ID group-ID neb Kspring keyword value — docs.lammps.org/fix_neb.html:
 * "Add nudging forces to atoms in the group for a multi-replica simulation
 * run via the neb command to perform a nudged elastic band (NEB) calculation
 * for finding the transition state." "The fix neb command must be used with
 * the "neb" command and defines how inter-replica nudging forces are
 * computed."
 *
 * The forces are not applied by this fix's own minimizer hook: they depend on
 * every replica's positions and energies, so the neb command (commands/neb.ts)
 * computes them after each replica's force evaluation. This class holds the
 * parsed settings only.
 *
 * Parsed keywords (docs.lammps.org/fix_neb.html): parallel takes neigh, ideal
 * or equal; perp takes Kspring2; end takes an estyle and Kspring3 and may be
 * given twice. The documented defaults are parallel neigh, perp 0.0 and no end
 * forces (the end replicas get no inter-replica force).
 */

export type NebParallel = 'neigh' | 'ideal' | 'equal';
export type NebEndStyle = 'first' | 'last' | 'last/efirst' | 'last/efirst/middle';

export class FixNeb extends Fix {
  readonly style = 'neb';
  readonly kspring: number;
  parallel: NebParallel = 'neigh';
  kspring2 = 0;
  /** end keyword(s): estyle and Kspring3 */
  ends: { estyle: NebEndStyle; kspring3: number }[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError('usage: fix ID group-ID neb Kspring [parallel neigh|ideal|equal] [perp Kspring2] [end estyle Kspring3]');
    this.kspring = numberOf(args[0], 'Kspring');
    for (let k = 1; k < args.length;) {
      const key = args[k];
      if (key === 'parallel') {
        const v = args[k + 1];
        if (v !== 'neigh' && v !== 'ideal' && v !== 'equal') throw new StyleError(`fix neb parallel must be neigh, ideal or equal (got '${v ?? ''}')`);
        this.parallel = v;
        k += 2;
      } else if (key === 'perp') {
        this.kspring2 = numberOf(args[k + 1], 'Kspring2');
        k += 2;
      } else if (key === 'end') {
        const estyle = args[k + 1];
        if (estyle !== 'first' && estyle !== 'last' && estyle !== 'last/efirst' && estyle !== 'last/efirst/middle') {
          throw new StyleError(`fix neb end estyle must be first, last, last/efirst or last/efirst/middle (got '${estyle ?? ''}')`);
        }
        this.ends.push({ estyle, kspring3: numberOf(args[k + 2], 'Kspring3') });
        k += 3;
      } else throw new StyleError(`illegal fix neb keyword '${key}'`);
    }
  }

  /** Set by the neb command while it runs; the forces are applied by the driver, not by these hooks. */
  active = false;

  // minimize and run call these hooks; the nudging forces exist only inside the neb command, so anywhere else
  // the fix is refused rather than ignored (an unsupported use is an error, not a silent no-op)
  minPostForce(): void { this.refuse(); }
  postForce(): void { this.refuse(); }

  private refuse(): void {
    if (!this.active) throw new StyleError(`fix neb ${this.id} acts only inside a neb command; minimize and run with fix neb are not supported`);
  }
}

const numberOf = (w: string | undefined, name: string): number => {
  const x = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(x)) throw new StyleError(`fix neb ${name} must be a number (got '${w ?? ''}')`);
  return x;
};
