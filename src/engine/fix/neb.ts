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
 * or equal; perp takes Kspring2; end takes an estyle and Kspring3 ("Note that
 * the end keyword can be used twice to add forces to both the first and last
 * replicas"). "The option defaults are parallel = neigh, perp = 0.0, ends is
 * not specified (no inter-replica force on the end replicas)." The estyles
 * map to four switches as native LAMMPS does (src/REPLICA/fix_neb.cpp at
 * b332f28): first sets the first replica's; last, last/efirst and
 * last/efirst/middle each replace the last replica's (a later end last... wins).
 * A bad keyword or value is "Illegal fix neb command", as native reports it.
 */

export type NebParallel = 'neigh' | 'ideal' | 'equal';

const ILLEGAL = 'Illegal fix neb command';

export class FixNeb extends Fix {
  readonly style = 'neb';
  readonly kspring: number;
  parallel: NebParallel = 'neigh';
  /** perp Kspring2 (0: no perpendicular spring) */
  kspringPerp = 0;
  /** end first: the first replica's free-end force, toward its energy at timestep 0 */
  freeEndIni = false;
  /** end last: the last replica's, toward its energy at timestep 0 */
  freeEndFinal = false;
  /** end last/efirst and last/efirst/middle: the last replica's, toward the first replica's current energy */
  freeEndFinalWithRespToEIni = false;
  /** end last/efirst/middle: an intermediate replica below the first replica's energy loses its gradient force */
  finalAndInterWithRespToEIni = false;
  kspringIni = 1;
  kspringFinal = 1;
  /**
   * The target energies of end first and end last, recorded when the nudging runs on timestep 0 (fix_neb.rst: "the
   * target energy ETarget is set to the initial energy of the replica"); they keep their value across neb commands.
   */
  readonly endEnergies = { eIniIni: 0, eFinalIni: 0 };

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!args.length) throw new StyleError(ILLEGAL);
    this.kspring = numberOf(args[0]);
    if (this.kspring <= 0) throw new StyleError(ILLEGAL);
    for (let k = 1; k < args.length;) {
      const key = args[k];
      if (key === 'parallel') {
        if (k + 2 > args.length) throw new StyleError(ILLEGAL);
        const v = args[k + 1];
        if (v !== 'neigh' && v !== 'ideal' && v !== 'equal') throw new StyleError(ILLEGAL);
        this.parallel = v;
        k += 2;
      } else if (key === 'perp') {
        if (k + 2 > args.length) throw new StyleError(ILLEGAL);
        this.kspringPerp = numberOf(args[k + 1]);
        if (this.kspringPerp < 0) throw new StyleError(ILLEGAL);
        k += 2;
      } else if (key === 'end') {
        if (k + 3 > args.length) throw new StyleError(ILLEGAL);
        const estyle = args[k + 1];
        if (estyle === 'first') {
          this.freeEndIni = true;
          this.kspringIni = numberOf(args[k + 2]);
        } else if (estyle === 'last' || estyle === 'last/efirst' || estyle === 'last/efirst/middle') {
          this.freeEndFinal = estyle === 'last';
          this.freeEndFinalWithRespToEIni = estyle !== 'last';
          this.finalAndInterWithRespToEIni = estyle === 'last/efirst/middle';
          this.kspringFinal = numberOf(args[k + 2]);
        } else throw new StyleError(ILLEGAL);
        k += 3;
      } else throw new StyleError(ILLEGAL);
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

/** A floating-point word as the input reader accepts it (native: utils::numeric). */
const numberOf = (w: string): number => {
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(w)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return Number(w);
};
