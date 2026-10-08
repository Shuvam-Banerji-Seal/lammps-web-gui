import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ID group-ID freeze — docs.lammps.org/fix_freeze.html (fix_freeze.rst):
 * "Zero out the force and torque on a granular particle.  This is useful for
 * preventing certain particles from moving in a simulation."
 * "This fix computes a global 3-vector of forces, which can be accessed by
 * various output commands.  This is the total force on the group of atoms
 * before the forces on individual atoms are changed by the fix.  The vector
 * values calculated by this fix are "extensive"."
 * "There can only be a single freeze fix defined."
 *
 * The force vector is summed over the group before zeroing. Measured with
 * native LAMMPS (black box, not part of the doc text): for a wall/gran contact
 * the vector equals the full wall force on the frozen atom; for a granular
 * pair contact (pair_style gran/hooke, two spheres, one frozen) the vector
 * is half of the pair force on the frozen atom (Kn-only case: pair force 200,
 * vector 100). That pair-side difference is not reproduced here (see the
 * report of the wall/gran work).
 *
 * The frozen atom's forces and torques are zeroed in post_force; a run start
 * (setup) does the same, as the doc's "fix freeze" zeroes the force in every
 * force evaluation. Fix order matters: a wall defined after freeze adds its
 * force after the zeroing (measured: the frozen atom keeps the wall force).
 * The fix is not invoked during energy minimization (not implemented here:
 * minimize does not call postForce for this fix).
 */

export class FixFreeze extends Fix {
  readonly style = 'freeze';
  vectorFlag = true;
  sizeVector = 3;
  extvector = 1;
  private readonly sum = new Float64Array(3);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError(`Illegal fix freeze command: unexpected argument '${args[0]}' (fix freeze takes no arguments)`);
    if (!sys.state.torque) throw new StyleError('fix freeze requires atom_style sphere (torque)');
    if (sys.fixes.some((f) => f.style === 'freeze')) throw new StyleError('There can only be a single freeze fix defined');
  }

  postForce(): void {
    const s = this.sys.state;
    const { f, mask } = s;
    const torque = s.torque!;
    const bit = this.groupBit;
    this.sum.fill(0);
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      this.sum[0] += f[3 * i];
      this.sum[1] += f[3 * i + 1];
      this.sum[2] += f[3 * i + 2];
      f[3 * i] = f[3 * i + 1] = f[3 * i + 2] = 0;
      torque[3 * i] = torque[3 * i + 1] = torque[3 * i + 2] = 0;
    }
  }

  computeVector(i: number): number { return this.sum[i]; }
}
