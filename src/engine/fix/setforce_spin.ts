import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { spinContribute, spinOwnerValue } from '../force/spin_book';
import { numeric } from '../force/util';

/*
 * fix ID group-ID setforce/spin fx fy fz — docs.lammps.org/fix_setforce.html: "Style *spin* suffix sets the
 * components of the magnetic precession vectors instead of the mechanical forces." and "This command can be
 * used to freeze the magnetic moment of certain atoms in the simulation by zeroing their precession vector."
 * Each component is a number or NULL (NULL leaves the component as computed by the other magnetic terms).
 * The fix must be defined after the other magnetic fixes and spin pair styles, so that it sees their sum: it
 * sets its own contribution so that the total precession vector equals the value. Variables (v_name) are refused.
 */
export class FixSetForceSpin extends Fix {
  readonly style: string = 'setforce/spin';
  private readonly comp: (number | null)[];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (!sys.state.sp) throw new StyleError('fix setforce/spin requires atom_style spin');
    if (args.length < 3) throw new StyleError('usage: fix ID group-ID setforce/spin fx fy fz');
    // As native LAMMPS does (src/fix_setforce.cpp FixSetForce::FixSetForce at b332f28, inherited by FixSetForceSpin): the optional words
    // after the three values are the keyword loop, whose only keyword is region; any other word is "Unknown fix setforce keyword: {}"
    for (let k = 3; k < args.length; k++) {
      if (args[k] !== 'region') throw new StyleError(`Unknown fix setforce keyword: ${args[k]}`);
      throw new StyleError('fix setforce/spin: the region keyword is not supported by the browser engine');
    }
    this.comp = args.slice(0, 3).map((w) => {
      if (w === 'NULL') return null;
      if (w.startsWith('v_')) throw new StyleError(`fix setforce/spin: variables (${w}) are not implemented in the browser engine`);
      // As native LAMMPS does (src/SPIN/fix_setforce_spin.cpp, via FixSetForce at b332f28): utils::numeric
      return numeric(w);
    });
  }

  /**
   * The value the spin integrator gives the precession vector of atom i once every other term is in (native
   * src/SPIN/fix_setforce_spin.cpp single_setforce_spin, called last by fix nve/spin's ComputeInteractionsSpin at
   * b332f28): the set components replace those of w for an atom of the group.
   */
  single(i: number, w: Float64Array | number[]): void {
    if (!(this.sys.state.mask[i] & this.groupBit)) return;
    for (let k = 0; k < 3; k++) {
      const c = this.comp[k];
      if (c !== null) w[k] = c;
    }
  }

  /** The minimizers apply the fix after each force evaluation too (native min_post_force). */
  minPostForce(): void { this.postForce(); }

  postForce(): void {
    const s = this.sys.state;
    const fm = s.fm!;
    const vals = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      for (let k = 0; k < 3; k++) {
        const own = spinOwnerValue(s, this, i, k);
        const others = fm[3 * i + k] - own;
        const c = this.comp[k];
        vals[3 * i + k] = c === null ? own : c - others;
      }
    }
    spinContribute(s, this, vals, 0);
  }
}
