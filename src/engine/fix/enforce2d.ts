import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * fix ID group enforce2d — docs.lammps.org/fix_enforce2d.html: "Zero out
 * the z-dimension velocity and force on each atom in the group. This is
 * useful when running a 2d simulation to ensure that atoms do not move from
 * their initial z coordinate." Applied after forces (post_force), at setup,
 * and during minimization.
 */

export class FixEnforce2d extends Fix {
  readonly style = 'enforce2d';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('fix enforce2d takes no arguments');
    if (sys.dimension !== 2) throw new StyleError('cannot use fix enforce2d with 3d simulation');
  }

  postForce(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      s.v[3 * i + 2] = 0;
      s.f[3 * i + 2] = 0;
    }
  }

  minPostForce(): void { this.postForce(); }
}
