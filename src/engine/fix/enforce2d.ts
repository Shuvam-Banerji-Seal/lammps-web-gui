import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { isEllipsoid } from '../atoms';

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

  /**
   * Aspherical particles: the angular momentum components along x and y are
   * zeroed too, so that rotation stays about z. Measured with native LAMMPS
   * (black box): after run 0 with fix enforce2d the x and y angmom of every
   * ellipsoid are 0 while z is kept, and without the fix they keep the values
   * from set angmom. The torque x and y are zero anyway for particles in the
   * plane, so this is the only aspherical term. The setup of a run applies
   * postForce too (fix.ts), so the zeroing is also done at setup.
   */
  private zeroAngmomPlane(): void {
    const s = this.sys.state;
    const L = s.angmom;
    if (!L) return;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i) || !isEllipsoid(s, i)) continue;
      L[3 * i] = 0;
      L[3 * i + 1] = 0;
    }
  }

  postForce(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!this.inGroup(i)) continue;
      s.v[3 * i + 2] = 0;
      s.f[3 * i + 2] = 0;
    }
    this.zeroAngmomPlane();
  }

  minPostForce(): void { this.postForce(); }
}
