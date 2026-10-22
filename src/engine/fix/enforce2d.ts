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
    if (args.length) throw new StyleError('Illegal fix enforce2d command');
  }

  init(): void {
    // the dimension is checked at init (as native LAMMPS does, src/fix_enforce2d.cpp at b332f28)
    if (this.sys.dimension === 3) throw new StyleError('Cannot use fix enforce2d with 3d simulation');
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

  /**
   * Spin: the x and y components of omega are zeroed for every atom of the group (spheres and point particles
   * alike). Measured with native LAMMPS (black box): after run 0 with fix enforce2d, two spheres with omega
   * (0.05, 0.05, 0.2) and (0.1, 0, -0.1) read back omega_x = omega_y = 0 and a point particle with omega
   * (0.1, 0.3, -0.1) reads back (0, 0, -0.1); without the fix the x and y spin are kept.
   */
  private zeroSpinPlane(): void {
    const w = this.sys.state.omega;
    if (!w) return;
    for (let i = 0; i < this.sys.state.n; i++) {
      if (!this.inGroup(i)) continue;
      w[3 * i] = 0;
      w[3 * i + 1] = 0;
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
    this.zeroSpinPlane();
  }

  minPostForce(): void { this.postForce(); }
}
