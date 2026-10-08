import { Compute } from './compute';
import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute erotate/sphere and temp/sphere for atom_style sphere.
 *
 * docs.lammps.org/compute_erotate_sphere.html: "The rotational energy is
 * computed as :math:`\frac12 I \omega^2`, where :math:`I` is the moment of
 * inertia for a sphere and :math:`\omega` is the particle's angular
 * velocity." and "For :doc:`2d models <dimension>`, particles are treated as
 * spheres, not disks, meaning their moment of inertia will be the same as in
 * 3d." — I = 2/5 m r^2 (solid sphere).
 *
 * docs.lammps.org/compute_temp_sphere.html: "Point particles do not rotate,
 * so they have only three translational degrees of freedom.  For 3d
 * spherical particles, each has six degrees of freedom (three translational,
 * three rotational).  For 2d spherical particles, each has three degrees of
 * freedom (two translational, one rotational)." "*dof* value = *all* or
 * *rotate*". The tensor is "the same as the above expression for
 * :math:`E_\mathrm{kin}`, except that the 1/2 factor is NOT included and the
 * :math:`v_i^2` and :math:`\omega^2` are replaced by :math:`v_x v_y` and
 * :math:`\omega_x \omega_y` for the :math:`xy` component".
 *
 * Measured with native LAMMPS (black box; 3d and 2d, a mixture of radii,
 * densities and a point particle, all three omega components set):
 * - dof all = dim N + R N_ext - extra/dof - fix DOFs (R = 3 in 3d, 1 in 2d);
 *   dof rotate = R N_ext with nothing subtracted;
 * - the rotational energy sums all three omega components in 2d as well;
 * - with dof rotate the tensor holds only the rotational part;
 * - errors (native LAMMPS messages, paraphrased here, not quoted from the
 *   docs) for a missing omega atom attribute (likewise erotate/sphere) and
 *   for an unknown temp/sphere dof keyword.
 */

const SPHERE_INERTIA = 0.4;

export class ComputeERotateSphere extends Compute {
  readonly style = 'erotate/sphere';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('compute erotate/sphere takes no arguments');
    if (!sys.state.omega) throw new StyleError('Compute erotate/sphere requires atom attribute omega');
    this.scalarFlag = true;
    this.extscalar = 1;
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    const w = s.omega!, r = s.radius!, m = s.rmass!;
    let e = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      e += m[i] * r[i] * r[i] * (w[3 * i] ** 2 + w[3 * i + 1] ** 2 + w[3 * i + 2] ** 2);
    }
    return 0.5 * SPHERE_INERTIA * s.units.mvv2e * e;
  }
}

export class ComputeTempSphere extends ComputeTemp {
  readonly style: string = 'temp/sphere';
  private rotateOnly = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (!sys.state.omega) throw new StyleError('Compute temp/sphere requires atom attribute omega');
    for (let k = 0; k < args.length; k += 2) {
      if (args[k] === 'dof') {
        if (args[k + 1] !== 'all' && args[k + 1] !== 'rotate') throw new StyleError(`Unknown compute temp/sphere dof keyword ${args[k + 1] ?? ''}`.trim());
        this.rotateOnly = args[k + 1] === 'rotate';
      } else if (args[k] === 'bias') {
        throw new StyleError('compute temp/sphere bias is not supported by the browser engine');
      } else throw new StyleError(`Illegal compute temp/sphere keyword ${args[k]}`);
    }
  }

  dofCompute(): void {
    const s = this.sys.state;
    const d = this.sys.dimension, rot = d === 3 ? 3 : 1;
    let n = 0, ext = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      n++;
      if (s.radius![i] > 0) ext++;
    }
    this.dof = this.rotateOnly ? rot * ext : d * n + rot * ext - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /** m v_a v_b (unless dof rotate) + I w_a w_b, components xx yy zz xy xz yz. */
  private tensor(): number[] {
    const s = this.sys.state;
    const { v } = s;
    const w = s.omega!, r = s.radius!, m = s.rmass!;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      if (!this.rotateOnly) {
        const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
        t[0] += m[i] * vx * vx; t[1] += m[i] * vy * vy; t[2] += m[i] * vz * vz;
        t[3] += m[i] * vx * vy; t[4] += m[i] * vx * vz; t[5] += m[i] * vy * vz;
      }
      const I = SPHERE_INERTIA * m[i] * r[i] * r[i];
      const wx = w[3 * i], wy = w[3 * i + 1], wz = w[3 * i + 2];
      t[0] += I * wx * wx; t[1] += I * wy * wy; t[2] += I * wz * wz;
      t[3] += I * wx * wy; t[4] += I * wx * wz; t[5] += I * wy * wz;
    }
    return t;
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const t = this.tensor();
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return (t[0] + t[1] + t[2]) * tfactor;
  }

  protected computeVector(): void {
    const t = this.tensor();
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * this.sys.state.units.mvv2e;
  }
}
