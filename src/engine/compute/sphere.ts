import { Compute } from './compute';
import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import { missingArgs } from '../force/util';
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
  private biasId: string | null = null;
  private biasCompute: Compute | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    // As native LAMMPS does (src/compute_temp_sphere.cpp at b332f28): the keywords, then the atom attributes
    for (let k = 0; k < args.length; k += 2) {
      if (args[k] === 'dof') {
        if (k + 2 > args.length) throw missingArgs('compute temp/sphere dof');
        if (args[k + 1] !== 'all' && args[k + 1] !== 'rotate') throw new StyleError(`Unknown compute temp/sphere dof keyword ${args[k + 1]}`);
        this.rotateOnly = args[k + 1] === 'rotate';
      } else if (args[k] === 'bias') {
        if (k + 2 > args.length) throw missingArgs('compute temp/sphere bias');
        this.biasId = args[k + 1];
      } else throw new StyleError(`Unknown compute temp/sphere keyword ${args[k]}`);
    }
    // native: dof rotate counts no extra degrees of freedom unless compute_modify extra/dof sets them
    if (this.rotateOnly) this.extraDof = 0;
    if (!sys.state.omega) throw new StyleError('Compute temp/sphere requires atom attribute omega');
    if (!sys.state.radius) throw new StyleError('Compute temp/sphere requires atom attribute radius');
  }

  /**
   * As native LAMMPS does (src/compute_temp_sphere.cpp init at b332f28): the bias compute must exist, compute a
   * temperature with a velocity bias, and have this compute's group.
   */
  override init(): void {
    this.biasCompute = null;
    if (this.biasId !== null) {
      const bc = this.sys.computes.find((c) => c.id === this.biasId);
      if (!bc) throw new StyleError(`Could not find compute ${this.biasId} for temperature bias`);
      if (!bc.tempFlag) throw new StyleError('Bias compute does not calculate temperature');
      if (!bc.hasBias()) throw new StyleError('Bias compute does not calculate a velocity bias');
      if (bc.groupBit !== this.groupBit) throw new StyleError('Bias compute group does not match compute group');
      this.biasCompute = bc;
    }
    super.init();
  }

  /**
   * As native LAMMPS does (src/compute_temp_sphere.cpp dof_compute at b332f28): 3 translational and 3 rotational degrees
   * of freedom per extended particle in 3d (2 and 1 in 2d), point particles translational only, dof rotate counting the
   * rotational ones; a bias takes its dof_remove (per atom for temp/region), then extra/dof and the fixes' share.
   */
  dofCompute(): void {
    const s = this.sys.state;
    const three = this.sys.dimension === 3;
    const all = !this.rotateOnly;
    const perAtom = (i: number): number => (s.radius![i] === 0 ? (all ? (three ? 3 : 2) : 0) : all ? (three ? 6 : 3) : three ? 3 : 1);
    let count = 0, natoms = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      natoms++;
      count += perAtom(i);
    }
    let dof = count;
    const bc = this.biasCompute;
    if (bc && bc.style === 'temp/region') {
      for (let i = 0; i < s.n; i++) if (this.counted(i) && bc.dofRemove(i)) dof -= perAtom(i);
    } else if (bc && all) {
      dof -= bc.dofRemove(-1) * natoms;
    }
    this.dof = dof - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /** Runs f with the bias velocity removed from the group (native remove_bias_all / restore_bias_all). */
  private withBias<T>(f: () => T): T {
    const bc = this.biasCompute;
    if (!bc) return f();
    bc.computeBias();
    bc.removeBiasAll();
    try { return f(); } finally { bc.restoreBiasAll(); }
  }

  /** A thermostat using this compute removes the bias compute's bias (native remove_bias delegates to it). */
  override hasBias(): boolean { return this.biasCompute !== null; }
  override computeBias(): void { this.biasCompute?.computeBias(); }
  override removeBiasAll(): void { this.biasCompute?.removeBiasAll(); }
  override restoreBiasAll(): void { this.biasCompute?.restoreBiasAll(); }
  override removeBias(i: number): void { this.biasCompute?.removeBias(i); }
  override restoreBias(i: number): void { this.biasCompute?.restoreBias(i); }

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
    // native recomputes the dof on every call with a temp/region bias (tempbias 2)
    if (this.dynamicDof || this.biasCompute?.style === 'temp/region') this.dofCompute();
    const s = this.sys.state;
    const t = this.withBias(() => this.tensor());
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return (t[0] + t[1] + t[2]) * tfactor;
  }

  protected computeVector(): void {
    const t = this.withBias(() => this.tensor());
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * this.sys.state.units.mvv2e;
  }
}
