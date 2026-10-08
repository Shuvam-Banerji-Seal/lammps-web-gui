import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group-ID viscosity/cos — docs.lammps.org/compute_viscosity_cos.html
 * (page text in plans/lammps-docs/compute_viscosity_cos.rst).
 *
 * Syntax (docs): "compute ID group-ID viscosity/cos", "viscosity/cos = style
 * name of this compute command". The compute takes no arguments.
 *
 * Description (docs): "Define a computation that calculates the velocity
 * amplitude of a group of atoms with an cosine-shaped velocity profile and the
 * temperature of them after subtracting out the velocity profile before
 * computing the kinetic energy." The profile is a function of z:
 *
 *   v_x(z) = V cos(2 pi z / l_z),   V = sum_i 2 m_i v_{i,x} cos(2 pi z_i / l_z) / sum_i m_i
 *
 * with l_z the z-length of the box. The temperature is KE = dim/2 N k_B T
 * after the cosine-shaped collective velocity in x has been subtracted for
 * each atom, so the scalar is the temperature with a velocity bias removed,
 * and it can be used by thermostats (fix nvt, fix langevin, ...) through the
 * bias API (removeBiasAll / restoreBiasAll).
 *
 * Output (docs, Output info): "a global scalar (the temperature) and a global
 * vector of length 7 ... The first six elements of the vector are those of
 * the symmetric tensor ... The seventh is the cosine-shaped velocity amplitude
 * V". The tensor is the kinetic energy tensor "except that the 1/2 factor is
 * NOT included", with the xy component built from v_{i,x} v_{i,y} and so on,
 * in the order xx, yy, zz, xy, xz, yz. Extensivity (docs): the scalar is
 * intensive, the first six vector values are extensive and the seventh is
 * intensive.
 *
 * Restrictions (docs): "Since this compute depends on fix accelerate/cos which
 * can only work for 3d systems, it cannot be used for 2d systems."
 *
 * Measured with native LAMMPS (black box, LJ fcc liquid with deterministic
 * velocities, run 0): the scalar is sum m (v_x - V cos)^2 + ... divided by
 * (3N - 3) with mvv2e = 1; the six tensor components are sums over the
 * bias-removed velocities; V matches the 2 m v cos / m formula above. The
 * degrees of freedom are 3N - 3 (the standard compute temp count).
 *
 * The bias is V cos(2 pi (z - zlo) / l_z) in x, computed from the velocities
 * before any removal. The positions are measured from the box lower face: the
 * cosine is periodic in lz, so wrapped or unwrapped z give the same value.
 */
export class ComputeViscosityCos extends ComputeTemp {
  readonly style = 'viscosity/cos';
  /** Velocity amplitude used by the current bias (set by computeBias). */
  private amp = 0;
  /** True between removeBiasAll and restoreBiasAll. */
  private biasActive = false;
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length) throw new StyleError(`compute viscosity/cos takes no arguments (got ${args.join(' ')})`);
    if (sys.dimension === 2) throw new StyleError('compute viscosity/cos cannot be used for 2d systems (it depends on fix accelerate/cos)');
    this.vectorFlag = true;
    this.sizeVector = 7;
    this.vector = new Float64Array(7);
    this.extscalar = 0;
    this.extvector = 0;
    this.extlist = [1, 1, 1, 1, 1, 1, 0];
  }

  /** cos(2 pi (z - zlo) / lz) for the box as it is now. */
  private cosOf(z: number): number {
    const b = this.sys.state.box;
    if (b.triclinic) throw new StyleError('compute viscosity/cos does not support triclinic boxes');
    const lz = b.hi[2] - b.lo[2];
    return Math.cos((2 * Math.PI * (z - b.lo[2])) / lz);
  }

  /** V = sum 2 m v_x cos / sum m over the group (intensive). */
  private amplitude(): number {
    const s = this.sys.state;
    let num = 0, mt = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      num += 2 * m * s.v[3 * i] * this.cosOf(s.x[3 * i + 2]);
      mt += m;
    }
    return mt > 0 ? num / mt : 0;
  }

  /** Bias-free x velocity of atom i using amplitude a (0 when the bias is already removed). */
  private vxFree(i: number, a: number): number {
    const s = this.sys.state;
    return s.v[3 * i] - a * this.cosOf(s.x[3 * i + 2]);
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const a = this.amp0();
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const vx = this.vxFree(i, a);
      t += m * (vx * vx + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const a = this.amp0();
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const vx = this.vxFree(i, a), vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
      t[0] += m * vx * vx; t[1] += m * vy * vy; t[2] += m * vz * vz;
      t[3] += m * vx * vy; t[4] += m * vx * vz; t[5] += m * vy * vz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
    this.vector[6] = this.biasActive ? this.amp : this.amplitude();
  }

  /**
   * Amplitude to subtract right now: none while the bias is removed from the
   * atoms (their velocities are already bias-free), else the amplitude of the
   * current velocities.
   */
  private amp0(): number {
    return this.biasActive ? 0 : this.amplitude();
  }

  // ---- velocity bias: x velocity of each group atom loses V cos(2 pi z / lz)
  hasBias(): boolean { return true; }

  /** Fixes the amplitude from the current velocities (before any removal). */
  computeBias(): void { if (!this.biasActive) this.amp = this.amplitude(); }

  removeBias(i: number): void {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (this.biasOn.length < s.n) {
      const on = new Uint8Array(s.n);
      on.set(this.biasOn);
      this.biasOn = on;
    }
    if (this.biasOn[i]) return;
    this.biasOn[i] = 1;
    s.v[3 * i] -= this.amp * this.cosOf(s.x[3 * i + 2]);
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] += this.amp * this.cosOf(s.x[3 * i + 2]);
  }

  removeBiasAll(): void {
    this.computeBias();
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.removeBias(i);
    this.biasActive = true;
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
    this.biasActive = false;
  }
}
