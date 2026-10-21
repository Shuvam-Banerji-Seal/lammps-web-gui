/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_temp_rotate.cpp
 * and compute_temp_rotate.h (ComputeTempRotate: compute temp/rotate, the temperature of the group after removing the
 * velocity of its rigid rotation about the centre of mass). Contributing author: Laurent Joly (U Lyon, France). Copyright
 * (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_temp_rotate.html: "compute ID group-ID temp/rotate" (no arguments). The bias of each atom is
 * the velocity of the group's rigid rotation, vcm + omega x (r - xcm), with r the unwrapped position; the temperature
 * uses the thermal velocity v - bias. The scalar is intensive, the vector (xx, yy, zz, xy, xz, yz) extensive. A
 * thermostat or velocity command with fix_modify temp uses the bias (remove_bias_all / restore_bias_all).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { ComputeTemp } from '../../engine/compute/temp';
import { groupMotion } from '../group/group_omega';
import type { V3 } from '../rigid/math_extra';

export class ComputeTempRotate extends ComputeTemp {
  readonly style = 'temp/rotate';
  /** Native vbiasall: the velocity bias of every group atom, from the last computeBias (3 per atom). */
  private vbias = new Float64Array(0);
  private biasReady = false;
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length) throw new StyleError('Illegal compute temp/rotate command');
  }

  hasBias(): boolean { return true; }

  /** dof = dim N - extra - fixes, with the group count kept for the negative-dof test (native dof_compute). */
  dofCompute(): void {
    super.dofCompute();
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) n++;
    this.natomsTemp = n;
  }

  /** The group members in native order, their unwrapped positions, velocities and masses, and the rotation (vcm, omega). */
  private motion(): { members: number[]; vcm: V3; xcm: V3; omega: V3 } {
    const s = this.sys.state;
    const members: number[] = [];
    for (const i of nativeOrder(s)) if (s.mask[i] & this.groupBit) members.push(i);
    const pos: V3[] = [], vel: V3[] = [], mass: number[] = [];
    for (const i of members) {
      const u = [0, 0, 0];
      this.sys.geom.unwrap(s.x, s.image, i, u);
      pos.push([u[0], u[1], u[2]]);
      vel.push([s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]]);
      mass.push(massOf(s, i));
    }
    // Group::mass and Group::vcm: the total mass, and the mass-weighted velocity divided by it
    let masstotal = 0.0;
    for (let k = 0; k < members.length; k++) masstotal += mass[k];
    const p: V3 = [0, 0, 0];
    for (let k = 0; k < members.length; k++) {
      p[0] += vel[k][0] * mass[k];
      p[1] += vel[k][1] * mass[k];
      p[2] += vel[k][2] * mass[k];
    }
    const vcm: V3 = [p[0], p[1], p[2]];
    if (masstotal > 0.0) { vcm[0] /= masstotal; vcm[1] /= masstotal; vcm[2] /= masstotal; }
    // Group::xcm, Group::angmom, Group::inertia and Group::omega (group_omega.ts)
    const g = groupMotion(pos, vel, mass);
    return { members, vcm, xcm: g.xcm, omega: g.omega };
  }

  /** Computes the velocity bias of the group atoms from the current state (native compute_scalar / compute_vector). */
  private biasFrom(m: { members: number[]; vcm: V3; xcm: V3; omega: V3 }): void {
    const s = this.sys.state;
    if (this.vbias.length < 3 * s.n) this.vbias = new Float64Array(3 * s.n);
    const { vcm, xcm, omega } = m;
    for (const i of m.members) {
      const u = [0, 0, 0];
      this.sys.geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - xcm[0], dy = u[1] - xcm[1], dz = u[2] - xcm[2];
      this.vbias[3 * i] = vcm[0] + dz * omega[1] - dy * omega[2];
      this.vbias[3 * i + 1] = vcm[1] + dx * omega[2] - dz * omega[0];
      this.vbias[3 * i + 2] = vcm[2] + dy * omega[0] - dx * omega[1];
    }
    this.biasReady = true;
  }

  protected computeScalar(): number {
    const s = this.sys.state;
    const m = this.motion();
    this.biasFrom(m);
    let t = 0.0;
    for (const i of m.members) {
      const v0 = s.v[3 * i] - this.vbias[3 * i];
      const v1 = s.v[3 * i + 1] - this.vbias[3 * i + 1];
      const v2 = s.v[3 * i + 2] - this.vbias[3 * i + 2];
      t += (v0 * v0 + v1 * v1 + v2 * v2) * massOf(s, i);
    }
    if (this.dynamicDof) this.dofCompute();
    if (this.dof < 0.0 && this.natomsTemp > 0) throw new StyleError('Temperature compute degrees of freedom < 0');
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0.0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const m = this.motion();
    this.biasFrom(m);
    const t = [0, 0, 0, 0, 0, 0];
    for (const i of m.members) {
      const v0 = s.v[3 * i] - this.vbias[3 * i];
      const v1 = s.v[3 * i + 1] - this.vbias[3 * i + 1];
      const v2 = s.v[3 * i + 2] - this.vbias[3 * i + 2];
      const massone = massOf(s, i);
      t[0] += massone * v0 * v0;
      t[1] += massone * v1 * v1;
      t[2] += massone * v2 * v2;
      t[3] += massone * v0 * v1;
      t[4] += massone * v0 * v2;
      t[5] += massone * v1 * v2;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  // ---- velocity bias (remove_bias / restore_bias of the native class)
  computeBias(): void {
    this.biasFrom(this.motion());
  }

  removeBias(i: number): void {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (!this.biasReady) this.computeBias();
    if (this.biasOn.length < s.n) { const b = new Uint8Array(s.n); b.set(this.biasOn); this.biasOn = b; }
    if (this.biasOn[i]) return;
    this.biasOn[i] = 1;
    s.v[3 * i] -= this.vbias[3 * i];
    s.v[3 * i + 1] -= this.vbias[3 * i + 1];
    s.v[3 * i + 2] -= this.vbias[3 * i + 2];
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] += this.vbias[3 * i];
    s.v[3 * i + 1] += this.vbias[3 * i + 1];
    s.v[3 * i + 2] += this.vbias[3 * i + 2];
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.removeBias(i);
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
  }
}
