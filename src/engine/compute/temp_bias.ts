import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import type { Region } from '../region';

/*
 * Temperature computes that remove a velocity bias, attached to thermostats
 * with fix_modify temp <compute-ID> (unquoted paraphrase). Written from the
 * LAMMPS documentation
 * only (docs.lammps.org/compute_temp_partial.html, compute_temp_com.html,
 * compute_temp_region.html, compute_modify.html). Comment lines marked "|"
 * are verbatim quotes from the cited .rst pages (verified with grep -F).
 *
 * All three styles share the documented temperature formula and tensor:
 *
 *   |\text{KE} = \frac{\text{dim}}{2} N k_B T,
 *
 *   |Note that because it lacks the 1/2
 *   |factor, these tensor components are twice those of the traditional
 *   |kinetic energy tensor.  The six components of the vector are ordered
 *   |:math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`xz`,
 *   |:math:`yz`.
 *
 *   |The scalar value calculated by this compute is "intensive".  The
 *   |vector values are "extensive".
 *
 * and the shared thermostat-bias contract
 * (docs.lammps.org/compute_temp_partial.html):
 *
 *   |this bias will be subtracted from
 *   |each atom, thermostatting of the remaining thermal velocity will be
 *   |performed, and the bias will be added back in.
 *
 * (docs.lammps.org/compute_temp_region.html):
 *
 *   |this bias will be subtracted from each atom, thermostatting of the
 *   |remaining thermal velocity will be performed, and the bias will be
 *   |added back in.
 */

/**
 * compute ID group-ID temp/partial xflag yflag zflag —
 * docs.lammps.org/compute_temp_partial.html:
 *
 *   |   compute ID group-ID temp/partial xflag yflag zflag
 *   |
 *   |* ID, group-ID are documented in :doc:`compute <compute>` command
 *   |* temp/partial = style name of this compute command
 *   |* xflag,yflag,zflag = 0/1 for whether to exclude/include this dimension
 *
 *   |The calculation of KE excludes the
 *   |:math:`x`, :math:`y`, or :math:`z` dimensions if *xflag*, *yflag*, or *zflag*
 *   |is 0.  The dim parameter is adjusted to give the correct number of
 *   |degrees of freedom.
 *
 * The extra degrees of freedom are scaled with the fraction of included
 * components (docs.lammps.org/compute_modify.html):
 *
 *   |For compute temp/partial, if one or
 *   |more velocity components are excluded, the value used for *extra/dof* is
 *   |scaled accordingly.
 *
 * Default (docs.lammps.org/compute_temp_partial.html): none.
 */
export class ComputeTempPartial extends ComputeTemp {
  readonly style = 'temp/partial';
  /** Included velocity components (x, y, z); a dimension beyond sys.dimension is never counted. */
  private readonly inc: boolean[];
  private readonly ninclude: number;
  /** Excluded-dim velocities of group atoms, saved while the bias is removed. */
  private saved = new Float64Array(0);
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length !== 3) throw new StyleError('usage: compute ID group-ID temp/partial xflag yflag zflag');
    const dim = sys.dimension;
    const flags = args.map((w) => {
      if (w !== '0' && w !== '1') throw new StyleError(`compute temp/partial: xflag/yflag/zflag must be 0 or 1 (got '${w}')`);
      return w === '1';
    });
    this.inc = [dim > 0 && flags[0], dim > 1 && flags[1], dim > 2 && flags[2]];
    this.ninclude = (this.inc[0] ? 1 : 0) + (this.inc[1] ? 1 : 0) + (this.inc[2] ? 1 : 0);
    if (this.ninclude === 0) throw new StyleError('compute temp/partial: all three flags are 0, no velocity component is left to thermostat');
  }

  /** dof = ninclude * N - scaled extra/dof - fix-removed DOF ("dim parameter is adjusted"). */
  dofCompute(): void {
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    const extra = (this.extraDof * this.ninclude) / this.sys.dimension;
    this.dof = this.ninclude * n - extra - this.sys.dofRemoved(this.groupBit);
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const { v, mask } = s;
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      if (this.inc[0]) t += m * v[3 * i] * v[3 * i];
      if (this.inc[1]) t += m * v[3 * i + 1] * v[3 * i + 1];
      if (this.inc[2]) t += m * v[3 * i + 2] * v[3 * i + 2];
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  /** Tensor components touching an excluded dimension are zero (their KE is excluded). */
  protected computeVector(): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const vx = this.inc[0] ? v[3 * i] : 0;
      const vy = this.inc[1] ? v[3 * i + 1] : 0;
      const vz = this.inc[2] ? v[3 * i + 2] : 0;
      t[0] += m * vx * vx; t[1] += m * vy * vy; t[2] += m * vz * vz;
      t[3] += m * vx * vy; t[4] += m * vx * vz; t[5] += m * vy * vz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  // ---- velocity bias: the excluded components are removed (set to zero)
  hasBias(): boolean { return true; }
  computeBias(): void {} // nothing to precompute; removeBias captures the saved components

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    this.ensureBias(s.n);
    this.biasOn[i] = 1;
    if (!this.inc[0]) { this.saved[3 * i] = s.v[3 * i]; s.v[3 * i] = 0; }
    if (!this.inc[1]) { this.saved[3 * i + 1] = s.v[3 * i + 1]; s.v[3 * i + 1] = 0; }
    if (!this.inc[2]) { this.saved[3 * i + 2] = s.v[3 * i + 2]; s.v[3 * i + 2] = 0; }
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    if (!this.inc[0]) s.v[3 * i] = this.saved[3 * i];
    if (!this.inc[1]) s.v[3 * i + 1] = this.saved[3 * i + 1];
    if (!this.inc[2]) s.v[3 * i + 2] = this.saved[3 * i + 2];
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.removeBias(i);
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
  }

  private ensureBias(n: number): void {
    if (this.biasOn.length >= n && this.saved.length >= 3 * n) return;
    const on = new Uint8Array(n);
    on.set(this.biasOn);
    this.biasOn = on;
    const sv = new Float64Array(3 * n);
    sv.set(this.saved);
    this.saved = sv;
  }
}

/**
 * compute ID group-ID temp/com — docs.lammps.org/compute_temp_com.html:
 *
 *   |   compute ID group-ID temp/com
 *   |
 *   |* ID, group-ID are documented in :doc:`compute <compute>` command
 *   |* temp/com = style name of this compute command
 *
 *   |Define a computation that calculates the temperature of a group of
 *   |atoms, after subtracting out the center-of-mass velocity of the group.
 *
 * The center-of-mass velocity of the group is the mass-weighted mean
 * velocity; the temperature and the tensor use v - v_com. Default
 * (docs.lammps.org/compute_temp_com.html): none.
 */
export class ComputeTempCom extends ComputeTemp {
  readonly style = 'temp/com';
  private vcm: [number, number, number] = [0, 0, 0];
  private vcmReady = false;
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length) throw new StyleError(`compute temp/com takes no arguments (got ${args.join(' ')})`);
  }

  /** Mass-weighted center-of-mass velocity of the group. */
  private vcmOf(): [number, number, number] {
    const s = this.sys.state;
    let mx = 0, my = 0, mz = 0, mt = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      mx += m * s.v[3 * i]; my += m * s.v[3 * i + 1]; mz += m * s.v[3 * i + 2]; mt += m;
    }
    return mt > 0 ? [mx / mt, my / mt, mz / mt] : [0, 0, 0];
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const [cx, cy, cz] = this.vcmOf();
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const dx = s.v[3 * i] - cx, dy = s.v[3 * i + 1] - cy, dz = s.v[3 * i + 2] - cz;
      t += m * (dx * dx + dy * dy + dz * dz);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const [cx, cy, cz] = this.vcmOf();
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const dx = s.v[3 * i] - cx, dy = s.v[3 * i + 1] - cy, dz = s.v[3 * i + 2] - cz;
      t[0] += m * dx * dx; t[1] += m * dy * dy; t[2] += m * dz * dz;
      t[3] += m * dx * dy; t[4] += m * dx * dz; t[5] += m * dy * dz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  // ---- velocity bias: the group's center-of-mass velocity is removed
  hasBias(): boolean { return true; }
  computeBias(): void { this.vcm = this.vcmOf(); this.vcmReady = true; }

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (!this.vcmReady) this.computeBias();
    this.ensureBias(s.n);
    this.biasOn[i] = 1;
    s.v[3 * i] -= this.vcm[0]; s.v[3 * i + 1] -= this.vcm[1]; s.v[3 * i + 2] -= this.vcm[2];
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] += this.vcm[0]; s.v[3 * i + 1] += this.vcm[1]; s.v[3 * i + 2] += this.vcm[2];
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.removeBias(i);
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
    this.vcmReady = false;
  }

  private ensureBias(n: number): void {
    if (this.biasOn.length >= n) return;
    const on = new Uint8Array(n);
    on.set(this.biasOn);
    this.biasOn = on;
  }
}

/**
 * compute ID group-ID temp/region region-ID — docs.lammps.org/compute_temp_region.html:
 *
 *   |   compute ID group-ID temp/region region-ID
 *   |
 *   |* ID, group-ID are documented in :doc:`compute <compute>` command
 *   |* temp/region = style name of this compute command
 *   |* region-ID = ID of region to use for choosing atoms
 *
 *   |where KE = is the total kinetic energy of the group of atoms (sum of
 *
 * (with the shared formula above; N is the number of atoms in both the group
 * and the region). The atom count is dynamic by definition and fix DOF are
 * not removed:
 *
 *   |The number of atoms contributing to the temperature is calculated each
 *   |time the temperature is evaluated since it is assumed atoms can
 *   |enter/leave the region.  Thus there is no need to use the *dynamic*
 *   |option of the :doc:`compute_modify <compute_modify>` command for this
 *   |compute style.
 *
 *   |Unlike other compute styles that calculate temperature, this compute
 *   |does not subtract out degrees-of-freedom due to fixes that constrain
 *   |motion, such as :doc:`fix shake <fix_shake>` and :doc:`fix rigid
 *   |<fix_rigid>`.
 *
 * The bias is the velocity of atoms outside the region:
 *
 *   |The removal of atoms outside the region by this fix is essentially
 *   |computing the temperature after a "bias" has been removed, which in
 *   |this case is the velocity of any atoms outside the region.
 *
 * Default (docs.lammps.org/compute_temp_region.html): none.
 */
export class ComputeTempRegion extends ComputeTemp {
  readonly style = 'temp/region';
  readonly regionId: string;
  private reg: Region | null = null;
  /** Velocities of out-of-region atoms, saved while the bias is removed. */
  private saved = new Float64Array(0);
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length !== 1) throw new StyleError('usage: compute ID group-ID temp/region region-ID');
    this.regionId = args[0];
  }

  init(): void {
    this.regionRef(); // fails at run setup when the region does not exist
    super.init();
  }

  private regionRef(): Region {
    const r = this.sys.region(this.regionId);
    this.reg = r;
    return r;
  }

  /** Atoms of the group that are inside the region (wrapped coordinates, as regions do not wrap). */
  protected counted(i: number): boolean {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return false;
    const r = this.reg ?? this.regionRef();
    return r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]);
  }

  /** No fix-removed DOF; N is re-derived on every call (atoms enter/leave the region). */
  dofCompute(): void {
    this.regionRef();
    const s = this.sys.state;
    let n = 0;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    this.dof = this.sys.dimension * n - this.extraDof;
  }

  protected computeScalar(): number {
    this.dofCompute(); // recomputed on every evaluation by documented definition
    const s = this.sys.state;
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      t += m * (s.v[3 * i] * s.v[3 * i] + s.v[3 * i + 1] * s.v[3 * i + 1] + s.v[3 * i + 2] * s.v[3 * i + 2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  // ---- velocity bias: the velocity of any atom outside the region is removed (set to zero)
  hasBias(): boolean { return true; }
  computeBias(): void { this.regionRef(); }

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    const r = this.reg ?? this.regionRef();
    if (r.match(s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2])) return; // inside: no bias
    this.ensureBias(s.n);
    this.biasOn[i] = 1;
    this.saved[3 * i] = s.v[3 * i];
    this.saved[3 * i + 1] = s.v[3 * i + 1];
    this.saved[3 * i + 2] = s.v[3 * i + 2];
    s.v[3 * i] = 0; s.v[3 * i + 1] = 0; s.v[3 * i + 2] = 0;
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] = this.saved[3 * i];
    s.v[3 * i + 1] = this.saved[3 * i + 1];
    s.v[3 * i + 2] = this.saved[3 * i + 2];
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.removeBias(i);
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
  }

  private ensureBias(n: number): void {
    if (this.biasOn.length >= n && this.saved.length >= 3 * n) return;
    const on = new Uint8Array(n);
    on.set(this.biasOn);
    this.biasOn = on;
    const sv = new Float64Array(3 * n);
    sv.set(this.saved);
    this.saved = sv;
  }
}
