import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import { numeric } from '../force/util';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group-ID temp/ramp vdim vlo vhi dim clo chi keyword value ...
 * — docs.lammps.org/compute_temp_ramp.html. Quoted lines are verbatim from
 * plans/lammps-docs/compute_temp_ramp.rst:
 *
 *   |   compute ID group-ID temp/ramp vdim vlo vhi dim clo chi keyword value ...
 *   |* vdim = *vx* or *vy* or *vz*
 *   |* vlo,vhi = subtract velocities between vlo and vhi (velocity units)
 *   |* dim = *x* or *y* or *z*
 *   |* clo,chi = lower and upper bound of domain to subtract from (distance units)
 *   |* keyword = *units*
 *   |*units* value = *lattice* or *box*
 *
 *   |The meaning of the arguments for this command which define the
 *   |velocity ramp are the same as for the :doc:`velocity ramp <velocity>`
 *   |command which was presumably used to impose the velocity.
 *
 *   |The *units* keyword determines the meaning of the distance units used
 *   |for coordinates (*clo*, *chi*) and velocities (*vlo*, *vhi*).  A *box* value
 *   |selects standard distance units as defined by the :doc:`units <units>`
 *   |command (e.g., :math:`\AA` for units = real or metal).  A
 *   |*lattice* value means the distance units are in lattice spacings (i.e.,
 *   |velocity in lattice spacings per unit time).
 *
 *   |The default option is units = lattice.
 *
 * The ramp itself is the one of the velocity ramp command, quoted from
 * plans/lammps-docs/velocity.rst (the ramp style is "similar to that used by
 * the compute temp/ramp command"):
 *
 *   |Velocities ramped uniformly
 *   |from vlo to vhi are applied to dimension vx, or vy, or vz.  The value
 *   |assigned to a particular atom depends on its relative coordinate value
 *   |(in dim) from clo to chi.
 *   |Atoms outside
 *   |the coordinate bounds (less than 5 or greater than 25 in this case),
 *   |are assigned velocities equal to vlo or vhi (0.0 or 5.0 in this case).
 *
 * so the bias of atom i is vlo + f (vhi - vlo) with f = (x_dim - clo) /
 * (chi - clo) clamped to [0, 1]. In lattice units clo, chi are multiplied by
 * the lattice spacing of dim and vlo, vhi by the spacing of vdim (1 without a
 * lattice command), as the velocity command does with units lattice.
 *
 * The KE and tensor follow the compute temp formulas (docs.lammps.org/compute_temp_ramp.html:
 * KE = dim/2 N k_B T; the tensor components are "twice those of the traditional kinetic
 * energy tensor", ordered xx, yy, zz, xy, xz, yz), evaluated on
 * the thermal velocity v - bias. The dof is the compute temp count (fixes that constrain
 * motion are subtracted, as in compute temp).
 *
 * Bias for thermostats (docs.lammps.org/compute_temp_ramp.html): "this bias will
 * be subtracted from each atom, thermostatting of the remaining thermal velocity
 * will be performed, and the bias will be added back in."
 */
export class ComputeTempRamp extends ComputeTemp {
  readonly style = 'temp/ramp';
  private readonly vdim: number;
  private readonly cdim: number;
  private readonly vlo: number;
  private readonly vhi: number;
  private readonly clo: number;
  private readonly chi: number;
  private readonly latticeUnits: boolean;
  /** Ramp velocity subtracted from each atom while the bias is removed (restored by +=). */
  private saved = new Float64Array(0);
  private biasOn = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    // As native LAMMPS does (src/compute_temp_ramp.cpp ComputeTempRamp::ComputeTempRamp at b332f28): the keywords are read
    // first, then vdim and dim; every malformed word is "Illegal compute temp/ramp command" (a short line "Illegal compute
    // temp command"), the bounds are read by utils::numeric, and clo and chi may come in either order
    if (args.length < 6) throw new StyleError('Illegal compute temp command');
    const illegal = 'Illegal compute temp/ramp command';
    let latticeUnits = true;
    for (let k = 6; k < args.length; k += 2) {
      if (args[k] !== 'units' || k + 1 >= args.length || (args[k + 1] !== 'lattice' && args[k + 1] !== 'box')) throw new StyleError(illegal);
      latticeUnits = args[k + 1] === 'lattice';
    }
    const vdim = ['vx', 'vy', 'vz'].indexOf(args[0]);
    if (vdim < 0) throw new StyleError(illegal);
    this.vlo = numeric(args[1]);
    this.vhi = numeric(args[2]);
    const cdim = ['x', 'y', 'z'].indexOf(args[3]);
    if (cdim < 0) throw new StyleError(illegal);
    this.clo = numeric(args[4]);
    this.chi = numeric(args[5]);
    this.vdim = vdim;
    this.cdim = cdim;
    this.latticeUnits = latticeUnits;
  }

  /** Bounds in box units: lattice spacings multiply the coordinates (dim) and velocities (vdim). */
  private bounds(): { vlo: number; vhi: number; clo: number; chi: number } {
    if (!this.latticeUnits) return { vlo: this.vlo, vhi: this.vhi, clo: this.clo, chi: this.chi };
    const sp = this.sys.lattice ? this.sys.lattice.spacing : [1, 1, 1];
    return {
      vlo: this.vlo * sp[this.vdim], vhi: this.vhi * sp[this.vdim],
      clo: this.clo * sp[this.cdim], chi: this.chi * sp[this.cdim],
    };
  }

  /** Ramp velocity of atom i along vdim (clamped to vlo/vhi outside the coordinate bounds). */
  private rampOf(i: number, b: { vlo: number; vhi: number; clo: number; chi: number }): number {
    const c = this.sys.state.x[3 * i + this.cdim];
    // clamped as native's MAX(fraction, 0.0) then MIN(fraction, 1.0): a NaN (an atom at clo == chi) becomes 0
    let f = (c - b.clo) / (b.chi - b.clo);
    f = f > 0 ? f : 0;
    f = f < 1 ? f : 1;
    return b.vlo + f * (b.vhi - b.vlo);
  }

  /**
   * Thermal velocity of atom i (velocity minus the ramp bias on vdim). While the
   * bias is removed (thermostats, velocity scale with bias yes) the stored
   * velocity is already thermal, so nothing is subtracted again.
   */
  private thermal(i: number, b: { vlo: number; vhi: number; clo: number; chi: number }, out: number[]): void {
    const v = this.sys.state.v;
    out[0] = v[3 * i]; out[1] = v[3 * i + 1]; out[2] = v[3 * i + 2];
    if (!this.biasOn[i]) out[this.vdim] -= this.rampOf(i, b);
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const b = this.bounds();
    const u = [0, 0, 0];
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.thermal(i, b, u);
      const m = massOf(s, i);
      t += m * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const b = this.bounds();
    const u = [0, 0, 0];
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.thermal(i, b, u);
      const m = massOf(s, i);
      t[0] += m * u[0] * u[0]; t[1] += m * u[1] * u[1]; t[2] += m * u[2] * u[2];
      t[3] += m * u[0] * u[1]; t[4] += m * u[0] * u[2]; t[5] += m * u[1] * u[2];
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  // ---- velocity bias: the ramp velocity on vdim, removed from each atom in the group
  hasBias(): boolean { return true; }
  computeBias(): void {}

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    this.ensureBias(s.n);
    const r = this.rampOf(i, this.bounds());
    this.biasOn[i] = 1;
    this.saved[i] = r;
    s.v[3 * i + this.vdim] -= r;
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    this.sys.state.v[3 * i + this.vdim] += this.saved[i];
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
    if (this.biasOn.length >= n && this.saved.length >= n) return;
    const on = new Uint8Array(n);
    on.set(this.biasOn);
    this.biasOn = on;
    const sv = new Float64Array(n);
    sv.set(this.saved);
    this.saved = sv;
  }
}
