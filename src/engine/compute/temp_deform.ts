import { ComputeTemp } from './temp';
import type { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';
import { isDeformFix, type FixDeform } from '../fix/deform';

/*
 * compute ID group-ID temp/deform — docs.lammps.org/compute_temp_deform.html.
 * Quoted lines are verbatim from that page (compute_temp_deform.rst in
 * plans/lammps-docs, checked with grep -F).
 *
 *   |   compute ID group-ID temp/deform
 *
 *   |Define a computation that calculates the temperature of a group of
 *   |atoms, after subtracting out a streaming velocity induced by the
 *   |simulation box changing size and/or shape, for example in a
 *   |non-equilibrium MD (NEMD) simulation.  The size/shape change is
 *   |induced by use of the :doc:`fix deform <fix_deform>` command.
 *
 *   |This position-dependent streaming velocity is
 *   |subtracted from each atom's actual velocity to yield a thermal
 *   |velocity, which is then used to compute the temperature.
 *
 * The streaming velocity is the one induced by the defined fix deform
 * (fix/deform.ts): v = d(r_lo)/dt + (dH/dt) lamda with the triclinic edge
 * matrix H — atoms frozen in fractional coordinates move with the box, so a
 * pure xy shear gives "atoms at the bottom of the box (low *y*) have a small *x*
 * velocity, while atoms at the top of the box (high *y*) have a large *x*
 * velocity" (compute_temp_deform.rst; "0 at the bottom to 10 at the top of the box" is in fix_deform.rst).
 *
 * After the subtraction the documented temperature and tensor formulas of
 * compute temp apply (compute_temp_deform.html):
 *
 *   |   \text{KE} = \frac{\text{dim}}{2} N k_B T,
 *
 *   |A symmetric tensor, stored as a six-element vector, is also calculated
 *   |by this compute for use in the computation of a pressure tensor by the
 *   |:doc:`compute pressue <compute_pressure>` command.
 *
 *   |The six components of the vector are ordered
 *   |:math:`xx`, :math:`yy`, :math:`zz`, :math:`xy`, :math:`xz`,
 *   |:math:`yz`.
 *
 *   |This compute calculates a global scalar (the temperature) and a global
 *   |vector of length 6 (symmetric tensor), which can be accessed by
 *   |indices 1--6.
 *
 *   |The scalar value calculated by this compute is "intensive".  The
 *   |vector values are "extensive".
 *
 * Bias for thermostatting fixes (compute_temp_deform.html):
 *
 *   |The removal of the box deformation velocity component by this fix is
 *   |essentially computing the temperature after a "bias" has been removed
 *   |from the velocity of the atoms.  If this compute is used with a fix
 *   |command that performs thermostatting then this bias will be subtracted
 *   |from each atom, thermostatting of the remaining thermal velocity will be
 *   |performed, and the bias will be added back in.
 *
 * Default (compute_temp_deform.html): "none".
 */
export class ComputeTempDeform extends ComputeTemp {
  readonly style = 'temp/deform';
  private deform: FixDeform | null = null;
  private vs = [0, 0, 0];
  /** Streaming velocity saved while the bias is removed (thermostat use). */
  private saved = new Float64Array(0);
  private biasOn = new Uint8Array(0);

  /** ID of the internal temperature compute (keyword temp), or null for compute temp. */
  private tempId: string | null = null;
  private internal: Compute | null = null;
  private thermalSaved = new Float64Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    // compute_temp_deform.rst: "keyword = *temp*" with "*temp* value = compute ID that calculates a temperature"
    for (let k = 0; k < args.length;) {
      if (args[k] !== 'temp') throw new StyleError(`Unknown compute temp/deform keyword: ${args[k]} (compute temp/deform takes no arguments other than the keyword temp)`);
      if (args[k + 1] === undefined) throw new StyleError('compute temp/deform keyword temp needs a compute ID');
      this.tempId = args[k + 1];
      k += 2;
    }
  }

  init(): void {
    this.deform = this.findDeform();
    this.internal = null;
    if (this.tempId !== null) {
      // Native messages (own words here): the compute must exist, must compute a temperature and must share the group.
      const c = this.sys.computes.find((x) => x.id === this.tempId);
      if (!c) throw new StyleError(`Temperature ID ${this.tempId} for compute temp/deform does not exist`);
      if (!c.tempFlag) throw new StyleError(`Temperature ID ${this.tempId} for compute temp/deform does not compute temperature`);
      if (c.groupBit !== this.groupBit) throw new StyleError(`Group of temperature compute with ID ${this.tempId} for compute temp/deform does not match`);
      this.internal = c;
    }
    super.init();
  }

  /**
   * With keyword temp, the internal compute evaluates the thermal velocities: the streaming
   * velocity is subtracted from the group's velocities for the duration of the call.
   */
  private withThermal<T>(fn: () => T): T {
    const s = this.sys.state;
    const df = this.deformFix();
    if (this.thermalSaved.length < 3 * s.n) this.thermalSaved = new Float64Array(3 * s.n);
    const vs = this.vs;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.thermalSaved[3 * i] = s.v[3 * i]; this.thermalSaved[3 * i + 1] = s.v[3 * i + 1]; this.thermalSaved[3 * i + 2] = s.v[3 * i + 2];
      this.streaming(df, i, vs);
      s.v[3 * i] -= vs[0]; s.v[3 * i + 1] -= vs[1]; s.v[3 * i + 2] -= vs[2];
    }
    try {
      return fn();
    } finally {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        s.v[3 * i] = this.thermalSaved[3 * i]; s.v[3 * i + 1] = this.thermalSaved[3 * i + 1]; s.v[3 * i + 2] = this.thermalSaved[3 * i + 2];
      }
    }
  }

  private findDeform(): FixDeform {
    const f = this.sys.fixes.find((x) => isDeformFix(x));
    if (!f) throw new StyleError('compute temp/deform requires a fix deform command (none defined)');
    return f as FixDeform;
  }

  private deformFix(): FixDeform {
    return this.deform ?? this.findDeform();
  }

  protected computeScalar(): number {
    if (this.internal) {
      const c = this.internal as unknown as { computeScalar(): number };
      return this.withThermal(() => c.computeScalar());
    }
    if (this.dynamicDof) this.dofCompute();
    const df = this.deformFix();
    const s = this.sys.state;
    const vs = this.vs;
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.streaming(df, i, vs);
      const m = massOf(s, i);
      const dx = s.v[3 * i] - vs[0], dy = s.v[3 * i + 1] - vs[1], dz = s.v[3 * i + 2] - vs[2];
      t += m * (dx * dx + dy * dy + dz * dz);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    if (this.internal) {
      const c = this.internal as unknown as { computeVector(): void };
      const src = this.internal.vector;
      this.withThermal(() => c.computeVector());
      for (let k = 0; k < 6; k++) this.vector[k] = src[k];
      return;
    }
    const df = this.deformFix();
    const s = this.sys.state;
    const vs = this.vs;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.streaming(df, i, vs);
      const m = massOf(s, i);
      const dx = s.v[3 * i] - vs[0], dy = s.v[3 * i + 1] - vs[1], dz = s.v[3 * i + 2] - vs[2];
      t[0] += m * dx * dx; t[1] += m * dy * dy; t[2] += m * dz * dz;
      t[3] += m * dx * dy; t[4] += m * dx * dz; t[5] += m * dy * dz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  /**
   * The streaming velocity still in atom i's stored velocity: none while the bias is removed
   * (velocity scale with bias yes reads the temperature after removing it), so it is not
   * subtracted twice.
   */
  private streaming(df: FixDeform, i: number, out: number[]): void {
    if (this.biasOn[i]) { out[0] = 0; out[1] = 0; out[2] = 0; return; }
    df.vstream(i, out);
  }

  // ---- velocity bias: the box deformation streaming velocity
  hasBias(): boolean { return true; }
  computeBias(): void {}

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    this.ensureBias(s.n);
    this.biasOn[i] = 1;
    const vs = [0, 0, 0];
    this.deformFix().vstream(i, vs);
    this.saved[3 * i] = vs[0]; this.saved[3 * i + 1] = vs[1]; this.saved[3 * i + 2] = vs[2];
    s.v[3 * i] -= vs[0]; s.v[3 * i + 1] -= vs[1]; s.v[3 * i + 2] -= vs[2];
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    s.v[3 * i] += this.saved[3 * i];
    s.v[3 * i + 1] += this.saved[3 * i + 1];
    s.v[3 * i + 2] += this.saved[3 * i + 2];
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
