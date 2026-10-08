import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';

/*
 * compute heat/flux — implemented only from docs.lammps.org/compute_heat_flux.html
 * (source plans/lammps-docs/compute_heat_flux.rst). Quotes are copied from that file.
 *
 * "compute ID group-ID heat/flux ke-ID pe-ID stress-ID"
 * "ke-ID = ID of a compute that calculates per-atom kinetic energy"; "pe-ID = ID of a compute
 * that calculates per-atom potential energy"; "stress-ID = ID of a compute that calculates
 * per-atom stress".
 *
 * the stress-tensor form of J on the page, where "e_i in
 * the first term of the equation is the per-atom energy (potential and kinetic)" and "S_i in
 * the second term is the per-atom stress tensor calculated by the compute stress-ID". The 1/V
 * factor is NOT included: "the :math:`1/V` scaling factor in the equation for :math:`\mathbf{J}` is
 * **not** included in the calculation performed by these computes". "The tensor multiplies" v_i "by a :math:`3\times3` matrix" "to yield a vector."
 *
 * Output: "This compute calculates a global vector of length 6. The first three components are
 * the x, y, and z components of the full heat flux vector (i.e., J_x, J_y, and J_z). The next three
 * components are the x, y, and z components of just the convective portion of the flux (i.e., the
 * first term in the equation for J)." "The vector values calculated by this compute are "extensive""
 *
 * Measured with native LAMMPS (black box): with stress/atom NULL virial (per-atom S = -W), a
 * 108-atom LJ fluid gives J = convective - sum_i S_i v_i to 1e-12 (at step 0 the two are equal
 * because the perfect lattice's virial is isotropic and the momentum is zero). The stress layout is
 * xx yy zz xy xz yz, as in compute stress/atom.
 */
export class ComputeHeatFlux extends Compute {
  readonly style = 'heat/flux';
  private readonly keId: string;
  private readonly peId: string;
  private readonly stressId: string;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('usage: compute ID group-ID heat/flux ke-ID pe-ID stress-ID');
    [this.keId, this.peId, this.stressId] = args;
    this.vectorFlag = true;
    this.sizeVector = 6;
    this.extvector = 1;
    this.vector = new Float64Array(6);
  }

  init(): void {
    const ke = this.sys.compute(this.keId);
    const pe = this.sys.compute(this.peId);
    const st = this.sys.compute(this.stressId);
    if (!ke.peratomFlag || ke.sizePeratomCols !== 0) throw new StyleError(`compute ${this.id}: ${this.keId} must be a per-atom vector (compute ke/atom)`);
    if (!pe.peratomFlag || pe.sizePeratomCols !== 0) throw new StyleError(`compute ${this.id}: ${this.peId} must be a per-atom vector (compute pe/atom)`);
    if (!st.peratomFlag || st.sizePeratomCols !== 6) throw new StyleError(`compute ${this.id}: ${this.stressId} must be a per-atom array with 6 columns (compute stress/atom)`);
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const e = this.sys.compute(this.keId).peratomValues();
    const p = this.sys.compute(this.peId).peratomValues();
    const S = this.sys.compute(this.stressId).peratomValues();
    // compute_heat_flux.html: "The vector values will be in energy*velocity units"; the per-atom stress
    // is in pressure*volume units (compute stress/atom), so it is converted back to energy units
    const inv = 1 / s.units.nktv2p;
    const v = s.v;
    const conv = [0, 0, 0];
    const sv = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ei = e[i] + p[i];
      for (let a = 0; a < 3; a++) conv[a] += ei * v[3 * i + a];
      const sxx = S[6 * i] * inv, syy = S[6 * i + 1] * inv, szz = S[6 * i + 2] * inv;
      const sxy = S[6 * i + 3] * inv, sxz = S[6 * i + 4] * inv, syz = S[6 * i + 5] * inv;
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      sv[0] += sxx * vx + sxy * vy + sxz * vz;
      sv[1] += sxy * vx + syy * vy + syz * vz;
      sv[2] += sxz * vx + syz * vy + szz * vz;
    }
    for (let a = 0; a < 3; a++) {
      this.vector[a] = conv[a] - sv[a];
      this.vector[3 + a] = conv[a];
    }
  }
}

