import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group-ID temp/cs group1 group2 — docs.lammps.org/compute_temp_cs.html.
 * Quoted lines are verbatim from plans/lammps-docs/compute_temp_cs.rst:
 *
 *   |   compute ID group-ID temp/cs group1 group2
 *   |* group1 = group-ID of either cores or shells
 *   |* group2 = group-ID of either shells or cores
 *
 *   |Define a computation that calculates the temperature of a system based
 *   |on the center-of-mass velocity of atom pairs that are bonded to each
 *   |other.
 *
 *   |For this compute, core and shell particles are specified by two
 *   |respective group IDs, which can be defined using the :doc:`group
 *   |<group>` command.  The number of atoms in the two groups must be the
 *   |same and there should be one bond defined between a pair of atoms in the
 *   |two groups.  Non-polarized ions which might also be included in the
 *   |treated system should not be included into either of these groups, they
 *   |are taken into account by the *group-ID* (second argument) of the
 *   |compute.
 *
 *   |Note that the velocity of each core or shell atom used in the KE
 *   |calculation is the velocity of the center-of-mass (COM) of the
 *   |core/shell pair the atom is part of.
 *
 *   |This "bias" is the velocity of the atom relative
 *   |to the center-of-mass velocity of the core/shell pair.  If this compute
 *   |is used with a fix command that performs thermostatting then this bias
 *   |will be subtracted from each atom, thermostatting of the remaining
 *   |center-of-mass velocity will be performed, and the bias will be added
 *   |back in.
 *
 *   |The number of core/shell pairs contributing to the temperature is
 *   |assumed to be constant for the duration of the run.
 *
 * Pairs are the bonds (sys.state.topo.bonds) with one atom in group1 and the
 * other in group2. The KE of each pair member uses the mass-weighted COM
 * velocity of its pair; N and the dof follow compute temp for the atoms of the
 * compute group. The KE, tensor and the thermostat bias follow the same
 * documented formulas (compute_temp_cs.html: \text{KE} = \frac{\text{dim}}{2} N k_B T).
 *
 * Restrictions enforced here (the doc words them as "should be" / "must be
 * the same", so a violation throws instead of being approximated):
 *   - group1 and group2 hold the same number of atoms;
 *   - every atom of group1 and group2 has exactly one bond to the other group;
 *   - a pair with one member in the compute group has its partner there too.
 */
export class ComputeTempCs extends ComputeTemp {
  readonly style = 'temp/cs';
  private readonly bit1: number;
  private readonly bit2: number;
  readonly group1: string;
  readonly group2: string;
  /** Per-atom COM velocity of its pair (3 per atom), set by computeBias. */
  private vcm = new Float64Array(0);
  /** Per-atom bias (relative velocity to the pair COM), saved while removed. */
  private bias = new Float64Array(0);
  private biasOn = new Uint8Array(0);
  private biasReady = false;
  /** 1 for group atoms of a pair covered by the bias (set by computeBias). */
  private pairedMask = new Uint8Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (args.length !== 2) throw new StyleError('usage: compute ID group-ID temp/cs group1 group2');
    this.group1 = args[0];
    this.group2 = args[1];
    this.bit1 = sys.groups.bit(args[0]);
    this.bit2 = sys.groups.bit(args[1]);
  }

  /**
   * Partner index of every atom of group1/group2 (-1 for other atoms), from the
   * bonds between the two groups. Throws when the documented pairing is violated.
   */
  private partners(): Int32Array {
    const s = this.sys.state;
    const n = s.n;
    const partner = new Int32Array(n).fill(-1);
    let n1 = 0, n2 = 0;
    for (let i = 0; i < n; i++) {
      if (s.mask[i] & this.bit1) n1++;
      if (s.mask[i] & this.bit2) n2++;
    }
    if (n1 !== n2) throw new StyleError(`compute ${this.id} (temp/cs): ${this.group1} has ${n1} atoms, ${this.group2} has ${n2}; they must be equal`);
    const bonds = s.topo.bonds;
    for (let k = 0; k < bonds.n; k++) {
      const ia = this.sys.indexOfId(bonds.atoms[2 * k]);
      const ib = this.sys.indexOfId(bonds.atoms[2 * k + 1]);
      if (ia < 0 || ib < 0) continue;
      const a1 = (s.mask[ia] & this.bit1) !== 0, a2 = (s.mask[ia] & this.bit2) !== 0;
      const b1 = (s.mask[ib] & this.bit1) !== 0, b2 = (s.mask[ib] & this.bit2) !== 0;
      const pairAB = (a1 && b2) || (a2 && b1);
      if (!pairAB) continue;
      if (partner[ia] >= 0 || partner[ib] >= 0) {
        throw new StyleError(`compute ${this.id} (temp/cs): an atom of ${this.group1} or ${this.group2} has more than one bond to the other group`);
      }
      partner[ia] = ib;
      partner[ib] = ia;
    }
    for (let i = 0; i < n; i++) {
      const in1 = (s.mask[i] & this.bit1) !== 0, in2 = (s.mask[i] & this.bit2) !== 0;
      if (!(in1 || in2)) continue;
      if (partner[i] < 0) {
        throw new StyleError(`compute ${this.id} (temp/cs): atom ${s.id[i]} of ${this.group1} or ${this.group2} has no bond to the other group`);
      }
      if (this.groupBit && (s.mask[i] & this.groupBit) && !(s.mask[partner[i]] & this.groupBit)) {
        throw new StyleError(`compute ${this.id} (temp/cs): the core/shell partner of atom ${s.id[i]} is not in group ${this.group}`);
      }
    }
    return partner;
  }

  /** Velocity of atom i used for the KE: its pair COM velocity if paired, else its own. */
  private kinVel(i: number, partner: Int32Array, out: number[]): void {
    const s = this.sys.state;
    const p = partner[i];
    if (p < 0) { out[0] = s.v[3 * i]; out[1] = s.v[3 * i + 1]; out[2] = s.v[3 * i + 2]; return; }
    const mi = massOf(s, i), mp = massOf(s, p);
    const inv = 1 / (mi + mp);
    for (let d = 0; d < 3; d++) out[d] = (mi * s.v[3 * i + d] + mp * s.v[3 * p + d]) * inv;
  }

  /**
   * dof = dim (N - Npairs) - extra - fix DOF: each core/shell pair counts as one
   * particle. Measured with native LAMMPS (black box): the temperature of the
   * 22-atom core/shell oracle system (8 pairs, 6 ions, dim 3) divides the
   * COM-based sum by 39 = 3 (22 - 8) - 3, not by 63 = 3 * 22 - 3.
   */
  dofCompute(): void {
    const s = this.sys.state;
    const partner = this.partners();
    let n = 0, pairs2 = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      n++;
      if (partner[i] >= 0 && (s.mask[partner[i]] & this.groupBit)) pairs2++;
    }
    const npairs = pairs2 / 2;
    this.dof = this.sys.dimension * (n - npairs) - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /** KE uses the pair COM velocity (docs: "the velocity of each core or shell atom used in the KE calculation is the velocity of the center-of-mass (COM) of the core/shell pair"). */
  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const partner = this.partners();
    const u = [0, 0, 0];
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      this.kinVel(i, partner, u);
      t += massOf(s, i) * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  /**
   * Measured with native LAMMPS (black box): the tensor of the same oracle
   * system equals the sum of m v_a v_b over the plain atom velocities (xx, yy,
   * xz match to 1e-15 after thermo's per-atom normalisation), not the pair-COM
   * velocities used by the scalar. The tensor is therefore computed on v.
   */
  protected computeVector(): void {
    const s = this.sys.state;
    const { v } = s;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      t[0] += m * vx * vx; t[1] += m * vy * vy; t[2] += m * vz * vz;
      t[3] += m * vx * vy; t[4] += m * vx * vz; t[5] += m * vy * vz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  // ---- velocity bias: the velocity relative to the pair COM velocity
  hasBias(): boolean { return true; }

  computeBias(): void {
    const s = this.sys.state;
    const partner = this.partners();
    this.ensureBias(s.n);
    this.pairedMask.fill(0);
    const u = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (partner[i] < 0 || !(s.mask[i] & this.groupBit)) continue;
      this.kinVel(i, partner, u);
      this.vcm[3 * i] = u[0]; this.vcm[3 * i + 1] = u[1]; this.vcm[3 * i + 2] = u[2];
      this.pairedMask[i] = 1;
    }
    this.biasReady = true;
  }

  removeBias(i: number): void {
    if (this.biasOn[i]) return;
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (!this.biasReady) this.computeBias();
    if (!this.pairedMask[i]) return;
    this.biasOn[i] = 1;
    for (let d = 0; d < 3; d++) {
      const k = 3 * i + d;
      this.bias[k] = s.v[k] - this.vcm[k];
      s.v[k] = this.vcm[k];
    }
  }

  restoreBias(i: number): void {
    if (!this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    for (let d = 0; d < 3; d++) s.v[3 * i + d] += this.bias[3 * i + d];
  }

  removeBiasAll(): void {
    const s = this.sys.state;
    if (!this.biasReady) this.computeBias();
    for (let i = 0; i < s.n; i++) this.removeBias(i);
  }

  restoreBiasAll(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) this.restoreBias(i);
    this.biasReady = false;
  }

  private ensureBias(n: number): void {
    if (this.biasOn.length >= n && this.bias.length >= 3 * n) return;
    const on = new Uint8Array(n);
    on.set(this.biasOn);
    this.biasOn = on;
    const b = new Float64Array(3 * n);
    b.set(this.bias);
    this.bias = b;
    const vc = new Float64Array(3 * n);
    vc.set(this.vcm);
    this.vcm = vc;
    const pm = new Uint8Array(n);
    pm.set(this.pairedMask);
    this.pairedMask = pm;
  }
}
