import { ComputeTemp } from './temp';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * compute ID group-ID temp/profile xflag yflag zflag binstyle args keyword value ...
 * — docs.lammps.org/compute_temp_profile.html (plans/lammps-docs-live/compute_temp_profile.rst).
 * Quoted sentences are verbatim from that page:
 *
 *   "xflag,yflag,zflag = 0/1 for whether to exclude/include this dimension"
 *   "The xflag, yflag, zflag settings determine which components of average velocity are subtracted out."
 *   "When a temperature is computed, the center-of-mass velocity for the set of atoms that are both in
 *    the compute group and in the same spatial bin is calculated.  This bias velocity is then subtracted
 *    from the velocities of individual atoms in the bin to yield a thermal velocity for each atom."
 *   "The :math:`N_s N_x N_y N_z` term is the number of degrees of freedom subtracted to adjust for the
 *    removal of the center-of-mass velocity in each direction of the *Nx\*Ny\*Nz* bins"
 *   "The extra term defaults to :math:`\text{dim} - N_s`"
 *   "Bins in :math:`x` vary fastest, then :math:`y`, then :math:`z`."
 *
 * Degrees of freedom, measured with native LAMMPS (black box): the group temperature divides
 * 2 KE by dim N - Ns Nx Ny Nz - extra - (fix constraints), with extra = dim - Ns by default (the
 * probe tests with 1 1 1 x 3 and 1 0 1 x 3 agreed to 13 digits). Per bin the degrees of freedom are
 * dim N_b - Ns - (extra + fix constraints) N_b / N, which measured native per-bin temperatures
 * reproduce. The KE formula printed on the doc page reads dim/N; the measured rule is dim N (the
 * native temperatures agree with it, not with the printed fraction).
 *
 * Measured with native LAMMPS (black box): the bin velocity is the mass-weighted (centre-of-mass) mean of the
 * bin's atoms (tests/oracle/w48cb_tp_mass.in, masses 1 and 3); a plain mean does not reproduce it.
 *
 * Measured with native LAMMPS (black box): an atom that has crossed a periodic boundary and is not yet wrapped
 * (between reneighborings, e.g. x slightly below lo) is binned periodically, i.e. into the last bin; clamping it into
 * the first bin departs from native from the step where that happens (tests/oracle/w48cb_tp_twobins.in, two x bins
 * under fix nve with neigh_modify every 10, 40 steps). Non-periodic dimensions clamp to the end bins.
 *
 * out tensor (default): six components as in compute temp, without the 1/2 factor.
 * out bin: array of Nx Ny Nz rows (x fastest), columns: atom count, temperature of the bin.
 *
 * Velocity bias: "If this compute is used with a fix command that performs thermostatting then this
 * bias will be subtracted from each atom, thermostatting of the remaining thermal velocity will be
 * performed, and the bias will be added back in."
 */

type Dims = { nx: number; ny: number; nz: number };

export class ComputeTempProfile extends ComputeTemp {
  readonly style = 'temp/profile';
  private readonly flags: [number, number, number];
  private readonly nb: Dims;
  private readonly nbins: number;
  private readonly ns: number;
  private readonly binDims: [boolean, boolean, boolean];
  /** Bias velocity of each atom (only flagged components non-zero), set by computeBias. */
  private bias = new Float64Array(0);
  private biasOn = new Uint8Array(0);
  /** Atoms of the group at the last dof count (per-bin degrees of freedom share the extra terms by this). */
  private groupCount = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    const usage = 'usage: compute ID group-ID temp/profile xflag yflag zflag binstyle args keyword value ...';
    if (args.length < 5) throw new StyleError(usage);
    const flags: number[] = [];
    for (let k = 0; k < 3; k++) {
      if (args[k] !== '0' && args[k] !== '1') throw new StyleError(`compute ${id} (temp/profile): ${['xflag', 'yflag', 'zflag'][k]} must be 0 or 1 (got '${args[k]}')`);
      flags.push(Number(args[k]));
    }
    this.flags = [flags[0], flags[1], flags[2]] as [number, number, number];
    const style = args[3];
    const counts: Record<string, string[]> = { x: ['x'], y: ['y'], z: ['z'], xy: ['x', 'y'], yz: ['y', 'z'], xz: ['x', 'z'], xyz: ['x', 'y', 'z'] };
    const dims = counts[style];
    if (!dims) throw new StyleError(`compute ${id} (temp/profile): binstyle must be x, y, z, xy, yz, xz or xyz (got '${style}')`);
    const nbin: Dims = { nx: 1, ny: 1, nz: 1 };
    let k = 4;
    for (const d of dims) {
      const w = args[k++];
      const v = Math.floor(parseNum(w, `compute ${id} (temp/profile) N${d}`));
      if (!(v >= 1) || String(v) !== String(parseNum(w, 'N'))) throw new StyleError(`compute ${id} (temp/profile): N${d} must be a positive integer (got '${w}')`);
      if (d === 'x') nbin.nx = v; else if (d === 'y') nbin.ny = v; else nbin.nz = v;
    }
    this.nb = nbin;
    this.nbins = nbin.nx * nbin.ny * nbin.nz;
    this.binDims = [dims.includes('x'), dims.includes('y'), dims.includes('z')];
    this.ns = flags[0] + flags[1] + flags[2];
    // compute temp/profile: the extra degrees of freedom default to dim - Ns (overridden by compute_modify extra)
    this.extraDof = sys.dimension - this.ns;
    let outBin = false;
    // As native LAMMPS does (src/compute_temp_profile.cpp ComputeTempProfile::ComputeTempProfile at b332f28): the only
    // optional words are "out tensor" or "out bin"; any other word, or a missing value, is "Illegal compute temp/profile command".
    while (k < args.length) {
      if (args[k] !== 'out' || k + 1 >= args.length) throw new StyleError('Illegal compute temp/profile command');
      const w = args[k + 1];
      if (w !== 'tensor' && w !== 'bin') throw new StyleError('Illegal compute temp/profile command');
      outBin = w === 'bin';
      k += 2;
    }
    if (k !== args.length) throw new StyleError(usage);
    if (outBin) {
      this.arrayFlag = true;
      this.vectorFlag = false;
      this.sizeArrayRows = this.nbins;
      this.sizeArrayCols = 2;
      this.array = new Float64Array(2 * this.nbins);
      this.extvector = 0;
    } else {
      this.vectorFlag = true;
      this.sizeVector = 6;
    }
    if (sys.state.box.triclinic) throw new StyleError(`compute ${id} (temp/profile): triclinic boxes are not supported (bins would be tilted)`);
    this.dof = 0;
  }

  /** dim N - Ns Nx Ny Nz - extra - (fix constraints), as in compute temp. */
  dofCompute(): void {
    let n = 0;
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) if (this.counted(i)) n++;
    this.groupCount = n;
    this.dof = this.sys.dimension * n - this.ns * this.nbins - this.extraDof - this.sys.dofRemoved(this.groupBit);
  }

  /** Bin index (x fastest) of atom i; periodic dimensions wrap the index (see the file comment). */
  private binOf(i: number): number {
    const s = this.sys.state;
    const box = this.sys.state.box;
    const idx = [0, 0, 0];
    const ns = [this.nb.nx, this.nb.ny, this.nb.nz];
    for (let d = 0; d < 3; d++) {
      if (!this.binDims[d]) continue;
      const L = box.hi[d] - box.lo[d];
      let f = Math.floor((s.x[3 * i + d] - box.lo[d]) / L * ns[d]);
      if (box.periodic[d]) f = ((f % ns[d]) + ns[d]) % ns[d];
      else { if (f < 0) f = 0; if (f >= ns[d]) f = ns[d] - 1; }
      idx[d] = f;
    }
    return idx[0] + this.nb.nx * (idx[1] + this.nb.ny * idx[2]);
  }

  /**
   * Centre-of-mass velocity of each bin (3 per bin, flagged components only are used by the caller): the
   * mass-weighted mean of the velocities of the group atoms in the bin. Atoms whose bias was removed by a
   * thermostat are left out (their velocities are already thermal).
   */
  private binVelocity(bin: Int32Array, skipBiased: boolean): Float64Array {
    const s = this.sys.state;
    const v = s.v;
    const sum = new Float64Array(3 * this.nbins), mass = new Float64Array(this.nbins);
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      if (skipBiased && this.biasOn.length && this.biasOn[i]) continue;
      const b = bin[i];
      const m = massOf(s, i);
      mass[b] += m;
      for (let d = 0; d < 3; d++) sum[3 * b + d] += m * v[3 * i + d];
    }
    for (let b = 0; b < this.nbins; b++) if (mass[b] > 0) for (let d = 0; d < 3; d++) sum[3 * b + d] /= mass[b];
    return sum;
  }

  /** Bin index of every atom of the group (-1 for atoms outside the group). */
  private binsOfAll(): Int32Array {
    const s = this.sys.state;
    const bin = new Int32Array(s.n);
    for (let i = 0; i < s.n; i++) bin[i] = this.counted(i) ? this.binOf(i) : -1;
    return bin;
  }

  /**
   * Thermal velocities of the group atoms into out (3 per atom): the velocity minus the bin
   * centre-of-mass velocity of the flagged components. Atoms whose bias was removed by a thermostat
   * already hold thermal velocities.
   */
  private thermal(out: Float64Array): void {
    const s = this.sys.state;
    const v = s.v;
    const bin = this.binsOfAll();
    const cm = this.binVelocity(bin, true);
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const b = bin[i];
      const biased = this.biasOn.length > 0 && this.biasOn[i] === 1;
      for (let d = 0; d < 3; d++) {
        const sub = biased || !this.flags[d] ? 0 : cm[3 * b + d];
        out[3 * i + d] = v[3 * i + d] - sub;
      }
    }
  }

  protected computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const u = new Float64Array(3 * s.n);
    this.thermal(u);
    let t = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      t += m * (u[3 * i] * u[3 * i] + u[3 * i + 1] * u[3 * i + 1] + u[3 * i + 2] * u[3 * i + 2]);
    }
    const tfactor = this.dof > 0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0;
    return t * tfactor;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const u = new Float64Array(3 * s.n);
    this.thermal(u);
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      const vx = u[3 * i], vy = u[3 * i + 1], vz = u[3 * i + 2];
      t[0] += m * vx * vx; t[1] += m * vy * vy; t[2] += m * vz * vz;
      t[3] += m * vx * vy; t[4] += m * vx * vz; t[5] += m * vy * vz;
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }

  /** Per bin: atom count and temperature with the fractional degrees of freedom of the bin. */
  protected computeArray(): void {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const u = new Float64Array(3 * s.n);
    this.thermal(u);
    const ke = new Float64Array(this.nbins);
    const cnt = new Float64Array(this.nbins);
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const b = this.binOf(i);
      const m = massOf(s, i);
      cnt[b]++;
      ke[b] += m * (u[3 * i] * u[3 * i] + u[3 * i + 1] * u[3 * i + 1] + u[3 * i + 2] * u[3 * i + 2]);
    }
    const d = this.sys.dimension;
    // the constraint and extra degrees of freedom are shared among the bins by atom fraction
    const shared = this.extraDof + this.sys.dofRemoved(this.groupBit);
    for (let b = 0; b < this.nbins; b++) {
      this.array[2 * b] = cnt[b];
      const dofB = d * cnt[b] - this.ns - (this.groupCount > 0 ? shared * cnt[b] / this.groupCount : 0);
      this.array[2 * b + 1] = cnt[b] > 0 && dofB > 0 ? s.units.mvv2e * ke[b] / (dofB * s.units.boltz) : 0;
    }
  }

  // ---- velocity bias: the bin average velocity, removed from each atom of the group
  hasBias(): boolean { return true; }

  computeBias(): void {
    const s = this.sys.state;
    if (this.bias.length !== 3 * s.n) this.bias = new Float64Array(3 * s.n);
    if (this.biasOn.length !== s.n) this.biasOn = new Uint8Array(s.n);
    const bin = this.binsOfAll();
    const cm = this.binVelocity(bin, true);
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i) || this.biasOn[i]) continue;
      for (let d = 0; d < 3; d++) this.bias[3 * i + d] = this.flags[d] ? cm[3 * bin[i] + d] : 0;
    }
  }

  removeBias(i: number): void {
    const s = this.sys.state;
    if (!(s.mask[i] & this.groupBit)) return;
    if (this.bias.length !== 3 * s.n) this.bias = new Float64Array(3 * s.n);
    if (this.biasOn.length !== s.n) this.biasOn = new Uint8Array(s.n);
    if (this.biasOn[i]) return;
    this.biasOn[i] = 1;
    for (let d = 0; d < 3; d++) s.v[3 * i + d] -= this.bias[3 * i + d];
  }

  restoreBias(i: number): void {
    if (!this.biasOn.length || !this.biasOn[i]) return;
    this.biasOn[i] = 0;
    const s = this.sys.state;
    for (let d = 0; d < 3; d++) s.v[3 * i + d] += this.bias[3 * i + d];
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
