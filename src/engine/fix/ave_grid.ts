import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { parseNum } from '../force/util';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';
import { massOf } from '../atoms';

/*
 * fix ave/grid — per-atom mode, implemented from docs.lammps.org/fix_ave_grid.html
 * (plans/lammps-docs/fix_ave_grid.rst). Quoted sentences are copied character
 * for character from that page.
 *
 * Syntax: "fix ID group-ID ave/grid Nevery Nrepeat Nfreq Nx Ny Nz value1 value2 ...
 * keyword args ...". "Overlay the 2d or 3d simulation box with a uniformly spaced
 * 2d or 3d grid and use it to either (a) time-average per-atom quantities for the
 * atoms in each grid cell". Grid cells are numbered with x varying fastest.
 *
 * Sampling: "The final averaged quantities are generated on time steps that are a
 * multiples of Nfreq. The average is over Nrepeat quantities, computed in the
 * preceding portion of the simulation every Nevery time steps." The page requires
 * Nfreq to be a multiple of a non-zero Nevery, and "Also, the time steps
 * contributing to the average value cannot overlap" (Nrepeat x Nevery <= Nfreq). Measured with native LAMMPS (black box): the
 * output on Nfreq steps uses the samples of Nrepeat steps; an output whose
 * window is not complete (step 0 with Nrepeat > 1) is left at zero.
 *
 * Normalization (keyword norm; "The output grid value on an Nfreq timestep is the sum
 * of the grid values in each of the Nrepeat samples, divided by Nrepeat."):
 *   all:    "Output = (Sum1 + Sum2 + ... + SumN) / (Count1 + Count2 + ... + CountN)"
 *   sample: "Output = (Sum1/Count1 + Sum2/Count2 + ... + SumN/CountN) / Nrepeat"
 *   none:   "Output = (Sum1 + Sum2 + ... SumN) / Nrepeat"
 * Measured with native LAMMPS (black box): an empty cell in a sample counts as
 * 0 in the sample average, and an empty cell gives 0 under norm all and none.
 * density/number and density/mass: "the result is also divided by the grid cell
 * volume". Measured: the output is (sum over samples of the count) / (Nrepeat x
 * cell volume at the output step) for norm all and none, and the per-sample
 * density averaged over Nrepeat for norm sample. The page gives the count output as
 * "Output count = (Count1 + Count2 + ... CountN) / Nrepeat".
 *
 * temp: the page says the fix "computes the temperature for each grid cell, by the formula" KE = DOF/2 k_B T,
 * with "The DOF is calculated as N\*adof + cdof" (defaults: adof = dimension,
 * cdof = 0). For norm all and none the DOF is summed over the samples; for
 * norm sample each sample gives its own temperature and the average over the
 * Nrepeat samples is the output (measured: an empty sample contributes 0).
 *
 * ave: one (the default) outputs each produced value as it is; running is the
 * cumulative average of all previous outputs; window M is the average of the M
 * most recent outputs. "Outputs on early steps will average over less than M
 * values if they are not available." (the running and window averages of count
 * are taken the same way; not measured).
 *
 * discard: "yes (the default), then the atom will be assigned to the closest grid
 * cell (lowest or highest) in that dimension. If discard is set to no the atom
 * will be ignored." Periodic dimensions wrap the atom position into the box.
 *
 * Output access: "For access by other commands, the name of the single grid
 * produced by this fix is" grid. The per-grid data is read by dump grid in
 * native LAMMPS; the browser engine has no grid dump yet, so the values are read
 * with gridValue() / gridCount(). This fix has no global scalar, vector or array,
 * so f_ID and f_ID[i][j] are refused, as native LAMMPS refuses them (measured:
 * the thermo keyword f_ag[1][1] fails with the message: fix ag does not compute an array).
 *
 * Not supported (StyleError): per-grid inputs c_ID:gname:dname (no engine compute
 * or fix defines a grid yet), bias (the temperature bias needs bias computes), and
 * triclinic boxes.
 */

type AttrName = 'vx' | 'vy' | 'vz' | 'fx' | 'fy' | 'fz' | 'mass';
type Column =
  | { t: 'attr'; name: AttrName; text: string }
  | { t: 'ref'; ref: Ref; text: string }
  | { t: 'dens'; mass: boolean; text: string }
  | { t: 'temp'; text: string };

const ATTRS: AttrName[] = ['vx', 'vy', 'vz', 'fx', 'fy', 'fz', 'mass'];
const KEYWORDS = new Set(['discard', 'norm', 'ave', 'bias', 'adof', 'cdof']);

const parseInt1 = (w: string | undefined, what: string, min: number): number => {
  const v = parseNum(w, what);
  if (!Number.isInteger(v) || v < min) throw new StyleError(`${what} must be an integer >= ${min}, got '${w}'`);
  return v;
};

export class FixAveGrid extends Fix {
  readonly style = 'ave/grid';
  readonly nEvery: number;
  readonly nRepeat: number;
  readonly nFreq: number;
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly ncell: number;
  private readonly cols: Column[] = [];
  private discard = true;
  private norm: 'all' | 'sample' | 'none' = 'all';
  private aveMode: 'one' | 'running' | 'window' = 'one';
  private windowM = 1;
  private adof: number;
  private cdof = 0;

  /** Current output: data[cell * ncol + col] and count[cell]. */
  private data: Float64Array;
  private count: Float64Array;
  private runData: Float64Array | null = null;
  private runCount: Float64Array | null = null;
  private runN = 0;
  private history: { data: Float64Array; count: Float64Array }[] = [];

  // accumulators over the samples of the current window
  private nSample = 0;
  private accCnt: Float64Array;
  private accCntSamp: Float64Array;
  private accMass: Float64Array;
  private accMassSamp: Float64Array;
  private accKE: Float64Array;
  private accDof: Float64Array;
  private accTsamp: Float64Array;
  private accSum: Float64Array[] = [];
  private accSamp: Float64Array[] = [];
  private lastStep = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} ave/grid`;
    if (args.length < 7) throw new StyleError(`${what}: missing argument(s): Nevery Nrepeat Nfreq Nx Ny Nz value1 ...`);
    this.nEvery = parseInt1(args[0], `${what} Nevery`, 1);
    this.nRepeat = parseInt1(args[1], `${what} Nrepeat`, 1);
    this.nFreq = parseInt1(args[2], `${what} Nfreq`, 1);
    this.nx = parseInt1(args[3], `${what} Nx`, 1);
    this.ny = parseInt1(args[4], `${what} Ny`, 1);
    this.nz = parseInt1(args[5], `${what} Nz`, 1);
    if (this.nFreq % this.nEvery !== 0 || this.nRepeat * this.nEvery > this.nFreq) {
      throw new StyleError(`${what}: Inconsistent nevery/nrepeat/pergrid_nfreq values: Nfreq must be a multiple of Nevery and Nrepeat*Nevery must not exceed Nfreq`);
    }
    this.nevery = this.nEvery;
    this.ncell = this.nx * this.ny * this.nz;
    this.adof = sys.dimension;

    if (this.sys.dimension === 2 && this.nz !== 1) throw new StyleError(`${what}: for 2d simulations Nz must be 1`);

    // values, then keywords
    let k = 6;
    const values: string[] = [];
    while (k < args.length && !KEYWORDS.has(args[k])) values.push(args[k++]);
    if (values.length === 0) throw new StyleError(`${what}: at least one input value is required`);
    for (const w of expandWildcards(sys, values, 'peratom')) this.cols.push(this.column(w, what));

    while (k < args.length) {
      const key = args[k];
      const val = args[k + 1];
      if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
      k += 2;
      if (key === 'discard') {
        if (val !== 'yes' && val !== 'no') throw new StyleError(`${what}: discard must be yes or no, got '${val}'`);
        this.discard = val === 'yes';
      } else if (key === 'norm') {
        if (val !== 'all' && val !== 'sample' && val !== 'none') throw new StyleError(`${what}: norm must be all, sample or none, got '${val}'`);
        this.norm = val;
      } else if (key === 'ave') {
        if (val === 'one' || val === 'running') this.aveMode = val;
        else if (val === 'window') {
          this.aveMode = 'window';
          this.windowM = parseInt1(args[k], `${what} ave window M`, 1);
          k++;
        } else throw new StyleError(`${what}: ave must be one, running or window, got '${val}'`);
      } else if (key === 'bias') {
        throw new StyleError(`${what}: keyword bias is not supported by the browser engine`);
      } else if (key === 'adof') {
        this.adof = parseNum(val, `${what} adof`);
      } else if (key === 'cdof') {
        this.cdof = parseNum(val, `${what} cdof`);
      } else throw new StyleError(`${what}: unknown keyword '${key}'`);
    }
    const ncol = this.cols.length;
    this.data = new Float64Array(this.ncell * ncol);
    this.count = new Float64Array(this.ncell);
    this.accCnt = new Float64Array(this.ncell);
    this.accCntSamp = new Float64Array(this.ncell);
    this.accMass = new Float64Array(this.ncell);
    this.accMassSamp = new Float64Array(this.ncell);
    this.accKE = new Float64Array(this.ncell);
    this.accDof = new Float64Array(this.ncell);
    this.accTsamp = new Float64Array(this.ncell);
    for (let j = 0; j < ncol; j++) {
      this.accSum.push(new Float64Array(this.ncell));
      this.accSamp.push(new Float64Array(this.ncell));
    }
  }

  private column(w: string, what: string): Column {
    if ((ATTRS as string[]).includes(w)) return { t: 'attr', name: w as AttrName, text: w };
    if (w === 'temp') return { t: 'temp', text: w };
    if (w === 'density/number' || w === 'density/mass') return { t: 'dens', mass: w === 'density/mass', text: w };
    if (w.includes(':')) throw new StyleError(`${what}: per-grid input ${w} is not supported by the browser engine (no compute or fix defines a grid yet)`);
    if (/^[cfv]_/.test(w)) return { t: 'ref', ref: parseRef(w), text: w };
    throw new StyleError(`${what}: invalid input value '${w}': expected vx, vy, vz, fx, fy, fz, mass, density/number, density/mass, temp, c_ID, c_ID[I], f_ID, f_ID[I] or v_name`);
  }

  /** Output for one grid cell (cell index x fastest) and column (0-based). */
  gridValue(cell: number, col: number): number {
    if (cell < 0 || cell >= this.ncell || col < 0 || col >= this.cols.length) throw new StyleError(`fix ${this.id}: grid index out of range`);
    return this.data[cell * this.cols.length + col];
  }

  /** Number of atoms per grid cell (averaged as the data are). */
  gridCount(cell: number): number {
    if (cell < 0 || cell >= this.ncell) throw new StyleError(`fix ${this.id}: grid index out of range`);
    return this.count[cell];
  }

  setup(): void { this.sampleStep(); }

  endOfStep(): void { this.sampleStep(); }

  private sampleStep(): void {
    const step = this.sys.state.step;
    if (step % this.nEvery !== 0 || step === this.lastStep) return;
    this.lastStep = step;
    const rem = step % this.nFreq;
    const r = rem === 0 ? this.nFreq : rem;
    if (r > this.nFreq - this.nRepeat * this.nEvery) this.sample();
    if (rem !== 0) return;
    if (this.nSample === this.nRepeat) this.produce();
    this.clearWindow();
  }

  private boxVolume(): number {
    const b = this.sys.state.box;
    let v = 1;
    for (let d = 0; d < this.sys.dimension; d++) v *= b.hi[d] - b.lo[d];
    return v;
  }

  /** Cell index along one dimension for a position, or -1 if the atom is dropped (discard no). */
  private cellIndex(x: number, lo: number, len: number, periodic: boolean, n: number): number {
    let f = (x - lo) / len;
    if (periodic) {
      f -= Math.floor(f);
    } else if (f < 0 || f >= 1) {
      if (!this.discard) return -1;
      f = f < 0 ? 0 : 1 - 1e-12;
    }
    return Math.min(n - 1, Math.max(0, Math.floor(f * n)));
  }

  /** One sample: per-cell sums of the group atoms at this step. */
  private sample(): void {
    const s = this.sys.state;
    if (s.box.triclinic) throw new StyleError(`fix ${this.id} ave/grid: triclinic boxes are not supported by the browser engine`);
    const ncell = this.ncell;
    const ncol = this.cols.length;
    const vol = this.boxVolume();
    const cellVol = vol / ncell;
    const cnt = new Float64Array(ncell);
    const sums: Float64Array[] = this.cols.map(() => new Float64Array(ncell));
    const mass = new Float64Array(ncell);
    const ke = new Float64Array(ncell);
    const perAtom: (Float64Array | null)[] = this.cols.map((c) => (c.t === 'ref' ? peratomValues(this.sys, c.ref) : null));
    const mvv2e = s.units.mvv2e;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ix = this.cellIndex(s.x[3 * i], s.box.lo[0], s.box.hi[0] - s.box.lo[0], s.box.periodic[0], this.nx);
      const iy = this.cellIndex(s.x[3 * i + 1], s.box.lo[1], s.box.hi[1] - s.box.lo[1], s.box.periodic[1], this.ny);
      const iz = this.sys.dimension === 2 ? 0 : this.cellIndex(s.x[3 * i + 2], s.box.lo[2], s.box.hi[2] - s.box.lo[2], s.box.periodic[2], this.nz);
      if (ix < 0 || iy < 0 || iz < 0) continue;
      const c = ix + this.nx * (iy + this.ny * iz);
      const m = massOf(s, i);
      cnt[c] += 1;
      mass[c] += m;
      const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
      ke[c] += 0.5 * mvv2e * m * (vx * vx + vy * vy + vz * vz);
      for (let j = 0; j < ncol; j++) {
        const col = this.cols[j];
        let val = 0;
        if (col.t === 'attr') {
          switch (col.name) {
            case 'vx': val = vx; break;
            case 'vy': val = vy; break;
            case 'vz': val = vz; break;
            case 'fx': val = s.f[3 * i]; break;
            case 'fy': val = s.f[3 * i + 1]; break;
            case 'fz': val = s.f[3 * i + 2]; break;
            case 'mass': val = m; break;
          }
        } else if (col.t === 'ref') {
          val = perAtom[j]![i];
        } else {
          continue; // density and temp use cnt, mass and ke
        }
        sums[j][c] += val;
      }
    }
    // fold the sample into the window
    for (let c = 0; c < ncell; c++) {
      const n = cnt[c];
      this.accCnt[c] += n;
      this.accCntSamp[c] += n / cellVol;
      this.accMass[c] += mass[c];
      this.accMassSamp[c] += mass[c] / cellVol;
      const dof = this.cdof + n * this.adof;
      this.accDof[c] += dof;
      this.accKE[c] += ke[c];
      this.accTsamp[c] += dof > 0 && this.sys.state.units.boltz > 0 ? (2 * ke[c]) / (dof * this.sys.state.units.boltz) : 0;
      for (let j = 0; j < ncol; j++) {
        const col = this.cols[j];
        if (col.t === 'dens' || col.t === 'temp') continue;
        this.accSum[j][c] += sums[j][c];
        this.accSamp[j][c] += n > 0 ? sums[j][c] / n : 0;
      }
    }
    this.nSample++;
  }

  private clearWindow(): void {
    this.nSample = 0;
    this.accCnt.fill(0);
    this.accCntSamp.fill(0);
    this.accMass.fill(0);
    this.accMassSamp.fill(0);
    this.accKE.fill(0);
    this.accDof.fill(0);
    this.accTsamp.fill(0);
    for (const a of this.accSum) a.fill(0);
    for (const a of this.accSamp) a.fill(0);
  }

  /** The output on an Nfreq step, then the averaging keyword applied to it. */
  private produce(): void {
    const ncell = this.ncell;
    const ncol = this.cols.length;
    const nr = this.nRepeat;
    const cellVol = this.boxVolume() / ncell;
    const data = new Float64Array(ncell * ncol);
    const count = new Float64Array(ncell);
    for (let c = 0; c < ncell; c++) {
      count[c] = this.accCnt[c] / nr;
      for (let j = 0; j < ncol; j++) {
        const col = this.cols[j];
        let out: number;
        if (col.t === 'dens') {
          const acc = col.mass ? this.accMass[c] : this.accCnt[c];
          const accS = col.mass ? this.accMassSamp[c] : this.accCntSamp[c];
          out = this.norm === 'sample' ? accS / nr : acc / (nr * cellVol);
        } else if (col.t === 'temp') {
          out = this.norm === 'sample' ? this.accTsamp[c] / nr
            : (this.accDof[c] > 0 && this.sys.state.units.boltz > 0 ? (2 * this.accKE[c]) / (this.accDof[c] * this.sys.state.units.boltz) : 0);
        } else if (this.norm === 'all') {
          out = this.accCnt[c] > 0 ? this.accSum[j][c] / this.accCnt[c] : 0;
        } else if (this.norm === 'sample') {
          out = this.accSamp[j][c] / nr;
        } else {
          out = this.accSum[j][c] / nr;
        }
        data[c * ncol + j] = out;
      }
    }
    this.publish(data, count);
  }

  /** Applies ave one / running / window to the produced output. */
  private publish(data: Float64Array, count: Float64Array): void {
    if (this.aveMode === 'one') {
      this.data = data;
      this.count = count;
      return;
    }
    if (this.aveMode === 'running') {
      if (!this.runData || !this.runCount) {
        this.runData = new Float64Array(data.length);
        this.runCount = new Float64Array(count.length);
      }
      for (let k = 0; k < data.length; k++) this.runData[k] += data[k];
      for (let k = 0; k < count.length; k++) this.runCount[k] += count[k];
      this.runN++;
      this.data = this.runData.map((v) => v / this.runN);
      this.count = this.runCount.map((v) => v / this.runN);
      return;
    }
    this.history.push({ data, count });
    if (this.history.length > this.windowM) this.history.shift();
    const n = this.history.length;
    const d = new Float64Array(data.length);
    const c = new Float64Array(count.length);
    for (const h of this.history) {
      for (let k = 0; k < d.length; k++) d[k] += h.data[k];
      for (let k = 0; k < c.length; k++) c[k] += h.count[k];
    }
    for (let k = 0; k < d.length; k++) d[k] /= n;
    for (let k = 0; k < c.length; k++) c[k] /= n;
    this.data = d;
    this.count = c;
  }
}
