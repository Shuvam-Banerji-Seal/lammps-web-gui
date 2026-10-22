import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { peratomValues, type Ref } from '../refs';
import { massOf } from '../atoms';
import { inumeric, missingArgs, numeric } from '../force/util';
import { argInfo, expandArgs } from '../argInfo';
import { checkCFormat, compileCFormat } from '../cformat';
import { errorUrl } from '../formula';
import { aveNextValid } from './util';
import { ComputeChunkAtom } from '../compute/chunk';
import type { Compute } from '../compute/compute';

/*
 * fix ave/chunk — implemented from docs.lammps.org/fix_ave_chunk.html only
 * (never from LAMMPS source). Quotations are copied from
 * plans/lammps-docs/fix_ave_chunk.rst, one source line per fragment.
 *
 *   "fix ID group-ID ave/chunk Nevery Nrepeat Nfreq chunkID value1 value2 ... keyword args ..."
 *   "Use one or more per-atom vectors as inputs every few timesteps, sum the values over the atoms in each chunk at each timestep, then average the per-chunk values over longer timescales."
 *   "Note that only atoms in the specified group contribute to the summing and averaging calculations."
 *   density/number: "The density/number value means the number density is computed for each chunk (i.e., number/volume)."
 *   volume: "the volume is the bin volume." and "Otherwise, it is the volume of the entire simulation box."
 *   temp: "By default, adof = 2 or 3 = dimensionality of system," and "cdof = 0.0."
 *   norm all: "as is the count of atoms in the chunk." and "Total-sum / Total-count."
 *   norm sample: "In other words, it is an average of an average."
 *   norm none: "A summed sample value is simply the chunk value summed over atoms in the sample, without dividing by the number of atoms in the sample."
 *   ave running: "Each output chunk value is thus the average of the chunk value produced on that timestep with all preceding values for the same chunk."
 *   ave window: "values for the same chunk are used to produce the output."
 *
 * Measured with native LAMMPS (black box, probes under /tmp/haiku-chunk):
 *   - the header is three lines: Chunk-averaged data for fix ID and group GROUP;
 *     Timestep Number-of-chunks Total-count; Chunk, then OrigID if compress is
 *     set, then Coord1.. for bins, then Ncount and the value names. Each output
 *     section starts with a line "step Nchunk Total-count", then one line per chunk.
 *   - Ncount is the average atom count over the Nrepeat samples (all norms).
 *   - Total-count is the number of group atoms in chunks at the last sample;
 *     under ave running / window it is summed over the outputs averaged.
 *   - norm all density = (sum of masses over samples) / (Nrepeat * volume);
 *     norm sample / none density = average over samples of mass / volume.
 *   - norm all temp = sum(m v^2) / sum(DOF) (times mvv2e/boltz); norm sample /
 *     none temp = average of the per-sample temperatures.
 *   - coordinates and Ncount are printed with " %g" whatever the format keyword
 *     says; only the value columns follow *format*.
 *   - an empty chunk outputs 0 for every value.
 */

/** The values of fix ave/chunk that are not references (native's V, F, MASS, DENSITY and TEMPERATURE kinds). */
const CHUNK_VALUES: Record<string, Val | undefined> = {
  vx: { t: 'vel', d: 0 }, vy: { t: 'vel', d: 1 }, vz: { t: 'vel', d: 2 },
  fx: { t: 'frc', d: 0 }, fy: { t: 'frc', d: 1 }, fz: { t: 'frc', d: 2 },
  mass: { t: 'mass' }, 'density/number': { t: 'dnum' }, 'density/mass': { t: 'dmass' }, temp: { t: 'temp' },
};

type Val =
  | { t: 'vel'; d: number }
  | { t: 'frc'; d: number }
  | { t: 'mass' }
  | { t: 'dnum' }
  | { t: 'dmass' }
  | { t: 'temp' }
  | { t: 'ref'; ref: Ref };

type Norm = 'all' | 'sample' | 'none';

/** Grows a per-chunk accumulator to n entries (zero-filled). */
const grow = (a: Float64Array, n: number): Float64Array<ArrayBuffer> => {
  if (a.length >= n && a.buffer instanceof ArrayBuffer) return a as Float64Array<ArrayBuffer>;
  const b = new Float64Array(n);
  b.set(a);
  return b;
};

export class FixAveChunk extends Fix {
  readonly style = 'ave/chunk';
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private readonly chunkId: string;
  private norm: Norm = 'all';
  private aveMode: 'one' | 'running' | 'window' = 'one';
  private windowM = 1;
  private adof: number | null = null;
  private cdof = 0;
  private fileName: string | null = null;
  private fileAppend = false;
  private overwriteFile = false;
  private fmt = ' %g';
  private title1: string | undefined;
  private title2: string | undefined;
  private title3: string | undefined;

  private chunk!: ComputeChunkAtom;
  private words: string[] = [];
  private vals: Val[] = [];
  private wantTemp = false;
  /**
   * bias keyword (docs.lammps.org/fix_ave_chunk.html): "The bias keyword specifies the ID of a temperature compute that
   * removes a "bias" velocity from each atom, specified as bias-ID. It is only used when the temp value is calculated".
   */
  private biasId: string | null = null;
  private bias: Compute | null = null;

  // sums over the samples of the current window (per chunk)
  private cntAll = new Float64Array(0);
  private sumAll: Float64Array[] = [];
  /** Per value: sum of per-sample values (norm sample/none) or per-sample temperature. */
  private acc: Float64Array[] = [];
  private ke2All = new Float64Array(0);
  private dofAll = new Float64Array(0);
  private nSample = 0;
  /** The next step that samples, and the samples taken of the current window (native nvalid and irepeat). */
  private nvalid: number;
  private irepeat = 0;
  private totalLast = 0;

  // outputs for ave running / window: the averaged part of each chunk row
  private runSum: number[][] = [];
  private runN = 0;
  private runTotal = 0;
  private ring: { rows: number[][]; total: number }[] = [];
  /** Current output table: per chunk [OrigID?, Coord..., Ncount, values...]. */
  private table: number[][] = [];
  private nFixed = 0;
  private header: string[] | null = null;
  private body: string[] = [];
  private headerWritten = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // the global array (rows = chunks) is available as f_ID[i][j] once output exists
    this.arrayFlag = true;
    // As native LAMMPS does (src/fix_ave_chunk.cpp constructor at b332f28): the words after the chunk ID are expanded,
    // the values are read up to the first word that is none, then the keywords, then the step counts and the sources
    // are checked
    if (args.length < 4) throw missingArgs('fix ave/chunk');
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    this.chunkId = args[3];
    // fix_ave_chunk.cpp: global_freq = nfreq, and the number of rows (chunks) varies (size_array_rows_variable)
    this.globalFreq = nfreq;
    this.sizeArrayRowsVariable = true;
    const words = expandArgs(sys, args.slice(4), 1).words;
    let k = 0;
    for (; k < words.length; k++) {
      const w = words[k];
      const fixed = CHUNK_VALUES[w];
      if (fixed) { this.vals.push(fixed); continue; }
      const a = argInfo(w);
      if (a.which === 'none') break;
      if (a.which === 'unknown' || a.dim > 1) throw new StyleError(`Unknown fix ave/chunk data value: ${w}`);
      if (a.index1 < 0) throw new StyleError(`fix ave/chunk: index ${a.index1} in ${w} is refused: native LAMMPS reads outside the array for it`);
      this.vals.push({ t: 'ref', ref: { kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: w } });
    }
    this.words = words.slice(0, k);
    if (!this.vals.length) throw new StyleError('No values from computes, fixes, or variables used in fix ave/chunk command');
    const need = (n: number, what: string): void => { if (k + n > words.length) throw missingArgs(what); };
    while (k < words.length) {
      const key = words[k];
      const v = words[k + 1];
      if (key === 'norm') {
        need(2, 'fix ave/chunk norm');
        if (v !== 'all' && v !== 'sample' && v !== 'none') throw new StyleError(`Unknown fix ave/chunk norm mode: ${v}`);
        this.norm = v;
        k += 2;
      } else if (key === 'ave') {
        need(2, 'fix ave/chunk ave');
        if (v !== 'one' && v !== 'running' && v !== 'window') throw new StyleError(`Unknown fix ave/chunk ave mode: ${v}`);
        this.aveMode = v;
        if (v === 'window') {
          need(3, 'fix ave/chunk ave window');
          this.windowM = inumeric(words[k + 2]);
          if (this.windowM <= 0) throw new StyleError(`Illegal fix ave/chunk number of windows: ${this.windowM}`);
        }
        k += this.aveMode === 'window' ? 3 : 2;
      } else if (key === 'bias') {
        need(2, 'fix ave/chunk bias');
        this.biasId = v;
        k += 2;
      } else if (key === 'adof') {
        need(2, 'fix ave/chunk adof');
        this.adof = numeric(v);
        k += 2;
      } else if (key === 'cdof') {
        need(2, 'fix ave/chunk cdof');
        this.cdof = numeric(v);
        k += 2;
      } else if (key === 'file' || key === 'append') {
        need(2, `fix ave/chunk ${key}`);
        this.fileName = v;
        this.fileAppend = key === 'append';
        k += 2;
      } else if (key === 'overwrite') {
        this.overwriteFile = true;
        k += 1;
      } else if (key === 'format') {
        need(2, 'fix ave/chunk format');
        checkCFormat(v, 'float', 'Invalid fix ave/chunk format argument');
        this.fmt = v;
        k += 2;
      } else if (key === 'title1' || key === 'title2' || key === 'title3') {
        need(2, `fix ave/chunk ${key}`);
        if (key === 'title1') this.title1 = v;
        else if (key === 'title2') this.title2 = v;
        else this.title3 = v;
        k += 2;
      } else throw new StyleError(`Unknown fix ave/chunk keyword: ${key}`);
    }
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/chunk nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal fix ave/chunk nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/chunk nfreq value: ${nfreq}`);
    if (nfreq % nevery || nrepeat * nevery > nfreq) throw new StyleError('Inconsistent fix ave/chunk nevery/nrepeat/nfreq values');
    if (this.aveMode !== 'running' && this.overwriteFile) throw new StyleError('Fix ave/chunk overwrite keyword requires ave running setting');
    this.nEvery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    this.nevery = nevery;
    if (this.biasId !== null) {
      const c = sys.computes.find((x) => x.id === this.biasId);
      if (!c) throw new StyleError(`Could not find compute ID ${this.biasId} for temperature bias`);
      if (!c.tempFlag) throw new StyleError(`Bias compute ${this.biasId} does not calculate temperature`);
      if (!c.hasBias()) throw new StyleError(`Bias compute ${this.biasId} does not calculate a velocity bias`);
    }
    for (const v of this.vals) {
      if (v.t !== 'ref') continue;
      const r = v.ref;
      if (r.kind === 'c' || r.kind === 'f') {
        const what = r.kind === 'c' ? 'compute' : 'fix';
        const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id) : sys.fixes.find((x) => x.id === r.id);
        if (!o) throw new StyleError(`${r.kind === 'c' ? 'Compute' : 'Fix'} ID ${r.id} for fix ave/chunk does not exist`);
        if (!o.peratomFlag) throw new StyleError(`Fix ave/chunk ${what} ${r.id} does not calculate per-atom values`);
        if (r.index === null && o.sizePeratomCols !== 0) throw new StyleError(`Fix ave/chunk ${what} ${r.id} does not calculate a per-atom vector`);
        if (r.index !== null && o.sizePeratomCols === 0) throw new StyleError(`Fix ave/chunk ${what} ${r.id} does not calculate a per-atom array`);
        if (r.index !== null && r.index > o.sizePeratomCols) throw new StyleError(`Fix ave/chunk ${what} ${r.id} vector is accessed out-of-range${errorUrl(20)}`);
      } else {
        const vv = sys.vars.get(r.id);
        if (!vv) throw new StyleError(`Variable name ${r.id} for fix ave/chunk does not exist`);
        if (vv.style !== 'atom') throw new StyleError(`Fix ave/chunk variable ${r.id} is not atom-style variable`);
      }
    }
    if (!(sys.computes.find((x) => x.id === this.chunkId) instanceof ComputeChunkAtom)) {
      throw new StyleError(`Chunk/atom compute ${this.chunkId} does not exist or is incorrect style for fix ave/chunk`);
    }
    this.wantTemp = this.vals.some((v) => v.t === 'temp');
    this.nvalid = aveNextValid(sys.state.step, this.nEvery, this.nRepeat, this.nFreq);
  }

  /** The chunk compute and the sources must still exist (native init); the bias compute is resolved. */
  init(): void {
    const c = this.sys.computes.find((x) => x.id === this.chunkId);
    if (!(c instanceof ComputeChunkAtom)) throw new StyleError(`Chunk/atom compute ${this.chunkId} does not exist or is incorrect style for fix ave/chunk`);
    this.chunk = c;
    this.bias = null;
    if (this.biasId !== null) {
      const b = this.sys.computes.find((x) => x.id === this.biasId);
      if (!b) throw new StyleError(`Could not find compute ID ${this.biasId} for temperature bias`);
      this.bias = b;
    }
    for (const v of this.vals) {
      if (v.t !== 'ref') continue;
      const r = v.ref;
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ave/chunk does not exist`);
      if (r.kind === 'f') {
        const f = this.sys.fixes.find((x) => x.id === r.id);
        if (!f) throw new StyleError(`Fix ID ${r.id} for fix ave/chunk does not exist`);
        if (f.peratomFreq > 0 && this.nEvery % f.peratomFreq) throw new StyleError(`Fix ${r.id} for fix ave/chunk not computed at compatible time${errorUrl(7)}`);
      }
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/chunk does not exist`);
    }
    // fix_ave_chunk.cpp: size_array_cols = colextra + 1 + nvalues (the compressed chunk ID, the bin coordinates, the
    // count, then the values), known before the first output
    this.sizeArrayCols = (this.chunk.compress ? 1 : 0) + (this.chunk.kind === 'bin' ? this.chunk.binDims : 0) + 1 + this.vals.length;
    // ave running / window: Nchunk must stay constant for the whole run (docs)
    if (this.aveMode !== 'one') this.chunk.holdNchunk();
    // The window (samples taken since the last output) and the running and window averages carry over a run
    // boundary, as native's init keeps them (src/fix_ave_chunk.cpp at b332f28); only a change of the value list
    // restarts them.
    if (this.sumAll.length !== this.vals.length || this.acc.length !== this.vals.length) {
      this.clearWindow();
      this.runSum = [];
      this.runN = 0;
      this.runTotal = 0;
      this.ring = [];
      this.table = [];
    }
    // a minimization past the next sample step restarts the window (native init)
    const step = this.sys.state.step;
    if (this.nvalid < step) {
      this.irepeat = 0;
      this.nvalid = aveNextValid(step, this.nEvery, this.nRepeat, this.nFreq);
    }
  }

  private clearWindow(): void {
    this.cntAll = new Float64Array(0);
    this.sumAll = this.vals.map(() => new Float64Array(0));
    this.acc = this.vals.map(() => new Float64Array(0));
    this.ke2All = new Float64Array(0);
    this.dofAll = new Float64Array(0);
    this.nSample = 0;
    this.totalLast = 0;
  }

  /** Box volume (3d) or area (2d), for values of non-binning chunks. */
  private boxVolume(): number {
    const b = this.sys.state.box;
    let v = 1;
    for (let d = 0; d < this.sys.dimension; d++) v *= b.hi[d] - b.lo[d];
    return v;
  }

  /** Volume of chunk c (0-based) for density values: the bin volume for binning chunks, else the box. */
  private volume(): (c: number) => number {
    if (this.chunk.kind === 'bin') return this.chunk.chunkVolumes();
    const v = this.boxVolume();
    return () => v;
  }

  setup(): void { this.endOfStep(); }

  /**
   * unfix: as native LAMMPS does (src/fix_ave_chunk.cpp at b332f28, the destructor), the lock of ave running and
   * window is released; an ave one window cut short by unfix keeps its lock.
   */
  destroy(): void {
    if (this.aveMode !== 'one') this.chunk.unlock(this);
  }

  /**
   * As native LAMMPS does (src/fix_ave_chunk.cpp end_of_step at b332f28): only the scheduled step samples; the first
   * sample of a window zeroes the sums (and sets up and locks the chunks), the Nrepeat-th averages them and schedules
   * the next window to end Nfreq steps later.
   */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    if (this.irepeat === 0) this.clearWindow();
    this.sample();
    this.irepeat++;
    if (this.irepeat < this.nRepeat) {
      this.nvalid += this.nEvery;
      return;
    }
    this.irepeat = 0;
    this.nvalid = step + this.nFreq - (this.nRepeat - 1) * this.nEvery;
    if (this.nRepeat > 1 && this.aveMode === 'one') this.chunk.unlock(this);
    this.produce(step);
  }

  /** ave running / window: the chunk/atom was locked for good on the first sample. */
  private lockedForever = false;

  /** One sample: per-chunk sums over the group atoms at this step. */
  private sample(): void {
    const s = this.sys.state;
    if (this.nSample === 0) {
      // As native LAMMPS does (src/fix_ave_chunk.cpp at b332f28): the first sample of a window sets up the chunks,
      // then locks the chunk/atom until the window's last sample (Nrepeat > 1, ave one), or for good (ave running /
      // window)
      this.chunk.ensure();
      if (this.nRepeat > 1 && this.aveMode === 'one') this.chunk.lock(this, s.step, s.step + (this.nRepeat - 1) * this.nEvery);
      else if (this.aveMode !== 'one' && !this.lockedForever) {
        this.chunk.lock(this, s.step, -1);
        this.lockedForever = true;
      }
    }
    this.chunk.ensure();
    const N = this.chunk.nchunk;
    const ids = this.chunk.ids;
    const cnt = new Float64Array(N);
    const sums = this.vals.map(() => new Float64Array(N));
    const ke2 = new Float64Array(N);
    const perAtom = this.vals.map((v) => (v.t === 'ref' ? peratomValues(this.sys, v.ref) : null));
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const c = ids[i];
      if (c <= 0 || c > N) continue;
      const m = massOf(s, i);
      cnt[c - 1] += 1;
      for (let j = 0; j < this.vals.length; j++) {
        const v = this.vals[j];
        let x = 0;
        switch (v.t) {
          case 'vel': x = s.v[3 * i + v.d]; break;
          case 'frc': x = s.f[3 * i + v.d]; break;
          case 'mass': case 'dmass': x = m; break;
          case 'dnum': x = 1; break;
          case 'ref': x = perAtom[j]![i]; break;
          case 'temp': continue;
        }
        sums[j][c - 1] += x;
      }
    }
    if (this.wantTemp) {
      // the kinetic energy of the temp value, with the velocity bias of the bias compute taken out while it is summed
      if (this.bias) { this.bias.computeBias(); this.bias.removeBiasAll(); }
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const c = ids[i];
        if (c <= 0 || c > N) continue;
        const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
        ke2[c - 1] += massOf(s, i) * (vx * vx + vy * vy + vz * vz);
      }
      if (this.bias) this.bias.restoreBiasAll();
    }
    const adof = this.adof ?? this.sys.dimension;
    const vol = this.volume();
    const mv2d = s.units.mv2d;
    this.totalLast = 0;
    this.cntAll = grow(this.cntAll, N);
    this.ke2All = grow(this.ke2All, N);
    this.dofAll = grow(this.dofAll, N);
    for (let c = 0; c < N; c++) {
      this.totalLast += cnt[c];
      this.cntAll[c] += cnt[c];
    }
    this.vals.forEach((v, j) => {
      this.sumAll[j] = grow(this.sumAll[j], N);
      this.acc[j] = grow(this.acc[j], N);
      for (let c = 0; c < N; c++) {
        if (v.t === 'temp') {
          if (this.norm === 'all') continue;
          const dof = cnt[c] * adof + this.cdof;
          this.acc[j][c] += this.tempOf(ke2[c], dof);
        } else if (v.t === 'dnum' || v.t === 'dmass') {
          // As native LAMMPS does (src/fix_ave_chunk.cpp at b332f28): a mass density is converted by mv2d
          if (this.norm === 'all') this.sumAll[j][c] += sums[j][c];
          else if (v.t === 'dnum') this.acc[j][c] += sums[j][c] / vol(c);
          else this.acc[j][c] += mv2d * (sums[j][c] / vol(c));
        } else if (this.norm === 'all') {
          this.sumAll[j][c] += sums[j][c];
        } else if (this.norm === 'sample') {
          this.acc[j][c] += cnt[c] > 0 ? sums[j][c] / cnt[c] : 0;
        } else {
          this.acc[j][c] += sums[j][c];
        }
      }
    });
    if (this.wantTemp && this.norm === 'all') {
      for (let c = 0; c < N; c++) {
        this.ke2All[c] += ke2[c];
        this.dofAll[c] += cnt[c] * adof + this.cdof;
      }
    }
    this.nSample++;
  }

  /** Temperature from sum(m v^2) and DOF: T = mvv2e * sum(m v^2) / (DOF * boltz). */
  private tempOf(ke2: number, dof: number): number {
    const u = this.sys.state.units;
    return dof > 0 ? (u.mvv2e * ke2) / (dof * u.boltz) : 0;
  }

  /** Averages the samples into the output table, then applies ave one / running / window. */
  private produce(step: number): void {
    const N = this.chunk.nchunk;
    const nr = this.nRepeat;
    const vol = this.volume();
    const u = this.sys.state.units;
    const origCol = this.chunk.origIds ? 1 : 0;
    const binDims = this.chunk.kind === 'bin' ? this.chunk.binDims : 0;
    this.nFixed = origCol + binDims;
    const cur: number[][] = [];
    for (let c = 0; c < N; c++) {
      const row: number[] = [];
      if (origCol) row.push(this.chunk.origOf(c + 1));
      if (binDims) row.push(...this.chunk.coordsOf(c + 1));
      row.push((this.cntAll[c] ?? 0) / nr);
      this.vals.forEach((v, j) => {
        const sumAll = this.sumAll[j][c] ?? 0;
        const acc = this.acc[j][c] ?? 0;
        const cntTot = this.cntAll[c] ?? 0;
        let x: number;
        if (v.t === 'temp') {
          if (this.norm === 'all') {
            const dof = this.dofAll[c] ?? 0;
            x = dof > 0 ? (u.mvv2e * (this.ke2All[c] ?? 0)) / (dof * u.boltz) : 0;
          } else x = acc / nr;
        } else if (v.t === 'dnum' || v.t === 'dmass') {
          if (this.norm !== 'all') x = acc / nr;
          else if (v.t === 'dnum') x = sumAll / vol(c) / nr;
          else x = (sumAll / vol(c)) * (u.mv2d / nr);
        } else if (this.norm === 'all') {
          x = cntTot > 0 ? sumAll / cntTot : 0;
        } else {
          x = acc / nr;
        }
        row.push(x);
      });
      cur.push(row);
    }
    const total = this.totalLast;
    let table = cur;
    let outTotal = total;
    if (this.aveMode === 'running') {
      this.runN++;
      this.runTotal += total;
      this.runSum = addAvgParts(this.runSum, cur, this.nFixed);
      table = cur.map((row, c) => [...row.slice(0, this.nFixed), ...(this.runSum[c] ?? []).map((x) => x / this.runN)]);
      outTotal = this.runTotal;
    } else if (this.aveMode === 'window') {
      this.ring.push({ rows: cur.map((row) => row.slice(this.nFixed)), total });
      if (this.ring.length > this.windowM) this.ring.shift();
      outTotal = this.ring.reduce((a, e) => a + e.total, 0);
      table = cur.map((row, c) => [
        ...row.slice(0, this.nFixed),
        ...row.slice(this.nFixed).map((_x, k) => this.ringMean(c, k)),
      ]);
    }
    this.table = table;
    this.sizeArrayRows = table.length;
    this.sizeArrayCols = table[0]?.length ?? 0;
    if (this.fileName) this.writeSection(step, outTotal);
    this.clearWindow();
  }

  /** Mean over the window entries (chunks missing from an entry count as 0). */
  private ringMean(c: number, k: number): number {
    let sum = 0;
    for (const e of this.ring) sum += e.rows[c]?.[k] ?? 0;
    return sum / this.ring.length;
  }

  private buildHeader(): string[] {
    const out = [this.title1 ?? `# Chunk-averaged data for fix ${this.id} and group ${this.group}`];
    out.push(this.title2 ?? '# Timestep Number-of-chunks Total-count');
    let head = '# Chunk';
    if (this.chunk.origIds) head += ' OrigID';
    if (this.chunk.kind === 'bin') for (let d = 1; d <= this.chunk.binDims; d++) head += ` Coord${d}`;
    head += ` Ncount ${this.words.join(' ')}`;
    out.push(this.title3 ?? head);
    return out;
  }

  private writeSection(step: number, total: number): void {
    if (this.header === null) this.header = this.buildHeader();
    const sec: string[] = [`${step} ${this.table.length} ${total}`];
    // measured with native LAMMPS: the OrigID column is an integer, the coordinates and
    // the Ncount column are always printed with " %g", and only the value columns use *format*
    const fixed = this.nFixed;
    const origN = this.chunk.origIds ? 1 : 0;
    this.table.forEach((row, c) => {
      let line = String(c + 1);
      row.forEach((x, k) => {
        if (k < origN) line += ` ${x}`;
        else if (k <= fixed) line += compileCFormat(' %g', 'float')(x); // coordinates, then Ncount
        else line += compileCFormat(this.fmt, 'float', 'Invalid fix ave/chunk format argument')(x);
      });
      sec.push(line);
    });
    if (this.overwriteFile) {
      this.sys.writeFile(this.fileName!, [...this.header, ...sec].join('\n') + '\n', false);
    } else if (this.fileAppend) {
      const head = this.headerWritten ? '' : this.header.join('\n') + '\n';
      this.headerWritten = true;
      this.sys.writeFile(this.fileName!, head + sec.join('\n') + '\n', true);
    } else {
      this.body.push(...sec);
      this.sys.writeFile(this.fileName!, [...this.header, ...this.body].join('\n') + '\n', false);
    }
  }

  /** f_ID[i][j]: row = chunk (0-based), column = [OrigID], [Coord...], Ncount, values. */
  computeArray(i: number, j: number): number {
    return this.table[i]?.[j] ?? 0;
  }
}

/** Adds the averaged parts (columns from `fixed` on) of each chunk row into the running sums. */
const addAvgParts = (sum: number[][], cur: number[][], fixed: number): number[][] => {
  const out: number[][] = [];
  const nc = Math.max(sum.length, cur.length);
  for (let c = 0; c < nc; c++) {
    const a = sum[c] ?? [];
    const b = (cur[c] ?? []).slice(fixed);
    const len = Math.max(a.length, b.length);
    const row: number[] = [];
    for (let k = 0; k < len; k++) row.push((a[k] ?? 0) + (b[k] ?? 0));
    out.push(row);
  }
  return out;
};
