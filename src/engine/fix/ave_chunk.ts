import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { expandWildcards, parseRef, peratomValues, type Ref } from '../refs';
import { formatNumber } from '../script';
import { massOf } from '../atoms';
import { chunkCompute, type ComputeChunkAtom } from '../compute/chunk';

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

const KEYWORDS = new Set(['norm', 'ave', 'bias', 'adof', 'cdof', 'file', 'append', 'overwrite', 'format', 'title1', 'title2', 'title3']);

type Val =
  | { t: 'vel'; d: number }
  | { t: 'frc'; d: number }
  | { t: 'mass' }
  | { t: 'dnum' }
  | { t: 'dmass' }
  | { t: 'temp' }
  | { t: 'ref'; ref: Ref };

type Norm = 'all' | 'sample' | 'none';

const posInt = (w: string | undefined, what: string): number => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  const n = Number(w);
  if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what} must be a positive integer, got '${w}'`);
  return n;
};

/** Applies a file format like " %g" / "%20.16g": leading spaces kept, then one C conversion. */
const fmtValue = (v: number, fmt: string, what: string): string => {
  const m = /^(\s*)(%[-+ 0#]*\d*(?:\.\d+)?[feEgGdi])$/.exec(fmt);
  if (!m) throw new StyleError(`${what}: invalid numeric format string '${fmt}'`);
  return m[1] + formatNumber(v, m[2]);
};

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
  private readonly valueWords: string[];
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

  // sums over the samples of the current window (per chunk)
  private cntAll = new Float64Array(0);
  private sumAll: Float64Array[] = [];
  /** Per value: sum of per-sample values (norm sample/none) or per-sample temperature. */
  private acc: Float64Array[] = [];
  private ke2All = new Float64Array(0);
  private dofAll = new Float64Array(0);
  private nSample = 0;
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
    const what = `fix ${id} (ave/chunk)`;
    if (args.length < 5) throw new StyleError(`usage: fix ${id} group-ID ave/chunk Nevery Nrepeat Nfreq chunkID value1 ... [keyword args ...]`);
    this.nEvery = posInt(args[0], `${what}: Nevery`);
    this.nRepeat = posInt(args[1], `${what}: Nrepeat`);
    this.nFreq = posInt(args[2], `${what}: Nfreq`);
    this.nevery = this.nEvery;
    if (this.nFreq % this.nEvery !== 0) throw new StyleError(`${what}: Nfreq must be a multiple of Nevery`);
    if (this.nRepeat * this.nEvery > this.nFreq) {
      throw new StyleError(`${what}: Nrepeat x Nevery cannot exceed Nfreq (the time steps contributing to the average cannot overlap)`);
    }
    this.chunkId = args[3];
    const words: string[] = [];
    let k = 4;
    while (k < args.length && !KEYWORDS.has(args[k])) words.push(args[k++]);
    if (!words.length) throw new StyleError(`${what}: no input values were given`);
    this.valueWords = words;
    for (; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      const need = (): string => {
        if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
        return val;
      };
      switch (key) {
        case 'norm': {
          const v = need();
          if (v !== 'all' && v !== 'sample' && v !== 'none') throw new StyleError(`${what}: norm must be all, sample or none, got '${v}'`);
          this.norm = v;
          k += 2;
          break;
        }
        case 'ave': {
          const v = need();
          if (v === 'one' || v === 'running') { this.aveMode = v; k += 2; break; }
          if (v === 'window') {
            this.aveMode = 'window';
            this.windowM = posInt(args[k + 2], `${what}: window M`);
            k += 3;
            break;
          }
          throw new StyleError(`${what}: ave must be one, running or window M, got '${v}'`);
        }
        case 'bias':
          throw new StyleError(`${what}: the bias keyword is not supported by the engine yet`);
        case 'adof':
          this.adof = Number(need());
          if (!Number.isFinite(this.adof)) throw new StyleError(`${what}: adof must be a number`);
          k += 2;
          break;
        case 'cdof':
          this.cdof = Number(need());
          if (!Number.isFinite(this.cdof)) throw new StyleError(`${what}: cdof must be a number`);
          k += 2;
          break;
        case 'file':
        case 'append':
          if (this.fileName !== null) throw new StyleError(`${what}: file and append cannot both be used`);
          this.fileName = need();
          this.fileAppend = key === 'append';
          k += 2;
          break;
        case 'overwrite':
          this.overwriteFile = true;
          k += 1;
          break;
        case 'format':
          this.fmt = need();
          fmtValue(0, this.fmt, what);
          k += 2;
          break;
        case 'title1': this.title1 = need(); k += 2; break;
        case 'title2': this.title2 = need(); k += 2; break;
        case 'title3': this.title3 = need(); k += 2; break;
        default:
          throw new StyleError(`${what}: unknown keyword '${key}'`);
      }
    }
    if (this.overwriteFile && this.aveMode !== 'running') {
      throw new StyleError(`${what}: the overwrite keyword can only be used with the ave running setting`);
    }
  }

  /** Resolves the chunk compute and the value list (wildcards need the referenced computes). */
  init(): void {
    const what = `fix ${this.id} (ave/chunk)`;
    this.chunk = chunkCompute(this.sys, this.chunkId, what);
    this.words = expandWildcards(this.sys, this.valueWords, 'peratom');
    this.vals = this.words.map((w): Val => {
      switch (w) {
        case 'vx': return { t: 'vel', d: 0 };
        case 'vy': return { t: 'vel', d: 1 };
        case 'vz': return { t: 'vel', d: 2 };
        case 'fx': return { t: 'frc', d: 0 };
        case 'fy': return { t: 'frc', d: 1 };
        case 'fz': return { t: 'frc', d: 2 };
        case 'mass': return { t: 'mass' };
        case 'density/number': return { t: 'dnum' };
        case 'density/mass': return { t: 'dmass' };
        case 'temp': return { t: 'temp' };
        default: {
          const r = parseRef(w);
          if (r.kind === 'attr') throw new StyleError(`${what}: unknown value '${w}'`);
          return { t: 'ref', ref: r };
        }
      }
    });
    this.wantTemp = this.vals.some((v) => v.t === 'temp');
    // ave running / window: Nchunk must stay constant for the whole run (docs)
    if (this.aveMode !== 'one') this.chunk.holdNchunk();
    this.clearWindow();
    this.runSum = [];
    this.runN = 0;
    this.runTotal = 0;
    this.ring = [];
    this.table = [];
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

  /** Volume for density values: the bin volume for binning chunks, else the box. */
  private volume(): number {
    return this.chunk.kind === 'bin' ? this.chunk.binVolume() : this.boxVolume();
  }

  setup(): void { this.endOfStep(); }

  endOfStep(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    const rem = step % this.nFreq;
    const nextOut = rem === 0 ? step : step + (this.nFreq - rem);
    if (step < nextOut - (this.nRepeat - 1) * this.nEvery) return; // not in the next output's window
    this.sample();
    if (step % this.nFreq !== 0) return;
    if (this.nSample === this.nRepeat) this.produce(step);
    else this.clearWindow();
  }

  /** One sample: per-chunk sums over the group atoms at this step. */
  private sample(): void {
    const s = this.sys.state;
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
      if (this.wantTemp) {
        const vx = s.v[3 * i], vy = s.v[3 * i + 1], vz = s.v[3 * i + 2];
        ke2[c - 1] += m * (vx * vx + vy * vy + vz * vz);
      }
    }
    const adof = this.adof ?? this.sys.dimension;
    const vol = this.volume();
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
          if (this.norm === 'all') this.sumAll[j][c] += sums[j][c];
          else this.acc[j][c] += sums[j][c] / vol;
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
          x = this.norm === 'all' ? sumAll / (nr * vol) : acc / nr;
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
    const what = `fix ${this.id} (ave/chunk)`;
    // measured with native LAMMPS: the OrigID column is an integer, the coordinates and
    // the Ncount column are always printed with " %g", and only the value columns use *format*
    const fixed = this.nFixed;
    const origN = this.chunk.origIds ? 1 : 0;
    this.table.forEach((row, c) => {
      let line = String(c + 1);
      row.forEach((x, k) => {
        if (k < origN) line += ` ${x}`;
        else if (k <= fixed) line += fmtValue(x, ' %g', what); // coordinates, then Ncount
        else line += fmtValue(x, this.fmt, what);
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
