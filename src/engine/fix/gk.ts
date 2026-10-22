import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { globalScalar, type Ref } from '../refs';
import { inumeric, missingArgs, numeric } from '../force/util';
import { argInfo, EQUAL_STYLES, expandArgs } from '../argInfo';
import { errorUrl } from '../formula';
import { formatNumber } from '../script';

/*
 * fix ave/correlate — implemented only from docs.lammps.org/fix_ave_correlate.html
 * (source plans/lammps-docs/fix_ave_correlate.rst). Every quoted fragment below is
 * copied character for character from that file.
 *
 * Syntax: "fix ID group-ID ave/correlate Nevery Nrepeat Nfreq value1 value2 ... keyword args ..."
 *
 * "The input values must be all scalars." "The :math:`N_\text{every}`, :math:`N_\text{repeat}`, and :math:`N_\text{freq}`" arguments
 * "specify on what timesteps the input values will be used to calculate
 * correlation data."
 * ":math:`N_\text{freq}` must be a multiple of :math:`N_\text{every}`;" and
 * ":math:`N_\text{every}` and :math:`N_\text{repeat}` must be non-zero." Also
 * "Also, if the *ave* keyword is set to *one* which is the default, then"
 * ":math:`N_\text{freq} \ge (N_\text{repeat} -1) N_\text{every}` is required."
 *
 * Correlation: C_ij(dt) = <V_i(t) V_j(t+dt)> ("Note that the second value
 * :math:`V_j` in the pair is always the one sampled at the later time.").
 * "Let :math:`S_{ij}` be a set of time correlation data for input values" ...
 * "As explained below, these data are output as one column of a global" "array".
 *
 * type keyword pair lists (quoted from the page, in output column order):
 *   auto: "C_{11}, C_{22}, \dotsc, C_{NN}" (N_pair = N)
 *   upper: "C_{12}, C_{13}, \dotsc, C_{1N}, C_{23}, \dotsc, C_{2N}," ... (N(N-1)/2)
 *   lower: "C_{21}, C_{31}, C_{32}, C_{41}, C_{42}, C_{43I}, \dotsc," (N(N-1)/2)
 *   auto/upper: "C_{11}, C_{12}, C_{13}, \dotsc, C_{1N}, C_{22}, C_{23}," (N(N+1)/2)
 *   auto/lower: "C_{11}, C_{21}, C_{22}, C_{31}, C_{32}, C_{33}, C_{41}," (N(N+1)/2)
 *   full: "C_{11}, C_{12}, \dotsc, C_{1N}, C_{21}, C_{22}, \dotsc, C_{2N}," (N^2)
 *   first: "C_{11}, C_{12}, \dotsc, C_{1N}" (N)
 *
 * Sampling: "The input values are sampled every :math:`N_\text{every}`" time steps and
 * the correlation data for the preceding samples is computed on time steps that
 * are a multiple of Nfreq. Measured with native LAMMPS (black box): a pair of
 * samples (a, b) contributes to the output at step T (a multiple of Nfreq) when
 * both a and b lie in the window [T_prev, T], where T_prev is the previous output
 * step (the first window starts at the first sample). With ave one the window
 * restarts at T, so the sample at T contributes to the next window as well
 * ("The exception is that the") and pairs with a < T are dropped. With ave running
 * every pair since the first sample is kept.
 *
 * "The *ave* keyword determines what happens to the accumulation of correlation"
 * "samples every :math:`N_\text{freq}` timesteps.  If the *ave* setting is *one*,"
 * "then the accumulation is restarted or zeroed every :math:`N_\text{freq}`"
 * "If the *ave* setting is *running*, then the accumulation is never zeroed."
 * "The *start* keyword specifies what time step the accumulation of"
 * "correlation samples will begin on.  The default is step 0."
 * "The *prefactor* keyword specifies a constant which will be used as a multiplier"
 * "on the correlation data after it is averaged."
 *
 * Output array: "The global array has # of rows" = N_repeat and "# of columns"
 * = N_pair + 2; column 1 is the time delta (in steps), column 2 the number of
 * samples, then the pairs. "The array values calculated by this fix are treated
 * as extensive."
 *
 * Output file: "Every :math:`N_\text{freq}` steps, an array of correlation data is
 * written to the file." Header (default, as the doc page prints it):
 * "# Time-correlated data for fix ID", "# TimeStep Number-of-time-windows",
 * "# Index TimeDelta Ncount valueI\*valueJ valueI\*valueJ ...".
 * Measured with native LAMMPS (black box): the second header line reads
 * # Timestep Number-of-time-windows (lower-case s). A section starts with the
 * step and Nrepeat, then one row per lag: 1-based row index, TimeDelta, Ncount, values;
 * values are written %g, except that a pair with Ncount 0 is written 0.0.
 * Measured: for type lower the third header line lists only the pairs with I - J >= 2
 * (N = 3 gives one label, N = 2 none) and for type auto/lower the pairs with I > J
 * (diagonal labels omitted); the value columns are unaffected. Reproduced here as printed.
 * "The *overwrite* keyword will continuously overwrite the output file" ... "This option
 * can only be used with the *ave running* setting."
 * "The *title1*, *title2*, and *title3* keywords allow specification of" the first three lines.
 *
 * "The group specified with this command is ignored."
 * "No parameter of this fix can be used with the *start/stop* keywords of"
 * "This fix is not invoked during"
 * Defaults: ave = one, type = auto, start = 0, no file, prefactor = 1.0.
 * "The *trap* function defined for :doc:`equal-style variables <variable>`
 * can be used to perform a time integration" of the output column (src/engine/groupfn.ts supports trap() on f_ID[I] columns).
 */

export const CORR_TYPES = ['auto', 'upper', 'lower', 'auto/upper', 'auto/lower', 'full', 'first'] as const;
export type CorrType = (typeof CORR_TYPES)[number];


/** Pairs [I, J] (0-based) in output column order for a type and N inputs. */
export const correlationPairs = (type: CorrType, n: number): [number, number][] => {
  const out: [number, number][] = [];
  switch (type) {
    case 'auto': for (let i = 0; i < n; i++) out.push([i, i]); break;
    case 'upper': for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push([i, j]); break;
    case 'lower': for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) out.push([i, j]); break;
    case 'auto/upper': for (let i = 0; i < n; i++) for (let j = i; j < n; j++) out.push([i, j]); break;
    case 'auto/lower': for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) out.push([i, j]); break;
    case 'full': for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out.push([i, j]); break;
    case 'first': for (let j = 0; j < n; j++) out.push([0, j]); break;
  }
  return out;
};

/** Pairs named in the third header line, as LAMMPS prints them (measured; see the file comment). */
export const correlationLabelPairs = (type: CorrType, n: number): [number, number][] => {
  if (type === 'lower') return correlationPairs(type, n).filter(([i, j]) => i - j >= 2);
  if (type === 'auto/lower') return correlationPairs('lower', n);
  return correlationPairs(type, n);
};


export class FixAveCorrelate extends Fix {
  readonly style = 'ave/correlate';
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private type: CorrType;
  private running: boolean;
  private startStep: number;
  private prefactor: number;
  private fileName: string | null;
  private overwrite: boolean;
  private readonly titles: (string | undefined)[];
  private inputs: Ref[] = [];
  private inputWords: string[] = [];
  private pairs: [number, number][] = [];
  /** The last Nrepeat samples (native cvalues), the newest at lastindex; nsample of them are filled. */
  private cvalues: Float64Array[] = [];
  private lastindex = -1;
  private nsample = 0;
  /** The correlation sums per lag and pair (native corr) and their sample counts per lag (native count). */
  private sums = new Float64Array(0);
  private counts = new Float64Array(0);
  /** The last output: per lag the time delta, the count and the correlations (native save_count and save_corr). */
  private out = new Float64Array(0);
  /** The next step that samples (native nvalid). */
  private nvalid: number;
  private headerWritten = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_ave_correlate.cpp constructor at b332f28): the words are expanded, the values
    // read up to the first word that is none (there may be none), then the keywords, the step counts and the sources
    if (args.length < 4) throw missingArgs('fix ave/correlate');
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    this.globalFreq = nfreq;
    const words = expandArgs(sys, args.slice(3), 0).words;
    let k = 0;
    for (; k < words.length; k++) {
      const a = argInfo(words[k]);
      if (a.which === 'none') break;
      if (a.which === 'unknown' || a.dim > 1) throw new StyleError(`Unknown fix ave/correlate data type: ${words[k]}`);
      if (a.index1 < 0) throw new StyleError(`fix ave/correlate: index ${a.index1} in ${words[k]} is refused: native LAMMPS reads outside the vector for it`);
      this.inputs.push({ kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: words[k] });
    }
    this.inputWords = words.slice(0, k);
    this.type = 'auto';
    this.running = false;
    this.startStep = 0;
    this.prefactor = 1;
    this.fileName = null;
    this.overwrite = false;
    this.titles = [undefined, undefined, undefined];
    const need = (what: string): void => { if (k + 2 > words.length) throw missingArgs(what); };
    while (k < words.length) {
      const key = words[k];
      const v = words[k + 1];
      if (key === 'type') {
        need('fix ave/correlate type');
        if (!(CORR_TYPES as readonly string[]).includes(v)) throw new StyleError(`Unknown fix ave/correlate type: ${v}`);
        this.type = v as CorrType;
        k += 2;
      } else if (key === 'ave') {
        need('fix ave/correlate ave');
        if (v !== 'one' && v !== 'running') throw new StyleError(`Unknown fix ave/correlate ave mode: ${v}`);
        this.running = v === 'running';
        k += 2;
      } else if (key === 'start') {
        need('fix ave/correlate start');
        this.startStep = inumeric(v);
        k += 2;
      } else if (key === 'prefactor') {
        need('fix ave/correlate prefactor');
        this.prefactor = numeric(v);
        k += 2;
      } else if (key === 'file') {
        need('fix ave/correlate file');
        this.fileName = v;
        k += 2;
      } else if (key === 'overwrite') {
        this.overwrite = true;
        k += 1;
      } else if (key === 'title1' || key === 'title2' || key === 'title3') {
        need(`fix ave/correlate ${key}`);
        this.titles[Number(key[5]) - 1] = v;
        k += 2;
      } else throw new StyleError(`Unkown fix ave/correlate keyword: ${key}`);
    }
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/correlate nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal fix ave/correlate nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/correlate nfreq value: ${nfreq}`);
    if (nfreq % nevery || nrepeat * nevery > nfreq) throw new StyleError('Inconsistent fix ave/correlate nevery/nrepeat/nfreq values');
    if (!this.running && this.overwrite) throw new StyleError('Fix ave/correlate overwrite keyword requires ave running setting');
    this.nEvery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    this.nevery = nevery;
    for (const r of this.inputs) {
      if (r.kind === 'c' || r.kind === 'f') {
        const what = r.kind === 'c' ? 'compute' : 'fix';
        const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id) : sys.fixes.find((x) => x.id === r.id);
        if (!o) throw new StyleError(`${r.kind === 'c' ? 'Compute' : 'Fix'} ID ${r.id} for fix ave/correlate does not exist`);
        if (r.index === null && !o.scalarFlag) throw new StyleError(`Fix ave/correlate ${what} ${r.id} does not calculate a scalar`);
        if (r.index !== null && !o.vectorFlag) throw new StyleError(`Fix ave/correlate ${what} ${r.id} does not calculate a vector`);
        // an engine compute that sizes its vector when it is first computed reports 0 here
        if (r.index !== null && o.sizeVector > 0 && r.index > o.sizeVector) {
          throw new StyleError(`Fix ave/correlate ${what} ${r.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
        if (r.kind === 'f' && (o as Fix).globalFreq > 0 && nevery % (o as Fix).globalFreq) {
          throw new StyleError(`Fix ${r.id} for fix ave/correlate not computed at compatible time${errorUrl(7)}`);
        }
      } else {
        const vv = sys.vars.get(r.id);
        if (!vv) throw new StyleError(`Variable name ${r.id} for fix ave/correlate does not exist`);
        if (r.index === null && !EQUAL_STYLES.has(vv.style)) throw new StyleError(`Fix ave/correlate variable ${r.id} is not equal-style variable`);
        if (r.index !== null && vv.style !== 'vector') throw new StyleError(`Fix ave/correlate variable ${r.id} is not vector-style variable`);
      }
    }
    this.nvalid = this.nextValid();
  }

  /** As native LAMMPS does (src/fix_ave_correlate.cpp nextvalid at b332f28): this step or start, up to a multiple of Nevery. */
  private nextValid(): number {
    let nvalid = Math.max(this.sys.state.step, this.startStep);
    if (nvalid % this.nEvery) nvalid = Math.floor(nvalid / this.nEvery) * this.nEvery + this.nEvery;
    return nvalid;
  }


  init(): void {
    // the sources must still exist (native init)
    for (const r of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ave/correlate does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for fix ave/correlate does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/correlate does not exist`);
    }
    this.pairs = correlationPairs(this.type, this.inputs.length);
    const npair = this.pairs.length;
    this.arrayFlag = true;
    this.extvector = 1; // "The array values calculated by this fix are treated as extensive."
    this.sizeArrayRows = this.nRepeat;
    this.sizeArrayCols = npair + 2;
    // As native LAMMPS does (src/fix_ave_correlate.cpp at b332f28): the samples, sums, counts and the last output
    // carry over a run boundary; only a change of their size reallocates them
    const nin = this.inputs.length;
    if (this.cvalues.length !== this.nRepeat || this.cvalues[0]?.length !== nin) {
      this.cvalues = Array.from({ length: this.nRepeat }, () => new Float64Array(nin));
    }
    if (this.sums.length !== this.nRepeat * npair) this.sums = new Float64Array(this.nRepeat * npair);
    if (this.counts.length !== this.nRepeat) this.counts = new Float64Array(this.nRepeat);
    if (this.out.length !== this.nRepeat * (npair + 2)) {
      this.out = new Float64Array(this.nRepeat * (npair + 2));
      for (let lag = 0; lag < this.nRepeat; lag++) this.out[lag * (npair + 2)] = lag * this.nEvery;
    }
    // native init: a minimization past the next sample step restarts the ring (the sums are kept)
    if (this.nvalid < this.sys.state.step) {
      this.lastindex = -1;
      this.nsample = 0;
      this.nvalid = this.nextValid();
    }
  }

  setup(): void { this.endOfStep(); }

  /**
   * As native LAMMPS does (src/fix_ave_correlate.cpp end_of_step at b332f28): every Nevery steps from the first
   * sample step the values go into the ring and the products with the earlier samples are added; on multiples of
   * Nfreq the averages are output, and with ave one the sums start again from the products of this step's sample.
   */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    this.lastindex++;
    if (this.lastindex === this.nRepeat) this.lastindex = 0;
    const vals = this.cvalues[this.lastindex];
    for (let k = 0; k < this.inputs.length; k++) vals[k] = this.inputValue(this.inputs[k]);
    if (this.nsample < this.nRepeat) this.nsample++;
    this.nvalid += this.nEvery;
    this.accumulate();
    if (step % this.nFreq) return;
    this.produce(step);
  }

  /** One input's value; an element past the end of a vector-style variable is 0 (native end_of_step). */
  private inputValue(r: Ref): number {
    if (r.kind === 'v' && r.index !== null) {
      const vec = this.sys.vars.evalVector(r.id, this.sys.formulaEnv);
      return r.index > vec.length ? 0 : vec[r.index - 1];
    }
    return globalScalar(this.sys, r);
  }

  /** Adds the products of the newest sample with each sample in the ring (lag 0 = itself); native accumulate. */
  private accumulate(): void {
    const npair = this.pairs.length;
    for (let k = 0; k < this.nsample; k++) this.counts[k]++;
    const now = this.cvalues[this.lastindex];
    let m = this.lastindex;
    for (let k = 0; k < this.nsample; k++) {
      const earlier = this.cvalues[m];
      for (let p = 0; p < npair; p++) {
        const [i, j] = this.pairs[p];
        this.sums[k * npair + p] += earlier[i] * now[j];
      }
      m--;
      if (m < 0) m = this.nRepeat - 1;
    }
  }

  private produce(step: number): void {
    const npair = this.pairs.length;
    const cols = npair + 2;
    for (let lag = 0; lag < this.nRepeat; lag++) {
      const c = this.counts[lag];
      const row = lag * cols;
      this.out[row] = lag * this.nEvery;
      this.out[row + 1] = c;
      for (let p = 0; p < npair; p++) this.out[row + 2 + p] = c > 0 ? (this.prefactor * this.sums[lag * npair + p]) / c : 0;
    }
    if (this.fileName) this.writeSection(step);
    if (!this.running) {
      this.sums.fill(0);
      this.counts.fill(0);
      this.nsample = 1;
      this.accumulate();
    }
  }

  private header(): string[] {
    const labels = correlationLabelPairs(this.type, this.inputs.length)
      .map(([i, j]) => `${this.inputWords[i]}*${this.inputWords[j]}`).join(' ');
    return [
      this.titles[0] ?? `# Time-correlated data for fix ${this.id}`,
      this.titles[1] ?? '# Timestep Number-of-time-windows',
      this.titles[2] ?? (labels ? `# Index TimeDelta Ncount ${labels}` : '# Index TimeDelta Ncount'),
    ];
  }

  private writeSection(step: number): void {
    const cols = this.pairs.length + 2;
    const lines = [`${step} ${this.nRepeat}`];
    for (let lag = 0; lag < this.nRepeat; lag++) {
      const row = lag * cols;
      const parts = [String(lag + 1), String(this.out[row]), String(this.out[row + 1])];
      for (let c = 2; c < cols; c++) {
        // A pair with no samples is written as the literal 0.0 (measured with native LAMMPS, black box).
        parts.push(this.out[row + 1] === 0 ? '0.0' : formatNumber(this.out[row + c], '%g'));
      }
      lines.push(parts.join(' '));
    }
    const section = lines.join('\n') + '\n';
    const name = this.fileName!;
    if (this.overwrite) {
      this.sys.writeFile(name, [...this.header(), section].join('\n'), false);
      return;
    }
    if (!this.headerWritten) {
      this.headerWritten = true;
      this.sys.writeFile(name, this.header().join('\n') + '\n' + section, false);
    } else {
      this.sys.writeFile(name, section, true);
    }
  }

  computeArray(i: number, j: number): number {
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} does not compute a global array`);
    return this.out[i * (this.pairs.length + 2) + j] ?? 0;
  }
}
