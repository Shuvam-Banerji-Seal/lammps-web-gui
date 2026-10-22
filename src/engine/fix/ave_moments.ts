import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { expandWildcards, globalScalar, parseRef, type Ref } from '../refs';

/*
 * fix ave/moments — docs.lammps.org/fix_ave_moments.html.
 *
 * Doc: "The :math:`N_\text{every}`, :math:`N_\text{repeat}`, and ..." — the
 * window: "on step 5 from steps 1-5, on step 10 from 3-10" (Nfreq 5, Nrepeat 8),
 * i.e. the last Nrepeat samples taken every Nevery steps, ending at the output
 * step.
 *
 * Doc (the moments): skew is "the adjusted Fisher--Pearson standardized
 * moment" G_1 = k3 / stddev^3 and kurtosis is the same kind of moment,
 * G_2 = k4 / k2^2 (k2 the Bessel-corrected variance; k3 and k4 the unbiased
 * sample cumulants, Cramer).
 *
 * As native LAMMPS does (src/EXTRA-FIX/fix_ave_moments.cpp at b332f28): the
 * samples go into a ring of Nrepeat entries; each output step (a multiple of
 * Nfreq, at or after start) is scheduled first and the first sample of its
 * window is the multiple of Nevery that leaves Nrepeat samples up to it, but
 * never before the current step, so a window longer than Nfreq keeps sampling
 * every Nevery steps (and with Nrepeat = Nfreq + 1 and Nevery 1 the next
 * sample step is the output step just done: the fix stops sampling). The
 * moments are taken over the filled entries in ring order, with the mean as
 * a sum of x / n, and only when at least 3 are filled (else the outputs keep
 * their values).
 */

const MOMENTS = ['mean', 'stddev', 'variance', 'skew', 'kurtosis'] as const;
type Moment = (typeof MOMENTS)[number];

/**
 * The moments of each input over its filled ring entries, as native LAMMPS does (src/EXTRA-FIX/fix_ave_moments.cpp
 * update_results at b332f28): series[k][j] for entry j of input k; the result is interleaved, input 1's moments first.
 */
const moments = (series: Float64Array[], count: number, kinds: Moment[]): Float64Array => {
  const res = new Float64Array(series.length * kinds.length);
  const invN = 1.0 / count;
  const fk2 = count / (count - 1);
  const fk3 = (count * count) / ((count - 1) * (count - 2));
  const np1Nm3 = (count + 1.0) / (count - 3.0);
  const nm1Nm3 = (3.0 * (count - 1.0)) / (count - 3.0);
  series.forEach((x, k) => {
    let mean = 0.0;
    for (let j = 0; j < count; j++) mean += x[j] * invN;
    let m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (let j = 0; j < count; j++) {
      const dx = x[j] - mean;
      let y = dx * dx * invN;
      m2 += y;
      y *= dx;
      m3 += y;
      y *= dx;
      m4 += y;
    }
    const k2 = fk2 * m2;
    const k3 = fk3 * m3;
    const k4 = fk3 * (np1Nm3 * m4 - nm1Nm3 * (m2 * m2));
    const stddev = Math.sqrt(k2);
    const vals: Record<Moment, number> = {
      mean, stddev, variance: k2, skew: k3 / (stddev * stddev * stddev), kurtosis: k4 / (k2 * k2),
    };
    kinds.forEach((m, j) => { res[k * kinds.length + j] = vals[m]; });
  });
  return res;
};

/** The first step at or after `after` that is ts plus a multiple of freq (native next_after). */
const nextAfter = (ts: number, after: number, freq: number): number =>
  ts >= after ? ts : ts + (Math.floor((after - ts) / freq) + 1) * freq;

export class FixAveMoments extends Fix {
  readonly style = 'ave/moments';
  private readonly nRepeat: number;
  private readonly startStep: number;
  private readonly nrecent: number;
  private readonly valueWords: string[];
  private readonly momentWords: Moment[];
  private inputs: Ref[] = [];
  /** The ring of samples, one array per input (native window_list), the next entry and whether it has wrapped. */
  private window: Float64Array[] = [];
  private iwindow = 0;
  private windowFilled = false;
  /** The ring of outputs (native result_list, history rows), and the next row to write. */
  private results: Float64Array[] = [];
  private iresult = 0;
  /** The next step that samples and the next output step (native nvalid and nvalid_comp_next). */
  private nvalid = -1;
  private nvalidCompNext = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (ave/moments)`;
    if (args.length < 4) throw new StyleError(`usage: fix ${id} group-ID ave/moments Nevery Nrepeat Nfreq value1 ... moment1 ... [start N] [history N]`);
    const posInt = (w: string, name: string): number => {
      const v = Number(w);
      if (!Number.isInteger(v) || v < 1) throw new StyleError(`${what}: ${name} must be a positive integer, got '${w}'`);
      return v;
    };
    this.nevery = posInt(args[0], 'Nevery');
    this.nRepeat = posInt(args[1], 'Nrepeat');
    const nFreq = posInt(args[2], 'Nfreq');
    this.nFreq = nFreq;
    const isValue = (w: string) => /^[cfv]_/.test(w);
    const isMoment = (w: string) => (MOMENTS as readonly string[]).includes(w);
    let k = 3;
    const values: string[] = [];
    while (k < args.length && isValue(args[k])) values.push(args[k++]);
    if (!values.length) throw new StyleError(`${what}: no input values (c_ID, f_ID or v_name) were given`);
    const moments: Moment[] = [];
    while (k < args.length && isMoment(args[k])) moments.push(args[k++] as Moment);
    if (!moments.length) throw new StyleError(`${what}: no moments were given (${MOMENTS.join(', ')})`);
    this.valueWords = values;
    this.momentWords = moments;
    let startStep = 0;
    let nrecent = 1;
    while (k < args.length) {
      const key = args[k];
      const value = args[k + 1];
      if (value === undefined) throw new StyleError(`${what}: keyword '${key}' needs a value`);
      if (key === 'start') {
        const v = Number(value);
        if (!Number.isInteger(v) || v < 0) throw new StyleError(`${what}: start must be an integer >= 0, got '${value}'`);
        startStep = v;
      } else if (key === 'history') {
        nrecent = posInt(value, 'history');
      } else {
        throw new StyleError(`${what}: unknown keyword '${key}'`);
      }
      k += 2;
    }
    this.startStep = startStep;
    this.nrecent = nrecent;
    this.vectorFlag = true;
    this.arrayFlag = true;
    // thermo f_ID[*] expands before init(): without wildcards the sizes are known now
    if (!values.some((w) => w.includes('*'))) {
      this.sizeVector = values.length * moments.length;
      this.sizeArrayRows = this.sizeVector;
      this.sizeArrayCols = nrecent;
    }
    this.setNextValid();
  }

  /**
   * As native LAMMPS does (setnextvalid): before the next output step the next sample is Nevery steps on; else the
   * next output step is scheduled and the window's first sample from it, not before this step.
   */
  private setNextValid(): void {
    const step = this.sys.state.step;
    if (this.nvalidCompNext > step) {
      this.nvalid = step + this.nevery;
      return;
    }
    const nextComp = Math.floor(step / this.nFreq) * this.nFreq + this.nFreq;
    this.nvalidCompNext = nextAfter(nextComp, this.startStep, this.nFreq);
    const ntake = this.nvalidCompNext - (this.nRepeat - 1) * this.nevery;
    this.nvalid = nextAfter(ntake, step, this.nevery);
  }

  /** Nfreq is the output interval (multiples of Nfreq compute the moments). */
  readonly nFreq: number;

  /** Resolves wildcards and sets the output sizes (computes must exist by now). */
  init(): void {
    this.inputs = expandWildcards(this.sys, this.valueWords, 'scalar').map((w) => parseRef(w));
    const size = this.inputs.length * this.momentWords.length;
    // As native LAMMPS does (src/EXTRA-FIX/fix_ave_moments.cpp at b332f28): every moment of a value takes that value's
    // flag (the vector's per-element list), and the array takes the one flag all values share, else it is an error
    let ext = -2;
    const list: number[] = [];
    for (const r of this.inputs) {
      let e = 0;
      if (r.kind !== 'v') {
        const src = r.kind === 'c' ? this.sys.compute(r.id) : this.sys.fix(r.id);
        e = r.index === null ? src.extscalar : src.extlist ? src.extlist[r.index - 1] ?? 0 : src.extvector;
      }
      if (ext > -2 && e !== ext) {
        throw new StyleError(`fix ${this.id} (ave/moments): Fix ave/moments cannot set output array intensive/extensive from these inputs`);
      }
      ext = e;
      for (let m = 0; m < this.momentWords.length; m++) list.push(e);
    }
    this.extvector = 0;
    this.extlist = list;
    this.extarray = ext < 0 ? 0 : ext;
    this.sizeVector = size;
    this.sizeArrayRows = size;
    this.sizeArrayCols = this.nrecent;
    // As native LAMMPS does (src/EXTRA-FIX/fix_ave_moments.cpp at b332f28): the window and the results are allocated
    // once (the constructor) and init() keeps them, so a later run starts with the samples and outputs of the last one
    if (this.results[0]?.length !== size || this.window.length !== this.inputs.length) {
      this.window = this.inputs.map(() => new Float64Array(this.nRepeat));
      this.iwindow = 0;
      this.windowFilled = false;
      this.results = Array.from({ length: this.nrecent }, () => new Float64Array(size));
      this.iresult = 0;
    }
    // a minimization past the next sample step: the schedule starts again from this step (native init)
    if (this.nvalid < this.sys.state.step) this.setNextValid();
  }

  setup(): void { this.endOfStep(); }

  /** As native LAMMPS does (end_of_step): a scheduled step samples, and an output step also takes the moments. */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    this.inputs.forEach((r, k) => { this.window[k][this.iwindow] = this.inputValue(r); });
    if (++this.iwindow >= this.nRepeat) {
      this.windowFilled = true;
      this.iwindow = 0;
    }
    if (step === this.nvalidCompNext) {
      const count = this.windowFilled ? this.nRepeat : this.iwindow;
      if (count >= 3) {
        this.results[this.iresult] = moments(this.window, count, this.momentWords);
        if (++this.iresult >= this.nrecent) this.iresult = 0;
      }
      this.setNextValid();
    } else {
      this.nvalid += this.nevery;
    }
  }

  /** One input's value; an element past the end of a vector-style variable is 0 (native get_values). */
  private inputValue(r: Ref): number {
    if (r.kind === 'v' && r.index !== null) {
      const vec = this.sys.vars.evalVector(r.id, this.sys.formulaEnv);
      return r.index > vec.length ? 0 : vec[r.index - 1];
    }
    return globalScalar(this.sys, r);
  }

  computeVector(i: number): number {
    if (!this.vectorFlag) throw new StyleError(`fix ${this.id} does not compute a global vector`);
    return this.computeArray(i, 0);
  }

  /** f_ID[i][j] (0-based): column j is the output j outputs before the latest (history keyword). */
  computeArray(i: number, j: number): number {
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} does not compute a global array`);
    if (i >= this.sizeVector || j >= this.nrecent) return 0;
    const row = (this.iresult - 1 - j + this.nrecent) % this.nrecent;
    return this.results[row]?.[i] ?? 0;
  }
}
