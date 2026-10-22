import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { globalScalar, type Ref } from '../refs';
import { inumeric, missingArgs } from '../force/util';
import { argInfo, EQUAL_STYLES, expandArgs } from '../argInfo';
import { errorUrl } from '../formula';

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
    // As native LAMMPS does (src/EXTRA-FIX/fix_ave_moments.cpp constructor at b332f28): the values (words that start
    // with c_, f_ or v_), the moments, the keywords; then the values are expanded and their sources checked. Native
    // does not compare Nevery, Nrepeat and Nfreq with each other.
    if (args.length < 5) throw missingArgs('fix ave/moments');
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    let k = 3;
    while (k < args.length && /^[cfv]_/.test(args[k])) k++;
    const values = args.slice(3, k);
    const noValues = 'No values from computes, fixes, or variables used in fix ave/moments command';
    if (!values.length) throw new StyleError(noValues);
    const moments: Moment[] = [];
    while (k < args.length && (MOMENTS as readonly string[]).includes(args[k])) moments.push(args[k++] as Moment);
    if (!moments.length) throw new StyleError(noValues);
    let startStep = 0;
    let nrecent = 1;
    while (k < args.length) {
      const key = args[k];
      if (key === 'history') {
        if (k + 2 > args.length) throw missingArgs('fix ave/moments history');
        nrecent = inumeric(args[k + 1]);
        if (nrecent <= 0) throw new StyleError(`Illegal ave/moments history argument ${nrecent}; must be > 0`);
      } else if (key === 'start') {
        if (k + 2 > args.length) throw missingArgs('fix ave/moments start');
        startStep = inumeric(args[k + 1]);
      } else throw new StyleError(`Unknown fix ave/moments keyword ${key}`);
      k += 2;
    }
    this.inputs = expandArgs(sys, values, 0).words.map((w): Ref => {
      const a = argInfo(w);
      if (a.which === 'none' || a.which === 'unknown' || a.dim > 1) throw new StyleError(`Invalid fix ave/moments argument: ${w}`);
      if (a.index1 < 0) throw new StyleError(`fix ave/moments: index ${a.index1} in ${w} is refused: native LAMMPS reads outside the vector for it`);
      return { kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: w };
    });
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/moments nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal fix ave/moments nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/moments nfreq value: ${nfreq}`);
    this.nevery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    for (const r of this.inputs) {
      if (r.kind === 'c' || r.kind === 'f') {
        const what = r.kind === 'c' ? 'compute' : 'fix';
        const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id) : sys.fixes.find((x) => x.id === r.id);
        if (!o) throw new StyleError(`${r.kind === 'c' ? 'Compute' : 'Fix'} ID ${r.id} for fix ave/moments does not exist`);
        if (r.index === null && !o.scalarFlag) throw new StyleError(`Fix ave/moments ${what} ${r.id} does not calculate a scalar`);
        if (r.index !== null && !o.vectorFlag) throw new StyleError(`Fix ave/moments ${what} ${r.id} does not calculate a vector`);
        if (r.kind === 'f' && r.index !== null && (o as Fix).sizeVectorVariable) throw new StyleError(`Fix ave/moments fix ${r.id} vector cannot be variable length`);
        // an engine compute that sizes its vector when it is first computed reports 0 here
        if (r.index !== null && o.sizeVector > 0 && r.index > o.sizeVector) {
          throw new StyleError(`Fix ave/moments ${what} ${r.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
        if (r.kind === 'f' && (o as Fix).globalFreq > 0 && nevery % (o as Fix).globalFreq) {
          throw new StyleError(`Fix ${r.id} for fix ave/moments not computed at compatible time${errorUrl(7)}`);
        }
      } else {
        const v = sys.vars.get(r.id);
        if (!v) throw new StyleError(`Variable name ${r.id} for fix ave/moments does not exist`);
        if (r.index === null && !EQUAL_STYLES.has(v.style)) throw new StyleError(`Fix ave/moments variable ${r.id} is not equal-style variable`);
        if (r.index !== null && v.style !== 'vector') throw new StyleError(`Fix ave/moments variable ${r.id} is not vector-style variable`);
      }
    }
    this.momentWords = moments;
    this.startStep = startStep;
    this.nrecent = nrecent;
    this.vectorFlag = true;
    this.arrayFlag = true;
    this.setShape();
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

  /** The sources must still exist (native init); the output sizes and flags are set again. */
  init(): void {
    for (const r of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ave/moments does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for fix ave/moments does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/moments does not exist`);
    }
    this.setShape();
    // a minimization past the next sample step: the schedule starts again from this step (native init)
    if (this.nvalid < this.sys.state.step) this.setNextValid();
  }

  /** The output sizes and intensive/extensive flags, and the window and result storage when their size changes. */
  private setShape(): void {
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
        throw new StyleError('Fix ave/moments cannot set output array intensive/extensive from these inputs');
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
