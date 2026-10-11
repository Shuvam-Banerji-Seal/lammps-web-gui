import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { expandWildcards, globalScalar, parseRef, type Ref } from '../refs';

/*
 * fix ave/moments — docs.lammps.org/fix_ave_moments.html. Written from the doc
 * page and from black-box runs of the native binary.
 *
 * Doc: "The :math:`N_\text{every}`, :math:`N_\text{repeat}`, and ..." — the
 * window: "on step 5 from steps 1-5, on step 10 from 3-10" (Nfreq 5, Nrepeat 8),
 * i.e. the last Nrepeat samples taken every Nevery steps, ending at the output
 * step. Measured: step 0 is never sampled, and the outputs appear only on
 * multiples of Nfreq (an output of 0 before the first one).
 *
 * Doc (the moments): skew is "the adjusted Fisher--Pearson standardized
 * moment" G_1 = k3 / stddev^3 and kurtosis is the same kind of moment,
 * G_2 = k4 / k2^2 (k2 the Bessel-corrected variance; k3 and k4 the unbiased
 * sample cumulants, Cramer). Measured native values agree with these formulas
 * for n = 8 (skew 0.879189153563413, kurtosis -0.663817520661157).
 *
 * Measured: a window with fewer than 3 samples outputs 0 for every moment; with
 * n = 3 the kurtosis is NaN (n - 3 = 0 in the cumulant correction). The
 * start keyword is "no earlier than" the step given (measured for Nrepeat <=
 * Nfreq, Nevery 1: an output on the start step itself is computed, except for
 * the single cell Nfreq 3, Nrepeat 3, start 6, where native gives 0).
 *
 * Not reproduced (measured): with Nrepeat > Nfreq and Nevery 1, native outputs
 * were measured to freeze after the first output when Nrepeat = Nfreq + 1, and
 * the first window can include step 0 (Nfreq 3, Nrepeat 4 gives the mean of
 * steps 0-3). The engine follows the documented window (last Nrepeat samples
 * of steps >= 1); the oracle cases avoid those windows.
 */

const MOMENTS = ['mean', 'stddev', 'variance', 'skew', 'kurtosis'] as const;
type Moment = (typeof MOMENTS)[number];

/** Moment of the sample x (length n). The formulas are those of the doc page (see the file comment). */
const moment = (kind: Moment, x: number[]): number => {
  const n = x.length;
  let sum = 0;
  for (const v of x) sum += v;
  const mean = sum / n;
  if (kind === 'mean') return mean;
  let m2 = 0, m3 = 0, m4 = 0;
  for (const v of x) {
    const d = v - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  // Bessel-corrected variance k2
  const k2 = m2 / (n - 1);
  if (kind === 'variance') return k2;
  if (kind === 'stddev') return Math.sqrt(k2);
  if (kind === 'skew') {
    // unbiased third cumulant k3 = n^2 / ((n-1)(n-2)) * (sum d^3 / n); G1 = k3 / stddev^3
    const k3 = ((n * n) / ((n - 1) * (n - 2))) * (m3 / n);
    return k3 / Math.pow(Math.sqrt(k2), 3);
  }
  // kurtosis: unbiased fourth cumulant k4 = n^2 [(n+1) m4 - 3 (n-1) m2^2] / ((n-1)(n-2)(n-3)), G2 = k4 / k2^2
  if (n < 4) return NaN;
  const mm2 = m2 / n, mm4 = m4 / n;
  const k4 = (n * n * ((n + 1) * mm4 - 3 * (n - 1) * mm2 * mm2)) / ((n - 1) * (n - 2) * (n - 3));
  return k4 / (k2 * k2);
};

export class FixAveMoments extends Fix {
  readonly style = 'ave/moments';
  private readonly nRepeat: number;
  private readonly startStep: number;
  private readonly nrecent: number;
  private readonly valueWords: string[];
  private readonly momentWords: Moment[];
  private inputs: Ref[] = [];
  /** Sample window: one entry per sample step, holding the value of every input. */
  private window: number[][] = [];
  /** Current output, interleaved: input 1's moments, then input 2's, ... */
  private out = new Float64Array(0);
  /** Outputs, most recent first (history keyword). */
  private hist: Float64Array[] = [];

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
    this.extvector = 0;
    // thermo f_ID[*] expands before init(): without wildcards the sizes are known now
    if (!values.some((w) => w.includes('*'))) {
      this.sizeVector = values.length * moments.length;
      this.sizeArrayRows = this.sizeVector;
      this.sizeArrayCols = nrecent;
    }
  }

  /** Nfreq is the output interval (multiples of Nfreq compute the moments). */
  readonly nFreq: number;

  /** Resolves wildcards and sets the output sizes (computes must exist by now). */
  init(): void {
    this.inputs = expandWildcards(this.sys, this.valueWords, 'scalar').map((w) => parseRef(w));
    const size = this.inputs.length * this.momentWords.length;
    this.sizeVector = size;
    this.sizeArrayRows = size;
    this.sizeArrayCols = this.nrecent;
    this.out = new Float64Array(size);
    this.hist = [];
    this.window = [];
  }

  /** Samples are taken in end_of_step; the moments are evaluated on multiples of Nfreq after the start step. */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    this.window.push(this.inputs.map((r) => globalScalar(this.sys, r)));
    if (this.window.length > this.nRepeat) this.window.shift();
    if (step % this.nFreq !== 0 || step < this.startStep) return;
    const M = this.momentWords.length;
    const res = new Float64Array(this.inputs.length * M);
    const n = this.window.length;
    if (n >= 3) {
      for (let k = 0; k < this.inputs.length; k++) {
        const x = this.window.map((w) => w[k]);
        for (let m = 0; m < M; m++) res[k * M + m] = moment(this.momentWords[m], x);
      }
    }
    this.out = res;
    this.hist.unshift(res.slice());
    if (this.hist.length > this.nrecent) this.hist.pop();
  }

  computeVector(i: number): number {
    if (!this.vectorFlag) throw new StyleError(`fix ${this.id} does not compute a global vector`);
    return this.out[i] ?? 0;
  }

  /** f_ID[i][j] (0-based): column 0 is the most recent output (history keyword). */
  computeArray(i: number, j: number): number {
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} does not compute a global array`);
    return this.hist[j]?.[i] ?? 0;
  }
}
