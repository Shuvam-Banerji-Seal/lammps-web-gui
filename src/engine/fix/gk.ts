import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { expandWildcards, globalScalar, parseRef, type Ref } from '../refs';
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

const CORR_KEYWORDS = new Set(['type', 'ave', 'start', 'prefactor', 'file', 'overwrite', 'title1', 'title2', 'title3']);

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

interface Sample { step: number; vals: Float64Array }

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
  private readonly valueWords: string[];
  private inputs: Ref[] = [];
  private inputWords: string[] = [];
  private pairs: [number, number][] = [];
  private ring: Sample[] = [];
  private sums = new Float64Array(0);
  private counts = new Float64Array(0);
  private out = new Float64Array(0);
  private windowStart: number | null = null;
  private lastSampled = -1;
  private headerWritten = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (ave/correlate)`;
    if (args.length < 4) throw new StyleError(`usage: fix ${id} group-ID ave/correlate Nevery Nrepeat Nfreq value1 ... [keyword args ...]`);
    const pos = (w: string, name: string): number => {
      const n = Number(w);
      if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what}: ${name} must be a positive integer, got '${w}'`);
      return n;
    };
    this.nEvery = pos(args[0], 'Nevery');
    this.nRepeat = pos(args[1], 'Nrepeat');
    this.nFreq = pos(args[2], 'Nfreq');
    this.nevery = this.nEvery;
    const words: string[] = [];
    let k = 3;
    while (k < args.length && !CORR_KEYWORDS.has(args[k])) { words.push(args[k]); k++; }
    if (!words.length) throw new StyleError(`${what}: no input values were given`);
    if (this.nFreq % this.nEvery !== 0) {
      throw new StyleError(`${what}: Nfreq must be a multiple of Nevery`);
    }
    this.valueWords = words;
    this.type = 'auto';
    this.running = false;
    this.startStep = 0;
    this.prefactor = 1;
    this.fileName = null;
    this.overwrite = false;
    this.titles = [undefined, undefined, undefined];
    while (k < args.length) {
      const key = args[k];
      const val = args[k + 1];
      const need = (): string => {
        if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
        return val;
      };
      switch (key) {
        case 'type': {
          const v = need();
          if (!(CORR_TYPES as readonly string[]).includes(v)) throw new StyleError(`${what}: type must be one of ${CORR_TYPES.join(' ')}, got '${v}'`);
          this.type = v as CorrType;
          k += 2;
          break;
        }
        case 'ave': {
          const v = need();
          if (v !== 'one' && v !== 'running') throw new StyleError(`${what}: ave must be one or running, got '${v}'`);
          this.running = v === 'running';
          k += 2;
          break;
        }
        case 'start': {
          const n = Number(need());
          if (!Number.isInteger(n) || n < 0) throw new StyleError(`${what}: start must be a timestep >= 0, got '${val}'`);
          this.startStep = n;
          k += 2;
          break;
        }
        case 'prefactor': {
          const x = Number(need());
          if (!Number.isFinite(x)) throw new StyleError(`${what}: prefactor must be a number, got '${val}'`);
          this.prefactor = x;
          k += 2;
          break;
        }
        case 'file':
          this.fileName = need();
          k += 2;
          break;
        case 'overwrite':
          this.overwrite = true;
          k += 1;
          break;
        case 'title1': case 'title2': case 'title3':
          this.titles[Number(key[5]) - 1] = need();
          k += 2;
          break;
        default:
          throw new StyleError(`${what}: unknown keyword '${key}'`);
      }
    }
    if (this.overwrite && !this.running) {
      throw new StyleError(`${what}: the overwrite keyword can only be used with the ave running setting`);
    }
    if (!this.running && this.nFreq < (this.nRepeat - 1) * this.nEvery) {
      throw new StyleError(`${what}: with ave one, Nfreq must be >= (Nrepeat-1)*Nevery`);
    }
  }


  init(): void {
    this.inputWords = expandWildcards(this.sys, this.valueWords, 'scalar');
    this.inputs = this.inputWords.map((w) => parseRef(w));
    this.pairs = correlationPairs(this.type, this.inputs.length);
    const npair = this.pairs.length;
    this.arrayFlag = true;
    this.extvector = 1; // "The array values calculated by this fix are treated as extensive."
    this.sizeArrayRows = this.nRepeat;
    this.sizeArrayCols = npair + 2;
    this.sums = new Float64Array(this.nRepeat * npair);
    this.counts = new Float64Array(this.nRepeat);
    this.out = new Float64Array(this.nRepeat * (npair + 2));
  }

  setup(): void { this.endOfStep(); }

  endOfStep(): void {
    const step = this.sys.state.step;
    if (step % this.nEvery !== 0 || step < this.startStep || step === this.lastSampled) return;
    this.lastSampled = step;
    const vals = new Float64Array(this.inputs.length);
    for (let k = 0; k < this.inputs.length; k++) vals[k] = globalScalar(this.sys, this.inputs[k]);
    this.ring.unshift({ step, vals });
    if (this.ring.length > this.nRepeat) this.ring.pop();
    if (this.windowStart === null) this.windowStart = step;
    const npair = this.pairs.length;
    const now = this.ring[0];
    for (let lag = 0; lag < this.ring.length; lag++) {
      const earlier = this.ring[lag];
      if (earlier.step < this.windowStart) break;
      for (let p = 0; p < npair; p++) {
        const [i, j] = this.pairs[p];
        this.sums[lag * npair + p] += earlier.vals[i] * now.vals[j];
      }
      this.counts[lag]++;
    }
    if (step % this.nFreq === 0 && step >= this.startStep) this.produce(step);
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
      // The sample at this step starts the next window (the "exception" of the doc page).
      this.sums.fill(0);
      this.counts.fill(0);
      this.windowStart = step;
      const now = this.ring[0];
      for (let p = 0; p < npair; p++) {
        const [i, j] = this.pairs[p];
        this.sums[p] += now.vals[i] * now.vals[j];
      }
      this.counts[0]++;
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
