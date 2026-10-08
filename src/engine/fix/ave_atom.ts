import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { expandWildcards, globalScalar, globalVector, parseRef, peratomValues, type Ref } from '../refs';
import { formatNumber } from '../script';

/*
 * fix ave/atom and fix ave/histo — implemented only from the cited
 * docs.lammps.org pages (never from LAMMPS source code).
 *
 * ------------------------------------------------------------------ fix ave/atom
 * docs.lammps.org/fix_ave_atom.html, Syntax:
 *
 *   fix ID group-ID ave/atom Nevery Nrepeat Nfreq value1 value2 ...
 *
 * with "Nevery = use input values every this many timesteps", "Nrepeat = # of
 * times to use input values for calculating averages", "Nfreq = calculate
 * averages every this many timesteps" and "value = x, y, z, vx, vy, vz, fx,
 * fy, fz, c_ID, c_ID[i], f_ID, f_ID[i], v_name" ("x,y,z,vx,vy,vz,fx,fy,fz =
 * atom attribute (position, velocity, force component)"; "c_ID = per-atom
 * vector calculated by a compute with ID"; "c_ID[I] = Ith column of per-atom
 * array calculated by a compute with ID"; "f_ID = per-atom vector calculated
 * by a fix with ID"; "f_ID[I] = Ith column of per-atom array calculated by a
 * fix with ID"; "v_name = per-atom vector calculated by an atom-style
 * variable with name").
 *
 * Sampling schedule: "The final averaged quantities are generated on
 * timesteps that are a multiple of :math:`N_\text{freq}`\ .  The average is
 * over :math:`N_\text{repeat}` quantities, computed in the preceding portion
 * of the simulation every :math:`N_\text{every}` timesteps.
 * :math:`N_\text{freq}` must be a multiple of :math:`N_\text{every}` and
 * :math:`N_\text{every}` must be non-zero even if :math:`N_\text{repeat}` is
 * 1.  Also, the timesteps contributing to the average value cannot overlap;
 * that is, :math:`N_\text{repeat} \times N_\text{every}` cannot exceed
 * :math:`N_\text{freq}`." — "For example, if Nevery=2,
 * Nrepeat=6, and Nfreq=100, then values on timesteps 90, 92, 94, 96, 98, and
 * 100 will be used to compute the final average on time step 100."
 *
 * Group: "The group specified with the command means only atoms within the
 * group have their averages computed.  Results are set to 0.0 for atoms not
 * in the group." Inputs: "the compute, fix, or variable must produce a
 * per-atom vector, not a global quantity or local quantity"; "Variables of
 * style atom are the only ones that can be used with this fix since they
 * produce per-atom vectors."
 *
 * Output: "This fix produces a per-atom vector or array which can be accessed
 * by various output commands.  A vector is produced if only a single quantity
 * is averaged by this fix.  If two or more quantities are averaged, then an
 * array of values is produced.  The per-atom values can only be accessed on
 * timesteps that are multiples of Nfreq since that is when averaging is
 * performed." "No global scalar or vector quantities are stored by this fix
 * for access by various output commands."
 * Default: "none".
 *
 * ------------------------------------------------------------------ fix ave/histo
 * docs.lammps.org/fix_ave_histo.html, Syntax:
 *
 *   fix ID group-ID style Nevery Nrepeat Nfreq lo hi Nbin value1 value2 ... keyword args ...
 *
 * with "style = ave/histo or ave/histo/weight"; "Nevery = use input values
 * every this many timesteps"; "Nrepeat = # of times to use input values for
 * calculating histogram"; "Nfreq = calculate histogram every this many
 * timesteps"; "lo,hi = lo/hi bounds within which to histogram"; "Nbin = #
 * of histogram bins"; "zero or more keyword/arg pairs may be appended";
 * "keyword = mode or kind or file or append or ave or start or beyond or
 * overwrite or title1 or title2 or title3"; "mode arg = scalar or vector";
 * "kind arg = global or peratom or local"; "ave args = one or running or
 * window"; "beyond arg = ignore or end or extra"; "overwrite arg = none".
 *
 * Group: "The group specified with this command is ignored for global and
 * local input values.  For per-atom input values, only atoms in the group
 * contribute to the histogram." Mixing: "The set of input values can be
 * either all global, all per-atom, or all local quantities.  Inputs of
 * different kinds (e.g. global and per-atom) cannot be mixed." Kind: "The
 * kind keyword only needs to be used if any of the specified input computes
 * or fixes produce more than one kind of output (global, per-atom, local)."
 *
 * Binning: "Values such that *lo* :math:`\le` value :math:`\le` *hi* are
 * assigned to one bin.  Values on a bin boundary are assigned to the lower of
 * the two bins." Beyond:
 * "If beyond is set to ignore then values < lo and values > hi are ignored
 * (i.e., they are not binned). If beyond is set to end, then values < lo are
 * counted in the first bin and values > hi are counted in the last bin. If
 * beyond is set to extend, then two extra bins are created so that there are
 * Nbins+2 total bins.  Values < lo are counted in the first bin and values >
 * hi are counted in the last bin (Nbins+2).  Values between lo and hi
 * (inclusive) are counted in bins 2 through Nbins+1.  The "coordinate"
 * stored and printed for these two extra bins is lo and hi." (The keyword
 * list spells the third setting "extra"; both spellings are accepted.)
 *
 * ave keyword: "If the ave setting is one, then the histograms produced on
 * timesteps that are multiples of Nfreq are independent of each other; they
 * are output as-is without further averaging." "If the ave setting is
 * running, then the histograms produced on timesteps that are multiples of
 * Nfreq are summed and averaged in a cumulative sense before being output."
 * "If the ave setting is window, then the histograms produced on timesteps
 * that are multiples of Nfreq are summed within a moving" window of the last
 * M histograms, which produce the output. Start: "The
 * start keyword specifies what timestep histogramming will begin on.  The
 * default is step 0."
 *
 * File output: "If file is used, then the filename is overwritten if it
 * already exists. If append is used, then the filename is appended to if it
 * already exists, or created if it does not exist.  Every Nfreq steps, one
 * histogram is written to the file.  This includes a leading line that
 * contains the timestep, number of bins, the total count of values
 * contributing to the histogram, the count of values that were not
 * histogrammed (see the beyond keyword), the minimum value encountered, and
 * the maximum value encountered.  The min/max values include values that were
 * not histogrammed.  Following the leading line, one line per bin is written
 * into the file.  Each line contains the bin #, the coordinate for the center
 * of the bin (between lo and hi), the count of values in the bin, and the
 * normalized count.  The normalized count is the bin count divided by the
 * total count (not including values not histogrammed), so that the normalized
 * values sum to 1.0 across all bins." "The overwrite keyword will
 * continuously overwrite the output file with the latest output, so that it
 * only contains one timestep worth of output.  This option can only be used
 * with the ave running setting." Titles: "By default, these header lines are
 * as follows:"
 *
 *    # Histogram for fix ID
 *    # TimeStep Number-of-bins Total-counts Missing-counts Min-value Max-value
 *    # Bin Coord Count Count/Total
 *
 * "In the first line, ID is replaced with the fix-ID." The engine's default
 * first line is # Histogrammed data for fix ID, which is what current
 * native LAMMPS writes (the oracle fixtures compare it token by token); the
 * other two defaults match the page.
 *
 * Output (fix_ave_histo.html): "This fix produces a global vector and global
 * array which can be accessed by various output commands.  The values can
 * only be accessed on timesteps that are multiples of Nfreq since that is
 * when a histogram is generated.  The global vector has four values:" total
 * counts in the histogram, values that were not histogrammed (see the beyond
 * keyword), and the min and max of all input values, including ones not
 * histogrammed. "The global array has Nbins rows and three columns.  The
 * first column has the bin coordinate, the second column has the count of
 * values in that histogram bin, and the third column has the bin count
 * divided by the total count (not including missing counts), so that the
 * values in the third column sum to 1.0." "The vector and array values
 * calculated by this fix are all treated as intensive."
 * "This fix is not invoked during energy minimization." Default: "The option
 * defaults are mode = scalar, kind = figured out from input arguments, ave =
 * one, start = 0, no file output, beyond = ignore, and title 1,2,3 = strings
 * as described above."
 *
 * Both fixes are not invoked during minimization ("This fix is not invoked
 * during energy minimization") and "No parameter of this fix can be used with
 * the start/stop keywords of the run command".
 */

const posInt = (w: string | undefined, what: string): number => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  const n = Number(w);
  if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what} must be a positive integer, got '${w}'`);
  return n;
};

const fmtG = (v: number): string => formatNumber(v, '%g');
const fmt16 = (v: number): string => formatNumber(v, '%.16g');

export class FixAveAtom extends Fix {
  readonly style = 'ave/atom';
  peratomFlag = true;
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private readonly inputs: Ref[];
  private readonly nvals: number;
  private sum = new Float64Array(0);
  private nSample = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (ave/atom)`;
    if (args.length < 4) {
      throw new StyleError(`usage: fix ${id} group-ID ave/atom Nevery Nrepeat Nfreq value1 value2 ...`);
    }
    this.nEvery = posInt(args[0], `${what}: Nevery`);
    this.nRepeat = posInt(args[1], `${what}: Nrepeat`);
    this.nFreq = posInt(args[2], `${what}: Nfreq`);
    this.nevery = this.nEvery;
    if (this.nFreq % this.nEvery !== 0) {
      throw new StyleError(`${what}: Nfreq must be a multiple of Nevery`);
    }
    if (this.nRepeat * this.nEvery > this.nFreq) {
      throw new StyleError(`${what}: Nrepeat x Nevery cannot exceed Nfreq (the time steps contributing to the average cannot overlap)`);
    }
    const words = expandWildcards(sys, args.slice(3), 'peratom');
    this.inputs = words.map((w) => this.parseInput(w));
    this.nvals = this.inputs.length;
    this.sizePeratomCols = this.nvals === 1 ? 0 : this.nvals;
    this.alloc();
  }

  /** One input: must be a per-atom producer ("not a global quantity or local quantity"). */
  private parseInput(w: string): Ref {
    const r = parseRef(w, true);
    const what = `fix ${this.id} (ave/atom)`;
    if (r.kind === 'v') {
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`${what}: variable ${r.id} does not exist`);
      if (v.style !== 'atom') {
        throw new StyleError(`${what}: variable ${r.id} must be atom-style (is ${v.style}-style); "Variables of style atom are the only ones that can be used with this fix"`);
      }
      if (r.index !== null) throw new StyleError(`${what}: ${w}: an atom-style variable has no columns`);
      return r;
    }
    if (r.kind === 'c' || r.kind === 'f') {
      const obj = r.kind === 'c' ? this.sys.compute(r.id) : this.sys.fix(r.id);
      const kind = r.kind === 'c' ? 'compute' : 'fix';
      if (!obj.peratomFlag) {
        throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate per-atom values ("must produce a per-atom vector, not a global quantity or local quantity")`);
      }
      if (r.index === null) {
        if (obj.sizePeratomCols !== 0) throw new StyleError(`${what}: ${w}: it calculates a per-atom array; give a column, e.g. ${w}[1]`);
      } else {
        if (obj.sizePeratomCols === 0) throw new StyleError(`${what}: ${w}: it calculates a per-atom vector, which has no columns`);
        if (r.index > obj.sizePeratomCols) throw new StyleError(`${what}: ${w}: column out of range 1..${obj.sizePeratomCols}`);
      }
      return r;
    }
    return r;
  }

  private alloc(): void {
    const n = this.sys.state.n;
    const need = this.nvals * n;
    const outOk = this.nvals === 1 ? this.vectorAtom.length === n : this.arrayAtom.length === need;
    if (this.sum.length === need && outOk) return;
    this.sum = new Float64Array(need);
    if (this.nvals === 1) this.vectorAtom = new Float64Array(n);
    else this.arrayAtom = new Float64Array(need);
    this.nSample = 0;
  }

  init(): void {
    this.alloc();
  }

  /** Runs the end-of-step logic at the setup step as well (values for step-0 output when the window is complete). */
  setup(): void {
    this.endOfStep();
  }

  endOfStep(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    // samples only within the next output's window ("cannot overlap")
    const rem = step % this.nFreq;
    const nextOut = rem === 0 ? step : step + (this.nFreq - rem);
    if (step < nextOut - (this.nRepeat - 1) * this.nEvery) return;
    this.sample();
    if (step % this.nFreq !== 0) return;
    if (this.nSample === this.nRepeat) this.produce();
    else {
      // window cannot be complete yet (e.g. step 0 with Nrepeat > 1): discard it
      this.sum.fill(0);
      this.nSample = 0;
    }
  }

  private sample(): void {
    const n = this.sys.state.n;
    for (let k = 0; k < this.nvals; k++) {
      const vals = peratomValues(this.sys, this.inputs[k]);
      const off = k * n;
      for (let i = 0; i < n; i++) {
        if (this.inGroup(i)) this.sum[off + i] += vals[i];
      }
    }
    this.nSample++;
  }

  private produce(): void {
    const n = this.sys.state.n;
    const cols = this.nvals;
    const out = cols === 1 ? this.vectorAtom : this.arrayAtom;
    for (let i = 0; i < n; i++) {
      // "Results are set to 0.0 for atoms not in the group"
      const inG = this.inGroup(i);
      for (let k = 0; k < cols; k++) {
        out[i * cols + k] = inG ? this.sum[k * n + i] / this.nRepeat : 0;
      }
    }
    this.sum.fill(0);
    this.nSample = 0;
  }
}

interface HistoInput {
  ref: Ref;
  peratom: boolean;
}

const HISTO_KEYWORDS = new Set([
  'mode', 'kind', 'file', 'append', 'ave', 'start', 'beyond', 'overwrite', 'title1', 'title2', 'title3',
]);

export class FixAveHisto extends Fix {
  readonly style = 'ave/histo';
  vectorFlag = true;
  sizeVector = 4;
  arrayFlag = true;
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private readonly lo: number;
  private readonly hi: number;
  private readonly nBins: number;
  private readonly delta: number;
  private readonly mode: 'scalar' | 'vector';
  private readonly aveMode: 'one' | 'running' | 'window';
  private readonly windowM: number;
  private readonly startStep: number;
  private readonly beyond: 'ignore' | 'end' | 'extra';
  private readonly fileName: string | null;
  private readonly fileAppend: boolean;
  private readonly overwriteFile: boolean;
  private readonly title1: string | undefined;
  private readonly title2: string | undefined;
  private readonly title3: string | undefined;
  private readonly inputs: HistoInput[];
  private readonly bins: Float64Array;
  private readonly runBins: Float64Array;
  private readonly outBins: Float64Array;
  private missing = 0;
  private winMin = Infinity;
  private winMax = -Infinity;
  private nSample = 0;
  private runMissing = 0;
  private runN = 0;
  private ring: { bins: Float64Array; missing: number }[] = [];
  private outMissing = 0;
  private outMin = 0;
  private outMax = 0;
  private headerWritten = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (ave/histo)`;
    if (args.length < 6) {
      throw new StyleError(`usage: fix ${id} group-ID ave/histo Nevery Nrepeat Nfreq lo hi Nbin value1 value2 ... [keyword args ...]`);
    }
    this.nEvery = posInt(args[0], `${what}: Nevery`);
    this.nRepeat = posInt(args[1], `${what}: Nrepeat`);
    this.nFreq = posInt(args[2], `${what}: Nfreq`);
    this.nevery = this.nEvery;
    if (this.nFreq % this.nEvery !== 0) {
      throw new StyleError(`${what}: Nfreq must be a multiple of Nevery`);
    }
    if (this.nRepeat * this.nEvery > this.nFreq) {
      throw new StyleError(`${what}: Nrepeat x Nevery cannot exceed Nfreq (the time steps contributing to the histogram cannot overlap)`);
    }
    this.lo = Number(args[3]);
    this.hi = Number(args[4]);
    if (!Number.isFinite(this.lo) || !Number.isFinite(this.hi) || !(this.hi > this.lo)) {
      throw new StyleError(`${what}: lo and hi must be numbers with hi > lo, got '${args[3]}' and '${args[4]}'`);
    }
    this.nBins = posInt(args[5], `${what}: Nbin`);
    this.delta = (this.hi - this.lo) / this.nBins;
    this.mode = 'scalar';
    this.aveMode = 'one';
    this.windowM = 1;
    this.startStep = 0;
    this.beyond = 'ignore';
    this.fileName = null;
    this.fileAppend = false;
    this.overwriteFile = false;
    let kindFlag: 'global' | 'peratom' | null = null;
    let file: string | null = null;
    let append: string | null = null;
    const words: string[] = [];
    let k = 6;
    while (k < args.length && !HISTO_KEYWORDS.has(args[k])) { words.push(args[k]); k++; }
    if (!words.length) throw new StyleError(`${what}: no input values were given`);
    for (; k < args.length;) {
      const key = args[k];
      const val = args[k + 1];
      const need = (): string => {
        if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
        return val;
      };
      switch (key) {
        case 'mode': {
          const v = need();
          if (v !== 'scalar' && v !== 'vector') throw new StyleError(`${what}: mode must be scalar or vector, got '${v}'`);
          this.mode = v;
          k += 2;
          break;
        }
        case 'kind': {
          const v = need();
          if (v !== 'global' && v !== 'peratom' && v !== 'local') {
            throw new StyleError(`${what}: kind must be global, peratom or local, got '${v}'`);
          }
          if (v === 'local') throw new StyleError(`${what}: kind local is not supported (the engine has no local quantities)`);
          kindFlag = v;
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
        case 'start': {
          const n = Number(need());
          if (!Number.isInteger(n) || n < 0) throw new StyleError(`${what}: start must be a timestep >= 0, got '${val}'`);
          this.startStep = n;
          k += 2;
          break;
        }
        case 'beyond': {
          const v = need();
          if (v === 'ignore' || v === 'end') { this.beyond = v; k += 2; break; }
          // the Description spells the extra-bins setting "extend"; the keyword list says "extra"
          if (v === 'extra' || v === 'extend') { this.beyond = 'extra'; k += 2; break; }
          throw new StyleError(`${what}: beyond must be ignore, end or extra, got '${v}'`);
        }
        case 'overwrite':
          this.overwriteFile = true;
          k += 1;
          break;
        case 'file':
          if (file !== null || append !== null) throw new StyleError(`${what}: file and append cannot both be used`);
          file = need();
          k += 2;
          break;
        case 'append':
          if (file !== null || append !== null) throw new StyleError(`${what}: file and append cannot both be used`);
          append = need();
          k += 2;
          break;
        case 'title1': this.title1 = need(); k += 2; break;
        case 'title2': this.title2 = need(); k += 2; break;
        case 'title3': this.title3 = need(); k += 2; break;
        default:
          throw new StyleError(`${what}: unknown keyword '${key}'`);
      }
    }
    this.fileName = file ?? append;
    this.fileAppend = append !== null;
    if (this.overwriteFile && this.aveMode !== 'running') {
      throw new StyleError(`${what}: the overwrite keyword can only be used with the ave running setting`);
    }
    const expanded = expandWildcards(sys, words, this.mode === 'vector' ? 'vector' : 'scalar');
    this.inputs = expanded.map((w) => this.resolveInput(w, kindFlag));
    if (this.inputs.length > 1) {
      const first = this.inputs[0].peratom;
      if (this.inputs.some((inp) => inp.peratom !== first)) {
        throw new StyleError(`${what}: "Inputs of different kinds (e.g. global and per-atom) cannot be mixed"`);
      }
    }
    const len = this.beyond === 'extra' ? this.nBins + 2 : this.nBins;
    this.bins = new Float64Array(len);
    this.runBins = new Float64Array(len);
    this.outBins = new Float64Array(len);
  }

  /** Resolves one input word to a global or per-atom source, per the mode/kind rules. */
  private resolveInput(w: string, kindFlag: 'global' | 'peratom' | null): HistoInput {
    const what = `fix ${this.id} (ave/histo)`;
    const r = parseRef(w, true);
    if (r.kind === 'attr') return { ref: r, peratom: true };
    if (r.kind === 'v') {
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`${what}: variable ${r.id} does not exist`);
      if (v.style === 'atom') {
        if (r.index !== null) throw new StyleError(`${what}: ${w}: an atom-style variable has no columns`);
        return { ref: r, peratom: true };
      }
      if (v.style === 'equal') {
        if (r.index !== null) throw new StyleError(`${what}: ${w}: an equal-style variable has no elements`);
        return { ref: r, peratom: false };
      }
      if (v.style === 'vector') {
        if (this.mode === 'scalar' && r.index === null) {
          throw new StyleError(`${what}: ${w}: "a vector-style variable requires a bracketed term to specify the Ith element of the vector" in mode scalar`);
        }
        if (this.mode === 'vector' && r.index !== null) {
          throw new StyleError(`${what}: ${w}: "The vector-style variable must be used without a bracketed term" in mode vector`);
        }
        return { ref: r, peratom: false };
      }
      throw new StyleError(`${what}: ${w}: variable ${r.id} must be equal-style, vector-style or atom-style (is ${v.style}-style)`);
    }
    const obj = r.kind === 'c' ? this.sys.compute(r.id) : this.sys.fix(r.id);
    const kind = r.kind === 'c' ? 'compute' : 'fix';
    let peratom: boolean;
    if (kindFlag === 'peratom') peratom = true;
    else if (kindFlag === 'global') peratom = false;
    else if (this.mode === 'scalar') {
      // "If mode = scalar, then if no bracketed term is appended, the global scalar
      // calculated by the compute is used.  If a bracketed term is appended, the Ith
      // element of the global vector calculated by the compute is used."
      peratom = false;
    } else if (r.index === null) {
      // "If mode = vector, then if no bracketed term is appended, the global or
      // per-atom or local vector calculated by the compute is used."
      const g = obj.vectorFlag;
      const p = obj.peratomFlag && obj.sizePeratomCols === 0;
      if (g && p) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} produces both global and per-atom vectors; use the kind keyword`);
      if (!g && !p) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global or per-atom vector`);
      peratom = p;
    } else {
      const g = obj.arrayFlag;
      const p = obj.peratomFlag && obj.sizePeratomCols > 0;
      if (g && p) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} produces both a global array and per-atom values; use the kind keyword`);
      if (!g && !p) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global array or a per-atom array column`);
      peratom = p;
    }
    if (peratom) {
      if (!obj.peratomFlag) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate per-atom values`);
      if (r.index === null) {
        if (obj.sizePeratomCols !== 0) throw new StyleError(`${what}: ${w}: it calculates a per-atom array; give a column, e.g. ${w}[1]`);
      } else if (obj.sizePeratomCols === 0) {
        throw new StyleError(`${what}: ${w}: it calculates a per-atom vector, which has no columns`);
      } else if (r.index > obj.sizePeratomCols) {
        throw new StyleError(`${what}: ${w}: column out of range 1..${obj.sizePeratomCols}`);
      }
    } else if (this.mode === 'scalar') {
      if (r.index === null) {
        if (!obj.scalarFlag) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global scalar`);
      } else if (!obj.vectorFlag) {
        throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global vector (the bracketed term selects a vector element in mode scalar)`);
      }
    } else if (r.index === null) {
      if (!obj.vectorFlag) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global vector`);
    } else {
      if (!obj.arrayFlag) throw new StyleError(`${what}: ${w}: ${kind} ${r.id} does not calculate a global array`);
      if (r.index > obj.sizeArrayCols) throw new StyleError(`${what}: ${w}: column out of range 1..${obj.sizeArrayCols}`);
    }
    return { ref: r, peratom };
  }

  setup(): void {
    this.endOfStep();
  }

  endOfStep(): void {
    const step = this.sys.state.step;
    if (step % this.nevery !== 0) return;
    if (step < this.startStep) return;
    const rem = step % this.nFreq;
    const nextOut = rem === 0 ? step : step + (this.nFreq - rem);
    if (step < nextOut - (this.nRepeat - 1) * this.nEvery) return;
    this.sample();
    if (step % this.nFreq !== 0) return;
    if (this.nSample === this.nRepeat) this.produce(step);
    else {
      // window cannot be complete yet (e.g. step 0 with Nrepeat > 1): discard it
      this.bins.fill(0);
      this.missing = 0;
      this.winMin = Infinity;
      this.winMax = -Infinity;
      this.nSample = 0;
    }
  }

  private sample(): void {
    const sys = this.sys;
    const n = sys.state.n;
    for (const inp of this.inputs) {
      if (inp.peratom) {
        // "For per-atom input values, only atoms in the group contribute to the histogram"
        const vals = peratomValues(sys, inp.ref);
        for (let i = 0; i < n; i++) {
          if (this.inGroup(i)) this.tally(vals[i]);
        }
      } else if (this.mode === 'scalar') {
        this.tally(globalScalar(sys, inp.ref));
      } else {
        const vec = globalVector(sys, inp.ref);
        for (let k = 0; k < vec.length; k++) this.tally(vec[k]);
      }
    }
    this.nSample++;
  }

  private tally(v: number): void {
    if (v < this.winMin) this.winMin = v;
    if (v > this.winMax) this.winMax = v;
    if (v < this.lo || v > this.hi) {
      if (this.beyond === 'ignore') {
        this.missing++;
        return;
      }
      this.bins[v < this.lo ? 0 : this.bins.length - 1]++;
      return;
    }
    // "Values on a bin boundary are assigned to the lower of the two bins"
    const t = (v - this.lo) / this.delta;
    let idx = Math.ceil(t) - 1;
    if (idx < 0) idx = 0;
    if (idx >= this.nBins) idx = this.nBins - 1;
    this.bins[this.beyond === 'extra' ? idx + 1 : idx]++;
  }

  /** Builds the output histogram from the window's samples and applies the ave one/running/window mode. */
  private produce(step: number): void {
    const outB = new Float64Array(this.bins.length);
    let outM: number;
    if (this.aveMode === 'one') {
      outB.set(this.bins);
      outM = this.missing;
    } else if (this.aveMode === 'running') {
      this.runN++;
      for (let i = 0; i < this.bins.length; i++) this.runBins[i] += this.bins[i];
      this.runMissing += this.missing;
      for (let i = 0; i < outB.length; i++) outB[i] = this.runBins[i] / this.runN;
      outM = this.runMissing / this.runN;
    } else {
      this.ring.push({ bins: this.bins.slice(), missing: this.missing });
      if (this.ring.length > this.windowM) this.ring.shift();
      let miss = 0;
      for (const e of this.ring) {
        for (let i = 0; i < outB.length; i++) outB[i] += e.bins[i];
        miss += e.missing;
      }
      const inv = 1 / this.ring.length;
      for (let i = 0; i < outB.length; i++) outB[i] *= inv;
      outM = miss * inv;
    }
    this.outBins.set(outB);
    this.outMissing = outM;
    this.outMin = Number.isFinite(this.winMin) ? this.winMin : 0;
    this.outMax = Number.isFinite(this.winMax) ? this.winMax : 0;
    this.bins.fill(0);
    this.missing = 0;
    this.winMin = Infinity;
    this.winMax = -Infinity;
    this.nSample = 0;
    if (this.fileName) this.writeSection(step);
  }

  /** Sum of the output bin counts — "the total count" the normalized counts divide by. */
  private total(): number {
    let t = 0;
    for (let i = 0; i < this.outBins.length; i++) t += this.outBins[i];
    return t;
  }

  /** The "coordinate for the center of the bin (between lo and hi)"; extra bins print lo and hi. */
  private coordOf(i: number): number {
    if (this.beyond === 'extra') {
      if (i === 0) return this.lo;
      if (i === this.nBins + 1) return this.hi;
      return this.lo + (i - 0.5) * this.delta;
    }
    return this.lo + (i + 0.5) * this.delta;
  }

  private buildHeader(): string[] {
    return [
      this.title1 ?? `# Histogrammed data for fix ${this.id}`,
      this.title2 ?? '# TimeStep Number-of-bins Total-counts Missing-counts Min-value Max-value',
      this.title3 ?? '# Bin Coord Count Count/Total',
    ];
  }

  private writeSection(step: number): void {
    const total = this.total();
    const lines = [
      `${step} ${this.outBins.length} ${fmtG(total)} ${fmtG(this.outMissing)} ${fmt16(this.outMin)} ${fmt16(this.outMax)}`,
    ];
    for (let i = 0; i < this.outBins.length; i++) {
      const c = this.outBins[i];
      lines.push(`${i + 1} ${fmtG(this.coordOf(i))} ${fmtG(c)} ${fmtG(total > 0 ? c / total : 0)}`);
    }
    const header = this.buildHeader();
    if (this.overwriteFile) {
      this.sys.writeFile(this.fileName!, [...header, ...lines].join('\n') + '\n', false);
      return;
    }
    const wasWritten = this.headerWritten;
    this.headerWritten = true;
    const head = wasWritten ? '' : header.join('\n') + '\n';
    this.sys.writeFile(this.fileName!, head + lines.join('\n') + '\n', this.fileAppend || wasWritten);
  }

  computeVector(i: number): number {
    if (!this.vectorFlag) throw new StyleError(`fix ${this.id} does not compute a global vector`);
    if (i === 0) return this.total();
    if (i === 1) return this.outMissing;
    if (i === 2) return this.outMin;
    if (i === 3) return this.outMax;
    throw new StyleError(`fix ${this.id} (ave/histo): vector index out of range 1..4`);
  }

  computeArray(i: number, j: number): number {
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} does not compute a global array`);
    if (i < 0 || i >= this.outBins.length || j < 0 || j > 2) {
      throw new StyleError(`fix ${this.id} (ave/histo): array index out of range (${this.outBins.length} rows, 3 columns)`);
    }
    if (j === 0) return this.coordOf(i);
    if (j === 1) return this.outBins[i];
    const t = this.total();
    return t > 0 ? this.outBins[i] / t : 0;
  }
}
