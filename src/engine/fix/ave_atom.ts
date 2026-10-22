import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { localOutputOf } from '../compute/local_dump';
import { globalScalar, globalVector, peratomValues, type Ref } from '../refs';
import { formatNumber } from '../script';
import { aveNextValid } from './util';
import { inumeric, missingArgs, numeric } from '../force/util';
import { argInfo, EQUAL_STYLES, expandArgs } from '../argInfo';
import { errorUrl } from '../formula';

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


const fmtG = (v: number): string => formatNumber(v, '%g');
// min and max of ave/histo are written with 17 significant digits (measured with native LAMMPS, black box: the file
// holds -2.0800551807377268 and 2.0859149625104054)
const fmt17 = (v: number): string => formatNumber(v, '%.17g');

export class FixAveAtom extends Fix {
  readonly style = 'ave/atom';
  peratomFlag = true;
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private readonly inputs: Ref[];
  private readonly nvals: number;
  /** The next step that samples, and the samples taken of the current window (native nvalid and irepeat). */
  private nvalid: number;
  private irepeat = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_ave_atom.cpp constructor at b332f28): the values (expanded first) are atom
    // attributes or c_, f_ and v_ references, then the step counts and the sources are checked
    if (args.length < 4) throw missingArgs('fix ave/atom');
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    const words = expandArgs(sys, args.slice(3), 1).words;
    this.inputs = words.map((w): Ref => {
      if (ATOM_WORDS.has(w)) return { kind: 'attr', id: w, index: null, text: w };
      const a = argInfo(w);
      if (a.which === 'unknown' || a.which === 'none' || a.dim > 1) throw new StyleError(`Invalid fix ave/atom argument: ${w}`);
      if (a.index1 < 0) throw new StyleError(`fix ave/atom: index ${a.index1} in ${w} is refused: native LAMMPS reads outside the array for it`);
      return { kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: w };
    });
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/atom nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal fix ave/atom nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/atom nfreq value: ${nfreq}`);
    if (nfreq % nevery || nrepeat * nevery > nfreq) throw new StyleError('Inconsistent fix ave/atom nevery/nrepeat/nfreq values');
    this.nEvery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    this.nevery = nevery;
    this.peratomFreq = nfreq;
    for (const r of this.inputs) {
      if (r.kind === 'c' || r.kind === 'f') {
        const what = r.kind === 'c' ? 'compute' : 'fix';
        const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id) : sys.fixes.find((x) => x.id === r.id);
        if (!o) throw new StyleError(`${r.kind === 'c' ? 'Compute' : 'Fix'} ID ${r.id} for fix ave/atom does not exist`);
        if (!o.peratomFlag) throw new StyleError(`Fix ave/atom ${what} ${r.id} does not calculate per-atom values`);
        if (r.index === null && o.sizePeratomCols !== 0) throw new StyleError(`Fix ave/atom ${what} ${r.id} does not calculate a per-atom vector`);
        if (r.index !== null && o.sizePeratomCols === 0) throw new StyleError(`Fix ave/atom ${what} ${r.id} does not calculate a per-atom array`);
        if (r.index !== null && r.index > o.sizePeratomCols) throw new StyleError(`Fix ave/atom ${what} ${r.id} array is accessed out-of-range${errorUrl(20)}`);
        if (r.kind === 'f' && (o as Fix).peratomFreq > 0 && nevery % (o as Fix).peratomFreq) {
          throw new StyleError(`Fix ${r.id} for fix ave/atom not computed at compatible time${errorUrl(7)}`);
        }
      } else if (r.kind === 'v') {
        const v = sys.vars.get(r.id);
        if (!v) throw new StyleError(`Variable name ${r.id} for fix ave/atom does not exist`);
        if (v.style !== 'atom') throw new StyleError(`Fix ave/atom variable ${r.id} is not atom-style variable`);
      }
    }
    this.nvals = this.inputs.length;
    this.sizePeratomCols = this.nvals === 1 ? 0 : this.nvals;
    this.alloc();
    this.nvalid = aveNextValid(sys.state.step, this.nEvery, this.nRepeat, this.nFreq);
  }

  /** The sources must still exist (native init). */
  private checkSources(): void {
    for (const r of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ave/atom does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for fix ave/atom does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/atom does not exist`);
    }
  }

  /**
   * Sizes the per-atom output to the atoms: atoms added since keep the values of the atoms before them (the engine
   * appends new atoms) and start at 0.
   */
  private alloc(): void {
    const n = this.sys.state.n;
    const need = this.nvals * n;
    const old = this.nvals === 1 ? this.vectorAtom : this.arrayAtom;
    if (old.length === need) return;
    const out = new Float64Array(need);
    out.set(old.subarray(0, Math.min(old.length, need)));
    if (this.nvals === 1) this.vectorAtom = out;
    else this.arrayAtom = out;
  }

  /** As native LAMMPS does (src/fix_ave_atom.cpp init at b332f28): a minimization past the next sample step restarts the window. */
  init(): void {
    this.checkSources();
    this.alloc();
    const step = this.sys.state.step;
    if (this.nvalid < step) {
      this.irepeat = 0;
      this.nvalid = aveNextValid(step, this.nEvery, this.nRepeat, this.nFreq);
    }
  }

  setup(): void {
    this.endOfStep();
  }

  /**
   * As native LAMMPS does (src/fix_ave_atom.cpp end_of_step at b332f28): the per-atom output itself holds the sums of
   * the window (zeroed by its first sample, so it reads as a partial sum until the Nrepeat-th sample divides it);
   * atoms outside the group stay 0.
   */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    this.alloc();
    const n = this.sys.state.n;
    const cols = this.nvals;
    const out = cols === 1 ? this.vectorAtom : this.arrayAtom;
    if (this.irepeat === 0) out.fill(0);
    for (let k = 0; k < cols; k++) {
      const vals = peratomValues(this.sys, this.inputs[k]);
      for (let i = 0; i < n; i++) {
        if (this.inGroup(i)) out[i * cols + k] += vals[i];
      }
    }
    this.irepeat++;
    if (this.irepeat < this.nRepeat) {
      this.nvalid += this.nEvery;
      return;
    }
    this.irepeat = 0;
    this.nvalid = step + this.nFreq - (this.nRepeat - 1) * this.nEvery;
    for (let i = 0; i < out.length; i++) out[i] /= this.nRepeat;
  }
}

/** The per-atom attributes fix ave/histo takes as values. */
const ATOM_WORDS = new Set(['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']);

interface HistoInput {
  ref: Ref;
  peratom: boolean;
  /** Local values of a compute (every row, or one column, is tallied). */
  local?: boolean;
}

export class FixAveHisto extends Fix {
  // typed as string: the weighted variant (src/engine-gpl, fix ave/histo/weight) is a subclass
  readonly style: string = 'ave/histo';
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
  protected readonly mode: 'scalar' | 'vector';
  private readonly aveMode: 'one' | 'running' | 'window';
  private readonly windowM: number;
  private readonly startStep: number;
  private readonly beyond: 'ignore' | 'end' | 'extra';
  private readonly fileName: string | null;
  private readonly fileAppend: boolean;
  private readonly overwriteFile: boolean;
  private title1: string | undefined;
  private title2: string | undefined;
  private title3: string | undefined;
  protected readonly inputs: HistoInput[];
  private readonly bins: Float64Array;
  private readonly runBins: Float64Array;
  private readonly outBins: Float64Array;
  private missing = 0;
  private winMin = Infinity;
  private winMax = -Infinity;
  /** The next step that samples, and the samples taken of the current window (native nvalid and irepeat). */
  private nvalid = 0;
  private irepeat = 0;
  private runMissing = 0;
  private runMin = 0;
  private runMax = 0;
  private ring: { bins: Float64Array; missing: number; total: number; min: number; max: number }[] = [];
  private outMissing = 0;
  /** The total weight of the window's samples (native stats[0]), kept apart from the bins' sum (round-off). */
  private winTotal = 0;
  private runTotal = 0;
  private ringTotal = 0;
  private outTotal = 0;
  private outMin = 0;
  private outMax = 0;
  private headerWritten = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_ave_histo.cpp constructor and options at b332f28); the weighted subclass names
    // its own style in the messages
    const style = (new.target as unknown as { histoStyle?: string }).histoStyle ?? 'ave/histo';
    const cmd = `fix ${style}`;
    if (args.length < 7) throw missingArgs(cmd);
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    this.globalFreq = nfreq;
    this.lo = numeric(args[3]);
    this.hi = numeric(args[4]);
    const nbins = inumeric(args[5]);
    let k = 6;
    while (k < args.length && (ATOM_WORDS.has(args[k]) || /^[cfv]_/.test(args[k]))) k++;
    const valueWords = args.slice(6, k);
    if (!valueWords.length) throw new StyleError(`No values in from computes, fixes, or variables used in ${cmd} command`);
    // options (native's messages name ave/histo for the keyword values, and its missing-argument messages for the
    // title keywords have two blanks)
    let kind: 'default' | 'global' | 'peratom' | 'local' = 'default';
    let aveMode: 'one' | 'running' | 'window' = 'one';
    let windowM = 1;
    let startStep = 0;
    let mode: 'scalar' | 'vector' = 'scalar';
    let beyond: 'ignore' | 'end' | 'extra' = 'ignore';
    let overwrite = false;
    let fileName: string | null = null;
    let fileAppend = false;
    const need = (n: number, what: string): void => { if (k + n > args.length) throw missingArgs(what); };
    while (k < args.length) {
      const key = args[k];
      const v = args[k + 1];
      if (key === 'file' || key === 'append') {
        need(2, `${cmd} ${key}`);
        fileName = v;
        fileAppend = key === 'append';
        k += 2;
      } else if (key === 'kind') {
        need(2, `${cmd} kind`);
        if (v !== 'global' && v !== 'peratom' && v !== 'local') throw new StyleError(`Unknown fix ave/histo kind option: ${v}`);
        kind = v;
        k += 2;
      } else if (key === 'ave') {
        need(2, `${cmd} ave`);
        if (v !== 'one' && v !== 'running' && v !== 'window') throw new StyleError(`Unknown fix ave/histo ave option: ${v}`);
        aveMode = v;
        if (aveMode === 'window') {
          need(3, `${cmd} ave window`);
          windowM = inumeric(args[k + 2]);
          if (windowM <= 0) throw new StyleError(`Illegal fix ave/histo ave window size: ${windowM}`);
        }
        k += aveMode === 'window' ? 3 : 2;
      } else if (key === 'start') {
        need(2, `${cmd} start`);
        startStep = inumeric(v);
        k += 2;
      } else if (key === 'mode') {
        need(2, `${cmd} mode`);
        if (v !== 'scalar' && v !== 'vector') throw new StyleError(`Unknown fix ave/histo mode option: ${v}`);
        mode = v;
        k += 2;
      } else if (key === 'beyond') {
        need(2, `${cmd} beyond`);
        if (v !== 'ignore' && v !== 'end' && v !== 'extra') throw new StyleError(`Unknown fix ave/histo beyond option: ${v}`);
        beyond = v;
        k += 2;
      } else if (key === 'overwrite') {
        overwrite = true;
        k += 1;
      } else if (key === 'title1' || key === 'title2' || key === 'title3') {
        need(2, `${cmd}  ${key}`);
        if (key === 'title1') this.title1 = v;
        else if (key === 'title2') this.title2 = v;
        else this.title3 = v;
        k += 2;
      } else throw new StyleError(`Unknown ${cmd} option: ${key}`);
    }
    this.mode = mode;
    this.aveMode = aveMode;
    this.windowM = windowM;
    this.startStep = startStep;
    this.beyond = beyond;
    this.fileName = fileName;
    this.fileAppend = fileAppend;
    this.overwriteFile = overwrite;
    // the values: atom attributes, then c_, f_ and v_ references (expanded first); a word that is no reference ends them
    const words = expandArgs(sys, valueWords, mode === 'vector' ? 1 : 0).words;
    const refs: Ref[] = [];
    for (const w of words) {
      if (ATOM_WORDS.has(w)) { refs.push({ kind: 'attr', id: w, index: null, text: w }); continue; }
      const a = argInfo(w);
      if (a.which === 'none') break;
      if (a.which === 'unknown' || a.dim > 1) throw new StyleError(`Invalid ${cmd} argument: ${w}`);
      if (a.index1 < 0) throw new StyleError(`${cmd}: index ${a.index1} in ${w} is refused: native LAMMPS reads outside the vector for it`);
      refs.push({ kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: w });
    }
    if (refs.length !== words.length) throw new StyleError(`Could not parse value data consistently for ${cmd}`);
    if (nevery <= 0) throw new StyleError(`Illegal ${cmd} nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal ${cmd} nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal ${cmd} nfreq value: ${nfreq}`);
    if (nfreq % nevery || nrepeat * nevery > nfreq) throw new StyleError(`Inconsistent ${cmd} nevery/nrepeat/nfreq values`);
    if (aveMode !== 'running' && overwrite) throw new StyleError(`${cmd} overwrite keyword requires ave running setting`);
    this.nEvery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    this.nevery = nevery;
    // the kind of the values: given, or what the first value produces; every value must produce it
    for (const r of refs) {
      let g = false, pa = false, lc = false;
      if (r.kind === 'attr') pa = true;
      else if (r.kind === 'c' || r.kind === 'f') {
        const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id) : sys.fixes.find((x) => x.id === r.id);
        if (!o) throw new StyleError(`${r.kind === 'c' ? 'Compute' : 'Fix'} ID ${r.id} for ${cmd} does not exist`);
        g = o.scalarFlag || o.vectorFlag || o.arrayFlag;
        pa = o.peratomFlag;
        lc = o.localFlag;
      } else {
        const v = sys.vars.get(r.id);
        if (!v) throw new StyleError(`Variable name ${r.id} for ${cmd} does not exist`);
        if (EQUAL_STYLES.has(v.style) || v.style === 'vector') g = true;
        else if (v.style === 'atom') pa = true;
        else throw new StyleError(`${cmd} variable ${r.id} is incompatible style`);
      }
      if (kind === 'default') {
        if (Number(g) + Number(pa) + Number(lc) > 1) throw new StyleError(`${cmd} input kind is ambiguous`);
        if (g) kind = 'global';
        if (pa) kind = 'peratom';
        if (lc) kind = 'local';
      } else if (kind === 'global' && !g) throw new StyleError(`${cmd} input kind is not global`);
      else if (kind === 'peratom' && !pa) throw new StyleError(`${cmd} input kind is not peratom`);
      else if (kind === 'local' && !lc) throw new StyleError(`${cmd} input kind is not local`);
    }
    if (kind === 'peratom' && mode === 'scalar') throw new StyleError(`${cmd} cannot process per-atom values in scalar mode`);
    if (kind === 'local' && mode === 'scalar') throw new StyleError(`${cmd} cannot process local values in scalar mode`);
    for (const r of refs) this.checkInput(cmd, r, kind);
    this.inputs = refs.map((ref) => ({ ref, peratom: kind === 'peratom', local: kind === 'local' }));
    // only a positive bin count and distinct bounds bin anything (native writes outside its histogram for the others)
    if (nbins <= 0) throw new StyleError(`${cmd}: Nbin ${nbins} is refused: native LAMMPS writes outside its histogram for it`);
    if (this.hi === this.lo) throw new StyleError(`${cmd}: lo = hi is refused: native LAMMPS bins with an infinite inverse width for it`);
    this.nBins = nbins;
    this.delta = (this.hi - this.lo) / this.nBins;
    const len = this.beyond === 'extra' ? this.nBins + 2 : this.nBins;
    // fix_ave_histo.html: "The global array has Nbins rows and three columns" (Nbins + 2 with beyond extra)
    this.sizeArrayRows = len;
    this.sizeArrayCols = 3;
    this.bins = new Float64Array(len);
    this.runBins = new Float64Array(len);
    this.outBins = new Float64Array(len);
    this.nvalid = aveNextValid(sys.state.step, this.nEvery, this.nRepeat, this.nFreq, this.startStep);
  }

  /** One value's source, checked as native's constructor does for the kind and the mode. */
  private checkInput(cmd: string, r: Ref, kind: string): void {
    const sys = this.sys;
    const vector = this.mode === 'vector';
    if (r.kind === 'c' || r.kind === 'f') {
      const what = r.kind === 'c' ? 'compute' : 'fix';
      const o = r.kind === 'c' ? sys.computes.find((x) => x.id === r.id)! : sys.fixes.find((x) => x.id === r.id)!;
      const j = r.index ?? 0;
      const url = r.kind === 'f' ? errorUrl(20) : '';
      if (kind === 'global' && !vector) {
        if (j === 0 && !o.scalarFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a global scalar`);
        if (j && !o.vectorFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a global vector`);
        // an engine compute that sizes its vector when it is first computed reports 0 here
        if (j && o.sizeVector > 0 && j > o.sizeVector) throw new StyleError(`${cmd} ${what} ${r.id} vector is accessed out-of-range${url}`);
      } else if (kind === 'global') {
        if (j === 0 && !o.vectorFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a global vector`);
        if (j && !o.arrayFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a global array`);
        if (j && j > o.sizeArrayCols) throw new StyleError(`${cmd} ${what} ${r.id} array is accessed out-of-range${url}`);
      } else if (kind === 'peratom') {
        if (!o.peratomFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate per-atom values`);
        // native's fix message starts with a blank here
        if (j === 0 && o.sizePeratomCols !== 0) throw new StyleError(`${r.kind === 'f' ? ' ' : ''}${cmd} ${what} ${r.id} does not calculate a per-atom vector`);
        if (j && o.sizePeratomCols === 0) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a per-atom array`);
        if (j && j > o.sizePeratomCols) throw new StyleError(`${cmd} ${what} ${r.id} array is accessed out-of-range${url}`);
      } else {
        if (!o.localFlag) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate local values`);
        if (j === 0 && o.sizeLocalCols !== 0) throw new StyleError(`${cmd} ${what} ${r.id} does not calculate a local vector`);
        // native's fix message has no ID here
        if (j && o.sizeLocalCols === 0) throw new StyleError(r.kind === 'f' ? `${cmd} fix does not calculate a local array` : `${cmd} compute ${r.id} does not calculate a local array`);
        if (j && j > o.sizeLocalCols) throw new StyleError(`${cmd} ${what} ${r.id} array is accessed out-of-range${errorUrl(20)}`);
      }
      // native compares Nevery with a fix's global frequency for every kind
      if (r.kind === 'f' && (o as Fix).globalFreq > 0 && this.nEvery % (o as Fix).globalFreq) {
        throw new StyleError(`Fix ${r.id} for ${cmd} not computed at compatible time${errorUrl(7)}`);
      }
    } else if (r.kind === 'v') {
      const v = sys.vars.get(r.id)!;
      if (kind === 'global' && !vector) {
        if (r.index === null && !EQUAL_STYLES.has(v.style)) throw new StyleError(`${cmd} variable ${r.id} is not equal-style variable`);
        if (r.index !== null && v.style !== 'vector') throw new StyleError(`${cmd} variable ${r.id} is not vector-style variable`);
      } else if (kind === 'global') {
        if (r.index === null && v.style !== 'vector') throw new StyleError(`${cmd} variable ${r.id} is not vector-style variable`);
        if (r.index !== null) throw new StyleError(`${cmd} variable ${r.id} cannot be indexed`);
      } else if (kind === 'peratom') {
        if (r.index === null && v.style !== 'atom') throw new StyleError(`${cmd} variable ${r.id} is not atom-style variable`);
        if (r.index !== null) throw new StyleError(`${cmd} variable ${r.id} cannot be indexed`);
      }
    }
  }

  /**
   * As native LAMMPS does (src/fix_ave_histo.cpp init at b332f28): the sources must still exist, and a minimization
   * past the next sample step restarts the window.
   */
  init(): void {
    const style = (this.constructor as unknown as { histoStyle?: string }).histoStyle ?? 'ave/histo';
    for (const { ref: r } of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ${style} does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for fix ${style} does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ${style} does not exist`);
    }
    const step = this.sys.state.step;
    if (this.nvalid < step) {
      this.irepeat = 0;
      this.nvalid = aveNextValid(step, this.nEvery, this.nRepeat, this.nFreq, this.startStep);
    }
  }

  setup(): void {
    this.endOfStep();
  }

  /**
   * As native LAMMPS does (src/fix_ave_histo.cpp end_of_step at b332f28): only the scheduled step samples; the first
   * sample of a window zeroes the histogram, the Nrepeat-th produces the output and schedules the next window to end
   * Nfreq steps later.
   */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    if (this.irepeat === 0) {
      this.bins.fill(0);
      this.missing = 0;
      this.winTotal = 0;
      this.winMin = Infinity;
      this.winMax = -Infinity;
    }
    this.sampleInputs();
    this.irepeat++;
    if (this.irepeat < this.nRepeat) {
      this.nvalid += this.nEvery;
      return;
    }
    this.irepeat = 0;
    this.nvalid = step + this.nFreq - (this.nRepeat - 1) * this.nEvery;
    this.produce(step);
  }

  /**
   * Tallies the values of one repeat (the inputs in order). The weighted variant (fix ave/histo/weight, in
   * src/engine-gpl) overrides this with a value/weight pair per sample.
   */
  protected sampleInputs(): void {
    const sys = this.sys;
    const n = sys.state.n;
    for (const inp of this.inputs) {
      if (inp.local) {
        const comp = localOutputOf(sys, inp.ref.kind === 'f' ? 'f' : 'c', inp.ref.id);
        const data = comp.localValues();
        const ncol = Math.max(1, comp.sizeLocalCols);
        const col = inp.ref.index === null ? 0 : inp.ref.index - 1;
        for (let r = 0; r < comp.localRows; r++) this.tally(data[r * ncol + col]);
      } else if (inp.peratom) {
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
  }

  /**
   * Bins one value with weight w (1 for the plain histogram). A value outside [lo, hi] with beyond ignore adds
   * its weight to the missing count; the min and max count every value whatever its weight.
   */
  protected tally(v: number, w = 1): void {
    if (v < this.winMin) this.winMin = v;
    if (v > this.winMax) this.winMax = v;
    if (v < this.lo || v > this.hi) {
      if (this.beyond === 'ignore') {
        this.missing += w;
        return;
      }
      this.bins[v < this.lo ? 0 : this.bins.length - 1] += w;
      this.winTotal += w;
      return;
    }
    this.winTotal += w;
    // Bin assignment: the docs (fix_ave_histo.html) say "Values on a bin boundary are assigned to the
    // lower of the two bins." Measured with native LAMMPS (black box): values exactly on an interior
    // edge go to the UPPER bin (lo 0, hi 3, 6 bins: 0.5 -> bin 2, 1.0 -> bin 3, 2.5 -> bin 6; lo -1.5,
    // hi 1.5: -1.0 -> bin 2, -0.5 -> bin 3), and v = hi goes to the last bin. Native multiplies by the
    // inverse width: lo 0, hi 1, 10 bins: 0.3 -> bin 4 (0.3/0.1 = 2.9999999999999996 would give bin 3),
    // 0.6 -> bin 7, 0.7 -> bin 8, 0.9 -> bin 10. So idx = floor((v - lo) * nBins / (hi - lo)).
    // As native LAMMPS does (src/fix_ave_histo.cpp bin_one at b332f28): the inverse of the bin width, truncated
    let idx = Math.trunc((v - this.lo) * (1.0 / this.delta));
    if (idx >= this.nBins) idx = this.nBins - 1;
    this.bins[this.beyond === 'extra' ? idx + 1 : idx] += w;
  }

  /**
   * Builds the output histogram from the window's samples and applies the ave one/running/window mode. Native
   * fix_ave_histo.cpp (b332f28, end_of_step) sums the histograms of ave running and ave window and does not divide
   * them: bin_total[i] += bin[i], stats_total[0] += stats[0] (the normalized counts divide by that total when written).
   * Its running min and max start from 0 (stats_total is zeroed in the constructor), the window's are the min and
   * max of the window's entries.
   */
  private produce(step: number): void {
    const outB = new Float64Array(this.bins.length);
    let outM: number;
    // an empty window keeps the native start values (fix_ave_histo.cpp end_of_step: "stats[2] = BIG; stats[3] = -BIG;", BIG = 1.0e20)
    let outMin = Number.isFinite(this.winMin) ? this.winMin : 1.0e20;
    let outMax = Number.isFinite(this.winMax) ? this.winMax : -1.0e20;
    if (this.aveMode === 'one') {
      outB.set(this.bins);
      outM = this.missing;
      this.outTotal = this.winTotal;
    } else if (this.aveMode === 'running') {
      this.runTotal += this.winTotal;
      this.runMissing += this.missing;
      for (let i = 0; i < this.bins.length; i++) this.runBins[i] += this.bins[i];
      this.runMin = Math.min(this.runMin, outMin);
      this.runMax = Math.max(this.runMax, outMax);
      outB.set(this.runBins);
      outM = this.runMissing;
      this.outTotal = this.runTotal;
      outMin = this.runMin;
      outMax = this.runMax;
    } else {
      // native (the window branch): the new entry is added and the one it overwrites taken out
      this.ringTotal += this.winTotal;
      this.ring.push({ bins: this.bins.slice(), missing: this.missing, total: this.winTotal, min: outMin, max: outMax });
      if (this.ring.length > this.windowM) {
        this.ringTotal -= this.ring[0].total;
        this.ring.shift();
      }
      this.outTotal = this.ringTotal;
      let miss = 0;
      let lo = Infinity;
      let hi = -Infinity;
      for (const e of this.ring) {
        for (let i = 0; i < outB.length; i++) outB[i] += e.bins[i];
        miss += e.missing;
        lo = Math.min(lo, e.min);
        hi = Math.max(hi, e.max);
      }
      outM = miss;
      outMin = lo;
      outMax = hi;
    }
    this.outBins.set(outB);
    this.outMissing = outM;
    this.outMin = outMin;
    this.outMax = outMax;
    this.bins.fill(0);
    this.missing = 0;
    this.winTotal = 0;
    this.winMin = Infinity;
    this.winMax = -Infinity;
    if (this.fileName) this.writeSection(step);
  }

  /** "The total count" the normalized counts divide by (native stats_total[0]). */
  private total(): number {
    return this.outTotal;
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
      // the leading line is printed with full precision: native prints it with "{}" (src/fix_ave_histo.cpp at
      // b332f28, utils::print of stats_total), so the total and missing counts are not cut to %g
      `${step} ${this.outBins.length} ${fmt17(total)} ${fmt17(this.outMissing)} ${fmt17(this.outMin)} ${fmt17(this.outMax)}`,
    ];
    for (let i = 0; i < this.outBins.length; i++) {
      const c = this.outBins[i];
      // native divides when the total is non-zero (src/fix_ave_histo.cpp at b332f28: `if (stats_total[0] != 0.0)`):
      // weighted totals can be negative
      lines.push(`${i + 1} ${fmtG(this.coordOf(i))} ${fmtG(c)} ${fmtG(total !== 0 ? c / total : 0)}`);
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
    return t !== 0 ? this.outBins[i] / t : 0;
  }
}
