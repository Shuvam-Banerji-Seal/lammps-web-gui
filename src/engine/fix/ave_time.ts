import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { aveNextValid } from './util';
import { globalScalar, globalVector, type Ref } from '../refs';
import { errorUrl, evaluateScalar } from '../formula';
import { inumeric, missingArgs } from '../force/util';
import { argInfo, expandArgs } from '../argInfo';
import { checkCFormat, compileCFormat } from '../cformat';
import { formatNumber, substituteVariables } from '../script';

/*
 * fix ave/time and fix print — implemented only from the cited docs.lammps.org
 * pages (never from LAMMPS source code). Every quoted fragment below is copied
 * character for character from plans/lammps-docs/fix_ave_time.rst or
 * fix_print.rst (the sources of docs.lammps.org/fix_ave_time.html and
 * docs.lammps.org/fix_print.html); each fragment lies within one source line,
 * and adjacent quoted lines are consecutive source lines.
 *
 * ---------------------------------------------------------------- fix ave/time
 * Syntax:
 *
 *   "fix ID group-ID ave/time Nevery Nrepeat Nfreq value1 value2 ... keyword args ..."
 *
 * "Nevery = use input values every this many time steps"; "Nrepeat = # of
 * times to use input values for calculating averages"; "Nfreq = calculate
 * averages every this many time steps"; "one or more input values can be
 * listed"; "value = c_ID, c_ID[N], f_ID, f_ID[N], v_name"; "keyword = *mode*
 * or *file* or *append* or *ave* or *start* or *off* or *overwrite* or
 * *format* or *title1* or *title2* or *title3*".
 * "*mode* arg = *scalar* or *vector*" ("scalar = all input values are global
 * scalars"; "vector = all input values are global vectors or global arrays");
 * "*ave* args = *one* or *running* or *window M*" ("one = output a new average
 * value every Nfreq steps"; "running = output cumulative average of all
 * previous Nfreq steps"; "window M = output average of M most recent Nfreq
 * steps"); "*start* args = Nstart" ("Nstart = start averaging on this time
 * step"); "*off* arg = M = do not average this value" ("M = value # from 1 to
 * Nvalues"); "*file* arg = filename" ("filename = name of file to output time
 * averages to"); "*append* arg = filename" ("filename = name of file to append
 * time averages to"); "*overwrite* arg = none = overwrite output file with
 * only latest output"; "*format* arg = string" ("string = C-style format
 * string"); "*title1* arg = string" ("string = text to print as 1st line of
 * output file").
 *
 * Inputs: "The input values must either be all scalars or all vectors depending"
 * "on the setting of the *mode* keyword." — the averaging is
 * "performed independently on each input value". "If *mode* = scalar, then the
 * input values must be scalars, or vectors" "with a bracketed term appended,
 * indicating the :math:`I^\text{th}` value of the" "vector is used."
 * "If *mode* = vector, then the input values must be vectors, or arrays"
 * "with a bracketed term appended, indicating the Ith column of the array"
 * "is used.  All vectors must be the same length, which is the length of"
 * "the vector or number of rows in the array."
 *
 * Sampling schedule:
 *   "contribute to the average.  The final averaged quantities are generated on"
 *   "time steps that are a multiple of :math:`N_\text{freq}`\ .  The average is over"
 *   ":math:`N_\text{repeat}` quantities, computed in the preceding portion of the"
 *   "simulation every :math:`N_\text{every}` time steps.  :math:`N_\text{freq}` must"
 *   "be a multiple of :math:`N_\text{every}` and :math:`N_\text{every}` must be"
 *   "non-zero even if :math:`N_\text{repeat} = 1`. Also, the time steps"
 *   "contributing to the average value cannot overlap (i.e.,"
 *   ":math:`N_\text{repeat} \times N_\text{every}` cannot exceed :math:`N_\text{freq}`)."
 *   "For example, if :math:`N_\text{every}=2`, :math:`N_\text{repeat}=6`, and"
 *   ":math:`N_\text{freq}=100`, then values on time steps 90, 92, 94, 96, 98, and"
 *   "100 will be used to compute the final average on time step 100."
 * So each output step T (a multiple of Nfreq) averages the samples at
 * T-(Nrepeat-1)*Nevery .. T, and no samples outside the next output's window
 * are taken.
 *
 * ave keyword:
 *   "If the *ave* setting is *one*, then the values produced on time steps"
 *   "that are multiples of :math:`N_\text{freq}` are independent of each other; they"
 *   "are output as-is without further averaging."
 *   "If the *ave* setting is *running*, then the values produced on"
 *   "time steps that are multiples of :math:`N_\text{freq}` are summed and averaged"
 *   "in a cumulative sense before being output.  Each output value is thus"
 *   "average of the value produced on that time step with all preceding"
 *   "values.  This running average begins when the fix is defined; it can"
 *   "only be restarted by deleting the fix via the :doc:`unfix <unfix>`"
 *   "command, or by re-defining the fix by re-specifying it."
 *   "If the *ave* setting is *window*, then the values produced on"
 *   "time steps that are multiples of *Nfreq* are summed and averaged within"
 *   "a moving \"window\" of time, so that the last M values are used to"
 *   "produce the output.  For example, if :math:`M = 3` and"
 *   ":math:`N_\text{freq} = 1000`, then the output on step 10000 will be the average"
 *   "of the individual values on steps 8000, 9000, and 10000.  Outputs on early"
 *   "steps will average over less than :math:`M` values if they are not available."
 *
 *   "The *start* keyword specifies what time step averaging will begin on."
 *   "The default is step 0."
 *   "The *off* keyword can be used to flag any of the input values.  If a"
 *   "value is flagged, it will not be time averaged.  Instead the most"
 *   "recent input value will always be stored and output."
 *
 * File output:
 *   "The *file* or *append* keywords allow a filename to be specified.  If"
 *   "*file* is used, then the filename is overwritten if it already exists."
 *   "If *append* is used, then the filename is appended to if it already"
 *   "exists, or created if it does not exist."
 *   "quantity or vector of quantities is written to the file for each input"
 *   "value specified in the fix ave/time command.  For *mode* = scalar, this"
 *   "means a single line is written each time output is performed."
 *   "The *overwrite* keyword will continuously overwrite the output file"
 *   "with the latest output, so that it only contains one time step worth of"
 *   "output.  This option can only be used with the *ave running* setting."
 *   "The *format* keyword sets the numeric format of each value when it is"
 *   "printed to a file via the *file* keyword." "The default format is \" %g\"."
 *
 * Title lines (defaults; "In the first line, ID is replaced with the fix-ID."):
 *   "By default, these header lines are as follows for *mode* = scalar:"
 *   "# Time-averaged data for fix ID"
 *   "# TimeStep value1 value2 ..."
 *   "There is no third line in the header of the file,"
 *   "so the *title3* setting is ignored when *mode* = scalar."
 *   "By default, these header lines are as follows for *mode* = vector:"
 *   "# Time-averaged data for fix ID"
 *   "# TimeStep Number-of-rows"
 *   "# Row value1 value2 ..."
 *
 * Output shape and extensivity:
 *   "The values can only be accessed on time steps that are multiples of"
 *   ":math:`N_\text{freq}` since that is when averaging is performed."
 *   "A scalar is produced if only a single input value is averaged and"
 *   "*mode* = scalar.  A vector is produced if multiple input values are"
 *   "averaged for *mode* = scalar, or a single input value for *mode* ="
 *   "vector.  In the first case, the length of the vector is the number of"
 *   "inputs.  In the second case, the length of the vector is the same as"
 *   "the length of the input vector.  An array is produced if multiple"
 *   "input values are averaged and *mode* = vector.  The global array has #"
 *   "of rows = length of the input vectors and # of columns = number of"
 *   "inputs." ("the length of the input vector." begins the next source line.)
 *   "If the fix produces a scalar or vector, then the scalar and each"
 *   "element of the vector can be either \"intensive\" or \"extensive\","
 *   "depending on whether the values contributing to the scalar or vector"
 *   "element are \"intensive\" or \"extensive\"."
 *   "Values produced by a variable are treated as intensive."
 *
 * "The group specified with this command is ignored."
 *   "No parameter of this fix can be used with the *start/stop* keywords of"
 *   "the :doc:`run <run>` command.  This fix is not invoked during"
 *   ":doc:`energy minimization <minimize>`."
 * Default: "The option defaults are mode = scalar, ave = one, start = 0, no file"
 * "output, format = %g, title 1,2,3 = strings as described above, and no"
 * "off settings for any input values."
 *
 * ------------------------------------------------------------------ fix print
 * Syntax:
 *
 *   "fix ID group-ID print N string keyword value ..."
 *
 * "N = print every N steps; N can be a variable (see below)"; "string = text
 * string to print with optional variable names"; "keyword = *file* or *append*
 * or *screen* or *title*" with "*file* value = filename", "*append* value =
 * filename", "*screen* value = *yes* or *no*" and "*title* value = string".
 * "Print a text string every N steps during a simulation run." For a variable
 * N the variable is evaluated at the beginning of a run to determine the next
 * timestep: "beginning of a run to determine the **next** timestep at which the"
 * "string will be written out.  On that timestep, the variable will be"
 * "evaluated again to determine the next timestep, etc." — the
 * "variable should return timestep values." The string is substituted at print
 * time ($x, ${name}, $(formula[:fmt]) per docs.lammps.org/Commands_parse.html);
 * vector-style "variables are printed in a bracketed, comma-separated format,"
 * "e.g. [1,2,3,4] or [12.5,2,4.6,10.1]."
 *
 * "If *file* is" "used, then the filename is overwritten if it already exists.";
 * "*append* is used, then the filename is appended to if it already"
 * "exists, or created if it does not exist." (screen:) "logfile can be turned
 * on or off as desired." Title: "keyword was used.  By default, the title line
 * is as follows:"
 *
 *   "# Fix print output for fix ID"
 *
 * "where ID is replaced with the fix-ID." "The specified group-ID is ignored by
 * this fix." "None of the :doc:`fix_modify <fix_modify>` options are"
 * "relevant to this fix.  No global or per-atom quantities are stored by"
 * "this fix for access by various :doc:`output commands <Howto_output>`."
 *   "No parameter of this fix can be used with the *start/stop* keywords of"
 *   "the :doc:`run <run>` command.  This fix is not invoked during"
 *   ":doc:`energy minimization <minimize>`."
 * Default: "The option defaults are no file output, screen = yes, and title
 * string" "as described above."
 */

export class FixAveTime extends Fix {
  readonly style = 'ave/time';
  private readonly nEvery: number;
  private readonly nRepeat: number;
  private readonly nFreq: number;
  private readonly mode: 'scalar' | 'vector';
  private readonly aveMode: 'one' | 'running' | 'window';
  private readonly windowM: number;
  private readonly startStep: number;
  private readonly offIdx = new Set<number>();
  private readonly fileName: string | null;
  private readonly fileAppend: boolean;
  private readonly overwriteFile: boolean;
  private readonly fmt: string;
  private title1: string | undefined;
  private title2: string | undefined;
  private title3: string | undefined;

  private inputs: Ref[] = [];
  private inputWords: string[] = [];
  /** Length of the input/output vectors (1 in scalar mode). */
  private len = 1;
  private lenSet = false;
  private acc: Float64Array[] = [];
  private latest: Float64Array[] = [];
  private values: Float64Array[] = [];
  private runSum: Float64Array[] = [];
  private runN = 0;
  private ring: Float64Array[][] = [];
  private nSample = 0;
  /** The next step that samples, and the samples taken of the current window (native nvalid and irepeat). */
  private nvalid = 0;
  private irepeat = 0;
  private header: string[] | null = null;
  private headerWritten = false;
  private body: string[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // As native LAMMPS does (src/fix_ave_time.cpp constructor and options at b332f28): the values are the words that
    // start with c_, f_ or v_, the keywords follow; wildcards expand before the values are read, and the step counts
    // are checked after them
    if (args.length < 4) throw missingArgs('fix ave/time');
    const nevery = inumeric(args[0]);
    const nrepeat = inumeric(args[1]);
    const nfreq = inumeric(args[2]);
    this.globalFreq = nfreq;
    let k = 3;
    while (k < args.length && /^[cfv]_/.test(args[k])) k++;
    const valueWords = args.slice(3, k);
    if (!valueWords.length) throw new StyleError('No values from computes, fixes, or variables used in fix ave/time command');
    let mode: 'scalar' | 'vector' = 'scalar';
    let aveMode: 'one' | 'running' | 'window' = 'one';
    let windowM = 1;
    let startStep = 0;
    let fileName: string | null = null;
    let fileAppend = false;
    let overwrite = false;
    let fmt = ' %g';
    const off: number[] = [];
    const need = (n: number, what: string): void => { if (k + n > args.length) throw missingArgs(what); };
    while (k < args.length) {
      const key = args[k];
      if (key === 'file' || key === 'append') {
        need(2, `fix ave/time ${key}`);
        fileName = args[k + 1];
        fileAppend = key === 'append';
        k += 2;
      } else if (key === 'ave') {
        need(2, 'fix ave/time ave');
        const v = args[k + 1];
        if (v !== 'one' && v !== 'running' && v !== 'window') throw new StyleError(`Unknown fix ave/time ave keyword ${v}`);
        aveMode = v;
        if (aveMode === 'window') {
          need(3, 'fix ave/time ave window');
          windowM = inumeric(args[k + 2]);
          if (windowM <= 0) throw new StyleError(`Illegal fix ave/time ave window argument ${windowM}; must be > 0`);
        }
        k += aveMode === 'window' ? 3 : 2;
      } else if (key === 'start') {
        need(2, 'fix ave/time start');
        startStep = inumeric(args[k + 1]);
        k += 2;
      } else if (key === 'mode') {
        need(2, 'fix ave/time mode');
        const v = args[k + 1];
        if (v !== 'scalar' && v !== 'vector') throw new StyleError(`Unknown fix ave/time mode ${v}`);
        mode = v;
        k += 2;
      } else if (key === 'off') {
        need(2, 'fix ave/time off');
        off.push(inumeric(args[k + 1]));
        k += 2;
      } else if (key === 'overwrite') {
        overwrite = true;
        k += 1;
      } else if (key === 'format') {
        need(2, 'fix ave/time format');
        checkCFormat(args[k + 1], 'float', 'Invalid fix ave/time format argument');
        fmt = args[k + 1];
        k += 2;
      } else if (key === 'title1' || key === 'title2' || key === 'title3') {
        need(2, `fix ave/time ${key}`);
        if (key === 'title1') this.title1 = args[k + 1];
        else if (key === 'title2') this.title2 = args[k + 1];
        else this.title3 = args[k + 1];
        k += 2;
      } else throw new StyleError(`Unknown fix ave/time keyword ${key}`);
    }
    this.mode = mode;
    this.aveMode = aveMode;
    this.windowM = windowM;
    this.startStep = startStep;
    this.fileName = fileName;
    this.fileAppend = fileAppend;
    this.overwriteFile = overwrite;
    this.fmt = fmt;
    this.inputWords = expandArgs(sys, valueWords, mode === 'vector' ? 1 : 0).words;
    this.inputs = this.inputWords.map((w) => {
      const a = argInfo(w);
      if (a.which === 'none' || a.which === 'unknown' || a.dim > 1) throw new StyleError(`Invalid fix ave/time argument: ${w}`);
      // an index of 0 reads the scalar (native argindex 0); a negative one would read before the vector
      if (a.index1 < 0) throw new StyleError(`fix ave/time: index ${a.index1} in ${w} is refused: native LAMMPS reads outside the vector for it`);
      return { kind: a.which as 'c' | 'f' | 'v', id: a.name, index: a.index1 === 0 ? null : a.index1, text: w };
    });
    for (const m of off) {
      if (m < 1 || m > this.inputs.length) throw new StyleError(`Invalid fix ave/time off column: ${m}`);
      this.offIdx.add(m - 1);
    }
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/time nevery value: ${nevery}`);
    if (nrepeat <= 0) throw new StyleError(`Illegal fix ave/time nrepeat value: ${nrepeat}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/time nfreq value: ${nfreq}`);
    if (nfreq % nevery || nrepeat * nevery > nfreq) throw new StyleError('Inconsistent fix ave/time nevery/nrepeat/nfreq values');
    if (aveMode !== 'running' && overwrite) throw new StyleError('Fix ave/time overwrite keyword requires ave running setting');
    this.nEvery = nevery;
    this.nRepeat = nrepeat;
    this.nFreq = nfreq;
    this.nevery = nevery;
    for (const r of this.inputs) this.checkInput(r);
    // the output shape is known when the fix is defined (native LAMMPS sets scalar_flag, vector_flag and
    // array_flag in the constructor), so thermo_style and variables can use it before the first run; init() repeats
    // this at each run
    this.nvalid = aveNextValid(sys.state.step, this.nEvery, this.nRepeat, this.nFreq, this.startStep);
    this.init();
  }

  /** The source of one value, checked as native's constructor does for the mode. */
  private checkInput(r: Ref): void {
    const vector = this.mode === 'vector';
    if (r.kind === 'c') {
      const c = this.sys.computes.find((x) => x.id === r.id);
      if (!c) throw new StyleError(`Compute ID ${r.id} for fix ave/time does not exist`);
      if (!vector) {
        if (r.index === null && !c.scalarFlag) throw new StyleError(`Fix ave/time compute ${r.id} does not calculate a scalar`);
        if (r.index !== null && !c.vectorFlag) throw new StyleError(`Fix ave/time compute ${r.id} does not calculate a vector`);
        // an engine compute that sizes its vector when it is first computed reports 0 here
        if (r.index !== null && c.sizeVector > 0 && r.index > c.sizeVector) {
          throw new StyleError(`Fix ave/time compute ${r.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
      } else {
        if (r.index === null && !c.vectorFlag) throw new StyleError(`Fix ave/time compute ${r.id} does not calculate a vector`);
        if (r.index !== null && !c.arrayFlag) throw new StyleError(`Fix ave/time compute ${r.id} does not calculate an array`);
        if (r.index !== null && c.sizeArrayCols > 0 && r.index > c.sizeArrayCols) {
          throw new StyleError(`Fix ave/time compute ${r.id} array is accessed out-of-range${errorUrl(20)}`);
        }
      }
    } else if (r.kind === 'f') {
      const f = this.sys.fixes.find((x) => x.id === r.id);
      if (!f) throw new StyleError(`Fix ID ${r.id} for fix ave/time does not exist`);
      if (!vector) {
        if (r.index === null && !f.scalarFlag) throw new StyleError(`Fix ave/time fix ${r.id} does not calculate a scalar`);
        if (r.index !== null && !f.vectorFlag) throw new StyleError(`Fix ave/time fix ${r.id} does not calculate a vector`);
        if (r.index !== null && f.sizeVectorVariable) throw new StyleError(`Fix ave/time fix ${r.id} vector cannot be variable length`);
        if (r.index !== null && (f.sizeVector > 0 || f.sizeExact) && r.index > f.sizeVector) {
          throw new StyleError(`Fix ave/time fix ${r.id} vector is accessed out-of-range${errorUrl(20)}`);
        }
      } else {
        if (r.index === null && !f.vectorFlag) throw new StyleError(`Fix ave/time fix ${r.id} does not calculate a vector`);
        if (r.index !== null && !f.arrayFlag) throw new StyleError(`Fix ave/time fix ${r.id} does not calculate an array`);
        if (r.index !== null && f.sizeArrayRowsVariable) throw new StyleError(`Fix ave/time fix ${r.id} array cannot have variable row length`);
        if (r.index !== null && f.sizeArrayCols > 0 && r.index > f.sizeArrayCols) {
          throw new StyleError(`Fix ave/time fix ${r.id} array is accessed out-of-range${errorUrl(20)}`);
        }
      }
      if (f.globalFreq > 0 && this.nEvery % f.globalFreq) {
        throw new StyleError(`Fix ${r.id} for fix ave/time not computed at compatible time${errorUrl(7)}`);
      }
    } else {
      const v = this.sys.vars.get(r.id);
      if (!v) throw new StyleError(`Variable name ${r.id} for fix ave/time does not exist`);
      const equal = v.style === 'equal' || v.style === 'timer' || v.style === 'internal';
      if (!vector) {
        if (r.index === null && !equal) throw new StyleError(`Fix ave/time variable ${r.id} is not equal-style variable`);
        if (r.index !== null && v.style !== 'vector') throw new StyleError(`Fix ave/time variable ${r.id} is not vector-style variable`);
      } else {
        if (r.index === null && v.style !== 'vector') throw new StyleError(`Fix ave/time variable ${r.id} is not vector-style variable`);
        if (r.index !== null) throw new StyleError(`Fix ave/time mode vector variable ${r.id} cannot be indexed`);
      }
    }
  }

  /** The inputs must still exist (native init), and the output shape is set from them. */
  init(): void {
    for (const r of this.inputs) {
      if (r.kind === 'c' && !this.sys.computes.some((x) => x.id === r.id)) throw new StyleError(`Compute ID ${r.id} for fix ave/time does not exist`);
      if (r.kind === 'f' && !this.sys.fixes.some((x) => x.id === r.id)) throw new StyleError(`Fix ID ${r.id} for fix ave/time does not exist`);
      if (r.kind === 'v' && !this.sys.vars.has(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/time does not exist`);
    }
    this.scalarFlag = false;
    this.vectorFlag = false;
    this.arrayFlag = false;
    this.extlist = null;
    this.extarray = 0;
    if (this.mode === 'scalar') {
      const exts = this.inputs.map((r) => this.extOf(r));
      const allExt = exts.length > 0 && exts.every((e) => e === 1);
      this.len = 1;
      this.lenSet = true;
      if (this.inputs.length === 1) {
        this.scalarFlag = true;
        this.extscalar = exts[0];
      } else {
        this.vectorFlag = true;
        this.sizeVector = this.inputs.length;
        this.extvector = allExt ? 1 : 0;
        // each element carries its own flag (thermo normalises the extensive ones)
        this.extlist = exts;
      }
    } else if (this.inputs.length === 1) {
      // As native LAMMPS does (src/fix_ave_time.cpp at b332f28, init): a whole vector keeps its flag (or its
      // per-element list), a column of an array takes the array's flag
      this.vectorFlag = true;
      const r = this.inputs[0];
      const src = r.kind === 'c' ? this.sys.compute(r.id) : r.kind === 'f' ? this.sys.fix(r.id) : null;
      if (!src) this.extvector = 0;
      else if (r.index !== null) this.extvector = src.extarray;
      else {
        this.extvector = src.extvector;
        this.extlist = src.extlist ? src.extlist.slice() : null;
      }
    } else {
      // an array: every input must have the same flag, and a vector with a per-element list has none
      this.arrayFlag = true;
      this.sizeArrayCols = this.inputs.length;
      this.extvector = 0;
      let ext = -2;
      for (const r of this.inputs) {
        const src = r.kind === 'c' ? this.sys.compute(r.id) : r.kind === 'f' ? this.sys.fix(r.id) : null;
        const e = !src ? 0 : r.index !== null ? src.extarray : src.extlist ? -1 : src.extvector;
        if (e === -1 || (ext > -2 && e !== ext)) {
          throw new StyleError(`fix ${this.id} (ave/time): Fix ave/time cannot set output array intensive/extensive from these inputs`);
        }
        ext = e;
      }
      this.extarray = ext;
    }
    if (this.mode === 'vector' && !this.lenSet) {
      // As native LAMMPS does (src/fix_ave_time.cpp at b332f28, column_length): the rows of the output are the
      // static length of the inputs (size_vector, or size_array_rows for a column), known when the fix is defined, so
      // compute slice and thermo can check against them before the first sample sets the length
      let L = 0;
      for (const r of this.inputs) {
        const src = r.kind === 'c' ? this.sys.compute(r.id) : r.kind === 'f' ? this.sys.fix(r.id) : null;
        const n = !src ? 0 : r.index === null ? src.sizeVector : src.sizeArrayRows;
        if (n > 0 && L === 0) L = n;
      }
      if (this.inputs.length === 1) this.sizeVector = L;
      else this.sizeArrayRows = L;
    }
    this.ensureAlloc();
    // As native LAMMPS does (src/fix_ave_time.cpp init at b332f28): a minimization that moved past the next sample
    // step starts a new window
    const step = this.sys.state.step;
    if (this.nvalid < step) {
      this.irepeat = 0;
      this.nvalid = aveNextValid(step, this.nEvery, this.nRepeat, this.nFreq, this.startStep);
    }
  }

  /** Intensive (0) or extensive (1) of one scalar input value (mode scalar). */
  private extOf(r: Ref): number {
    if (r.kind === 'v') return 0; // "Values produced by a variable are treated as intensive"
    const src = r.kind === 'c' ? this.sys.compute(r.id) : this.sys.fix(r.id);
    if (r.index === null) return src.extscalar;
    return src.extlist ? src.extlist[r.index - 1] ?? 0 : src.extvector;
  }

  private ensureAlloc(): void {
    if (
      this.values.length === this.inputs.length && this.acc.length === this.inputs.length &&
      this.latest.length === this.inputs.length && this.runSum.length === this.inputs.length &&
      this.values[0] !== undefined && this.values[0].length === this.len
    ) return;
    const mk = (): Float64Array[] => {
      const a: Float64Array[] = [];
      for (let k = 0; k < this.inputs.length; k++) a.push(new Float64Array(this.len));
      return a;
    };
    this.acc = mk();
    this.latest = mk();
    this.values = mk();
    this.runSum = mk();
    this.ring = [];
    this.runN = 0;
    this.nSample = 0;
  }

  setup(): void { this.endOfStep(); }

  /**
   * As native LAMMPS does (src/fix_ave_time.cpp end_of_step at b332f28): only the scheduled step samples; the first
   * sample of a window zeroes the sums, the Nrepeat-th averages them and schedules the next window to end Nfreq steps
   * later.
   */
  endOfStep(): void {
    const step = this.sys.state.step;
    if (step !== this.nvalid) return;
    if (this.irepeat === 0) {
      for (const a of this.acc) a.fill(0);
      this.nSample = 0;
    }
    this.sampleInputs();
    this.irepeat++;
    if (this.irepeat < this.nRepeat) {
      this.nvalid += this.nEvery;
      return;
    }
    this.irepeat = 0;
    this.nvalid = step + this.nFreq - (this.nRepeat - 1) * this.nEvery;
    this.produceOutput(step);
  }

  /** Evaluates every input on this sample step. */
  private sampleInputs(): void {
    const sys = this.sys;
    if (this.mode === 'scalar') {
      for (let k = 0; k < this.inputs.length; k++) {
        const v = globalScalar(sys, this.inputs[k]);
        if (this.offIdx.has(k)) this.latest[k][0] = v;
        else this.acc[k][0] += v;
      }
      this.nSample++;
      return;
    }
    if (!this.lenSet) {
      const vecs: Float64Array[] = [];
      for (const r of this.inputs) vecs.push(globalVector(sys, r));
      const L = vecs[0].length;
      for (let k = 1; k < vecs.length; k++) {
        if (vecs[k].length !== L) {
          throw new StyleError(`fix ${this.id} (ave/time): input ${this.inputWords[k]} has ${vecs[k].length} values, expected ${L} ("All vectors must be the same length")`);
        }
      }
      this.len = L;
      this.lenSet = true;
      if (this.inputs.length === 1) this.sizeVector = L;
      else this.sizeArrayRows = L;
      this.ensureAlloc();
      for (let k = 0; k < vecs.length; k++) (this.offIdx.has(k) ? this.latest[k] : this.acc[k]).set(vecs[k]);
      this.nSample = 1;
      return;
    }
    for (let k = 0; k < this.inputs.length; k++) {
      const vec = globalVector(sys, this.inputs[k]);
      if (vec.length !== this.len) {
        throw new StyleError(`fix ${this.id} (ave/time): input ${this.inputWords[k]} has ${vec.length} values, expected ${this.len} ("All vectors must be the same length")`);
      }
      if (this.offIdx.has(k)) this.latest[k].set(vec);
      else {
        const a = this.acc[k];
        for (let i = 0; i < this.len; i++) a[i] += vec[i];
      }
    }
    this.nSample++;
  }

  /** Averages the collected samples and applies the ave one/running/window mode. */
  private produceOutput(step: number): void {
    const n = this.inputs.length;
    const out: Float64Array[] = [];
    for (let k = 0; k < n; k++) {
      if (this.offIdx.has(k)) { out.push(this.latest[k]); continue; }
      const a = this.acc[k];
      const m = new Float64Array(this.len);
      for (let i = 0; i < this.len; i++) m[i] = a[i] / this.nRepeat;
      out.push(m);
    }
    if (this.aveMode === 'running') {
      this.runN++;
      for (let k = 0; k < n; k++) {
        if (this.offIdx.has(k)) continue;
        const r = this.runSum[k], m = out[k];
        for (let i = 0; i < this.len; i++) { r[i] += m[i]; m[i] = r[i] / this.runN; }
      }
    } else if (this.aveMode === 'window') {
      this.ring.push(out.map((m) => m.slice()));
      if (this.ring.length > this.windowM) this.ring.shift();
      for (let k = 0; k < n; k++) {
        if (this.offIdx.has(k)) continue;
        const v = this.values[k];
        v.fill(0);
        for (const entry of this.ring) {
          const e = entry[k];
          for (let i = 0; i < this.len; i++) v[i] += e[i];
        }
        for (let i = 0; i < this.len; i++) v[i] /= this.ring.length;
      }
    }
    for (let k = 0; k < n; k++) {
      if (this.offIdx.has(k)) this.values[k].set(this.latest[k]);
      else if (this.aveMode !== 'window') this.values[k].set(out[k]);
    }
    for (const a of this.acc) a.fill(0);
    this.nSample = 0;
    if (this.fileName) this.writeSection(step);
  }

  private buildHeader(): string[] {
    const words = this.inputWords.join(' ');
    const out = [this.title1 ?? `# Time-averaged data for fix ${this.id}`];
    if (this.mode === 'scalar') {
      out.push(this.title2 ?? `# TimeStep ${words}`);
    } else {
      out.push(this.title2 ?? '# TimeStep Number-of-rows');
      out.push(this.title3 ?? `# Row ${words}`);
    }
    return out;
  }

  private writeSection(step: number): void {
    const sys = this.sys;
    if (this.header === null) this.header = this.buildHeader();
    const sec: string[] = [];
    if (this.mode === 'scalar') {
      let line = String(step);
      for (let k = 0; k < this.inputs.length; k++) line += compileCFormat(this.fmt, 'float', 'Invalid fix ave/time format argument')(this.values[k][0]);
      sec.push(line);
    } else {
      sec.push(`${step} ${this.len}`);
      for (let i = 0; i < this.len; i++) {
        let line = String(i + 1);
        for (let k = 0; k < this.inputs.length; k++) line += compileCFormat(this.fmt, 'float', 'Invalid fix ave/time format argument')(this.values[k][i]);
        sec.push(line);
      }
    }
    if (this.overwriteFile) {
      sys.writeFile(this.fileName!, [...this.header, ...sec].join('\n') + '\n', false);
    } else if (this.fileAppend) {
      const head = this.headerWritten ? '' : this.header.join('\n') + '\n';
      this.headerWritten = true;
      sys.writeFile(this.fileName!, head + sec.join('\n') + '\n', true);
    } else {
      this.body.push(...sec);
      sys.writeFile(this.fileName!, [...this.header, ...this.body].join('\n') + '\n', false);
    }
  }

  computeScalar(): number {
    if (!this.scalarFlag) throw new StyleError(`fix ${this.id} does not compute a global scalar`);
    return this.values[0]?.[0] ?? 0;
  }

  computeVector(i: number): number {
    if (!this.vectorFlag) throw new StyleError(`fix ${this.id} does not compute a global vector`);
    const v = this.mode === 'scalar' ? this.values[i] : this.values[0];
    return v?.[this.mode === 'scalar' ? 0 : i] ?? 0;
  }

  computeArray(i: number, j: number): number {
    if (!this.arrayFlag) throw new StyleError(`fix ${this.id} does not compute a global array`);
    return this.values[j]?.[i] ?? 0;
  }
}

export class FixPrint extends Fix {
  readonly style = 'print';
  private readonly everyN: number;
  private readonly everyVar: string | null;
  private readonly text: string;
  private readonly fileName: string | null;
  private readonly fileAppend: boolean;
  private readonly screen: boolean;
  private title: string;
  private titleWritten = false;
  private nextStep = 0;
  private lines: string[] = [];

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const what = `fix ${id} (print)`;
    if (args.length < 2) throw new StyleError(`usage: fix ${id} group-ID print N string [keyword value ...]`);
    const nWord = args[0];
    if (nWord.startsWith('v_')) {
      this.everyVar = nWord.slice(2);
      this.everyN = 0;
      this.nevery = 1;
    } else {
      const n = Number(nWord);
      if (!Number.isInteger(n) || n < 1) throw new StyleError(`${what}: N must be a positive integer or v_name, got '${nWord}'`);
      this.everyVar = null;
      this.everyN = n;
      this.nevery = n;
    }
    this.text = args[1];
    this.screen = true;
    this.title = `# Fix print output for fix ${id}`;
    let file: string | null = null;
    let append: string | null = null;
    let title: string | null = null;
    for (let k = 2; k < args.length; k += 2) {
      const key = args[k];
      const val = args[k + 1];
      if (val === undefined) throw new StyleError(`${what}: keyword ${key} needs a value`);
      if (key === 'file') file = val;
      else if (key === 'append') append = val;
      else if (key === 'screen') {
        if (val !== 'yes' && val !== 'no') throw new StyleError(`${what}: screen must be yes or no, got '${val}'`);
        this.screen = val === 'yes';
      } else if (key === 'title') title = val;
      else throw new StyleError(`${what}: unknown keyword '${key}'`);
    }
    if (file !== null && append !== null) throw new StyleError(`${what}: file and append cannot both be used`);
    this.fileName = file ?? append;
    this.fileAppend = append !== null;
    if (title !== null) this.title = title;
  }

  init(): void {
    if (this.everyVar === null) return;
    const def = this.sys.vars.get(this.everyVar);
    if (!def) throw new StyleError(`fix ${this.id} (print): variable ${this.everyVar} does not exist`);
    if (def.style !== 'equal') throw new StyleError(`fix ${this.id} (print): variable ${this.everyVar} must be an equal-style variable, not ${def.style}-style`);
    this.nextStep = Math.trunc(this.sys.equalVariable(this.everyVar));
  }

  setup(): void {
    const step = this.sys.state.step;
    if (this.everyVar !== null) {
      if (step >= this.nextStep) { this.printLine(); this.advance(); }
    } else if (step % this.everyN === 0) {
      this.printLine();
    }
  }

  endOfStep(): void {
    if (this.everyVar !== null) {
      if (this.sys.state.step >= this.nextStep) { this.printLine(); this.advance(); }
    } else {
      this.printLine(); // the run loop only calls this on multiples of N
    }
  }

  /** The variable decides the next timestep ("On that timestep, the variable will be evaluated again"). */
  private advance(): void {
    const step = this.sys.state.step;
    const v = Math.trunc(this.sys.equalVariable(this.everyVar!));
    this.nextStep = v > step ? v : step + 1;
  }

  private printLine(): void {
    const sys = this.sys;
    const env = sys.formulaEnv;
    const line = substituteVariables(
      this.text,
      (n) => sys.vars.text(n, env),
      (f, fmt) => formatNumber(evaluateScalar(f, env), fmt ?? '%.20g'),
    );
    if (this.screen) sys.log(line);
    if (this.fileName) {
      if (this.fileAppend) {
        const head = this.titleWritten ? '' : this.title + '\n';
        this.titleWritten = true;
        sys.writeFile(this.fileName, head + line + '\n', true);
      } else {
        this.lines.push(line);
        sys.writeFile(this.fileName, [this.title, ...this.lines].join('\n') + '\n', false);
      }
    }
  }
}
