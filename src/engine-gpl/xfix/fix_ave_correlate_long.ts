/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_ave_correlate_long.cpp and src/EXTRA-FIX/fix_ave_correlate_long.h at commit b332f28 (class
 * FixAveCorrelateLong, fix ave/correlate/long: the constructor and its keywords, init, setup, end_of_step, evaluate,
 * accumulate and the one- and two-value add routines, the file header). Contributing authors Jorge Ramirez (Universidad
 * Politecnica de Madrid) and Christian Krekeler (TU Darmstadt), with others named in the source. Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Multi-tau correlator (docs.lammps.org/fix_ave_correlate_long.html): values are sampled every nevery steps, the
 * samples are block-averaged by m (ncount) on each of numcorrelators levels, and the correlations of nlen lags per level
 * are written every nfreq steps to the file (the "file" keyword). The values may be c_ID, c_ID[I], f_ID, f_ID[I],
 * v_name, v_name[I]. Global outputs (scalar or vector) do not exist, as in native LAMMPS (f_ID is refused by thermo).
 * The keywords type (auto, upper, lower, auto/upper, auto/lower, full, first), start, ncorr, nlen, ncount, file,
 * overwrite, title1, title2 are implemented.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import { missingArgs } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { globalScalar, parseRef, expandWildcards, type Ref } from '../../engine/refs';
import { formatNumber } from '../../engine/cformat';

/** fix_ave_correlate_long.h: the correlation pair types (AUTO = auto, UPPER = upper, ...). */
type CorrType = 'auto' | 'upper' | 'lower' | 'auto/upper' | 'auto/lower' | 'full' | 'first';

/** The shift value of an empty slot (fix_ave_correlate_long.cpp: shift initialised to -2e10). */
const EMPTY = -2e10;
const EMPTY_TEST = -1e10;

function inumeric(w: string | undefined): number {
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return Number(w);
}

/** powint(m, k) of the source: an integer power of a double. */
function powint(m: number, k: number): number {
  let r = 1.0;
  for (let i = 0; i < k; i++) r *= m;
  return r;
}

/** C "%lg" of one value (printf %g). */
const fmtG = (v: number): string => formatNumber(v, '%g');

export class FixAveCorrelateLong extends Fix {
  readonly style = 'ave/correlate/long';
  private readonly nfreq: number;
  private readonly refs: Ref[];
  private readonly names: string[];
  private readonly nvalues: number;
  private readonly type: CorrType;
  private readonly startstep: number;
  private readonly numcorrelators: number;
  private readonly p: number;
  private readonly m: number;
  private readonly dmin: number;
  private readonly length: number;
  private readonly npair: number;
  private readonly overwrite: boolean;
  private readonly fileName: string | null;
  private readonly header: string[] = [];
  private readonly cvalues: Float64Array;
  private readonly shift: Float64Array;
  private readonly shift2: Float64Array;
  private readonly correlation: Float64Array;
  private readonly accumulator: Float64Array;
  private readonly accumulator2: Float64Array;
  private readonly ncorrelation: Float64Array;
  private readonly naccumulator: Int32Array;
  private readonly insertindex: Int32Array;
  private readonly t: Float64Array;
  private readonly f: Float64Array;
  private kmax = 0;
  private npcorr = 0;
  private nvalid: number;
  private lastAccumulatedStep = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix ave/correlate/long command: missing arguments');
    const nevery = inumeric(args[0]);
    const nfreq = inumeric(args[1]);
    this.nevery = nevery;
    this.nfreq = nfreq;

    // the value words: everything up to the first keyword (a word that is not c_, f_ or v_)
    const rest = args.slice(2);
    let iarg = 0;
    const valueWords: string[] = [];
    while (iarg < rest.length && /^[cfv]_/.test(rest[iarg])) {
      valueWords.push(rest[iarg]);
      iarg++;
    }
    const expanded = expandWildcards(sys, valueWords, 'scalar');
    this.refs = expanded.map((w) => {
      try {
        return parseRef(w);
      } catch {
        throw new StyleError(`Unknown fix ave/correlate/long data type: ${w}`);
      }
    });
    this.names = expanded;
    this.nvalues = expanded.length;

    let type: CorrType = 'auto';
    let startstep = 0;
    let numcorrelators = 20;
    let p = 16;
    let m = 2;
    let fileName: string | null = null;
    let overwrite = false;
    let title1: string | null = null;
    let title2: string | null = null;
    while (iarg < rest.length) {
      const key = rest[iarg];
      const val = rest[iarg + 1];
      if (key === 'type') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long type');
        if (!['auto', 'upper', 'lower', 'auto/upper', 'auto/lower', 'full', 'first'].includes(val)) {
          throw new StyleError(`Unknown fix ave/correlate/long type: ${val}`);
        }
        type = val as CorrType;
        iarg += 2;
      } else if (key === 'start') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long start');
        startstep = inumeric(val);
        iarg += 2;
      } else if (key === 'ncorr') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long ncorr');
        numcorrelators = inumeric(val);
        iarg += 2;
      } else if (key === 'nlen') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long nlen');
        p = inumeric(val);
        iarg += 2;
      } else if (key === 'ncount') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long ncount');
        m = inumeric(val);
        iarg += 2;
      } else if (key === 'file') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long file');
        fileName = val;
        iarg += 2;
      } else if (key === 'overwrite') {
        overwrite = true;
        iarg += 1;
      } else if (key === 'title1') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long title1');
        title1 = val;
        iarg += 2;
      } else if (key === 'title2') {
        if (iarg + 2 > rest.length) throw missingArgs('fix ave/correlate/long title2');
        title2 = val;
        iarg += 2;
      } else {
        throw new StyleError(`Unknown fix ave/correlate/long keyword: ${key}`);
      }
    }
    this.type = type;
    this.startstep = startstep;
    this.numcorrelators = numcorrelators;
    this.p = p;
    this.m = m;
    this.overwrite = overwrite;
    this.fileName = fileName;
    if (p % m !== 0) throw new StyleError('Fix ave/correlate/long: nlen must be divisible by ncount');
    this.dmin = Math.trunc(p / m);
    this.length = numcorrelators * p;
    if (nevery <= 0) throw new StyleError(`Illegal fix ave/correlate/long nevery value: ${nevery}`);
    if (nfreq <= 0) throw new StyleError(`Illegal fix ave/correlate/long nfreq value: ${nfreq}`);
    if (nfreq % nevery) throw new StyleError('Inconsistent fix ave/correlate/long nevery/nfreq values');

    for (const r of this.refs) {
      if (r.kind === 'c') {
        const c = sys.compute(r.id);
        if (r.index === null && !c.scalarFlag) throw new StyleError(`Fix ave/correlate/long compute ${r.id} does not calculate a scalar`);
        if (r.index !== null && !c.vectorFlag) throw new StyleError(`Fix ave/correlate/long compute ${r.id} does not calculate a vector`);
        if (r.index !== null && r.index > c.sizeVector) {
          throw new StyleError(`Fix ave/correlate/long compute ${r.id} vector is accessed out-of-range`);
        }
      } else if (r.kind === 'f') {
        const f = sys.fix(r.id);
        if (r.index === null && !f.scalarFlag) throw new StyleError(`Fix ave/correlate/long fix ${r.id} does not calculate a scalar`);
        if (r.index !== null && !f.vectorFlag) throw new StyleError(`Fix ave/correlate/long fix ${r.id} does not calculate a vector`);
        if (r.index !== null && r.index > f.sizeVector) {
          throw new StyleError(`Fix ave/correlate/long fix ${r.id} vector is accessed out-of-range`);
        }
        if (f.globalFreq > 0 && nevery % f.globalFreq) {
          throw new StyleError(`Fix ${r.id} for fix ave/correlate/long not computed at compatible time`);
        }
      } else {
        if (r.index === null && !sys.vars.get(r.id)) throw new StyleError(`Variable name ${r.id} for fix ave/correlate/long does not exist`);
      }
    }

    if (type === 'auto' || type === 'first') this.npair = this.nvalues;
    else if (type === 'upper' || type === 'lower') this.npair = (this.nvalues * (this.nvalues - 1)) / 2;
    else if (type === 'auto/upper' || type === 'auto/lower') this.npair = (this.nvalues * (this.nvalues + 1)) / 2;
    else this.npair = this.nvalues * this.nvalues;

    // header (fix_ave_correlate_long.cpp constructor: title1, the names of the pairs as the native loops print them)
    if (fileName !== null) {
      this.header.push(title1 ?? `# Time-correlated data for fix ${id}`);
      if (title2 !== null) {
        this.header.push(title2);
      } else {
        let line = '# Time';
        const e = this.names;
        const n = this.nvalues;
        if (type === 'auto') for (let i = 0; i < n; i++) line += ` ${e[i]}*${e[i]}`;
        else if (type === 'upper') for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) line += ` ${e[i]}*${e[j]}`;
        else if (type === 'lower') for (let i = 0; i < n; i++) for (let j = 0; j < i - 1; j++) line += ` ${e[i]}*${e[j]}`;
        else if (type === 'auto/upper') for (let i = 0; i < n; i++) for (let j = i; j < n; j++) line += ` ${e[i]}*${e[j]}`;
        else if (type === 'auto/lower') for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) line += ` ${e[i]}*${e[j]}`;
        else if (type === 'full') for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) line += ` ${e[i]}*${e[j]}`;
        else for (let i = 0; i < n; i++) line += ` ${e[0]}*${e[i]}`;
        this.header.push(line);
      }
    }

    const np = this.npair;
    this.cvalues = new Float64Array(this.nvalues);
    this.shift = new Float64Array(np * numcorrelators * p).fill(EMPTY);
    this.shift2 = new Float64Array(np * numcorrelators * p);
    this.correlation = new Float64Array(np * numcorrelators * p);
    this.accumulator = new Float64Array(np * numcorrelators);
    this.accumulator2 = new Float64Array(np * numcorrelators);
    this.ncorrelation = new Float64Array(numcorrelators * p);
    this.naccumulator = new Int32Array(numcorrelators);
    this.insertindex = new Int32Array(numcorrelators);
    this.t = new Float64Array(this.length);
    this.f = new Float64Array(np * this.length);
    this.nvalid = this.nextvalid(sys.state.step);
    if (fileName !== null) this.writeFile(this.header.join('\n') + '\n', false);
  }

  /** The output file (the native fp): the header at construction, then one block per output. */
  private writeFile(text: string, append: boolean): void {
    this.sys.writeFile?.(this.fileName!, text, append);
  }

  /** nextvalid: the next step on which end_of_step does something (a multiple of nevery, not before startstep). */
  private nextvalid(step: number): number {
    let nvalid = step;
    if (this.startstep > nvalid) nvalid = this.startstep;
    if (nvalid % this.nevery) nvalid = Math.floor(nvalid / this.nevery) * this.nevery + this.nevery;
    return nvalid;
  }

  init(): void {
    const sys = this.sys;
    for (const r of this.refs) {
      if (r.kind === 'c') sys.compute(r.id);
      else if (r.kind === 'f') sys.fix(r.id);
    }
    if (this.nvalid < sys.state.step) this.nvalid = this.nextvalid(sys.state.step);
  }

  /** setup: end_of_step (the sample of the start step, if it is on the schedule). */
  setup(): void { this.endOfStep(); }

  endOfStep(): void {
    const sys = this.sys;
    const ntimestep = sys.state.step;
    if (ntimestep !== this.nvalid) return;
    for (let i = 0; i < this.nvalues; i++) {
      const r = this.refs[i];
      let scalar = 0.0;
      if (r.kind === 'v') {
        if (r.index === null) {
          scalar = sys.equalVariable(r.id);
        } else {
          const vec = sys.vars.evalVector(r.id, sys.formulaEnv);
          scalar = r.index > vec.length ? 0.0 : vec[r.index - 1];
        }
      } else {
        scalar = globalScalar(sys, r);
      }
      this.cvalues[i] = scalar;
    }
    this.nvalid += this.nevery;
    this.accumulate(ntimestep);
    if (ntimestep % this.nfreq) return;
    this.evaluate();
    // block: "# Timestep: N", then one row per lag time (t * dt * nevery) with the pair correlations
    let block = `# Timestep: ${ntimestep}\n`;
    const dt = sys.state.dt;
    for (let i = 0; i < this.npcorr; i++) {
      let line = `${fmtG(this.t[i] * dt * this.nevery)} `;
      for (let j = 0; j < this.npair; j++) line += `${fmtG(this.f[j * this.length + i])} `;
      block += line + '\n';
    }
    // overwrite: the file keeps only the header and the latest block (native fseek to filepos and truncate)
    if (this.overwrite) this.writeFile(this.header.join('\n') + '\n' + block, false);
    else this.writeFile(block, true);
  }

  /** evaluate: the lag times and the averaged correlations of every level with data. */
  private evaluate(): void {
    const p = this.p, m = this.m, np = this.npair, nc = this.numcorrelators, len = this.length;
    let jm = 0;
    for (let j = 0; j < p; j++) {
      if (this.ncorrelation[j] > 0) {
        this.t[jm] = j;
        for (let i = 0; i < np; i++) {
          this.f[i * len + jm] = this.correlation[(i * nc + 0) * p + j] / this.ncorrelation[j];
        }
        jm++;
      }
    }
    for (let k = 1; k < this.kmax; k++) {
      for (let j = this.dmin; j < p; j++) {
        const nk = this.ncorrelation[k * p + j];
        if (nk > 0) {
          this.t[jm] = j * powint(m, k);
          for (let i = 0; i < np; i++) {
            this.f[i * len + jm] = this.correlation[(i * nc + k) * p + j] / nk;
          }
          jm++;
        }
      }
    }
    this.npcorr = jm;
  }

  /** accumulate: the correlations of the current sample values, per the pair type. */
  private accumulate(step: number): void {
    if (step <= this.lastAccumulatedStep) return;
    const c = this.cvalues;
    const n = this.nvalues;
    switch (this.type) {
      case 'auto':
        for (let i = 0; i < n; i++) this.add1(i, c[i], 0);
        break;
      case 'upper': {
        let ipair = 0;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) this.add2(ipair++, c[j], c[i], 0);
        break;
      }
      case 'lower': {
        let ipair = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) this.add2(ipair++, c[j], c[i], 0);
        break;
      }
      case 'auto/upper': {
        let ipair = 0;
        for (let i = 0; i < n; i++) {
          for (let j = i; j < n; j++) {
            if (i === j) this.add1(ipair++, c[i], 0);
            else this.add2(ipair++, c[j], c[i], 0);
          }
        }
        break;
      }
      case 'auto/lower': {
        let ipair = 0;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j <= i; j++) {
            if (i === j) this.add1(ipair++, c[i], 0);
            else this.add2(ipair++, c[j], c[i], 0);
          }
        }
        break;
      }
      case 'full': {
        let ipair = 0;
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            if (i === j) this.add1(ipair++, c[i], 0);
            else this.add2(ipair++, c[j], c[i], 0);
          }
        }
        break;
      }
      case 'first':
        for (let i = 0; i < n; i++) this.add2(i, c[i], c[0], 0);
        break;
    }
    this.lastAccumulatedStep = step;
  }

  /** add (one value): the autocorrelator k of pair i with the value w. */
  private add1(i: number, w: number, k: number): void {
    const nc = this.numcorrelators, p = this.p, m = this.m, np = this.npair;
    if (k === nc) return;
    if (k > this.kmax) this.kmax = k;
    const sh = (kk: number, jj: number) => (i * nc + kk) * p + jj;
    this.shift[sh(k, this.insertindex[k])] = w;
    this.accumulator[i * nc + k] += w;
    if (i === 0) this.naccumulator[k]++;
    if (this.naccumulator[k] === m) {
      this.add1(i, this.accumulator[i * nc + k] / m, k + 1);
      this.accumulator[i * nc + k] = 0;
      if (i === np - 1) this.naccumulator[k] = 0;
    }
    const ind1 = this.insertindex[k];
    if (k === 0) {
      // the first correlator is different: lags 0..p-1 over the whole window
      let ind2 = ind1;
      for (let j = 0; j < p; j++) {
        if (this.shift[sh(k, ind2)] > EMPTY_TEST) {
          this.correlation[sh(k, j)] += this.shift[sh(k, ind1)] * this.shift[sh(k, ind2)];
          if (i === 0) this.ncorrelation[k * p + j]++;
        }
        ind2--;
        if (ind2 < 0) ind2 += p;
      }
    } else {
      let ind2 = ind1 - this.dmin;
      for (let j = this.dmin; j < p; j++) {
        if (ind2 < 0) ind2 += p;
        if (this.shift[sh(k, ind2)] > EMPTY_TEST) {
          this.correlation[sh(k, j)] += this.shift[sh(k, ind1)] * this.shift[sh(k, ind2)];
          if (i === 0) this.ncorrelation[k * p + j]++;
        }
        ind2--;
      }
    }
    if (i === np - 1) {
      this.insertindex[k]++;
      if (this.insertindex[k] === p) this.insertindex[k] = 0;
    }
  }

  /** add (two values): the cross-correlator k of pair i, wA now and wB at the lags. */
  private add2(i: number, wA: number, wB: number, k: number): void {
    const nc = this.numcorrelators, p = this.p, m = this.m, np = this.npair;
    if (k === nc) return;
    if (k > this.kmax) this.kmax = k;
    const sh = (kk: number, jj: number) => (i * nc + kk) * p + jj;
    this.shift[sh(k, this.insertindex[k])] = wA;
    this.shift2[sh(k, this.insertindex[k])] = wB;
    this.accumulator[i * nc + k] += wA;
    this.accumulator2[i * nc + k] += wB;
    if (i === 0) this.naccumulator[k]++;
    if (this.naccumulator[k] === m) {
      this.add2(i, this.accumulator[i * nc + k] / m, this.accumulator2[i * nc + k] / m, k + 1);
      this.accumulator[i * nc + k] = 0;
      this.accumulator2[i * nc + k] = 0;
      if (i === np - 1) this.naccumulator[k] = 0;
    }
    const ind1 = this.insertindex[k];
    if (k === 0) {
      let ind2 = ind1;
      for (let j = 0; j < p; j++) {
        if (this.shift[sh(k, ind2)] > EMPTY_TEST) {
          this.correlation[sh(k, j)] += this.shift[sh(k, ind1)] * this.shift2[sh(k, ind2)];
          if (i === 0) this.ncorrelation[k * p + j]++;
        }
        ind2--;
        if (ind2 < 0) ind2 += p;
      }
    } else {
      let ind2 = ind1 - this.dmin;
      for (let j = this.dmin; j < p; j++) {
        if (ind2 < 0) ind2 += p;
        if (this.shift[sh(k, ind2)] > EMPTY_TEST) {
          this.correlation[sh(k, j)] += this.shift[sh(k, ind1)] * this.shift2[sh(k, ind2)];
          if (i === 0) this.ncorrelation[k * p + j]++;
        }
        ind2--;
      }
    }
    if (i === np - 1) {
      this.insertindex[k]++;
      if (this.insertindex[k] === p) this.insertindex[k] = 0;
    }
  }
}
