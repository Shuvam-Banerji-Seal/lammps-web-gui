/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/APIP/fix_lambda_apip.cpp (fix
 * lambda/apip: the arguments, fix_modify, init, post_constructor, setup_pre_force, post_integrate, end_of_step, the
 * lambda_input and lambda histories, get_lambda_average, calculate_lambda_input_ta, switching_function_poly,
 * write_peratom_stats and extract), at commit b332f28. Author named in the source: David Immel (FZ Juelich), with the
 * LAMMPS Sandia copyright notice (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/fix_lambda_apip.html. The switching parameter lambda of each atom follows from the time average
 * of lambda_input (pair lambda/input/apip or lambda/input/csp/apip) through a switching polynomial between the two
 * thresholds, spread over a transition zone by pair lambda/zone/apip and averaged again over time. The two histories
 * are per-atom ring buffers (native FixStoreAtom arrays: history_length + 2 and history2_length + 1 columns), kept here
 * as internal per-atom props so that they follow deleted and added atoms. Ghost atoms read their owners' values, which
 * is what the native forward communication of lambda_input_ta, lambda and lambda_const gives at the time it is used.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { APIP_LAMBDA, APIP_LAMBDA_CONST, APIP_LAMBDA_INPUT, APIP_LAMBDA_INPUT_TA } from '../../engine/atoms';
import { pow as libmPow } from '../libm/pow';
import type { LambdaInputHistory } from './pair_lambda_input_apip';
import type { LambdaZoneSettings } from './pair_lambda_zone_apip';
import type { PairLambdaInputAPIP } from './pair_lambda_input_apip';
import type { PairLambdaZoneAPIP } from './pair_lambda_zone_apip';

/** The pair styles of the force field (the sub-styles of a hybrid). */
export const pairStyles = (sys: System): unknown[] => {
  const pair = sys.ff.pair as unknown as { subs?: { style: unknown }[] } | null;
  if (!pair) return [];
  return pair.subs ? pair.subs.map((s) => s.style) : [pair];
};

const nameOf = (p: unknown): string => (p as { name?: string }).name ?? '';

export class FixLambdaAPIP extends Fix implements LambdaInputHistory, LambdaZoneSettings {
  readonly style = 'lambda/apip';
  private readonly thresholdLo: number;
  private readonly thresholdHi: number;
  private readonly thresholdWidth: number;
  cutLo = 4.0;
  cutHi = 12.0;
  cutHiSq = 144;
  cutWidth = 8;
  lambdaNonGroup = 1;
  readonly historyLength: number = 100;
  private readonly history2Length: number = 100;
  private historyLast = -1;
  private history2Last = -1;
  private historyUsed = 0;
  private history2Used = 0;
  private readonly minDeltaLambda: number = 0;
  private readonly groupNameSimple: string | null = null;
  private readonly groupNameComplex: string | null = null;
  private readonly groupNameIgnore: string | null = null;
  private groupBitSimple = 0;
  private groupBitComplex = 0;
  groupBitIgnoreLambdaInput = 0;
  private readonly dumpHistory: boolean = false;
  private invokedHistoryUpdate = -1;
  private invokedHistory2Update = -1;
  /** Per-atom prop keys of the two histories. */
  readonly keyInput: string;
  private readonly keyLambda: string;
  private pairInput: PairLambdaInputAPIP | null = null;
  private pairZone: PairLambdaZoneAPIP | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const num = (w: string | undefined): number => {
      if (w === undefined || w === '') throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
      const v = Number(w);
      if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
      return v;
    };
    const int = (w: string | undefined): number => {
      if (w === undefined || w === '') throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
      if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
      return parseInt(w, 10);
    };
    if (args.length < 1) throw new StyleError('fix lambda/apip requires two arguments');
    this.thresholdLo = num(args[0]);
    this.thresholdHi = num(args[1]);
    this.thresholdWidth = this.thresholdHi - this.thresholdLo;
    let historyLength = 100, history2Length = 100, minDelta = 0, peratom = false, dump = false;
    let gs: string | null = null, gc: string | null = null, gi: string | null = null;
    const bitOf = (name: string): number => {
      if (!sys.groups.names.includes(name)) throw new StyleError(`fix lambda/apip: group ${name} does not exist`);
      return sys.groups.bit(name);
    };
    // native: for (iarg = 5; iarg < narg; iarg++), arguments here start after the two thresholds
    for (let k = 2; k < args.length; k++) {
      const w = args[k];
      if (w === 'time_averaged_zone') {
        if (k + 4 >= args.length) throw new StyleError('fix lambda/apip: time_averaged_zone requires four arguments');
        this.cutLo = num(args[k + 1]);
        this.cutHi = num(args[k + 2]);
        historyLength = int(args[k + 3]);
        history2Length = int(args[k + 4]);
        k += 4;
      } else if (w === 'min_delta_lambda') {
        if (k + 1 >= args.length) throw new StyleError('fix lambda/apip: min_delta_lambda requires one argument');
        minDelta = num(args[k + 1]);
        k += 1;
      } else if (w === 'lambda_non_group') {
        if (k + 1 >= args.length) throw new StyleError('fix lambda/apip: lambda_non_group requires an argument');
        const v = args[k + 1];
        if (v === 'precise') this.lambdaNonGroup = 0;
        else if (v === 'fast') this.lambdaNonGroup = 1;
        else {
          this.lambdaNonGroup = num(v);
          if (this.lambdaNonGroup < 0 || this.lambdaNonGroup > 1) throw new StyleError('fix lambda/apip: Illegal value of lambda_non_group');
        }
        k++;
      } else if (w === 'store_atomic_stats') {
        peratom = true;
      } else if (w === 'dump_atomic_history') {
        peratom = true;
        dump = true;
      } else if (w === 'group_fast') {
        gs = args[k + 1];
        this.groupBitSimple = bitOf(gs);
        k++;
      } else if (w === 'group_precise') {
        gc = args[k + 1];
        this.groupBitComplex = bitOf(gc);
        k++;
      } else if (w === 'group_ignore_lambda_input') {
        gi = args[k + 1];
        this.groupBitIgnoreLambdaInput = bitOf(gi);
        k++;
      } else {
        throw new StyleError(`fix lambda/apip: unknown argument ${w}`);
      }
    }
    this.historyLength = historyLength;
    this.history2Length = history2Length;
    this.minDeltaLambda = minDelta;
    this.groupNameSimple = gs;
    this.groupNameComplex = gc;
    this.groupNameIgnore = gi;
    this.dumpHistory = dump;
    this.cutHiSq = this.cutHi * this.cutHi;
    this.cutWidth = this.cutHi - this.cutLo;
    if (this.thresholdLo > this.thresholdHi || this.thresholdLo < 0) throw new StyleError('fix lambda: Illegal or missing threshold values');
    if (minDelta < 0) throw new StyleError(`fix lambda: min_delta_lambda >= 0 required instead of ${minDelta}`);
    if (this.cutLo < 0 || this.cutHi < this.cutLo) throw new StyleError('fix lambda: Illegal cutoff values');
    if (historyLength < 2 || history2Length < 2) throw new StyleError('fix lambda: history_length > 1 required');
    if ((this.groupBitSimple !== 0 || this.groupBitComplex !== 0) && this.groupBitIgnoreLambdaInput === 0) {
      sys.warn('group_ignore_lambda_input should be used to prevent the calculation of lambda_input for atoms that are in the groups group_fast and group_precise.');
    }
    const c = sys.state.custom;
    if (!c.has(APIP_LAMBDA_CONST)) throw new StyleError('fix lambda requires atomic style with lambda_const.');
    if (!c.has(APIP_LAMBDA)) throw new StyleError('fix lambda requires atomic style with lambda.');
    if (!c.has(APIP_LAMBDA_INPUT)) throw new StyleError('fix lambda requires atomic style with lambda_input.');
    if (!c.has(APIP_LAMBDA_INPUT_TA)) throw new StyleError('fix lambda requires atomic style with lambda_input_ta.');
    if (peratom) {
      this.peratomFlag = true;
      this.sizePeratomCols = 5 + (dump ? historyLength + history2Length + 2 : 0);
      this.arrayAtom = new Float64Array(sys.state.n * this.sizePeratomCols);
    }
    // post_constructor: the two FixStoreAtom arrays, zero
    this.keyInput = `#apip_lambda_input_history:${id}`;
    this.keyLambda = `#apip_lambda_history:${id}`;
    const n = sys.state.n;
    c.set(this.keyInput, { int: false, cols: historyLength + 2, data: new Float64Array(n * (historyLength + 2)) });
    c.set(this.keyLambda, { int: false, cols: history2Length + 1, data: new Float64Array(n * (history2Length + 1)) });
  }

  /** modify_param: fix_modify ID cut_lo cut_hi (the key is cut_lo; one more value is consumed). */
  modify(key: string, values: string[]): number {
    if (key === 'energy' || key === 'virial') return super.modify(key, values);
    if (values.length < 1) throw new StyleError('Illegal fix_modify lambda/apip command: missing argument(s)');
    const num = (w: string): number => {
      const v = Number(w);
      if (w === '' || !Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
      return v;
    };
    const lo = num(key), hi = num(values[0]);
    this.cutLo = lo;
    this.cutHi = hi;
    this.cutHiSq = hi * hi;
    this.cutWidth = hi - lo;
    if (lo < 0 || hi < lo) throw new StyleError('fix lambda/apip: Illegal cutoff values');
    // force->pair->cutforce: zero until the pair style has been initialised by a run, as in native
    const nb = this.sys.nb;
    const cutforce = nb.cutneighmax > 0 ? nb.cutneighmax - nb.skin : 0;
    if (cutforce < hi) throw new StyleError('fix lambda/apip: cutoff of potential smaller than cutoff of switching region');
    return 1;
  }

  destroy(): void {
    this.sys.state.custom.delete(this.keyInput);
    this.sys.state.custom.delete(this.keyLambda);
  }

  init(): void {
    const sys = this.sys;
    if (!sys.ff.pair) throw new StyleError('Fix lambda/apip requires a pair style be defined');
    if (sys.fixes.filter((f) => f.style === 'lambda/apip').length > 1) throw new StyleError('More than one fix lambda/apip.');
    if (!sys.fixes.some((f) => f.style === 'lambda_thermostat/apip')) {
      sys.warn('The energy is not conserved when lambda changes as fix lambda_thermostat/apip is not used.');
    }
    const pairs = pairStyles(sys);
    const input = pairs.find((p) => nameOf(p).startsWith('lambda/input/'));
    if (!input) throw new StyleError('fix lambda requires a `pair lambda_input`');
    this.pairInput = input as PairLambdaInputAPIP;
    const zone = pairs.find((p) => nameOf(p) === 'lambda/zone/apip');
    if (!zone) throw new StyleError('fix lambda requires a `pair lambda`');
    this.pairZone = zone as PairLambdaZoneAPIP;
    if (sys.atomStyle !== 'apip') throw new StyleError('fix lambda requires atom style apip');
    const bitOf = (name: string): number => {
      if (!sys.groups.names.includes(name)) throw new StyleError(`fix lambda/apip: group ${name} does not exist`);
      return sys.groups.bit(name);
    };
    if (this.groupNameSimple) this.groupBitSimple = bitOf(this.groupNameSimple);
    if (this.groupNameComplex) this.groupBitComplex = bitOf(this.groupNameComplex);
    if (this.groupNameIgnore) this.groupBitIgnoreLambdaInput = bitOf(this.groupNameIgnore);
  }

  /** setup(): native has none (the engine default would call postForce). */
  setup(): void {}

  setupPreForce(): void {
    const sys = this.sys;
    const nb = sys.nb;
    // init: "cutoff of potential smaller than cutoff of switching region" (force->pair->cutforce < cut_hi)
    const cutforce = nb.cutneighmax > 0 ? nb.cutneighmax - nb.skin : 0;
    if (cutforce < this.cutHi) throw new StyleError('fix lambda/apip: cutoff of potential smaller than cutoff of switching region');
    this.pairInput!.calculateLambdaInput(sys.state, nb, sys.geom);
    this.updateLambdaInputHistory();
    this.calculateLambdaInputTa();
    this.pairZone!.calculateLambda(sys.state, nb);
    this.postIntegrate();
    // comm_forward_lambda: lambda_const = lambda after the first average (history2_used == 1)
    if (this.history2Used === 1) {
      const s = sys.state;
      const lam = s.custom.get(APIP_LAMBDA)!.data, lc = s.custom.get(APIP_LAMBDA_CONST)!.data;
      for (let i = 0; i < s.n; i++) lc[i] = lam[i];
    }
    this.calculateLambdaInputTa();
    this.writePeratomStats();
  }

  /** post_integrate: lambda_const = lambda, lambda = the time average of lambda_input_ta (with min_delta_lambda). */
  postIntegrate(): void {
    const s = this.sys.state;
    const lam = s.custom.get(APIP_LAMBDA)!.data;
    const lc = s.custom.get(APIP_LAMBDA_CONST)!.data;
    const lta = s.custom.get(APIP_LAMBDA_INPUT_TA)!.data;
    this.updateLambdaHistory();
    this.getLambdaAverage();
    for (let i = 0; i < s.n; i++) {
      lc[i] = lam[i];
      lam[i] = lta[i];
      if (Math.abs(lam[i] - lc[i]) < this.minDeltaLambda && lam[i] !== 1 && lam[i] !== 0) lam[i] = lc[i];
    }
    this.calculateLambdaInputTa();
  }

  endOfStep(): void {
    this.writePeratomStats();
  }

  private writePeratomStats(): void {
    if (!this.peratomFlag) return;
    const s = this.sys.state;
    const hl = this.historyLength, h2 = this.history2Length;
    const hi = s.custom.get(this.keyInput)!.data, hlam = s.custom.get(this.keyLambda)!.data;
    const w1 = hl + 2, w2 = h2 + 1;
    const cols = this.sizePeratomCols;
    if (this.arrayAtom.length !== s.n * cols) this.arrayAtom = new Float64Array(s.n * cols);
    const out = this.arrayAtom;
    for (let i = 0; i < s.n; i++) {
      out[i * cols] = hi[i * w1 + this.historyLast];
      out[i * cols + 1] = hi[i * w1 + hl] / this.historyUsed;
      out[i * cols + 2] = hi[i * w1 + hl + 1];
      out[i * cols + 3] = hlam[i * w2 + this.history2Last];
      out[i * cols + 4] = hlam[i * w2 + h2] / this.history2Used;
    }
    if (this.dumpHistory) {
      for (let i = 0; i < s.n; i++) {
        for (let j = 0; j <= hl; j++) out[i * cols + j + 5] = hi[i * w1 + j];
        for (let j = 0; j <= h2; j++) out[i * cols + j + 6 + hl] = hlam[i * w2 + j];
      }
    }
  }

  /** update_lambda_input_history (once per step): the running sum of lambda_input and the switching value. */
  updateLambdaInputHistory(): void {
    const s = this.sys.state;
    if (this.invokedHistoryUpdate === s.step) return;
    this.invokedHistoryUpdate = s.step;
    const li = s.custom.get(APIP_LAMBDA_INPUT)!.data;
    const hist = s.custom.get(this.keyInput)!.data;
    const hl = this.historyLength, w = hl + 2;
    this.historyLast = (this.historyLast + 1) % hl;
    this.historyUsed = Math.min(this.historyUsed + 1, hl);
    const gb = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      const o = i * w;
      const m = s.mask[i];
      if (!(m & gb)) {
        hist[o + hl + 1] = this.lambdaNonGroup;
        continue;
      }
      hist[o + hl] -= hist[o + this.historyLast];
      hist[o + this.historyLast] = li[i];
      hist[o + hl] += hist[o + this.historyLast];
      if (this.groupNameComplex && (m & this.groupBitComplex)) {
        hist[o + hl + 1] = 0;
      } else if (this.groupNameSimple && (m & this.groupBitSimple)) {
        hist[o + hl + 1] = 1;
      } else if ((m & gb) && !(this.groupNameIgnore && (m & this.groupBitIgnoreLambdaInput))) {
        hist[o + hl + 1] = this.switchingFunctionPoly(hist[o + hl] / this.historyUsed);
      } else {
        hist[o + hl + 1] = this.lambdaNonGroup;
      }
    }
  }

  /** update_lambda_history (once per step): the running sum of lambda_input_ta. */
  private updateLambdaHistory(): void {
    const s = this.sys.state;
    if (this.invokedHistory2Update === s.step) return;
    this.invokedHistory2Update = s.step;
    const lta = s.custom.get(APIP_LAMBDA_INPUT_TA)!.data;
    const hist = s.custom.get(this.keyLambda)!.data;
    const h2 = this.history2Length, w = h2 + 1;
    this.history2Last = (this.history2Last + 1) % h2;
    this.history2Used = Math.min(this.history2Used + 1, h2);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const o = i * w;
      hist[o + h2] -= hist[o + this.history2Last];
      hist[o + this.history2Last] = lta[i];
      hist[o + h2] += hist[o + this.history2Last];
    }
  }

  /** get_lambda_average: lambda_input_ta = the average (1 above 0.9999, 0 below 0.0001); the sum is renewed per lap. */
  private getLambdaAverage(): void {
    const s = this.sys.state;
    const lta = s.custom.get(APIP_LAMBDA_INPUT_TA)!.data;
    const hist = s.custom.get(this.keyLambda)!.data;
    const h2 = this.history2Length, w = h2 + 1;
    if (this.history2Last === h2 - 1) {
      for (let i = 0; i < s.n; i++) {
        let sum = 0;
        for (let j = 0; j < h2; j++) sum += hist[i * w + j];
        hist[i * w + h2] = sum;
      }
    }
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const avg = hist[i * w + h2] / this.history2Used;
      lta[i] = avg > 0.9999 ? 1 : avg < 0.0001 ? 0 : avg;
    }
  }

  /** calculate_lambda_input_ta: the switching value of the lambda_input history. */
  private calculateLambdaInputTa(): void {
    const s = this.sys.state;
    const lta = s.custom.get(APIP_LAMBDA_INPUT_TA)!.data;
    const hist = s.custom.get(this.keyInput)!.data;
    const hl = this.historyLength, w = hl + 2;
    for (let i = 0; i < s.n; i++) lta[i] = hist[i * w + hl + 1];
  }

  private switchingFunctionPoly(input: number): number {
    if (input <= this.thresholdLo) return 1;
    if (input >= this.thresholdHi) return 0;
    const d = 1 - 2 * (1 + (input - this.thresholdHi) / this.thresholdWidth);
    return 0.5 + 7.5 / 2. * (d / 4. - libmPow(d, 3) / 6. + libmPow(d, 5) / 20.);
  }

  /** extract("fix_lambda:lambda_input_history") for fix atom_weight/apip: column history_length + 1 of atom i. */
  lambdaOfInputHistory(i: number): number {
    const hl = this.historyLength;
    return this.sys.state.custom.get(this.keyInput)!.data[i * (hl + 2) + hl + 1];
  }
}
