/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/pair_tracker.cpp and pair_tracker.h at commit b332f28 (Hudson Oliveira-Bosco and Eric Ferrari
 * for the pair style tracker; the LAMMPS developers). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style tracker: no force; it records, for every neighbor pair, when the pair came into contact (inside the
 * cutoff), when it broke apart and the minimum and sum of its distances, and writes one row per broken contact
 * into the fix named by the style (fix STORE/LOCAL, fix_store_local.ts). finite cutoffs (granular radii) are not
 * supported (StyleError).
 */
import { Pair, pairCoeffArgs, StyleError, typeBounds, mixDistance, type PairCompute, type StyleContext } from '../../engine/force/types';
import { inumeric, numeric, parseNum } from '../../engine/force/util';
import type { System } from '../../engine/system';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { FixStoreLocal } from './fix_store_local';

/** Output values in the order of the keywords (pair_tracker.cpp saved_choices numbering). */
const CHOICES = ['id1', 'id2', 'time/created', 'time/broken', 'time/total', 'x', 'y', 'z', 'r/min', 'r/ave'] as const;


/** Key of an unordered tag pair (tags below 2^21 keep the key exact). */
const pairKey = (a: number, b: number): number => (a < b ? a * 4294967296 + b : b * 4294967296 + a);

export class PairTracker extends Pair {
  readonly name = 'tracker';
  /** No energy or force: the global virial is zero. */
  virialFdotr = true;
  private fixId = '';
  private freq = 0;
  private choices: number[] = [];
  private tmin = -1;
  /** type/include filter (1-based square of ntypes+1), null when not given. */
  private typeFilter: Uint8Array | null = null;
  private set = new Uint8Array(0);
  private cutOwn = new Float64Array(0);
  /**
   * The history of each entry of the list (native's order), as fix neigh/history keeps it (src/fix_neigh_history.cpp at
   * b332f28): pair_tracker.cpp data[0] (created step), data[1] (sum of r), data[2] (min r) and touch, with the entry's
   * atom-ID pair key; carried over to the next build's entries by atom-ID pair (ensureSlots).
   */
  private slotPairs: Int32Array | null = null;
  private slotBuild = -1;
  private sKey = new Float64Array(0);
  private sCreated = new Float64Array(0);
  private sSum = new Float64Array(0);
  private sMin = new Float64Array(0);
  private sTouch = new Uint8Array(0);
  /** The fix STORE/LOCAL the style writes its rows to (created by settings through ctx.addFix). */
  private fix: FixStoreLocal | null = null;
  private row = new Float64Array(CHOICES.length);
  private readonly order = new NativePairOrder();
  /** type/include lists, resolved when the types are known (allocate). */
  private typeFilterArgs: [string, string] | null = null;
  /** The compute() call in progress (its arrays), read by process_data. */
  private cur: PairCompute | null = null;

  settings(args: string[], ctx: StyleContext): void {
    if (args.length < 2) throw new StyleError('Illegal pair_style command');
    if (!ctx.addFix) throw new StyleError('pair_style tracker needs a system to create its fix');
    this.fixId = args[0];
    const freq = inumeric(args[1]);
    if (!Number.isInteger(freq) || freq <= 0) throw new StyleError(`pair_style tracker: output frequency must be a positive integer (got ${args[1]})`);
    this.freq = freq;
    this.choices = [];
    this.clearSlots();
    this.tmin = -1;
    this.typeFilter = null;
    // the keywords are read as the source reads them; finite (granular radii) is refused after
    let finite = false;
    let k = 2;
    while (k < args.length) {
      const w = args[k];
      if (w === 'finite') {
        finite = true;
        k++;
        continue;
      }
      const c = CHOICES.indexOf(w as (typeof CHOICES)[number]);
      if (c >= 0) {
        this.choices.push(c);
      } else if (w === 'time/min') {
        if (k + 1 >= args.length) throw new StyleError('Invalid keyword in pair tracker command');
        this.tmin = numeric(args[k + 1]);
        k++;
      } else if (w === 'type/include') {
        if (k + 2 >= args.length) throw new StyleError('Invalid keyword in pair tracker command');
        this.typeFilterArgs = [args[k + 1], args[k + 2]];
        k += 2;
      } else {
        throw new StyleError('Illegal pair_style command');
      }
      k++;
    }
    if (this.choices.length === 0) throw new StyleError('Must request at least one value to output');
    if (finite) throw new StyleError('pair_style tracker keyword finite is not supported by the browser engine (granular radii)');
    // pair_tracker.cpp settings: modify->add_fix("ID all STORE/LOCAL N nvalues")
    const choices = this.choices.length;
    const id = this.fixId;
    const freqN = this.freq;
    ctx.addFix((sys: System) => (this.fix = new FixStoreLocal(sys, id, freqN, choices)));
  }


  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.set = new Uint8Array(nt * nt);
    this.cutOwn = new Float64Array(nt * nt);
    this.typeFilter = null;
    if (this.typeFilterArgs) {
      const filt = new Uint8Array(nt * nt);
      const [iw, jw] = this.typeFilterArgs;
      for (const ifield of iw.split(',')) {
        const [ilo, ihi] = typeBounds(ifield, ntypes);
        for (const jfield of jw.split(',')) {
          const [jlo, jhi] = typeBounds(jfield, ntypes);
          for (let it = ilo; it <= ihi; it++) for (let jt = jlo; jt <= jhi; jt++) filt[it * nt + jt] = filt[jt * nt + it] = 1;
        }
      }
      this.typeFilter = filt;
    }
  }

  coeff(args: string[]): void {
    // pair_tracker.cpp coeff: "pair_coeff I J cutoff" (the finite form is refused in settings)
    if (args.length !== 3) throw pairCoeffArgs();
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the atom types first');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const cut = parseNum(args[2], 'cutoff');
    const nt = this.ntypes + 1;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.set[i * nt + j] = 1;
        this.cutOwn[i * nt + j] = cut;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients (pair tracker: no type pairs)');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    if (!this.set[i * nt + j]) {
      // pair_tracker.cpp init_one: "if (setflag[i][j] == 0) cut[i][j] = mix_distance(cut[i][i], cut[j][j])"
      this.cutOwn[i * nt + j] = mixDistance(this.mix, this.cutOwn[i * nt + i], this.cutOwn[j * nt + j]);
    }
    this.cutOwn[j * nt + i] = this.cutOwn[i * nt + j];
    return this.cutOwn[i * nt + j];
  }

  private clearSlots(): void {
    this.slotPairs = null;
    this.slotBuild = -1;
    this.sKey = new Float64Array(0); this.sCreated = new Float64Array(0); this.sSum = new Float64Array(0);
    this.sMin = new Float64Array(0); this.sTouch = new Uint8Array(0);
  }

  /**
   * The slots of this build's entries: a touching contact of the previous entries carries over by its atom-ID pair (the
   * first entry of that pair, as fix neigh/history's post_neighbor takes the first partner with the tag); every other
   * entry starts untouched with zeros, which is what a non-touching contact holds. A pair that left the list is gone.
   */
  private ensureSlots(pairs: Int32Array, nbuild: number, tag: ArrayLike<number>, owner: Int32Array): void {
    if (pairs === this.slotPairs && nbuild === this.slotBuild) return;
    const oldKey = this.sKey, oldTouch = this.sTouch, oldCreated = this.sCreated, oldSum = this.sSum, oldMin = this.sMin;
    let prev: Map<number, number> | null = null;
    for (let k = 0; k < oldTouch.length; k++) {
      if (!oldTouch[k]) continue;
      prev ??= new Map();
      if (!prev.has(oldKey[k])) prev.set(oldKey[k], k);
    }
    const n = pairs.length >> 1;
    const key = new Float64Array(n), created = new Float64Array(n), sum = new Float64Array(n), min = new Float64Array(n);
    const touch = new Uint8Array(n);
    for (let e = 0; e < n; e++) {
      const k = pairKey(tag[owner[pairs[2 * e]]], tag[owner[pairs[2 * e + 1] & NEIGHMASK]]);
      key[e] = k;
      const o = prev?.get(k);
      if (o === undefined) continue;
      created[e] = oldCreated[o]; sum[e] = oldSum[o]; min[e] = oldMin[o]; touch[e] = 1;
    }
    this.sKey = key; this.sCreated = created; this.sSum = sum; this.sMin = min; this.sTouch = touch;
    this.slotPairs = pairs;
    this.slotBuild = nbuild;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const { x } = pc;
    const type = pc.type;
    const step = s.step;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const update = pc.historyUpdate === true;
    const tag = s.id;
    const owner = pc.nb.owner;
    // the pairs in native's order (nativeOrder.ts): the row order and the i/j of each row follow native's list
    const pairs = this.order.pairs(s, pc.nb, pc.geom, pc.half);
    this.ensureSlots(pairs, pc.nb.nbuild, tag, owner);
    const sCreated = this.sCreated, sSum = this.sSum, sMin = this.sMin, sTouch = this.sTouch;
    this.cur = pc;
    for (let p = 0, e = 0; p < pairs.length; p += 2, e++) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const r = Math.sqrt(rsq);
      if (rsq >= cutsq[type[i] * nt + type[j]]) {
        // the contact is broken: write its row (when touched), then reset the pair
        if (sTouch[e]) this.processData(i, j, sCreated[e], sSum[e], sMin[e], step);
        sTouch[e] = 0;
        sCreated[e] = 0;
        sSum[e] = 0;
        sMin[e] = 0;
      } else {
        if (!sTouch[e]) {
          sCreated[e] = step;
          sSum[e] = r;
          sMin[e] = r;
        } else if (update) {
          sSum[e] += r;
          if (sMin[e] > r) sMin[e] = r;
        }
        sTouch[e] = 1;
      }
    }
    this.cur = null;
  }

  /** pair_tracker.cpp process_data: the tmin and type filters, then one row into the fix. */
  private processData(i: number, j: number, created: number, sum: number, min: number, step: number): void {
    if (step - created < this.tmin) return;
    const pc = this.cur!;
    if (this.typeFilter) {
      // "if (type_filter[type[i]][type[j]] == 0) return;"
      const nt = this.ntypes + 1;
      if (this.typeFilter[pc.type[i] * nt + pc.type[j]] === 0) return;
    }
    const owner = pc.nb.owner;
    const tag = pc.s.id;
    const x = pc.x;
    const out = this.row;
    for (let n = 0; n < this.choices.length; n++) {
      switch (this.choices[n]) {
        case 0: out[n] = tag[owner[i]]; break;
        case 1: out[n] = tag[owner[j]]; break;
        case 2: out[n] = created; break;
        case 3: out[n] = step; break;
        case 4: out[n] = step - created; break;
        case 5: out[n] = (x[3 * i] + x[3 * j]) * 0.5; break;
        case 6: out[n] = (x[3 * i + 1] + x[3 * j + 1]) * 0.5; break;
        case 7: out[n] = (x[3 * i + 2] + x[3 * j + 2]) * 0.5; break;
        case 8: out[n] = min; break;
        default: out[n] = sum / (step - created); break;
      }
    }
    this.fix!.addData(out);
  }
}
