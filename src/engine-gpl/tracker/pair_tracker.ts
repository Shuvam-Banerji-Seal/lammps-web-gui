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

/** One pair's history: pair_tracker.cpp data[0] (created step), data[1] (sum of r), data[2] (min r) and touch. */
interface Contact {
  created: number;
  sum: number;
  min: number;
  touch: boolean;
  gen: number;
}

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
  private hist = new Map<number, Contact>();
  private gen = 0;
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
    this.hist.clear();
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
    const gen = ++this.gen;
    this.cur = pc;
    let visited = 0;
    for (let p = 0; p < pairs.length; p += 2) {
      const i = pairs[p];
      const j = pairs[p + 1] & NEIGHMASK;
      const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const r = Math.sqrt(rsq);
      const key = pairKey(tag[owner[i]], tag[owner[j]]);
      let c = this.hist.get(key);
      if (!c) {
        c = { created: 0, sum: 0, min: 0, touch: false, gen };
        this.hist.set(key, c);
      }
      c.gen = gen;
      visited++;
      if (rsq >= cutsq[type[i] * nt + type[j]]) {
        // the contact is broken: write its row (when touched), then reset the pair
        if (c.touch) this.processData(i, j, c, step);
        c.touch = false;
        c.created = 0;
        c.sum = 0;
        c.min = 0;
      } else {
        if (!c.touch) {
          c.created = step;
          c.sum = r;
          c.min = r;
        } else if (update) {
          c.sum += r;
          if (c.min > r) c.min = r;
        }
        c.touch = true;
      }
    }
    // a pair that left the neighbor list (only at a rebuild) loses its history (fix neigh/history drops it)
    if (this.hist.size !== visited) {
      for (const [key, c] of this.hist) if (c.gen !== gen) this.hist.delete(key);
    }
    this.cur = null;
  }

  /** pair_tracker.cpp process_data: the tmin and type filters, then one row into the fix. */
  private processData(i: number, j: number, c: Contact, step: number): void {
    if (step - c.created < this.tmin) return;
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
        case 2: out[n] = c.created; break;
        case 3: out[n] = step; break;
        case 4: out[n] = step - c.created; break;
        case 5: out[n] = (x[3 * i] + x[3 * j]) * 0.5; break;
        case 6: out[n] = (x[3 * i + 1] + x[3 * j + 1]) * 0.5; break;
        case 7: out[n] = (x[3 * i + 2] + x[3 * j + 2]) * 0.5; break;
        case 8: out[n] = c.min; break;
        default: out[n] = c.sum / (step - c.created); break;
      }
    }
    this.fix!.addData(out);
  }
}
