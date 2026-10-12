/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/fix_adapt_fep.cpp and src/FEP/fix_adapt_fep.h (class FixAdaptFEP, fix adapt/fep) at commit
 * b332f28, by Agilio Padua (ENS de Lyon & CNRS) and the LAMMPS developers. Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix ID group-ID adapt/fep N keyword args ... keyword value ... (docs.lammps.org/fix_adapt_fep.html). Differences
 * from fix adapt (src/engine/fix/adapt.ts): the pair keyword takes the soft-core parameter lambda of the
 * lj/cut/coul/long/soft family (any pair parameter table of the defined style is accepted); the keywords
 * reset, scale and after; the atom keyword takes charge only; the timing follows fix_adapt_fep.cpp
 * (pre_force: "if (afterflag) ... update at n+1"). Not supported (StyleError): kspace, atom diameter, dynamic
 * groups with atom attributes, and respa.
 *
 * Source notes kept from fix_adapt_fep.cpp: setup applies the variables at every run start (setup_pre_force);
 * pre_force applies at steps where ntimestep % N == 0, or with "after yes" at steps N+1, 2N+1, ... (and
 * always when N is 1 or at step 0); post_run restores the original values when reset is yes. The original
 * per-atom charges are the values the fix saw when it was defined (the STORE/ATOM vector of the native fix).
 */
import { Fix } from '../../engine/fix/fix';
import { Pair, StyleError, typeBounds } from '../../engine/force/types';
import { hasChargeStyle } from '../../engine/atoms';
import type { System } from '../../engine/system';
import { refreshChargeSums, reinitPair, resolvePairTable, setPairValue, typePairs, type PairTable } from './pair_table';

const USAGE = 'usage: fix ID group-ID adapt/fep N keyword args ... keyword value ...';

const logical = (key: string, w: string | undefined): boolean => {
  if (w !== 'yes' && w !== 'no') throw new StyleError(`fix adapt/fep keyword ${key} must be yes or no, got '${w ?? ''}'`);
  return w === 'yes';
};

interface PairAdapt {
  vname: string;
  table: PairTable;
  pairs: Array<[number, number]>;
  orig: Float64Array;
}

interface ChargeAdapt {
  vname: string;
  /** Type range as given (resolved against ntypes at run start). */
  iw: string;
}

export class FixAdaptFEP extends Fix {
  readonly style = 'adapt/fep';
  private readonly pending: Array<{ pstyle: string; pparam: string; iw: string; jw: string; vname: string }> = [];
  private readonly charges: ChargeAdapt[] = [];
  private pairs: PairAdapt[] = [];
  private reset = false;
  private scale = false;
  private after = false;
  /** Charges as the fix saw them when defined, by atom ID (reset yes restores them). */
  private readonly origQ = new Map<number, number>();
  private definedQ = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const n = Number(args[0]);
    if (args[0] === undefined || !Number.isInteger(n) || n < 0) throw new StyleError(`fix ${id} (adapt/fep): N must be a non-negative integer, got '${args[0] ?? ''}' (${USAGE})`);
    this.nevery = n;
    let k = 1;
    let nattr = 0;
    while (k < args.length) {
      const w = args[k];
      if (w === 'reset' || w === 'scale' || w === 'after') {
        const v = logical(w, args[k + 1]);
        if (w === 'reset') this.reset = v;
        else if (w === 'scale') this.scale = v;
        else this.after = v;
        k += 2;
        continue;
      }
      if (w === 'pair') {
        const pstyle = args[k + 1], pparam = args[k + 2], iw = args[k + 3], jw = args[k + 4], vw = args[k + 5];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`fix ${id} (adapt/fep): pair needs pstyle pparam I J v_name (${USAGE})`);
        this.pending.push({ pstyle, pparam, iw, jw, vname: vw.slice(2) });
        k += 6; nattr++;
        continue;
      }
      if (w === 'atom') {
        const kind = args[k + 1], iw = args[k + 2], vw = args[k + 3];
        if (vw === undefined || !vw.startsWith('v_')) throw new StyleError(`fix ${id} (adapt/fep): atom needs atomparam type v_name (${USAGE})`);
        if (kind === 'diameter') throw new StyleError(`fix ${id} (adapt/fep): atom diameter is not supported by this engine (only atom charge)`);
        if (kind !== 'charge') throw new StyleError(`fix ${id} (adapt/fep): atom parameter '${kind}' is not supported (charge)`);
        this.charges.push({ vname: vw.slice(2), iw });
        k += 4; nattr++;
        continue;
      }
      if (w === 'kspace') throw new StyleError(`fix ${id} (adapt/fep): attribute kspace is not supported by this engine (pair and atom charge are)`);
      throw new StyleError(`fix ${id} (adapt/fep): unknown attribute or keyword '${w}' (${USAGE})`);
    }
    if (nattr === 0) throw new StyleError(`fix ${id} (adapt/fep): nothing to adapt (${USAGE})`);
    if (this.scale && this.charges.length) throw new StyleError(`fix ${id} (adapt/fep): keyword 'scale yes' is not supported with 'atom'`);
    // the STORE/ATOM vector of the native fix is filled when the fix is defined
    if (this.reset && this.charges.length) this.storeCharges();
  }

  private storeCharges(): void {
    const s = this.sys.state;
    this.origQ.clear();
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.origQ.set(s.id[i], s.q[i]);
    this.definedQ = true;
  }

  /**
   * Resolves the targets, copies the original pair values and applies the variables. The native fix does the
   * latter in setup_pre_force, before the first force evaluation of each run; this engine runs init() at the
   * start of every run, before the setup forces.
   */
  init(): void {
    const s = this.sys.state;
    if (this.charges.length) {
      if (!hasChargeStyle(s.atomStyle)) throw new StyleError(`fix ${this.id} (adapt/fep): requires atom attribute charge`);
    }
    this.pairs = [];
    for (const p of this.pending) {
      const table = resolvePairTable(this.sys, `fix ${this.id} (adapt/fep)`, p.pstyle, p.pparam);
      const pairs = typePairs(p.iw, p.jw, s.ntypes, true);
      if (!pairs.length) throw new StyleError(`fix ${this.id} (adapt/fep): no type pairs with I <= J in ${p.iw} ${p.jw}`);
      const orig = new Float64Array(pairs.length);
      pairs.forEach(([i, j], k) => { orig[k] = table.table.get(table.name, i, j); });
      this.pairs.push({ vname: p.vname, table, pairs, orig });
    }
    if (this.reset && this.charges.length && !this.definedQ) this.storeCharges();
    this.apply();
  }

  /** The variables are written to the tables and the atoms (change_settings). */
  private apply(): void {
    const sys = this.sys, s = sys.state;
    const owners = new Set<Pair>();
    for (const p of this.pairs) {
      const v = sys.equalVariable(p.vname);
      p.pairs.forEach(([i, j], k) => {
        const val = this.scale ? v * p.orig[k] : v;
        setPairValue(p.table, i, j, val);
      });
      owners.add(p.table.owner);
    }
    for (const owner of owners) reinitPair(sys, owner);
    if (this.charges.length) {
      for (const c of this.charges) {
        const v = sys.equalVariable(c.vname);
        const [ilo, ihi] = typeBounds(c.iw, s.ntypes);
        for (let i = 0; i < s.n; i++) {
          if (s.type[i] >= ilo && s.type[i] <= ihi && (s.mask[i] & this.groupBit)) s.q[i] = v;
        }
      }
      refreshChargeSums(sys);
    }
  }

  preForce(): void {
    if (this.nevery === 0) return;
    const step = this.sys.state.step;
    if (this.after) {
      if (this.nevery === 1 || step === 0) this.apply();
      else if (step > 1 && (step - 1) % this.nevery === 0) this.apply();
    } else if (step % this.nevery === 0) this.apply();
  }

  /** post_run: reset yes restores the values of the run start (pair orig, charges as defined). */
  postRun(): void {
    if (!this.reset) return;
    const sys = this.sys, s = sys.state;
    const owners = new Set<Pair>();
    for (const p of this.pairs) {
      p.pairs.forEach(([i, j], k) => setPairValue(p.table, i, j, p.orig[k]));
      owners.add(p.table.owner);
    }
    for (const owner of owners) reinitPair(sys, owner);
    if (this.charges.length) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        const q = this.origQ.get(s.id[i]);
        if (q !== undefined) s.q[i] = q;
      }
      refreshChargeSums(sys);
    }
  }
}
