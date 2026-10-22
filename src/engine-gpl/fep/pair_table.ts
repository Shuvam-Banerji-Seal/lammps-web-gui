/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/fix_adapt_fep.cpp and src/FEP/compute_fep.cpp (pair lookup through pair->extract(), and the
 * type-pair ranges of utils::bounds_typelabel) at commit b332f28, by Agilio Padua (ENS de Lyon & CNRS) and the
 * LAMMPS developers. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Shared helpers of fix adapt/fep and compute fep: the pair parameter table a "pair pstyle param I J" argument
 * names (the defined pair style, or a hybrid sub-style with the same name rules as fix adapt), the type-pair
 * list of an I J range, and the energy bookkeeping of compute fep.
 */
import { Pair, PairParams, StyleError, typeBounds } from '../../engine/force/types';
import { PairHybrid } from '../../engine/force/pair/hybrid';
import type { System } from '../../engine/system';

/** A pair parameter table (PairParams) of a defined pair style or of one hybrid sub-style. */
export interface PairTable {
  /** The (sub-)style that owns the table. */
  style: Pair;
  /** The defined pair style; init() of this one re-derives the tables. */
  owner: Pair;
  /** Engine name of the parameter (pair_coeff spelling). */
  name: string;
  table: PairParams;
}

/** Looks up pstyle (style or style:N) and the parameter pparam (case-insensitive) for the fix or compute named id. */
export const resolvePairTable = (sys: System, id: string, pstyle: string, pparam: string): PairTable => {
  const owner = sys.ff.pair;
  if (!owner) throw new StyleError(`${id}: no pair style is defined (pair ${pstyle})`);
  const m = /^([^:]+)(?::(\d+))?$/.exec(pstyle);
  if (!m) throw new StyleError(`${id}: pair style selection '${pstyle}' must be style or style:N`);
  const base = m[1];
  const inst = m[2] === undefined ? undefined : Number(m[2]);
  let style: Pair;
  if (owner instanceof PairHybrid) {
    const subs = owner.subs.filter((x) => x.name === base);
    if (!subs.length) throw new StyleError(`${id}: pair sub-style ${base} is not in pair_style ${owner.name}`);
    if (inst === undefined && subs.length > 1) throw new StyleError(`${id}: pair sub-style ${base} is used ${subs.length} times; select one with ${base}:N`);
    const sub = inst === undefined ? subs[0] : subs.find((x) => x.instance === inst);
    if (!sub) throw new StyleError(`${id}: pair sub-style ${base}:${inst} does not exist`);
    style = sub.style;
  } else {
    if (owner.name !== base) throw new StyleError(`${id}: pair style ${base} is not the defined pair style (${owner.name})`);
    if (inst !== undefined && inst !== 1) throw new StyleError(`${id}: pair style ${base}:${inst} does not exist (the pair style is not a hybrid style)`);
    style = owner;
  }
  const lname = pparam.toLowerCase();
  const table = Object.values(style).find((v): v is PairParams => v instanceof PairParams && v.names.some((n) => n.toLowerCase() === lname));
  if (!table) throw new StyleError(`${id}: pair style ${base} param ${pparam} is not supported`);
  const name = table.names.find((n) => n.toLowerCase() === lname)!;
  return { style, owner, name, table };
};

/**
 * The type pairs (I <= J) of an I J range. "*" and "n*m" ranges as in typeBounds. A range with one
 * type on each side is swapped when I > J (fix adapt/fep does this; compute fep does not).
 */
export const typePairs = (iw: string, jw: string, ntypes: number, swap: boolean): Array<[number, number]> => {
  let [ilo, ihi] = typeBounds(iw, ntypes);
  let [jlo, jhi] = typeBounds(jw, ntypes);
  if (swap && ilo === ihi && jlo === jhi && ilo > jlo) [ilo, ihi, jlo, jhi] = [jlo, jhi, ilo, ihi];
  const out: Array<[number, number]> = [];
  for (let i = ilo; i <= ihi; i++) for (let j = Math.max(jlo, i); j <= jhi; j++) out.push([i, j]);
  return out;
};

/** Writes one value to the (i, j) and (j, i) entries of a table. */
export const setPairValue = (t: PairTable, i: number, j: number, value: number): void => {
  const arr = t.table.p(t.name);
  arr[t.table.idx(i, j)] = value;
  arr[t.table.idx(j, i)] = value;
};

/** pair_modify tail yes on the style or a hybrid sub-style (fix adapt and compute fep check it). */
export const hasTail = (p: Pair): boolean => {
  if (p.tail) return true;
  return p instanceof PairHybrid && p.subs.some((x) => x.style.tail);
};

/** After a table changed: re-initializes the defined pair style (it re-derives its tables), updates the tail. */
export const reinitPair = (sys: System, owner: Pair): void => {
  owner.init(sys.styleContext());
  if (hasTail(owner)) sys.ff.reinitTail(sys.state);
  owner.version++;
  sys.ff.pairStale = true;
};

/**
 * Re-derives the kspace charge sums after the charges changed (native kspace->qsum_qsq()). The engine's kspace
 * styles keep qsum and qsqsum from init(), and their constant energy term (the self energy) reads those, so a
 * changed charge would otherwise leave the k-space energy at its old value.
 */
export const refreshChargeSums = (sys: System): void => {
  sys.nb.refreshCharges(sys.state);
  const k = sys.ff.kspace as { qsum?: number; qsqsum?: number } | null;
  if (!k || !('qsqsum' in k)) return;
  const s = sys.state;
  let qs = 0, q2 = 0;
  for (let i = 0; i < s.n; i++) { qs += s.q[i]; q2 += s.q[i] * s.q[i]; }
  k.qsum = qs;
  k.qsqsum = q2;
};
