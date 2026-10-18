/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the argument helpers of the
 * EXTRA-COMPUTE styles (utils::numeric, utils::inumeric, utils::logical, utils::bounds of src/utils.cpp) and the
 * Numerical Recipes select2 that compute_ackland_atom.cpp and compute_ti.cpp use. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import type { NeighList } from '../../engine/neighbor';
import type { System } from '../../engine/system';

/** utils::numeric: a floating-point argument; the native error text for anything else. */
export const argNumeric = (w: string | undefined): number => {
  const v = w === undefined || w.trim() === '' ? NaN : Number(w);
  if (!Number.isFinite(v)) throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  return v;
};

/** utils::inumeric: an integer argument. */
export const argInt = (w: string | undefined): number => {
  if (w === undefined || !/^[+-]?\d+$/.test(w.trim())) throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  return Number(w);
};

/** utils::logical: yes/no, on/off, true/false, 1/0 (case-sensitive as in native). */
export const argLogical = (w: string | undefined): boolean => {
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w ?? ''}' in input script or data file`);
};

/**
 * utils::bounds for atom types: "*" (all), "*n" (1..n), "m*" (m..ntypes), "m*n" (m..n) or "n". The native error text
 * for an index outside 1..ntypes is kept.
 */
export const argBounds = (w: string, ntypes: number, what: string): [number, number] => {
  const bad = (): never => { throw new StyleError(`Illegal ${what} type range '${w}'`); };
  const num = (t: string): number => {
    if (!/^\d+$/.test(t)) bad();
    return Number(t);
  };
  let lo: number, hi: number;
  const star = w.indexOf('*');
  if (star < 0) {
    lo = hi = num(w);
  } else {
    const a = w.slice(0, star), b = w.slice(star + 1);
    if (a === '' && b === '') { lo = 1; hi = ntypes; }
    else if (a === '') { lo = 1; hi = num(b); }
    else if (b === '') { lo = num(a); hi = ntypes; }
    else { lo = num(a); hi = num(b); }
  }
  if (lo < 1 || hi > ntypes || lo > hi) throw new StyleError(`Numeric index ${lo < 1 ? lo : hi} is out of bounds (1-${ntypes})`);
  return [lo, hi];
};

/**
 * select2 (Numerical Recipes, "slightly modified" in compute_ackland_atom.cpp): the k smallest values of arr[0..n-1]
 * are moved to the front, arr and iarr are permuted together. The algorithm runs on 1-based working copies, as the C
 * code does with its decremented pointers, so the permutation (and hence any tie) is the same.
 */
export const select2 = (k: number, n: number, arr: Float64Array, iarr: Int32Array): void => {
  const a = new Float64Array(n + 2);
  const ia = new Int32Array(n + 2);
  for (let t = 0; t < n; t++) { a[t + 1] = arr[t]; ia[t + 1] = iarr[t]; }
  const swap = (p: number, q: number): void => {
    const tv = a[p]; a[p] = a[q]; a[q] = tv;
    const ti = ia[p]; ia[p] = ia[q]; ia[q] = ti;
  };
  let l = 1, ir = n;
  for (;;) {
    if (ir <= l + 1) {
      if (ir === l + 1 && a[ir] < a[l]) swap(l, ir);
      break;
    }
    const mid = (l + ir) >> 1;
    swap(mid, l + 1);
    if (a[l] > a[ir]) swap(l, ir);
    if (a[l + 1] > a[ir]) swap(l + 1, ir);
    if (a[l] > a[l + 1]) swap(l, l + 1);
    let i = l + 1, j = ir;
    const pivot = a[l + 1];
    const pivotI = ia[l + 1];
    for (;;) {
      do i++; while (a[i] < pivot);
      do j--; while (a[j] > pivot);
      if (j < i) break;
      swap(i, j);
    }
    a[l + 1] = a[j];
    a[j] = pivot;
    ia[l + 1] = ia[j];
    ia[j] = pivotI;
    if (j >= k) ir = j - 1;
    if (j <= k) l = i;
  }
  for (let t = 0; t < n; t++) { arr[t] = a[t + 1]; iarr[t] = ia[t + 1]; }
};

/** The largest force cutoff of the pair style over its type pairs (Pair::cutforce). */
export const pairCutforce = (cut: Float64Array): number => {
  let c = 0;
  for (let k = 0; k < cut.length; k++) if (cut[k] > c) c = cut[k];
  return c;
};

/**
 * The occasional neighbour list of a compute, as native LAMMPS keeps it (Neighbor::build_one, src/neighbor.cpp at
 * b332f28: "no need to build this occasional list if already built since last comm->exchange() and re-neighbor";
 * neighbor.cpp init: an occasional list takes its cutoff without the skin, "if (!irq->occasional) irq->cutoff += skin").
 * The membership is fixed by the first request after each re-neighbour (the pairs within the cutoff at that time) and
 * reused until the next one; the distances are the current ones, and the compute applies its own cutoff. The engine
 * rebuilds its ghosts and lists only at a re-neighbour (Neighbor.nbuild), so the snapshot is keyed on that counter.
 */
export class OccasionalSnapshot {
  private list: NeighList | null = null;
  private build = -1;
  private cut = Number.NaN;

  /** The list for `cutoff` (a fixed cutoff, or the pair cutoff plus the skin for the default list). */
  get(sys: System, cutoff: number): NeighList {
    const nb = sys.nb;
    if (this.list && this.build === nb.nbuild && this.cut === cutoff) return this.list;
    this.list = nb.occasionalList(sys.state, cutoff, true, sys.state.ntypes);
    this.build = nb.nbuild;
    this.cut = cutoff;
    return this.list;
  }
}

/** The occasional list cutoff of a compute with cutoff keyword: fixed when cutflag, else the pair cutoff plus the skin. */
export const occasionalCutoff = (sys: System, cutoff: number, cutflag: boolean): number => {
  const pair = sys.ff.pair;
  if (cutflag || !pair) return cutoff;
  return pairCutforce(pair.cut) + sys.nb.skin;
};
