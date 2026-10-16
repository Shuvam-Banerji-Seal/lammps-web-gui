/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the order in which std::sort of the GCC
 * libstdc++ (GCC 11, the compiler of the native binary) rearranges equal keys. The PTM neighbour ordering
 * (src/PTM/ptm_neighbour_ordering.cpp at b332f28) calls std::sort with a boolean predicate on records that can
 * have equal keys, so the native order of ties is the libstdc++ introsort order. This file reproduces that
 * algorithm (bits/stl_algo.h: std::sort, __introsort_loop, __unguarded_partition_pivot, __final_insertion_sort,
 * __insertion_sort, __heap_select, __sort_heap; threshold 16 elements). libstdc++ is GPL-3.0 with the GCC Runtime
 * Library Exception; the algorithm (not its code) is the reference here, and the code is written from it. This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

/** Strict weak ordering predicate, as in std::sort: comp(a, b) is true when a must come before b. */
export type LessFn<T> = (a: T, b: T) => boolean;

const THRESHOLD = 16;

/** __lg: floor(log2(n)) for n >= 1. */
const lg = (n: number): number => Math.floor(Math.log2(n));

/** std::sort(first, last, comp) on a[first..last) in place. */
export function std_sort<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  if (last - first > 1) {
    introsort_loop(a, first, last, lg(last - first) * 2, comp);
    final_insertion_sort(a, first, last, comp);
  }
}

function introsort_loop<T>(a: T[], first: number, last: number, depth_limit: number, comp: LessFn<T>): void {
  while (last - first > THRESHOLD) {
    if (depth_limit === 0) {
      partial_sort(a, first, last, last, comp);
      return;
    }
    depth_limit--;
    const cut = unguarded_partition_pivot(a, first, last, comp);
    introsort_loop(a, cut, last, depth_limit, comp);
    last = cut;
  }
}

function unguarded_partition_pivot<T>(a: T[], first: number, last: number, comp: LessFn<T>): number {
  const mid = first + Math.floor((last - first) / 2);
  move_median_to_first(a, first, first + 1, mid, last - 1, comp);
  return unguarded_partition(a, first + 1, last, first, comp);
}

function swap<T>(a: T[], i: number, j: number): void {
  const t = a[i];
  a[i] = a[j];
  a[j] = t;
}

function move_median_to_first<T>(a: T[], result: number, x: number, y: number, z: number, comp: LessFn<T>): void {
  if (comp(a[x], a[y])) {
    if (comp(a[y], a[z])) swap(a, result, y);
    else if (comp(a[x], a[z])) swap(a, result, z);
    else swap(a, result, x);
  } else if (comp(a[x], a[z])) swap(a, result, x);
  else if (comp(a[y], a[z])) swap(a, result, z);
  else swap(a, result, y);
}

function unguarded_partition<T>(a: T[], first: number, last: number, pivot: number, comp: LessFn<T>): number {
  for (;;) {
    while (comp(a[first], a[pivot])) first++;
    last--;
    while (comp(a[pivot], a[last])) last--;
    if (!(first < last)) return first;
    swap(a, first, last);
    first++;
  }
}

function final_insertion_sort<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  if (last - first > THRESHOLD) {
    insertion_sort(a, first, first + THRESHOLD, comp);
    unguarded_insertion_sort(a, first + THRESHOLD, last, comp);
  } else {
    insertion_sort(a, first, last, comp);
  }
}

function insertion_sort<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  for (let i = first + 1; i < last; i++) {
    if (comp(a[i], a[first])) {
      // move_backward(first, i, i + 1); *first = val
      const val = a[i];
      for (let k = i; k > first; k--) a[k] = a[k - 1];
      a[first] = val;
    } else {
      unguarded_linear_insert(a, i, comp);
    }
  }
}

function unguarded_linear_insert<T>(a: T[], last: number, comp: LessFn<T>): void {
  const val = a[last];
  let next = last - 1;
  while (comp(val, a[next])) {
    a[last] = a[next];
    last = next;
    next--;
  }
  a[last] = val;
}

function unguarded_insertion_sort<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  for (let i = first; i < last; i++) unguarded_linear_insert(a, i, comp);
}

/** std::partial_sort(first, middle, last): heap select then sort the heap. */
function partial_sort<T>(a: T[], first: number, middle: number, last: number, comp: LessFn<T>): void {
  heap_select(a, first, middle, last, comp);
  sort_heap(a, first, middle, comp);
}

function heap_select<T>(a: T[], first: number, middle: number, last: number, comp: LessFn<T>): void {
  make_heap(a, first, middle, comp);
  for (let i = middle; i < last; i++) {
    if (comp(a[i], a[first])) pop_heap_into(a, first, middle, i, comp);
  }
}

function sort_heap<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  let n = last;
  while (n - first > 1) {
    n--;
    pop_heap_into(a, first, n, n, comp);
  }
}

/** __pop_heap(first, last, result): move the top to result and restore the heap on [first, last). */
function pop_heap_into<T>(a: T[], first: number, last: number, result: number, comp: LessFn<T>): void {
  const value = a[result];
  a[result] = a[first];
  adjust_heap(a, first, 0, last - first, value, comp);
}

function make_heap<T>(a: T[], first: number, last: number, comp: LessFn<T>): void {
  const len = last - first;
  if (len < 2) return;
  let parent = Math.floor((len - 2) / 2);
  for (;;) {
    adjust_heap(a, first, parent, len, a[first + parent], comp);
    if (parent === 0) return;
    parent--;
  }
}

/** __adjust_heap(first, holeIndex, len, value, comp) with the indices relative to first. */
function adjust_heap<T>(a: T[], first: number, holeIndex: number, len: number, value: T, comp: LessFn<T>): void {
  const topIndex = holeIndex;
  let child = holeIndex;
  while (child < Math.floor((len - 1) / 2)) {
    child = 2 * (child + 1);
    if (comp(a[first + child], a[first + child - 1])) child--;
    a[first + holeIndex] = a[first + child];
    holeIndex = child;
  }
  if ((len & 1) === 0 && child === Math.floor((len - 2) / 2)) {
    child = 2 * (child + 1);
    a[first + holeIndex] = a[first + child - 1];
    holeIndex = child - 1;
  }
  push_heap(a, first, holeIndex, topIndex, value, comp);
}

function push_heap<T>(a: T[], first: number, holeIndex: number, topIndex: number, value: T, comp: LessFn<T>): void {
  let parent = Math.floor((holeIndex - 1) / 2);
  while (holeIndex > topIndex && comp(a[first + parent], value)) {
    a[first + holeIndex] = a[first + parent];
    holeIndex = parent;
    parent = Math.floor((holeIndex - 1) / 2);
  }
  a[first + holeIndex] = value;
}
