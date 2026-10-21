/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the random stream of the overstep of fix
 * bond/react (src/REACTION/fix_bond_react.cpp post_integrate at b332f28: std::minstd_rand seeded with shuffle_seed at
 * every call, and std::shuffle of the list of the step's reactions of a max_rxn or rate_limit that spans several
 * reactions). The native binary is built with GCC 11, so the sequence is that of libstdc++ 11: std::minstd_rand
 * (bits/random.h: linear_congruential_engine<uint_fast32_t, 48271, 0, 2147483647>, seeded with s mod m, or 1 when that
 * is 0), std::uniform_int_distribution<unsigned long> (bits/uniform_int_dist.h: a generator of range 2^31 - 3 takes
 * the downscaling path with two divisions: scaling = urngrange / (urange + 1), past = (urange + 1) * scaling, draw
 * until below past, divide by scaling) and std::shuffle (bits/stl_algo.h: two swap positions per draw through
 * __gen_two_uniform_ints when the generator range covers the square of the list length, else one draw per
 * position). libstdc++ is GPL-3.0-or-later with the GCC Runtime Library Exception; the algorithms (not the code) are
 * the reference, and the code is written from them. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

const MODULUS = 2147483647;
const MULTIPLIER = 48271;
/** max() - min() of std::minstd_rand. */
const URNGRANGE = MODULUS - 2;

/** std::minstd_rand: x <- 48271 x mod (2^31 - 1), values in [1, 2^31 - 2]. */
export class MinstdRand {
  private x: number;

  /** seed(s): s converted to the unsigned result type, then reduced mod 2^31 - 1 (0 becomes 1). */
  constructor(seed: number) {
    const s = BigInt.asUintN(64, BigInt(Math.trunc(seed)));
    const r = Number(s % BigInt(MODULUS));
    this.x = r === 0 ? 1 : r;
  }

  next(): number {
    // 48271 * x < 2^47: exact in double precision
    this.x = (MULTIPLIER * this.x) % MODULUS;
    return this.x;
  }
}

/** uniform_int_distribution<unsigned long>{a, b}(g): the two-division downscaling of a 31-bit generator. */
export const uniformIntMinstd = (g: MinstdRand, a: number, b: number): number => {
  const urange = b - a;
  if (urange > URNGRANGE) throw new Error('uniformIntMinstd: the range exceeds the generator range');
  if (urange === URNGRANGE) return g.next() - 1 + a;
  const uerange = urange + 1;
  const scaling = Math.floor(URNGRANGE / uerange);
  const past = uerange * scaling;
  let ret: number;
  do ret = g.next() - 1;
  while (ret >= past);
  return Math.floor(ret / scaling) + a;
};

/** std::shuffle(list.begin(), list.end(), g). */
export const shuffleMinstd = (list: number[], g: MinstdRand): void => {
  const n = list.length;
  if (n === 0) return;
  const swap = (i: number, j: number): void => {
    const t = list[i];
    list[i] = list[j];
    list[j] = t;
  };
  if (Math.floor(URNGRANGE / n) >= n) {
    let i = 1;
    // an even length leaves an odd number of positions to swap: the first one alone
    if (n % 2 === 0) {
      swap(i, uniformIntMinstd(g, 0, 1));
      i++;
    }
    while (i !== n) {
      const swapRange = i + 1;
      // __gen_two_uniform_ints(swapRange, swapRange + 1, g)
      const x = uniformIntMinstd(g, 0, swapRange * (swapRange + 1) - 1);
      const p1 = Math.floor(x / (swapRange + 1));
      const p2 = x % (swapRange + 1);
      swap(i, p1);
      i++;
      swap(i, p2);
      i++;
    }
    return;
  }
  for (let i = 1; i < n; i++) swap(i, uniformIntMinstd(g, 0, i));
};
