import { describe, expect, it } from 'vitest';
import { MinstdRand, shuffleMinstd } from '../src/engine-gpl/react/minstd';

/*
 * The overstep shuffle of fix bond/react (src/engine-gpl/react/minstd.ts): std::minstd_rand and std::shuffle as
 * libstdc++ 11 computes them. The expected lists are the output of std::shuffle on 0..n-1 with g.seed(seed), compiled
 * with GCC 11.4 (the compiler of the native binary); the seeds continue across the lists as one generator.
 */

const EXPECTED: [number, number, number[]][] = [
  [30000, 2, [0, 1]],
  [30000, 5, [0, 4, 3, 2, 1]],
  [30000, 8, [3, 6, 5, 4, 1, 2, 0, 7]],
  [77, 2, [1, 0]],
  [77, 5, [2, 4, 0, 1, 3]],
  [77, 8, [5, 2, 3, 4, 7, 0, 1, 6]],
  [1000000, 2, [1, 0]],
  [1000000, 5, [1, 0, 4, 2, 3]],
  [1000000, 8, [3, 0, 2, 4, 1, 6, 7, 5]],
];

describe('libstdc++ std::minstd_rand and std::shuffle', () => {
  it('draws 48271 x mod (2^31 - 1) from the seed', () => {
    const g = new MinstdRand(30000);
    expect([g.next(), g.next(), g.next()]).toEqual([1448130000, 2090520150, 1241588120]);
  });

  it('shuffles as std::shuffle does', () => {
    const gens = new Map<number, MinstdRand>();
    for (const [seed, n, want] of EXPECTED) {
      if (!gens.has(seed)) gens.set(seed, new MinstdRand(seed));
      const list = Array.from({ length: n }, (_, i) => i);
      shuffleMinstd(list, gens.get(seed)!);
      expect(list, `seed ${seed} n ${n}`).toEqual(want);
    }
  });
});
