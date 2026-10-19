/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the random stream of fix lambda_thermostat/apip
 * (src/APIP/fix_lambda_thermostat_apip.cpp at b332f28: std::mt19937 seeded with the seed keyword, and std::shuffle of
 * each local neighbour list). The native binary is built with GCC 11, so the sequence is that of libstdc++ 11:
 * std::mt19937 (bits/random.h, random.tcc: the 32-bit Mersenne Twister of the C++ standard, seeded with the standard
 * linear recurrence), std::shuffle (bits/stl_algo.h: the variant that draws two swap positions per call through
 * __gen_two_uniform_ints when the generator range covers the square of the list length) and
 * std::uniform_int_distribution<unsigned long> (bits/uniform_int_dist.h: for a 32-bit generator, Lemire's nearly
 * divisionless downscaling with a 64-bit product, _S_nd). libstdc++ is GPL-3.0-or-later with the GCC Runtime Library
 * Exception; the algorithms (not the code) are the reference, and the code is written from them. This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 */

const N = 624;
const M = 397;

/** std::mt19937 (result in [0, 2^32 - 1]). */
export class Mt19937 {
  private readonly mt = new Uint32Array(N);
  private mti = N;

  constructor(seed: number) {
    const mt = this.mt;
    mt[0] = seed >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      // 1812433253 * prev + i modulo 2^32
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = N;
  }

  private twist(): void {
    const mt = this.mt;
    for (let k = 0; k < N; k++) {
      const y = (mt[k] & 0x80000000) | (mt[(k + 1) % N] & 0x7fffffff);
      mt[k] = (mt[(k + M) % N] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0)) >>> 0;
    }
    this.mti = 0;
  }

  /** operator(): the next 32-bit value. */
  next(): number {
    if (this.mti >= N) this.twist();
    let y = this.mt[this.mti++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

/**
 * uniform_int_distribution<unsigned long>{0, b}(g) for b < 2^32 - 1 with a 32-bit generator: _S_nd<uint64>(g, b + 1).
 */
export const uniformInt = (g: Mt19937, b: number): number => {
  const range = BigInt(b + 1);
  const mask32 = 0xffffffffn;
  let product = BigInt(g.next()) * range;
  let low = product & mask32;
  if (low < range) {
    // -range % range in 32-bit unsigned arithmetic
    const threshold = ((mask32 + 1n - range) & mask32) % range;
    while (low < threshold) {
      product = BigInt(g.next()) * range;
      low = product & mask32;
    }
  }
  return Number(product >> 32n);
};

/** std::shuffle(a, a + n, g) for a list of n < 2^16 entries (the two-positions-per-draw path). */
export const stdShuffle = (a: Int32Array, n: number, g: Mt19937): void => {
  if (n === 0) return;
  const urngrange = 0xffffffff;
  if (Math.floor(urngrange / n) < n) throw new Error('stdShuffle: list too long for the paired path');
  const swap = (i: number, j: number): void => {
    const t = a[i];
    a[i] = a[j];
    a[j] = t;
  };
  let i = 1;
  if (n % 2 === 0) {
    swap(i, uniformInt(g, 1));
    i++;
  }
  while (i !== n) {
    const swapRange = i + 1;
    // __gen_two_uniform_ints(swapRange, swapRange + 1, g)
    const x = uniformInt(g, swapRange * (swapRange + 1) - 1);
    const p1 = Math.floor(x / (swapRange + 1));
    const p2 = x % (swapRange + 1);
    swap(i, p1);
    i++;
    swap(i, p2);
    i++;
  }
};
