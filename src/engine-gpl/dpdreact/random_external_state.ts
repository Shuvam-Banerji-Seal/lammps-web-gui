/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-REACT/random_external_state.h (the
 * random_external_state namespace: es_urand, es_urand64, es_rand, es_drand, es_normal, es_init and
 * es_genNextParallelState; derived from the Kokkos xorshift64* generator). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The 64-bit state is kept as two unsigned 32-bit halves (hi, lo) so that the generator runs without BigInt: the
 * arithmetic is the same modulo 2^64. A state is a plain pair of numbers owned by the caller (the native code keeps
 * no internal state in the class).
 */
import { log as libmLog } from '../libm/log';

/** A 64-bit generator state (hi and lo 32-bit halves, unsigned). */
export interface ExtState { hi: number; lo: number }

/** 2685821657736338717 = 0x2545F4914F6CDD1D, the xorshift64* multiplier. */
const M_HI = 0x2545f491;
const M_LO = 0x4f6cdd1d;

/** Full 64-bit product of two unsigned 32-bit numbers, as [high 32 bits, low 32 bits]. */
const mulFull = (a: number, b: number): [number, number] => {
  const al = a & 0xffff, ah = a >>> 16;
  const bl = b & 0xffff, bh = b >>> 16;
  const p0 = al * bl;
  const p1 = al * bh + ah * bl;
  const p2 = ah * bh;
  const t = p0 + (p1 % 65536) * 65536;
  const lo = t % 4294967296;
  const carry = Math.floor(t / 4294967296);
  const hi = p2 + Math.floor(p1 / 65536) + carry;
  return [hi % 4294967296, lo];
};

/** One xorshift step, then state * M (mod 2^64) written to out; the state itself is updated in place. */
const step = (s: ExtState): { hi: number; lo: number } => {
  // state ^= state >> 12
  let lo = (s.lo ^ (((s.lo >>> 12) | (s.hi << 20)) >>> 0)) >>> 0;
  let hi = (s.hi ^ (s.hi >>> 12)) >>> 0;
  // state ^= state << 25
  hi = (hi ^ (((hi << 25) | (lo >>> 7)) >>> 0)) >>> 0;
  lo = (lo ^ ((lo << 25) >>> 0)) >>> 0;
  // state ^= state >> 27
  lo = (lo ^ (((lo >>> 27) | (hi << 5)) >>> 0)) >>> 0;
  hi = (hi ^ (hi >>> 27)) >>> 0;
  s.hi = hi;
  s.lo = lo;
  // tmp = state * 2685821657736338717 (mod 2^64)
  const [ph, pl] = mulFull(lo, M_LO);
  const outHi = (ph + Math.imul(hi, M_LO) + Math.imul(lo, M_HI)) >>> 0;
  return { hi: outHi, lo: pl };
};

/** es_urand: the high 32 bits of the product after a 16-bit right shift (uint32). */
export const esUrand = (s: ExtState): number => {
  const p = step(s);
  return ((p.lo >>> 16) | (p.hi << 16)) >>> 0;
};

/** es_urand64: product minus one (mod 2^64), as a double value of the 64-bit integer. */
const urand64AsDouble = (s: ExtState): number => {
  const p = step(s);
  let lo = p.lo - 1, hi = p.hi;
  if (lo < 0) { lo += 4294967296; hi = (hi - 1 + 4294967296) % 4294967296; }
  return hi * 4294967296 + lo;
};

/**
 * es_drand: urand64 / MAX_URAND64. MAX_URAND64 = 2^64 - 2 converts to the double 2^64 (its distance from 2^64 is
 * below the double spacing there), so the division is by 2^64.
 */
export const esDrand = (s: ExtState): number => urand64AsDouble(s) / 18446744073709551616;

/** es_rand: the uint32 urand halved (an int in [0, 2^31)). */
export const esRand = (s: ExtState): number => Math.floor(esUrand(s) / 2);

/** es_normal: a Gaussian deviate by the polar rejection method; one draw of the state per uniform. */
export const esNormal = (s: ExtState): number => {
  let U: number, S: number;
  do {
    U = 2.0 * esDrand(s) - 1.0;
    const V = 2.0 * esDrand(s) - 1.0;
    S = U * U + V * V;
    if (S < 1.0 && S !== 0.0) return U * Math.sqrt(-2.0 * libmLog(S) / S);
  } while (true);
};

/** es_init: seeds the serial state, then discards 17 values (0 seeds with 1318319). */
export const esInit = (seed: number): ExtState => {
  const sd = seed === 0 ? 1318319 : seed;
  const s: ExtState = { hi: Math.floor(sd / 4294967296) >>> 0, lo: (sd % 4294967296) >>> 0 };
  for (let i = 0; i < 17; i++) esRand(s);
  return s;
};

/** es_genNextParallelState: four 16-bit pieces from four esRand values make a new state. */
export const esGenNextParallelState = (serial: ExtState): ExtState => {
  const n1 = esRand(serial), n2 = esRand(serial), n3 = esRand(serial), n4 = esRand(serial);
  return {
    lo: ((n1 & 0xffff) | ((n2 & 0xffff) << 16)) >>> 0,
    hi: ((n3 & 0xffff) | ((n4 & 0xffff) << 16)) >>> 0,
  };
};
