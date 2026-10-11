/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/random_park.cpp and
 * src/random_park.h (class RanPark: uniform(), gaussian(), reset(int), reset(int, double *)).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The Park-Miller minimal standard generator, and the Jenkins one-at-a-time hash that seeds it from a user seed
 * and an atom's coordinates. Every arithmetic step is exact in a JavaScript number (all values stay below 2^53),
 * and the hash keeps 32 unsigned bits, so the streams are the same in every browser.
 */

const IA = 16807;
const IM = 2147483647;
const AM = 1.0 / IM;
const IQ = 127773;
const IR = 2836;

/** Bytes of the seed and the coordinates are read as signed chars, little-endian (x86-64 layout). */
const HASH_BYTES = 4 + 3 * 8;

export class RanPark {
  private seed: number;
  private save = 0;
  private second = 0;

  /** RanPark::RanPark(lmp, seed_init): "Invalid seed for Park random # generator" for a seed <= 0. */
  constructor(seedInit: number) {
    if (seedInit <= 0) throw new Error('Invalid seed for Park random # generator');
    this.seed = seedInit;
  }

  /** RanPark::uniform(): one number in (0, 1). */
  uniform(): number {
    const k = Math.trunc(this.seed / IQ);
    this.seed = IA * (this.seed - k * IQ) - IR * k;
    if (this.seed < 0) this.seed += IM;
    return AM * this.seed;
  }

  /** RanPark::gaussian(): the first value of a polar pair; the second is kept for the next call. */
  gaussian(): number {
    let first: number;
    if (!this.save) {
      let v1: number, v2: number, rsq: number;
      do {
        v1 = 2.0 * this.uniform() - 1.0;
        v2 = 2.0 * this.uniform() - 1.0;
        rsq = v1 * v1 + v2 * v2;
      } while (rsq >= 1.0 || rsq === 0.0);
      const fac = Math.sqrt((-2.0 * Math.log(rsq)) / rsq);
      this.second = v1 * fac;
      first = v2 * fac;
      this.save = 1;
    } else {
      first = this.second;
      this.save = 0;
    }
    return first;
  }

  /** RanPark::reset(int seed_init): a new seed, and the saved gaussian is dropped. */
  reset(seedInit: number): void {
    if (seedInit <= 0) throw new Error('Invalid seed for Park random # generator');
    this.seed = seedInit;
    this.save = 0;
  }

  /**
   * RanPark::reset(int ibase, double *coord): the seed of one atom from the user seed ibase and its coordinates
   * x, y, z (the stored doubles, bit for bit). The hash is Jenkins' one-at-a-time hash over the 4 bytes of ibase
   * and then the 24 bytes of the three doubles, each byte a signed char. Then the seed keeps 27 bits, 0 becomes
   * 1, and the generator is warmed up with 5 uniform() calls.
   */
  resetAtom(ibase: number, x: number, y: number, z: number): void {
    const view = new DataView(new ArrayBuffer(HASH_BYTES));
    view.setInt32(0, ibase | 0, true);
    view.setFloat64(4, x, true);
    view.setFloat64(12, y, true);
    view.setFloat64(20, z, true);
    let hash = 0;
    for (let i = 0; i < HASH_BYTES; i++) {
      hash = (hash + view.getInt8(i)) >>> 0;
      hash = (hash + (hash << 10)) >>> 0;
      hash = (hash ^ (hash >>> 6)) >>> 0;
    }
    hash = (hash + (hash << 3)) >>> 0;
    hash = (hash ^ (hash >>> 11)) >>> 0;
    hash = (hash + (hash << 15)) >>> 0;
    // "keep 31 bits of unsigned int as new seed" (the source masks with 0x7ffffff, which keeps 27 bits)
    this.seed = hash & 0x7ffffff;
    if (!this.seed) this.seed = 1;
    for (let i = 0; i < 5; i++) this.uniform();
    this.save = 0;
  }
}
