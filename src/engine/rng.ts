/**
 * Seeded pseudo-random numbers for the engine: xoshiro128** (Blackman &
 * Vigna, 2018), state initialised from the seed with splitmix32. Pure 32-bit
 * integer arithmetic, so a seed gives the same stream in every browser.
 *
 * This is NOT LAMMPS's generator, so the same seed gives different (equally
 * valid) velocities than LAMMPS would; only the statistics match.
 */
export class Rng {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;
  private spare: number | null = null;

  constructor(seed: number) {
    let z = seed | 0;
    const next = () => {
      z = (z + 0x9e3779b9) | 0;
      let t = z ^ (z >>> 16);
      t = Math.imul(t, 0x21f0aaad);
      t ^= t >>> 15;
      t = Math.imul(t, 0x735a2d97);
      return (t ^ (t >>> 15)) >>> 0;
    };
    this.s0 = next(); this.s1 = next(); this.s2 = next(); this.s3 = next();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  /** Uniform 32-bit unsigned integer. */
  nextU32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5), 7), 9) >>> 0;
    const t = this.s1 << 9;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1) with 53 random bits. */
  uniform(): number {
    const hi = this.nextU32() >>> 5;   // 27 bits
    const lo = this.nextU32() >>> 6;   // 26 bits
    return (hi * 67108864 + lo) / 9007199254740992;
  }

  /** Standard normal deviate (Marsaglia polar method). */
  gaussian(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u: number, v: number, q: number;
    do {
      u = 2 * this.uniform() - 1;
      v = 2 * this.uniform() - 1;
      q = u * u + v * v;
    } while (q >= 1 || q === 0);
    const m = Math.sqrt((-2 * Math.log(q)) / q);
    this.spare = v * m;
    return u * m;
  }
}

const rotl = (x: number, k: number): number => (x << k) | (x >>> (32 - k));
