/**
 * Seeded pseudo-random numbers for the engine: xoshiro128** (Blackman &
 * Vigna, 2018), state initialised from the seed with splitmix32. Pure 32-bit
 * integer arithmetic, so a seed gives the same stream in every browser.
 *
 * This is NOT LAMMPS's generator (that is RanPark below): with the same seed
 * it gives different, equally valid, numbers; only the statistics match.
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

/**
 * The Park-Miller minimal standard generator (S. K. Park and K. W. Miller,
 * Random number generators: good ones are hard to find, Commun. ACM 31,
 * 1192 (1988)): seed <- 16807 seed mod (2^31 - 1), uniform = seed / (2^31 - 1),
 * evaluated without overflow by Schrage's factorisation; gaussian() is the
 * polar (Marsaglia) method in its Numerical Recipes form, which returns
 * v2 * f and keeps v1 * f for the next call.
 *
 * LAMMPS names a "Park random # generator" whose "initial seed ... must be a
 * positive integer" (docs.lammps.org/Errors_messages.html). Measured with
 * native LAMMPS as a black box: velocity create loop all draws exactly this
 * stream (velocity ratios vx/vy agree to 10 digits for uniform and gaussian
 * distributions), so the engine uses it wherever that matters.
 */
export class RanPark {
  private seed: number;
  private saved: number | null = null;

  constructor(seed: number) {
    if (!Number.isInteger(seed) || seed <= 0 || seed >= PM_M) throw new Error('Invalid seed for Park random # generator');
    this.seed = seed;
  }

  uniform(): number {
    const k = Math.floor(this.seed / PM_Q);
    this.seed = PM_A * (this.seed - k * PM_Q) - PM_R * k;
    if (this.seed < 0) this.seed += PM_M;
    return this.seed / PM_M;
  }

  gaussian(): number {
    if (this.saved !== null) {
      const s = this.saved;
      this.saved = null;
      return s;
    }
    let v1: number, v2: number, rsq: number;
    do {
      v1 = 2 * this.uniform() - 1;
      v2 = 2 * this.uniform() - 1;
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1 || rsq === 0);
    const fac = Math.sqrt((-2 * Math.log(rsq)) / rsq);
    this.saved = v1 * fac;
    return v2 * fac;
  }

  /** Drop the kept second value of a polar pair, so the next gaussian() starts a new pair. */
  discardGaussian(): void {
    this.saved = null;
  }
}

const PM_A = 16807, PM_M = 2147483647, PM_Q = 127773, PM_R = 2836;

/**
 * Marsaglia-Zaman lagged-Fibonacci generator with carry, in the form published
 * by F. James (Comput. Phys. Commun. 60, 329 (1990)) for RANMAR: a 97-element
 * state, seeded from one integer through IJ = (seed-1)/30082, KL = seed-1-30082*IJ.
 * fix_langevin.html: "A Marsaglia random number generator is used." Measured
 * with native LAMMPS (black box): with seed 12345 the fix's force components
 * are (uniform() - 0.5) times the amplitude below, drawn in this stream, which
 * is why the engine uses it for fix langevin (Rng above is a different generator).
 */
export class RanMars {
  private readonly u = new Float64Array(98);
  private c = 362436 / 16777216;
  private readonly cd = 7654321 / 16777216;
  private readonly cm = 16777213 / 16777216;
  private ui = 97;
  private uj = 33;

  constructor(seed: number) {
    const ij = Math.floor((seed - 1) / 30082);
    const kl = seed - 1 - 30082 * ij;
    let i = (Math.floor(ij / 177) % 177) + 2;
    let j = (ij % 177) + 2;
    let k = (Math.floor(kl / 169) % 178) + 1;
    let l = kl % 169;
    for (let ii = 1; ii <= 97; ii++) {
      let x = 0;
      let t = 0.5;
      for (let jj = 1; jj <= 24; jj++) {
        const m = ((i * j) % 179) * k % 179;
        i = j; j = k; k = m;
        l = (53 * l + 1) % 169;
        if ((l * m) % 64 >= 32) x += t;
        t *= 0.5;
      }
      this.u[ii] = x;
    }
  }

  uniform(): number {
    let uni = this.u[this.ui] - this.u[this.uj];
    if (uni < 0) uni += 1;
    this.u[this.ui] = uni;
    if (--this.ui === 0) this.ui = 97;
    if (--this.uj === 0) this.uj = 97;
    this.c -= this.cd;
    if (this.c < 0) this.c += this.cm;
    uni -= this.c;
    if (uni < 0) uni += 1;
    return uni;
  }

  /**
   * The first accepted Marsaglia polar pair, [v2 f, v1 f] with f = sqrt(-2 ln s / s); rejected
   * pairs consume their draws. Measured with native LAMMPS (black box) by fix
   * wall/reflect/stochastic, whose tangential velocities are this pair.
   */
  polarPair(): [number, number] {
    for (;;) {
      const v1 = 2 * this.uniform() - 1;
      const v2 = 2 * this.uniform() - 1;
      const s = v1 * v1 + v2 * v2;
      if (s < 1 && s > 0) {
        const f = Math.sqrt((-2 * Math.log(s)) / s);
        return [v2 * f, v1 * f];
      }
    }
  }
}
