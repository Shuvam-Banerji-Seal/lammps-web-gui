/*
 * erfc(x) for the real-space Ewald/PPPM sum, accurate to ~1e-12 absolute:
 * a table built once from the Taylor series of erf (x < 2, no cancellation
 * problem there) and the continued fraction for erfc (x >= 2, Abramowitz &
 * Stegun 7.1.14), read with cubic Hermite interpolation using the exact
 * derivative erfc'(x) = -2/sqrt(pi) exp(-x^2). Beyond the table erfc < 1e-17.
 */

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Direct evaluation (slow, for the table). */
export const erfcExact = (x: number): number => {
  if (x < 0) return 2 - erfcExact(-x);
  if (x < 2) {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    let term = x, sum = x;
    const x2 = x * x;
    for (let n = 1; n < 200; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
    }
    return 1 - TWO_OVER_SQRTPI * sum;
  }
  // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + (1/2)/(x + 1/(x + (3/2)/(x + 2/(x + ...)))))
  let f = x;
  for (let k = 400; k >= 1; k--) f = x + (k / 2) / f;
  return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
};

const XMAX = 6.25;
const N = 1600;            // h = 1/256
const H = XMAX / N;
const table = new Float64Array(N + 2);
for (let k = 0; k <= N + 1; k++) table[k] = erfcExact(k * H);

/** erfc(x) for x >= 0, given ex = exp(-x^2) (callers need it for the force anyway). */
export const erfcFast = (x: number, ex: number): number => {
  if (x >= XMAX) return 0;
  const u = x / H;
  const k = u | 0;
  const t = u - k;
  const x0 = k * H;
  const f0 = table[k], f1 = table[k + 1];
  const d0 = -TWO_OVER_SQRTPI * Math.exp(-x0 * x0) * H;
  const d1 = -TWO_OVER_SQRTPI * Math.exp(-(x0 + H) * (x0 + H)) * H;
  void ex;
  const t2 = t * t, t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * f0 + (t3 - 2 * t2 + t) * d0 + (-2 * t3 + 3 * t2) * f1 + (t3 - t2) * d1;
};

/**
 * 2/sqrt(pi) as used in the Ewald real-space force term C q_i q_j / r
 * (erfc(g r) + EWALD_F g r exp(-g^2 r^2)). Native LAMMPS uses it rounded to 8
 * digits: with the exact value the pair virial of oracle case ewald_nacl
 * (pair_modify table 0) is off by 6e-9 relative on every diagonal component
 * while energies agree to 1e-15; with 1.12837917 every pressure component
 * agrees to 1e-15 (measured 2026-10-07).
 */
export const EWALD_F = 1.12837917;

/**
 * pair_modify table 0 — pair_modify.html: "For N = 0, forces and energies are
 * computed directly, using a polynomial fit for the needed erfc() function
 * evaluation": the Abramowitz & Stegun 7.1.26 rational fit,
 * erfc(x) = t (a1 + t (a2 + t (a3 + t (a4 + t a5)))) exp(-x^2), t = 1/(1 + p x)
 * (absolute error <= 1.5e-7).
 */
export const erfcPoly = (x: number, ex: number): number => {
  const t = 1 / (1 + 0.3275911 * x);
  return t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * ex;
};

/**
 * pair_modify table N emulation (Coulomb real-space kernels).
 *
 * pair_modify.html: "If N is non-zero, a table of length 2\^N is pre-computed
 * for forces and energies ... The table is indexed via a bit-mapping technique
 * (Wolff) and a linear interpolation is performed between adjacent table
 * values." and "The default value of 12 (table of length 4096) gives
 * approximately the same accuracy as the no-table (N = 0) option. For N = 0,
 * forces and energies are computed directly, using a polynomial fit for the
 * needed erfc() function evaluation".
 * pair_modify.html (tabinner): "The default cutoff value is sqrt(2.0) distance
 * units", i.e. table lookup is used for rsq >= 2 and the polynomial below.
 *
 * Measured with native LAMMPS (black box, 2 atoms, coul/long, gewald 0.3, real
 * units, 2026-10): the table nodes are the rsq values whose float32 mantissa is
 * truncated to N-3 bits within each octave, i.e. with s in [2^e, 2^(e+1)) the
 * node spacing is 2^(e-(N-3)) (N = 12: spacing 2^(e-9); checked in octaves
 * e = 2, 3, 4, 5 and 16); the node values are the exact kernel
 * at the node, and the value between nodes is linear in rsq using the float32
 * rounding of rsq. Energy kernel: the table of erfc(g r)/r; agreement with the
 * native table-12 energies is 1e-14 absolute on the points tested. Force kernel:
 * (erfc + EWALD_F g r exp(-g^2 r^2))/r with the same node structure; the native
 * force difference (table 12 minus table 0) agrees with this to ~3e-6 absolute
 * (a few per cent of the table error itself, ~1e-8 of the total force).
 * The octave exponent is not checked beyond e = 5 (Wolff bit-mapping with 3
 * exponent bits); N other than 12 is not measured.
 */
export const TABLE_INNER_RSQ = 2.0;

export interface ErfcTable {
  /** erfc(g r)/r at rsq (no charge or unit prefactor). */
  energy(rsq: number): number;
  /** (erfc(g r) + EWALD_F g r exp(-g^2 r^2))/r; the force is qq * force(rsq) / rsq. */
  force(rsq: number): number;
  /**
   * 1/r, for the special-bond correction -(1 - f_coul) C q_i q_j / r. Measured with native LAMMPS
   * (black box, a bonded pair with coul weight 0 at r = 2.3, 3.1 and 4.7 A, g 0.3, table 12): the
   * correction uses this tabulated 1/r too (energies agree to 1e-14; the exact 1/r is 2e-5 off).
   */
  coul(rsq: number): number;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/**
 * The table for pair_modify table N (N >= 4) and g_ewald g, covering rsq up to cutsq. Node k of
 * octave e (float32 exponent bits) sits at rsq = 2^(e-127) (1 + k / 2^(N-3)); a value is the
 * linear interpolation between the two nodes around float32(rsq).
 */
export const makeErfcTable = (N: number, g: number, cutsq: number): ErfcTable => {
  if (!(N > 3 && N <= 24)) throw new RangeError('pair_modify table N must be 4..24 for the emulation');
  const M = N - 3;
  const perOctave = 1 << M;
  f32[0] = TABLE_INNER_RSQ;
  const eMin = u32[0] >>> 23;
  f32[0] = Math.max(cutsq, TABLE_INNER_RSQ);
  const eMax = (u32[0] >>> 23) + 1;
  const n = (eMax - eMin + 1) * perOctave + 1;
  const eTab = new Float64Array(n), fTab = new Float64Array(n), cTab = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const e = eMin + Math.floor(k / perOctave), top = k % perOctave;
    const s = 2 ** (e - 127) * (1 + top / perOctave);
    const r = Math.sqrt(s);
    const x = g * r;
    const ec = x >= 6.25 ? 0 : erfcExact(x);
    eTab[k] = ec / r;
    fTab[k] = (ec + EWALD_F * x * Math.exp(-x * x)) / r;
    cTab[k] = 1 / r;
  }
  const shift = 23 - M;
  const direct = (rsq: number, which: 0 | 1 | 2): number => {
    const r = Math.sqrt(rsq);
    if (which === 2) return 1 / r;
    const x = g * r, ex = Math.exp(-x * x), ec = erfcPoly(x, ex);
    return which === 0 ? ec / r : (ec + EWALD_F * x * ex) / r;
  };
  const interp = (rsq: number, tab: Float64Array, which: 0 | 1 | 2): number => {
    if (rsq < TABLE_INNER_RSQ) return direct(rsq, which);
    f32[0] = rsq;
    const bits = u32[0];
    const e = bits >>> 23;
    const top = (bits & 0x7fffff) >>> shift;
    const k = (e - eMin) * perOctave + top;
    if (k + 1 >= n) return direct(rsq, which);
    const sp = 2 ** (e - 127 - M);
    const s0 = 2 ** (e - 127) * (1 + top / perOctave);
    const t = (f32[0] - s0) / sp;
    return tab[k] + (tab[k + 1] - tab[k]) * t;
  };
  return {
    energy: (rsq) => interp(rsq, eTab, 0),
    force: (rsq) => interp(rsq, fTab, 1),
    coul: (rsq) => interp(rsq, cTab, 2),
  };
};

/** One pair style's table, rebuilt when pair_modify table N, g_ewald or the Coulomb cutoff change. */
export class ErfcTableCache {
  private key = '';
  private table: ErfcTable | null = null;

  /** null for pair_modify table 0 (direct evaluation). */
  get(N: number, g: number, cutsq: number): ErfcTable | null {
    if (N <= 0) return null;
    const key = `${N} ${g} ${cutsq}`;
    if (key !== this.key) { this.table = makeErfcTable(N, g, cutsq); this.key = key; }
    return this.table;
  }
}
