/*
 * erfc(x) for the real-space Ewald/PPPM sum, accurate to ~1e-12 absolute:
 * a table built once from the Taylor series of erf (x < 2, no cancellation
 * problem there) and the continued fraction for erfc (x >= 2, Abramowitz &
 * Stegun 7.1.14), read with cubic Hermite interpolation using the exact
 * derivative erfc'(x) = -2/sqrt(pi) exp(-x^2). Beyond the table erfc < 1e-17.
 */
import { erfcExact, libm } from '../libm';

const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** Direct evaluation (slow, for the table): defined with the libm table, which uses it outside a session. */
export { erfcExact };

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
  const d0 = -TWO_OVER_SQRTPI * libm.exp(-x0 * x0) * H;
  const d1 = -TWO_OVER_SQRTPI * libm.exp(-(x0 + H) * (x0 + H)) * H;
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
 * units", and the table is used for pairs "above" that cutoff. Measured with
 * native LAMMPS (black box, 2026-10-13): at rsq exactly 2 (atoms at (1,1,0)) the
 * energy, the force and the special-bond 1/r correction equal the table-0
 * polynomial values for pair_modify table 12, so the table is used only for
 * rsq > 2 (strict); the engine's TABLE_INNER_RSQ tests are strict for that reason.
 *
 * Measured with native LAMMPS (black box, 2 atoms, coul/long, gewald 0.3, real
 * units, 2026-10): the table nodes are the rsq values whose float32 mantissa is
 * truncated to M bits (M = N - nexpbits, see below) within each octave, i.e. with s in [2^e, 2^(e+1)) the
 * node spacing is 2^(e-M), M = N - nexpbits with nexpbits set by the cutoff (makeErfcTable; N = 12 and cut 12
 * give M = 9, spacing 2^(e-9); checked in octaves
 * e = 2, 3, 4, 5 and 16); the node values are the exact kernel
 * at the node, and the value between nodes is linear in rsq using the float32
 * rounding of rsq. Energy kernel: the table of erfc(g r)/r; agreement with the
 * native table-12 energies is 1e-14 absolute on the points tested. Force kernel:
 * (erfc + TWO_OVER_SQRTPI g r exp(-g^2 r^2))/r with the same node structure,
 * where TWO_OVER_SQRTPI is the exact 2/sqrt(pi), not EWALD_F. Measured with
 * native LAMMPS (black box, 2026-10-13): two atoms, charges +1 and -1, pair
 * coul/long 12.0, kspace ewald 1e-8, gewald 0.3, real units, separations
 * r = 1.41 to 11.3 (1035 points: random in octaves rsq = 2..128, rsq at the node
 * grid 2^e (1 + k/512), and sqrt(2) + {0, 1e-12, 1e-9, 1e-6, 1e-4}), force on
 * atom 2 with %.17g. The difference table 12 minus table 0 (the kspace part
 * cancels) equals the emulation's difference to 7e-14 absolute at worst (the
 * native output precision). The same difference with EWALD_F in the node values
 * misses by up to 1.7e-7; with float32-rounded node radii r the miss is 6e-6.
 * So the table nodes carry the exact constant, the node radius is the double
 * sqrt of the node rsq, and the interpolation is linear in float32(rsq). The
 * direct (table 0) branch keeps EWALD_F, which the same native difference
 * confirms.
 * The octave exponent is checked for e = 1..6 by the force scan above (and 2..5,
 * 16 for the node spacing); N other than 12 is not measured. The cut-26.5 case (M = 8) is checked by w117cf_coul_long_table12_cut26.
 */
export const TABLE_INNER_RSQ = 2.0;

export interface ErfcTable {
  /** rsq from which the table is used (pair_modify tabinner squared; below it the direct polynomial). */
  readonly inner: number;
  /** erfc(g r)/r at rsq (no charge or unit prefactor). */
  energy(rsq: number): number;
  /** (erfc(g r) + 2/sqrt(pi) g r exp(-g^2 r^2))/r; the force is qq * force(rsq) / rsq. */
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
 * linear interpolation between the two nodes around float32(rsq). The node is always found from
 * float32(rsq); the interpolation fraction is (float32(rsq) - node) / spacing, or with doubleRsq
 * (rsq - node) / spacing. As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp and
 * pair_buck_long_coul_long.cpp compute at b332f28, "double fraction = (rsq-rtable[k])*drtable[k]"),
 * lj/long/coul/long and buck/long/coul/long take the double rsq; lj/cut/coul/long and the other
 * styles take the float one ("fraction = ((double) rsq_lookup.f - rtable[itable]) * drtable[itable]").
 */
export const makeErfcTable = (N: number, g: number, cutsq: number, inner: number = TABLE_INNER_RSQ, doubleRsq = false): ErfcTable => {
  if (!(N > 3 && N <= 24)) throw new RangeError('pair_modify table N must be 4..24 for the emulation');
  // Mantissa bits per octave: native Pair::init_bitmap (src/pair.cpp at b332f28) takes nexpbits, the
  // exponent bits the table spans, from the outer cutoff: nlowermin is the power of two at or below
  // inner^2, required_range = outer^2 / 2^nlowermin, and nexpbits is the least value with
  // 2^(2^nexpbits) >= required_range; nmantbits = ntablebits - nexpbits. With cut_coul 12 that is
  // nexpbits 3 (9 mantissa bits, the spacing measured above); with cut_coul 26.5 it is 4 (8 bits),
  // measured with native LAMMPS (black box, w104cf_sds_coul_alone: the table-12 minus table-0 ecoul
  // differs from the 9-bit emulation by 1.2e-8).
  let nlowermin = 1;
  while (2 ** nlowermin > inner || 2 ** (nlowermin + 1) <= inner) {
    if (2 ** nlowermin <= inner) nlowermin++;
    else nlowermin--;
  }
  const required = cutsq / 2 ** nlowermin;
  let nexp = 0, available = 2;
  while (available < required) {
    nexp++;
    available = 2 ** (2 ** nexp);
  }
  const M = N - nexp;
  if (M < 3) throw new RangeError('pair_modify table N is too small for this cutoff');
  const perOctave = 1 << M;
  f32[0] = inner;
  const eMin = u32[0] >>> 23;
  f32[0] = Math.max(cutsq, inner);
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
    fTab[k] = (ec + TWO_OVER_SQRTPI * x * libm.exp(-x * x)) / r;
    cTab[k] = 1 / r;
  }
  const shift = 23 - M;
  const direct = (rsq: number, which: 0 | 1 | 2): number => {
    const r = Math.sqrt(rsq);
    if (which === 2) return 1 / r;
    const x = g * r, ex = libm.exp(-x * x), ec = erfcPoly(x, ex);
    return which === 0 ? ec / r : (ec + EWALD_F * x * ex) / r;
  };
  // per node: its rsq (node), the inverse of its spacing (an exact power of two, so multiplying by it is the same as
  // dividing by the spacing) and the difference to the next node of each table, computed once instead of per lookup
  const node = new Float64Array(n), invsp = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const e = eMin + Math.floor(k / perOctave), top = k % perOctave;
    node[k] = 2 ** (e - 127) * (1 + top / perOctave);
    invsp[k] = 1 / 2 ** (e - 127 - M);
  }
  const diffs = (tab: Float64Array): Float64Array => {
    const d = new Float64Array(n);
    for (let k = 0; k + 1 < n; k++) d[k] = tab[k + 1] - tab[k];
    return d;
  };
  const dE = diffs(eTab), dF = diffs(fTab), dC = diffs(cTab);
  const interp = (rsq: number, tab: Float64Array, dtab: Float64Array, which: 0 | 1 | 2): number => {
    if (rsq <= inner) return direct(rsq, which);
    f32[0] = rsq;
    const bits = u32[0];
    const k = ((bits >>> 23) - eMin) * perOctave + ((bits & 0x7fffff) >>> shift);
    if (k + 1 >= n) return direct(rsq, which);
    const t = ((doubleRsq ? rsq : f32[0]) - node[k]) * invsp[k];
    return tab[k] + dtab[k] * t;
  };
  return {
    inner,
    energy: (rsq) => interp(rsq, eTab, dE, 0),
    force: (rsq) => interp(rsq, fTab, dF, 1),
    coul: (rsq) => interp(rsq, cTab, dC, 2),
  };
};

/** One pair style's table, rebuilt when pair_modify table N, g_ewald or the Coulomb cutoff change. */
export class ErfcTableCache {
  private built = false;
  private keyN = 0;
  private keyG = 0;
  private keyCut = 0;
  private keyInner = 0;
  private keyDouble = false;
  private table: ErfcTable | null = null;

  /**
   * null for pair_modify table 0 (direct evaluation). The key is compared as numbers: this runs for every pair
   * in the Coulomb terms, and building a string key per call dominated the real-space loop. Two numbers have the
   * same string form exactly when they are equal (apart from NaN, which then rebuilds the same table).
   */
  get(N: number, g: number, cutsq: number, inner: number = TABLE_INNER_RSQ, doubleRsq = false): ErfcTable | null {
    if (N <= 0) return null;
    if (!this.built || N !== this.keyN || g !== this.keyG || cutsq !== this.keyCut || inner !== this.keyInner || doubleRsq !== this.keyDouble) {
      this.table = makeErfcTable(N, g, cutsq, inner, doubleRsq);
      this.built = true;
      this.keyN = N; this.keyG = g; this.keyCut = cutsq; this.keyInner = inner; this.keyDouble = doubleRsq;
    }
    return this.table;
  }
}
