/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp (PairSMTBQ::gammas, css, coeffs, binm, caintgs, cbintgs) and
 * src/SMTBQ/pair_smtbq.h, at commit b332f28; the factorial table is src/math_special.cpp (MathSpecial::factorial,
 * nfac_table) and powint is MathSpecial::powint (math_special.h). Authors credited in the C++ file headers:
 * Julian Gale (Imperial College, 1997, gammas and css) and the SMTBQ authors (Nicolas Salles, Olivier Politano,
 * Robert Tetot, E. Maras). Copyright (2015) Universite de Bourgogne (SMTBQ) and Sandia Corporation (LAMMPS).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Slater-orbital two-centre overlap integrals of the second-moment tight-binding charge model. Arrays use the
 * 1-based indexing of the C++ source (a[1] .. a[n]); fct(n+1) holds n! (set by css, read by coeffs and binm).
 */

/** nfac_table of math_special.cpp, entries 0..31 (n! for n = index); the literals are copied as the C++ source has them. */
const NFAC: readonly number[] = [
  1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200,
  1307674368000, 20922789888000, 355687428096000, 6.402373705728e+15, 1.21645100408832e+17,
  2.43290200817664e+18, 5.10909421717094e+19, 1.12400072777761e+21, 2.5852016738885e+22,
  6.20448401733239e+23, 1.5511210043331e+25, 4.03291461126606e+26, 1.08888694504184e+28,
  3.04888344611714e+29, 8.8417619937397e+30, 2.65252859812191e+32, 8.22283865417792e+33,
];

/** MathSpecial::factorial for the orders the SMTBQ integrals use (n <= 31). */
export const factorial = (n: number): number => {
  if (n < 0 || n >= NFAC.length) throw new Error(`smtbq factorial order ${n} outside the table`);
  return NFAC[n];
};

/** MathSpecial::powint: x^n by repeated squaring (n may be negative). */
export const powint = (x: number, n: number): number => {
  if (n === 0) return 1.0;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

/** Outputs of gammas (the derivative terms the C++ routine also returns; tabqeq reads gam and dgam only). */
export interface GammaOut {
  gam: number; dgam: number; dza: number; dzb: number; d2zaa: number; d2zab: number; d2zbb: number;
  d2zra: number; d2zrb: number; d2gamr2: number;
}

/** Outputs of css. */
interface CssOut {
  s: number; deriv: number; dzeta1: number; dzeta2: number; d2zeta11: number; d2zeta12: number;
  d2zeta22: number; d2zeta1r: number; d2zeta2r: number; deriv2: number;
}

const newGamma = (): GammaOut => ({
  gam: 0, dgam: 0, dza: 0, dzb: 0, d2zaa: 0, d2zab: 0, d2zbb: 0, d2zra: 0, d2zrb: 0, d2gamr2: 0,
});

/** The Slater integral routines; fct is the factorial table (fct[n+1] = n!) that css fills. */
export class SlaterIntegrals {
  /** fct[1..30] = (0..29)!; css sets it (as the C++ fct common block) and it is constant, so it is filled once here. */
  readonly fct = (() => {
    const f = new Float64Array(31);
    for (let i = 1; i <= 30; i++) f[i] = factorial(i - 1);
    return f;
  })();
  private readonly a = new Float64Array(32);
  private readonly b = new Float64Array(32);
  private readonly da1 = new Float64Array(32);
  private readonly da2 = new Float64Array(32);
  private readonly db1 = new Float64Array(32);
  private readonly db2 = new Float64Array(32);
  private readonly d2a11 = new Float64Array(32);
  private readonly d2a12 = new Float64Array(32);
  private readonly d2a22 = new Float64Array(32);
  private readonly dar = new Float64Array(32);
  private readonly d2b11 = new Float64Array(32);
  private readonly d2b12 = new Float64Array(32);
  private readonly d2b22 = new Float64Array(32);
  private readonly dbr = new Float64Array(32);
  private readonly d2a1r = new Float64Array(32);
  private readonly d2a2r = new Float64Array(32);
  private readonly d2b1r = new Float64Array(32);
  private readonly d2b2r = new Float64Array(32);
  private readonly d2ar2 = new Float64Array(32);
  private readonly d2br2 = new Float64Array(32);
  private readonly cssOut: CssOut = {
    s: 0, deriv: 0, dzeta1: 0, dzeta2: 0, d2zeta11: 0, d2zeta12: 0, d2zeta22: 0, d2zeta1r: 0, d2zeta2r: 0,
    deriv2: 0,
  };
  private readonly out: GammaOut = newGamma();

  /**
   * gammas (pair_smtbq.cpp): the integral over two s orbitals of Slater functions, with its derivatives. Julian
   * Gale's routine. Returns the shared output object (valid until the next call).
   */
  gammas(na: number, nb: number, za: number, zb: number, r: number): GammaOut {
    const o = this.out;
    o.gam = 0; o.dgam = 0; o.dza = 0; o.dzb = 0; o.d2zaa = 0; o.d2zab = 0; o.d2zbb = 0;
    o.d2zra = 0; o.d2zrb = 0; o.d2gamr2 = 0;
    // This routine only handles two centre integrals
    if (r < 1.0e-10) return o;

    const z2ra = 2.0 * za * r;
    const z2rb = 2.0 * zb * r;
    const na2 = 2 * na;
    const nb2 = 2 * nb;
    const halfr = 0.5 * r;
    const cs = this.cssOut;
    let d2rtrm = powint(halfr, na2 - 2);
    let drtrm = d2rtrm * halfr;
    let rtrm = drtrm * halfr;

    // First term
    this.css(na2 - 1, 0, z2ra, 0.0, r);
    o.gam = rtrm * cs.s;
    o.dgam = rtrm * cs.deriv + na * drtrm * cs.s;
    o.dza = rtrm * cs.dzeta1;
    o.d2zaa = rtrm * cs.d2zeta11;
    o.d2zra = rtrm * cs.d2zeta1r + na * drtrm * cs.dzeta1;
    o.d2gamr2 = o.d2gamr2 + 0.5 * (na * (na2 - 1)) * d2rtrm * cs.s + 2.0 * na * drtrm * cs.deriv + rtrm * cs.deriv2;

    // Sum over 2*nb
    rtrm = drtrm;
    drtrm = d2rtrm;
    let ztrm = 0.5 / (zb * nb2);
    for (let i = nb2; i >= 1; i--) {
      rtrm = rtrm * halfr;
      drtrm = drtrm * halfr;
      ztrm = ztrm * 2.0 * zb;
      const ctrm = ztrm / factorial(nb2 - i);

      this.css(na2 - 1, nb2 - i, z2ra, z2rb, r);

      const trm1 = i * ctrm;
      const trm2 = trm1 * rtrm;
      o.gam = o.gam - trm2 * cs.s;
      const trm3 = trm1 * (na2 + nb2 - i) * drtrm;
      o.dgam = o.dgam - trm2 * cs.deriv - 0.5 * trm3 * cs.s;
      o.d2gamr2 = o.d2gamr2 - trm2 * cs.deriv2 - trm3 * cs.deriv - 0.5 * trm3 * (na2 + nb2 - i - 1) * cs.s / r;
      o.dza = o.dza - trm2 * cs.dzeta1;
      o.dzb = o.dzb - (trm2 / zb) * ((nb2 - i) * cs.s + zb * cs.dzeta2);
      o.d2zaa = o.d2zaa - trm2 * cs.d2zeta11;
      o.d2zab = o.d2zab - (trm2 / zb) * ((nb2 - i) * cs.dzeta1 + zb * cs.d2zeta12);
      o.d2zbb = o.d2zbb - (trm2 / zb) * (2.0 * (nb2 - i) * cs.dzeta2 + zb * cs.d2zeta22 +
                                         ((nb2 - i - 1) * (nb2 - i)) * cs.s / zb);
      o.d2zra = o.d2zra - trm2 * cs.d2zeta1r - 0.5 * trm3 * cs.dzeta1;
      o.d2zrb = o.d2zrb - (trm2 / zb) * ((nb2 - i) * cs.deriv + zb * cs.d2zeta2r) -
        0.5 * (trm3 / zb) * ((nb2 - i) * cs.s + zb * cs.dzeta2);
    }

    // Multiply by coefficients
    const trm3 = powint(2.0 * za, na2 + 1) / factorial(na2);
    o.gam = o.gam * trm3;
    o.dgam = o.dgam * trm3;
    const rfct1 = (na2 + 1) / za;
    const rgam1 = rfct1 * o.gam;
    o.dza = o.dza * trm3;
    const rdza1 = 2.0 * rfct1 * o.dza;
    o.dza = o.dza + rgam1;
    o.dzb = o.dzb * trm3;
    const rgam2 = rgam1 * na2 / za;
    o.d2zaa = o.d2zaa * trm3 + rgam2 + rdza1;
    o.d2zab = o.d2zab * trm3 + rfct1 * o.dzb;
    o.d2zbb = o.d2zbb * trm3;
    o.d2zra = o.d2zra * trm3 + rfct1 * o.dgam;
    o.d2zrb = o.d2zrb * trm3;
    o.d2gamr2 = o.d2gamr2 * trm3;
    return o;
  }

  /**
   * The values gam and dgam of gammas, without the second-derivative outputs. The table builder needs only these,
   * and each is computed by the same operations, in the same order, as in gammas (the derivative sums it skips
   * do not feed gam or dgam). Returns the shared output object.
   */
  gammaValues(na: number, nb: number, za: number, zb: number, r: number): GammaOut {
    const o = this.out;
    o.gam = 0; o.dgam = 0;
    if (r < 1.0e-10) return o;
    const z2ra = 2.0 * za * r;
    const z2rb = 2.0 * zb * r;
    const na2 = 2 * na;
    const nb2 = 2 * nb;
    const halfr = 0.5 * r;
    const cs = this.cssOut;
    const d2rtrm = powint(halfr, na2 - 2);
    let drtrm = d2rtrm * halfr;
    let rtrm = drtrm * halfr;
    this.cssValues(na2 - 1, 0, z2ra, 0.0, r);
    o.gam = rtrm * cs.s;
    o.dgam = rtrm * cs.deriv + na * drtrm * cs.s;
    rtrm = drtrm;
    drtrm = d2rtrm;
    let ztrm = 0.5 / (zb * nb2);
    for (let i = nb2; i >= 1; i--) {
      rtrm = rtrm * halfr;
      drtrm = drtrm * halfr;
      ztrm = ztrm * 2.0 * zb;
      const ctrm = ztrm / factorial(nb2 - i);
      this.cssValues(na2 - 1, nb2 - i, z2ra, z2rb, r);
      const trm1 = i * ctrm;
      const trm2 = trm1 * rtrm;
      o.gam = o.gam - trm2 * cs.s;
      const trm3 = trm1 * (na2 + nb2 - i) * drtrm;
      o.dgam = o.dgam - trm2 * cs.deriv - 0.5 * trm3 * cs.s;
    }
    const trm3 = powint(2.0 * za, na2 + 1) / factorial(na2);
    o.gam = o.gam * trm3;
    o.dgam = o.dgam * trm3;
    return o;
  }

  /** css without the derivatives with respect to the zetas and the second derivatives (s and deriv only). */
  private cssValues(nn1: number, nn2: number, alpha: number, beta: number, r: number): void {
    const o = this.cssOut;
    let n1 = nn1;
    let n2 = nn2;
    let p = (alpha + beta) * 0.5;
    let pt = (alpha - beta) * 0.5;
    const zeta1 = alpha / r;
    const zeta2 = beta / r;
    const sumzeta = zeta1 + zeta2;
    const difzeta = zeta1 - zeta2;
    const dpdz1 = r;
    const dpdz2 = r;
    let dptdz1 = r;
    let dptdz2 = -r;
    const dpdr = 0.5 * sumzeta;
    let dptdr = 0.5 * difzeta;
    o.deriv = 0.0;
    if (n2 < n1) {
      const k = n1;
      n1 = n2;
      n2 = k;
      pt = -pt;
      dptdr = -dptdr;
      dptdz1 = -dptdz1;
      dptdz2 = -dptdz2;
    }
    if (p > 86.0 || pt > 86.0) {
      o.s = 0.0;
      return;
    }
    const a = this.a, b = this.b;
    this.caintgs(p, n1 + n2 + 3, a);
    this.cbintgs(pt, n1 + n2 + 3, b);
    const ulim = n1 + n2 + 1;
    for (let i = 1; i <= ulim; i++) {
      this.da1[i] = -a[i + 1] * dpdz1;
      this.da2[i] = -a[i + 1] * dpdz2;
      this.db1[i] = -b[i + 1] * dptdz1;
      this.db2[i] = -b[i + 1] * dptdz2;
      this.dar[i] = -a[i + 1] * dpdr;
      this.dbr[i] = -b[i + 1] * dptdr;
    }
    let x = 0.0;
    for (let i1 = 1; i1 <= ulim; i1++) {
      const nni1 = n1 + n2 - i1 + 2;
      const coff = this.coeffs(n1, n2, i1 - 1);
      o.deriv = o.deriv + coff * (this.dar[i1] * b[nni1] + a[i1] * this.dbr[nni1]);
      x = x + coff * a[i1] * b[nni1];
    }
    o.s = x * 0.5;
    o.deriv = 0.5 * o.deriv;
  }

  /**
   * css (pair_smtbq.cpp): the modified overlap integral for Slater orbitals with derivatives (S orbitals only).
   * Fills this.cssOut. Julian Gale, Imperial College, December 1997.
   */
  private css(nn1: number, nn2: number, alpha: number, beta: number, r: number): void {
    const o = this.cssOut;
    o.dzeta1 = 0.0; o.dzeta2 = 0.0; o.d2zeta11 = 0.0; o.d2zeta12 = 0.0; o.d2zeta22 = 0.0;
    o.d2zeta1r = 0.0; o.d2zeta2r = 0.0; o.deriv = 0.0; o.deriv2 = 0.0;
    let n1 = nn1;
    let n2 = nn2;
    let p = (alpha + beta) * 0.5;
    let pt = (alpha - beta) * 0.5;
    let x = 0.0;
    const zeta1 = alpha / r;
    const zeta2 = beta / r;
    const sumzeta = zeta1 + zeta2;
    const difzeta = zeta1 - zeta2;

    // Partial derivative terms for zeta derivatives
    const dpdz1 = r;
    const dpdz2 = r;
    let dptdz1 = r;
    let dptdz2 = -r;
    const dpdr = 0.5 * sumzeta;
    let dptdr = 0.5 * difzeta;
    const d2pdz1r = 1.0;
    const d2pdz2r = 1.0;
    let d2ptdz1r = 1.0;
    let d2ptdz2r = -1.0;

    // Reverse quantum numbers if necessary; also change the sign of difzeta to match the change in sign of pt
    if (n2 < n1) {
      const k = n1;
      n1 = n2;
      n2 = k;
      pt = -pt;
      // native also negates difzeta here; it is not read again, so the assignment is left out
      dptdr = -dptdr;
      dptdz1 = -dptdz1;
      dptdz2 = -dptdz2;
      d2ptdz1r = -d2ptdz1r;
      d2ptdz2r = -d2ptdz2r;
    }

    // Trap for enormously long distances which would cause caintgs or cbintgs to crash with an overflow
    if (p > 86.0 || pt > 86.0) {
      o.s = 0.0;
      return;
    }

    // Find a and b integrals
    const a = this.a, b = this.b;
    this.caintgs(p, n1 + n2 + 3, a);
    this.cbintgs(pt, n1 + n2 + 3, b);

    // Convert derivatives with respect to p and pt into derivatives with respect to zeta1 and zeta2
    const ulim = n1 + n2 + 1;
    for (let i = 1; i <= ulim; i++) {
      this.da1[i] = -a[i + 1] * dpdz1;
      this.da2[i] = -a[i + 1] * dpdz2;
      this.db1[i] = -b[i + 1] * dptdz1;
      this.db2[i] = -b[i + 1] * dptdz2;
      this.d2a11[i] = a[i + 2] * dpdz1 * dpdz1;
      this.d2a12[i] = a[i + 2] * dpdz1 * dpdz2;
      this.d2a22[i] = a[i + 2] * dpdz2 * dpdz2;
      this.d2b11[i] = b[i + 2] * dptdz1 * dptdz1;
      this.d2b12[i] = b[i + 2] * dptdz1 * dptdz2;
      this.d2b22[i] = b[i + 2] * dptdz2 * dptdz2;
      this.dar[i] = -a[i + 1] * dpdr;
      this.dbr[i] = -b[i + 1] * dptdr;
      this.d2a1r[i] = a[i + 2] * dpdz1 * dpdr - a[i + 1] * d2pdz1r;
      this.d2a2r[i] = a[i + 2] * dpdz2 * dpdr - a[i + 1] * d2pdz2r;
      this.d2b1r[i] = b[i + 2] * dptdz1 * dptdr - b[i + 1] * d2ptdz1r;
      this.d2b2r[i] = b[i + 2] * dptdz2 * dptdr - b[i + 1] * d2ptdz2r;
      this.d2ar2[i] = a[i + 2] * dpdr * dpdr;
      this.d2br2[i] = b[i + 2] * dptdr * dptdr;
    }

    // Begin section used for overlap integrals involving s functions
    for (let i1 = 1; i1 <= ulim; i1++) {
      const nni1 = n1 + n2 - i1 + 2;
      const coff = this.coeffs(n1, n2, i1 - 1);
      o.deriv = o.deriv + coff * (this.dar[i1] * b[nni1] + a[i1] * this.dbr[nni1]);
      x = x + coff * a[i1] * b[nni1];
      o.dzeta1 = o.dzeta1 + coff * (this.da1[i1] * b[nni1] + a[i1] * this.db1[nni1]);
      o.dzeta2 = o.dzeta2 + coff * (this.da2[i1] * b[nni1] + a[i1] * this.db2[nni1]);
      o.d2zeta11 = o.d2zeta11 + coff * (this.d2a11[i1] * b[nni1] + a[i1] * this.d2b11[nni1] +
                                        2.0 * this.da1[i1] * this.db1[nni1]);
      o.d2zeta12 = o.d2zeta12 + coff * (this.d2a12[i1] * b[nni1] + a[i1] * this.d2b12[nni1] +
                                        this.da1[i1] * this.db2[nni1] + this.da2[i1] * this.db1[nni1]);
      o.d2zeta22 = o.d2zeta22 + coff * (this.d2a22[i1] * b[nni1] + a[i1] * this.d2b22[nni1] +
                                        2.0 * this.da2[i1] * this.db2[nni1]);
      o.d2zeta1r = o.d2zeta1r + coff * (this.d2a1r[i1] * b[nni1] + this.dar[i1] * this.db1[nni1] +
                                        this.da1[i1] * this.dbr[nni1] + a[i1] * this.d2b1r[nni1]);
      o.d2zeta2r = o.d2zeta2r + coff * (this.d2a2r[i1] * b[nni1] + this.dar[i1] * this.db2[nni1] +
                                        this.da2[i1] * this.dbr[nni1] + a[i1] * this.d2b2r[nni1]);
      o.deriv2 = o.deriv2 + coff * (this.d2ar2[i1] * b[nni1] + a[i1] * this.d2br2[nni1] +
                                    2.0 * this.dar[i1] * this.dbr[nni1]);
    }
    o.s = x * 0.5;
    o.deriv = 0.5 * o.deriv;
    o.deriv2 = 0.5 * o.deriv2;
    o.dzeta1 = 0.5 * o.dzeta1;
    o.dzeta2 = 0.5 * o.dzeta2;
    o.d2zeta11 = 0.5 * o.d2zeta11;
    o.d2zeta12 = 0.5 * o.d2zeta12;
    o.d2zeta22 = 0.5 * o.d2zeta22;
    o.d2zeta1r = 0.5 * o.d2zeta1r;
    o.d2zeta2r = 0.5 * o.d2zeta2r;
  }

  /** coeffs (pair_smtbq.cpp): the coefficient of the k-th term of the overlap sum (uses fct). */
  private coeffs(na: number, nb: number, k: number): number {
    let coeffs = 0.0;
    const l = na + nb - k;
    const ie = Math.min(l, na) + 1;
    const je = Math.min(l, nb);
    const ia = l - je + 1;
    for (let il = ia; il <= ie; il++) {
      const i = il - 1;
      const j = l - i;  // D'ou vient le i
      coeffs = coeffs + this.binm(na, i) * this.binm(nb, j) * powint(-1.0, j);
    }
    return coeffs;
  }

  /** binm(n,i) = fct(n+1)/(fct(n-i+1)*fct(i+1)) (pair_smtbq.cpp). */
  private binm(n: number, i: number): number {
    return this.fct[n + 1] / (this.fct[n - i + 1] * this.fct[i + 1]);
  }

  /** caintgs (pair_smtbq.cpp): a(i+1) integrals for the exponent x. */
  private caintgs(x: number, k: number, a: Float64Array): void {
    const cste = Math.exp(-x);
    const rx = 1.0 / x;
    a[1] = cste * rx;
    for (let i = 1; i <= k; i++) {
      a[i + 1] = (a[i] * i + cste) * rx;
    }
  }

  /**
   * cbintgs (pair_smtbq.cpp): fills b(i+1) (the "b(i) is b(i-1) in the usual notation" integrals). The branch
   * structure follows the C++ goto labels (exponential formula, 6/7/12/15-term series, x extremely small).
   */
  private cbintgs(x: number, k: number, b: Float64Array): void {
    const i0 = 0;
    const absx = Math.abs(x);
    let last = 0;
    let branch: 'exp' | 'series' | 'small';
    if (absx > 3.0) branch = 'exp';
    else if (absx > 2.0) {
      if (k <= 10) branch = 'exp';
      else { branch = 'series'; last = 15; }
    } else if (absx > 1.0) {
      if (k <= 7) branch = 'exp';
      else { branch = 'series'; last = 12; }
    } else if (absx > 0.5) {
      if (k <= 5) branch = 'exp';
      else { branch = 'series'; last = 7; }
    } else if (absx > 1.0e-8) { branch = 'series'; last = 6; }
    else branch = 'small';

    if (branch === 'exp') {
      const expx = Math.exp(x);
      const expmx = 1.0 / expx;
      const rx = 1.0 / x;
      b[1] = (expx - expmx) * rx;
      for (let i = 1; i <= k; i++) {
        b[i + 1] = (i * b[i] + powint(-1.0, i) * expx - expmx) * rx;
      }
    } else if (branch === 'series') {
      for (let i = i0; i <= k; i++) {
        let y = 0.0;
        for (let m = i0; m <= last; m++) {
          const ytrm = powint(-x, m - 1) * (1.0 - powint(-1.0, m + i + 1)) / (this.fct[m + 1] * (m + i + 1));
          y = y + ytrm * (-x);
        }
        b[i + 1] = y;
      }
    } else {
      // x extremely small
      for (let i = i0; i <= k; i++) {
        b[i + 1] = (1.0 - powint(-1.0, i + 1)) / (i + 1);
      }
    }
  }
}
