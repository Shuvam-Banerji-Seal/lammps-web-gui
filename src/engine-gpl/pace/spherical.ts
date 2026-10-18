/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use: ML-PACE/ace-evaluator/ace_spherical_cart.cpp and
 * ace_spherical_cart.h of lammps-user-pace v.2025.12.4.p1 (checked against LAMMPS's CMake SHA256). The LAMMPS package that
 * uses it is src/ML-PACE/pair_pace.cpp at commit b332f28. Library authors (from its headers): Yury Lysogorskiy, Anton
 * Bochkarev, Sarath Menon, Ralf Drautz and co-authors (ICAMS, Ruhr-University Bochum). Used here under GPL-2.0
 * (src/engine-gpl/LICENSE).
 *
 * Cartesian spherical harmonics Y_lm(r_hat) for 0 <= m <= l (the evaluator reads only m >= 0 and builds m < 0 by
 * conjugation), with their Cartesian gradients. Recurrences and operation order follow the C++ source.
 */

/** Y00 in ace_spherical_cart.h is 1. */
const Y00 = 1;
const sq3 = 1.73205080756887719318; // sqrt(3)
const sq3o2 = 1.22474487139158894067; // sqrt(3/2)

/** Index of (l, m) in a (lmax+1) x (lmax+1) table: l * (lmax + 1) + m. */
export class CartSphericalHarmonics {
  readonly lmax: number;
  private readonly dim: number;
  private alm: Float64Array;
  private blm: Float64Array;
  private cl: Float64Array;
  private dl: Float64Array;
  private plm: Float64Array;
  private dplm: Float64Array;
  /** ylm(l, m) real and imaginary parts */
  readonly ylmRe: Float64Array;
  readonly ylmIm: Float64Array;
  /**
   * dylm(l, m).a[c] for c = 0..2 (x, y, z): six numbers per (l, m), [re0, im0, re1, im1, re2, im2] at 6 * (l*dim + m).
   */
  readonly dylm: Float64Array;

  constructor(lmax: number) {
    this.lmax = lmax;
    const d = lmax + 1;
    this.dim = d;
    this.alm = new Float64Array(d * d);
    this.blm = new Float64Array(d * d);
    this.cl = new Float64Array(d + 1);
    this.dl = new Float64Array(d + 1);
    this.plm = new Float64Array(d * d);
    this.dplm = new Float64Array(d * d);
    this.ylmRe = new Float64Array(d * d);
    this.ylmIm = new Float64Array(d * d);
    this.dylm = new Float64Array(6 * d * d);
    this.preCompute();
  }

  /** pre_compute (ace_spherical_cart.cpp): the recurrence coefficients alm, blm, cl, dl. */
  private preCompute(): void {
    const d = this.dim, lmax = this.lmax;
    for (let l = 1; l <= lmax; l++) {
      const lsq = l * l;
      const ld = 2 * l;
      const l1 = 4 * lsq - 1;
      const l2 = lsq - ld + 1;
      for (let m = 0; m < l - 1; m++) {
        const msq = m * m;
        const a = Math.sqrt(l1 / (lsq - msq));
        const b = -Math.sqrt((l2 - msq) / (4 * l2 - 1));
        this.alm[l * d + m] = a;
        this.blm[l * d + m] = b;
      }
    }
    for (let l = 1; l <= lmax; l++) {
      this.cl[l] = -Math.sqrt(1.0 + 0.5 / l);
      this.dl[l] = Math.sqrt(2 * (l - 1) + 3);
    }
  }

  /** compute_barplm: the normalised associated Legendre functions plm and their z-derivatives dplm. */
  private computeBarplm(rz: number, lmaxi: number): void {
    const d = this.dim, plm = this.plm, dplm = this.dplm;
    plm[0] = Y00;
    dplm[0] = 0.0;
    if (lmaxi > 0) {
      plm[1 * d + 0] = Y00 * sq3 * rz;
      dplm[1 * d + 0] = Y00 * sq3;
      plm[1 * d + 1] = -sq3o2 * Y00;
      dplm[1 * d + 1] = 0.0;
      for (let l = 2; l <= lmaxi; l++) {
        for (let m = 0; m < l - 1; m++) {
          const a = this.alm[l * d + m], b = this.blm[l * d + m];
          plm[l * d + m] = a * (rz * plm[(l - 1) * d + m] + b * plm[(l - 2) * d + m]);
          dplm[l * d + m] = a * (plm[(l - 1) * d + m] + rz * dplm[(l - 1) * d + m] + b * dplm[(l - 2) * d + m]);
        }
        const t = this.dl[l] * plm[(l - 1) * d + (l - 1)];
        plm[l * d + (l - 1)] = t * rz;
        dplm[l * d + (l - 1)] = t;
        plm[l * d + l] = this.cl[l] * plm[(l - 1) * d + (l - 1)];
        dplm[l * d + l] = 0.0;
      }
    }
  }

  /** compute_ylm (ace_spherical_cart.cpp): Y_lm and dY_lm/dx, dY_lm/dy, dY_lm/dz of the unit vector (rx, ry, rz). */
  computeYlm(rx: number, ry: number, rz: number, lmaxi: number): void {
    const d = this.dim, plm = this.plm, dplm = this.dplm;
    const ylmRe = this.ylmRe, ylmIm = this.ylmIm, dy = this.dylm;
    const phaseRe0 = rx, phaseIm0 = ry;
    this.computeBarplm(rz, lmaxi);
    // m = 0
    let m = 0;
    for (let l = 0; l <= lmaxi; l++) {
      const k = l * d + m;
      ylmRe[k] = plm[k];
      ylmIm[k] = 0.0;
      const dyzRe = dplm[k];
      const rdyRe = dyzRe * rz;
      const o = 6 * k;
      dy[o + 0] = -rdyRe * rx;
      dy[o + 1] = 0.0;
      dy[o + 2] = -rdyRe * ry;
      dy[o + 3] = 0.0;
      dy[o + 4] = dyzRe - rdyRe * rz;
      dy[o + 5] = 0;
    }
    // m = 1
    m = 1;
    for (let l = 1; l <= lmaxi; l++) {
      const k = l * d + m;
      const p = plm[k];
      // ylm(l, 1) = phase * plm(l, 1)
      ylmRe[k] = phaseRe0 * p;
      ylmIm[k] = phaseIm0 * p;
      const dyxRe = p, dyxIm = 0.0;
      const dyyRe = 0.0, dyyIm = p;
      const dpz = dplm[k];
      const dyzRe = phaseRe0 * dpz, dyzIm = phaseIm0 * dpz;
      const rdyRe = rx * dyxRe + rz * dyzRe;
      const rdyIm = ry * dyyIm + rz * dyzIm;
      const o = 6 * k;
      dy[o + 0] = dyxRe - rdyRe * rx;
      dy[o + 1] = -rdyIm * rx;
      dy[o + 2] = -rdyRe * ry;
      dy[o + 3] = dyyIm - rdyIm * ry;
      dy[o + 4] = dyzRe - rdyRe * rz;
      dy[o + 5] = dyzIm - rdyIm * rz;
      void dyxIm;
    }
    // m >= 2: phasem = phase^m (complex powers by repeated complex products)
    let phRe = phaseRe0, phIm = phaseIm0;
    for (let mm = 2; mm <= lmaxi; mm++) {
      const mRe = phRe * mm, mIm = phIm * mm;
      // phasem = phasem * phase (ACEComplex::operator*)
      const nRe = phRe * phaseRe0 - phIm * phaseIm0;
      const nIm = phRe * phaseIm0 + phIm * phaseRe0;
      phRe = nRe;
      phIm = nIm;
      for (let l = mm; l <= lmaxi; l++) {
        const k = l * d + mm;
        const p = plm[k];
        ylmRe[k] = phRe * p;
        ylmIm[k] = phIm * p;
        const dyxRe = mRe * p, dyxIm = mIm * p;
        const dyyRe = -dyxIm, dyyIm = dyxRe;
        const dz = dplm[k];
        const dyzRe = phRe * dz, dyzIm = phIm * dz;
        const rdyRe = rx * dyxRe + ry * dyyRe + rz * dyzRe;
        const rdyIm = rx * dyxIm + ry * dyyIm + rz * dyzIm;
        const o = 6 * k;
        dy[o + 0] = dyxRe - rdyRe * rx;
        dy[o + 1] = dyxIm - rdyIm * rx;
        dy[o + 2] = dyyRe - rdyRe * ry;
        dy[o + 3] = dyyIm - rdyIm * ry;
        dy[o + 4] = dyzRe - rdyRe * rz;
        dy[o + 5] = dyzIm - rdyIm * rz;
      }
    }
  }
}
