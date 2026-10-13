/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BOCS/ldd_indicator.h, ldd_indicator.cpp,
 * ldd_indicator_lucy.cpp and ldd_indicator_dpd.cpp (the LddIndicator base and its "lucy" and "dpd" weight functions of
 * pair_style ldd). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributed by Michael R. DeLyser and Maria C.
 * Lesniewski (Pennsylvania State University), from the BOCS package sources.
 *
 * The weight w(r) of a species pair is a normalised function that vanishes at rc; the local density of an atom is the
 * sum of w over its neighbours (docs.lammps.org/pair_ldd.html). Only the indicators the example inputs use are here
 * ("lucy" and "dpd"); the others are refused by the pair style by name.
 */
import { StyleError } from '../../engine/force/types';

/** The weight function interface (LddIndicator): w, w' and w'' with the normalisation of the dimension. */
export interface LddIndicator {
  readonly r0: number;
  readonly rc: number;
  readonly norm: number;
  readonly invnorm: number;
  w(r: number): number;
  wp(r: number): number;
  wp2(r: number): number;
}

/** ldd_indicator_lucy.cpp: a polynomial in r^2 with w(rc) = w'(rc) = 0 (Lucy-type weight). */
export class LddIndicatorLucy implements LddIndicator {
  readonly r0: number;
  readonly rc: number;
  readonly norm: number;
  readonly invnorm: number;
  private readonly c: Float64Array;

  constructor(r0: number, rc: number, dim: 2 | 3) {
    this.r0 = r0;
    this.rc = rc;
    this.c = new Float64Array([1.0, 0.0, -6.0 / (rc * rc), 8.0 / (rc * rc * rc), -3.0 / (rc * rc * rc * rc)]);
    this.norm = dim === 2 ? 0.2 * Math.PI * (rc * rc) : 16.0 * Math.PI / 105.0 * (rc * rc * rc);
    this.invnorm = 1.0 / this.norm;
  }

  w(r: number): number {
    if (r >= this.rc) return 0.0;
    const c = this.c;
    const w = (c[0] + c[2] * (r * r) + c[3] * (r * r * r) + c[4] * (r * r * r * r)) * this.invnorm;
    // ldd_indicator_lucy.cpp: "return (w > 0.0 ? w : 0.0);" (round-off below zero near rc)
    return w > 0.0 ? w : 0.0;
  }

  wp(r: number): number {
    if (r >= this.rc) return 0.0;
    const c = this.c;
    return (2.0 * c[2] * r + 3.0 * c[3] * (r * r) + 4.0 * c[4] * (r * r * r)) * this.invnorm;
  }

  wp2(r: number): number {
    if (r > this.rc) return 0.0;
    const c = this.c;
    return (2.0 * c[2] + 6.0 * c[3] * r + 12.0 * c[4] * (r * r)) * this.invnorm;
  }
}

/** ldd_indicator_dpd.cpp: w(r) = (1 - r/rc)^2 normalised (the DPD weight). */
export class LddIndicatorDpd implements LddIndicator {
  readonly r0: number;
  readonly rc: number;
  readonly norm: number;
  readonly invnorm: number;
  private readonly c: Float64Array;

  constructor(r0: number, rc: number, dim: 2 | 3) {
    this.r0 = r0;
    this.rc = rc;
    this.c = new Float64Array([1.0, -2.0 / rc, 1.0 / (rc * rc)]);
    this.norm = dim === 2 ? Math.PI * rc * rc / 6.0 : 2.0 * Math.PI * (rc * rc * rc) / 15.0;
    this.invnorm = 1.0 / this.norm;
  }

  w(r: number): number {
    if (r > this.rc) return 0.0;
    const c = this.c;
    return (c[0] + c[1] * r + c[2] * (r * r)) * this.invnorm;
  }

  wp(r: number): number {
    if (r > this.rc) return 0.0;
    const c = this.c;
    return (c[1] + 2.0 * c[2] * r) * this.invnorm;
  }

  wp2(r: number): number {
    if (r > this.rc) return 0.0;
    return (2.0 * this.c[2]) * this.invnorm;
  }
}

/** The indicator names pair_coeff accepts here (ldd_indicator_register.cpp lists five more, refused by name). */
const SUPPORTED_INDICATORS = ['dpd', 'lucy'];
const OTHER_INDICATORS = ['shell', 'smooth', 'sphere'];

/** new_indicator + init_coeffs (pair_ldd.cpp coeff_ldd: "indicator wtype r0 rc"). */
export const makeIndicator = (name: string, r0: number, rc: number, dim: 2 | 3): LddIndicator => {
  if (name === 'lucy') return new LddIndicatorLucy(r0, rc, dim);
  if (name === 'dpd') return new LddIndicatorDpd(r0, rc, dim);
  if (OTHER_INDICATORS.includes(name)) {
    throw new StyleError(`ldd indicator '${name}' is not supported by the browser engine (supported: ${SUPPORTED_INDICATORS.join(', ')})`);
  }
  throw new StyleError(`Unrecognized ldd indicator style '${name}'`);
};
