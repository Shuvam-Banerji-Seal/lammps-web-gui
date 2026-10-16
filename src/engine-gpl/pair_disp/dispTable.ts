/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/pair.cpp (Pair::init_bitmap and Pair::init_tables_disp) at commit b332f28, the dispersion lookup table of
 * pair_style lj/long/coul/long (src/KSPACE/pair_lj_long_coul_long.cpp, the "table real space" branch of compute). Copyright
 * Sandia Corporation (LAMMPS development team; the dispersion table is by Wayne Mitchell, Loyola University). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_modify table/disp N (docs.lammps.org/pair_modify.html): "The *table/disp* keyword ... sets the size of the lookup
 * table ... for long-range dispersion". As native: the table has 2^N entries, bitmapped on the float32 bits of rsq (the
 * mantissa bits of each octave are chosen from the inner and outer cutoffs), with value and delta arrays for linear
 * interpolation. Values are stored at the lower edge of each bin, and the bins are connected periodically.
 *
 * Native quirk kept as is: the last-bin correction at itablemax uses the rsq left over from the table loop (the variable
 * rsq of Pair::init_tables_disp, the last table node), not the cutoff squared that the correction then stores.
 */

import type { DispTable } from '../../engine/force/pair/lj_long';
import { exp as libmExp } from '../libm/exp';

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
/** The float32 bits of x (C union_int_float_t). */
const bitsOf = (x: number): number => {
  f32[0] = x;
  return u32[0];
};
/** The float32 value of bits b. */
const floatOf = (b: number): number => {
  u32[0] = b >>> 0;
  return f32[0];
};

const FLT_MANT_DIG = 24;

/** Pair::init_bitmap: the bit masks and the shift of a bitmapped table from the inner and outer cutoffs. */
const initBitmap = (inner: number, outer: number, ntablebits: number): { masklo: number; maskhi: number; nmask: number; nshift: number } => {
  if (ntablebits > 32) throw new RangeError('Too many total bits for bitmapped Dispersion lookup table');
  let nlowermin = 1;
  while (2 ** nlowermin > inner * inner || 2 ** (nlowermin + 1) <= inner * inner) {
    if (2 ** nlowermin <= inner * inner) nlowermin++;
    else nlowermin--;
  }
  let nexpbits = 0;
  const requiredRange = (outer * outer) / 2 ** nlowermin;
  let availableRange = 2.0;
  while (availableRange < requiredRange) {
    nexpbits++;
    availableRange = 2 ** (2 ** nexpbits);
  }
  const nmantbits = ntablebits - nexpbits;
  if (nexpbits > 32 - FLT_MANT_DIG) throw new RangeError('Too many exponent bits for lookup table');
  if (nmantbits + 1 > FLT_MANT_DIG) throw new RangeError('Too many mantissa bits for lookup table');
  if (nmantbits < 3) throw new RangeError('Too few bits for lookup table');
  const nshift = FLT_MANT_DIG - (nmantbits + 1);
  let nmask = 1;
  for (let j = 0; j < ntablebits + nshift; j++) nmask *= 2;
  nmask -= 1;
  const maskhi = (bitsOf(outer * outer) & ~nmask) >>> 0;
  const masklo = (bitsOf(inner * inner) & ~nmask) >>> 0;
  return { masklo, maskhi, nmask, nshift };
};

/**
 * Pair::init_tables_disp for the dispersion G-ewald g (g_ewald_6), the global LJ cutoff cutLJ, pair_modify
 * table/disp ntablebits and the inner cutoff tabinnerDisp (pair_modify tabinner/disp, default sqrt(2)).
 */
export const makeDispTable = (ntablebits: number, g: number, cutLJ: number, tabinnerDisp: number): DispTable => {
  const g2 = g * g, g6 = g2 * g2 * g2, g8 = g6 * g2;
  const tabinnerdispsq = tabinnerDisp * tabinnerDisp;
  const { masklo, maskhi, nmask, nshift } = initBitmap(tabinnerDisp, cutLJ, ntablebits);
  const ntable = 2 ** ntablebits;
  const rdisp = new Float64Array(ntable), fdisp = new Float64Array(ntable), edisp = new Float64Array(ntable);
  const drdisp = new Float64Array(ntable), dfdisp = new Float64Array(ntable), dedisp = new Float64Array(ntable);
  // the lookup bits of the table nodes: the masklo bits, or the maskhi bits below the inner cutoff
  let minrsq = floatOf(maskhi);
  let rsq = 0;
  for (let i = 0; i < ntable; i++) {
    let bits = ((i << nshift) | masklo) >>> 0;
    if (floatOf(bits) < tabinnerdispsq) bits = ((i << nshift) | maskhi) >>> 0;
    rsq = floatOf(bits);
    let x2 = g2 * rsq;
    const a2 = 1.0 / x2;
    x2 = a2 * libmExp(-x2);
    rdisp[i] = rsq;
    fdisp[i] = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * x2 * rsq;
    edisp[i] = g6 * ((a2 + 1.0) * a2 + 0.5) * x2;
    if (rsq < minrsq) minrsq = rsq;
  }
  const inner = minrsq;
  const ntablem1 = ntable - 1;
  for (let i = 0; i < ntablem1; i++) {
    drdisp[i] = 1.0 / (rdisp[i + 1] - rdisp[i]);
    dfdisp[i] = fdisp[i + 1] - fdisp[i];
    dedisp[i] = edisp[i + 1] - edisp[i];
  }
  // the periodic connection between the last and the first node
  drdisp[ntablem1] = 1.0 / (rdisp[0] - rdisp[ntablem1]);
  dfdisp[ntablem1] = fdisp[0] - fdisp[ntablem1];
  dedisp[ntablem1] = edisp[0] - edisp[ntablem1];
  // the bin of the smallest rsq is itablemin; its predecessor (or the last node) is itablemax, whose deltas run to the cutoff
  const itablemin = (bitsOf(minrsq) & nmask) >>> nshift;
  const itablemax = itablemin === 0 ? ntablem1 : itablemin - 1;
  const cutLJsq = cutLJ * cutLJ;
  if (floatOf(((itablemax << nshift) | maskhi) >>> 0) < cutLJsq) {
    const rcut = floatOf(bitsOf(cutLJsq));
    let x2 = g2 * rsq;
    const a2 = 1.0 / x2;
    x2 = a2 * libmExp(-x2);
    const fTmp = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * x2 * rsq;
    const eTmp = g6 * ((a2 + 1.0) * a2 + 0.5) * x2;
    drdisp[itablemax] = 1.0 / (rcut - rdisp[itablemax]);
    dfdisp[itablemax] = fTmp - fdisp[itablemax];
    dedisp[itablemax] = eTmp - edisp[itablemax];
  }
  const lookup = (rsqv: number): number => ((bitsOf(rsqv) & nmask) >>> nshift);
  return {
    inner,
    force(rsqv: number): number {
      const k = lookup(rsqv);
      const fr = (rsqv - rdisp[k]) * drdisp[k];
      return fdisp[k] + fr * dfdisp[k];
    },
    energy(rsqv: number): number {
      const k = lookup(rsqv);
      const fr = (rsqv - rdisp[k]) * drdisp[k];
      return edisp[k] + fr * dedisp[k];
    },
  };
};
