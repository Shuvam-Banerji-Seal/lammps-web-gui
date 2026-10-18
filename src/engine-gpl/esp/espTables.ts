/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/pair.cpp at b332f28, the parts used
 * by the esp pair styles with pair_modify table N (N > 0, the default 12): Pair::init_bitmap and Pair::init_tables
 * with the espflag branches (the polynomial kernels of kspace style esp in place of the erfc of Ewald and PPPM). The
 * lookup is the bitmapped float table of native LAMMPS: rsq is stored as a 32-bit float and its bits index the table.
 * Copyright of the LAMMPS sources: Sandia Corporation, GPL-2.0. This file is GPL-2.0 too (see src/engine-gpl/LICENSE).
 */
import { pow as libmPow } from '../libm/pow';

const FLT_MANT_DIG = 24;
const f32 = new Float32Array(1);
const i32 = new Int32Array(f32.buffer);

/** (float) x, as the value of a 32-bit float (rsq_lookup.f = x). */
export const toFloat = (x: number): number => { f32[0] = x; return f32[0]; };
/** The 32-bit pattern of (float) x (rsq_lookup.i after rsq_lookup.f = x). */
export const floatBits = (x: number): number => { f32[0] = x; return i32[0]; };
/** The float with bit pattern i (rsq_lookup.f after rsq_lookup.i = i). */
export const bitsToFloat = (i: number): number => { i32[0] = i; return f32[0]; };

/** powint(x, n) of math_special (x = 2 here: exact). */
const powint = (x: number, n: number): number => {
  let yy: number, ww: number;
  if (x === 0.0) return 0.0;
  let nn = n > 0 ? n : -n;
  ww = x;
  for (yy = 1.0; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

export interface BitmapInfo { masklo: number; maskhi: number; nmask: number; nshiftbits: number }

/** Pair::init_bitmap(inner, outer, ntablebits) (src/pair.cpp at b332f28). */
export const initBitmap = (inner: number, outer: number, ntablebits: number): BitmapInfo => {
  let nlowermin = 1;
  while ((powint(2.0, nlowermin) > inner * inner) || (powint(2.0, nlowermin + 1) <= inner * inner)) {
    if (powint(2.0, nlowermin) <= inner * inner) nlowermin++;
    else nlowermin--;
  }
  let nexpbits = 0;
  const requiredRange = outer * outer / powint(2.0, nlowermin);
  let availableRange = 2.0;
  while (availableRange < requiredRange) {
    nexpbits++;
    availableRange = libmPow(2.0, powint(2.0, nexpbits));
  }
  const nmantbits = ntablebits - nexpbits;
  if (nmantbits + 1 > FLT_MANT_DIG) throw new Error('Too many mantissa bits for lookup table');
  if (nmantbits < 3) throw new Error('Too few bits for lookup table');
  const nshiftbits = FLT_MANT_DIG - (nmantbits + 1);
  let nmask = 1;
  for (let j = 0; j < ntablebits + nshiftbits; j++) nmask *= 2;
  nmask -= 1;
  const maskhi = floatBits(outer * outer) & ~nmask;
  const masklo = floatBits(inner * inner) & ~nmask;
  return { masklo, maskhi, nmask, nshiftbits };
};

export interface EspTable {
  ncoultablebits: number;
  ncoulmask: number;
  ncoulshiftbits: number;
  /** tabinnersq after init (the float minimum of rsq_lookup over the table). */
  tabinnersq: number;
  rtable: Float64Array; ftable: Float64Array; ctable: Float64Array; etable: Float64Array;
  drtable: Float64Array; dftable: Float64Array; dctable: Float64Array; detable: Float64Array;
}

/** esp kernels at r (pair.cpp espflag branch of init_tables): fesp and eesp. */
const espAt = (r: number, cut: number, fpoly: number[], epoly: number[]): { fesp: number; eesp: number } => {
  const rCoul = 2.0 * r / cut - 1.0;
  let polyR = 1.0;
  let fesp = fpoly[0];
  for (let ii = 1; ii < fpoly.length; ii++) {
    polyR *= rCoul;
    fesp += fpoly[ii] * polyR;
  }
  polyR = 1.0;
  let eesp = epoly[0];
  for (let ii = 1; ii < epoly.length; ii++) {
    polyR *= rCoul;
    eesp += epoly[ii] * polyR;
  }
  return { fesp, eesp };
};

/** Pair::init_tables(cut_coul, nullptr) with espflag (src/pair.cpp at b332f28). */
export const initEspTables = (cutCoul: number, tabinner: number, ncoultablebits: number, qqrd2e: number,
  fpoly: number[], epoly: number[]): EspTable => {
  const cutCoulsq = cutCoul * cutCoul;
  const tabinnersq0 = tabinner * tabinner;
  const bm = initBitmap(tabinner, cutCoul, ncoultablebits);
  const ncoulshiftbits = bm.nshiftbits;
  const ncoulmask = bm.nmask;
  const { masklo, maskhi } = bm;
  let ntable = 1;
  for (let i = 0; i < ncoultablebits; i++) ntable *= 2;
  const rtable = new Float64Array(ntable), ftable = new Float64Array(ntable), ctable = new Float64Array(ntable);
  const etable = new Float64Array(ntable), drtable = new Float64Array(ntable), dftable = new Float64Array(ntable);
  const dctable = new Float64Array(ntable), detable = new Float64Array(ntable);
  // minrsq_lookup.i = 0 << shift | maskhi
  let minrsqF = bitsToFloat((0 << ncoulshiftbits) | maskhi);
  for (let i = 0; i < ntable; i++) {
    let bits = ((i << ncoulshiftbits) | masklo);
    let rsqF = bitsToFloat(bits);
    if (rsqF < tabinnersq0) {
      bits = ((i << ncoulshiftbits) | maskhi);
      rsqF = bitsToFloat(bits);
    }
    const r = Math.sqrt(rsqF);
    const k = espAt(r, cutCoul, fpoly, epoly);
    rtable[i] = rsqF;
    ctable[i] = qqrd2e / r;
    ftable[i] = qqrd2e / r * k.fesp;
    etable[i] = qqrd2e / r * k.eesp;
    minrsqF = minrsqF < rsqF ? minrsqF : rsqF;
  }
  const tabinnersq = minrsqF;
  const ntablem1 = ntable - 1;
  for (let i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0 / (rtable[i + 1] - rtable[i]);
    dftable[i] = ftable[i + 1] - ftable[i];
    dctable[i] = ctable[i + 1] - ctable[i];
    detable[i] = etable[i + 1] - etable[i];
  }
  drtable[ntablem1] = 1.0 / (rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  // deltas at itablemax: from the last bin to the cutoff squared
  let itablemin = (floatBits(minrsqF) & ncoulmask);
  itablemin >>= ncoulshiftbits;
  let itablemax = itablemin - 1;
  if (itablemin === 0) itablemax = ntablem1;
  let rsqF = bitsToFloat((itablemax << ncoulshiftbits) | maskhi);
  if (rsqF < cutCoulsq) {
    rsqF = toFloat(cutCoulsq);
    const r = Math.sqrt(rsqF);
    const k = espAt(r, cutCoul, fpoly, epoly);
    const cTmp = qqrd2e / r;
    const fTmp = qqrd2e / r * k.fesp;
    const eTmp = qqrd2e / r * k.eesp;
    drtable[itablemax] = 1.0 / (rsqF - rtable[itablemax]);
    dftable[itablemax] = fTmp - ftable[itablemax];
    dctable[itablemax] = cTmp - ctable[itablemax];
    detable[itablemax] = eTmp - etable[itablemax];
  }
  return { ncoultablebits, ncoulmask, ncoulshiftbits, tabinnersq, rtable, ftable, ctable, etable,
    drtable, dftable, dctable, detable };
};

/** Lazily built esp lookup table of one pair style, rebuilt when the cutoff, the table size or the kspace polynomials change. */
export class EspTableCache {
  private tab: EspTable | null = null;
  private key: [number, number, number, number, number[] | null, number[] | null] | null = null;

  get(cutCoul: number, tabinnerRsq: number, ntablebits: number, qqrd2e: number, fpoly: number[], epoly: number[]): EspTable {
    const k = this.key;
    if (!this.tab || !k || k[0] !== cutCoul || k[1] !== tabinnerRsq || k[2] !== ntablebits || k[3] !== qqrd2e ||
        k[4] !== fpoly || k[5] !== epoly) {
      // tabinner is squared on the pair (pair_modify tabinner); native squares the unsquared value
      this.tab = initEspTables(cutCoul, Math.sqrt(tabinnerRsq), ntablebits, qqrd2e, fpoly, epoly);
      this.key = [cutCoul, tabinnerRsq, ntablebits, qqrd2e, fpoly, epoly];
    }
    return this.tab;
  }
}
