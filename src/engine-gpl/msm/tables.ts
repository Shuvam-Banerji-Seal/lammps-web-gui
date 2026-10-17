/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/pair.cpp at commit b332f28 (Pair::init_bitmap and the msmflag branch of Pair::init_tables, without the
 * rRESPA tables), and src/kspace.h (KSpace::gamma and KSpace::dgamma) with the gcons/dgcons constants of src/kspace.cpp.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The MSM splitting function gamma(rho) (Hardy, Parallel Computing 35 (2009) 164-177, eq. 4) and the bitmapped lookup
 * tables the coul/msm pair styles use for pair_modify table N > 0: node rsq values are float32 numbers whose top
 * mantissa bits index the table, the value between nodes is linear in float32(rsq).
 */

/** kspace.cpp gcons[split_order][n]: the even polynomial of gamma for rho <= 1. */
const GCONS: readonly (readonly number[])[] = [
  [], [],
  [15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0],
  [35.0 / 16.0, -35.0 / 16.0, 21.0 / 16.0, -5.0 / 16.0],
  [315.0 / 128.0, -105.0 / 32.0, 189.0 / 64.0, -45.0 / 32.0, 35.0 / 128.0],
  [693.0 / 256.0, -1155.0 / 256.0, 693.0 / 128.0, -495.0 / 128.0, 385.0 / 256.0, -63.0 / 256.0],
  [3003.0 / 1024.0, -3003.0 / 512.0, 9009.0 / 1024.0, -2145.0 / 256.0, 5005.0 / 1024.0, -819.0 / 512.0, 231.0 / 1024.0],
];

/** kspace.cpp dgcons[split_order][n]: the odd polynomial of dgamma/drho for rho <= 1. */
const DGCONS: readonly (readonly number[])[] = [
  [], [],
  [-5.0 / 2.0, 3.0 / 2.0],
  [-35.0 / 8.0, 21.0 / 4.0, -15.0 / 8.0],
  [-105.0 / 16.0, 189.0 / 16.0, -135.0 / 16.0, 35.0 / 16.0],
  [-1155.0 / 128.0, 693.0 / 32.0, -1485.0 / 64.0, 385.0 / 32.0, -315.0 / 128.0],
  [-3003.0 / 256.0, 9009.0 / 256.0, -6435.0 / 128.0, 5005.0 / 128.0, -4095.0 / 256.0, 693.0 / 256.0],
];

/** KSpace::gamma for the given MSM order (kspace.h). */
export const msmGamma = (order: number, rho: number): number => {
  if (rho <= 1.0) {
    const splitOrder = order >> 1;
    const c = GCONS[splitOrder];
    const rho2 = rho * rho;
    let g = c[0];
    let rhoN = rho2;
    for (let n = 1; n <= splitOrder; n++) {
      g += c[n] * rhoN;
      rhoN *= rho2;
    }
    return g;
  }
  return 1.0 / rho;
};

/** KSpace::dgamma for the given MSM order (kspace.h). */
export const msmDgamma = (order: number, rho: number): number => {
  if (rho <= 1.0) {
    const splitOrder = order >> 1;
    const c = DGCONS[splitOrder];
    const rho2 = rho * rho;
    let dg = c[0] * rho;
    let rhoN = rho * rho2;
    for (let n = 1; n < splitOrder; n++) {
      dg += c[n] * rhoN;
      rhoN *= rho2;
    }
    return dg;
  }
  return -1.0 / rho / rho;
};

const f32 = new Float32Array(1);
const i32 = new Int32Array(f32.buffer);
/** float32 bits of x (union_int_float_t: rsq_lookup.f = x; rsq_lookup.i). */
export const floatBits = (x: number): number => { f32[0] = x; return i32[0]; };
const bitsFloat = (i: number): number => { i32[0] = i; return f32[0]; };

/** The lookup tables of Pair::init_tables (no rRESPA): value at the lower edge of a bin and the delta to the next. */
export interface CoulTables {
  /** Squared radius above which the table is used (the smallest node rsq). */
  readonly tabinnersq: number;
  readonly ncoulmask: number;
  readonly ncoulshiftbits: number;
  readonly rtable: Float64Array; readonly drtable: Float64Array;
  readonly ftable: Float64Array; readonly dftable: Float64Array;
  readonly ctable: Float64Array; readonly dctable: Float64Array;
  readonly etable: Float64Array; readonly detable: Float64Array;
}

/** Pair::init_bitmap: the float bit masks and shift of an ntablebits table spanning inner..outer. */
const initBitmap = (inner: number, outer: number, ntablebits: number): { masklo: number; maskhi: number; nmask: number; nshiftbits: number } => {
  if (ntablebits > 32) throw new Error('Too many total bits for bitmapped lookup table');
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
  // FLT_MANT_DIG = 24; 32 - 24 = 8 exponent bits
  if (nexpbits > 8) throw new Error('Too many exponent bits for lookup table');
  if (nmantbits + 1 > 24) throw new Error('Too many mantissa bits for lookup table');
  if (nmantbits < 3) throw new Error('Too few bits for lookup table');
  const nshiftbits = 24 - (nmantbits + 1);
  let nmask = 1;
  for (let j = 0; j < ntablebits + nshiftbits; j++) nmask *= 2;
  nmask -= 1;
  const maskhi = floatBits(outer * outer) & ~nmask;
  const masklo = floatBits(inner * inner) & ~nmask;
  return { masklo, maskhi, nmask: nmask | 0, nshiftbits };
};

/**
 * Pair::init_tables with msmflag set: ftable = qqrd2e/r fgamma, etable = qqrd2e/r egamma and ctable = qqrd2e/r at the
 * nodes, egamma = 1 - (r/rc) gamma(r/rc) and fgamma = 1 + (r/rc)^2 dgamma(r/rc).
 */
export const initMsmTables = (ncoultablebits: number, tabinner: number, cutCoul: number, qqrd2e: number, order: number): CoulTables => {
  const cutCoulsq = cutCoul * cutCoul;
  const tabinnersq0 = tabinner * tabinner;
  const { masklo, maskhi, nmask, nshiftbits } = initBitmap(tabinner, cutCoul, ncoultablebits);
  const ntable = 2 ** ncoultablebits;
  const rtable = new Float64Array(ntable), ftable = new Float64Array(ntable), ctable = new Float64Array(ntable), etable = new Float64Array(ntable);
  const drtable = new Float64Array(ntable), dftable = new Float64Array(ntable), dctable = new Float64Array(ntable), detable = new Float64Array(ntable);
  let minrsqBits = (0 << nshiftbits) | maskhi;
  for (let i = 0; i < ntable; i++) {
    let bits = (i << nshiftbits) | masklo;
    if (bitsFloat(bits) < tabinnersq0) bits = (i << nshiftbits) | maskhi;
    const rsq = bitsFloat(bits);
    const r = Math.sqrt(rsq);
    const egamma = 1.0 - (r / cutCoul) * msmGamma(order, r / cutCoul);
    const fgamma = 1.0 + (rsq / cutCoulsq) * msmDgamma(order, r / cutCoul);
    rtable[i] = rsq;
    ctable[i] = qqrd2e / r;
    ftable[i] = (qqrd2e / r) * fgamma;
    etable[i] = (qqrd2e / r) * egamma;
    // minrsq_lookup.f = MIN(minrsq_lookup.f, rsq_lookup.f)
    if (rsq < bitsFloat(minrsqBits)) minrsqBits = bits;
  }
  const tabinnersq = bitsFloat(minrsqBits);
  const ntablem1 = ntable - 1;
  for (let i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0 / (rtable[i + 1] - rtable[i]);
    dftable[i] = ftable[i + 1] - ftable[i];
    dctable[i] = ctable[i + 1] - ctable[i];
    detable[i] = etable[i + 1] - etable[i];
  }
  // tables are connected periodically between 0 and ntablem1
  drtable[ntablem1] = 1.0 / (rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  // the bin holding the largest r ends at the cutoff
  const itablemin = (minrsqBits & nmask) >> nshiftbits;
  const itablemax = itablemin === 0 ? ntablem1 : itablemin - 1;
  const topBits = (itablemax << nshiftbits) | maskhi;
  if (bitsFloat(topBits) < cutCoulsq) {
    f32[0] = cutCoulsq;
    const rsq = f32[0];
    const r = Math.sqrt(rsq);
    const egamma = 1.0 - (r / cutCoul) * msmGamma(order, r / cutCoul);
    const fgamma = 1.0 + (rsq / cutCoulsq) * msmDgamma(order, r / cutCoul);
    const cTmp = qqrd2e / r;
    const fTmp = (qqrd2e / r) * fgamma;
    const eTmp = (qqrd2e / r) * egamma;
    drtable[itablemax] = 1.0 / (rsq - rtable[itablemax]);
    dftable[itablemax] = fTmp - ftable[itablemax];
    dctable[itablemax] = cTmp - ctable[itablemax];
    detable[itablemax] = eTmp - etable[itablemax];
  }
  return { tabinnersq, ncoulmask: nmask, ncoulshiftbits: nshiftbits, rtable, drtable, ftable, dftable, ctable, dctable, etable, detable };
};
