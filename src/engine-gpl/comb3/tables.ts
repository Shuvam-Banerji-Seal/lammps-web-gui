/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp (PairComb3::tables, potal_calc, tri_point, vdwaals) and
 * src/MANYBODY/pair_comb3.h, at commit b332f28. Authors credited in the C++ file headers: Tzu-Ray Shan
 * (Sandia) and the COMB3 authors. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { erfcExact } from '../../engine/force/erfc';
import type { Comb3Param } from './params';

/** Look-up tables of PairComb3::tables(). Two-dimensional tables are row-major with nntypes columns. */
export interface Comb3Tables {
  nel: number;
  nntypes: number;
  /** intype[i*n+j]: interaction-type index of element pair (i, j). */
  intype: Int32Array;
  ncoul: number;
  ncoulLim: number;
  fafb: Float64Array;
  dfafb: Float64Array;
  ddfafb: Float64Array;
  phin: Float64Array;
  dphin: Float64Array;
  afb: Float64Array;
  dafb: Float64Array;
  /** erpaw[2*i] = erfc(alf r), erpaw[2*i+1] = exp(-(alf r)^2). */
  erpaw: Float64Array;
  /** Van der Waals look-up table and its derivative, indexed [k*nntypes + inty] for k < ncoul. */
  vvdw: Float64Array;
  vdvdw: Float64Array;
}

/** Interpolation indices and weights of tri_point. */
export interface TriPoint { mr1: number; mr2: number; mr3: number; sr1: number; sr2: number; sr3: number }

/**
 * tri_point: three-point interpolation indices in the 0.001 A grid. Writes into `tp` (no allocation per pair).
 */
export const triPoint = (rsq: number, cutmax: number, tp: TriPoint): void => {
  const rin = 0.1000, dr = 0.0010;
  let r = Math.sqrt(rsq);
  if (r < rin + 2.0 * dr) r = rin + 2.0 * dr;
  if (r > cutmax - 2.0 * dr) r = cutmax - 2.0 * dr;
  let rridr = (r - rin) / dr;
  let mr1 = Math.trunc(rridr);
  const dd = rridr - mr1;
  if (dd > 0.5) mr1 += 1;
  const rr1 = mr1 * dr;
  rridr = (r - rin - rr1) / dr;
  const rridr2 = rridr * rridr;
  tp.sr1 = (rridr2 - rridr) * 0.50;
  tp.sr2 = 1.0 - rridr2;
  tp.sr3 = (rridr2 + rridr) * 0.50;
  tp.mr1 = mr1;
  tp.mr2 = mr1 + 1;
  tp.mr3 = mr1 + 2;
};

/**
 * PairComb3::tables: Slater 1/r overlap tables, erfc table and the van der Waals spline. `typeElem[t]` is the
 * element index of atom type t (1-based; -1 for NULL), as the C++ map.
 */
export const comb3Tables = (
  params: Comb3Param[], elem3param: Int32Array, nel: number, typeElem: Int32Array, cutmax: number,
): Comb3Tables => {
  const n = nel;
  const dra = 0.001, drin = 0.100, drbuf = 0.100;
  const nnbuf = Math.trunc(drbuf / dra) + 1;
  const rc = cutmax;
  const alf = 0.20;
  const nntypes = Math.trunc((n + 1) * n / 2.0) + 1;
  const ncoul = Math.trunc((rc - drin) / dra) + nnbuf;
  const ncoulLim = Math.trunc(ncoul * 1.20);
  const intype = new Int32Array(n * n);
  let m = 0, k = n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j === i) { intype[i * n + j] = m; m += 1; } else if (j > i) { intype[i * n + j] = k; k += 1; } else { intype[i * n + j] = intype[j * n + i]; }
    }
  }
  const T = (rows: number): Float64Array => new Float64Array(rows * nntypes);
  const fafb = T(ncoulLim), dfafb = T(ncoulLim), ddfafb = T(ncoulLim);
  const phin = T(ncoulLim), dphin = T(ncoulLim), afb = T(ncoulLim), dafb = T(ncoulLim);
  const vvdw = T(ncoul), vdvdw = T(ncoul);
  const erpaw = new Float64Array(ncoulLim * 2);
  const at = (row: number, col: number): number => row * nntypes + col;

  // direct 1/r energy with Slater 1S orbital overlap (single-element terms, per element i)
  for (let i = 0; i < n; i++) {
    if (typeElem[i + 1] < 0) continue;
    let r = drin - dra;
    const itype = typeElem[i + 1];
    const iparamI = elem3param[(itype * n + itype) * n + itype];
    const z = params[iparamI].esm;
    const exp2ershift = Math.exp(-2.0 * z * rc);
    const afbshift = -exp2ershift * (z + 1.0 / rc);
    const dafbshift = exp2ershift * (2.0 * z * z + 2.0 * z / rc + 1.0 / (rc * rc));
    for (let j = 0; j < ncoul; j++) {
      const exp2er = Math.exp(-2.0 * z * r);
      phin[at(j, i)] = 1.0 - exp2er * (1.0 + 2.0 * z * r * (1.0 + z * r));
      dphin[at(j, i)] = (4.0 * exp2er * z * z * z * r * r);
      afb[at(j, i)] = -exp2er * (z + 1.0 / r) - afbshift - (r - rc) * dafbshift;
      dafb[at(j, i)] = -(exp2er * (2.0 * z * z + 2.0 * z / r + 1.0 / (r * r)) - dafbshift);
      r += dra;
    }
  }
  for (let i = 0; i < n; i++) {
    if (typeElem[i + 1] < 0) continue;
    for (let j = 0; j < n; j++) {
      if (typeElem[j + 1] < 0) continue;
      let r = drin - dra;
      if (j === i) {
        const itype = typeElem[i + 1];
        const inty = intype[itype * n + itype];
        const iparamI = elem3param[(itype * n + itype) * n + itype];
        const z = params[iparamI].esm;
        const zrc = z * rc;
        const exp2ersh = Math.exp(-2.0 * zrc);
        const fafash = -exp2ersh * (1.0 / rc + z * (11.0 / 8.0 + 3.0 / 4.0 * zrc + zrc * zrc / 6.0));
        const dfafash = exp2ersh * (1.0 / (rc * rc) + 2.0 * z / rc + z * z * (2.0 + 7.0 / 6.0 * zrc + zrc * zrc / 3.0));
        for (let kk = 0; kk < ncoul; kk++) {
          const zr = z * r;
          const exp2er = Math.exp(-2.0 * zr);
          const F1 = -exp2er * (1.0 / r + z * (11.0 / 8.0 + 3.0 / 4.0 * zr + zr * zr / 6.0));
          const dF1 = exp2er * (1.0 / (r * r) + 2.0 * z / r + z * z * (2.0 + 7.0 / 6.0 * zr + zr * zr / 3.0));
          const ddF1 = -exp2er * (2.0 / (r * r * r) + 4.0 * z / (r * r) + 4.0 * z * z / r +
            z * z * z / 3.0 * (17.0 / 2.0 + 5.0 * zr + 2.0 * zr * zr));
          fafb[at(kk, inty)] = F1 - fafash - (r - rc) * dfafash;
          dfafb[at(kk, inty)] = -(dF1 - dfafash);
          ddfafb[at(kk, inty)] = ddF1;
          r += dra;
        }
      } else {
        const itype = typeElem[i + 1];
        const jtype = typeElem[j + 1];
        const inty = intype[itype * n + jtype];
        const ea = params[elem3param[(itype * n + jtype) * n + jtype]].esm;
        const ea3 = ea * ea * ea;
        const eb = params[elem3param[(jtype * n + itype) * n + itype]].esm;
        const eb3 = eb * eb * eb;
        const E1 = ea * eb3 * eb / ((ea + eb) * (ea + eb) * (ea - eb) * (ea - eb));
        const E2 = eb * ea3 * ea / ((ea + eb) * (ea + eb) * (eb - ea) * (eb - ea));
        const E3 = (3.0 * ea * ea * eb3 * eb - eb3 * eb3) /
          ((ea + eb) * (ea + eb) * (ea + eb) * (ea - eb) * (ea - eb) * (ea - eb));
        const E4 = (3.0 * eb * eb * ea3 * ea - ea3 * ea3) /
          ((ea + eb) * (ea + eb) * (ea + eb) * (eb - ea) * (eb - ea) * (eb - ea));
        const exp2earsh = Math.exp(-2.0 * ea * rc);
        const exp2ebrsh = Math.exp(-2.0 * eb * rc);
        const fafbsh = -exp2earsh * (E1 + E3 / rc) - exp2ebrsh * (E2 + E4 / rc);
        const dfafbsh = exp2earsh * (2.0 * ea * (E1 + E3 / rc) + E3 / (rc * rc)) +
          exp2ebrsh * (2.0 * eb * (E2 + E4 / rc) + E4 / (rc * rc));
        for (let kk = 0; kk < ncoul; kk++) {
          const exp2ear = Math.exp(-2.0 * ea * r);
          const exp2ebr = Math.exp(-2.0 * eb * r);
          fafb[at(kk, inty)] = -exp2ear * (E1 + E3 / r) - exp2ebr * (E2 + E4 / r) - fafbsh - (r - rc) * dfafbsh;
          dfafb[at(kk, inty)] = -(exp2ear * (2.0 * ea * (E1 + E3 / r) + E3 / (r * r)) +
            exp2ebr * (2.0 * eb * (E2 + E4 / r) + E4 / (r * r)) - dfafbsh);
          ddfafb[at(kk, inty)] = -exp2ear * (4.0 * ea * ea * (E1 + E3 / r) + 4.0 * ea * E3 / (r * r) + 2.0 * E3 / (r * r * r)) -
            exp2ebr * (4.0 * eb * eb * (E2 + E4 / r) + 4.0 * eb * E4 / (r * r) + 2.0 * E4 / (r * r * r));
          r += dra;
        }
      }
    }
  }
  for (let i = 0; i < ncoulLim; i++) {
    const r = dra * (i - 1) + drin;
    erpaw[2 * i] = erfcExact(r * alf);
    erpaw[2 * i + 1] = Math.exp(-r * r * alf * alf);
  }

  // van der Waals spline (vrc[i] = rc^(i+1), as the C++ array is filled)
  const vrc = new Float64Array(13);
  vrc[0] = rc;
  for (let i = 1; i < 13; i++) vrc[i] = vrc[i - 1] * vrc[0];
  const cc2 = new Float64Array(nntypes), cc3 = new Float64Array(nntypes);
  const rvdw0 = new Float64Array(nntypes), rvdw1 = new Float64Array(nntypes);
  const rrc = new Float64Array(13);
  for (let ii = 0; ii < n; ii++) {
    for (let jj = ii; jj < n; jj++) {
      const inty = intype[ii * n + jj];
      const p = params[elem3param[(ii * n + jj) * n + jj]];
      if (p.vdwflag > 0) {
        rvdw0[inty] = p.vdwflag === 1 ? p.bigr + p.bigd : p.bigr - p.bigd;
        rvdw1[inty] = p.vsig * 0.950;
        if (rvdw0[inty] > rvdw1[inty]) throw new Error('Error in vdw spline: inner radius > outer radius');
        rrc[0] = rvdw1[inty];
        for (let i = 1; i < 13; i++) rrc[i] = rrc[i - 1] * rrc[0];
        const drcc = rrc[0] - rvdw0[inty];
        const temp6 = 1.0 / rrc[5] - 1.0 / vrc[5] + 6.0 * (rrc[0] - vrc[0]) / vrc[6];
        const temp7 = 6.0 * (1.0 / vrc[6] - 1.0 / rrc[6]);
        const temp12 = 1.0 / rrc[11] - 1.0 / vrc[11] + (rrc[0] - vrc[0]) * 12.0 / vrc[12];
        const temp13 = 12.0 * (1.0 / vrc[12] - 1.0 / rrc[12]);
        let vsigt = p.vsig;
        const vepst = p.veps;
        vsigt = vsigt * vsigt * vsigt * vsigt * vsigt * vsigt;
        const vdwt = vepst * (vsigt * vsigt * temp12 - vsigt * temp6);
        const dvdwt = vepst * (vsigt * vsigt * temp13 - vsigt * temp7);
        cc2[inty] = (3.0 / drcc * vdwt - dvdwt) / drcc;
        cc3[inty] = (vdwt / (drcc * drcc) - cc2[inty]) / drcc;
      }
    }
  }
  for (let ii = 0; ii < n; ii++) {
    for (let jj = ii; jj < n; jj++) {
      const inty = intype[ii * n + jj];
      const p = params[elem3param[(ii * n + jj) * n + jj]];
      let r = drin;
      for (let kk = 0; kk < ncoul; kk++) {
        const r6 = r * r * r * r * r * r;
        const r7 = r6 * r;
        const rf6 = 1.0 / r6 - 1.0 / vrc[5] + (r - vrc[0]) * 6.0 / vrc[6];
        const drf7 = 6.0 * (1.0 / vrc[6] - 1.0 / r7);
        let vsigt = p.vsig;
        const vepst = p.veps;
        vsigt = vsigt * vsigt * vsigt * vsigt * vsigt * vsigt;
        if (p.vdwflag > 0) {
          if (r <= rvdw0[inty]) {
            vvdw[at(kk, inty)] = 0.0;
            vdvdw[at(kk, inty)] = 0.0;
          } else if (r > rvdw0[inty] && r <= rvdw1[inty]) {
            const drcc = r - rvdw0[inty];
            vvdw[at(kk, inty)] = drcc * drcc * (drcc * cc3[inty] + cc2[inty]);
            vdvdw[at(kk, inty)] = drcc * (3.0 * drcc * cc3[inty] + 2.0 * cc2[inty]);
          } else {
            const r12 = r6 * r6;
            const r13 = r6 * r7;
            const rf12 = 1.0 / r12 - 1.0 / vrc[11] + (r - vrc[0]) * 12.0 / vrc[12];
            const drf13 = 12.0 * (1.0 / vrc[12] - 1.0 / r13);
            vvdw[at(kk, inty)] = vepst * (vsigt * vsigt * rf12 - vsigt * rf6);
            vdvdw[at(kk, inty)] = vepst * (vsigt * vsigt * drf13 - vsigt * drf7);
          }
        } else {
          vvdw[at(kk, inty)] = 0.0;
          vdvdw[at(kk, inty)] = 0.0;
        }
        r += dra;
      }
    }
  }
  return { nel: n, nntypes, intype, ncoul, ncoulLim, fafb, dfafb, ddfafb, phin, dphin, afb, dafb, erpaw, vvdw, vdvdw };
};

/** potal_calc: the Wolf-type constants fac11 (1/r^2 force term) and fac11e (self-energy shift) at the Coulomb cutoff. */
export const potalFactors = (rcoul: number, esucon: number): { fac11: number; fac11e: number } => {
  const alf = 0.20;
  const fac11 = (erfcExact(rcoul * alf) / rcoul / rcoul + 2.0 * alf / Math.sqrt(Math.PI) * Math.exp(-alf * alf * rcoul * rcoul) / rcoul) * esucon / rcoul;
  const fac11e = (erfcExact(rcoul * alf) / rcoul) * esucon;
  return { fac11, fac11e };
};

/** vdwaals: van der Waals energy and force/r of one pair from the tables. */
export const vdwaals = (t: Comb3Tables, inty: number, tp: TriPoint, r: number): { eng: number; fforce: number } => {
  const nt = t.nntypes;
  const v = (mr: number): number => t.vvdw[(mr - 1) * nt + inty];
  const dv = (mr: number): number => t.vdvdw[(mr - 1) * nt + inty];
  const eng = 1.0 * (tp.sr1 * v(tp.mr1) + tp.sr2 * v(tp.mr2) + tp.sr3 * v(tp.mr3));
  const fforce = -1.0 / r * (tp.sr1 * dv(tp.mr1) + tp.sr2 * dv(tp.mr2) + tp.sr3 * dv(tp.mr3));
  return { eng, fforce };
};
