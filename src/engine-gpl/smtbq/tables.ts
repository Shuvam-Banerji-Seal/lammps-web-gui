/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SMTBQ/pair_smtbq.cpp (PairSMTBQ::tabqeq, tabsm, and the cutting functions fcoupure, fcoupured,
 * fcoup2, Primfcoup2, Intfcoup2) at commit b332f28. Authors credited in the C++ file headers: N. Salles,
 * O. Politano, E. Amzallag, R. Tetot (SMTBQ), E. Maras (Aalto University) and Julian Gale (gammas, Imperial
 * College). Copyright (2015) Universite de Bourgogne and Sandia Corporation (LAMMPS). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * The tables are the C++ tables, indexed [k * columns + m]. In tabqeq the Slater overlap gam and its derivative
 * dgam are needed only up to the first r where dij < 0.01 (the cutting point rc): past it dij and ddij are
 * replaced by the cut-off polynomial (r > rc), so the C++ values there do not depend on gam. The overlap is
 * therefore evaluated only while no cutting point has been found (the same values, fewer integral calls).
 */
import { erfc } from './erfc';
import { SlaterIntegrals } from './slater';
import type { SmtbqFile } from './potfile';

/** MathConst::MY_PIS (sqrt(pi)). */
const MY_PIS = 1.77245385090551602729;

/** PairSMTBQ::fcoupure: the cut-off function of the second-moment terms (1 below r1, 0 above r2). */
export function fcoupure(r: number, dc1: number, dc2: number): number {
  if (r < dc1) return 1;
  if (r > dc2) return 0;
  const ddc = dc2 - dc1;
  const x = r - dc2;
  const a3 = -10 / (ddc * ddc * ddc);
  const a4 = -15 / (ddc * ddc * ddc * ddc);
  const a5 = -6 / (ddc * ddc * ddc * ddc * ddc);
  return x * x * x * (a3 + x * (a4 + x * a5));
}

/** PairSMTBQ::fcoupured: derivative of fcoupure (zero outside (dc1, dc2)). */
export function fcoupured(r: number, dc1: number, dc2: number): number {
  if (r > dc1 && r < dc2) {
    const ddc = dc2 - dc1;
    const x = r - dc2;
    const a3 = -10 / (ddc * ddc * ddc);
    const a4 = -15 / (ddc * ddc * ddc * ddc);
    const a5 = -6 / (ddc * ddc * ddc * ddc * ddc);
    return x * x * (3 * a3 + x * (4 * a4 + 5 * x * a5));
  }
  return 0;
}

/** PairSMTBQ::fcoup2: the cut-off of the force integral (charge forces of the oxide). */
export function fcoup2(c: number, x: number, delta: number): number {
  if (c < x - delta) return 1;
  if (c > x + delta) return 0;
  const dc = c - x - delta;
  return dc * dc * (3 * delta + dc) / (4 * delta * delta * delta);
}

/** PairSMTBQ::Primfcoup2: primitive of fcoup2. */
export function primfcoup2(c: number, x: number, delta: number): number {
  return (c * (c * c * c - 4 * c * c * x - 4 * (x - 2 * delta) * (x + delta) * (x + delta) +
               6 * c * (x * x - delta * delta))) / (16 * delta * delta * delta);
}

/** PairSMTBQ::Intfcoup2: integral of fcoup2 between x and c. */
export function intfcoup2(c: number, x: number, delta: number): number {
  if (c < x - delta) {
    return c - x + delta + primfcoup2(x - delta, x, delta) - primfcoup2(x, x, delta);
  } else if (c > x + delta) {
    return primfcoup2(x + delta, x, delta) - primfcoup2(x, x, delta);
  }
  return primfcoup2(c, x, delta) - primfcoup2(x, x, delta);
}

/** The tables of PairSMTBQ (tabqeq and tabsm). */
export interface SmtbqTables {
  /** Number of table entries (kmax); the arrays hold kmax + 5 rows. */
  kmax: number;
  rows: number;
  ds: number;
  /** Self energy of the Coulomb charge (Vself). */
  Vself: number;
  /** Columns of fafb / dfafb: nntype = (n+1)n/2 element pairs. */
  nntype: number;
  /** coultype[i * ntypes + j]: column of the element pair. */
  coultype: Int32Array;
  fafb: Float64Array;
  dfafb: Float64Array;
  fafbOxOxSurf: Float64Array;
  dfafbOxOxSurf: Float64Array;
  fafbTiOxSurf: Float64Array;
  dfafbTiOxSurf: Float64Array;
  fafbOxOxBB: Float64Array;
  dfafbOxOxBB: Float64Array;
  fafbTiOxBB: Float64Array;
  dfafbTiOxBB: Float64Array;
  /** tabsm arrays, [k * (maxintsm + 1) + sm]. */
  tabsmb: Float64Array;
  dtabsmb: Float64Array;
  tabsmr: Float64Array;
  dtabsmr: Float64Array;
  /** Columns of the second-moment tables (maxintsm + 1). */
  smCols: number;
}

/**
 * The overlap-table loop of tabqeq (one table: the bulk table or one of the two oxygen tables): the cut-off
 * search over k, shared by the three C++ blocks. store(k, value, dvalue) receives fafb-type and dfafb-type values.
 */
function cutTable(
  slater: SlaterIntegrals, na: number, nb: number, za: number, zb: number, cutmax: number, rcoupe: number,
  ds: number, kmax: number, potqn: Float64Array, dpotqn: Float64Array,
  store: (k: number, v: number, dv: number) => void,
): void {
  let rc = cutmax;
  let ii = 0;
  let nang = 5.0;
  let aCoeff = 0.0;
  let bCoeff = 0.0;
  for (let k = 0; k < kmax + 5; k++) {
    let gam = 0.0, dgam = 0.0;
    let dij = 0.0;
    const s = k * ds;
    let r = Math.sqrt(s);
    if (k === 0) r = 10e-30;
    if (ii === 0) {
      const g = slater.gammaValues(na, nb, za, zb, r);
      gam = g.gam;
      dgam = g.dgam;
    }
    dij = 14.4 * (1.0 / r - gam);
    let ddij = 14.4 * (-1.0 / (r * r) - dgam);

    // Cutting function
    if (dij < 0.01 && ii === 0) {
      // native sets ii = 2 and then ii = 1 before it is read; only the second value matters
      rc = r;
      ii = 1;
      if (rc + nang > rcoupe) nang = rcoupe - rc;
      bCoeff = (2 * dij + ddij * nang) / (dij * nang);
      aCoeff = dij * Math.exp(-bCoeff * rc) / (nang * nang);
    }
    if (r > rc) {
      const d = r - rc - nang;
      dij = aCoeff * (d * d) * Math.exp(bCoeff * r);
      ddij = aCoeff * d * (2 + bCoeff * d) * Math.exp(bCoeff * r);
    }
    if (r > rc + nang) {
      dij = 0.0;
      ddij = 0.0;
    }
    store(k, potqn[k] - dij, dpotqn[k] - ddij / r);
  }
}

/** PairSMTBQ::tabqeq and PairSMTBQ::tabsm, for the parameters read from the file. */
export function buildTables(file: SmtbqFile, slater: SlaterIntegrals): SmtbqTables {
  const n = file.ntypes;
  const params = file.params;
  const cutmax = file.cutmax;
  const rmin = file.rmin;
  const dr = file.dr;
  const kmax = Math.trunc((cutmax * cutmax) / (2.0 * dr * rmin));
  const ds = (cutmax * cutmax) / kmax;
  const rows = kmax + 5;
  const nntype = Math.trunc(((n + 1) * n) / 2);
  const alf = 0.3;

  // interaction numbers: 0-0 = 0, 1-1 = 1, 0-1 = 1-0 = 2 (C++ numbering: diagonal first, then off-diagonal)
  const coultype = new Int32Array(n * n);
  {
    let m = 0;
    let k = n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (j === i) { coultype[i * n + j] = m; m += 1; }
        else if (j > i) { coultype[i * n + j] = k; k += 1; }
        else coultype[i * n + j] = coultype[j * n + i];
      }
    }
  }

  // Coulomb 1/r energy and force look-up (tabqeq)
  const mu = erfc(alf * cutmax) / cutmax;
  const potqn = new Float64Array(rows);
  const dpotqn = new Float64Array(rows);
  for (let k = 0; k < rows; k++) {
    const s = k * ds;
    let r = Math.sqrt(s);
    if (k === 0) r = 10e-30;
    potqn[k] = 14.4 * (erfc(alf * r) / r - mu);
    dpotqn[k] = -14.4 * ((erfc(alf * r) / (r * r) + 2.0 * alf / MY_PIS / r * Math.exp(-alf * alf * r * r)) / r);
  }
  const Vself = -14.4 * (alf / MY_PIS + mu * 0.5);

  const fafb = new Float64Array(rows * nntype);
  const dfafb = new Float64Array(rows * nntype);
  const fafbOxOxSurf = new Float64Array(rows), dfafbOxOxSurf = new Float64Array(rows);
  const fafbTiOxSurf = new Float64Array(rows), dfafbTiOxSurf = new Float64Array(rows);
  const fafbOxOxBB = new Float64Array(rows), dfafbOxOxBB = new Float64Array(rows);
  const fafbTiOxBB = new Float64Array(rows), dfafbTiOxBB = new Float64Array(rows);

  const rcoupe = cutmax;
  const isO = params.map((p) => p.nom === 'O');
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const m = coultype[i * n + j];
      const na = params[i].ne;
      const nb = params[j].ne;
      // bulk table
      cutTable(slater, na, nb, params[i].dzeta, params[j].dzeta, cutmax, rcoupe, ds, kmax, potqn, dpotqn,
        (k, v, dv) => {
          fafb[k * nntype + m] = v;
          if (k === 1) fafb[0 * nntype + m] = fafb[1 * nntype + m];
          dfafb[k * nntype + m] = dv;
        });
      if (!isO[i] && !isO[j]) continue;
      // oxygen tables: the O orbital radius replaces R (surface, then bulk-to-bulk)
      const zOf = (e: number, r: number): number => (2.0 * params[e].ne + 1.0) / (4.0 * r);
      const zaS = isO[i] ? zOf(i, file.ROxSurf) : params[i].dzeta;
      const zbS = isO[j] ? zOf(j, file.ROxSurf) : params[j].dzeta;
      const bothO = isO[i] && isO[j];
      cutTable(slater, na, nb, zaS, zbS, cutmax, rcoupe, ds, kmax, potqn, dpotqn, (k, v, dv) => {
        if (bothO) {
          fafbOxOxSurf[k] = v;
          if (k === 1) fafbOxOxSurf[0] = fafbOxOxSurf[k];
          dfafbOxOxSurf[k] = dv;
        } else {
          fafbTiOxSurf[k] = v;
          if (k === 1) fafbTiOxSurf[0] = fafbTiOxSurf[k];
          dfafbTiOxSurf[k] = dv;
        }
      });
      const zaB = isO[i] ? zOf(i, file.ROxBB) : params[i].dzeta;
      const zbB = isO[j] ? zOf(j, file.ROxBB) : params[j].dzeta;
      cutTable(slater, na, nb, zaB, zbB, cutmax, rcoupe, ds, kmax, potqn, dpotqn, (k, v, dv) => {
        if (bothO) {
          fafbOxOxBB[k] = v;
          if (k === 1) fafbOxOxBB[0] = fafbOxOxBB[k];
          dfafbOxOxBB[k] = dv;
        } else {
          fafbTiOxBB[k] = v;
          if (k === 1) fafbTiOxBB[0] = fafbTiOxBB[k];
          dfafbTiOxBB[k] = dv;
        }
      });
    }
  }

  // second-moment tables (tabsm); the column is the interaction's intsm number
  const smCols = file.maxintsm + 1;
  const tabsmb = new Float64Array(rows * smCols), dtabsmb = new Float64Array(rows * smCols);
  const tabsmr = new Float64Array(rows * smCols), dtabsmr = new Float64Array(rows * smCols);
  for (let m = 0; m <= file.maxintparam; m++) {
    const it = file.intparams[m];
    if (it.intsm === 0) continue;
    const rc1 = it.dc1;
    const rc2 = it.dc2;
    const A = it.a;
    const p = it.p;
    const Ksi = it.ksi;
    const q = it.q;
    const rzero = it.r0;
    const sm = it.intsm;
    for (let k = 0; k < kmax; k++) {
      const s = k * ds;
      let r = Math.sqrt(s);
      if (k === 0) r = 10e-30;
      const tmpb = Math.exp(-2.0 * q * (r / rzero - 1.0));
      const tmpr = Math.exp(-p * (r / rzero - 1.0));
      const at = k * smCols + sm;
      if (r <= rc1) {
        tabsmb[at] = Ksi * Ksi * tmpb;
        tabsmr[at] = A * tmpr;
        dtabsmb[at] = -2.0 * Ksi * Ksi * q / rzero * tmpb / r;
        dtabsmr[at] = -A * p / rzero * tmpr / r;
      } else if (r > rc1 && r <= rc2) {
        // C++ indexes intparams by the interaction's intsm number here (not by m)
        const sI = file.intparams[sm];
        const fcv = fcoupure(r, sI.dc1, sI.dc2);
        tabsmb[at] = fcv * Ksi * Ksi * tmpb;
        tabsmr[at] = fcv * A * tmpr;
        const fcdv = fcoupured(r, sI.dc1, sI.dc2);
        dtabsmb[at] = (fcv * (-2.0 * Ksi * Ksi * q / rzero * tmpb) + fcdv * Ksi * Ksi * tmpb) / r;
        dtabsmr[at] = (fcv * (-A * p / rzero * tmpr) + fcdv * A * tmpr) / r;
      } else {
        tabsmb[at] = 0.0;
        tabsmr[at] = 0.0;
        dtabsmb[at] = 0.0;
        dtabsmr[at] = 0.0;
      }
    }
  }

  return {
    kmax, rows, ds, Vself, nntype, coultype, fafb, dfafb, fafbOxOxSurf, dfafbOxOxSurf, fafbTiOxSurf,
    dfafbTiOxSurf, fafbOxOxBB, dfafbOxOxBB, fafbTiOxBB, dfafbTiOxBB, tabsmb, dtabsmb, tabsmr, dtabsmr, smCols,
  };
}
