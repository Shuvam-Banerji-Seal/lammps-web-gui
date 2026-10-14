/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_airebo_bc.cpp and pair_airebo_bc.h (PairAIREBObc: bcflag = 1),
 * src/MANYBODY/pair_airebo.cpp (PairAIREBO::Pij_eval, PairAIREBO::bondorder_Pij_cross,
 * PairAIREBO::read_file_extra, the bc patch build of spline_init), and the bcflag branch of
 * PairAIREBO::PijSpline, at commit b332f28. The AIREBO-BC bond-centric P_CC term of
 * Hur and Stuart, J. Chem. Phys. 137, 054102 (2012).
 * Authors credited in the C++ file headers: Ase Henry (MIT); bugfixes and optimizations:
 * Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer (Temple U), Markus Hoehnerbach
 * (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M: Thomas C. O'Connor (JHU) 2014.
 * Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import { PijSpline, newRef, type ReboCore } from './rebo';
import { sp, spbicubicPatchCoeffs, type Ref } from './splines';
import type { SplineTables } from './tables';

/** Number of knots of the bond-centric P_CC grid (PCCf_bc[7][7], doubled coordinate 0..6). */
export const BC_KNOTS = 49;

/**
 * PairAIREBO::read_file_extra for AIREBO-BC: the count N and N triples "mC mH value" (mC = 2*N_C,
 * mH = 2*N_H, integer 0..6) that follow the Tij section. `rest` holds the value lines after the
 * standard data, one entry per line. Knots not listed stay zero.
 */
export const readBcKnots = (T: SplineTables, rest: string[], fileName: string): void => {
  const v = rest.join(' ').split(/\s+/).filter((s) => s !== '');
  const bad = (): never => {
    throw new StyleError(`reading derived-class extra parameters section in ${fileName} file: unexpected end of file or bad number`);
  };
  if (v.length === 0) bad();
  const n = Number(v[0]);
  if (!Number.isInteger(n)) bad();
  if (n < 0 || n > BC_KNOTS) throw new StyleError(`AIREBO-BC pCC knot count out of range: ${n}`);
  if (v.length < 1 + 3 * n) bad();
  T.bcKnots.fill(0.0);
  for (let k = 0; k < n; k++) {
    const mC = Math.round(Number(v[1 + 3 * k]));
    const mH = Math.round(Number(v[2 + 3 * k]));
    const val = Number(v[3 + 3 * k]);
    if (!Number.isFinite(val)) bad();
    if (mC < 0 || mC > 6 || mH < 0 || mH > 6) {
      throw new StyleError(`AIREBO-BC pCC knot index out of range: ${mC} ${mH}`);
    }
    T.bcKnots[mC * 7 + mH] = val;
  }
};

/**
 * The bond-centric bicubic P_CC patches on the half-integer grid (the bc block of spline_init):
 * 6x6 cells, cell (mC, mH) spans [mC, mC+1] x [mH, mH+1] in doubled coordinates, knot derivatives zero.
 */
export const splineInitBc = (T: SplineTables): void => {
  const y = new Float64Array(4), y1 = new Float64Array(4), y2 = new Float64Array(4);
  for (let mH = 0; mH < 6; mH++) {
    for (let mC = 0; mC < 6; mC++) {
      y[0] = T.bcKnots[mC * 7 + mH];
      y[1] = T.bcKnots[mC * 7 + mH + 1];
      y[2] = T.bcKnots[(mC + 1) * 7 + mH];
      y[3] = T.bcKnots[(mC + 1) * 7 + mH + 1];
      spbicubicPatchCoeffs(mC, mC + 1, mH, mH + 1, y, y1, y2, T.pCCbc, (mC * 6 + mH) * 16);
    }
  }
};

/**
 * PairAIREBO::Pij_eval: the P term of the i-j bond. For AIREBO-BC the C-C term is evaluated at the
 * bond-averaged coordination numbers 1/2 (N_ij + N_ji); every other case is the stock PijSpline.
 * thisC/thisH are the counts of the side whose P is formed, othC/othH those of the opposite side.
 */
export const pijEval = (
  core: ReboCore, thisC: number, thisH: number, othC: number, othH: number,
  typei: number, typej: number, dN2: Float64Array,
): number => {
  if (core.bcflag && typei === 0 && typej === 0) {
    return PijSpline(core, 0.5 * (thisC + othC), 0.5 * (thisH + othH), typei, typej, dN2);
  }
  return PijSpline(core, thisC, thisH, typei, typej, dN2);
};

const v3 = (): Float64Array => new Float64Array(3);
const rik = v3(), rjl = v3();
const rS: Ref = newRef();

/**
 * PairAIREBO::bondorder_Pij_cross: the bond-centric P cross forces. For a bond-averaged P the
 * pji term acts on the REBO neighbors k of i and the pij term on the neighbors l of j; both are
 * central forces along the bond carried by the cutoff derivative. Nothing for stock P.
 */
export const bondorderPijCross = (
  core: ReboCore, i: number, j: number, itype: number, jtype: number, VA: number,
  tmppij: number, tmppji: number, dN2PIJ: Float64Array, dN2PJI: Float64Array,
): void => {
  if (!(core.bcflag && itype === 0 && jtype === 0)) return;
  const { P, x, f, mt, rstart, rlist } = core;
  const rcmin = P.rcmin, rcmax = P.rcmax;
  // pji term: P-force on i's neighbors k
  for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
    const atomk = rlist[kk];
    if (atomk === j) continue;
    const ktype = mt[atomk];
    rik[0] = x[3 * i] - x[3 * atomk];
    rik[1] = x[3 * i + 1] - x[3 * atomk + 1];
    rik[2] = x[3 * i + 2] - x[3 * atomk + 2];
    const rikmag = Math.sqrt((rik[0] * rik[0]) + (rik[1] * rik[1]) + (rik[2] * rik[2]));
    sp(rikmag, rcmin[itype * 2 + ktype], rcmax[itype * 2 + ktype], rS);
    const dwik = rS.v;
    const tmp2 = VA * 0.5 * (tmppji * dN2PJI[ktype] * dwik) / rikmag;
    f[3 * i] -= tmp2 * rik[0];
    f[3 * i + 1] -= tmp2 * rik[1];
    f[3 * i + 2] -= tmp2 * rik[2];
    f[3 * atomk] += tmp2 * rik[0];
    f[3 * atomk + 1] += tmp2 * rik[1];
    f[3 * atomk + 2] += tmp2 * rik[2];
  }
  // pij term: P-force on j's neighbors l
  for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
    const atoml = rlist[ll];
    if (atoml === i) continue;
    const ltype = mt[atoml];
    rjl[0] = x[3 * j] - x[3 * atoml];
    rjl[1] = x[3 * j + 1] - x[3 * atoml + 1];
    rjl[2] = x[3 * j + 2] - x[3 * atoml + 2];
    const rjlmag = Math.sqrt((rjl[0] * rjl[0]) + (rjl[1] * rjl[1]) + (rjl[2] * rjl[2]));
    sp(rjlmag, rcmin[jtype * 2 + ltype], rcmax[jtype * 2 + ltype], rS);
    const dwjl = rS.v;
    const tmp2 = VA * 0.5 * (tmppij * dN2PIJ[ltype] * dwjl) / rjlmag;
    f[3 * j] -= tmp2 * rjl[0];
    f[3 * j + 1] -= tmp2 * rjl[1];
    f[3 * j + 2] -= tmp2 * rjl[2];
    f[3 * atoml] += tmp2 * rjl[0];
    f[3 * atoml + 1] += tmp2 * rjl[1];
    f[3 * atoml + 2] += tmp2 * rjl[2];
  }
};
