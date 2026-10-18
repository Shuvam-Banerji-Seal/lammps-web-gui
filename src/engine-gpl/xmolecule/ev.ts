/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/bond.cpp (Bond::ev_tally), src/angle.cpp (Angle::ev_tally) and src/dihedral.cpp / src/improper.cpp
 * (Dihedral::ev_tally, Improper::ev_tally, the same expressions) at commit b332f28. Copyright (2003) Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The tally helpers of the xmolecule styles: the energy and virial sums of native LAMMPS for newton_bond on, in the
 * same expressions and order, so the per-term virial bookkeeping matches native bit for bit.
 */
import type { BondedCompute } from '../../engine/force/types';

/** Angle::ev_tally: THIRD = 1.0/3.0 (math_const.h). */
const THIRD = 1.0 / 3.0;

/** Bond::ev_tally (bond.cpp) for the pair i, j with delx, dely, delz = x_i - x_j. */
export const evTallyBond = (
  bc: BondedCompute, i: number, j: number, ebond: number, fbond: number, d: readonly number[],
): void => {
  const delx = d[0], dely = d[1], delz = d[2];
  bc.acc.ebond += ebond;
  const v = bc.virial;
  const v0 = delx * delx * fbond, v1 = dely * dely * fbond, v2 = delz * delz * fbond;
  const v3 = delx * dely * fbond, v4 = delx * delz * fbond, v5 = dely * delz * fbond;
  v[0] += v0; v[1] += v1; v[2] += v2; v[3] += v3; v[4] += v4; v[5] += v5;
  if (bc.eatom) { bc.eatom[i] += 0.5 * ebond; bc.eatom[j] += 0.5 * ebond; }
  if (bc.vatom) {
    for (const a of [i, j]) {
      bc.vatom[6 * a] += 0.5 * v0; bc.vatom[6 * a + 1] += 0.5 * v1; bc.vatom[6 * a + 2] += 0.5 * v2;
      bc.vatom[6 * a + 3] += 0.5 * v3; bc.vatom[6 * a + 4] += 0.5 * v4; bc.vatom[6 * a + 5] += 0.5 * v5;
    }
  }
};

/**
 * Angle::ev_tally (angle.cpp) for the angle i-j-k: d1 = x_i - x_j, d2 = x_k - x_j, f1 and f3 the forces on i and k
 * from the native angle loop.
 */
export const evTallyAngle = (
  bc: BondedCompute, i: number, j: number, k: number, eangle: number,
  f1: readonly number[], f3: readonly number[], d1: readonly number[], d2: readonly number[],
): void => {
  bc.acc.eangle += eangle;
  const v = bc.virial;
  const w0 = d1[0] * f1[0] + d2[0] * f3[0];
  const w1 = d1[1] * f1[1] + d2[1] * f3[1];
  const w2 = d1[2] * f1[2] + d2[2] * f3[2];
  const w3 = d1[0] * f1[1] + d2[0] * f3[1];
  const w4 = d1[0] * f1[2] + d2[0] * f3[2];
  const w5 = d1[1] * f1[2] + d2[1] * f3[2];
  v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
  if (bc.eatom) {
    const e = THIRD * eangle;
    bc.eatom[i] += e; bc.eatom[j] += e; bc.eatom[k] += e;
  }
  if (bc.vatom) {
    for (const a of [i, j, k]) {
      bc.vatom[6 * a] += THIRD * w0; bc.vatom[6 * a + 1] += THIRD * w1; bc.vatom[6 * a + 2] += THIRD * w2;
      bc.vatom[6 * a + 3] += THIRD * w3; bc.vatom[6 * a + 4] += THIRD * w4; bc.vatom[6 * a + 5] += THIRD * w5;
    }
  }
};

/**
 * Dihedral::ev_tally / Improper::ev_tally (dihedral.cpp, improper.cpp) for the quadruplet i1..i4: vb1, vb2, vb3 are
 * the bond vectors of the native call, f1, f3, f4 the forces on atoms 1, 3 and 4 used by the virial.
 * kind 'dihed' adds to the dihedral energy and virial, 'imp' to the improper ones.
 */
export const evTallyFour = (
  bc: BondedCompute, atoms: readonly number[], kind: 'dihed' | 'imp', e: number,
  f1: readonly number[], f3: readonly number[], f4: readonly number[],
  vb1: readonly number[], vb2: readonly number[], vb3: readonly number[],
): void => {
  if (kind === 'dihed') bc.acc.edihed += e; else bc.acc.eimp += e;
  const v = bc.virial;
  const w0 = vb1[0] * f1[0] + vb2[0] * f3[0] + (vb3[0] + vb2[0]) * f4[0];
  const w1 = vb1[1] * f1[1] + vb2[1] * f3[1] + (vb3[1] + vb2[1]) * f4[1];
  const w2 = vb1[2] * f1[2] + vb2[2] * f3[2] + (vb3[2] + vb2[2]) * f4[2];
  const w3 = vb1[0] * f1[1] + vb2[0] * f3[1] + (vb3[0] + vb2[0]) * f4[1];
  const w4 = vb1[0] * f1[2] + vb2[0] * f3[2] + (vb3[0] + vb2[0]) * f4[2];
  const w5 = vb1[1] * f1[2] + vb2[1] * f3[2] + (vb3[1] + vb2[1]) * f4[2];
  v[0] += w0; v[1] += w1; v[2] += w2; v[3] += w3; v[4] += w4; v[5] += w5;
  const q = 0.25 * e;
  if (bc.eatom) for (const a of atoms) bc.eatom[a] += q;
  if (bc.vatom) {
    for (const a of atoms) {
      bc.vatom[6 * a] += 0.25 * w0; bc.vatom[6 * a + 1] += 0.25 * w1; bc.vatom[6 * a + 2] += 0.25 * w2;
      bc.vatom[6 * a + 3] += 0.25 * w3; bc.vatom[6 * a + 4] += 0.25 * w4; bc.vatom[6 * a + 5] += 0.25 * w5;
    }
  }
};
