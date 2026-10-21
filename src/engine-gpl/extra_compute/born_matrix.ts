/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_born_matrix.cpp (ComputeBornMatrix::compute_vector with the numdiff branch,
 * compute_numdiff, displace_atoms, restore_atoms, update_virial, virial_addon) at commit b332f28. Copyright Sandia
 * Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The numdiff method of compute born/matrix (compute_born_matrix.ts calls it for the numdiff keyword). For each of the six strain directions the cell is strained by +delta and
 * -delta about its centre; the atoms follow the same affine map (native displaces the owned and ghost atoms of its
 * fixed box, which is the same geometry as the strained periodic cell), the full force field is evaluated, and the
 * pressure of the virial-ID compute is read. The difference divided by -2 delta gives the second derivatives. The
 * pressure keeps the undeformed volume, as native does (its box is not changed), and the virial_addon terms are
 * added at the undeformed state. The result is scaled by the volume and 1/nktv2p to energy units.
 */
import type { System } from '../../engine/system';
import { ComputePressure } from '../../engine/compute/pressure';
import { StyleError } from '../../engine/force/types';

/** The 21 values of the Born matrix in energy units, for delta and the pressure compute virialId. */
export type BornNumdiffHook = (sys: System, id: string, delta: number, virialId: string) => Float64Array;

/** Native dirlist: the (k, l) index pair of each Voigt strain direction (xx yy zz yz xz xy). */
const DIRLIST: readonly (readonly [number, number])[] = [[0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1]];
/** Native virialVtoV: the LAMMPS virial index (xx yy zz xy xz yz) of each Voigt stress component. */
const VIRIAL_OF_VOIGT = [0, 1, 2, 5, 4, 3];
/** Native C_albe order of the 21 outputs, as Voigt index pairs. */
const ALBE: readonly (readonly [number, number])[] = [
  [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5],
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 2], [1, 3], [1, 4], [1, 5],
  [2, 3], [2, 4], [2, 5],
  [3, 4], [3, 5],
  [4, 5],
];

/** native revalbe: the output index of the Voigt pair (a, b). */
const REVALBE: number[][] = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
ALBE.forEach(([a, b], m) => { REVALBE[a][b] = m; REVALBE[b][a] = m; });

/**
 * The 21 Born values in energy units (native: values_global after `*= inv_nktv2p * volume`). Throws naming the compute
 * when the virial-ID is not a pressure compute (native: "does not compute pressure").
 */
export const bornMatrixNumdiff: BornNumdiffHook = (sys: System, id: string, delta: number, virialId: string): Float64Array => {
  const c = sys.compute(virialId);
  if (!(c instanceof ComputePressure)) throw new StyleError(`compute ${id} born/matrix: compute ${virialId} is not a pressure compute`);
  const s = sys.state;
  const box = s.box;
  const vol0 = sys.geom.volume(3);
  const nktv2p = s.units.nktv2p;
  const lo0 = [...box.lo] as [number, number, number];
  const hi0 = [...box.hi] as [number, number, number];
  const tilt0 = [...box.tilt] as [number, number, number];
  const minLo0 = [...box.minLo] as [number, number, number];
  const minHi0 = [...box.minHi] as [number, number, number];
  const triclinic0 = box.triclinic;
  // native fixedpoint = 0.5 (boxlo + boxhi)
  const fp = [0, 1, 2].map((k) => 0.5 * (lo0[k] + hi0[k]));
  const x0 = s.x.slice();
  const image0 = s.image.slice();
  const n = s.n;

  const restore = (): void => {
    s.x.set(x0);
    s.image.set(image0);
    box.triclinic = triclinic0;
    sys.setBox(lo0, hi0, tilt0, 0);
    box.minLo = [...minLo0];
    box.minHi = [...minHi0];
    sys.bump();
  };

  // Strain direction idir with magnitude mag: D is the symmetric strain (axial D_kk = delta mag; shear
  // D_kl = D_lk = delta mag / 2, as native's displace_atoms), x -> x + D (x - fp). Native keeps the box and displaces
  // the periodic images with their atoms, so the periodic cell is M = (I + D) H0, which is not in LAMMPS's upper
  // triangular form for a shear. The strained system is rotated by Q from M = Q R (R upper triangular, positive
  // diagonal): positions y = Q^T (x' - x'_lo) and the cell R. The virial is rotated back with the same Q (the
  // physics is invariant under the rotation). Returns Q (row-major, Q[row][col]).
  const deform = (idir: number, mag: number): number[][] => {
    const [k, l] = DIRLIST[idir];
    const D = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    if (k === l) D[k][k] = delta * mag;
    else { D[k][l] = 0.5 * delta * mag; D[l][k] = 0.5 * delta * mag; }
    const map = (x: readonly number[]): number[] => [0, 1, 2].map((r) => x[r] + D[r][0] * (x[0] - fp[0]) + D[r][1] * (x[1] - fp[1]) + D[r][2] * (x[2] - fp[2]));
    const flo = map(lo0);
    const H0 = [[hi0[0] - lo0[0], tilt0[0], tilt0[1]], [0, hi0[1] - lo0[1], tilt0[2]], [0, 0, hi0[2] - lo0[2]]];
    const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let r = 0; r < 3; r++) for (let cc = 0; cc < 3; cc++) {
      for (let q = 0; q < 3; q++) M[r][cc] += (r === q ? 1 : 0) * H0[q][cc] + D[r][q] * H0[q][cc];
    }
    const { Q, R } = qr3(M);
    for (let i = 0; i < n; i++) {
      const fx = map([x0[3 * i], x0[3 * i + 1], x0[3 * i + 2]]);
      const dd = [fx[0] - flo[0], fx[1] - flo[1], fx[2] - flo[2]];
      for (let r = 0; r < 3; r++) s.x[3 * i + r] = Q[0][r] * dd[0] + Q[1][r] * dd[1] + Q[2][r] * dd[2];
    }
    const lo: [number, number, number] = [0, 0, 0];
    const hi: [number, number, number] = [R[0][0], R[1][1], R[2][2]];
    const tilt: [number, number, number] = [R[0][1], R[0][2], R[1][2]];
    // the geometry takes the triclinic flag when setBox rebuilds it
    box.triclinic = triclinic0 || tilt.some((t) => t !== 0);
    sys.setBox(lo, hi, tilt, 0);
    return Q;
  };

  // the virial of the strained (rotated) state, rotated back to the lab frame, with the undeformed volume (native's
  // box is not changed); Q = null for the undeformed state
  const strainedVirial = (Q: number[][] | null): Float64Array => {
    sys.bump();
    const w = c.rawVirial();
    if (!Q) return c.vectorAt(w, vol0);
    // symmetric tensor from Voigt (xx yy zz xy xz yz), rotated: W = Q W' Q^T
    const Wp = [[w[0], w[3], w[4]], [w[3], w[1], w[5]], [w[4], w[5], w[2]]];
    const W = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      let v = 0;
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) v += Q[a][p] * Wp[p][q] * Q[b][q];
      W[a][b] = v;
    }
    return c.vectorAt(Float64Array.from([W[0][0], W[1][1], W[2][2], W[0][1], W[0][2], W[1][2]]), vol0);
  };

  const values = new Float64Array(21);
  try {
    for (let idir = 0; idir < 6; idir++) {
      // forward
      let sig = strainedVirial(deform(idir, 1.0));
      for (let jdir = 0; jdir < 6; jdir++) values[REVALBE[idir][jdir]] = sig[VIRIAL_OF_VOIGT[jdir]];
      restore();
      // backward
      sig = strainedVirial(deform(idir, -1.0));
      for (let jdir = 0; jdir < 6; jdir++) values[REVALBE[idir][jdir]] -= sig[VIRIAL_OF_VOIGT[jdir]];
      restore();
    }
    for (let m = 0; m < 21; m++) values[m] *= -0.5 / delta;

    // recompute the virial at the undeformed state; the addon terms use it (native virial_addon)
    const sigv = strainedVirial(null);
    virialAddon(values, sigv);
  } finally {
    restore();
  }
  for (let m = 0; m < 21; m++) values[m] *= (1 / nktv2p) * vol0;
  return values;
};

/** QR of a 3x3 matrix with positive diagonal R: M = Q R, Q orthogonal with det +1 (Gram-Schmidt on the columns). */
const qr3 = (M: number[][]): { Q: number[][]; R: number[][] } => {
  const col = (c: number): number[] => [M[0][c], M[1][c], M[2][c]];
  const dot = (a: number[], b: number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const a = col(0), b = col(1), c = col(2);
  const na = Math.sqrt(dot(a, a));
  const q1 = a.map((v) => v / na);
  const r01 = dot(q1, b);
  const v2 = [b[0] - r01 * q1[0], b[1] - r01 * q1[1], b[2] - r01 * q1[2]];
  const r11 = Math.sqrt(dot(v2, v2));
  const q2 = v2.map((v) => v / r11);
  const q3 = [q1[1] * q2[2] - q1[2] * q2[1], q1[2] * q2[0] - q1[0] * q2[2], q1[0] * q2[1] - q1[1] * q2[0]];
  const R = [[na, r01, dot(q1, c)], [0, r11, dot(q2, c)], [0, 0, dot(q3, c)]];
  const Q = [[q1[0], q2[0], q3[0]], [q1[1], q2[1], q3[1]], [q1[2], q2[2], q3[2]]];
  return { Q, R };
};

/**
 * Native virial_addon (src/EXTRA-COMPUTE/compute_born_matrix.cpp, b332f28): the symmetric shear-strain fields of the
 * numdiff give the Born matrix except for the virial stress terms, which are added from the undeformed sigma.
 * sigv is in LAMMPS virial order (xx yy zz xy xz yz).
 */
const virialAddon = (values: Float64Array, sigv: Float64Array): void => {
  values[0] += 2.0 * sigv[0];
  values[1] += 2.0 * sigv[1];
  values[2] += 2.0 * sigv[2];
  values[3] += 0.5 * (sigv[1] + sigv[2]);
  values[4] += 0.5 * (sigv[0] + sigv[2]);
  values[5] += 0.5 * (sigv[0] + sigv[1]);
  values[9] += sigv[4];
  values[10] += sigv[3];
  values[12] += sigv[5];
  values[14] += sigv[3];
  values[15] += sigv[5];
  values[16] += sigv[4];
  values[18] += 0.5 * sigv[3];
  values[19] += 0.5 * sigv[4];
  values[20] += 0.5 * sigv[5];
};
