/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_born_matrix.cpp (ComputeBornMatrix::compute_vector with the numdiff branch,
 * compute_numdiff, displace_atoms, restore_atoms, update_virial, virial_addon) at commit b332f28. Copyright Sandia
 * Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The numdiff method of compute born/matrix (compute_born_matrix.ts calls it for the numdiff keyword). For each of the
 * six strain directions the owned and ghost atoms are displaced by +delta and -delta of the strain about the centre
 * of the unchanged box, with the neighbor lists of the step, as native does; the force field is evaluated and the
 * pressure of the virial-ID compute is read. The difference divided by -2 delta gives the second derivatives. The
 * pressure keeps the undeformed volume, and the virial_addon terms are added at the undeformed state. The result is
 * scaled by the volume and 1/nktv2p to energy units.
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
  // the step's force evaluation, its lists and its ghosts (native: the compute is invoked after it)
  sys.forces();
  const s = sys.state, nb = sys.nb, geom = sys.geom, ff = sys.ff;
  const box = s.box;
  const vol0 = geom.volume(3);
  const nktv2p = s.units.nktv2p;
  const fixedpoint = [0, 1, 2].map((k) => 0.5 * (box.lo[k] + box.hi[k]));
  const n = s.n, nall = nb.nall;
  const xa = nb.xall;
  // native compute_numdiff: "store copy of current forces for owned and ghost atoms" (temp_x, temp_f); the force
  // field's accumulators and per-atom outputs are kept too, so the step's values are as before afterwards
  const tempX = xa.slice(0, 3 * nall);
  const x0 = s.x.slice(0, 3 * n);
  const f0 = s.f.slice(0, 3 * n);
  const t0 = s.torque ? s.torque.slice(0, 3 * n) : null;
  const acc = ff.acc;
  const accNum = [acc.evdwl, acc.ecoul, acc.elong, acc.ebond, acc.eangle, acc.edihed, acc.eimp];
  const accVec = [acc.virial, acc.vbond, acc.vangle, acc.vdihed, acc.vimp, acc.vlong].map((v) => v.slice());
  const { eatom, vatom, eatomTerm, vatomTerm, cvatomTerm } = ff;

  // native displace_atoms: the owned and ghost atoms of the unchanged box move by the strain of direction idir about
  // the fixed point, "x[i][k] = temp_x[i][k] + numdelta * magnitude * (temp_x[i][l] - fixedpoint[l])" for an axial
  // strain and half of it on both components for a symmetric shear; the neighbor lists are kept
  const displace = (idir: number, magnitude: number): void => {
    const [k, l] = DIRLIST[idir];
    if (l === k) {
      for (let i = 0; i < nall; i++) xa[3 * i + k] = tempX[3 * i + k] + delta * magnitude * (tempX[3 * i + l] - fixedpoint[l]);
    } else {
      for (let i = 0; i < nall; i++) {
        xa[3 * i + k] = tempX[3 * i + k] + 0.5 * delta * magnitude * (tempX[3 * i + l] - fixedpoint[l]);
        xa[3 * i + l] = tempX[3 * i + l] + 0.5 * delta * magnitude * (tempX[3 * i + k] - fixedpoint[k]);
      }
    }
    // the owned atoms' own coordinates (the kspace term reads them)
    s.x.set(xa.subarray(0, 3 * n));
  };
  const restoreAtoms = (): void => {
    xa.set(tempX);
    s.x.set(x0);
  };
  // native update_virial: the pair, bonded and kspace terms with the virial only, then the pressure compute's vector
  // (its volume is the undeformed box's: native does not change the box)
  const updateVirial = (): Float64Array => {
    ff.compute(s, nb, geom, {});
    return c.vectorAt(c.rawVirial(), vol0);
  };

  const values = new Float64Array(21);
  try {
    for (let idir = 0; idir < 6; idir++) {
      // forward
      displace(idir, 1.0);
      let sig = updateVirial();
      for (let jdir = 0; jdir < 6; jdir++) values[REVALBE[idir][jdir]] = sig[VIRIAL_OF_VOIGT[jdir]];
      restoreAtoms();
      // backward
      displace(idir, -1.0);
      sig = updateVirial();
      for (let jdir = 0; jdir < 6; jdir++) values[REVALBE[idir][jdir]] -= sig[VIRIAL_OF_VOIGT[jdir]];
      restoreAtoms();
    }
    for (let m = 0; m < 21; m++) values[m] *= -0.5 / delta;

    // native: "recompute virial so all virial and energy contributions are as before", then virial_addon
    const sigv = updateVirial();
    virialAddon(values, sigv);
  } finally {
    restoreAtoms();
    // native: "restore original forces for owned and ghost atoms"; the accumulators and per-atom outputs as well
    s.f.set(f0);
    if (t0) s.torque!.set(t0);
    [acc.evdwl, acc.ecoul, acc.elong, acc.ebond, acc.eangle, acc.edihed, acc.eimp] = accNum;
    [acc.virial, acc.vbond, acc.vangle, acc.vdihed, acc.vimp, acc.vlong].forEach((v, m) => v.set(accVec[m]));
    ff.eatom = eatom; ff.vatom = vatom; ff.eatomTerm = eatomTerm; ff.vatomTerm = vatomTerm; ff.cvatomTerm = cvatomTerm;
  }
  for (let m = 0; m < 21; m++) values[m] *= (1 / nktv2p) * vol0;
  return values;
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
