/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/boundary_correction.cpp and boundary_correction.h, src/ELECTRODE/slab_dipole.cpp (the EW3DC
 * dipole correction) and src/ELECTRODE/slab_2d.cpp (the EW2D correction) at commit b332f28. ELECTRODE contributing
 * authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meissner (TUHH). Copyright Sandia Corporation (GNU
 * General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { SimState } from '../../engine/types';
import { MY_2PI, MY_4PI, MY_PIS, erfAccurate } from './electrode_math';

/** Quantities the corrections need from the k-space style (single process: all atoms are local). */
export interface BoundaryEnv {
  /** Volume with the slab extension (get_volume() of boundary_correction.cpp). */
  volume: number;
  /** zprd * slab_volfactor. */
  zprdSlab: number;
  /** xprd * yprd (Slab2d area). */
  area: number;
  gEwald: number;
  /** force->qqrd2e. */
  qqrd2e: number;
}

/** True if atom i belongs to the source group (invert flips the test, as in the ELECTRODE sources). */
export const inGroup = (mask: number, bit: number, invert: boolean): boolean => ((mask & bit) !== 0) !== invert;

/** boundary_correction.h: the base class; a dummy correction for systems without a slab or wire correction. */
export abstract class BoundaryCorrection {
  /** Energy and force of the correction; adds forces to f and returns the energy (qqrd2e units). */
  abstract computeCorr(s: SimState, f: Float64Array, qsum: number, env: BoundaryEnv): number;
  vectorCorr(_s: SimState, _vec: Float64Array, _sensorBit: number, _sourceBit: number, _invert: boolean, _env: BoundaryEnv): void {}
  matrixCorr(_s: SimState, _imat: Int32Array, _matrix: Float64Array, _ngroup: number, _env: BoundaryEnv): void {}
}

/** Dummy BoundaryCorrection (no slab correction; the ffield case). */
export class NoBoundaryCorrection extends BoundaryCorrection {
  computeCorr(): number { return 0; }
}

/** Group atom indices in index order with their matrix index (imat >= 0). */
const groupList = (imat: Int32Array): number[] => {
  const out: number[] = [];
  for (let i = 0; i < imat.length; i++) if (imat[i] >= 0) out.push(i);
  return out;
};

/** slab_dipole.cpp: EW3DC dipole correction (Ballenegger et al.), boundary p p f. */
export class SlabDipole extends BoundaryCorrection {
  computeCorr(s: SimState, f: Float64Array, qsum: number, env: BoundaryEnv): number {
    const V = env.volume;
    const q = s.q, x = s.x, n = s.n;
    const zprdSlab = env.zprdSlab;
    let dipoleAll = 0;
    for (let i = 0; i < n; i++) dipoleAll += q[i] * x[3 * i + 2];
    let dipoleR2 = 0;
    for (let i = 0; i < n; i++) dipoleR2 += q[i] * x[3 * i + 2] * x[3 * i + 2];
    // need to make non-neutral systems translationally invariant (slab_dipole.cpp)
    const eSlabcorr = MY_2PI * (dipoleAll * dipoleAll - qsum * dipoleR2 - qsum * qsum * zprdSlab * zprdSlab / 12.0) / V;
    const scale = 1.0;
    const qscale = env.qqrd2e * scale;
    const energy = qscale * eSlabcorr;
    // force corrections
    const ffact = qscale * (-4.0 * Math.PI / V);
    for (let i = 0; i < n; i++) f[3 * i + 2] += ffact * q[i] * (dipoleAll - qsum * x[3 * i + 2]);
    return energy;
  }

  vectorCorr(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean, env: BoundaryEnv): void {
    const V = env.volume;
    const q = s.q, x = s.x, mask = s.mask, n = s.n;
    let dipole = 0;
    for (let i = 0; i < n; i++) if (inGroup(mask[i], sourceBit, invert)) dipole += q[i] * x[3 * i + 2];
    dipole *= MY_4PI / V;
    for (let i = 0; i < n; i++) if (mask[i] & sensorBit) vec[i] += x[3 * i + 2] * dipole;
  }

  matrixCorr(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number, env: BoundaryEnv): void {
    const prefac = MY_4PI / env.volume;
    const x = s.x;
    const group = groupList(imat);
    for (const i of group) {
      for (const j of group) {
        if (imat[j] > imat[i]) continue;   // matrix is symmetric
        const aij = prefac * x[3 * i + 2] * x[3 * j + 2];
        matrix[imat[i] * ngroup + imat[j]] += aij;
        if (imat[i] !== imat[j]) matrix[imat[j] * ngroup + imat[i]] += aij;
      }
    }
  }
}

/** slab_2d.cpp: EW2D correction (Yeh and Berkowitz style 2d Ewald with the Metalwalls form), boundary p p f. */
export class Slab2d extends BoundaryCorrection {
  computeCorr(s: SimState, f: Float64Array, _qsum: number, env: BoundaryEnv): number {
    const q = s.q, x = s.x, n = s.n;
    const gEwald = env.gEwald;
    const gInv = 1.0 / gEwald;
    const scale = 1.0;
    const qscale = env.qqrd2e * scale;
    const area = env.area;
    const ffact = qscale * MY_2PI / area;
    const efact = qscale * MY_PIS / area;
    let eKeq0 = 0;
    for (let i = 0; i < n; i++) {
      let potIJ = 0.0;
      for (let j = 0; j < n; j++) {
        const zij = x[3 * j + 2] - x[3 * i + 2];
        const gzij = gEwald * zij;
        // coulomb potential; see eq. (4) in metalwalls parallelization doc
        potIJ += q[j] * (Math.exp(-gzij * gzij) * gInv + MY_PIS * zij * erfAccurate(gzij));
        f[3 * i + 2] -= ffact * q[i] * q[j] * erfAccurate(gzij);
      }
      // per-atom energy; see eq. (20) in metalwalls ewald doc
      eKeq0 -= q[i] * potIJ;
    }
    return efact * eKeq0;
  }

  vectorCorr(s: SimState, vec: Float64Array, sensorBit: number, sourceBit: number, invert: boolean, env: BoundaryEnv): void {
    const x = s.x, q = s.q, mask = s.mask, n = s.n;
    const zAll: number[] = [], qAll: number[] = [];
    for (let i = 0; i < n; i++) {
      if (inGroup(mask[i], sourceBit, invert)) { zAll.push(x[3 * i + 2]); qAll.push(q[i]); }
    }
    const gEwald = env.gEwald;
    const prefac = 2 * MY_PIS / env.area;
    for (let i = 0; i < n; i++) {
      if (!(mask[i] & sensorBit)) continue;
      let b = 0;
      const zi = x[3 * i + 2];
      for (let j = 0; j < zAll.length; j++) {
        const zij = zAll[j] - zi;
        const gzij = gEwald * zij;
        // avoid exp underflow
        const zfac = gzij > 7 ? MY_PIS * zij : Math.exp(-(gzij * gzij)) / gEwald + MY_PIS * zij * erfAccurate(gzij);
        b += qAll[j] * zfac;
      }
      vec[i] -= prefac * b;
    }
  }

  matrixCorr(s: SimState, imat: Int32Array, matrix: Float64Array, ngroup: number, env: BoundaryEnv): void {
    const x = s.x;
    const gEwald = env.gEwald;
    const gInv = 1.0 / gEwald;
    const gSq = gEwald * gEwald;
    const prefac = 2.0 * MY_PIS / env.area;
    const group = groupList(imat);
    for (const i of group) {
      for (const j of group) {
        if (imat[j] > imat[i]) continue;   // matrix is symmetric
        const dij = x[3 * j + 2] - x[3 * i + 2];
        const aij = prefac * (Math.exp(-dij * dij * gSq) * gInv + MY_PIS * dij * erfAccurate(dij * gEwald));
        matrix[imat[i] * ngroup + imat[j]] -= aij;
        if (imat[i] !== imat[j]) matrix[imat[j] * ngroup + imat[i]] -= aij;
      }
    }
  }
}
