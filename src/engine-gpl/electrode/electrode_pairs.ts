/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ELECTRODE/electrode_matrix.cpp and electrode_vector.cpp (the real-space pair and self terms of the
 * elastance matrix and the b vector), and the gausscorr/self_energy/v_tally helpers of src/ELECTRODE/fix_electrode_conp.cpp,
 * at commit b332f28. ELECTRODE contributing authors: Ludwig Ahrens-Iwers (TUHH), Shern Tee (UQ), Robert Meissner (TUHH).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The C++ loops over a half neighbour list with ghost images. Here every periodic image of every pair within the
 * cut-off is enumerated directly (the same set of pairs), so each ordered (i, j, image) term is visited once and the
 * half-list bookkeeping of the C++ (adding to both matrix entries, weighting by 1/2) is written out as ordered sums.
 */
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { MY_PIS, safeErfc } from './electrode_math';

/**
 * Calls fn(i, j, rsq, dx, dy, dz) for every ordered pair (i, j) and periodic image within the type cut-off
 * (rsq < cutsq[type_i * nt + type_j], self pair with zero image excluded). d = x_i - (x_j + image).
 * The visiting order is i, then j, then the image shifts (x, y, z), as in the C++ pair loops of this file's header.
 * Pruning: a shift whose dx*dx (or dx*dx + dy*dy) already reaches the cut-off cannot give rsq < cut2, because
 * rsq is the same sum with non-negative terms added, so those shifts are skipped without changing any call.
 */
export const eachImagePair = (
  s: SimState, geom: Geometry, cutsq: Float64Array, nt: number,
  fn: (i: number, j: number, rsq: number, dx: number, dy: number, dz: number) => void,
): void => {
  const n = s.n, x = s.x, type = s.type;
  const Lx = geom.lx, Ly = geom.ly, Lz = geom.lz;
  const perX = !!geom.periodic[0], perY = !!geom.periodic[1], perZ = !!geom.periodic[2];
  let cutmax2 = 0;
  for (let t = 0; t < cutsq.length; t++) if (cutsq[t] > cutmax2) cutmax2 = cutsq[t];
  const rmax = Math.sqrt(cutmax2);
  // a non-periodic direction has no images: its range is the single shift 0
  const nx = perX ? Math.ceil(rmax / Lx) : 0;
  const ny = perY ? Math.ceil(rmax / Ly) : 0;
  const nz = perZ ? Math.ceil(rmax / Lz) : 0;
  for (let i = 0; i < n; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    for (let j = 0; j < n; j++) {
      const cut2 = cutsq[type[i] * nt + type[j]];
      if (!(cut2 > 0)) continue;
      const d0x = xi - x[3 * j], d0y = yi - x[3 * j + 1], d0z = zi - x[3 * j + 2];
      const bx = perX ? Math.round(d0x / Lx) : 0;
      const by = perY ? Math.round(d0y / Ly) : 0;
      const bz = perZ ? Math.round(d0z / Lz) : 0;
      const x0 = bx - nx, x1 = bx + nx, y0 = by - ny, y1 = by + ny, z0 = bz - nz, z1 = bz + nz;
      for (let sx = x0; sx <= x1; sx++) {
        const dx = d0x - sx * Lx;
        const dx2 = dx * dx;
        if (dx2 >= cut2) continue;
        for (let sy = y0; sy <= y1; sy++) {
          const dy = d0y - sy * Ly;
          const dxy = dx2 + dy * dy;
          if (dxy >= cut2) continue;
          for (let sz = z0; sz <= z1; sz++) {
            if (i === j && sx === 0 && sy === 0 && sz === 0) continue;
            const dz = d0z - sz * Lz;
            const rsq = dxy + dz * dz;
            if (rsq < cut2) fn(i, j, rsq, dx, dy, dz);
          }
        }
      }
    }
  }
};

/** Electrode-electrode matrix pair term: M[i][j] += (erfc(g r) - erfc(eta_ij r)) / r over ordered images (+ self). */
export const matrixPairTerms = (
  s: SimState, geom: Geometry, cutsq: Float64Array, nt: number, elecBit: number,
  imat: Int32Array, ngroup: number, matrix: Float64Array, gEwald: number,
  etaOf: (i: number) => number, sameTypeOnly = false,
): void => {
  const mask = s.mask, type = s.type;
  eachImagePair(s, geom, cutsq, nt, (i, j, rsq) => {
    if (!((mask[i] & elecBit) && (mask[j] & elecBit))) return;
    if (sameTypeOnly && type[i] !== type[j]) return;
    const etaI = etaOf(i), etaJ = etaOf(j);
    const etaij = etaI * etaJ / Math.sqrt(etaI * etaI + etaJ * etaJ);
    const r = Math.sqrt(rsq);
    const rinv = 1.0 / r;
    let aij = rinv;
    aij *= safeErfc(gEwald * r);
    aij -= safeErfc(etaij * r) * rinv;
    matrix[imat[i] * ngroup + imat[j]] += aij;
  });
};

/** Self term of the elastance matrix: M[i][i] += sqrt(2)/sqrt(pi) eta_i - 2 g / sqrt(pi). */
export const matrixSelfTerms = (
  s: SimState, elecBit: number, imat: Int32Array, ngroup: number, matrix: Float64Array, gEwald: number,
  etaOf: (i: number) => number,
): void => {
  const selfint = 2.0 / MY_PIS * gEwald;
  const preta = Math.SQRT2 / MY_PIS;
  for (let i = 0; i < s.n; i++) {
    if (!(s.mask[i] & elecBit)) continue;
    const etaI = etaOf(i);
    matrix[imat[i] * ngroup + imat[i]] += preta * etaI - selfint;
  }
};

/**
 * b-vector pair term (ElectrodeVector::pair_contribution): vec[i] += aij q_j for sensor i and source j. The
 * screening eta is the harmonic mean when both atoms are sensors, else the eta of the sensor atom.
 */
export const vectorPairTerms = (
  s: SimState, geom: Geometry, cutsq: Float64Array, nt: number, sensorBit: number, sourceBit: number,
  invert: boolean, vec: Float64Array, gEwald: number, etaOf: (i: number) => number, sameTypeOnly = false,
): void => {
  const q = s.q, mask = s.mask, type = s.type;
  const inSource = (i: number) => ((mask[i] & sourceBit) !== 0) !== invert;
  eachImagePair(s, geom, cutsq, nt, (i, j, rsq) => {
    const iSens = (mask[i] & sensorBit) !== 0, jSens = (mask[j] & sensorBit) !== 0;
    const jSrc = inSource(j);
    if (!(iSens && jSrc)) return;
    // etypes yes with the electrode-electrode list (mat_neighlist): only pairs of equal type are in it
    if (sameTypeOnly && type[i] !== type[j]) return;   // ordered term for the sensor atom i (the reverse term is its own visit)
    const etaI = etaOf(i), etaJ = etaOf(j);
    let etaij: number;
    if (iSens && jSens) etaij = etaI * etaJ / Math.sqrt(etaI * etaI + etaJ * etaJ);
    else etaij = etaI;
    const r = Math.sqrt(rsq);
    const rinv = 1.0 / r;
    let aij = rinv;
    aij *= safeErfc(gEwald * r);
    aij -= safeErfc(etaij * r) * rinv;
    vec[i] += aij * q[j];
  });
};

/** Self term of the b vector: vec[i] += (sqrt(2)/sqrt(pi) eta_i - 2 g / sqrt(pi)) q_i for sensor and source atoms. */
export const vectorSelfTerms = (
  s: SimState, sensorBit: number, sourceBit: number, invert: boolean, vec: Float64Array, gEwald: number,
  etaOf: (i: number) => number,
): void => {
  const selfint = 2.0 / MY_PIS * gEwald;
  const preta = Math.SQRT2 / MY_PIS;
  for (let i = 0; i < s.n; i++) {
    const sens = (s.mask[i] & sensorBit) !== 0;
    const src = ((s.mask[i] & sourceBit) !== 0) !== invert;
    if (sens && src) vec[i] += (preta * etaOf(i) - selfint) * s.q[i];
  }
};
