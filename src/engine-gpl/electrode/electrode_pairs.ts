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
 * The candidates come from a cell list of the periodic images (see the body), sorted back into that order.
 */
/**
 * Optional atom selections of eachImagePair: i (only these atoms as the first atom), j (only these as the second) and
 * either (only pairs with at least one selected atom). A caller passes the selections its fn would test anyway, so
 * the skipped calls are calls that would have returned at once: the remaining calls are the same, in the same order.
 */
export interface PairSelection {
  i?: Uint8Array;
  j?: Uint8Array;
  either?: Uint8Array;
}

export const eachImagePair = (
  s: SimState, geom: Geometry, cutsq: Float64Array, nt: number,
  fn: (i: number, j: number, rsq: number, dx: number, dy: number, dz: number) => void,
  sel: PairSelection = {},
): void => {
  const n = s.n, x = s.x, type = s.type;
  const Lx = geom.lx, Ly = geom.ly, Lz = geom.lz;
  const perX = !!geom.periodic[0], perY = !!geom.periodic[1], perZ = !!geom.periodic[2];
  let cutmax2 = 0;
  for (let t = 0; t < cutsq.length; t++) if (cutsq[t] > cutmax2) cutmax2 = cutsq[t];
  if (!(cutmax2 > 0) || n === 0) return;
  const rmax = Math.sqrt(cutmax2);
  // a non-periodic direction has no images: its range is the single shift 0
  const nx = perX ? Math.ceil(rmax / Lx) : 0;
  const ny = perY ? Math.ceil(rmax / Ly) : 0;
  const nz = perZ ? Math.ceil(rmax / Lz) : 0;
  // Every (j, image) within rmax of some atom i lies within rmax of the atoms' bounding box. The images of every atom
  // in that region are binned in cells of side >= rmax; each atom i collects the candidates of its 27 neighbour cells,
  // computes rsq exactly as the direct loop over j and shifts (d0 = x_i - x_j, d = d0 - shift L), keeps those with
  // rsq < the pair cut-off, and calls fn in the direct loop's order (j ascending, then the shifts x, y, z ascending).
  // The direct loop visits every image within the cut-off (|shift - round(d0/L)| <= ceil(rmax/L)), so the calls are the
  // same, with the same arguments, in the same order.
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      const v = x[3 * i + d];
      if (v < lo[d]) lo[d] = v;
      if (v > hi[d]) hi[d] = v;
    }
  }
  for (let d = 0; d < 3; d++) { lo[d] -= rmax; hi[d] += rmax; }
  // cells of side >= rmax / 3; the stencil holds the cell offsets whose nearest points are closer than rmax
  const nc = [0, 1, 2].map((d) => Math.max(1, Math.floor((hi[d] - lo[d]) / (rmax / 3))));
  const side = [0, 1, 2].map((d) => (hi[d] - lo[d]) / nc[d]);
  const reach = [0, 1, 2].map((d) => Math.ceil(rmax / side[d]));
  const stencil: number[] = [];
  for (let oz = -reach[2]; oz <= reach[2]; oz++) {
    for (let oy = -reach[1]; oy <= reach[1]; oy++) {
      for (let ox = -reach[0]; ox <= reach[0]; ox++) {
        const gx = Math.max(Math.abs(ox) - 1, 0) * side[0], gy = Math.max(Math.abs(oy) - 1, 0) * side[1], gz = Math.max(Math.abs(oz) - 1, 0) * side[2];
        // a small margin: a point on a cell boundary may be binned on either side by rounding
        if (gx * gx + gy * gy + gz * gz < cutmax2 * (1 + 1e-6)) stencil.push(ox, oy, oz);
      }
    }
  }
  const cellOf = (v: number, d: number): number => {
    let c = Math.floor((v - lo[d]) / (hi[d] - lo[d]) * nc[d]);
    if (c < 0) c = 0; else if (c >= nc[d]) c = nc[d] - 1;
    return c;
  };
  // the images inside the region: atom j and its shift (sx, sy, sz), packed in one row per image
  const imJ: number[] = [], imS: number[] = [], imCell: number[] = [];
  const L = [Lx, Ly, Lz];
  const nimg = [nx, ny, nz];
  for (let j = 0; j < n; j++) {
    if (sel.j && !sel.j[j]) continue;
    const p0 = x[3 * j], p1 = x[3 * j + 1], p2 = x[3 * j + 2];
    for (let sx = -nimg[0] - 1; sx <= nimg[0] + 1; sx++) {
      if (sx !== 0 && !perX) continue;
      const q0 = p0 + sx * L[0];
      if (q0 < lo[0] || q0 > hi[0]) continue;
      for (let sy = -nimg[1] - 1; sy <= nimg[1] + 1; sy++) {
        if (sy !== 0 && !perY) continue;
        const q1 = p1 + sy * L[1];
        if (q1 < lo[1] || q1 > hi[1]) continue;
        for (let sz = -nimg[2] - 1; sz <= nimg[2] + 1; sz++) {
          if (sz !== 0 && !perZ) continue;
          const q2 = p2 + sz * L[2];
          if (q2 < lo[2] || q2 > hi[2]) continue;
          imJ.push(j);
          imS.push(sx, sy, sz);
          imCell.push((cellOf(q2, 2) * nc[1] + cellOf(q1, 1)) * nc[0] + cellOf(q0, 0));
        }
      }
    }
  }
  const ncell = nc[0] * nc[1] * nc[2];
  // the images binned per cell (all images, and with an either selection also the images of selected atoms only)
  const bin = (keep: (k: number) => boolean): { head: Int32Array; members: Int32Array } => {
    const head = new Int32Array(ncell + 1);
    for (let k = 0; k < imJ.length; k++) if (keep(k)) head[imCell[k] + 1]++;
    for (let c = 0; c < ncell; c++) head[c + 1] += head[c];
    const fill = head.slice(0, ncell);
    const members = new Int32Array(head[ncell]);
    for (let k = 0; k < imJ.length; k++) if (keep(k)) members[fill[imCell[k]]++] = k;
    return { head, members };
  };
  const allBins = bin(() => true);
  const selBins = sel.either ? bin((k) => sel.either![imJ[k]] !== 0) : allBins;
  // the direct loop's order is (j, sx, sy, sz) ascending: a numeric key per candidate, sorted with its index
  const M = Math.max(nimg[0], nimg[1], nimg[2]) + 1, W = 2 * M + 1;
  let cap = 1024;
  let candJ = new Int32Array(cap), candR = new Float64Array(cap), candD = new Float64Array(3 * cap), keys = new Float64Array(cap);
  for (let i = 0; i < n; i++) {
    if (sel.i && !sel.i[i]) continue;
    const eitherI = sel.either ? sel.either[i] !== 0 : true;
    // an atom outside the either selection pairs only with selected atoms
    const { head, members } = eitherI ? allBins : selBins;
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const ci0 = cellOf(xi, 0), ci1 = cellOf(yi, 1), ci2 = cellOf(zi, 2);
    let m = 0;
    for (let st = 0; st < stencil.length; st += 3) {
      const cx = ci0 + stencil[st], cy = ci1 + stencil[st + 1], cz = ci2 + stencil[st + 2];
      if (cx < 0 || cy < 0 || cz < 0 || cx >= nc[0] || cy >= nc[1] || cz >= nc[2]) continue;
      {
        {
          const c = (cz * nc[1] + cy) * nc[0] + cx;
          for (let q = head[c]; q < head[c + 1]; q++) {
            const k = members[q];
            const j = imJ[k];
            const sx = imS[3 * k], sy = imS[3 * k + 1], sz = imS[3 * k + 2];
            if (i === j && sx === 0 && sy === 0 && sz === 0) continue;
            const cut2 = cutsq[type[i] * nt + type[j]];
            if (!(cut2 > 0)) continue;
            // the arithmetic of the direct loop
            const d0x = xi - x[3 * j], d0y = yi - x[3 * j + 1], d0z = zi - x[3 * j + 2];
            const dx = d0x - sx * Lx;
            const dx2 = dx * dx;
            if (dx2 >= cut2) continue;
            const dy = d0y - sy * Ly;
            const dxy = dx2 + dy * dy;
            if (dxy >= cut2) continue;
            const dz = d0z - sz * Lz;
            const rsq = dxy + dz * dz;
            if (!(rsq < cut2)) continue;
            if (m === cap) {
              cap *= 2;
              const nj = new Int32Array(cap); nj.set(candJ); candJ = nj;
              const nr = new Float64Array(cap); nr.set(candR); candR = nr;
              const nd = new Float64Array(3 * cap); nd.set(candD); candD = nd;
              const nk = new Float64Array(cap); nk.set(keys); keys = nk;
            }
            candJ[m] = j;
            candR[m] = rsq;
            candD[3 * m] = dx; candD[3 * m + 1] = dy; candD[3 * m + 2] = dz;
            // key (j, sx, sy, sz) ascending, with the candidate index in the low part (unique keys, exact in a double)
            keys[m] = ((j * W + (sx + M)) * W + (sy + M)) * W + (sz + M);
            m++;
          }
        }
      }
    }
    const order = new Float64Array(m);
    for (let k = 0; k < m; k++) order[k] = keys[k] * cap + k;
    order.sort();
    for (let q = 0; q < m; q++) {
      const v = order[q];
      const k = v - Math.floor(v / cap) * cap;
      fn(i, candJ[k], candR[k], candD[3 * k], candD[3 * k + 1], candD[3 * k + 2]);
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
  const elec = new Uint8Array(s.n);
  for (let i = 0; i < s.n; i++) elec[i] = (mask[i] & elecBit) ? 1 : 0;
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
  }, { i: elec, j: elec });
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
  const sens = new Uint8Array(s.n), src = new Uint8Array(s.n);
  for (let i = 0; i < s.n; i++) { sens[i] = (mask[i] & sensorBit) ? 1 : 0; src[i] = inSource(i) ? 1 : 0; }
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
  }, { i: sens, j: src });
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
