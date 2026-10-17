/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/reset_atoms_id.cpp (ResetAtomsID::sort() and sort_bins() with compare_coords(), one process) at commit
 * b332f28. Copyright Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * The order of reset_atoms id sort yes: the atom indices in the order that receives the IDs 1..N. The bounding box of
 * the atoms is widened where a side has no extent, then grown by 1e-4 of its length on each side (the high side from
 * the grown low side). The bin count estimate is natoms/10 + 1 (integer division) and binsize = (volume / estimate)^(1/dim);
 * nbin = int(length / binsize) + 1 per side. An atom's bin is (iz * nby + iy) * nbx + ix with ix = int((x - lo) * invx *
 * nbx) (the same for y and z; invz is 0 in 2d). Atoms are ordered by bin, then by x, y, z, and equal positions keep their
 * storage order (the merge sort of sort_bins is stable).
 */
import { pow as libmPow } from '../libm/pow';
export const spatialSortOrder = (x: Float64Array, n: number, dim: number): number[] => {
  const BIG = 1.0e20;
  const lo = [BIG, BIG, BIG], hi = [-BIG, -BIG, -BIG];
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      const v = x[3 * i + d];
      if (v < lo[d]) lo[d] = v;
      if (v > hi[d]) hi[d] = v;
    }
  }
  if (dim === 2) lo[2] = hi[2] = 0;
  for (let d = 0; d < 3; d++) {
    if (lo[d] === hi[d]) { lo[d] -= 0.5; hi[d] += 0.5; }
  }
  const blo = [0, 0, 0], bhi = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    blo[d] = lo[d] - 0.0001 * (hi[d] - lo[d]);
    bhi[d] = hi[d] + 0.0001 * (hi[d] - blo[d]);
  }
  const nbinEstimate = Math.floor(n / 10) + 1;
  const ext = [bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]];
  const vol = dim === 2 ? ext[0] * ext[1] : ext[0] * ext[1] * ext[2];
  const binsize = libmPow(vol / nbinEstimate, 1 / dim);
  const nbx = Math.trunc(ext[0] / binsize) + 1;
  const nby = Math.trunc(ext[1] / binsize) + 1;
  const nbz = Math.trunc(ext[2] / binsize) + 1;
  const invx = 1 / ext[0], invy = 1 / ext[1];
  const invz = dim === 2 ? 0 : 1 / ext[2];
  const bin = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ix = Math.trunc((x[3 * i] - blo[0]) * invx * nbx);
    const iy = Math.trunc((x[3 * i + 1] - blo[1]) * invy * nby);
    const iz = Math.trunc((x[3 * i + 2] - blo[2]) * invz * nbz);
    bin[i] = iz * nby * nbx + iy * nbx + ix;
  }
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => {
    if (bin[a] !== bin[b]) return bin[a] - bin[b];
    for (let d = 0; d < 3; d++) {
      const xa = x[3 * a + d], xb = x[3 * b + d];
      if (xa !== xb) return xa < xb ? -1 : 1;
    }
    return a - b;
  });
  return idx;
};
