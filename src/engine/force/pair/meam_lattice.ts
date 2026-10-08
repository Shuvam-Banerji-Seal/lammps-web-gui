/*
 * Reference-structure geometry for MEAM (pair_style meam). The reference structure of a MEAM
 * entry is scaled so that its nearest-neighbour distance is r; this module lists the neighbour
 * vectors of one atom of that structure within the cutoff rc.
 *
 * Documented lattice names (docs.lammps.org/pair_meam.html, plans/lammps-docs/pair_meam.rst):
 *   "fcc = face centered cubic"
 *   "bcc = body centered cubic"
 *   "dia = diamond (interlaced fcc for alloy)"
 * Only these three are implemented; the other names of the doc list (hcp, dia3, b1, c11, l12, b2,
 * ch4, lin, zig, tri, sc, dim) are rejected by the caller with a StyleError.
 *
 * Measured with native LAMMPS (black box): single-element bcc and dia crystals and dimers agree
 * with the reference-structure model in meam.ts (the bcc 2NN screening weight is 0 at every
 * scale, the dia 2NN shell at a/sqrt(2) is screened to 0 as well).
 */

export type ReferenceLattice = 'fcc' | 'bcc' | 'dia';

export interface ReferenceVector {
  dx: number;
  dy: number;
  dz: number;
  r: number;
}

export const SUPPORTED_REFERENCE_LATTICES: readonly ReferenceLattice[] = ['fcc', 'bcc', 'dia'];

/**
 * Neighbour vectors (within rc, origin excluded) of one atom in the reference lattice whose
 * nearest-neighbour distance is r. For dia the list holds both sublattices (A atoms and the
 * B atoms shifted by (a/4)(1,1,1)), so the 4 nearest neighbours are the B atoms.
 */
export function referenceVectors(lat: ReferenceLattice, r: number, rc: number): ReferenceVector[] {
  const out: ReferenceVector[] = [];
  let a: number;
  let parity: (i: number, j: number, k: number) => boolean;
  let m: number;
  if (lat === 'fcc') {
    a = r * Math.SQRT2;
    m = Math.ceil(rc / a) + 1;
    parity = (i, j, k) => ((i + j + k) & 1) === 0;
  } else if (lat === 'bcc') {
    a = (2 * r) / Math.sqrt(3);
    m = Math.ceil(rc / (a / 2)) + 1;
    parity = (i, j, k) => (((i ^ j) | (j ^ k)) & 1) === 0;
  } else {
    a = (4 * r) / Math.sqrt(3);
    m = Math.ceil(rc / (a / 2)) + 1;
    parity = (i, j, k) => ((i + j + k) & 1) === 0;
  }
  const h = a / 2;
  for (let i = -m; i <= m; i++)
    for (let j = -m; j <= m; j++)
      for (let k = -m; k <= m; k++) {
        if (!parity(i, j, k) || (i === 0 && j === 0 && k === 0)) continue;
        const dx = i * h, dy = j * h, dz = k * h;
        const rr = Math.hypot(dx, dy, dz);
        if (rr < rc) out.push({ dx, dy, dz, r: rr });
      }
  if (lat === 'dia') {
    const s = a / 4;
    for (let i = -m; i <= m; i++)
      for (let j = -m; j <= m; j++)
        for (let k = -m; k <= m; k++) {
          if (!parity(i, j, k)) continue;
          const dx = i * h + s, dy = j * h + s, dz = k * h + s;
          const rr = Math.hypot(dx, dy, dz);
          if (rr < rc) out.push({ dx, dy, dz, r: rr });
        }
  }
  return out;
}
