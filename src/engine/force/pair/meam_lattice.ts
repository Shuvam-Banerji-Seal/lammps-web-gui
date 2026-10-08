/*
 * Reference-structure geometry for MEAM (pair_style meam). The reference structure of a MEAM
 * entry is scaled so that its nearest-neighbour distance is r; this module lists the neighbour
 * vectors of one atom of that structure within the cutoff rc.
 *
 * Documented lattice names (docs.lammps.org/pair_meam.html, plans/lammps-docs/pair_meam.rst):
 *   "fcc = face centered cubic"
 *   "bcc = body centered cubic"
 *   "dia = diamond (interlaced fcc for alloy)"
 *   "hcp = hexagonal close-packed"
 *   "sc  = simple cubic"
 * The remaining names of the doc list (dia3, b1, c11, l12, b2, ch4, lin, zig, tri, dim) are
 * rejected by the caller with a StyleError unless implemented here.
 *
 * Measured with native LAMMPS (black box): single-element bcc and dia crystals and dimers agree
 * with the reference-structure model in meam.ts (the bcc 2NN screening weight is 0 at every
 * scale, the dia 2NN shell at a/sqrt(2) is screened to 0 as well). For the 2NN formulation the
 * reference structure of the pair term holds exactly the first two neighbour shells (measured
 * with native LAMMPS, black box: the bcc reference of the WL library entry is reproduced by the
 * 8 first and 6 second neighbours with screening among those 14 only).
 */

export type ReferenceLattice = 'fcc' | 'bcc' | 'dia' | 'hcp' | 'sc';

export interface ReferenceVector {
  dx: number;
  dy: number;
  dz: number;
  r: number;
}

export const SUPPORTED_REFERENCE_LATTICES: readonly ReferenceLattice[] = ['fcc', 'bcc', 'dia', 'hcp', 'sc'];

/**
 * Ratio r2/r1 of the second to the first neighbour-shell distance of the reference lattice, used
 * by the 2NN (nn2 = 1) pair term. Ideal c/a is assumed for hcp (docs.lammps.org/pair_meam.html
 * does not carry a c/a parameter).
 */
export const PAIR_SHELL_RATIO: Readonly<Record<ReferenceLattice, number>> = {
  fcc: Math.SQRT2,
  bcc: 2 / Math.sqrt(3),
  sc: Math.SQRT2,
  dia: Math.sqrt(8 / 3),
  // hcp with nn2 = 1 is refused by the caller (meam.ts), so this entry is never used for the 2NN shell.
  hcp: 1,
};

/** Ideal c/a of the hcp reference structure. */
export const HCP_C_OVER_A = Math.sqrt(8 / 3);

/**
 * Neighbour vectors (within rc, origin excluded) of one atom in the reference lattice whose
 * nearest-neighbour distance is r. For dia the list holds both sublattices (A atoms and the
 * B atoms shifted by (a/4)(1,1,1)), so the 4 nearest neighbours are the B atoms.
 */
export function referenceVectors(lat: ReferenceLattice, r: number, rc: number): ReferenceVector[] {
  if (lat === 'hcp') return hcpVectors(r, rc);
  const out: ReferenceVector[] = [];
  let a: number;
  let parity: (i: number, j: number, k: number) => boolean;
  let m: number;
  let h: number;
  if (lat === 'fcc') {
    a = r * Math.SQRT2;
    m = Math.ceil(rc / a) + 1;
    h = a / 2;
    parity = (i, j, k) => ((i + j + k) & 1) === 0;
  } else if (lat === 'bcc') {
    a = (2 * r) / Math.sqrt(3);
    m = Math.ceil(rc / (a / 2)) + 1;
    h = a / 2;
    parity = (i, j, k) => (((i ^ j) | (j ^ k)) & 1) === 0;
  } else if (lat === 'sc') {
    a = r;
    m = Math.ceil(rc / a) + 1;
    h = a;
    parity = () => true;
  } else {
    a = (4 * r) / Math.sqrt(3);
    m = Math.ceil(rc / (a / 2)) + 1;
    h = a / 2;
    parity = (i, j, k) => ((i + j + k) & 1) === 0;
  }
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

/**
 * Neighbours of one atom of an ideal hcp reference structure with nearest-neighbour distance r.
 * Two interpenetrating hexagonal sublattices: basis (0,0,0) and (a/2, a/(2 sqrt3), c/2), with
 * a = r and c = a sqrt(8/3).
 */
function hcpVectors(r: number, rc: number): ReferenceVector[] {
  const a = r;
  const c = a * HCP_C_OVER_A;
  const a1 = [a, 0, 0];
  const a2 = [-a / 2, (a * Math.sqrt(3)) / 2, 0];
  const a3 = [0, 0, c];
  const basis = [
    [0, 0, 0],
    [a / 2, a / (2 * Math.sqrt(3)), c / 2],
  ];
  const out: ReferenceVector[] = [];
  const m = Math.ceil(rc / a) + 1;
  for (let p = 0; p < 2; p++) {
    for (let i = -m; i <= m; i++)
      for (let j = -m; j <= m; j++)
        for (let k = -m; k <= m; k++) {
          if (p === 0 && i === 0 && j === 0 && k === 0) continue;
          const dx = i * a1[0] + j * a2[0] + k * a3[0] + basis[p][0];
          const dy = i * a1[1] + j * a2[1] + k * a3[1] + basis[p][1];
          const dz = i * a1[2] + j * a2[2] + k * a3[2] + basis[p][2];
          const rr = Math.hypot(dx, dy, dz);
          if (rr > 1e-9 && rr < rc) out.push({ dx, dy, dz, r: rr });
        }
  }
  return out;
}
