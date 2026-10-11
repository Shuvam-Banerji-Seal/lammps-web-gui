/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the neighbor lists that src/MANYBODY/pair_airebo.cpp reads (list->firstneigh of
 * owned atoms in PairAIREBO::FLJ, cutoff cutljsq per element pair), built here with a binned
 * search over owned and ghost positions. Authors credited in the C++ file headers: Ase Henry
 * (MIT); bugfixes and optimizations: Marcel Fallet and Steve Stuart (Clemson), Axel Kohlmeyer
 * (Temple U), Markus Hoehnerbach (RWTH Aachen), Cyril Falvo (Universite Paris Sud); AIREBO-M:
 * Thomas C. O'Connor (JHU) 2014. Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */

/**
 * Pairs (a, b) with a < nrows (owned rows) and b any owned or ghost atom, with
 * |x_a - x_b|^2 < cutsq[mt[a]*2 + mt[b]]. CSR result: row a is list[start[a] .. start[a+1]).
 */
export const rowNeighbors = (
  x: Float64Array, nrows: number, nall: number, mt: Int32Array, cutsq: Float64Array, cut: number,
): { start: Int32Array<ArrayBuffer>; list: Int32Array<ArrayBuffer> } => {
  const start = new Int32Array(nrows + 1);
  if (nall === 0 || nrows === 0) return { start, list: new Int32Array(0) };
  const cs = cut > 0 ? cut : 1;
  let xmin = Infinity, ymin = Infinity, zmin = Infinity;
  let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
  for (let a = 0; a < nall; a++) {
    const px = x[3 * a], py = x[3 * a + 1], pz = x[3 * a + 2];
    if (px < xmin) xmin = px;
    if (px > xmax) xmax = px;
    if (py < ymin) ymin = py;
    if (py > ymax) ymax = py;
    if (pz < zmin) zmin = pz;
    if (pz > zmax) zmax = pz;
  }
  const nx = Math.floor((xmax - xmin) / cs) + 1;
  const ny = Math.floor((ymax - ymin) / cs) + 1;
  const nz = Math.floor((zmax - zmin) / cs) + 1;
  const ncell = nx * ny * nz;
  const cellOf = new Int32Array(nall);
  const cstart = new Int32Array(ncell + 1);
  for (let a = 0; a < nall; a++) {
    const cx = Math.floor((x[3 * a] - xmin) / cs);
    const cy = Math.floor((x[3 * a + 1] - ymin) / cs);
    const cz = Math.floor((x[3 * a + 2] - zmin) / cs);
    const c = cx + nx * (cy + ny * cz);
    cellOf[a] = c;
    cstart[c + 1]++;
  }
  for (let c = 0; c < ncell; c++) cstart[c + 1] += cstart[c];
  const fill = cstart.slice(0, ncell);
  const cellAtoms = new Int32Array(nall);
  for (let a = 0; a < nall; a++) cellAtoms[fill[cellOf[a]]++] = a;
  const out: number[] = [];
  for (let a = 0; a < nrows; a++) {
    start[a] = out.length;
    if (mt[a] < 0) continue;
    const itype = mt[a];
    const xa = x[3 * a], ya = x[3 * a + 1], za = x[3 * a + 2];
    const cx = Math.floor((xa - xmin) / cs);
    const cy = Math.floor((ya - ymin) / cs);
    const cz = Math.floor((za - zmin) / cs);
    for (let dz = -1; dz <= 1; dz++) {
      const z = cz + dz;
      if (z < 0 || z >= nz) continue;
      for (let dy = -1; dy <= 1; dy++) {
        const y = cy + dy;
        if (y < 0 || y >= ny) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = cx + dx;
          if (xx < 0 || xx >= nx) continue;
          const c = xx + nx * (y + ny * z);
          for (let q = cstart[c]; q < cstart[c + 1]; q++) {
            const b = cellAtoms[q];
            if (b === a || mt[b] < 0) continue;
            const delx = xa - x[3 * b], dely = ya - x[3 * b + 1], delz = za - x[3 * b + 2];
            const rsq = delx * delx + dely * dely + delz * delz;
            if (rsq < cutsq[itype * 2 + mt[b]]) out.push(b);
          }
        }
      }
    }
  }
  start[nrows] = out.length;
  return { start, list: Int32Array.from(out) };
};
