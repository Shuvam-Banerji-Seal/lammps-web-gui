/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_canonical_coloured.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { PTM_MAX_NBRS, PTM_MAX_EDGES, PTM_NO_ERROR } from './ptm_constants';
import { build_facet_map } from './ptm_graph_tools';

const SCHAR_MAX = 127;

// Stores a value into an int8_t the way the C assignment does (two's complement truncation).
function toInt8(x: number): number {
  return (x << 24) >> 24;
}

function weinberg_coloured(num_nodes: number, num_edges: number, common: number[][], colours: number[],
  best_code: number[], canonical_labelling: number[], a: number, b: number): boolean {
  const m: boolean[][] = [];
  for (let i = 0; i < PTM_MAX_NBRS; i++) m.push(new Array<boolean>(PTM_MAX_NBRS).fill(false));

  const index: number[] = new Array<number>(PTM_MAX_NBRS).fill(-1);

  let n = 0;
  index[a] = toInt8(colours[a] * num_nodes + n++);
  if (index[a] > best_code[0]) return false;

  let winning = false;
  if (index[a] < best_code[0]) {
    best_code[0] = index[a];
    winning = true;
  }

  let c = -1;
  for (let it = 1; it < 2 * num_edges; it++) {
    const newvertex = index[b] === -1;

    if (newvertex) index[b] = toInt8(colours[b] * num_nodes + n++);

    if (!winning && index[b] > best_code[it]) return false;

    if (winning || index[b] < best_code[it]) {
      winning = true;
      best_code[it] = index[b];
    }

    if (newvertex) {
      //When a new vertex is reached, take the right-most edge
      //relative to the edge on which the vertex is reached.

      c = common[a][b];
    } else if (!m[b][a]) {
      //When an old vertex is reached on a new path, go back
      //in the opposite direction.

      c = a;
    } else {
      //When an old vertex is reached on an old path, leave the
      //vertex on the right-most edge that has not previously
      //been traversed in that direction.

      c = common[a][b];
      while (m[b][c]) c = common[c][b];
    }

    m[a][b] = true;
    a = b;
    b = c;
  }

  if (winning) {
    for (let i = 0; i < num_nodes; i++) canonical_labelling[i] = index[i];
    return true;
  }

  return false;
}

export function canonical_form_coloured(num_facets: number, facets: number[][], num_nodes: number, degree: number[],
  colours: number[], canonical_labelling: number[], best_code: number[], hashOut: { value: bigint }): number {
  const common: number[][] = [];
  for (let i = 0; i < PTM_MAX_NBRS; i++) common.push(new Array<number>(PTM_MAX_NBRS).fill(0));
  const num_edges = Math.trunc((3 * num_facets) / 2);
  if (!build_facet_map(num_facets, facets, common)) return -1;

  // memset(best_code, SCHAR_MAX, sizeof(int8_t) * 2 * PTM_MAX_EDGES)
  for (let i = 0; i < 2 * PTM_MAX_EDGES; i++) best_code[i] = SCHAR_MAX;

  let equal = true;
  for (let i = 1; i < num_nodes; i++) {
    if (degree[i] !== degree[0] || colours[i] !== colours[0]) equal = false;
  }

  if (equal) {
    weinberg_coloured(num_nodes, num_edges, common, colours, best_code, canonical_labelling, facets[0][0], facets[0][1]);
  } else {
    let best_degree = 0;
    for (let i = 0; i < num_facets; i++) {
      const a = facets[i][0];
      const b = facets[i][1];
      const c = facets[i][2];

      //int da = colours[a] * num_nodes + degree[a];
      //int db = colours[b] * num_nodes + degree[b];
      //int dc = colours[c] * num_nodes + degree[c];

      const da = degree[a];
      const db = degree[b];
      const dc = degree[c];

      best_degree = Math.max(best_degree, (da << 16) | (db << 8) | (dc << 0));
      best_degree = Math.max(best_degree, (da << 0) | (db << 16) | (dc << 8));
      best_degree = Math.max(best_degree, (da << 8) | (db << 0) | (dc << 16));
    }

    for (let i = 0; i < num_facets; i++) {
      const a = facets[i][0];
      const b = facets[i][1];
      const c = facets[i][2];

      //int da = colours[a] * num_nodes + degree[a];
      //int db = colours[b] * num_nodes + degree[b];
      //int dc = colours[c] * num_nodes + degree[c];

      const da = degree[a];
      const db = degree[b];
      const dc = degree[c];

      if (best_degree === ((da << 16) | (db << 8) | (dc << 0))) {
        weinberg_coloured(num_nodes, num_edges, common, colours, best_code, canonical_labelling, a, b);
      }

      if (best_degree === ((da << 0) | (db << 16) | (dc << 8))) {
        weinberg_coloured(num_nodes, num_edges, common, colours, best_code, canonical_labelling, b, c);
      }

      if (best_degree === ((da << 8) | (db << 0) | (dc << 16))) {
        weinberg_coloured(num_nodes, num_edges, common, colours, best_code, canonical_labelling, c, a);
      }
    }
  }

  for (let i = num_nodes - 1; i >= 0; i--) {
    canonical_labelling[i + 1] = (canonical_labelling[i] % num_nodes) + 1;
  }
  canonical_labelling[0] = 0;

  let hash = 0n;
  for (let i = 0; i < 2 * num_edges; i++) {
    let e = BigInt.asUintN(64, BigInt(best_code[i]) + BigInt(i % 8));
    e &= 0xFn;
    e = BigInt.asUintN(64, e << BigInt((4 * i) % 64));
    hash ^= e;
  }

  hashOut.value = hash;
  return PTM_NO_ERROR;
}
