/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_graph_tools.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { PTM_MAX_NBRS } from './ptm_constants';

export function build_facet_map(num_facets: number, facets: number[][], common: number[][]): boolean {
  // memset(common, -1, sizeof(int8_t) * PTM_MAX_NBRS * PTM_MAX_NBRS)
  for (let i = 0; i < PTM_MAX_NBRS; i++) {
    for (let j = 0; j < PTM_MAX_NBRS; j++) common[i][j] = -1;
  }

  for (let i = 0; i < num_facets; i++) {
    const a = facets[i][0];
    const b = facets[i][1];
    const c = facets[i][2];

    if (common[a][b] !== -1 || common[b][c] !== -1 || common[c][a] !== -1) return false;

    common[a][b] = c;
    common[b][c] = a;
    common[c][a] = b;
  }

  return true;
}

export function graph_degree(num_facets: number, facets: number[][], num_nodes: number, degree: number[]): number {
  // memset(degree, 0, sizeof(int8_t) * num_nodes)
  for (let i = 0; i < num_nodes; i++) degree[i] = 0;

  for (let i = 0; i < num_facets; i++) {
    const a = facets[i][0];
    const b = facets[i][1];
    const c = facets[i][2];

    degree[a]++;
    degree[b]++;
    degree[c]++;
  }

  let max_degree = 0;
  for (let i = 0; i < num_nodes; i++) max_degree = Math.max(max_degree, degree[i]);

  return max_degree;
}
