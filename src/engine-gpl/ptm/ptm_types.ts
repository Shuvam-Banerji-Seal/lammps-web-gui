/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), the struct typedefs of
 * src/PTM/ptm_graph_data.h, ptm_initialize_data.h, ptm_convex_hull_incremental.h and ptm_structure_matcher.h at
 * LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm).
 * This file is part of the GPL-2.0 module src/engine-gpl (see its LICENSE).
 *
 * C struct fields keep their C names (snake_case) so the translation can be checked line by line. C pointers to
 * arrays become arrays of arrays (number[][]); the caller's arrays are mutated in place, as the C code does.
 * uint64_t hashes are bigint (they are only compared for equality).
 */

/** graph_t (ptm_graph_data.h): canonical graph of one template neighbourhood. */
export interface Graph {
  id: number;
  hash: bigint;
  automorphism_index: number;
  num_automorphisms: number;
  /** PTM_MAX_POINTS entries */
  canonical_labelling: number[];
  /** PTM_MAX_FACETS rows of 3 */
  facets: number[][];
}

/** refdata_t (ptm_initialize_data.h): one reference structure. Optional C pointers are null when nullptr. */
export interface RefData {
  type: number;
  num_nbrs: number;
  num_facets: number;
  max_degree: number;
  num_graphs: number;
  graphs: Graph[] | null;
  points: number[][] | null;
  points_alt1: number[][] | null;
  points_alt2: number[][] | null;
  points_alt3: number[][] | null;
  penrose: number[][] | null;
  penrose_alt1: number[][] | null;
  penrose_alt2: number[][] | null;
  penrose_alt3: number[][] | null;
  num_mappings: number;
  mapping: number[][] | null;
  mapping_conventional: number[][] | null;
  template_indices: number[] | null;
}

/** convexhull_t (ptm_convex_hull_incremental.h) */
export interface ConvexHull {
  /** PTM_MAX_FACETS rows of 3 */
  facets: number[][];
  /** PTM_MAX_FACETS rows of 3 */
  plane_normal: number[][];
  /** PTM_MAX_POINTS entries */
  processed: boolean[];
  initial_vertices: number[];
  barycentre: number[];
  num_facets: number;
  num_prev: number;
  ok: boolean;
}

/** result_t (ptm_structure_matcher.h) */
export interface MatchResult {
  rmsd: number;
  scale: number;
  /** rotation in quaternion form */
  q: number[];
  /** PTM_MAX_POINTS entries */
  mapping: number[];
  ref_struct: RefData | null;
}

/**
 * get_neighbours callback of ptm_index (ptm_functions.h): fills nbr_indices, numbers and nbr_pos for the central
 * atom (num entries), returns the number of entries written. vdata is the caller's neighbour-list handle.
 */
export type GetNeighbours = (
  vdata: unknown,
  central_index: number,
  atom_index: number,
  num: number,
  nbr_indices: number[],
  numbers: number[],
  nbr_pos: number[][],
) => number;
