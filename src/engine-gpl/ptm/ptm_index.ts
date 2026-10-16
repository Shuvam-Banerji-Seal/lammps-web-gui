/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_index.cpp at LAMMPS
 * commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS
 * compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * ptm_index(): the structure identification for one atom. Only the outputs that compute ptm/atom reads are produced
 * (type, alloy type, scale, rmsd, quaternion, interatomic distance): the C calls compute_ptm_atom.cpp makes pass nullptr
 * for F, F_res, U, P, p_lattice_constant and output_indices, and the C output_data() branches that fill them are not
 * translated. The quaternion is rotated into the fundamental zone exactly as output_data() does
 * (output_conventional_orientation is always false in compute ptm/atom).
 */
import {
  PTM_CHECK_BCC, PTM_CHECK_DCUB, PTM_CHECK_DHEX, PTM_CHECK_FCC, PTM_CHECK_GRAPHENE, PTM_CHECK_HCP, PTM_CHECK_ICO,
  PTM_CHECK_SC, PTM_MATCH_BCC, PTM_MATCH_DCUB, PTM_MATCH_DHEX, PTM_MATCH_FCC, PTM_MATCH_GRAPHENE, PTM_MATCH_HCP,
  PTM_MATCH_ICO, PTM_MATCH_NONE, PTM_MATCH_SC, PTM_MAX_FACETS, PTM_MAX_INPUT_POINTS, PTM_MAX_POINTS,
  PTM_NUM_NBRS_DCUB, PTM_NUM_POINTS_BCC, PTM_NUM_POINTS_FCC, PTM_NUM_POINTS_SC, PTM_ALLOY_NONE,
} from './ptm_constants';
import type { ConvexHull, GetNeighbours, MatchResult } from './ptm_types';
import { find_alloy_type } from './ptm_alloy_types';
import { match_dcub_dhex, match_fcc_hcp_ico, match_general, match_graphene } from './ptm_structure_matcher';
import { normalize_vertices } from './ptm_normalize_vertices';
import { calculate_neighbour_ordering, calculate_two_shell_neighbour_ordering } from './ptm_neighbour_ordering';
import { rotate_quaternion_into_cubic_fundamental_zone, rotate_quaternion_into_diamond_cubic_fundamental_zone, rotate_quaternion_into_diamond_hexagonal_fundamental_zone, rotate_quaternion_into_hcp_fundamental_zone, rotate_quaternion_into_icosahedral_fundamental_zone } from './ptm_quat';
import { ptm_global, structure_bcc, structure_sc } from './ptm_initialize_data';
import type { voronoicell_neighbor } from './ptm_voronoi_cell';

/** Output of ptm_index() for the quantities compute ptm/atom reads (C out-parameters). */
export interface PtmIndexResult {
  p_type: number;
  p_alloy_type: number;
  p_scale: number;
  p_rmsd: number;
  q: number[];
  p_interatomic_distance: number;
}

/** static calculate_interatomic_distance (ptm_index.cpp) */
function calculate_interatomic_distance(type: number, scale: number): number {
  const c = [
    0,
    1,
    1,
    7.0 - 3.5 * Math.sqrt(3.0),
    1,
    1,
    (Math.sqrt(3.0) * 4.0) / (6 * Math.sqrt(2.0) + Math.sqrt(3.0)),
    (Math.sqrt(3.0) * 4.0) / (6 * Math.sqrt(2.0) + Math.sqrt(3.0)),
    -3.0 / 11 + (6 * Math.sqrt(3.0)) / 11,
  ];
  return c[type] / scale;
}

/** rotate_into_fundamental_zone (ptm_index.cpp) with output_conventional_orientation = false. */
function rotate_into_fundamental_zone(type: number, q: number[]): number {
  if (type === PTM_MATCH_SC) return rotate_quaternion_into_cubic_fundamental_zone(q);
  if (type === PTM_MATCH_FCC) return rotate_quaternion_into_cubic_fundamental_zone(q);
  if (type === PTM_MATCH_BCC) return rotate_quaternion_into_cubic_fundamental_zone(q);
  if (type === PTM_MATCH_ICO) return rotate_quaternion_into_icosahedral_fundamental_zone(q);
  if (type === PTM_MATCH_HCP || type === PTM_MATCH_GRAPHENE) return rotate_quaternion_into_hcp_fundamental_zone(q);
  if (type === PTM_MATCH_DCUB) return rotate_quaternion_into_diamond_cubic_fundamental_zone(q);
  if (type === PTM_MATCH_DHEX) return rotate_quaternion_into_diamond_hexagonal_fundamental_zone(q);
  return -1;
}

/** output_data() of ptm_index.cpp, restricted to the outputs compute ptm/atom uses. */
function output_data(res: MatchResult, numbers: number[], out: PtmIndexResult): void {
  const ref = res.ref_struct;
  if (ref === null) return;

  out.p_type = ref.type;
  out.p_alloy_type = find_alloy_type(ref, res.mapping, numbers);

  rotate_into_fundamental_zone(ref.type, res.q);

  const interatomic_distance = calculate_interatomic_distance(ref.type, res.scale);
  out.p_interatomic_distance = interatomic_distance;
  out.p_rmsd = res.rmsd;
  out.p_scale = res.scale;
  for (let k = 0; k < 4; k++) out.q[k] = res.q[k];
}

/** A fresh convexhull_t with zero-filled arrays (the C struct is a stack object with no initialiser). */
function newConvexHull(): ConvexHull {
  const facets: number[][] = [];
  const plane_normal: number[][] = [];
  for (let i = 0; i < PTM_MAX_FACETS; i++) {
    facets.push([0, 0, 0]);
    plane_normal.push([0, 0, 0]);
  }
  return {
    facets,
    plane_normal,
    processed: new Array<boolean>(PTM_MAX_POINTS).fill(false),
    initial_vertices: [0, 0, 0, 0],
    barycentre: [0, 0, 0],
    num_facets: 0,
    num_prev: 0,
    ok: false,
  };
}

/** A fresh point array of n rows of [x, y, z]. */
function newPoints(n: number): number[][] {
  const p: number[][] = [];
  for (let i = 0; i < n; i++) p.push([0, 0, 0]);
  return p;
}

/** A fresh MatchResult (res.ref_struct = nullptr, rmsd = INFINITY). */
function newMatchResult(): MatchResult {
  return {
    rmsd: Infinity,
    scale: 0,
    q: [0, 0, 0, 0],
    mapping: new Array<number>(PTM_MAX_POINTS).fill(0),
    ref_struct: null,
  };
}

/**
 * ptm_index(): identify the structure of atom atom_index. local_handle is the Voronoi handle from ptm_initialize_local
 *. flags is a PTM_CHECK_* mask. Returns the C result (PTM_NO_ERROR) and fills the
 * outputs: p_type is PTM_MATCH_NONE when nothing matched; p_rmsd keeps the C initial value (Infinity) in that case.
 */
export function ptm_index(
  local_handle: voronoicell_neighbor,
  atom_index: number,
  get_neighbours: GetNeighbours,
  nbrlist: unknown,
  flags: number,
): PtmIndexResult {
  if (!ptm_global.initialized) throw new Error('ptm_index: ptm_initialize_global() has not run');

  const res = newMatchResult();

  const ordering: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const numbers: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const points = newPoints(PTM_MAX_INPUT_POINTS);

  const dordering: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const dnumbers: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const dpoints = newPoints(PTM_MAX_INPUT_POINTS);

  const gordering: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const gnumbers: number[] = new Array<number>(PTM_MAX_INPUT_POINTS).fill(0);
  const gpoints = newPoints(PTM_MAX_INPUT_POINTS);

  const ch = newConvexHull();
  const ch_points = newPoints(PTM_MAX_INPUT_POINTS);
  let num_lpoints = 0;

  if (flags & (PTM_CHECK_SC | PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO | PTM_CHECK_BCC)) {
    let min_points = PTM_NUM_POINTS_SC;
    if (flags & (PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO)) min_points = PTM_NUM_POINTS_FCC;
    if (flags & PTM_CHECK_BCC) min_points = PTM_NUM_POINTS_BCC;

    num_lpoints = calculate_neighbour_ordering(
      local_handle, atom_index, min_points, get_neighbours, nbrlist, false, ordering, points, numbers,
    );
    if (num_lpoints >= min_points) {
      normalize_vertices(num_lpoints, points, ch_points);
      ch.ok = false;

      if (flags & PTM_CHECK_SC) match_general(structure_sc, ch_points, points, ch, res);

      if (flags & (PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO)) match_fcc_hcp_ico(ch_points, points, flags, ch, res);

      if (flags & PTM_CHECK_BCC) match_general(structure_bcc, ch_points, points, ch, res);
    }
  }

  if (flags & (PTM_CHECK_DCUB | PTM_CHECK_DHEX)) {
    const num_inner = 4;
    const num_outer = 3;
    const max_snbrs = 12;
    const ret = calculate_two_shell_neighbour_ordering(
      local_handle, atom_index, get_neighbours, nbrlist, num_inner, num_outer, max_snbrs, false, dordering, dpoints, dnumbers,
    );
    if (ret === 0) {
      normalize_vertices(PTM_NUM_NBRS_DCUB + 1, dpoints, ch_points);
      ch.ok = false;

      match_dcub_dhex(ch_points, dpoints, flags, ch, res);
    }
  }

  if (flags & PTM_CHECK_GRAPHENE) {
    const num_inner = 3;
    const num_outer = 2;
    const max_snbrs = 12;
    const ret = calculate_two_shell_neighbour_ordering(
      local_handle, atom_index, get_neighbours, nbrlist, num_inner, num_outer, max_snbrs, true, gordering, gpoints, gnumbers,
    );
    if (ret === 0) {
      match_graphene(gpoints, res);
    }
  }

  const out: PtmIndexResult = {
    p_type: PTM_MATCH_NONE,
    p_alloy_type: PTM_ALLOY_NONE,
    p_scale: 0,
    p_rmsd: Infinity,
    q: [0, 0, 0, 0],
    p_interatomic_distance: 0,
  };

  if (res.ref_struct === null) return out;

  // The C code calls output_data() with the point set and numbers of the branch that matched.
  if (res.ref_struct.type === PTM_MATCH_DCUB || res.ref_struct.type === PTM_MATCH_DHEX) {
    output_data(res, dnumbers, out);
  } else if (res.ref_struct.type === PTM_MATCH_GRAPHENE) {
    output_data(res, gnumbers, out);
  } else {
    output_data(res, numbers, out);
  }
  return out;
}
