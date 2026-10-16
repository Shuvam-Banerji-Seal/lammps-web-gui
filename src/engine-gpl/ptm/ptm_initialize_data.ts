/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_initialize_data.h and src/PTM/ptm_initialize_data.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import {
  NUM_BCC_GRAPHS,
  NUM_CUBIC_MAPPINGS,
  NUM_DCUB_GRAPHS,
  NUM_DCUB_MAPPINGS,
  NUM_DHEX_GRAPHS,
  NUM_DHEX_MAPPINGS,
  NUM_FCC_GRAPHS,
  NUM_HCP_GRAPHS,
  NUM_HEX_MAPPINGS,
  NUM_ICO_GRAPHS,
  NUM_ICO_MAPPINGS,
  NUM_SC_GRAPHS,
  PTM_MATCH_BCC,
  PTM_MATCH_DCUB,
  PTM_MATCH_DHEX,
  PTM_MATCH_FCC,
  PTM_MATCH_GRAPHENE,
  PTM_MATCH_HCP,
  PTM_MATCH_ICO,
  PTM_MATCH_SC,
  PTM_MAX_EDGES,
  PTM_MAX_NBRS,
  PTM_MAX_POINTS,
  PTM_NO_ERROR,
} from './ptm_constants';
import { Graph, RefData } from './ptm_types';
import {
  graphs_bcc,
  graphs_dcub,
  graphs_dhex,
  graphs_fcc,
  graphs_hcp,
  graphs_ico,
  graphs_sc,
  mapping_bcc,
  mapping_dcub,
  mapping_dcub_conventional,
  mapping_dhex,
  mapping_dhex_conventional,
  mapping_fcc,
  mapping_graphene,
  mapping_graphene_conventional,
  mapping_hcp,
  mapping_hcp_conventional,
  mapping_ico,
  mapping_sc,
  penrose_bcc,
  penrose_dcub,
  penrose_dcub_alt1,
  penrose_dhex,
  penrose_dhex_alt1,
  penrose_dhex_alt2,
  penrose_dhex_alt3,
  penrose_fcc,
  penrose_graphene,
  penrose_graphene_alt1,
  penrose_hcp,
  penrose_hcp_alt1,
  penrose_ico,
  penrose_sc,
  ptm_template_bcc,
  ptm_template_dcub,
  ptm_template_dcub_alt1,
  ptm_template_dhex,
  ptm_template_dhex_alt1,
  ptm_template_dhex_alt2,
  ptm_template_dhex_alt3,
  ptm_template_fcc,
  ptm_template_graphene,
  ptm_template_graphene_alt1,
  ptm_template_hcp,
  ptm_template_hcp_alt1,
  ptm_template_ico,
  ptm_template_sc,
  template_indices_dcub,
  template_indices_dhex,
  template_indices_graphene,
  template_indices_hcp,
} from './ptm_data';
import { voronoi_initialize_local, voronoi_uninitialize_local } from './ptm_neighbour_ordering';
import type { voronoicell_neighbor } from './ptm_voronoi_cell';
import { canonical_form_coloured } from './ptm_canonical_coloured';
import { add_facet } from './ptm_convex_hull_incremental';
import { graph_degree } from './ptm_graph_tools';

export const structure_sc: RefData = {
  type: PTM_MATCH_SC,
  num_nbrs: 6,
  num_facets: 8,
  max_degree: 4,
  num_graphs: NUM_SC_GRAPHS,
  graphs: graphs_sc,
  points: ptm_template_sc,
  points_alt1: null,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_sc,
  penrose_alt1: null,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_CUBIC_MAPPINGS,
  mapping: mapping_sc,
  mapping_conventional: null,
  template_indices: null,
};

export const structure_fcc: RefData = {
  type: PTM_MATCH_FCC,
  num_nbrs: 12,
  num_facets: 20,
  max_degree: 6,
  num_graphs: NUM_FCC_GRAPHS,
  graphs: graphs_fcc,
  points: ptm_template_fcc,
  points_alt1: null,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_fcc,
  penrose_alt1: null,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_CUBIC_MAPPINGS,
  mapping: mapping_fcc,
  mapping_conventional: null,
  template_indices: null,
};

export const structure_hcp: RefData = {
  type: PTM_MATCH_HCP,
  num_nbrs: 12,
  num_facets: 20,
  max_degree: 6,
  num_graphs: NUM_HCP_GRAPHS,
  graphs: graphs_hcp,
  points: ptm_template_hcp,
  points_alt1: ptm_template_hcp_alt1,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_hcp,
  penrose_alt1: penrose_hcp_alt1,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_HEX_MAPPINGS,
  mapping: mapping_hcp,
  mapping_conventional: mapping_hcp_conventional,
  template_indices: template_indices_hcp,
};

export const structure_ico: RefData = {
  type: PTM_MATCH_ICO,
  num_nbrs: 12,
  num_facets: 20,
  max_degree: 6,
  num_graphs: NUM_ICO_GRAPHS,
  graphs: graphs_ico,
  points: ptm_template_ico,
  points_alt1: null,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_ico,
  penrose_alt1: null,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_ICO_MAPPINGS,
  mapping: mapping_ico,
  mapping_conventional: null,
  template_indices: null,
};

export const structure_bcc: RefData = {
  type: PTM_MATCH_BCC,
  num_nbrs: 14,
  num_facets: 24,
  max_degree: 8,
  num_graphs: NUM_BCC_GRAPHS,
  graphs: graphs_bcc,
  points: ptm_template_bcc,
  points_alt1: null,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_bcc,
  penrose_alt1: null,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_CUBIC_MAPPINGS,
  mapping: mapping_bcc,
  mapping_conventional: null,
  template_indices: null,
};

export const structure_dcub: RefData = {
  type: PTM_MATCH_DCUB,
  num_nbrs: 16,
  num_facets: 28,
  max_degree: 8,
  num_graphs: NUM_DCUB_GRAPHS,
  graphs: graphs_dcub,
  points: ptm_template_dcub,
  points_alt1: ptm_template_dcub_alt1,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_dcub,
  penrose_alt1: penrose_dcub_alt1,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: NUM_DCUB_MAPPINGS,
  mapping: mapping_dcub,
  mapping_conventional: mapping_dcub_conventional,
  template_indices: template_indices_dcub,
};

export const structure_dhex: RefData = {
  type: PTM_MATCH_DHEX,
  num_nbrs: 16,
  num_facets: 28,
  max_degree: 8,
  num_graphs: NUM_DHEX_GRAPHS,
  graphs: graphs_dhex,
  points: ptm_template_dhex,
  points_alt1: ptm_template_dhex_alt1,
  points_alt2: ptm_template_dhex_alt2,
  points_alt3: ptm_template_dhex_alt3,
  penrose: penrose_dhex,
  penrose_alt1: penrose_dhex_alt1,
  penrose_alt2: penrose_dhex_alt2,
  penrose_alt3: penrose_dhex_alt3,
  num_mappings: NUM_DHEX_MAPPINGS,
  mapping: mapping_dhex,
  mapping_conventional: mapping_dhex_conventional,
  template_indices: template_indices_dhex,
};

export const structure_graphene: RefData = {
  type: PTM_MATCH_GRAPHENE,
  num_nbrs: 9,
  num_facets: -1,
  max_degree: -1,
  num_graphs: -1,
  graphs: null,
  points: ptm_template_graphene,
  points_alt1: ptm_template_graphene_alt1,
  points_alt2: null,
  points_alt3: null,
  penrose: penrose_graphene,
  penrose_alt1: penrose_graphene_alt1,
  penrose_alt2: null,
  penrose_alt3: null,
  num_mappings: -1,
  mapping: mapping_graphene,
  mapping_conventional: mapping_graphene_conventional,
  template_indices: template_indices_graphene,
};

/** ptm_initialized (C global): set once ptm_initialize_global() has succeeded. */
export const ptm_global = { initialized: false };

function make_facets_clockwise(num_facets: number, facets: number[][], points: number[][]): void {
  const plane_normal = [0, 0, 0];
  const origin = [0, 0, 0];

  for (let i = 0; i < num_facets; i++) add_facet(points, facets[i][0], facets[i][1], facets[i][2], facets[i], plane_normal, origin);
}

function initialize_graphs(s: RefData, colours: number[]): number {
  for (let i = 0; i < s.num_graphs; i++) {
    const graph: Graph = s.graphs![i];
    const code = new Array<number>(2 * PTM_MAX_EDGES).fill(0);
    const degree = new Array<number>(PTM_MAX_NBRS).fill(0);
    const _max_degree = graph_degree(s.num_facets, graph.facets, s.num_nbrs, degree);
    // C assert(_max_degree <= s->max_degree): a violation is a corrupt template table, so fail loudly.
    if (!(_max_degree <= s.max_degree)) throw new Error(`PTM initialize_graphs: max degree ${_max_degree} exceeds ${s.max_degree}`);

    // &s->points[1]: the neighbour points start at index 1 (index 0 is the central atom).
    make_facets_clockwise(s.num_facets, graph.facets, s.points!.slice(1));
    const hashOut = { value: graph.hash };
    const ret = canonical_form_coloured(s.num_facets, graph.facets, s.num_nbrs, degree, colours, graph.canonical_labelling, code, hashOut);
    graph.hash = hashOut.value;
    if (ret !== 0) return ret;
  }

  return PTM_NO_ERROR;
}

export function ptm_initialize_global(): number {
  if (ptm_global.initialized) return PTM_NO_ERROR;

  const colours = new Array<number>(PTM_MAX_POINTS).fill(0);
  const dcolours = new Array<number>(PTM_MAX_POINTS).fill(0);
  dcolours[0] = 1;
  dcolours[1] = 1;
  dcolours[2] = 1;
  dcolours[3] = 1;

  let ret = initialize_graphs(structure_sc, colours);
  ret |= initialize_graphs(structure_fcc, colours);
  ret |= initialize_graphs(structure_hcp, colours);
  ret |= initialize_graphs(structure_ico, colours);
  ret |= initialize_graphs(structure_bcc, colours);
  ret |= initialize_graphs(structure_dcub, dcolours);
  ret |= initialize_graphs(structure_dhex, dcolours);

  if (ret === PTM_NO_ERROR) ptm_global.initialized = true;

  return ret;
}

export function ptm_initialize_local(): voronoicell_neighbor {
  if (!ptm_global.initialized) throw new Error('ptm_initialize_local called before ptm_initialize_global');
  return voronoi_initialize_local();
}

export function ptm_uninitialize_local(handle: voronoicell_neighbor): void {
  voronoi_uninitialize_local(handle);
}
