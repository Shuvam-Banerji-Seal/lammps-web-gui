/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_structure_matcher.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import {
  PTM_CHECK_DCUB,
  PTM_CHECK_DHEX,
  PTM_CHECK_FCC,
  PTM_CHECK_HCP,
  PTM_CHECK_ICO,
  PTM_MATCH_SC,
  PTM_MAX_EDGES,
  PTM_MAX_FACETS,
  PTM_MAX_NBRS,
  PTM_MAX_POINTS,
  PTM_NO_ERROR,
} from './ptm_constants';
import { ConvexHull, MatchResult, RefData } from './ptm_types';
import { get_convex_hull } from './ptm_convex_hull_incremental';
import { canonical_form_coloured } from './ptm_canonical_coloured';
import { automorphisms } from './ptm_data';
import { graph_degree } from './ptm_graph_tools';
import { subtract_barycentre } from './ptm_normalize_vertices';
import { FastCalcRMSDAndRotation, InnerProduct } from './ptm_polar';
import { structure_dcub, structure_dhex, structure_fcc, structure_graphene, structure_hcp, structure_ico } from './ptm_initialize_data';

function calc_rmsd(
  num_points: number,
  ideal_points: number[][],
  normalized: number[][],
  mapping: number[],
  G1: number,
  G2: number,
  E0: number,
  q: number[],
): { rmsd: number; scale: number } {
  const A0 = new Array<number>(9).fill(0);
  InnerProduct(A0, num_points, ideal_points, normalized, mapping);

  const rot = new Array<number>(9).fill(0);
  FastCalcRMSDAndRotation(A0, E0, q, rot);

  let k0 = 0;
  for (let i = 0; i < num_points; i++) {
    for (let j = 0; j < 3; j++) {
      let v = 0.0;
      for (let k = 0; k < 3; k++) v += rot[j * 3 + k] * ideal_points[i][k];

      k0 += v * normalized[mapping[i]][j];
    }
  }

  const scale = k0 / G2;
  return { rmsd: Math.sqrt(Math.abs(G1 - scale * k0) / num_points), scale };
}

function check_graphs(s: RefData, hash: bigint, canonical_labelling: number[], normalized: number[][], res: MatchResult): void {
  const num_points = s.num_nbrs + 1;
  const ideal_points = s.points!;
  const inverse_labelling = new Array<number>(PTM_MAX_POINTS).fill(0);
  const mapping = new Array<number>(PTM_MAX_POINTS).fill(0);

  for (let i = 0; i < num_points; i++) inverse_labelling[canonical_labelling[i]] = i;

  let G1 = 0, G2 = 0;
  for (let i = 0; i < num_points; i++) {
    const x1 = ideal_points[i][0];
    const y1 = ideal_points[i][1];
    const z1 = ideal_points[i][2];

    const x2 = normalized[i][0];
    const y2 = normalized[i][1];
    const z2 = normalized[i][2];

    G1 += x1 * x1 + y1 * y1 + z1 * z1;
    G2 += x2 * x2 + y2 * y2 + z2 * z2;
  }
  const E0 = (G1 + G2) / 2;

  for (let i = 0; i < s.num_graphs; i++) {
    if (hash !== s.graphs![i].hash) continue;

    const gref = s.graphs![i];
    for (let j = 0; j < gref.num_automorphisms; j++) {
      for (let k = 0; k < num_points; k++)
        mapping[automorphisms[gref.automorphism_index + j][k]] = inverse_labelling[gref.canonical_labelling[k]];

      const q = [0, 0, 0, 0];
      const { rmsd, scale } = calc_rmsd(num_points, ideal_points, normalized, mapping, G1, G2, E0, q);
      if (rmsd < res.rmsd) {
        res.rmsd = rmsd;
        res.scale = scale;
        res.ref_struct = s;
        for (let k = 0; k < 4; k++) res.q[k] = q[k];
        for (let k = 0; k < num_points; k++) res.mapping[k] = mapping[k];
      }
    }
  }
}

export function match_general(s: RefData, ch_points: number[][], points: number[][], ch: ConvexHull, res: MatchResult): number {
  const degree = new Array<number>(PTM_MAX_NBRS).fill(0);
  const facets = Array.from({ length: PTM_MAX_FACETS }, () => [0, 0, 0]);

  let ret = get_convex_hull(s.num_nbrs + 1, ch_points, ch, facets);
  ch.ok = ret >= 0;
  if (ret !== 0) return PTM_NO_ERROR;

  if (ch.num_facets !== s.num_facets) return PTM_NO_ERROR; //incorrect number of facets in convex hull

  const max_degree = graph_degree(s.num_facets, facets, s.num_nbrs, degree);
  if (max_degree > s.max_degree) return PTM_NO_ERROR;

  if (s.type === PTM_MATCH_SC)
    for (let i = 0; i < s.num_nbrs; i++) if (degree[i] !== 4) return PTM_NO_ERROR;

  const normalized = Array.from({ length: PTM_MAX_POINTS }, () => [0, 0, 0]);
  subtract_barycentre(s.num_nbrs + 1, points, normalized);

  const code = new Array<number>(2 * PTM_MAX_EDGES).fill(0);
  const colours = new Array<number>(PTM_MAX_POINTS).fill(0);
  const canonical_labelling = new Array<number>(PTM_MAX_POINTS).fill(0);
  const hashOut = { value: 0n };
  ret = canonical_form_coloured(s.num_facets, facets, s.num_nbrs, degree, colours, canonical_labelling, code, hashOut);
  if (ret !== PTM_NO_ERROR) return ret;

  check_graphs(s, hashOut.value, canonical_labelling, normalized, res);
  return PTM_NO_ERROR;
}

export function match_fcc_hcp_ico(ch_points: number[][], points: number[][], flags: number, ch: ConvexHull, res: MatchResult): number {
  const num_nbrs = structure_fcc.num_nbrs;
  const num_facets = structure_fcc.num_facets;
  const max_degree = structure_fcc.max_degree;

  const degree = new Array<number>(PTM_MAX_NBRS).fill(0);
  const facets = Array.from({ length: PTM_MAX_FACETS }, () => [0, 0, 0]);

  let ret = get_convex_hull(num_nbrs + 1, ch_points, ch, facets);
  ch.ok = ret >= 0;
  if (ret !== 0) return PTM_NO_ERROR;

  if (ch.num_facets !== num_facets) return PTM_NO_ERROR; //incorrect number of facets in convex hull

  const _max_degree = graph_degree(num_facets, facets, num_nbrs, degree);
  if (_max_degree > max_degree) return PTM_NO_ERROR;

  const normalized = Array.from({ length: PTM_MAX_POINTS }, () => [0, 0, 0]);
  subtract_barycentre(num_nbrs + 1, points, normalized);

  const code = new Array<number>(2 * PTM_MAX_EDGES).fill(0);
  const colours = new Array<number>(PTM_MAX_POINTS).fill(0);
  const canonical_labelling = new Array<number>(PTM_MAX_POINTS).fill(0);
  const hashOut = { value: 0n };
  ret = canonical_form_coloured(num_facets, facets, num_nbrs, degree, colours, canonical_labelling, code, hashOut);
  if (ret !== PTM_NO_ERROR) return ret;

  if (flags & PTM_CHECK_FCC) check_graphs(structure_fcc, hashOut.value, canonical_labelling, normalized, res);
  if (flags & PTM_CHECK_HCP) check_graphs(structure_hcp, hashOut.value, canonical_labelling, normalized, res);
  if (flags & PTM_CHECK_ICO) check_graphs(structure_ico, hashOut.value, canonical_labelling, normalized, res);
  return PTM_NO_ERROR;
}

export function match_dcub_dhex(ch_points: number[][], points: number[][], flags: number, ch: ConvexHull, res: MatchResult): number {
  const num_nbrs = structure_dcub.num_nbrs;
  const num_facets = structure_fcc.num_facets;
  const max_degree = structure_dcub.max_degree;

  const facets = Array.from({ length: PTM_MAX_FACETS }, () => [0, 0, 0]);
  let ret = get_convex_hull(num_nbrs + 1, ch_points, ch, facets);
  ch.ok = ret >= 0;
  if (ret !== 0) return PTM_NO_ERROR;

  //check for facets with multiple inner atoms
  const inverted = [false, false, false, false];
  for (let i = 0; i < ch.num_facets; i++) {
    let n = 0;
    for (let j = 0; j < 3; j++) {
      if (facets[i][j] <= 3) {
        inverted[facets[i][j]] = true;
        n++;
      }
    }
    if (n > 1) return PTM_NO_ERROR;
  }

  let num_inverted = 0;
  for (let i = 0; i < 4; i++) num_inverted += inverted[i] ? 1 : 0;

  if (ch.num_facets !== num_facets + 2 * num_inverted) return PTM_NO_ERROR; //incorrect number of facets in convex hull

  const degree = new Array<number>(PTM_MAX_NBRS).fill(0);
  let _max_degree = graph_degree(num_facets, facets, num_nbrs, degree);
  if (_max_degree > max_degree) return PTM_NO_ERROR;

  let num_found = 0;
  const toadd = Array.from({ length: 4 }, () => [0, 0, 0]);
  for (let i = 0; i < ch.num_facets; i++) {
    const a = facets[i][0];
    const b = facets[i][1];
    const c = facets[i][2];
    if (a <= 3 || b <= 3 || c <= 3) continue;

    const i0 = Math.trunc((a - 4) / 3);
    const i1 = Math.trunc((b - 4) / 3);
    const i2 = Math.trunc((c - 4) / 3);

    if (i0 === i1 && i0 === i2) {
      if (num_found + num_inverted >= 4) return PTM_NO_ERROR;

      toadd[num_found][0] = a;
      toadd[num_found][1] = b;
      toadd[num_found][2] = c;
      num_found++;

      for (let k = 0; k < 3; k++) facets[i][k] = facets[ch.num_facets - 1][k];
      ch.num_facets--;
      i--;
    }
  }

  if (num_found + num_inverted !== 4) return PTM_NO_ERROR;

  for (let i = 0; i < num_found; i++) {
    const a = toadd[i][0];
    const b = toadd[i][1];
    const c = toadd[i][2];

    const i0 = Math.trunc((a - 4) / 3);

    facets[ch.num_facets][0] = i0;
    facets[ch.num_facets][1] = b;
    facets[ch.num_facets][2] = c;
    ch.num_facets++;

    facets[ch.num_facets][0] = a;
    facets[ch.num_facets][1] = i0;
    facets[ch.num_facets][2] = c;
    ch.num_facets++;

    facets[ch.num_facets][0] = a;
    facets[ch.num_facets][1] = b;
    facets[ch.num_facets][2] = i0;
    ch.num_facets++;
  }

  _max_degree = graph_degree(ch.num_facets, facets, num_nbrs, degree);
  if (_max_degree > max_degree) return PTM_NO_ERROR;

  const normalized = Array.from({ length: PTM_MAX_POINTS }, () => [0, 0, 0]);
  subtract_barycentre(num_nbrs + 1, points, normalized);

  const code = new Array<number>(2 * PTM_MAX_EDGES).fill(0);
  const colours = new Array<number>(PTM_MAX_POINTS).fill(0);
  colours[0] = 1;
  colours[1] = 1;
  colours[2] = 1;
  colours[3] = 1;
  const canonical_labelling = new Array<number>(PTM_MAX_POINTS).fill(0);
  const hashOut = { value: 0n };
  ret = canonical_form_coloured(ch.num_facets, facets, num_nbrs, degree, colours, canonical_labelling, code, hashOut);
  if (ret !== PTM_NO_ERROR) return ret;

  if (flags & PTM_CHECK_DCUB) check_graphs(structure_dcub, hashOut.value, canonical_labelling, normalized, res);
  if (flags & PTM_CHECK_DHEX) check_graphs(structure_dhex, hashOut.value, canonical_labelling, normalized, res);

  return PTM_NO_ERROR;
}

function check_graphs_graphene(
  s: RefData,
  num_points: number,
  ideal_points: number[][],
  normalized: number[][],
  mapping: number[],
  res: MatchResult,
): void {
  let G1 = 0, G2 = 0;
  for (let i = 0; i < num_points; i++) {
    const x1 = ideal_points[i][0];
    const y1 = ideal_points[i][1];
    const z1 = ideal_points[i][2];

    const x2 = normalized[i][0];
    const y2 = normalized[i][1];
    const z2 = normalized[i][2];

    G1 += x1 * x1 + y1 * y1 + z1 * z1;
    G2 += x2 * x2 + y2 * y2 + z2 * z2;
  }
  const E0 = (G1 + G2) / 2;

  const q = [0, 0, 0, 0];
  const { rmsd, scale } = calc_rmsd(num_points, ideal_points, normalized, mapping, G1, G2, E0, q);
  if (rmsd < res.rmsd) {
    res.rmsd = rmsd;
    res.scale = scale;
    res.ref_struct = s;
    for (let k = 0; k < 4; k++) res.q[k] = q[k];
    for (let k = 0; k < num_points; k++) res.mapping[k] = mapping[k];
  }
}

export function match_graphene(points: number[][], res: MatchResult): number {
  const num_nbrs = structure_graphene.num_nbrs;
  const num_points = num_nbrs + 1;
  const ideal_points = structure_graphene.points!;

  const normalized = Array.from({ length: PTM_MAX_POINTS }, () => [0, 0, 0]);
  subtract_barycentre(num_points, points, normalized);

  const mapping = new Array<number>(PTM_MAX_POINTS).fill(0);
  for (let i = 0; i < num_points; i++) mapping[i] = i;

  for (let i = 0; i < 2; i++) {
    [mapping[4], mapping[5]] = [mapping[5], mapping[4]];

    for (let j = 0; j < 2; j++) {
      [mapping[6], mapping[7]] = [mapping[7], mapping[6]];

      for (let k = 0; k < 2; k++) {
        [mapping[8], mapping[9]] = [mapping[9], mapping[8]];

        check_graphs_graphene(structure_graphene, num_points, ideal_points, normalized, mapping, res);
      }
    }
  }

  return PTM_NO_ERROR;
}
