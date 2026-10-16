import { describe, expect, it } from 'vitest';
import {
  PTM_CHECK_BCC, PTM_CHECK_DEFAULT, PTM_CHECK_FCC, PTM_CHECK_HCP, PTM_CHECK_ICO,
  PTM_MATCH_BCC, PTM_MATCH_FCC, PTM_MATCH_HCP, PTM_MATCH_ICO, PTM_MATCH_SC,
  PTM_NUM_POINTS_BCC, PTM_NUM_POINTS_FCC, PTM_NUM_POINTS_HCP, PTM_NUM_POINTS_ICO, PTM_NUM_POINTS_SC,
  PTM_MAX_POINTS,
} from '../src/engine-gpl/ptm/ptm_constants';
import { ptm_template_bcc, ptm_template_fcc, ptm_template_hcp, ptm_template_ico, ptm_template_sc } from '../src/engine-gpl/ptm/ptm_data';
import { ptm_initialize_global, ptm_uninitialize_local, structure_bcc, structure_sc } from '../src/engine-gpl/ptm/ptm_initialize_data';
import { ptm_index } from '../src/engine-gpl/ptm/ptm_index';
import { match_fcc_hcp_ico, match_general } from '../src/engine-gpl/ptm/ptm_structure_matcher';
import { normalize_vertices } from '../src/engine-gpl/ptm/ptm_normalize_vertices';
import { ptm_initialize_local } from '../src/engine-gpl/ptm/ptm_initialize_data';
import { add_facet, get_convex_hull } from '../src/engine-gpl/ptm/ptm_convex_hull_incremental';
import type { ConvexHull, MatchResult } from '../src/engine-gpl/ptm/ptm_types';

/** Fresh convex hull and match result, as ptm_index() creates them. */
const newHull = (): ConvexHull => ({
  facets: Array.from({ length: 28 }, () => [0, 0, 0]),
  plane_normal: Array.from({ length: 28 }, () => [0, 0, 0]),
  processed: new Array<boolean>(PTM_MAX_POINTS).fill(false),
  initial_vertices: [0, 0, 0, 0],
  barycentre: [0, 0, 0],
  num_facets: 0,
  num_prev: 0,
  ok: false,
});
const newRes = (): MatchResult => ({ rmsd: Infinity, scale: 0, q: [0, 0, 0, 0], mapping: new Array<number>(PTM_MAX_POINTS).fill(0), ref_struct: null });

/** Ideal template point set (origin first), as a fresh copy. */
const copy = (p: number[][], n: number): number[][] => Array.from({ length: n }, (_, i) => [p[i][0], p[i][1], p[i][2]]);

describe('PTM structure matching on ideal neighbourhoods (no Voronoi ordering)', () => {
  it('initialises the reference graphs', () => {
    expect(ptm_initialize_global()).toBe(0);
    expect(structure_bcc.num_nbrs).toBe(14);
  });

  it('identifies fcc, hcp, ico from the ideal templates with rmsd ~ 0', () => {
    ptm_initialize_global();
    const cases: [number[][], number, number][] = [
      [ptm_template_fcc, PTM_NUM_POINTS_FCC, PTM_MATCH_FCC],
      [ptm_template_hcp, PTM_NUM_POINTS_HCP, PTM_MATCH_HCP],
      [ptm_template_ico, PTM_NUM_POINTS_ICO, PTM_MATCH_ICO],
    ];
    for (const [tpl, n, want] of cases) {
      const points = copy(tpl, n);
      const chPoints = copy(tpl, n);
      normalize_vertices(n, points, chPoints);
      const res = newRes();
      match_fcc_hcp_ico(chPoints, points, PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO, newHull(), res);
      expect(res.ref_struct?.type).toBe(want);
      expect(res.rmsd).toBeLessThan(1e-8);
    }
  });

  it('identifies bcc and sc from their ideal templates', () => {
    ptm_initialize_global();
    const bcc = copy(ptm_template_bcc, PTM_NUM_POINTS_BCC);
    const bccN = copy(ptm_template_bcc, PTM_NUM_POINTS_BCC);
    normalize_vertices(PTM_NUM_POINTS_BCC, bcc, bccN);
    const rb = newRes();
    match_general(structure_bcc, bccN, bcc, newHull(), rb);
    expect(rb.ref_struct?.type).toBe(PTM_MATCH_BCC);

    const sc = copy(ptm_template_sc, PTM_NUM_POINTS_SC);
    const scN = copy(ptm_template_sc, PTM_NUM_POINTS_SC);
    normalize_vertices(PTM_NUM_POINTS_SC, sc, scN);
    const rs = newRes();
    match_general(structure_sc, scN, sc, newHull(), rs);
    expect(rs.ref_struct?.type).toBe(PTM_MATCH_SC);
    // the C sc template is stored to ~8 digits, so its ideal rmsd is ~1e-8 rather than round-off
    expect(rs.rmsd).toBeLessThan(1e-6);
  });

  it('keeps the detection flag set used by compute ptm/atom default', () => {
    expect(PTM_CHECK_DEFAULT).toBe(PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO | PTM_CHECK_BCC);
  });

  it('builds a convex hull over a tetrahedron (hull code path)', () => {
    const pts = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const hull = newHull();
    const simplex = Array.from({ length: 28 }, () => [0, 0, 0]);
    expect(get_convex_hull(4, pts, hull, simplex)).toBeGreaterThanOrEqual(0);
    expect(hull.num_facets).toBeGreaterThan(0);
    const scratch = newHull();
    add_facet(pts, 0, 1, 2, scratch.facets[0], scratch.plane_normal[0], scratch.barycentre);
  });

  it('identifies an ideal fcc neighbourhood end to end (Voronoi neighbour ordering, ptm_index)', () => {
    ptm_initialize_global();
    const handle = ptm_initialize_local();
    // central atom at the origin, its 12 nearest neighbours at the ideal fcc template positions (no periodic images)
    const tpl = copy(ptm_template_fcc, PTM_NUM_POINTS_FCC);
    const get_neighbours = (_v: unknown, _c: number, _a: number, num: number, idx: number[], nums: number[], pos: number[][]): number => {
      const m = Math.min(num, tpl.length);
      for (let k = 0; k < m; k++) {
        pos[k][0] = tpl[k][0];
        pos[k][1] = tpl[k][1];
        pos[k][2] = tpl[k][2];
        idx[k] = k;
        nums[k] = 0;
      }
      return m;
    };
    // With PTM_CHECK_BCC in the flags ptm_index.cpp raises min_points to the bcc count (15), so a 13-neighbour
    // environment is skipped there; the fcc/hcp/ico flags need 13 points, as used here.
    const r = ptm_index(handle, 0, get_neighbours, null, PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_ICO);
    expect(r.p_type).toBe(PTM_MATCH_FCC);
    expect(r.p_rmsd).toBeLessThan(1e-6);
    ptm_uninitialize_local(handle);
  });
});
