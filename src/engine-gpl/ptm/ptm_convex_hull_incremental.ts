/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_convex_hull_incremental.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { ConvexHull } from './ptm_types';
import { PTM_MAX_FACETS, PTM_MAX_POINTS } from './ptm_constants';

const VISIBLE = 1;
const INVISIBLE = 2;
const BOTH = 3;
const TOLERANCE = 1e-8;

function norm_squared(p: number[]): number {
  const x = p[0];
  const y = p[1];
  const z = p[2];

  return x * x + y * y + z * z;
}

function dot_product(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross_product(a: number[], b: number[], c: number[]): void {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

function calculate_plane_normal(points: number[][], a: number, b: number, c: number, plane_normal: number[]): void {
  const u = [points[b][0] - points[a][0], points[b][1] - points[a][1], points[b][2] - points[a][2]];

  const v = [points[c][0] - points[a][0], points[c][1] - points[a][1], points[c][2] - points[a][2]];

  cross_product(u, v, plane_normal);
  const norm = Math.sqrt(norm_squared(plane_normal));
  plane_normal[0] /= norm;
  plane_normal[1] /= norm;
  plane_normal[2] /= norm;
}

function point_plane_distance(w: number[], plane_point: number[], plane_cross: number[]): number {
  return plane_cross[0] * (plane_point[0] - w[0]) + plane_cross[1] * (plane_point[1] - w[1]) +
    plane_cross[2] * (plane_point[2] - w[2]);
}

function calc_max_extent(num_points: number, points: number[][], min_index: number[], max_index: number[]): boolean {
  for (let j = 0; j < 3; j++) {
    let dmin = Number.MAX_VALUE;
    let dmax = -Number.MAX_VALUE;
    let imin = 0;
    let imax = 0;

    for (let i = 0; i < num_points; i++) {
      const d = points[i][j];
      if (d < dmin) {
        dmin = d;
        imin = i;
      }
      if (d > dmax) {
        dmax = d;
        imax = i;
      }
    }

    if (imin === imax) return false;    //degenerate point set

    min_index[j] = imin;
    max_index[j] = imax;
  }

  return true;
}

// p_c is the C out-parameter (int *p_c): p_c[0] receives the index.
function find_third_point(num_points: number, points: number[][], a: number, b: number, p_c: number[]): boolean {
  const x1 = points[a];
  const x2 = points[b];

  const x2x1 = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]];
  const ns_x2x1 = norm_squared(x2x1);

  let bi = -1;
  let max_dist = 0.0;
  for (let i = 0; i < num_points; i++) {
    if (i === a || i === b) continue;

    const x0 = points[i];

    const x1x0 = [x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]];
    const dot = dot_product(x1x0, x2x1);
    const dist = (norm_squared(x1x0) * ns_x2x1 - dot * dot) / ns_x2x1;

    if (dist > max_dist) {
      max_dist = dist;
      bi = i;
    }
  }

  p_c[0] = bi;
  return max_dist > TOLERANCE;
}

// p_d is the C out-parameter (int *p_d): p_d[0] receives the index.
function find_fourth_point(num_points: number, points: number[][], a: number, b: number, c: number, p_d: number[]): boolean {
  const plane_normal = [0, 0, 0];
  calculate_plane_normal(points, a, b, c, plane_normal);

  let bi = -1;
  let max_dist = 0.0;
  for (let i = 0; i < num_points; i++) {
    if (i === a || i === b || i === c) continue;

    const x0 = points[i];
    const dist = Math.abs(point_plane_distance(x0, points[a], plane_normal));
    if (dist > max_dist) {
      max_dist = dist;
      bi = i;
    }
  }

  p_d[0] = bi;
  return max_dist > TOLERANCE;
}

function initial_simplex(num_points: number, points: number[][], initial_vertices: number[]): number {
  const min_index = [0, 0, 0];
  const max_index = [0, 0, 0];
  if (!calc_max_extent(num_points, points, min_index, max_index)) return -1;

  let bi = -1;
  let max_dist = 0.0;
  for (let i = 0; i < 3; i++) {
    const a = min_index[i];
    const b = max_index[i];
    const delta = [points[a][0] - points[b][0], points[a][1] - points[b][1], points[a][2] - points[b][2]];
    const dist = norm_squared(delta);
    if (dist > max_dist) {
      bi = i;
      max_dist = dist;
    }
  }

  //first two points are (a, b)
  const a = min_index[bi];
  const b = max_index[bi];
  const c = [-1];
  const d = [-1];

  if (!find_third_point(num_points, points, a, b, c)) return -2;

  if (!find_fourth_point(num_points, points, a, b, c[0], d)) return -3;

  initial_vertices[0] = a;
  initial_vertices[1] = b;
  initial_vertices[2] = c[0];
  initial_vertices[3] = d[0];
  return 0;
}

function visible(w: number[], plane_point: number[], plane_normal: number[]): boolean {
  return point_plane_distance(w, plane_point, plane_normal) > 0;
}

export function add_facet(points: number[][], a: number, b: number, c: number, facet: number[], plane_normal: number[],
  barycentre: number[]): void {
  calculate_plane_normal(points, a, b, c, plane_normal);
  if (visible(barycentre, points[a], plane_normal)) {
    plane_normal[0] = -plane_normal[0];
    plane_normal[1] = -plane_normal[1];
    plane_normal[2] = -plane_normal[2];

    facet[0] = b;
    facet[1] = a;
    facet[2] = c;
  } else {
    facet[0] = a;
    facet[1] = b;
    facet[2] = c;
  }
}

function initialize_convex_hull(num_points: number, points: number[][], facets: number[][], plane_normal: number[][],
  processed: boolean[], initial_vertices: number[], barycentre: number[]): number {
  // memset(processed, 0, PTM_MAX_POINTS * sizeof(bool))
  for (let i = 0; i < PTM_MAX_POINTS; i++) processed[i] = false;
  // memset(barycentre, 0, 3 * sizeof(double))
  barycentre[0] = 0;
  barycentre[1] = 0;
  barycentre[2] = 0;
  const ret = initial_simplex(num_points, points, initial_vertices);
  if (ret !== 0) return ret;

  for (let i = 0; i < 4; i++) {
    const a = initial_vertices[i];
    processed[a] = true;

    barycentre[0] += points[a][0];
    barycentre[1] += points[a][1];
    barycentre[2] += points[a][2];
  }
  barycentre[0] /= 4;
  barycentre[1] /= 4;
  barycentre[2] /= 4;

  add_facet(points, initial_vertices[0], initial_vertices[1], initial_vertices[2], facets[0], plane_normal[0], barycentre);
  add_facet(points, initial_vertices[0], initial_vertices[1], initial_vertices[3], facets[1], plane_normal[1], barycentre);
  add_facet(points, initial_vertices[0], initial_vertices[2], initial_vertices[3], facets[2], plane_normal[2], barycentre);
  add_facet(points, initial_vertices[1], initial_vertices[2], initial_vertices[3], facets[3], plane_normal[3], barycentre);
  return 0;
}

// The C assert(num_points == PTM_NUM_POINTS_*) is compiled out in optimised (NDEBUG) builds and is not reproduced.
export function get_convex_hull(num_points: number, points: number[][], ch: ConvexHull, simplex: number[][]): number {
  let ret = 0;
  let num_prev = ch.num_prev;
  ch.num_prev = num_points;
  if (!ch.ok) {
    ret = initialize_convex_hull(num_points, points, ch.facets, ch.plane_normal, ch.processed, ch.initial_vertices,
      ch.barycentre);
    if (ret !== 0) return ret;

    ch.num_facets = 4;
    num_prev = 0;
  }

  for (let i = num_prev; i < num_points; i++) {
    if (ch.processed[i]) continue;
    ch.processed[i] = true;

    const to_add: number[][] = [];
    // int8_t edge_visible[PTM_MAX_POINTS][PTM_MAX_POINTS], zeroed by memset for every point
    const edge_visible: number[][] = [];
    for (let r = 0; r < PTM_MAX_POINTS; r++) edge_visible.push(new Array<number>(PTM_MAX_POINTS).fill(0));

    for (let j = 0; j < ch.num_facets; j++) {
      const a = ch.facets[j][0];
      const b = ch.facets[j][1];
      const c = ch.facets[j][2];

      let u: number;
      let v: number;
      let w: number;

      const distance = point_plane_distance(points[i], points[a], ch.plane_normal[j]);
      const vis = distance > TOLERANCE;
      if (vis) {
        u = edge_visible[a][b] |= VISIBLE;
        edge_visible[b][a] |= VISIBLE;

        v = edge_visible[b][c] |= VISIBLE;
        edge_visible[c][b] |= VISIBLE;

        w = edge_visible[c][a] |= VISIBLE;
        edge_visible[a][c] |= VISIBLE;

        // memcpy(ch->facets[j], ch->facets[ch->num_facets - 1], 3 * sizeof(int8_t)) and the same for plane_normal
        const last = ch.num_facets - 1;
        for (let k = 0; k < 3; k++) ch.facets[j][k] = ch.facets[last][k];
        for (let k = 0; k < 3; k++) ch.plane_normal[j][k] = ch.plane_normal[last][k];
        ch.num_facets--;
        j--;
      } else {
        u = edge_visible[a][b] |= INVISIBLE;
        edge_visible[b][a] |= INVISIBLE;

        v = edge_visible[b][c] |= INVISIBLE;
        edge_visible[c][b] |= INVISIBLE;

        w = edge_visible[c][a] |= INVISIBLE;
        edge_visible[a][c] |= INVISIBLE;
      }

      if (u === BOTH) to_add.push([i, a, b]);

      if (v === BOTH) to_add.push([i, b, c]);

      if (w === BOTH) to_add.push([i, c, a]);
    }

    for (let j = 0; j < to_add.length; j++) {
      if (ch.num_facets >= PTM_MAX_FACETS) return -4;

      add_facet(points, to_add[j][0], to_add[j][1], to_add[j][2], ch.facets[ch.num_facets],
        ch.plane_normal[ch.num_facets], ch.barycentre);
      ch.num_facets++;
    }
  }

  for (let i = 0; i < ch.num_facets; i++) {
    const a = ch.facets[i][0];
    const b = ch.facets[i][1];
    const c = ch.facets[i][2];
    if (a === 0 || b === 0 || c === 0) return 1;    //central atom contained in convex hull

    simplex[i][0] = a - 1;
    simplex[i][1] = b - 1;
    simplex[i][2] = c - 1;
  }

  return ret;
}
