/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the PTM library (Polyhedral Template Matching, PM Larsen, MIT licence), src/PTM/ptm_neighbour_ordering.cpp at LAMMPS commit b332f28 (https://github.com/lammps/lammps). Copyright (c) 2016 PM Larsen (MIT, see LICENSE.ptm). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

import { PTM_MAX_INPUT_POINTS } from './ptm_constants';
import type { GetNeighbours } from './ptm_types';
import { std_sort } from './libstdcxx_sort';
import { voronoicell_neighbor } from './ptm_voronoi_cell';

const MAX_INNER = 4;
const MAX_SNBRS = 12;

interface sorthelper_t {
  area: number;
  dist: number;
  index: number;
  inner: number;
  number: number;
  offset: number[];
}

interface solidnbr_t {
  index: number;
  number: number;
  area: number;
  offset: number[];
}

/* C assert(): kept as a thrown error. */
function ptm_assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ptm_neighbour_ordering: assertion failed: ${msg}`);
}

function zeros(n: number): number[] {
  const a: number[] = [];
  for (let i = 0; i < n; i++) a.push(0);
  return a;
}

function zeros_vec3(n: number): number[][] {
  const a: number[][] = [];
  for (let i = 0; i < n; i++) a.push([0, 0, 0]);
  return a;
}

function new_sorthelper(): sorthelper_t {
  return { area: 0, dist: 0, index: 0, inner: 0, number: 0, offset: [0, 0, 0] };
}

function new_sorthelper_array(n: number): sorthelper_t[] {
  const a: sorthelper_t[] = [];
  for (let i = 0; i < n; i++) a.push(new_sorthelper());
  return a;
}

function new_solidnbr(): solidnbr_t {
  return { index: 0, number: 0, area: 0, offset: [0, 0, 0] };
}

function new_solidnbr_array(n: number): solidnbr_t[] {
  const a: solidnbr_t[] = [];
  for (let i = 0; i < n; i++) a.push(new_solidnbr());
  return a;
}

/* memcpy(dst, src, 3 * sizeof(double)) */
function copy3(dst: number[], src: number[]): void {
  for (let k = 0; k < 3; k++) dst[k] = src[k];
}

function sorthelper_compare(a: sorthelper_t, b: sorthelper_t): boolean {
  if (a.area > b.area)
    return true;

  if (a.area < b.area)
    return false;

  if (a.dist < b.dist)
    return true;

  return false;
}

/**
 * Sorts data[0..n) in place, as std::sort(data, data + n, &sorthelper_compare) does: the libstdc++ introsort of
 * libstdcxx_sort.ts, so equal (area, dist) entries end in the same order as native.
 */
function sort_sorthelpers(data: sorthelper_t[], n: number): void {
  std_sort(data, 0, n, sorthelper_compare);
}

function dot_product(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross_product(a: number[], b: number[], c: number[]): void {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

function calculate_solid_angle(R1: number[], R2: number[], R3: number[]): number {  // norms of R1-R3 must be 1
  const R2R3: number[] = [0, 0, 0];
  cross_product(R2, R3, R2R3);
  const numerator = dot_product(R1, R2R3);

  const r1r2 = dot_product(R1, R2);
  const r2r3 = dot_product(R2, R3);
  const r3r1 = dot_product(R3, R1);

  const denominator = 1 + r1r2 + r3r1 + r2r3;
  return Math.abs(2 * Math.atan2(numerator, denominator));
}

// todo: change voronoi code to return errors rather than exiting
function calculate_voronoi_face_areas(
  num_points: number,
  _points: number[][],
  normsq: number[],
  max_norm: number,
  v: voronoicell_neighbor,
  calc_solid_angles: boolean,
  nbr_indices: number[],
  face_areas: number[],
): number {
  const k = 10 * max_norm;
  v.init(-k, k, -k, k, -k, k);

  for (let i = 1; i < num_points; i++) {
    const x = _points[i][0] - _points[0][0];
    const y = _points[i][1] - _points[0][1];
    const z = _points[i][2] - _points[0][2];
    v.nplane(x, y, z, normsq[i], i);
  }

  v.neighbors(nbr_indices);

  // v->face_areas(face_areas);
  if (!calc_solid_angles) {
    v.face_areas(face_areas);
    return 0;
  } else {
    const face_vertices: number[] = [];
    const vertices: number[] = [];

    v.face_vertices(face_vertices);
    v.vertices(0, 0, 0, vertices);

    const num_vertices = Math.floor(vertices.length / 3);
    for (let i = 0; i < num_vertices; i++) {
      const x = vertices[i * 3 + 0];
      const y = vertices[i * 3 + 1];
      const z = vertices[i * 3 + 2];

      const s = Math.sqrt(x * x + y * y + z * z);
      vertices[i * 3 + 0] /= s;
      vertices[i * 3 + 1] /= s;
      vertices[i * 3 + 2] /= s;
    }

    const num_faces = v.number_of_faces();

    // #ifdef DEBUG printf blocks are omitted (DEBUG is not defined in the native build).

    let c = 0;
    for (let current_face = 0; current_face < num_faces; current_face++) {
      const num = face_vertices[c++];

      const point_index = nbr_indices[current_face];
      if (point_index > 0) {
        let solid_angle = 0;
        let u = face_vertices[c];
        let v = face_vertices[c + 1];
        for (let i = 2; i < num; i++) {
          const w = face_vertices[c + i];
          const omega = calculate_solid_angle(
            vertices.slice(u * 3, u * 3 + 3),
            vertices.slice(v * 3, v * 3 + 3),
            vertices.slice(w * 3, w * 3 + 3),
          );
          solid_angle += omega;

          v = w;
        }

        face_areas[current_face] = solid_angle;
        // solids[current_face] = solid_angle;
        // face_areas[point_index] = solid_angle;
      }

      c += num;
    }

    ptm_assert(c === face_vertices.length, 'c == face_vertices.size()');
    return 0;
  }
}

function _calculate_neighbour_ordering(
  voronoi_handle: voronoicell_neighbor,
  num_points: number,
  _points: number[][],
  calc_solid_angles: boolean,
  data: sorthelper_t[],
): number {
  ptm_assert(num_points <= PTM_MAX_INPUT_POINTS, 'num_points <= PTM_MAX_INPUT_POINTS');

  let max_norm = 0;
  const points = zeros_vec3(PTM_MAX_INPUT_POINTS);
  const normsq = zeros(PTM_MAX_INPUT_POINTS);

  for (let i = 0; i < num_points; i++) {
    const x = _points[i][0] - _points[0][0];
    const y = _points[i][1] - _points[0][1];
    const z = _points[i][2] - _points[0][2];
    points[i][0] = x;
    points[i][1] = y;
    points[i][2] = z;

    normsq[i] = x * x + y * y + z * z;
    max_norm = Math.max(max_norm, normsq[i]);
  }

  max_norm = Math.sqrt(max_norm);

  const nbr_indices = zeros(num_points + 6);
  const face_areas = zeros(num_points + 6);
  const ret = calculate_voronoi_face_areas(
    num_points,
    points,
    normsq,
    max_norm,
    voronoi_handle,
    calc_solid_angles,
    nbr_indices,
    face_areas,
  );
  if (ret !== 0)
    return ret;

  const areas = zeros(PTM_MAX_INPUT_POINTS);
  areas[0] = Infinity;
  for (let i = 0; i < nbr_indices.length; i++) {
    const index = nbr_indices[i];
    if (index > 0)
      areas[index] = face_areas[i];
  }

  for (let i = 0; i < num_points; i++) {
    ptm_assert(areas[i] === areas[i], 'areas[i] == areas[i]');
    data[i].area = areas[i];
    data[i].dist = normsq[i];
    data[i].index = i;
    copy3(data[i].offset, points[i]);
  }

  sort_sorthelpers(data, num_points);
  return ret;
}

export function calculate_neighbour_ordering(
  voronoi_handle: voronoicell_neighbor,
  atom_index: number,
  min_points: number,
  get_neighbours: GetNeighbours,
  nbrlist: unknown,
  calc_solid_angles: boolean,
  indices: number[],
  points: number[][],
  numbers: number[],
): number {
  const nbr_indices = zeros(PTM_MAX_INPUT_POINTS);
  const nbr_numbers = zeros(PTM_MAX_INPUT_POINTS);
  const nbr_pos = zeros_vec3(PTM_MAX_INPUT_POINTS);
  const num_points = get_neighbours(nbrlist, atom_index, atom_index, PTM_MAX_INPUT_POINTS, nbr_indices, nbr_numbers, nbr_pos);
  if (num_points < min_points)
    return -1;

  const data = new_sorthelper_array(PTM_MAX_INPUT_POINTS);
  const ret = _calculate_neighbour_ordering(voronoi_handle, num_points, nbr_pos, calc_solid_angles, data);
  if (ret !== 0)
    return ret;

  for (let i = 0; i < num_points; i++) {
    const index = data[i].index;
    indices[i] = nbr_indices[index];
    numbers[i] = nbr_numbers[index];
    copy3(points[i], nbr_pos[index]);
  }

  return num_points;
}

function find_diamond_neighbours(
  voronoi_handle: voronoicell_neighbor,
  num_points: number,
  _points: number[][],
  nbr_indices: number[],
  nbr_numbers: number[],
  num_solid_nbrs: number,
  calc_solid_angles: boolean,
  nbrlist: solidnbr_t[],
): number {
  const data = new_sorthelper_array(PTM_MAX_INPUT_POINTS);
  const ret = _calculate_neighbour_ordering(voronoi_handle, num_points, _points, calc_solid_angles, data);
  if (ret !== 0)
    return ret;

  const n = Math.min(num_solid_nbrs, num_points - 1);
  for (let i = 0; i < n; i++) {
    nbrlist[i].index = nbr_indices[data[i + 1].index];
    nbrlist[i].area = data[i + 1].area;
    nbrlist[i].number = nbr_numbers[data[i + 1].index];
    copy3(nbrlist[i].offset, data[i + 1].offset);
  }

  return ret;
}

export function calculate_two_shell_neighbour_ordering(
  voronoi_handle: voronoicell_neighbor,
  atom_index: number,
  get_neighbours: GetNeighbours,
  nbrlist: unknown,
  num_inner: number,
  num_outer: number,
  max_snbrs: number,
  calc_solid_angles: boolean,
  nbr_indices: number[],
  points: number[][],
  numbers: number[],
): number {
  ptm_assert(num_inner <= MAX_INNER, 'num_inner <= MAX_INNER');

  const central_nbr_indices = zeros(MAX_SNBRS + 1);
  const central_nbr_numbers = zeros(MAX_SNBRS + 1);
  const central_nbr_pos = zeros_vec3(MAX_SNBRS + 1);
  const num_points = get_neighbours(nbrlist, atom_index, atom_index, max_snbrs + 1, central_nbr_indices, central_nbr_numbers, central_nbr_pos);
  if (num_points < num_inner + 1)
    return -1;

  const central_solid = new_solidnbr_array(MAX_SNBRS);
  let ret = find_diamond_neighbours(voronoi_handle, num_points, central_nbr_pos, central_nbr_indices, central_nbr_numbers, max_snbrs, calc_solid_angles, central_solid);
  if (ret !== 0)
    return ret;

  const data = new_sorthelper_array(MAX_INNER * 6);
  nbr_indices[0] = atom_index;
  numbers[0] = central_nbr_numbers[0];
  for (let k = 0; k < 3; k++) points[0][k] = 0;

  const claimed = new Set<number>();
  claimed.add(atom_index);
  for (let i = 0; i < num_inner; i++) {
    nbr_indices[i + 1] = central_solid[i].index;
    numbers[i + 1] = central_solid[i].number;
    copy3(points[i + 1], central_solid[i].offset);

    claimed.add(central_solid[i].index);
  }

  let index = 0;
  for (let i = 0; i < num_inner; i++) {
    const inner_index = central_solid[i].index;

    //----------------------------------------------
    const inner_nbr_indices = zeros(MAX_SNBRS + 1);
    const inner_nbr_numbers = zeros(MAX_SNBRS + 1);
    const inner_nbr_pos = zeros_vec3(MAX_SNBRS + 1);
    const num_points = get_neighbours(nbrlist, atom_index, inner_index, max_snbrs + 1, inner_nbr_indices, inner_nbr_numbers, inner_nbr_pos);
    if (num_points < num_inner + 1)
      return -1;

    const inner_solid = new_solidnbr_array(MAX_SNBRS);
    ret = find_diamond_neighbours(voronoi_handle, num_points, inner_nbr_pos, inner_nbr_indices, inner_nbr_numbers, max_snbrs, calc_solid_angles, inner_solid);
    if (ret !== 0)
      return ret;
    //----------------------------------------------

    const n = Math.min(6, num_points);
    for (let j = 0; j < n; j++) {
      const already_claimed = claimed.has(inner_solid[j].index);
      if (already_claimed)
        continue;

      data[index].inner = i;
      data[index].index = inner_solid[j].index;
      data[index].area = inner_solid[j].area;
      data[index].dist = 0;
      data[index].number = inner_solid[j].number;

      copy3(data[index].offset, inner_solid[j].offset);
      for (let k = 0; k < 3; k++)
        data[index].offset[k] += central_solid[i].offset[k];

      index++;
    }
  }

  const n = index;
  sort_sorthelpers(data, n);

  let num_found = 0;
  const counts = zeros(MAX_INNER);
  for (let i = 0; i < n; i++) {
    const inner = data[i].inner;
    const nbr_atom_index = data[i].index;

    const already_claimed = claimed.has(nbr_atom_index);
    if (counts[inner] >= num_outer || already_claimed)
      continue;

    nbr_indices[1 + num_inner + num_outer * inner + counts[inner]] = nbr_atom_index;
    numbers[1 + num_inner + num_outer * inner + counts[inner]] = data[i].number;
    copy3(points[1 + num_inner + num_outer * inner + counts[inner]], data[i].offset);
    claimed.add(nbr_atom_index);

    counts[inner]++;
    num_found++;
    if (num_found >= num_inner * num_outer)
      break;
  }

  if (num_found !== num_inner * num_outer)
    return -1;

  return 0;
}

/** voronoi_initialize_local() (ptm_neighbour_ordering.cpp at b332f28): a new Voronoi cell object for one evaluation. */
export function voronoi_initialize_local(): voronoicell_neighbor {
  return new voronoicell_neighbor();
}

/** voronoi_uninitialize_local() (ptm_neighbour_ordering.cpp): the C deletes the cell; the GC does the same here. */
export function voronoi_uninitialize_local(_ptr: voronoicell_neighbor): void {}
