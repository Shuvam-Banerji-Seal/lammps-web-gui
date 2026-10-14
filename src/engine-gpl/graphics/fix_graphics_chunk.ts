/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_chunk.cpp
 * (FixGraphicsChunk: keywords radius, alpha, region, clip, maxreplace, mindist, shading; end_of_step() assigns the
 * chunks, splits them into clusters, replaces the atoms of small clusters by icosahedra, builds the convex hull of
 * each cluster and emits the triangles that lie inside the box). The clustering (splitByDistance) and the hull
 * (src/GRAPHICS/image_objects.cpp, see convex_hull.ts) are the same algorithms. LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The chunk assignment is an input (the engine's compute chunk/atom result, 1-based, 0 for none) and the region is
 * a predicate, so this module does not depend on the compute or region classes. One process is assumed (no ghost
 * atoms), so the atoms passed in are the atoms of the native nall loop.
 *
 * Quirks kept from the C++ on purpose: mindist is stored once (the first snapshot's 4 * maxradius when it was not
 * given), and the colour index of a vertex is its point index divided by 12 even when the cluster was not replaced
 * by icosahedra (so the atom type of such vertices is the type of atom index/12, as in the C++).
 */
import { buildConvexHull } from './convex_hull';
import type { GraphicsObject, TriNormObject, Vec3 } from './objects';

/** Options of fix graphics/chunk after parsing. */
export interface ChunkOptions {
  /** Global radius, or null when radius was not given (per-atom or per-type radius then, else the default). */
  radius: number | null;
  /** Default radius 0.5 * xlattice (xlattice is 1.0 without a lattice command). */
  defaultRadius: number;
  alpha: number;
  maxreplace: number;
  /** Integer mindist as given (must be > 0), or null for the automatic 4 * maxradius. */
  mindist: number | null;
  smooth: boolean;
  clip: boolean;
  /** The region test, or null when no region keyword was given. */
  region: ((x: Vec3) => boolean) | null;
  /** The pair style sigma of a type (sigma[type][type]) when the pair style has one, else null. */
  sigmaOfType: ((type: number) => number) | null;
}

export interface ChunkAtom {
  type: number;
  x: Vec3;
  inGroup: boolean;
  /** 1-based chunk ID of the atom, 0 when it has none (the compute chunk/atom ichunk). */
  chunk: number;
  /** Per-atom radius (the atom style sphere radius), when the atom style has one. */
  radius?: number;
}

export interface ChunkBox {
  lo: Vec3;
  hi: Vec3;
  dimension: 2 | 3;
}

const A = 0.5257311121191336;
const B = 0.8506508083520399;
const ICOSAHEDRON: Vec3[] = [
  [-A, B, 0.0], [A, B, 0.0], [-A, -B, 0.0], [A, -B, 0.0], [0.0, -A, B], [0.0, A, B],
  [0.0, -A, -B], [0.0, A, -B], [B, 0.0, -A], [B, 0.0, A], [-B, 0.0, -A], [-B, 0.0, A],
];
const NUM_POINTS = 12;

/** Parse the keywords after "fix graphics/chunk nevery chunkID" (the constructor checks). */
export function parseChunkArgs(args: string[], regionOf: (id: string) => ((x: Vec3) => boolean) | null): ChunkOptions {
  const opt: ChunkOptions = {
    radius: null,
    defaultRadius: 0.5,
    alpha: 0.0,
    maxreplace: 100,
    mindist: null,
    smooth: true,
    clip: true,
    region: null,
    sigmaOfType: null,
  };
  const num = (v: string, what: string): number => {
    const n = Number(v);
    if (!Number.isFinite(n)) throw new Error(`Expected floating point parameter instead of '${v}' for ${what}`);
    return n;
  };
  const int = (v: string, what: string): number => {
    if (!/^[+-]?\d+$/.test(v)) throw new Error(`Expected integer parameter instead of '${v}' for ${what}`);
    return Number(v);
  };
  let i = 0;
  while (i < args.length) {
    const key = args[i];
    if (i + 2 > args.length) throw new Error(`Illegal fix graphics/chunk ${key}: missing argument`);
    const val = args[i + 1];
    if (key === 'radius') {
      const r = num(val, 'radius');
      if (r <= 0.0) throw new Error('Fix graphics/chunk radius value must be > 0');
      opt.radius = r;
    } else if (key === 'alpha') {
      const a = num(val, 'alpha');
      if (a < 0.0) throw new Error('Fix graphics/chunk alpha value must be >= 0');
      opt.alpha = a;
    } else if (key === 'region') {
      const reg = regionOf(val);
      if (!reg) throw new Error(`Region ${val} for fix graphics/chunk does not exist`);
      opt.region = reg;
    } else if (key === 'clip') {
      if (/^(on|yes|true|1)$/i.test(val)) opt.clip = true;
      else if (/^(off|no|false|0)$/i.test(val)) opt.clip = false;
      else throw new Error(`Expected boolean parameter instead of '${val}' in fix graphics/chunk`);
    } else if (key === 'maxreplace') {
      opt.maxreplace = int(val, 'maxreplace');
      if (opt.maxreplace < 0) throw new Error('Fix graphics/chunk maxreplace value must be >= 0');
    } else if (key === 'mindist') {
      opt.mindist = int(val, 'mindist');
      if (opt.mindist <= 0) throw new Error('Fix graphics/chunk mindist value must be > 0');
    } else if (key === 'shading') {
      if (val === 'smooth') opt.smooth = true;
      else if (val === 'flat') opt.smooth = false;
      else throw new Error(`Unknown fix graphics/chunk shading setting ${val}`);
    } else {
      throw new Error(`Unknown fix graphics/chunk keyword ${key}`);
    }
    i += 2;
  }
  return opt;
}

/** Cluster the points: connected components of pairs closer than dist (splitByDistance). */
function clusterByDistance(indices: number[], pos: (i: number) => Vec3, dist: number): number[][] {
  const n = indices.length;
  const parent = Array.from({ length: n }, (_, k) => k);
  const find = (k: number): number => {
    while (parent[k] !== k) {
      parent[k] = parent[parent[k]];
      k = parent[k];
    }
    return k;
  };
  const d2 = dist * dist;
  const grid = new Map<string, number[]>();
  for (let k = 0; k < n; k++) {
    const p = pos(indices[k]);
    const c = [Math.floor(p[0] / dist), Math.floor(p[1] / dist), Math.floor(p[2] / dist)];
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const list = grid.get(`${c[0] + dx},${c[1] + dy},${c[2] + dz}`);
          if (!list) continue;
          for (const j of list) {
            const q = pos(indices[j]);
            const ddx = p[0] - q[0];
            const ddy = p[1] - q[1];
            const ddz = p[2] - q[2];
            if (ddx * ddx + ddy * ddy + ddz * ddz < d2) {
              const ra = find(k);
              const rb = find(j);
              if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb);
            }
          }
        }
      }
    }
    const key = `${c[0]},${c[1]},${c[2]}`;
    const list = grid.get(key);
    if (list) list.push(k);
    else grid.set(key, [k]);
  }
  const groups = new Map<number, number[]>();
  for (let k = 0; k < n; k++) {
    const r = find(k);
    const list = groups.get(r);
    if (list) list.push(indices[k]);
    else groups.set(r, [indices[k]]);
  }
  return [...groups.values()].sort((a, b) => a[0] - b[0]);
}

export class FixGraphicsChunk {
  readonly options: ChunkOptions;
  private mindistStored: number | null;

  constructor(options: ChunkOptions) {
    this.options = options;
    this.mindistStored = options.mindist;
  }

  /** The triangle objects of one snapshot (end_of_step). */
  objects(atoms: ChunkAtom[], box: ChunkBox, nchunk: number): GraphicsObject[] {
    if (box.dimension === 2) {
      throw new Error('Fix graphics/chunk is currently not compatible with 2d systems');
    }
    const opt = this.options;
    // per-atom radius: the global value, else the atom radius, else the default; <= 0 falls back to the default
    let maxradius = 0.0;
    const rad: number[] = atoms.map((a) => {
      let r = opt.radius ?? opt.defaultRadius;
      if (opt.radius === null) {
        // per-atom radius first, then the per-type sigma of the pair style (0.5 * sigma[type][type])
        if (a.radius !== undefined) r = a.radius;
        else if (opt.sigmaOfType) r = 0.5 * opt.sigmaOfType(a.type);
      }
      if (r <= 0.0) r = opt.radius ?? opt.defaultRadius;
      maxradius = Math.max(maxradius, r);
      return r;
    });

    // atoms of the group, in a chunk, inside the region
    const chunkAtoms: number[][] = Array.from({ length: nchunk }, () => []);
    atoms.forEach((a, i) => {
      if (!a.inGroup) return;
      if (opt.region && !opt.region(a.x)) return;
      if (a.chunk < 1 || a.chunk > nchunk) return;
      chunkAtoms[a.chunk - 1].push(i);
    });

    if (this.mindistStored === null) this.mindistStored = 4.0 * maxradius;
    const mindist = this.mindistStored;

    const newChunks: number[][] = [];
    for (const members of chunkAtoms) {
      if (members.length === 0) continue;
      for (const c of clusterByDistance(members, (i) => atoms[i].x, mindist)) newChunks.push(c);
    }

    const out: TriNormObject[] = [];
    for (const iatoms of newChunks) {
      const natoms = iatoms.length;
      let pts: Vec3[] = [];
      if (natoms <= opt.maxreplace) {
        for (const i of iatoms) {
          for (const ico of ICOSAHEDRON) {
            const p: Vec3 = [
              rad[i] * ico[0] + atoms[i].x[0],
              rad[i] * ico[1] + atoms[i].x[1],
              rad[i] * ico[2] + atoms[i].x[2],
            ];
            if (!opt.clip || inside(p, box)) pts.push(p);
          }
        }
      } else {
        const center: Vec3 = [0, 0, 0];
        for (const i of iatoms) {
          center[0] += atoms[i].x[0];
          center[1] += atoms[i].x[1];
          center[2] += atoms[i].x[2];
        }
        center[0] /= natoms;
        center[1] /= natoms;
        center[2] /= natoms;
        pts = [];
        for (const i of iatoms) {
          const e: Vec3 = [atoms[i].x[0] - center[0], atoms[i].x[1] - center[1], atoms[i].x[2] - center[2]];
          const val = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
          if (val > 0.0) {
            const s = 1.0 / Math.sqrt(val);
            e[0] *= s;
            e[1] *= s;
            e[2] *= s;
          }
          const p: Vec3 = [rad[i] * e[0] + atoms[i].x[0], rad[i] * e[1] + atoms[i].x[1], rad[i] * e[2] + atoms[i].x[2]];
          if (!opt.clip || inside(p, box)) pts.push(p);
        }
      }

      const hull = buildConvexHull(pts, opt.smooth, opt.alpha);
      for (let t = 0; t < hull.triangles.length; t++) {
        const idx = hull.colorIdx[t].map((k) => {
          const c = Math.floor(k / NUM_POINTS);
          return c < 0 || c >= natoms ? 0 : c;
        });
        const tri = hull.triangles[t];
        let addme = true;
        for (let v = 0; v < 3; v++) {
          for (let j = 0; j < 3; j++) {
            if (tri[v][j] < box.lo[j] - mindist || tri[v][j] > box.hi[j] + mindist) addme = false;
          }
        }
        if (!addme) continue;
        const nrm = hull.normals[t];
        out.push({
          kind: 'trinorm',
          types: [atoms[iatoms[idx[0]]].type, atoms[iatoms[idx[1]]].type, atoms[iatoms[idx[2]]].type],
          vertices: [tri[0], tri[1], tri[2]],
          normals: [nrm[0], nrm[1], nrm[2]],
        });
      }
    }
    return out;
  }
}

/** Domain::inside for an orthogonal box: lo <= x < hi in every dimension. */
function inside(p: Vec3, box: ChunkBox): boolean {
  for (let d = 0; d < 3; d++) {
    if (p[d] < box.lo[d] || p[d] >= box.hi[d]) return false;
  }
  return true;
}
