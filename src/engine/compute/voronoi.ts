import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNum, parseInt_ } from '../force/util';

/*
 * compute ID group-ID voronoi/atom keyword arg ... — docs.lammps.org/compute_voronoi_atom.html
 *
 * Quoted from the page:
 *   "Define a computation that calculates the Voronoi tessellation of the"
 *   "atom's Voronoi cell is closer to that atom than any other."
 *   "the volume of the Voronoi cell around each atom."
 *   "is the number of faces of the Voronoi cell."
 *   "This is equal to the"
 *   "number of nearest neighbors of the central atom, plus any exterior"
 *   "The tessellation is calculated using all"
 *   "atoms in the simulation, but non-zero values are only stored for atoms"
 *   "in the group."
 *   "If the *only_group* keyword is specified the tessellation is performed"
 *   "only with respect to the atoms contained in the compute group."
 *   "If the *radius* keyword is specified with an atom style variable as"
 *   "the argument, a poly-disperse Voronoi tessellation is"
 *   "The *edge_histo* keyword activates the compilation of a histogram of"
 *   "The *edge_threshold* and *face_threshold* keywords allow the"
 *   "The Voronoi cell volume will be in distance :doc:`units <units>` cubed."
 *   "The Voronoi face area will be in distance :doc:`units <units>` squared."
 *   "The default for the neighbors keyword is no."
 *   "The *peratom* keyword was removed as it is no longer required."
 *
 * Scope of this implementation: 3d, periodic orthogonal boxes, the default
 * per-atom output (volume, number of faces), surface (third column), only_group,
 * radius (radical tessellation with an atom-style variable), edge_histo (global
 * vector), edge_threshold and face_threshold. Not implemented, each an error:
 * occupation, neighbors yes, non-periodic boundaries, triclinic boxes, 2d.
 *
 * Algorithm (textbook): the cell of atom i is the intersection of half-spaces
 * bounded by the bisecting planes with each other atom j (its periodic images
 * included). For radical (radius) tessellations the plane is the radical plane
 * |x-x_i|^2 - r_i^2 = |x-x_j|^2 - r_j^2. The cell is built by clipping a cube
 * of half-side D (centred on atom i) with those planes, nearest first. A plane
 * at signed distance h from atom i cannot cut a cell that lies within radius
 * R of atom i when h > R; with rmax the largest radius in the pool, every plane
 * from an atom farther than D has h >= f(D) = (D^2 + r_i^2 - rmax^2)/(2D), which
 * increases with D. So when the cell's farthest vertex R satisfies R < f(D) the
 * result is exact (the cube faces are then inactive too, as f(D) < D). Otherwise
 * D grows by 1.5 and the cell is rebuilt.
 *
 * Volume: fan triangulation of the faces about the atom centre (det/6). Face
 * area: Newell's formula. A face counts when its area exceeds face_threshold
 * (default 0); its edges count when their length exceeds edge_threshold.
 */

interface Face { idx: number[]; plane: number }
interface Poly { v: number[]; f: Face[] }

/** Cube of half-side h centred on the origin (the atom), faces oriented outward. */
const cubePoly = (h: number): Poly => {
  const v = [-h, -h, -h, h, -h, -h, h, h, -h, -h, h, -h, -h, -h, h, h, -h, h, h, h, h, -h, h, h];
  const quads: [number[], number[]][] = [
    [[1, 2, 6, 5], [1, 0, 0]], [[0, 3, 7, 4], [-1, 0, 0]],
    [[3, 2, 6, 7], [0, 1, 0]], [[0, 4, 5, 1], [0, -1, 0]],
    [[4, 5, 6, 7], [0, 0, 1]], [[0, 1, 2, 3], [0, 0, -1]],
  ];
  const f: Face[] = quads.map(([idx, nrm]) => ({ idx: orientOutward(v, idx, nrm), plane: -1 }));
  return { v, f };
};

/** Reverses a polygon's vertex order when its Newell normal points against the wanted outward normal. */
const orientOutward = (v: number[], idx: number[], nrm: number[]): number[] => {
  const nv = newell(v, idx);
  const dot = nv[0] * nrm[0] + nv[1] * nrm[1] + nv[2] * nrm[2];
  return dot < 0 ? idx.slice().reverse() : idx;
};

/** Newell vector of a polygon (twice the area times the unit normal). */
const newell = (v: number[], idx: number[]): [number, number, number] => {
  let nx = 0, ny = 0, nz = 0;
  const m = idx.length;
  for (let k = 0; k < m; k++) {
    const a = 3 * idx[k], b = 3 * idx[(k + 1) % m];
    nx += v[a + 1] * v[b + 2] - v[a + 2] * v[b + 1];
    ny += v[a + 2] * v[b] - v[a] * v[b + 2];
    nz += v[a] * v[b + 1] - v[a + 1] * v[b];
  }
  return [nx, ny, nz];
};

/** Removes consecutive repeated vertices (cyclically) from a polygon loop. */
const dedupeLoop = (idx: number[]): number[] => {
  const out: number[] = [];
  for (const k of idx) if (out.length === 0 || out[out.length - 1] !== k) out.push(k);
  while (out.length > 1 && out[out.length - 1] === out[0]) out.pop();
  return out;
};

/**
 * Keeps the part of a convex polyhedron with n.x <= c (eps: tolerance on the
 * signed value). The new cap face takes the clipping plane's id. Vertices are
 * compacted so that only referenced ones remain.
 */
const clipPoly = (p: Poly, nx: number, ny: number, nz: number, c: number, plane: number, eps: number): Poly => {
  const v = p.v.slice();
  const nv0 = p.v.length / 3;
  const s = new Float64Array(nv0);
  let anyOut = false, anyIn = false;
  for (let k = 0; k < nv0; k++) {
    const sv = nx * p.v[3 * k] + ny * p.v[3 * k + 1] + nz * p.v[3 * k + 2] - c;
    s[k] = sv;
    if (sv > eps) anyOut = true; else anyIn = true;
  }
  if (!anyOut) return p;
  if (!anyIn) return { v: [], f: [] };
  const edgeNew = new Map<number, number>();
  const capNext = new Map<number, number>();
  const faces: Face[] = [];
  for (const face of p.f) {
    const idx = face.idx;
    const m = idx.length;
    const out: number[] = [];
    const cross: { x: number; exit: boolean }[] = [];
    for (let k = 0; k < m; k++) {
      const a = idx[k], b = idx[(k + 1) % m];
      const inA = s[a] <= eps, inB = s[b] <= eps;
      if (inA) out.push(a);
      if (inA !== inB) {
        // the kept end of the edge: when it lies on the plane it is itself the crossing point
        const p = inA ? a : b;
        let X: number;
        if (Math.abs(s[p]) <= eps) {
          X = p; // already in out (pushed as an inside vertex)
        } else {
          const key = Math.min(a, b) * nv0 + Math.max(a, b);
          const known = edgeNew.get(key);
          if (known === undefined) {
            const t = s[a] / (s[a] - s[b]);
            X = v.length / 3;
            for (let d = 0; d < 3; d++) v.push(v[3 * a + d] + t * (v[3 * b + d] - v[3 * a + d]));
            edgeNew.set(key, X);
            out.push(X);
          } else {
            X = known;
            out.push(X);
          }
        }
        cross.push({ x: X, exit: inA });
      }
    }
    const clean = dedupeLoop(out);
    if (clean.length >= 3) faces.push({ idx: clean, plane: face.plane });
    if (cross.length === 2) {
      const enter = cross[0].exit ? cross[1].x : cross[0].x;
      const exit = cross[0].exit ? cross[0].x : cross[1].x;
      if (enter !== exit) capNext.set(enter, exit); // a face touching the plane at one vertex adds no cap edge
    } else if (cross.length !== 0) {
      throw new Error('voronoi/atom: a convex face crossed the clipping plane more than twice');
    }
  }
  while (capNext.size > 0) {
    const start = capNext.keys().next().value as number;
    const loop: number[] = [];
    let cur = start;
    do {
      loop.push(cur);
      const nxt = capNext.get(cur);
      capNext.delete(cur);
      if (nxt === undefined) throw new Error('voronoi/atom: open cap loop');
      cur = nxt;
    } while (cur !== start);
    faces.push({ idx: loop, plane });
  }
  // compact: keep only referenced vertices
  const remap = new Int32Array(v.length / 3).fill(-1);
  const cv: number[] = [];
  let cnt = 0;
  for (const f of faces) {
    f.idx = f.idx.map((k) => {
      if (remap[k] < 0) {
        remap[k] = cnt++;
        cv.push(v[3 * k], v[3 * k + 1], v[3 * k + 2]);
      }
      return remap[k];
    });
  }
  return { v: cv, f: faces };
};

/** Largest distance of a referenced vertex from the centre. */
const maxRadius = (p: Poly): number => {
  let r2 = 0;
  for (let k = 0; k < p.v.length; k += 3) {
    const q = p.v[k] * p.v[k] + p.v[k + 1] * p.v[k + 1] + p.v[k + 2] * p.v[k + 2];
    if (q > r2) r2 = q;
  }
  return Math.sqrt(r2);
};

/** Volume by fan triangulation about the centre (origin): sum det(a, b, c) / 6. */
const polyVolume = (p: Poly): number => {
  const v = p.v;
  let vol = 0;
  for (const f of p.f) {
    const idx = f.idx;
    const a = 3 * idx[0];
    for (let k = 1; k + 1 < idx.length; k++) {
      const b = 3 * idx[k], c = 3 * idx[k + 1];
      vol += v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1])
        - v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c])
        + v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c]);
    }
  }
  return vol / 6;
};

/** Area of a face from its Newell vector. */
const faceArea = (p: Poly, f: Face): number => {
  const nv = newell(p.v, f.idx);
  return 0.5 * Math.sqrt(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]);
};

/** Image of a pool atom: position, radius, owning atom index. */
interface Images {
  x: Float64Array; y: Float64Array; z: Float64Array;
  r: Float64Array; atom: Int32Array;
  /** Bin grid over the window, cell size D; order holds image indices sorted by bin. */
  wlo: number[]; nb: number[]; start: Int32Array; order: Int32Array;
}

interface Candidate { h: number; dx: number; dy: number; dz: number; c: number; atom: number }

export class ComputeVoronoiAtom extends Compute {
  readonly style = 'voronoi/atom';
  peratomFlag = true;

  private readonly onlyGroup: boolean;
  /** Group bit of the surface group, or null when the surface keyword is absent. */
  private readonly surfaceBit: number | null;
  private readonly radiusName: string | null;
  private readonly edgeMax: number | null;
  private readonly edgeThreshold: number;
  private readonly faceThreshold: number;
  private cachedEpoch = -1;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let onlyGroup = false;
    let surfaceBit: number | null = null;
    let radiusName: string | null = null;
    let edgeMax: number | null = null;
    let edgeThreshold = 0;
    let faceThreshold = 0;
    for (let k = 0; k < args.length; k++) {
      const kw = args[k];
      const where = `compute ${id} (voronoi/atom)`;
      if (kw === 'only_group') { onlyGroup = true; continue; }
      if (kw === 'occupation') {
        throw new StyleError(`${where}: keyword occupation is not supported (it needs the tessellation stored from the first invocation)`);
      }
      if (kw === 'neighbors') {
        const w = args[++k];
        if (w === 'no') continue;
        if (w === 'yes') throw new StyleError(`${where}: keyword neighbors yes is not supported (the local face array is not implemented)`);
        throw new StyleError(`${where}: neighbors must be yes or no (got '${w}')`);
      }
      if (kw === 'peratom') {
        throw new StyleError(`${where}: keyword peratom was removed from LAMMPS and is not accepted`);
      }
      if (kw === 'surface') {
        const g = args[++k];
        if (g === undefined) throw new StyleError(`${where}: surface needs a group ID`);
        surfaceBit = sys.groups.bit(g);
        continue;
      }
      if (kw === 'radius') {
        const w = args[++k];
        if (w === undefined || !w.startsWith('v_')) throw new StyleError(`${where}: radius needs an atom-style variable v_name (got '${w}')`);
        radiusName = w.slice(2);
        continue;
      }
      if (kw === 'edge_histo') {
        const v = parseInt_(args[++k], `${where} edge_histo maxedge`);
        if (v < 1) throw new StyleError(`${where}: edge_histo maxedge must be >= 1 (got ${args[k]})`);
        edgeMax = v;
        continue;
      }
      if (kw === 'edge_threshold') {
        const v = parseNum(args[++k], `${where} edge_threshold minlength`);
        if (v < 0) throw new StyleError(`${where}: edge_threshold must be >= 0 (got ${args[k]})`);
        edgeThreshold = v;
        continue;
      }
      if (kw === 'face_threshold') {
        const v = parseNum(args[++k], `${where} face_threshold minarea`);
        if (v < 0) throw new StyleError(`${where}: face_threshold must be >= 0 (got ${args[k]})`);
        faceThreshold = v;
        continue;
      }
      throw new StyleError(`${where}: unknown keyword '${kw}' (use only_group, surface, radius, edge_histo, edge_threshold, face_threshold, neighbors)`);
    }
    this.onlyGroup = onlyGroup;
    this.surfaceBit = surfaceBit;
    this.radiusName = radiusName;
    this.edgeMax = edgeMax;
    this.edgeThreshold = edgeThreshold;
    this.faceThreshold = faceThreshold;
    this.sizePeratomCols = surfaceBit === null ? 2 : 3;
    if (edgeMax !== null) {
      this.vectorFlag = true;
      this.sizeVector = edgeMax + 1;
      this.vector = new Float64Array(edgeMax + 1);
    }
  }

  protected computePeratom(): void {
    this.ensure();
  }

  protected computeVector(): void {
    this.ensure();
  }

  /** Runs the tessellation once per state epoch and fills both outputs. */
  private ensure(): void {
    if (this.cachedEpoch === this.sys.epoch) return;
    const sys = this.sys;
    const g = sys.geom;
    const s = sys.state;
    const where = `compute ${this.id} (voronoi/atom)`;
    if (sys.dimension !== 3) throw new StyleError(`${where}: 2d systems are not supported (the cell would need the z extent of the box)`);
    if (g.triclinic) throw new StyleError(`${where}: triclinic boxes are not supported`);
    for (let d = 0; d < 3; d++) {
      if (!g.periodic[d]) throw new StyleError(`${where}: non-periodic boundaries are not supported (dimension ${'xyz'[d]} is not periodic)`);
    }
    const n = s.n;
    const L = [g.hi[0] - g.lo[0], g.hi[1] - g.lo[1], g.hi[2] - g.lo[2]];
    const rad = new Float64Array(n);
    if (this.radiusName !== null) {
      const rv = sys.atomVariable(this.radiusName);
      for (let i = 0; i < n; i++) rad[i] = rv[i];
    }
    const pool: number[] = [];
    const targets: number[] = [];
    let rmax = 0;
    for (let i = 0; i < n; i++) {
      const inGroup = (s.mask[i] & this.groupBit) !== 0;
      if (inGroup) targets.push(i);
      if (!this.onlyGroup || inGroup) {
        pool.push(i);
        if (Math.abs(rad[i]) > rmax) rmax = Math.abs(rad[i]);
      }
    }
    const cols = this.sizePeratomCols;
    const out = new Float64Array(cols * n);
    const histo = new Float64Array(this.edgeMax !== null ? this.edgeMax + 1 : 0);
    if (targets.length > 0) {
      // window over all owned positions
      const wlo = [Infinity, Infinity, Infinity], whi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < 3; d++) {
          const v = s.x[3 * i + d];
          if (v < wlo[d]) wlo[d] = v;
          if (v > whi[d]) whi[d] = v;
        }
      }
      const vol = L[0] * L[1] * L[2];
      let D = 1.5 * Math.cbrt(vol / Math.max(1, pool.length));
      let results: { vol: number; nf: number; surf: number; edges: number[] }[] | null = null;
      for (let attempt = 0; attempt < 80 && results === null; attempt++) {
        const img = this.buildImages(D, pool, rad, s.x, wlo, whi, L);
        results = this.tessellateAll(D, targets, img, rad, rmax, s.x, s.mask);
        if (results === null) D *= 1.5;
      }
      if (results === null) throw new StyleError(`${where}: the Voronoi search radius did not converge`);
      for (let t = 0; t < targets.length; t++) {
        const i = targets[t];
        const r = results[t];
        out[cols * i] = r.vol;
        out[cols * i + 1] = r.nf;
        if (cols === 3) out[cols * i + 2] = r.surf;
        if (this.edgeMax !== null) {
          // vector entry k (1-based) counts faces with k edges; the last entry (maxedge+1) counts faces with more than maxedge edges
          for (const ne of r.edges) {
            if (ne > this.edgeMax) histo[this.edgeMax]++;
            else if (ne >= 1) histo[ne - 1]++;
          }
        }
      }
    }
    this.arrayAtom = out;
    if (this.edgeMax !== null) this.vector = histo;
    this.cachedEpoch = sys.epoch;
  }

  /** Periodic images of the pool atoms inside the window [wlo - D, whi + D], binned with cell size D. */
  private buildImages(D: number, pool: number[], rad: Float64Array, x: Float64Array,
    wlo0: number[], whi0: number[], L: number[]): Images {
    const wlo = [wlo0[0] - D, wlo0[1] - D, wlo0[2] - D];
    const whi = [whi0[0] + D, whi0[1] + D, whi0[2] + D];
    const X: number[] = [], Y: number[] = [], Z: number[] = [], R: number[] = [], A: number[] = [];
    for (const j of pool) {
      const kr: [number, number][] = [];
      for (let d = 0; d < 3; d++) {
        const xj = x[3 * j + d];
        kr.push([Math.ceil((wlo[d] - xj) / L[d]), Math.floor((whi[d] - xj) / L[d])]);
      }
      for (let kx = kr[0][0]; kx <= kr[0][1]; kx++) {
        for (let ky = kr[1][0]; ky <= kr[1][1]; ky++) {
          for (let kz = kr[2][0]; kz <= kr[2][1]; kz++) {
            X.push(x[3 * j] + kx * L[0]);
            Y.push(x[3 * j + 1] + ky * L[1]);
            Z.push(x[3 * j + 2] + kz * L[2]);
            R.push(rad[j]);
            A.push(j);
          }
        }
      }
    }
    const m = X.length;
    const nb = [0, 1, 2].map((d) => Math.floor((whi[d] - wlo[d]) / D) + 1);
    const nbins = nb[0] * nb[1] * nb[2];
    const binOf = new Int32Array(m);
    const start = new Int32Array(nbins + 1);
    for (let q = 0; q < m; q++) {
      const b = this.binIndex([X[q], Y[q], Z[q]], wlo, nb, D);
      binOf[q] = b;
      start[b + 1]++;
    }
    for (let b = 0; b < nbins; b++) start[b + 1] += start[b];
    const fill = start.slice(0, nbins);
    const order = new Int32Array(m);
    for (let q = 0; q < m; q++) order[fill[binOf[q]]++] = q;
    return {
      x: Float64Array.from(X), y: Float64Array.from(Y), z: Float64Array.from(Z),
      r: Float64Array.from(R), atom: Int32Array.from(A), wlo, nb, start, order,
    };
  }

  private binIndex(p: number[], wlo: number[], nb: number[], D: number): number {
    const c = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      let b = Math.floor((p[d] - wlo[d]) / D);
      if (b < 0) b = 0;
      if (b > nb[d] - 1) b = nb[d] - 1;
      c[d] = b;
    }
    return (c[2] * nb[1] + c[1]) * nb[0] + c[0];
  }

  /**
   * Tessellates every target atom with search radius D. Returns null when one
   * cell is not certified exact at this D (its farthest vertex is not inside
   * f(D)), so the caller grows D and retries.
   */
  private tessellateAll(D: number, targets: number[], img: Images, rad: Float64Array, rmax: number,
    x: Float64Array, mask: Int32Array): { vol: number; nf: number; surf: number; edges: number[] }[] | null {
    const D2 = D * D;
    const eps = 1e-12 * D2;
    const res: { vol: number; nf: number; surf: number; edges: number[] }[] = [];
    const cands: Candidate[] = [];
    const [nbx, nby, nbz] = img.nb;
    const wlo = img.wlo;
    for (const i of targets) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ri = rad[i];
      cands.length = 0;
      const bx0 = Math.min(nbx - 1, Math.max(0, Math.floor((xi - wlo[0]) / D)));
      const by0 = Math.min(nby - 1, Math.max(0, Math.floor((yi - wlo[1]) / D)));
      const bz0 = Math.min(nbz - 1, Math.max(0, Math.floor((zi - wlo[2]) / D)));
      for (let bz = Math.max(0, bz0 - 1); bz <= Math.min(nbz - 1, bz0 + 1); bz++) {
        for (let by = Math.max(0, by0 - 1); by <= Math.min(nby - 1, by0 + 1); by++) {
          for (let bx = Math.max(0, bx0 - 1); bx <= Math.min(nbx - 1, bx0 + 1); bx++) {
            const b = (bz * nby + by) * nbx + bx;
            for (let t = img.start[b]; t < img.start[b + 1]; t++) {
              const q = img.order[t];
              const dx = img.x[q] - xi, dy = img.y[q] - yi, dz = img.z[q] - zi;
              const d2 = dx * dx + dy * dy + dz * dz;
              const j = img.atom[q];
              if (d2 < 1e-24) {
                if (j === i) continue; // the atom itself
                throw new StyleError(`compute ${this.id} (voronoi/atom): atoms ${i} and ${j} coincide`);
              }
              if (d2 > D2) continue;
              const d = Math.sqrt(d2);
              const rj = img.r[q];
              const c = 0.5 * (d2 + ri * ri - rj * rj);
              cands.push({ h: c / d, dx, dy, dz, c, atom: j });
            }
          }
        }
      }
      cands.sort((a, b) => a.h - b.h);
      let poly: Poly = cubePoly(D);
      let R = D * Math.sqrt(3);
      for (let k = 0; k < cands.length; k++) {
        const cd = cands[k];
        if (cd.h > R + 1e-12 * D) break;
        poly = clipPoly(poly, cd.dx, cd.dy, cd.dz, cd.c, k, eps);
        R = maxRadius(poly);
        if (poly.f.length === 0) break;
      }
      // safe bound: every plane from an atom farther than D has h >= f(D)
      const fD = (D2 + ri * ri - rmax * rmax) / (2 * D);
      if (!(R < fD)) return null;
      let vol = 0, nf = 0, surf = 0;
      const edges: number[] = [];
      for (const f of poly.f) {
        const area = faceArea(poly, f);
        if (f.plane < 0) continue; // cube face: cannot remain once R < fD
        const cand = cands[f.plane];
        const j = cand ? cand.atom : -1;
        if (this.surfaceBit !== null && j >= 0 && (mask[j] & this.surfaceBit) !== 0) surf += area;
        if (area > this.faceThreshold) {
          nf++;
          if (this.edgeMax !== null) {
            let ne = 0;
            const m = f.idx.length;
            for (let k = 0; k < m; k++) {
              const a = 3 * f.idx[k], b = 3 * f.idx[(k + 1) % m];
              const ex = poly.v[a] - poly.v[b], ey = poly.v[a + 1] - poly.v[b + 1], ez = poly.v[a + 2] - poly.v[b + 2];
              if (Math.sqrt(ex * ex + ey * ey + ez * ez) > this.edgeThreshold) ne++;
            }
            edges.push(ne);
          }
        }
      }
      vol = polyVolume(poly);
      res.push({ vol, nf, surf, edges });
    }
    return res;
  }
}
