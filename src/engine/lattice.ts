import type { UnitSystem } from './types';
import { StyleError } from './force/types';

/*
 * Lattices — docs.lammps.org/lattice.html.
 *
 * "style = none or sc or bcc or fcc or hcp or diamond or sq or sq2 or hex or
 * custom". Cubic styles "define a cubic unit cell with edge length = 1.0.
 * This means a1 = 1 0 0, a2 = 0 1 0, and a3 = 0 0 1. Style hcp has a1 = 1 0
 * 0, a2 = 0 sqrt(3) 0, and a3 = 0 0 sqrt(8/3)." hex: "a1 = 1 0 0 and a2 = 0
 * sqrt(3) 0. It has 2 basis atoms, one at the corner and one at the center of
 * the rectangle." "A lattice of style custom allows you to specify a1, a2,
 * a3, and a list of basis atoms". "The position vector x of a basis atom
 * within the unit cell is a linear combination of the unit cell's 3 edge
 * vectors, i.e. x = bx a1 + by a2 + bz a3".
 * scale: "For all unit styles except lj, the scale argument is specified in
 * the distance units"; for lj "factor^dim = rho/rho*, where rho = N/V with
 * V = the volume of the lattice unit cell and N = the number of basis atoms".
 * origin: "x,y,z = fractions of a unit cell (0 <= x,y,z < 1)". orient: "dim
 * i j k ... E.g. "orient x 2 1 0" means the x-axis in the simulation box will
 * be the [210] lattice direction ... they must be mutually orthogonal and
 * obey the right-hand rule". spacing: "If the spacing option is not
 * specified, the lattice spacings are computed by LAMMPS in the following
 * way. A unit cell of the lattice is mapped into the simulation box (scaled
 * and rotated) ... The lattice spacing in X is defined as the difference
 * between the min/max extent of the x coordinates of the 8 corner points of
 * the modified unit cell (4 in 2d)." "By default, a "lattice none 1.0" is
 * defined, which means the lattice spacing is the same as one distance unit".
 * Standard basis coordinates: bcc (0,0,0),(1/2,1/2,1/2); fcc corner + face
 * centers; hcp (0,0,0),(1/2,1/2,0),(1/2,5/6,1/2),(0,1/3,1/2); diamond = fcc
 * plus fcc shifted by (1/4,1/4,1/4); sq2 corner + center.
 */

export type LatticeStyle = 'none' | 'sc' | 'bcc' | 'fcc' | 'hcp' | 'diamond' | 'sq' | 'sq2' | 'hex' | 'custom';
export const LATTICE_STYLES: LatticeStyle[] = ['none', 'sc', 'bcc', 'fcc', 'hcp', 'diamond', 'sq', 'sq2', 'hex', 'custom'];
export const isLatticeStyle = (s: string): s is LatticeStyle => (LATTICE_STYLES as string[]).includes(s);

type V3 = [number, number, number];
const SQRT3 = Math.sqrt(3);

const DEFS: Record<Exclude<LatticeStyle, 'none' | 'custom'>, { dim: 2 | 3; a: [V3, V3, V3]; basis: V3[] }> = {
  sc: { dim: 3, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [[0, 0, 0]] },
  bcc: { dim: 3, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [[0, 0, 0], [0.5, 0.5, 0.5]] },
  fcc: { dim: 3, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]] },
  hcp: { dim: 3, a: [[1, 0, 0], [0, SQRT3, 0], [0, 0, Math.sqrt(8 / 3)]], basis: [[0, 0, 0], [0.5, 0.5, 0], [0.5, 5 / 6, 0.5], [0, 1 / 3, 0.5]] },
  diamond: {
    dim: 3, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    basis: [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.25, 0.25, 0.25], [0.25, 0.75, 0.75], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25]],
  },
  sq: { dim: 2, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [[0, 0, 0]] },
  sq2: { dim: 2, a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [[0, 0, 0], [0.5, 0.5, 0]] },
  hex: { dim: 2, a: [[1, 0, 0], [0, SQRT3, 0], [0, 0, 1]], basis: [[0, 0, 0], [0.5, 0.5, 0]] },
};

export interface Lattice {
  style: LatticeStyle;
  /** Multiplicative factor applied to the unit cell (lattice constant). */
  factor: number;
  /** Lattice spacings along x, y, z ("units lattice" scale factors). */
  spacing: V3;
  a: [V3, V3, V3];
  basis: V3[];
  origin: V3;
  /** Rows: the lattice direction along x, y, z (unnormalized). */
  orient: [V3, V3, V3];
  /** Box position of a lattice position (i, j, k as fractions of a1, a2, a3). */
  toBox(f: V3): V3;
  /** Inverse of toBox. */
  fromBox(x: V3): V3;
}

/** lattice style scale [keywords]; throws StyleError for bad input. */
export const makeLattice = (style: LatticeStyle, scale: number, units: UnitSystem, dimension: 2 | 3, kw: string[] = []): Lattice => {
  if (!(scale > 0)) throw new StyleError(`lattice scale must be > 0, got ${scale}`);
  if (style === 'none') {
    if (kw.length) throw new StyleError('lattice none takes no keywords');
    const f = scale;
    return {
      style, factor: f, spacing: [f, f, f], a: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], basis: [], origin: [0, 0, 0],
      orient: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], toBox: (p) => [p[0] * f, p[1] * f, p[2] * f], fromBox: (x) => [x[0] / f, x[1] / f, x[2] / f],
    };
  }
  let a: [V3, V3, V3];
  let basis: V3[];
  if (style === 'custom') { a = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; basis = []; } else {
    const d = DEFS[style];
    if (d.dim !== dimension) throw new StyleError(`lattice ${style} is ${d.dim}d but the simulation is ${dimension}d`);
    a = d.a.map((v) => [...v]) as [V3, V3, V3];
    basis = d.basis.map((v) => [...v] as V3);
  }
  let origin: V3 = [0, 0, 0];
  const orient: [V3, V3, V3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  let spacing: V3 | null = null;
  const num = (w: string | undefined, what: string) => {
    const v = Number(w);
    if (w === undefined || !Number.isFinite(v)) throw new StyleError(`lattice: expected a number for ${what}, got '${w ?? ''}'`);
    return v;
  };
  for (let k = 0; k < kw.length;) {
    const key = kw[k];
    const v3 = (): V3 => [num(kw[k + 1], key), num(kw[k + 2], key), num(kw[k + 3], key)];
    switch (key) {
      case 'origin': {
        origin = v3();
        if (origin.some((o) => o < 0 || o >= 1)) throw new StyleError('lattice origin values must be >= 0 and < 1');
        k += 4;
        break;
      }
      case 'orient': {
        const d = 'xyz'.indexOf(kw[k + 1]);
        if (d < 0) throw new StyleError("lattice orient needs a dimension x, y or z");
        const v: V3 = [num(kw[k + 2], 'orient'), num(kw[k + 3], 'orient'), num(kw[k + 4], 'orient')];
        if (v.some((c) => !Number.isInteger(c))) throw new StyleError('lattice orient directions must be integers');
        orient[d] = v;
        k += 5;
        break;
      }
      case 'spacing': spacing = v3(); k += 4; break;
      case 'a1': case 'a2': case 'a3':
        if (style !== 'custom') throw new StyleError(`lattice ${key} can only be used with style custom`);
        a[Number(key[1]) - 1] = v3();
        k += 4;
        break;
      case 'basis': {
        if (style !== 'custom') throw new StyleError('lattice basis can only be used with style custom');
        const b = v3();
        if (b.some((c) => c < 0 || c >= 1)) throw new StyleError('lattice basis coordinates must be >= 0 and < 1');
        basis.push(b);
        k += 4;
        break;
      }
      case 'triclinic/general':
        throw new StyleError('lattice triclinic/general is not supported (use a restricted triclinic box)');
      default:
        throw new StyleError(`unknown lattice keyword '${key}'`);
    }
  }
  if (style === 'custom' && !basis.length) throw new StyleError('lattice custom needs at least one basis atom');
  if (dimension === 2) {
    if (origin[2] !== 0) throw new StyleError('lattice origin z must be 0.0 for 2d');
    if (orient[0][2] !== 0 || orient[1][2] !== 0 || orient[2][0] !== 0 || orient[2][1] !== 0) {
      throw new StyleError('lattice orient: in 2d the x and y vectors need a 0 third component and z must be 0 0 k');
    }
    if (basis.some((b) => b[2] !== 0)) throw new StyleError('lattice basis z must be 0.0 for 2d');
  }
  // orthogonality and right-handedness of orient
  const dot = (p: V3, q: V3) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const cross = (p: V3, q: V3): V3 => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  if (dot(orient[0], orient[1]) !== 0 || dot(orient[1], orient[2]) !== 0 || dot(orient[0], orient[2]) !== 0) {
    throw new StyleError('lattice orient vectors are not orthogonal');
  }
  if (dot(cross(orient[0], orient[1]), orient[2]) <= 0) throw new StyleError('lattice orient vectors are not right-handed');
  const R = orient.map((v) => { const n = Math.sqrt(dot(v, v)); return [v[0] / n, v[1] / n, v[2] / n] as V3; });
  // scale factor
  let factor = scale;
  if (units.style === 'lj') {
    const vol = dimension === 3
      ? Math.abs(dot(a[0], cross(a[1], a[2])))
      : Math.abs(a[0][0] * a[1][1] - a[0][1] * a[1][0]);
    factor = Math.pow((basis.length / vol) / scale, 1 / dimension);
  }
  // unit-cell position -> box: rotate(factor * (f0 a1 + f1 a2 + f2 a3))
  const toBoxRaw = (f: V3): V3 => {
    const p: V3 = [0, 0, 0];
    for (let d = 0; d < 3; d++) p[d] = factor * (f[0] * a[0][d] + f[1] * a[1][d] + f[2] * a[2][d]);
    return [dot(R[0], p), dot(R[1], p), dot(R[2], p)];
  };
  // spacings from the 8 (4 in 2d) corners of the transformed unit cell
  if (!spacing) {
    const mn: V3 = [Infinity, Infinity, Infinity], mx: V3 = [-Infinity, -Infinity, -Infinity];
    for (const c of [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]] as V3[]) {
      if (dimension === 2 && c[2] === 1) continue;
      const p = toBoxRaw(c);
      for (let d = 0; d < 3; d++) { mn[d] = Math.min(mn[d], p[d]); mx[d] = Math.max(mx[d], p[d]); }
    }
    spacing = [mx[0] - mn[0], mx[1] - mn[1], dimension === 2 ? factor * Math.sqrt(dot(a[2], a[2])) : mx[2] - mn[2]];
  }
  // origin: fractions of a unit cell
  const toBox = (f: V3): V3 => toBoxRaw([f[0] + origin[0], f[1] + origin[1], f[2] + origin[2]]);
  // inverse: undo the rotation, then solve with the unit cell
  const M = [0, 1, 2].map((d) => [a[0][d] * factor, a[1][d] * factor, a[2][d] * factor]);
  const inv = invert3(M);
  const fromBox = (x: V3): V3 => {
    const p: V3 = [R[0][0] * x[0] + R[1][0] * x[1] + R[2][0] * x[2], R[0][1] * x[0] + R[1][1] * x[1] + R[2][1] * x[2], R[0][2] * x[0] + R[1][2] * x[1] + R[2][2] * x[2]];
    const f: V3 = [0, 0, 0];
    for (let r = 0; r < 3; r++) f[r] = inv[r][0] * p[0] + inv[r][1] * p[1] + inv[r][2] * p[2];
    return [f[0] - origin[0], f[1] - origin[1], f[2] - origin[2]];
  };
  return { style, factor, spacing, a, basis, origin, orient, toBox, fromBox };
};

const invert3 = (m: number[][]): number[][] => {
  const [[a, b, c], [d, e, f], [g, h, k]] = m;
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new StyleError('lattice unit cell is degenerate (a1, a2, a3 are not independent)');
  const s = 1 / det;
  return [
    [A * s, -(b * k - c * h) * s, (b * f - c * e) * s],
    [B * s, (a * k - c * g) * s, -(a * f - c * d) * s],
    [C * s, -(a * h - b * g) * s, (a * e - b * d) * s],
  ];
};

/**
 * Lattice points (with their basis index) inside an axis-aligned bounding
 * box [lo, hi], tested by `accept`. Points are generated in a fixed order
 * (k, j, i, basis) so that atom IDs are reproducible.
 */
export const latticeSites = (
  lat: Lattice, lo: readonly number[], hi: readonly number[], dimension: 2 | 3,
  accept: (x: number, y: number, z: number) => boolean,
): { x: number[]; basis: number[] } => {
  // range of unit cells covering the box corners
  const lmin = [Infinity, Infinity, Infinity], lmax = [-Infinity, -Infinity, -Infinity];
  for (const cx of [lo[0], hi[0]]) for (const cy of [lo[1], hi[1]]) for (const cz of [lo[2], hi[2]]) {
    const f = lat.fromBox([cx, cy, dimension === 2 ? 0 : cz]);
    for (let d = 0; d < 3; d++) { lmin[d] = Math.min(lmin[d], f[d]); lmax[d] = Math.max(lmax[d], f[d]); }
  }
  const i0 = Math.floor(lmin[0]) - 1, i1 = Math.ceil(lmax[0]) + 1;
  const j0 = Math.floor(lmin[1]) - 1, j1 = Math.ceil(lmax[1]) + 1;
  const k0 = dimension === 2 ? 0 : Math.floor(lmin[2]) - 1, k1 = dimension === 2 ? 0 : Math.ceil(lmax[2]) + 1;
  const x: number[] = [];
  const basis: number[] = [];
  for (let k = k0; k <= k1; k++) {
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        lat.basis.forEach((b, m) => {
          const p = lat.toBox([i + b[0], j + b[1], k + b[2]]);
          if (dimension === 2) p[2] = 0;
          if (accept(p[0], p[1], p[2])) { x.push(p[0], p[1], p[2]); basis.push(m); }
        });
      }
    }
  }
  return { x, basis };
};

/** v1 helper kept for the GPU harness and tests: points in [lo, hi) of a default lattice. */
export const latticePoints = (
  lat: Lattice, lo: readonly number[], hi: readonly number[], dimension: 2 | 3,
  region?: { lo: readonly number[]; hi: readonly number[] },
): Float64Array => {
  const tol = (d: number) => 1e-9 * Math.max(1, Math.abs(hi[d] - lo[d]));
  const { x } = latticeSites(lat, lo, hi, dimension, (px, py, pz) => {
    const p = [px, py, pz];
    for (let d = 0; d < (dimension === 2 ? 2 : 3); d++) {
      const t = tol(d);
      if (p[d] < lo[d] - t || p[d] >= hi[d] - t) return false;
      if (region && (p[d] < region.lo[d] - t || p[d] > region.hi[d] + t)) return false;
    }
    return true;
  });
  return Float64Array.from(x);
};
