import { Atom, BoxBounds, MoleculeData, TrajectoryFrame } from '../types';

/**
 * Trajectory analysis service — pure functions for RDF, MSD, density, etc.
 * All functions are unit-testable and have no three.js dependency.
 *
 * These run off the main thread via `services/analysisClient` (see
 * `workers/analysis.worker.ts`); they are still safe to call directly in
 * tests or as a no-worker fallback.
 *
 * Performance notes — these all used to be hot:
 *  - RDF uses a periodic CELL LIST, so it is O(N·k) in the number of atoms
 *    within the cutoff rather than O(N²) over every pair.
 *  - MSD precomputes a stable atom ordering once instead of rebuilding an
 *    id→atom Map inside the (lag × origin) loop.
 *  - No function spreads an atom array into Math.min/Math.max: that throws
 *    RangeError once the array passes the engine's argument limit, which a
 *    60k-atom frame already approaches.
 */

export interface RDFPoint { r: number; g: number; count: number; }
export interface MSDPoint { t: number; msd: number; }
export interface HistogramBin { x0: number; x1: number; count: number; density: number; }
export interface DensityProfile { bins: HistogramBin[]; axis: 'x' | 'y' | 'z'; }

/**
 * True when the frame carries LAMMPS image flags, so displacements can be
 * unwrapped exactly instead of falling back to the minimum-image convention.
 */
export const hasImageFlags = (atoms: Atom[]): boolean =>
  atoms.length > 0 && atoms[0].ix !== undefined && atoms[0].iy !== undefined &&
  atoms[0].iz !== undefined;

/**
 * True when MSD can measure displacement EXACTLY — either the coordinates are
 * already absolute, or image flags let them be reconstructed. False means the
 * minimum-image fallback, whose MSD saturates near (L/2)².
 */
export const msdIsExact = (
  frames: TrajectoryFrame[],
  box?: BoxBounds,
): boolean => {
  if (frames.length === 0) return false;
  if (frames[0].coordsUnwrapped === true) return true;
  return !!box && hasImageFlags(frames[0].atoms);
};

/** Largest value in a numeric array, floored at 1 (used for normalisation). */
const maxOf = (values: number[]): number => {
  let m = 1;
  for (let i = 0; i < values.length; i++) if (values[i] > m) m = values[i];
  return m;
};

/** Extent of one coordinate over an atom list, without argument spreading. */
const extentOf = (atoms: Atom[], axis: 'x' | 'y' | 'z'): { lo: number; hi: number } => {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < atoms.length; i++) {
    const v = atoms[i][axis];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return Number.isFinite(lo) ? { lo, hi } : { lo: 0, hi: 1 };
};

/** Minimum image convention for PBC */
const pbcDelta = (d: number, boxLen: number): number => {
  if (boxLen <= 0) return d;
  // bring into [-L/2, L/2]
  return d - Math.round(d / boxLen) * boxLen;
};

/**
 * Minimum-image distance between two atoms. The cell list inlines this for
 * speed; it stays exported so tests can check the accelerated RDF against a
 * brute-force reference that uses exactly the same convention.
 *
 * A thin z slab (lz <= 2, i.e. a 2D run) is deliberately NOT wrapped in z.
 */
export const distancePBC = (
  a: Atom, b: Atom, box?: BoxBounds,
): number => {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  let dz = a.z - b.z;
  if (box) {
    const lx = box.xhi - box.xlo;
    const ly = box.yhi - box.ylo;
    const lz = box.zhi - box.zlo;
    dx = pbcDelta(dx, lx);
    dy = pbcDelta(dy, ly);
    if (lz > 2) dz = pbcDelta(dz, lz);
  }
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

/* ------------------------------------------------------------------ */
/* Periodic cell list                                                  */
/* ------------------------------------------------------------------ */

interface CellList {
  n: number;
  x: Float64Array;
  y: Float64Array;
  z: Float64Array;
  nx: number; ny: number; nz: number;
  lx: number; ly: number; lz: number;
  /** Prefix offsets into `items`, length nx*ny*nz + 1. */
  start: Int32Array;
  /** Atom indices ordered by cell. */
  items: Int32Array;
  px: boolean; py: boolean; pz: boolean;
  /** Distinct neighbour offsets per axis — see `axisOffsets`. */
  ox: Int32Array; oy: Int32Array; oz: Int32Array;
}

/**
 * Distinct neighbour-cell offsets along one axis.
 *
 * With 3 or more cells the usual -1/0/+1 shell is right. With 1 or 2 cells a
 * periodic -1/0/+1 scan would visit the same cell twice and double-count
 * pairs, so enumerate the axis once instead.
 */
const axisOffsets = (n: number, periodic: boolean): Int32Array => {
  if (!periodic || n >= 3) return Int32Array.from([-1, 0, 1]);
  return Int32Array.from({ length: n }, (_, k) => k);
};

const buildCellList = (
  atoms: Atom[],
  box: BoxBounds | undefined,
  cutoff: number,
): CellList => {
  const n = atoms.length;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const z = new Float64Array(n);

  const ex = box ? { lo: box.xlo, hi: box.xhi } : extentOf(atoms, 'x');
  const ey = box ? { lo: box.ylo, hi: box.yhi } : extentOf(atoms, 'y');
  const ez = box ? { lo: box.zlo, hi: box.zhi } : extentOf(atoms, 'z');

  const lx = Math.max(ex.hi - ex.lo, 1e-12);
  const ly = Math.max(ey.hi - ey.lo, 1e-12);
  const lz = Math.max(ez.hi - ez.lo, 1e-12);

  // Matches distancePBC: a thin z slab (2D run) is not wrapped.
  const px = !!box;
  const py = !!box;
  const pz = !!box && lz > 2;

  const wrap = (v: number, lo: number, len: number, periodic: boolean): number => {
    if (!periodic) return Math.min(Math.max(v - lo, 0), len * (1 - 1e-12));
    let t = (v - lo) % len;
    if (t < 0) t += len;
    return t;
  };

  for (let i = 0; i < n; i++) {
    x[i] = wrap(atoms[i].x, ex.lo, lx, px);
    y[i] = wrap(atoms[i].y, ey.lo, ly, py);
    z[i] = wrap(atoms[i].z, ez.lo, lz, pz);
  }

  const nx = Math.max(1, Math.floor(lx / cutoff));
  const ny = Math.max(1, Math.floor(ly / cutoff));
  const nz = Math.max(1, Math.floor(lz / cutoff));
  const cells = nx * ny * nz;

  const cellOf = new Int32Array(n);
  const counts = new Int32Array(cells);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(nx - 1, (x[i] / lx * nx) | 0);
    const cy = Math.min(ny - 1, (y[i] / ly * ny) | 0);
    const cz = Math.min(nz - 1, (z[i] / lz * nz) | 0);
    const c = (cz * ny + cy) * nx + cx;
    cellOf[i] = c;
    counts[c]++;
  }

  const start = new Int32Array(cells + 1);
  for (let c = 0; c < cells; c++) start[c + 1] = start[c] + counts[c];
  const cursor = start.slice(0, cells);
  const items = new Int32Array(n);
  for (let i = 0; i < n; i++) items[cursor[cellOf[i]]++] = i;

  return {
    n, x, y, z, nx, ny, nz, lx, ly, lz, start, items, px, py, pz,
    ox: axisOffsets(nx, px), oy: axisOffsets(ny, py), oz: axisOffsets(nz, pz),
  };
};

/**
 * Visits every unordered pair (i, j) closer than `rMax` exactly once, through the periodic cell list. `dx, dy, dz`
 * is the minimum-image separation r_i - r_j and `d2` its squared length (the convention of distancePBC: a thin z
 * slab is not wrapped in z).
 */
const forEachPair = (
  cl: CellList,
  rMax: number,
  visit: (i: number, j: number, dx: number, dy: number, dz: number, d2: number) => void,
): void => {
  const r2max = rMax * rMax;
  const { nx, ny, nz, lx, ly, lz, x, y, z, start, items, ox, oy, oz } = cl;
  const halfX = lx / 2, halfY = ly / 2, halfZ = lz / 2;

  for (let cz = 0; cz < nz; cz++) {
    for (let cy = 0; cy < ny; cy++) {
      for (let cx = 0; cx < nx; cx++) {
        const c = (cz * ny + cy) * nx + cx;
        const cLo = start[c], cHi = start[c + 1];
        if (cLo === cHi) continue;

        for (let dzi = 0; dzi < oz.length; dzi++) {
          let kz = oz.length === 3 ? cz + oz[dzi] : oz[dzi];
          if (cl.pz) { if (kz < 0) kz += nz; else if (kz >= nz) kz -= nz; }
          else if (kz < 0 || kz >= nz) continue;

          for (let dyi = 0; dyi < oy.length; dyi++) {
            let ky = oy.length === 3 ? cy + oy[dyi] : oy[dyi];
            if (cl.py) { if (ky < 0) ky += ny; else if (ky >= ny) ky -= ny; }
            else if (ky < 0 || ky >= ny) continue;

            for (let dxi = 0; dxi < ox.length; dxi++) {
              let kx = ox.length === 3 ? cx + ox[dxi] : ox[dxi];
              if (cl.px) { if (kx < 0) kx += nx; else if (kx >= nx) kx -= nx; }
              else if (kx < 0 || kx >= nx) continue;

              const k = (kz * ny + ky) * nx + kx;
              const kLo = start[k], kHi = start[k + 1];
              if (kLo === kHi) continue;

              for (let ii = cLo; ii < cHi; ii++) {
                const i = items[ii];
                const xi = x[i], yi = y[i], zi = z[i];
                for (let jj = kLo; jj < kHi; jj++) {
                  const j = items[jj];
                  if (j <= i) continue;            // each unordered pair once
                  let dx = xi - x[j];
                  let dy = yi - y[j];
                  let dz = zi - z[j];
                  if (cl.px) { if (dx > halfX) dx -= lx; else if (dx < -halfX) dx += lx; }
                  if (cl.py) { if (dy > halfY) dy -= ly; else if (dy < -halfY) dy += ly; }
                  if (cl.pz) { if (dz > halfZ) dz -= lz; else if (dz < -halfZ) dz += lz; }
                  const d2 = dx * dx + dy * dy + dz * dz;
                  if (d2 >= r2max) continue;
                  visit(i, j, dx, dy, dz, d2);
                }
              }
            }
          }
        }
      }
    }
  }
};

/**
 * Histogram every pair closer than `rMax` into `hist`, adding 2 per unordered
 * pair so the caller's per-atom normalisation sees ordered-pair counts.
 */
const accumulatePairs = (
  cl: CellList,
  rMax: number,
  dr: number,
  bins: number,
  hist: number[],
): void => {
  forEachPair(cl, rMax, (_i, _j, _dx, _dy, _dz, d2) => {
    const bin = Math.floor(Math.sqrt(d2) / dr);
    if (bin >= 0 && bin < bins) hist[bin] += 2;
  });
};

/**
 * Minimum-image neighbour vectors of a frame within `cutoff`: for every atom, the flat list
 * [x, y, z, ...] of r_j - r_i over its neighbours j (each pair found once through the cell list).
 */
const neighbourVectors = (frame: TrajectoryFrame, cutoff: number): number[][] => {
  const vec: number[][] = Array.from({ length: frame.atoms.length }, () => []);
  forEachPair(buildCellList(frame.atoms, frame.box, cutoff), cutoff, (i, j, dx, dy, dz) => {
    // d = r_i - r_j, so r_j - r_i = -d and r_i - r_j = d
    vec[i].push(-dx, -dy, -dz);
    vec[j].push(dx, dy, dz);
  });
  return vec;
};

export interface CoordinationResult {
  /** Coordination number of each atom (neighbours closer than the cutoff). */
  perAtom: number[];
  /** atomsWith[k] = number of atoms whose coordination number is k, for k = 0 .. the largest. */
  atomsWith: number[];
  /** Mean coordination number, (1/N) sum_i CN_i. */
  mean: number;
  cutoff: number;
}

/**
 * Coordination number: CN_i = #{ j != i : |r_ij| < cutoff }, with the minimum-image distance on the frame's own box
 * (periodic dimensions only, as in computeRDF). Each unordered pair adds one neighbour to both atoms.
 */
export const computeCoordination = (frame: TrajectoryFrame, cutoff: number): CoordinationResult => {
  const n = frame.atoms.length;
  const perAtom = new Array<number>(n).fill(0);
  if (n > 0 && cutoff > 0) {
    forEachPair(buildCellList(frame.atoms, frame.box, cutoff), cutoff, (i, j) => { perAtom[i]++; perAtom[j]++; });
  }
  let max = 0;
  let sum = 0;
  for (const c of perAtom) { if (c > max) max = c; sum += c; }
  const atomsWith = new Array<number>(max + 1).fill(0);
  for (const c of perAtom) atomsWith[c]++;
  return { perAtom, atomsWith, mean: n > 0 ? sum / n : 0, cutoff };
};

/**
 * Bond-angle distribution. For every atom j and every pair of its neighbours i, k within `cutoff` (minimum image),
 * the angle theta_ijk = arccos( (r_i - r_j) . (r_k - r_j) / (|r_i - r_j| |r_k - r_j|) ) in degrees, binned over
 * [0, 180]. The density of bin b is count_b / (total * width), width = 180 / nbins, so sum(density * width) = 1: a
 * probability density per degree. Empty when no angle exists.
 */
export const computeBondAngleDistribution = (frame: TrajectoryFrame, cutoff: number, nbins = 90): HistogramBin[] => {
  if (!(cutoff > 0) || nbins < 1 || frame.atoms.length < 3) return [];
  const width = 180 / nbins;
  const counts = new Array<number>(nbins).fill(0);
  let total = 0;
  for (const v of neighbourVectors(frame, cutoff)) {
    const m = v.length / 3;
    for (let a = 0; a < m; a++) {
      const ax = v[3 * a], ay = v[3 * a + 1], az = v[3 * a + 2];
      const la = Math.sqrt(ax * ax + ay * ay + az * az);
      for (let b = a + 1; b < m; b++) {
        const bx = v[3 * b], by = v[3 * b + 1], bz = v[3 * b + 2];
        const lb = Math.sqrt(bx * bx + by * by + bz * bz);
        if (la === 0 || lb === 0) continue;
        const cos = (ax * bx + ay * by + az * bz) / (la * lb);
        const deg = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
        counts[Math.min(nbins - 1, Math.floor(deg / width))]++;
        total++;
      }
    }
  }
  if (total === 0) return [];
  return counts.map((count, b) => ({
    x0: b * width, x1: (b + 1) * width, count, density: count / (total * width),
  }));
};

export interface StructurePoint { q: number; s: number }

/**
 * Static structure factor from g(r), for an isotropic system of number density rho = N/V:
 *   S(q) = 1 + 4 pi rho * integral_0^rMax r^2 (g(r) - 1) sin(qr) / (qr) dr
 * The integral is trapezoidal on the g(r) grid, and sin(x)/x = 1 at x = 0. q runs from 0 to qMax in nq equal steps
 * (nq values including both ends). An ideal gas, g = 1, gives S = 1 for every q. With `lorch`, the integrand is
 * multiplied by the Lorch window M(r) = sin(pi r / R) / (pi r / R), R the last r of the grid, which damps the
 * ripples (and the unphysical S < 0 at small q) that cutting g(r) at R puts into the transform.
 */
export const computeStructureFactor = (
  rdf: { r: readonly number[]; g: readonly number[] },
  rho: number,
  qMax: number,
  nq = 200,
  lorch = false,
): StructurePoint[] => {
  const R0 = Math.min(rdf.r.length, rdf.g.length);
  if (nq < 2 || R0 < 2 || !(rho > 0) || !(qMax > 0)) return [];
  const sinc = (x: number): number => (Math.abs(x) < 1e-12 ? 1 : Math.sin(x) / x);
  const R = rdf.r[R0 - 1];
  const out: StructurePoint[] = [];
  for (let k = 0; k < nq; k++) {
    const q = qMax * k / (nq - 1);
    let integral = 0;
    let prev = 0;
    for (let i = 0; i < R0; i++) {
      const r = rdf.r[i];
      const f = r * r * (rdf.g[i] - 1) * sinc(q * r) * (lorch ? sinc(Math.PI * r / R) : 1);
      if (i > 0) integral += 0.5 * (prev + f) * (r - rdf.r[i - 1]);
      prev = f;
    }
    out.push({ q, s: 1 + 4 * Math.PI * rho * integral });
  }
  return out;
};

/**
 * Default neighbour cutoff from g(r): the position of the first minimum after the first peak. g(r) is first smoothed
 * with a 5-point moving average; the first peak is the first local maximum above 1, and the minimum is the lowest
 * smoothed value between the peak and twice its distance (so sampling noise is not taken for a shell). When g(r) is
 * still falling at twice the peak distance, 1.25 times the peak position; with no peak, 3.0.
 */
export const defaultCutoff = (rdf: readonly RDFPoint[]): number => {
  const n = rdf.length;
  const g = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, cnt = 0;
    for (let k = Math.max(0, i - 2); k <= Math.min(n - 1, i + 2); k++) { sum += rdf[k].g; cnt++; }
    g[i] = sum / cnt;
  }
  let p = -1;
  for (let i = 1; i + 1 < n; i++) {
    if (g[i] > 1 && g[i] > g[i - 1] && g[i] >= g[i + 1]) { p = i; break; }
  }
  if (p < 0) return 3.0;
  let m = -1, last = -1, gm = Infinity;
  for (let i = p + 1; i < n && rdf[i].r <= 2 * rdf[p].r; i++) {
    last = i;
    if (g[i] < gm) { gm = g[i]; m = i; }
  }
  return m < 0 || m === last ? 1.25 * rdf[p].r : rdf[m].r;
};

/** Coordination, bond angles for one frame (the worker's neighbour analysis). */
export interface NeighbourResult {
  coordination: CoordinationResult;
  angles: HistogramBin[];
}

/** Coordination and bond-angle distribution of one frame at a cutoff. */
export const computeNeighbourAnalysis = (frame: TrajectoryFrame, cutoff: number, nbins = 90): NeighbourResult => ({
  coordination: computeCoordination(frame, cutoff),
  angles: computeBondAngleDistribution(frame, cutoff, nbins),
});

/**
 * Radial Distribution Function g(r)
 * Averaged over the provided frames (caller should sample for speed).
 *  - rMax: max distance (default 12 Å, or half the smallest box side)
 *  - bins: number of histogram bins (default 100)
 *  - For 2D thin boxes (lz < 2), uses 2D normalization (2πr dr) vs 3D (4πr² dr)
 */
export const computeRDF = (
  frames: TrajectoryFrame[],
  box?: BoxBounds,
  opts?: { rMax?: number; bins?: number },
): RDFPoint[] => {
  if (frames.length === 0 || frames[0].atoms.length === 0) return [];
  const N = frames[0].atoms.length;
  if (N < 2) return [];

  const bins = opts?.bins ?? 100;
  let rMax = opts?.rMax ?? 12;
  if (box) {
    const lx = box.xhi - box.xlo;
    const ly = box.yhi - box.ylo;
    const lz = box.zhi - box.zlo;
    const halfMin = Math.min(lx, ly, lz > 2 ? lz : Math.min(lx, ly)) / 2;
    rMax = Math.min(rMax, halfMin * 0.95);
  }
  const dr = rMax / bins;
  const hist = new Array(bins).fill(0);

  // 2D detection: thin z (our 2D sims have lz=1)
  const is2D = box ? (box.zhi - box.zlo) < 2 : false;
  let volume = 1;
  let area = 1;
  if (box) {
    const lx = box.xhi - box.xlo;
    const ly = box.yhi - box.ylo;
    const lz = box.zhi - box.zlo;
    if (is2D) area = lx * ly;
    else volume = lx * ly * lz;
  }

  // Count pairs (i<j) per frame, then average
  let pairFrames = 0;
  for (const frame of frames) {
    const atoms = frame.atoms;
    if (atoms.length !== N) continue; // skip inconsistent frames
    pairFrames++;
    accumulatePairs(buildCellList(atoms, box, rMax), rMax, dr, bins, hist);
  }
  if (pairFrames === 0) return [];

  const points: RDFPoint[] = [];
  const rho = is2D ? N / area : N / volume;
  for (let b = 0; b < bins; b++) {
    const r = (b + 0.5) * dr;
    const count = hist[b] / pairFrames; // average per frame
    // Ideal gas count in shell
    let ideal: number;
    if (is2D) {
      // 2D: annulus area 2πr dr, density N/A
      ideal = rho * 2 * Math.PI * r * dr * N;
      // But hist counts pairs (2 per pair), and we divided by frames, so for RDF:
      // g(r) = (hist / (N * rho * 2πr dr)) ; hist is average count per frame (both directions)
      // For 2D, the count of pairs at distance r is N * rho * 2πr dr * g(r)
      // Our hist is total pairs (both directions) per frame, so per atom it's hist/N
      // So g = hist / (N * rho * 2πr dr)
    } else {
      ideal = rho * 4 * Math.PI * r * r * dr * N;
    }
    // Actually hist is total pairs (both directions) per frame, average. The RDF is per atom.
    // Standard: g(r) = V/N * (hist / (4πr² dr * N)) ??? Let's use the standard per-pair normalization.
    // Simpler: g = (hist / N) / (rho * shellVolume) where shellVolume is 4πr²dr (3D) or 2πr dr (2D)
    // and hist/N is average neighbors per atom at distance r.
    let g: number;
    if (is2D) {
      const shellArea = 2 * Math.PI * r * dr;
      g = hist[b] / pairFrames / N / (rho * shellArea);
    } else {
      const shellVol = 4 * Math.PI * r * r * dr;
      g = hist[b] / pairFrames / N / (rho * shellVol);
    }
    // Clamp absurd values from r~0 bin
    if (!Number.isFinite(g) || r < dr * 0.5) g = 0;
    points.push({ r, g: Math.max(0, g), count: hist[b] / pairFrames });
  }
  return points;
};

/**
 * Mean Squared Displacement MSD(t) = <|r_i(t) - r_i(0)|²>
 * Averaged over atoms, with optional time-origin averaging (every 10 frames).
 * Returns MSD in Å² (or LJ units) vs frame index (caller maps to time).
 */
export const computeMSD = (
  frames: TrajectoryFrame[],
  box?: BoxBounds,
  opts?: { timeOriginStride?: number },
): MSDPoint[] => {
  if (frames.length < 2) return [];
  const N = frames[0].atoms.length;
  if (N === 0) return [];
  const stride = opts?.timeOriginStride ?? Math.max(1, Math.floor(frames.length / 20));

  // Resolve a STABLE atom ordering once. The old implementation rebuilt an
  // id -> atom Map inside the (lag x origin) loop, i.e. O(lags * origins * N)
  // Map insertions before any arithmetic happened.
  const indexOfId = new Map<number, number>();
  frames[0].atoms.forEach((a, i) => indexOfId.set(a.id, i));

  // perm[f][stableIndex] = position in frames[f].atoms, or null when the
  // frame is already in the same order (the usual case for a LAMMPS dump).
  const perm: (Int32Array | null)[] = frames.map((f, fi) => {
    if (fi === 0 || f.atoms.length !== N) {
      return f.atoms.length === N ? null : new Int32Array(N).fill(-1);
    }
    let identity = true;
    const p = new Int32Array(N).fill(-1);
    for (let k = 0; k < f.atoms.length; k++) {
      const si = indexOfId.get(f.atoms[k].id);
      if (si === undefined) continue;
      p[si] = k;
      if (si !== k) identity = false;
    }
    return identity ? null : p;
  });

  const lx = box ? box.xhi - box.xlo : 0;
  const ly = box ? box.yhi - box.ylo : 0;
  const lz = box ? box.zhi - box.zlo : 0;
  const xy = box?.xy ?? 0, xz = box?.xz ?? 0, yz = box?.yz ?? 0;
  const wrapZ = !!box && lz > 2;

  /*
   * Three displacement modes, in order of fidelity.
   *
   * 1. Coordinates ALREADY absolute (`xu yu zu` / `xsu ysu zsu`, flagged by
   *    the parser as `coordsUnwrapped`): take plain differences. Applying the
   *    minimum-image convention here would be actively destructive — a
   *    genuine one-box drift folds to exactly zero.
   * 2. Wrapped coordinates PLUS image flags: reconstruct the absolute position
   *    (docs.lammps.org/dump.html — "A value of 2 means add 2 box lengths to
   *    get the true value"). MSD is then unbounded and a diffusive system
   *    shows the linear growth it should.
   * 3. Wrapped coordinates, no flags: minimum image is all that is available,
   *    and it caps any single displacement at half a box — so MSD SATURATES
   *    near (L/2)² and cannot represent diffusion beyond that. Dump
   *    `ix iy iz` or `xu yu zu` if you need long-time MSD.
   */
  const alreadyAbsolute = frames[0].coordsUnwrapped === true;
  const useImageFlags = !alreadyAbsolute && !!box && hasImageFlags(frames[0].atoms);
  const minimumImage = !alreadyAbsolute && !useImageFlags && !!box;

  const msd: MSDPoint[] = [];
  for (let dt = 0; dt < frames.length; dt++) {
    let sum = 0;
    let count = 0;
    for (let t0 = 0; t0 + dt < frames.length; t0 += stride) {
      const a0 = frames[t0].atoms;
      const a1 = frames[t0 + dt].atoms;
      if (a0.length !== N || a1.length !== N) continue;
      const p0 = perm[t0];
      const p1 = perm[t0 + dt];
      for (let si = 0; si < N; si++) {
        const i0 = p0 ? p0[si] : si;
        const i1 = p1 ? p1[si] : si;
        if (i0 < 0 || i1 < 0) continue;
        const from = a0[i0];
        const to = a1[i1];
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let dz = to.z - from.z;
        if (useImageFlags) {
          // Restricted-triclinic image offset; degenerates to the orthogonal
          // case when the tilt factors are zero.
          const dix = (to.ix ?? 0) - (from.ix ?? 0);
          const diy = (to.iy ?? 0) - (from.iy ?? 0);
          const diz = (to.iz ?? 0) - (from.iz ?? 0);
          dx += dix * lx + diy * xy + diz * xz;
          dy += diy * ly + diz * yz;
          dz += diz * lz;
        } else if (minimumImage) {
          dx = pbcDelta(dx, lx);
          dy = pbcDelta(dy, ly);
          if (wrapZ) dz = pbcDelta(dz, lz);
        }
        sum += dx * dx + dy * dy + dz * dz;
        count++;
      }
    }
    if (count === 0) continue;
    msd.push({ t: dt, msd: sum / count });
  }
  return msd;
};

/** MSD input: uniformly strided frames and the stride, in real frames. */
export interface MsdSample {
  frames: TrajectoryFrame[];
  stride: number;
}

/**
 * MSD with lags expressed in REAL frames. `computeMSD` counts lags in steps of
 * whatever list it is given; when that list is every `stride`-th frame, lag k
 * really spans k * stride frames.
 */
export const msdInRealFrames = (
  sample: MsdSample,
  box: BoxBounds | undefined,
  timeOriginStride: number,
): MSDPoint[] =>
  computeMSD(sample.frames, box, { timeOriginStride })
    .map(p => ({ t: p.t * sample.stride, msd: p.msd }));

/** Density profile along an axis (histogram) */
export const computeDensityProfile = (
  frames: TrajectoryFrame[],
  box: BoxBounds | undefined,
  axis: 'x' | 'y' | 'z' = 'y',
  bins = 30,
): DensityProfile => {
  if (frames.length === 0) return { bins: [], axis };
  const boxLo = axis === 'x' ? box?.xlo : axis === 'y' ? box?.ylo : box?.zlo;
  const boxHi = axis === 'x' ? box?.xhi : axis === 'y' ? box?.yhi : box?.zhi;
  const fallback = boxLo === undefined || boxHi === undefined
    ? extentOf(frames[0].atoms, axis)
    : null;
  const lo = boxLo ?? fallback!.lo;
  const hi = boxHi ?? fallback!.hi;
  const range = hi - lo || 1;
  const hist = new Array(bins).fill(0);
  let totalAtoms = 0;
  for (const frame of frames) {
    for (const a of frame.atoms) {
      const v = axis === 'x' ? a.x : axis === 'y' ? a.y : a.z;
      const bin = Math.floor(((v - lo) / range) * bins);
      const b = Math.max(0, Math.min(bins - 1, bin));
      hist[b]++;
      totalAtoms++;
    }
  }
  const binWidth = range / bins;
  // Density = count / (binVolume) ; for profile we just show count density per bin
  const maxCount = maxOf(hist);
  const resultBins: HistogramBin[] = hist.map((count, i) => ({
    x0: lo + i * binWidth,
    x1: lo + (i + 1) * binWidth,
    count,
    density: count / maxCount, // normalized 0-1 for display
  }));
  return { bins: resultBins, axis };
};

/** Velocity speed distribution (if vx/vy/vz present) */
export const computeSpeedDistribution = (
  atoms: Atom[],
  bins = 30,
): HistogramBin[] | null => {
  // Build into a typed array and track the extent in the same pass: a 60k-atom
  // frame is already close to the engine's argument limit for Math.min(...).
  const speeds = new Float64Array(atoms.length);
  let n = 0;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < atoms.length; i++) {
    const a = atoms[i];
    let v: number | undefined;
    if (a.vx !== undefined && a.vy !== undefined && a.vz !== undefined) {
      v = Math.sqrt(a.vx * a.vx + a.vy * a.vy + a.vz * a.vz);
    } else if (a.vx !== undefined || a.vy !== undefined) {
      const vx = a.vx ?? 0;
      const vy = a.vy ?? 0;
      v = Math.sqrt(vx * vx + vy * vy);
    }
    if (v === undefined || !Number.isFinite(v)) continue;
    speeds[n++] = v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (n === 0) return null;
  const range = max - min || 1;
  const hist = new Array(bins).fill(0);
  for (let i = 0; i < n; i++) {
    const bin = Math.floor(((speeds[i] - min) / range) * bins);
    const b = Math.max(0, Math.min(bins - 1, bin));
    hist[b]++;
  }
  const maxCount = maxOf(hist);
  const binWidth = range / bins;
  return hist.map((count, i) => ({
    x0: min + i * binWidth,
    x1: min + (i + 1) * binWidth,
    count,
    density: count / maxCount,
  }));
};

/** Simple stats for the trajectory */
export const trajStats = (data: MoleculeData) => {
  const frames = data.frames ?? [];
  const N = data.atoms.length;
  const frameCount = frames.length || 1;
  const box = data.box;
  let density: number | undefined;
  if (box) {
    const V = (box.xhi - box.xlo) * (box.yhi - box.ylo) * (box.zhi - box.zlo);
    if (V > 0) density = N / V;
  }
  // For 2D, area density
  let areaDensity: number | undefined;
  if (box && (box.zhi - box.zlo) < 2) {
    const A = (box.xhi - box.xlo) * (box.yhi - box.ylo);
    if (A > 0) areaDensity = N / A;
  }
  return {
    atoms: N,
    frames: frameCount,
    box,
    density,
    areaDensity,
    bonds: data.bonds.length,
    hasVelocities: data.atoms.some(a => a.vx !== undefined),
  };
};
