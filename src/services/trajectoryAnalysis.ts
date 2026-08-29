import { Atom, BoxBounds, MoleculeData, TrajectoryFrame } from '../types';

/**
 * Trajectory analysis service — pure functions for RDF, MSD, density, etc.
 * All functions are unit-testable and have no three.js dependency.
 * For large systems, callers should sample frames (e.g., every 10th) or
 * run in a Web Worker (the viewer already parses in a worker).
 */

export interface RDFPoint { r: number; g: number; count: number; }
export interface MSDPoint { t: number; msd: number; }
export interface HistogramBin { x0: number; x1: number; count: number; density: number; }
export interface DensityProfile { bins: HistogramBin[]; axis: 'x' | 'y' | 'z'; }

/** Minimum image convention for PBC */
const pbcDelta = (d: number, boxLen: number): number => {
  if (boxLen <= 0) return d;
  // bring into [-L/2, L/2]
  return d - Math.round(d / boxLen) * boxLen;
};

const distPBC = (
  a: Atom, b: Atom, box?: BoxBounds,
): number => {
  let dx = a.x - b.x;
  let dy = a.y - b.y;
  let dz = a.z - b.z;
  if (box) {
    const lx = box.xhi - box.xlo;
    const ly = box.yhi - box.ylo;
    const lz = box.zhi - box.zlo;
    // For 2D thin box, don't PBC in z if lz < 2 (our 2D sims have lz=1)
    dx = pbcDelta(dx, lx);
    dy = pbcDelta(dy, ly);
    if (lz > 2) dz = pbcDelta(dz, lz);
  }
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

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
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        const r = distPBC(atoms[i], atoms[j], box);
        if (r < rMax && r >= 0) {
          const bin = Math.floor(r / dr);
          if (bin >= 0 && bin < bins) hist[bin] += 2; // count both i->j and j->i for normalization
        }
      }
    }
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

  // Build id -> index map for first frame (assume stable ids)
  const idToIdx = new Map<number, number>();
  frames[0].atoms.forEach((a, idx) => idToIdx.set(a.id, idx));

  const msd: MSDPoint[] = [];
  // For each lag dt, average over time origins
  for (let dt = 0; dt < frames.length; dt++) {
    let sum = 0;
    let count = 0;
    for (let t0 = 0; t0 + dt < frames.length; t0 += stride) {
      const f0 = frames[t0];
      const f1 = frames[t0 + dt];
      if (f0.atoms.length !== N || f1.atoms.length !== N) continue;
      // Build pos maps for quick lookup by id (in case order changes, though dump is sorted)
      const pos1 = new Map<number, Atom>();
      f1.atoms.forEach(a => pos1.set(a.id, a));
      for (const a0 of f0.atoms) {
        const a1 = pos1.get(a0.id);
        if (!a1) continue;
        let dx = a1.x - a0.x;
        let dy = a1.y - a0.y;
        let dz = a1.z - a0.z;
        if (box) {
          const lx = box.xhi - box.xlo;
          const ly = box.yhi - box.ylo;
          const lz = box.zhi - box.zlo;
          dx = pbcDelta(dx, lx);
          dy = pbcDelta(dy, ly);
          if (lz > 2) dz = pbcDelta(dz, lz);
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

/** Density profile along an axis (histogram) */
export const computeDensityProfile = (
  frames: TrajectoryFrame[],
  box: BoxBounds | undefined,
  axis: 'x' | 'y' | 'z' = 'y',
  bins = 30,
): DensityProfile => {
  if (frames.length === 0) return { bins: [], axis };
  const lo = axis === 'x' ? box?.xlo ?? Math.min(...frames[0].atoms.map(a => a.x))
    : axis === 'y' ? box?.ylo ?? Math.min(...frames[0].atoms.map(a => a.y))
    : box?.zlo ?? Math.min(...frames[0].atoms.map(a => a.z));
  const hi = axis === 'x' ? box?.xhi ?? Math.max(...frames[0].atoms.map(a => a.x))
    : axis === 'y' ? box?.yhi ?? Math.max(...frames[0].atoms.map(a => a.y))
    : box?.zhi ?? Math.max(...frames[0].atoms.map(a => a.z));
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
  const maxCount = Math.max(...hist, 1);
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
  const speeds = atoms
    .map(a => (a.vx !== undefined && a.vy !== undefined && a.vz !== undefined
      ? Math.sqrt(a.vx * a.vx + a.vy * a.vy + a.vz * a.vz)
      : a.vx !== undefined || a.vy !== undefined ? Math.sqrt((a.vx ?? 0) ** 2 + (a.vy ?? 0) ** 2)
      : undefined))
    .filter((v): v is number => v !== undefined && Number.isFinite(v));
  if (speeds.length === 0) return null;
  const min = Math.min(...speeds);
  const max = Math.max(...speeds);
  const range = max - min || 1;
  const hist = new Array(bins).fill(0);
  for (const v of speeds) {
    const bin = Math.floor(((v - min) / range) * bins);
    const b = Math.max(0, Math.min(bins - 1, bin));
    hist[b]++;
  }
  const maxCount = Math.max(...hist, 1);
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
