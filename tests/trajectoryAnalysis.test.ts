import { describe, it, expect } from 'vitest';
import {
  computeRDF, computeMSD, computeDensityProfile, computeSpeedDistribution,
  trajStats, distancePBC, hasImageFlags,
} from '../src/services/trajectoryAnalysis';
import type { Atom, BoxBounds, TrajectoryFrame, MoleculeData } from '../src/types';

const atom = (id: number, x: number, y: number, z: number, v?: [number, number, number]): Atom => ({
  id, molId: 1, type: 1, charge: 0, x, y, z,
  ...(v ? { vx: v[0], vy: v[1], vz: v[2] } : {}),
});

/** Deterministic LCG so the "random" gas is reproducible across runs. */
const lcg = (seed: number) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

const gasFrame = (n: number, L: number, seed = 7): TrajectoryFrame => {
  const rnd = lcg(seed);
  const atoms: Atom[] = [];
  for (let i = 0; i < n; i++) atoms.push(atom(i + 1, rnd() * L, rnd() * L, rnd() * L));
  return { atoms };
};

const cube = (L: number): BoxBounds => ({ xlo: 0, xhi: L, ylo: 0, yhi: L, zlo: 0, zhi: L });

/**
 * Brute-force reference for RDFPoint.count — the per-frame ordered-pair count
 * per bin. The accelerated cell-list version must agree with it bin for bin.
 */
const rdfReference = (
  frames: TrajectoryFrame[], box: BoxBounds | undefined, rMax: number, bins: number,
) => {
  const dr = rMax / bins;
  const hist = new Array(bins).fill(0);
  for (const f of frames) {
    for (let i = 0; i < f.atoms.length; i++) {
      for (let j = i + 1; j < f.atoms.length; j++) {
        const r = distancePBC(f.atoms[i], f.atoms[j], box);
        if (r < rMax) {
          const b = Math.floor(r / dr);
          if (b >= 0 && b < bins) hist[b] += 2;
        }
      }
    }
  }
  return hist.map(h => h / frames.length);
};

describe('RDF — cell list matches brute force', () => {
  it('agrees bin-for-bin on a periodic 3D gas', () => {
    const L = 20;
    const frames = [gasFrame(400, L, 11)];
    const box = cube(L);
    const rMax = 5;
    const bins = 50;
    const got = computeRDF(frames, box, { rMax, bins });
    const ref = rdfReference(frames, box, rMax, bins);
    expect(got).toHaveLength(bins);
    got.forEach((p, i) => expect(p.count).toBeCloseTo(ref[i], 9));
  });

  it('agrees when the box is only ~2 cells wide (the wrap-around edge case)', () => {
    // nx = floor(L/rMax) = 2 -> a naive -1/0/+1 periodic scan double-counts.
    const L = 11;
    const frames = [gasFrame(200, L, 23)];
    const box = cube(L);
    const rMax = 5;
    const got = computeRDF(frames, box, { rMax, bins: 40 });
    const ref = rdfReference(frames, box, rMax, 40);
    got.forEach((p, i) => expect(p.count).toBeCloseTo(ref[i], 9));
  });

  it('agrees when the box is 1 cell wide', () => {
    const L = 6;
    const frames = [gasFrame(120, L, 31)];
    const box = cube(L);
    // computeRDF clamps rMax to 0.95 * half-min-side, so read it back from ref
    const rMax = Math.min(5, (L / 2) * 0.95);
    const got = computeRDF(frames, box, { rMax: 5, bins: 30 });
    const ref = rdfReference(frames, box, rMax, 30);
    got.forEach((p, i) => expect(p.count).toBeCloseTo(ref[i], 9));
  });

  it('agrees for a non-periodic (boxless) structure', () => {
    const frames = [gasFrame(250, 15, 41)];
    const got = computeRDF(frames, undefined, { rMax: 4, bins: 40 });
    const ref = rdfReference(frames, undefined, 4, 40);
    got.forEach((p, i) => expect(p.count).toBeCloseTo(ref[i], 9));
  });

  it('agrees for a thin-z 2D slab, which must not wrap in z', () => {
    const rnd = lcg(53);
    const atoms: Atom[] = [];
    for (let i = 0; i < 300; i++) atoms.push(atom(i + 1, rnd() * 20, rnd() * 20, rnd() * 0.4 - 0.2));
    const box: BoxBounds = { xlo: 0, xhi: 20, ylo: 0, yhi: 20, zlo: -0.5, zhi: 0.5 };
    const got = computeRDF([{ atoms }], box, { rMax: 5, bins: 40 });
    const ref = rdfReference([{ atoms }], box, 5, 40);
    got.forEach((p, i) => expect(p.count).toBeCloseTo(ref[i], 9));
  });
});

describe('RDF — physics', () => {
  it('g(r) of a random gas is ~1 beyond the first bins', () => {
    const L = 24;
    const frames = [gasFrame(1500, L, 3), gasFrame(1500, L, 5), gasFrame(1500, L, 9)];
    const pts = computeRDF(frames, cube(L), { rMax: 8, bins: 32 });
    const tail = pts.filter(p => p.r > 2);
    const mean = tail.reduce((a, p) => a + p.g, 0) / tail.length;
    expect(mean).toBeGreaterThan(0.85);
    expect(mean).toBeLessThan(1.15);
  });

  it('a simple cubic lattice puts a sharp peak at the lattice spacing', () => {
    const a = 2;
    const nCell = 8;
    const atoms: Atom[] = [];
    let id = 1;
    for (let i = 0; i < nCell; i++)
      for (let j = 0; j < nCell; j++)
        for (let k = 0; k < nCell; k++) atoms.push(atom(id++, i * a, j * a, k * a));
    const L = nCell * a;
    const pts = computeRDF([{ atoms }], cube(L), { rMax: 6, bins: 120 });
    // g(r) divides by 4*pi*r^2, so the 6-neighbour first shell and the
    // 12-neighbour second shell of a simple cubic lattice are equally tall.
    // The physically meaningful assertion is where the FIRST shell sits.
    const firstShell = pts.find(p => p.g > 1);
    expect(firstShell).toBeDefined();
    expect(firstShell!.r).toBeCloseTo(a, 1);
    expect(firstShell!.g).toBeGreaterThan(5);
    // nothing at all between the origin and that first shell
    expect(pts.filter(p => p.r > 0.3 && p.r < a - 0.3).every(p => p.g < 1e-9)).toBe(true);
    // the second shell lands at a*sqrt(2)
    const shells = pts.filter(p => p.g > 1).map(p => p.r);
    expect(shells[1]).toBeCloseTo(a * Math.SQRT2, 1);
  });

  it('returns nothing for degenerate input', () => {
    expect(computeRDF([], cube(10))).toEqual([]);
    expect(computeRDF([{ atoms: [] }], cube(10))).toEqual([]);
    expect(computeRDF([{ atoms: [atom(1, 0, 0, 0)] }], cube(10))).toEqual([]);
  });
});

describe('MSD', () => {
  it('is exactly (v·t)² for uniform drift with no box', () => {
    const v = 0.25;
    const frames: TrajectoryFrame[] = Array.from({ length: 10 }, (_, t) => ({
      atoms: [atom(1, v * t, 0, 0), atom(2, 10 + v * t, 0, 0)],
    }));
    const pts = computeMSD(frames, undefined, { timeOriginStride: 1 });
    for (const p of pts) expect(p.msd).toBeCloseTo((v * p.t) ** 2, 10);
  });

  it('is zero for a frozen structure', () => {
    const f = { atoms: [atom(1, 1, 2, 3), atom(2, 4, 5, 6)] };
    const pts = computeMSD([f, f, f, f], cube(20), { timeOriginStride: 1 });
    expect(pts.every(p => p.msd === 0)).toBe(true);
  });

  it('follows atom IDs when a later frame reorders them', () => {
    // frame 1 lists the same two atoms in the opposite order; MSD must track
    // ids, not array positions, or it reports a bogus jump.
    const frames: TrajectoryFrame[] = [
      { atoms: [atom(1, 0, 0, 0), atom(2, 5, 0, 0)] },
      { atoms: [atom(2, 5, 0, 0), atom(1, 0, 0, 0)] },
    ];
    const pts = computeMSD(frames, undefined, { timeOriginStride: 1 });
    expect(pts.find(p => p.t === 1)!.msd).toBeCloseTo(0, 10);
  });

  it('applies the minimum image convention across a periodic wrap', () => {
    // one atom steps from x=9.5 to x=0.5 in a box of length 10: that is a
    // +1.0 step through the boundary, not a -9.0 jump.
    const frames: TrajectoryFrame[] = [
      { atoms: [atom(1, 9.5, 0, 0)] },
      { atoms: [atom(1, 0.5, 0, 0)] },
    ];
    const pts = computeMSD(frames, cube(10), { timeOriginStride: 1 });
    expect(pts.find(p => p.t === 1)!.msd).toBeCloseTo(1, 10);
  });

  it('needs at least two frames', () => {
    expect(computeMSD([{ atoms: [atom(1, 0, 0, 0)] }])).toEqual([]);
    expect(computeMSD([])).toEqual([]);
  });
});

describe('density profile and speed distribution', () => {
  it('a uniform slab gives a flat profile', () => {
    const frames = [gasFrame(2000, 20, 17)];
    const prof = computeDensityProfile(frames, cube(20), 'y', 10);
    expect(prof.bins).toHaveLength(10);
    const counts = prof.bins.map(b => b.count);
    const mean = counts.reduce((a, c) => a + c, 0) / counts.length;
    for (const c of counts) expect(Math.abs(c - mean) / mean).toBeLessThan(0.35);
  });

  it('falls back to the atom extent when there is no box', () => {
    const frames = [{ atoms: [atom(1, 0, -3, 0), atom(2, 0, 7, 0)] }];
    const prof = computeDensityProfile(frames, undefined, 'y', 4);
    expect(prof.bins[0].x0).toBeCloseTo(-3, 6);
    expect(prof.bins[3].x1).toBeCloseTo(7, 6);
  });

  it('bins speeds and reports null without velocities', () => {
    const bins = computeSpeedDistribution(
      [atom(1, 0, 0, 0, [1, 0, 0]), atom(2, 0, 0, 0, [0, 2, 0]), atom(3, 0, 0, 0, [0, 0, 3])],
      3,
    );
    expect(bins).not.toBeNull();
    expect(bins!.reduce((a, b) => a + b.count, 0)).toBe(3);
    expect(computeSpeedDistribution([atom(1, 0, 0, 0)], 3)).toBeNull();
  });

  it('survives a frame far past the argument-spread limit', () => {
    // Math.min(...arr) throws RangeError around this size — the old
    // implementation did exactly that.
    const n = 200_000;
    const atoms: Atom[] = new Array(n);
    for (let i = 0; i < n; i++) atoms[i] = atom(i + 1, i % 97, i % 89, i % 83, [i % 7, 1, 2]);
    expect(() => computeSpeedDistribution(atoms, 16)).not.toThrow();
    expect(() => computeDensityProfile([{ atoms }], undefined, 'x', 16)).not.toThrow();
  });
});

describe('trajStats', () => {
  it('reports counts, density and velocity presence', () => {
    const data: MoleculeData = {
      atoms: [atom(1, 0, 0, 0, [1, 0, 0]), atom(2, 1, 0, 0)],
      bonds: [], atomTypes: {},
      min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 0, z: 0 }, center: { x: 0.5, y: 0, z: 0 },
      box: cube(10),
      frames: [{ atoms: [] }, { atoms: [] }],
    };
    const st = trajStats(data);
    expect(st.atoms).toBe(2);
    expect(st.frames).toBe(2);
    expect(st.density).toBeCloseTo(2 / 1000, 9);
    expect(st.hasVelocities).toBe(true);
  });
});

describe('MSD with LAMMPS image flags', () => {
  const L = 10;
  const withImage = (id: number, x: number, ix: number): Atom =>
    ({ ...atom(id, x, 0, 0), ix, iy: 0, iz: 0 });

  it('reports image flags when the frame carries them', () => {
    expect(hasImageFlags([withImage(1, 0, 0)])).toBe(true);
    expect(hasImageFlags([atom(1, 0, 0, 0)])).toBe(false);
    expect(hasImageFlags([])).toBe(false);
  });

  it('unwraps exactly across many box crossings', () => {
    // One atom drifting +2 per frame through a box of length 10, wrapped, with
    // honest image flags. True displacement after 10 frames is 20 = 2 boxes.
    const frames: TrajectoryFrame[] = [];
    for (let t = 0; t <= 10; t++) {
      const trueX = 2 * t;
      frames.push({ atoms: [withImage(1, trueX % L, Math.floor(trueX / L))] });
    }
    const pts = computeMSD(frames, cube(L), { timeOriginStride: 1 });
    for (const p of pts) expect(p.msd).toBeCloseTo((2 * p.t) ** 2, 6);
    // and the last lag exceeds (L/2)^2 = 25, which minimum image cannot reach
    expect(pts[pts.length - 1].msd).toBeCloseTo(400, 6);
  });

  it('WITHOUT image flags the same motion saturates below (L/2)²', () => {
    const frames: TrajectoryFrame[] = [];
    for (let t = 0; t <= 10; t++) frames.push({ atoms: [atom(1, (2 * t) % L, 0, 0)] });
    const pts = computeMSD(frames, cube(L), { timeOriginStride: 1 });
    // minimum image caps every displacement at L/2, so MSD can never exceed 25
    expect(Math.max(...pts.map(p => p.msd))).toBeLessThanOrEqual((L / 2) ** 2 + 1e-9);
    // which is exactly why the caption warns about it
    expect(pts[pts.length - 1].msd).toBeLessThan(400);
  });

  it('applies the triclinic tilt to image offsets', () => {
    const box: BoxBounds = { xlo: 0, xhi: 10, ylo: 0, yhi: 10, zlo: 0, zhi: 10, xy: 3, xz: 0, yz: 0 };
    // one y-image step carries an extra xy = 3 in x
    const frames: TrajectoryFrame[] = [
      { atoms: [{ ...atom(1, 0, 0, 0), ix: 0, iy: 0, iz: 0 }] },
      { atoms: [{ ...atom(1, 0, 0, 0), ix: 0, iy: 1, iz: 0 }] },
    ];
    const pts = computeMSD(frames, box, { timeOriginStride: 1 });
    // displacement = (xy, ly, 0) = (3, 10, 0) -> 9 + 100
    expect(pts.find(p => p.t === 1)!.msd).toBeCloseTo(109, 6);
  });
});
