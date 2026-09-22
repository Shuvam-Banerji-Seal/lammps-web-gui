import { describe, it, expect } from 'vitest';
import {
  analyzeSync, analyzeTrajectory, inFlightAnalyses, AnalysisOptions,
} from '../src/services/analysisClient';
import type { Atom, BoxBounds, TrajectoryFrame } from '../src/types';

const atom = (id: number, x: number, y: number, z: number): Atom =>
  ({ id, molId: 1, type: 1, charge: 0, x, y, z });

const lcg = (seed: number) => () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const frame = (n: number, L: number, seed: number): TrajectoryFrame => {
  const r = lcg(seed);
  const atoms: Atom[] = [];
  for (let i = 0; i < n; i++) atoms.push(atom(i + 1, r() * L, r() * L, r() * L));
  return { atoms };
};
const box: BoxBounds = { xlo: 0, xhi: 15, ylo: 0, yhi: 15, zlo: 0, zhi: 15 };
const opts: AnalysisOptions = {
  rdfRMax: 5, rdfBins: 30, msdStride: 1, densityAxis: 'y', densityBins: 12,
};

describe('analysis client', () => {
  const frames = [frame(200, 15, 1), frame(200, 15, 2), frame(200, 15, 3)];

  it('analyzeSync returns all three analyses', () => {
    const r = analyzeSync(frames, box, opts);
    expect(r.rdf).toHaveLength(30);
    expect(r.density.bins).toHaveLength(12);
    expect(r.density.axis).toBe('y');
    expect(r.msd.length).toBeGreaterThan(0);
    expect(r.onMainThread).toBe(true);
    expect(r.ms).toBeGreaterThanOrEqual(0);
  });

  it('falls back to the main thread when Workers are unavailable', async () => {
    // jsdom provides no Worker constructor — the exact degraded path a
    // locked-down embed hits. It must still produce results, not reject.
    const r = await analyzeTrajectory(frames, box, opts);
    expect(r.onMainThread).toBe(true);
    expect(r.rdf).toHaveLength(30);
    expect(inFlightAnalyses()).toBe(0);
  });

  it('matches analyzeSync exactly through the async entry point', async () => {
    const direct = analyzeSync(frames, box, opts);
    const viaClient = await analyzeTrajectory(frames, box, opts);
    expect(viaClient.rdf.map(p => p.g)).toEqual(direct.rdf.map(p => p.g));
    expect(viaClient.msd.map(p => p.msd)).toEqual(direct.msd.map(p => p.msd));
  });

  it('handles an empty trajectory without throwing', () => {
    const r = analyzeSync([], undefined, opts);
    expect(r.rdf).toEqual([]);
    expect(r.msd).toEqual([]);
    expect(r.density.bins).toEqual([]);
  });
});
