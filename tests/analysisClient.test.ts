import { describe, it, expect } from 'vitest';
import {
  analyzeSync, analyzeTrajectory, inFlightAnalyses, AnalysisOptions,
} from '../src/services/analysisClient';
import { msdInRealFrames } from '../src/services/trajectoryAnalysis';
import { uniformFrames } from '../src/hooks/useTrajectoryAnalysis';
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

describe('MSD frame sampling — uniform stride, lags in REAL frames', () => {
  // atom drifting +0.1 per REAL frame, 51 frames, no box
  const drift: TrajectoryFrame[] = Array.from({ length: 51 }, (_, t) => ({
    atoms: [atom(1, 0.1 * t, 0, 0)],
  }));

  it('uniformFrames uses an integer stride, never floor(i * 3.4)', () => {
    const { frames, stride } = uniformFrames(drift, 1);
    expect(Number.isInteger(stride)).toBe(true);
    // consecutive picks are exactly `stride` real frames apart
    const idx = frames.map(f => Math.round(f.atoms[0].x / 0.1));
    for (let i = 1; i < idx.length; i++) expect(idx[i] - idx[i - 1]).toBe(stride);
  });

  it('caps atom·frames so a huge trajectory is not cloned wholesale', () => {
    const big = Array.from({ length: 5000 }, () => ({ atoms: [] as Atom[] }));
    const { frames, stride } = uniformFrames(big, 60_000);
    expect(frames.length * 60_000).toBeLessThanOrEqual(2_000_000 + 60_000);
    expect(stride).toBeGreaterThan(1);
  });

  it('reports lags in real frames, so the slope is per REAL frame', () => {
    // Force a stride of 4 and check lag k*4 carries (0.1 * 4k)^2.
    const sample = { frames: drift.filter((_, i) => i % 4 === 0), stride: 4 };
    const pts = msdInRealFrames(sample, undefined, 1);
    for (const p of pts) {
      expect(p.t % 4).toBe(0);
      expect(p.msd).toBeCloseTo((0.1 * p.t) ** 2, 10);
    }
  });

  it('REGRESSION: non-uniform sampling gave inconsistent lags', () => {
    // The old floor(i*3.4) sampling of 51 frames: gaps alternate 3 and 4.
    const idx = Array.from({ length: 15 }, (_, i) => Math.floor(i * (51 / 15)));
    const gaps = new Set(idx.slice(1).map((v, i) => v - idx[i]));
    expect(gaps.size).toBeGreaterThan(1);    // why it was wrong
    const { stride } = uniformFrames(drift, 1);
    expect(stride).toBe(1);                   // 51 frames fit: use them all
  });

  it('analyzeSync scales MSD lags by the sample stride end to end', () => {
    const sample = { frames: drift.filter((_, i) => i % 5 === 0), stride: 5 };
    const r = analyzeSync(drift.slice(0, 3), undefined, opts, sample);
    const last = r.msd[r.msd.length - 1];
    expect(last.t).toBe(50);
    expect(last.msd).toBeCloseTo(25, 8);      // (0.1 * 50)^2
  });
});
