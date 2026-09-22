import { useEffect, useMemo, useRef, useState } from 'react';
import type { MoleculeData } from '../types';
import {
  AnalysisOptions,
  AnalysisResult,
  analyzeTrajectory,
} from '../services/analysisClient';
import {
  HistogramBin,
  computeSpeedDistribution,
} from '../services/trajectoryAnalysis';

/**
 * Frames sampled for the frame-averaged analyses. RDF and the density profile
 * converge quickly, so averaging ~15 evenly spaced frames looks the same as
 * averaging 500 and costs 30x less.
 */
const SAMPLE_FRAMES = 15;

export type AnalysisStatus = 'idle' | 'running' | 'ready' | 'error';

export interface TrajectoryAnalysis {
  status: AnalysisStatus;
  result: AnalysisResult | null;
  error: string | null;
  /** Frames actually averaged (for the "averaged over N frames" caption). */
  sampledFrames: number;
  /** Speed histogram for the CURRENT frame — cheap, so it stays on this thread. */
  speeds: HistogramBin[] | null;
}

const sampleFrames = <T,>(frames: T[], want: number): T[] => {
  if (frames.length <= want) return frames;
  const step = frames.length / want;
  const out: T[] = [];
  for (let i = 0; i < want; i++) out.push(frames[Math.floor(i * step)]);
  return out;
};

/**
 * Run the trajectory analyses off the main thread, once per structure.
 *
 * Previously the Analysis panel called computeRDF/computeMSD/
 * computeDensityProfile inline in its JSX, so they re-ran on every React
 * render — including every playback tick. This hook runs them once per
 * loaded structure, in a worker, and discards results that arrive after the
 * structure has been replaced.
 */
export const useTrajectoryAnalysis = (
  data: MoleculeData | null,
  frameIdx: number,
): TrajectoryAnalysis => {
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);

  const frames = data?.frames;
  const hasTrajectory = !!frames && frames.length > 1;

  const sampled = useMemo(
    () => (hasTrajectory ? sampleFrames(frames!, SAMPLE_FRAMES) : []),
    [frames, hasTrajectory],
  );

  const opts = useMemo<AnalysisOptions>(() => {
    const box = data?.box;
    const thinZ = box ? box.zhi - box.zlo < 2 : false;
    return {
      rdfRMax: 10,
      rdfBins: 80,
      // Cap the number of time origins so MSD stays linear in frame count.
      msdStride: Math.max(1, Math.floor((frames?.length ?? 1) / SAMPLE_FRAMES)),
      // A 2D slab has no meaningful y-profile; profile along x instead.
      densityAxis: thinZ ? 'x' : 'y',
      densityBins: 24,
    };
  }, [data?.box, frames?.length]);

  useEffect(() => {
    if (!hasTrajectory) {
      setStatus('idle');
      setResult(null);
      setError(null);
      return;
    }
    const gen = ++generation.current;
    setStatus('running');
    setError(null);
    let cancelled = false;

    analyzeTrajectory(sampled, data?.box, opts)
      .then(r => {
        // Ignore a result for a structure the user has already replaced.
        if (cancelled || gen !== generation.current) return;
        setResult(r);
        setStatus('ready');
      })
      .catch((e: unknown) => {
        if (cancelled || gen !== generation.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setStatus('error');
      });

    return () => { cancelled = true; };
  }, [hasTrajectory, sampled, data?.box, opts]);

  // Single-frame, O(N): no worker round-trip needed.
  const speeds = useMemo(() => {
    const atoms = frames ? frames[frameIdx]?.atoms ?? data?.atoms : data?.atoms;
    if (!atoms || atoms.length === 0) return null;
    return computeSpeedDistribution(atoms, 24);
  }, [frames, frameIdx, data?.atoms]);

  return {
    status,
    result,
    error,
    sampledFrames: sampled.length,
    speeds,
  };
};
