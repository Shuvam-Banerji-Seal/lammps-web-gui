import type { BoxBounds, TrajectoryFrame } from '../types';
import type { DensityProfile, MSDPoint, NeighbourResult, RDFPoint } from './trajectoryAnalysis';
import {
  computeDensityProfile,
  computeNeighbourAnalysis,
  computeRDF,
  msdIsExact,
  msdInRealFrames,
} from './trajectoryAnalysis';
import type { MsdSample } from './trajectoryAnalysis';

/**
 * Client side of the trajectory-analysis worker. Same id-routing contract as
 * `parserClient`, so concurrent requests cannot cross wires and a stale
 * response for an unloaded structure is simply dropped.
 *
 * Falls back to computing on the calling thread when Workers are unavailable
 * (tests, locked-down embeds).
 */

export interface AnalysisOptions {
  rdfRMax: number;
  rdfBins: number;
  msdStride: number;
  densityAxis: 'x' | 'y' | 'z';
  densityBins: number;
}

export interface AnalysisRequest {
  id: number;
  /** Frames for RDF and density — any even spread will do (plain averages). */
  frames: TrajectoryFrame[];
  box?: BoxBounds;
  opts: AnalysisOptions;
  /** Frames for MSD — must be a UNIFORM stride of the trajectory. */
  msdFrames: TrajectoryFrame[];
  /** Real frames between consecutive `msdFrames`; MSD lags are scaled by it. */
  msdFrameStride: number;
}

export type { MsdSample } from './trajectoryAnalysis';

/** Coordination and bond angles of one frame at a cutoff (a second request kind, see NeighbourRequest). */
export interface NeighbourRequest {
  id: number;
  kind: 'neighbours';
  frame: TrajectoryFrame;
  cutoff: number;
  nbins: number;
}

export interface AnalysisResponse {
  id: number;
  ok: boolean;
  rdf?: RDFPoint[];
  msd?: MSDPoint[];
  density?: DensityProfile;
  neighbours?: NeighbourResult;
  /** True when MSD used image flags (exact) rather than minimum image. */
  msdUnwrapped?: boolean;
  /** Wall-clock milliseconds spent in the worker. */
  ms?: number;
  error?: string;
}

export interface AnalysisResult {
  rdf: RDFPoint[];
  msd: MSDPoint[];
  density: DensityProfile;
  /**
   * True when MSD unwrapped displacements with LAMMPS image flags, so it is
   * exact at long lag. False means minimum-image, which saturates near
   * (L/2)² — the caption tells the user which they are looking at.
   */
  msdUnwrapped: boolean;
  ms: number;
  /** True when the work ran on the calling thread instead of a worker. */
  onMainThread: boolean;
}

let worker: Worker | null = null;
let workerUnavailable = false;
let nextId = 1;
const pending = new Map<
  number,
  { resolve: (r: AnalysisResponse) => void; reject: (e: Error) => void }
>();

const failAll = (message: string) => {
  for (const [, entry] of pending) entry.reject(new Error(message));
  pending.clear();
};

const ensureWorker = (): Worker | null => {
  if (worker) return worker;
  if (workerUnavailable) return null;
  try {
    worker = new Worker(new URL('../workers/analysis.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (ev: MessageEvent<AnalysisResponse>) => {
      const entry = pending.get(ev.data.id);
      if (!entry) return; // superseded request — the caller moved on
      pending.delete(ev.data.id);
      if (ev.data.ok) entry.resolve(ev.data);
      else entry.reject(new Error(ev.data.error ?? 'Analysis failed'));
    };
    worker.onerror = () => {
      // Don't try to respawn: fall back to the main thread from here on.
      workerUnavailable = true;
      worker = null;
      failAll('worker-crashed');
    };
    return worker;
  } catch {
    workerUnavailable = true;
    worker = null;
    return null;
  }
};

/** Compute on the calling thread — the no-worker fallback, also used by tests. */
export const analyzeSync = (
  frames: TrajectoryFrame[],
  box: BoxBounds | undefined,
  opts: AnalysisOptions,
  msd: MsdSample = { frames, stride: 1 },
): AnalysisResult => {
  const started = Date.now();
  return {
    rdf: computeRDF(frames, box, { rMax: opts.rdfRMax, bins: opts.rdfBins }),
    msd: msdInRealFrames(msd, box, opts.msdStride),
    density: computeDensityProfile(frames, box, opts.densityAxis, opts.densityBins),
    msdUnwrapped: msdIsExact(msd.frames, box),
    ms: Date.now() - started,
    onMainThread: true,
  };
};

/** Run the trajectory analyses off the main thread where possible. */
export const analyzeTrajectory = (
  frames: TrajectoryFrame[],
  box: BoxBounds | undefined,
  opts: AnalysisOptions,
  msd: MsdSample = { frames, stride: 1 },
): Promise<AnalysisResult> => {
  const w = ensureWorker();
  if (!w) return Promise.resolve(analyzeSync(frames, box, opts, msd));
  return new Promise<AnalysisResult>((resolve, reject) => {
    const id = nextId++;
    pending.set(id, {
      resolve: (r) => {
        if (!r.rdf || !r.msd || !r.density) { reject(new Error('Analysis failed')); return; }
        resolve({
          rdf: r.rdf, msd: r.msd, density: r.density, msdUnwrapped: r.msdUnwrapped === true,
          ms: r.ms ?? 0, onMainThread: false,
        });
      },
      reject,
    });
    try {
      w.postMessage({
        id, frames, box, opts, msdFrames: msd.frames, msdFrameStride: msd.stride,
      } satisfies AnalysisRequest);
    } catch (err) {
      // Structured clone can fail on exotic inputs; degrade rather than hang.
      pending.delete(id);
      reject(err instanceof Error ? err : new Error('postMessage failed'));
    }
  });
};

/**
 * Number of requests still awaiting a worker response. Superseded requests
 * are not cancelled — clearing `pending` would leave their promises forever
 * unsettled — so callers discard stale results by request generation instead
 * (see `useTrajectoryAnalysis`).
 */
export const inFlightAnalyses = (): number => pending.size;

/**
 * Coordination numbers and the bond-angle distribution of one frame at `cutoff`, computed in the analysis worker
 * (O(N * neighbours)). Falls back to the calling thread when Workers are unavailable.
 */
export const analyzeNeighbours = (
  frame: TrajectoryFrame,
  cutoff: number,
  nbins = 90,
): Promise<NeighbourResult> => {
  const w = ensureWorker();
  if (!w) return Promise.resolve(computeNeighbourAnalysis(frame, cutoff, nbins));
  return new Promise<NeighbourResult>((resolve, reject) => {
    const id = nextId++;
    pending.set(id, {
      resolve: (r) => {
        if (!r.neighbours) { reject(new Error('Neighbour analysis failed')); return; }
        resolve(r.neighbours);
      },
      reject,
    });
    try {
      w.postMessage({ id, kind: 'neighbours', frame, cutoff, nbins } satisfies NeighbourRequest);
    } catch (err) {
      pending.delete(id);
      reject(err instanceof Error ? err : new Error('postMessage failed'));
    }
  });
};
