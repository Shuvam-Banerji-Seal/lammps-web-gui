import {
  computeNeighbourAnalysis,
  computeRDF,
  computeDensityProfile,
  msdIsExact,
  msdInRealFrames,
} from '../services/trajectoryAnalysis';
import type {
  AnalysisRequest,
  AnalysisResponse,
  NeighbourRequest,
} from '../services/analysisClient';

/**
 * Off-main-thread trajectory analysis.
 *
 * RDF, MSD and the density profile are O(frames · N) or worse. They used to
 * run inline in the Analysis panel's render, so every slider nudge, theme
 * toggle and playback tick recomputed all of them — up to 30x/second with a
 * trajectory playing. Now the panel asks once per loaded structure and the
 * main thread stays free for rendering.
 *
 * Protocol: AnalysisRequest (or NeighbourRequest, for coordination and bond angles) -> AnalysisResponse, routed by `id`.
 */
self.onmessage = (e: MessageEvent<AnalysisRequest | NeighbourRequest>) => {
  // A dedicated worker hears only the page that created it, and its message events carry an empty origin; anything
  // else (a foreign origin) is not a request of this app.
  if (e.origin !== '' && e.origin !== self.location.origin) return;
  const post = (r: AnalysisResponse) => (self as unknown as Worker).postMessage(r);
  const started = Date.now();
  const req = e.data;
  if ('kind' in req) {
    try {
      post({ id: req.id, ok: true, neighbours: computeNeighbourAnalysis(req.frame, req.cutoff, req.nbins), ms: Date.now() - started });
    } catch (err) {
      post({ id: req.id, ok: false, error: err instanceof Error ? err.message : String(err), ms: Date.now() - started });
    }
    return;
  }
  const { id, frames, box, opts, msdFrames, msdFrameStride } = req;
  const msdSample = { frames: msdFrames ?? frames, stride: msdFrameStride ?? 1 };
  try {
    post({
      id,
      ok: true,
      rdf: computeRDF(frames, box, { rMax: opts.rdfRMax, bins: opts.rdfBins }),
      msd: msdInRealFrames(msdSample, box, opts.msdStride),
      density: computeDensityProfile(frames, box, opts.densityAxis, opts.densityBins),
      msdUnwrapped: msdIsExact(msdSample.frames, box),
      ms: Date.now() - started,
    });
  } catch (err) {
    post({
      id,
      ok: false,
      error: err instanceof Error ? err.message : String(err),
      ms: Date.now() - started,
    });
  }
};
