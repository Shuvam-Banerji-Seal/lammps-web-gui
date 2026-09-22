import {
  computeRDF,
  computeMSD,
  computeDensityProfile,
  hasImageFlags,
} from '../services/trajectoryAnalysis';
import type {
  AnalysisRequest,
  AnalysisResponse,
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
 * Protocol: AnalysisRequest -> AnalysisResponse, routed by `id`.
 */
self.onmessage = (e: MessageEvent<AnalysisRequest>) => {
  const { id, frames, box, opts } = e.data;
  const post = (r: AnalysisResponse) => (self as unknown as Worker).postMessage(r);
  const started = Date.now();
  try {
    post({
      id,
      ok: true,
      rdf: computeRDF(frames, box, { rMax: opts.rdfRMax, bins: opts.rdfBins }),
      msd: computeMSD(frames, box, { timeOriginStride: opts.msdStride }),
      density: computeDensityProfile(frames, box, opts.densityAxis, opts.densityBins),
      msdUnwrapped: !!box && frames.length > 0 && hasImageFlags(frames[0].atoms),
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
