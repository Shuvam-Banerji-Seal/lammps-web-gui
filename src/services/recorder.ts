/**
 * High-quality canvas → MP4 (or WebM fallback) recording via MediaRecorder.
 *
 * Codec strategy: prefer H.264-in-MP4 for the requested "high quality mp4"
 * output; fall back to WebM VP9/VP8 on browsers without MP4 muxing
 * (e.g. Firefox). Bitrate defaults target visually lossless 1080p-class
 * footage (~24 Mbps at 60 fps).
 */

export interface RecordingFormat {
  mime: string;
  ext: 'mp4' | 'webm';
  label: string;
}

const CANDIDATES: RecordingFormat[] = [
  { mime: 'video/mp4;codecs=avc1.640028', ext: 'mp4', label: 'MP4 · H.264 High' },
  { mime: 'video/mp4;codecs=avc1.42E01E', ext: 'mp4', label: 'MP4 · H.264' },
  { mime: 'video/mp4', ext: 'mp4', label: 'MP4' },
  { mime: 'video/webm;codecs=vp9', ext: 'webm', label: 'WebM · VP9' },
  { mime: 'video/webm;codecs=vp8', ext: 'webm', label: 'WebM · VP8' },
  { mime: 'video/webm', ext: 'webm', label: 'WebM' },
];

/** Every candidate the injected `isSupported` predicate accepts, best first (the Render panel's format list). */
export const recordingFormats = (
  isSupported: (mime: string) => boolean
): RecordingFormat[] =>
  CANDIDATES.filter(c => {
    try {
      return isSupported(c.mime);
    } catch {
      return false;
    }
  });

/** Pure negotiation over an injected `isSupported` predicate — unit-testable. */
export const negotiateRecordingFormat = (
  isSupported: (mime: string) => boolean
): RecordingFormat | null =>
  CANDIDATES.find(c => {
    try {
      return isSupported(c.mime);
    } catch {
      return false;
    }
  }) ?? null;

export interface CanvasRecordingOptions {
  fps?: number;
  /** Target bitrate in bits/second. Default ~24 Mbps ("high quality"). */
  videoBitsPerSecond?: number;
  /** Use this format instead of negotiating one (the caller checked it is supported). */
  format?: RecordingFormat;
  /**
   * Capture only when `capture()` is called (one frame per call), so every drawn frame is recorded once and
   * none is repeated. Without it the canvas is sampled at `fps`.
   */
  manualFrames?: boolean;
}

export interface RecordingResult {
  blob: Blob;
  ext: 'mp4' | 'webm';
  label: string;
  durationMs: number;
}

export interface RecordingHandle {
  format: RecordingFormat;
  stop: () => Promise<RecordingResult>;
  /** With `manualFrames`: record the canvas's current content as one frame. */
  capture?: () => void;
}

/**
 * Start capturing a canvas. The caller keeps rendering frames (the app
 * forces continuous rendering while a recording is active).
 */
export const startCanvasRecording = (
  canvas: HTMLCanvasElement,
  options: CanvasRecordingOptions = {}
): RecordingHandle => {
  if (typeof MediaRecorder === 'undefined') {
    throw new Error('MediaRecorder is not supported in this browser.');
  }
  const format = options.format ?? negotiateRecordingFormat(
    m => MediaRecorder.isTypeSupported(m)
  );
  if (!format) throw new Error('No supported video codec found.');

  const fps = options.fps ?? 60;
  const manual = options.manualFrames === true;
  const stream = canvas.captureStream(manual ? 0 : fps);
  const track = manual
    ? (stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack | undefined)
    : undefined;
  let recorder: MediaRecorder;
  try {
    recorder = new MediaRecorder(stream, {
      mimeType: format.mime,
      videoBitsPerSecond: options.videoBitsPerSecond ?? 24_000_000,
    });
  } catch (e) {
    // the constructor can still reject the stream: release its live tracks
    stream.getTracks().forEach(t => t.stop());
    throw e;
  }

  const chunks: BlobPart[] = [];
  let finished: ((r: RecordingResult) => void) | null = null;
  let startedAt = 0;

  recorder.ondataavailable = e => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const stopped = new Promise<RecordingResult>(resolve => {
    finished = resolve;
  });

  recorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    finished?.({
      blob: new Blob(chunks, { type: format.mime }),
      ext: format.ext,
      label: format.label,
      durationMs: performance.now() - startedAt,
    });
  };

  recorder.start(250); // gather chunks every 250 ms for smooth long captures
  startedAt = performance.now();

  return {
    format,
    capture: manual ? () => track?.requestFrame() : undefined,
    stop: () => {
      return new Promise<RecordingResult>(resolve => {
        if (recorder.state === 'inactive') {
          resolve({
            blob: new Blob(chunks, { type: format.mime }),
            ext: format.ext,
            label: format.label,
            durationMs: performance.now() - startedAt,
          });
          return;
        }
        stopped.then(resolve);
        // flush final chunk then stop
        recorder.requestData?.();
        recorder.stop();
      });
    },
  };
};
