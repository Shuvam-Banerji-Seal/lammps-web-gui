/**
 * Pure helpers for the Render panel: which recorded frames a render takes (range, stride, loops), the names of
 * exported PNGs, the step and time overlay text, the output size and bitrate presets and the zip of frames.
 * No DOM and no React, so they are unit-tested directly (tests/notebookRender.test.ts).
 */
import { buildZip, type ZipEntry } from '../../../lammps/zip';

export interface FrameRange {
  /** 1-based, inclusive, as the panel shows them. */
  start: number;
  end: number;
  stride: number;
}

export type FrameSelection =
  | { ok: true; indices: number[] }
  | { ok: false; error: string };

/**
 * The 0-based buffer indices a range takes: start, start + stride, ... up to end. Frame numbers are 1-based
 * (what the panel and the file names show). Rejects a range outside 1..count, start after end, or a stride < 1.
 */
export const selectFrames = (count: number, range: FrameRange): FrameSelection => {
  if (count <= 0) return { ok: false, error: 'No recorded frames to render yet.' };
  const { start, end, stride } = range;
  if (![start, end, stride].every((v) => Number.isInteger(v))) {
    return { ok: false, error: 'Start, end and stride must be whole numbers.' };
  }
  if (stride < 1) return { ok: false, error: 'Stride must be at least 1.' };
  if (start < 1 || end > count) return { ok: false, error: `Frames run from 1 to ${count}.` };
  if (start > end) return { ok: false, error: 'Start must not be after end.' };
  const indices: number[] = [];
  for (let i = start - 1; i <= end - 1; i += stride) indices.push(i);
  return { ok: true, indices };
};

/** The selection repeated `loops` times (a video that plays the range again and again). */
export const repeatIndices = (indices: readonly number[], loops: number): number[] => {
  const n = Math.max(1, Math.floor(loops));
  const out: number[] = [];
  for (let r = 0; r < n; r++) for (const i of indices) out.push(i);
  return out;
};

/** PNG name of a frame: its 1-based position in the recording and its step, e.g. frame_0001_step_100.png. */
export const frameFileName = (position: number, step: number): string =>
  `frame_${String(position).padStart(4, '0')}_step_${step}.png`;

/** Output size presets. `current` keeps the 3D view's own drawing size (null); the others are 16:9. */
export type ResolutionPreset = 'current' | '720p' | '1080p';

export const RESOLUTION_LABEL: Record<ResolutionPreset, string> = {
  current: 'Current view',
  '720p': '720p (1280 × 720)',
  '1080p': '1080p (1920 × 1080)',
};

export const RESOLUTION_SIZE: Record<Exclude<ResolutionPreset, 'current'>, { width: number; height: number }> = {
  '720p': { width: 1280, height: 720 },
  '1080p': { width: 1920, height: 1080 },
};

/** The output size for a preset, or null for the current view. */
export const resolveSize = (preset: ResolutionPreset): { width: number; height: number } | null =>
  preset === 'current' ? null : RESOLUTION_SIZE[preset];

export type VideoQuality = 'low' | 'medium' | 'high';

export const QUALITY_LABEL: Record<VideoQuality, string> = {
  low: 'Low (6 Mbps)',
  medium: 'Medium (12 Mbps)',
  high: 'High (24 Mbps)',
};

/** Target video bitrate in bits per second for a quality preset. */
export const bitrateFor = (quality: VideoQuality): number =>
  quality === 'low' ? 6_000_000 : quality === 'medium' ? 12_000_000 : 24_000_000;

/** Short unit for the overlay's time, from the thermo time unit name (null when unknown). */
const TIME_ABBREV: Record<string, string> = {
  femtoseconds: 'fs',
  picoseconds: 'ps',
  nanoseconds: 'ns',
  microseconds: 'µs',
  seconds: 's',
};

/**
 * The overlay line for a frame: "Step 100 · t = 0.5 fs". Time is step × dt in the run's units (lj is reduced,
 * shown as τ). Without a timestep only the step is shown.
 */
export const overlayText = (step: number, dt: number | null, units: string | undefined, timeUnit: string | null): string => {
  const stepPart = `Step ${step}`;
  if (dt === null || !Number.isFinite(dt)) return stepPart;
  const time = Number((step * dt).toPrecision(4));
  const unit = units === 'lj' ? 'τ' : timeUnit ? TIME_ABBREV[timeUnit] ?? timeUnit : '';
  return `${stepPart} · t = ${time}${unit ? ` ${unit}` : ''}`;
};

/** A degrees-per-frame orbit setting as radians per frame (0 when off or not a number). */
export const orbitRadians = (degreesPerFrame: number): number =>
  Number.isFinite(degreesPerFrame) && degreesPerFrame > 0 ? (degreesPerFrame * Math.PI) / 180 : 0;

/** Zip entries for a set of exported PNG frames, named by their position and step. */
export const frameZipEntries = (
  frames: readonly { position: number; step: number; png: Uint8Array }[],
): ZipEntry[] => frames.map((f) => ({ name: frameFileName(f.position, f.step), data: f.png }));

/** The frames as one stored zip archive. */
export const framesZip = (frames: readonly { position: number; step: number; png: Uint8Array }[], when?: Date): Uint8Array<ArrayBuffer> =>
  buildZip(frameZipEntries(frames), when);
