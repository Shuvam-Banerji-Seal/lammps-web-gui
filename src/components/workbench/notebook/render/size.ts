/**
 * Output size for the Render panel: aspect presets, the 720p and 1080p short-side presets, custom width and
 * height, supersampling, and the cap to what the GPU canvas allows. Pure: no DOM, no three.js.
 */

export type AspectChoice = 'native' | '16:9' | '1:1' | '4:3' | '9:16';
export type SizeChoice = 'current' | '720p' | '1080p' | 'custom';
export type Supersample = 1 | 2;

export interface Size {
  width: number;
  height: number;
}

export const ASPECT_LABEL: Record<AspectChoice, string> = {
  native: 'Native',
  '16:9': '16:9',
  '1:1': '1:1',
  '4:3': '4:3',
  '9:16': '9:16 (vertical)',
};

/** width / height for each fixed aspect. */
export const ASPECT_RATIO: Record<Exclude<AspectChoice, 'native'>, number> = {
  '16:9': 16 / 9,
  '1:1': 1,
  '4:3': 4 / 3,
  '9:16': 9 / 16,
};

export const SIZE_LABEL: Record<SizeChoice, string> = {
  current: 'Current view',
  '720p': '720p',
  '1080p': '1080p',
  custom: 'Custom width × height',
};

/** The short side in pixels of the 720p and 1080p presets (the long side follows the aspect). */
export const SHORT_SIDE: Record<'720p' | '1080p', number> = { '720p': 720, '1080p': 1080 };

/** Video encoders need even dimensions (H.264 and VP8/VP9 chroma subsampling); rounds to the nearest even, at least 2. */
export const even = (n: number): number => Math.max(2, 2 * Math.round(n / 2));

/** Width and height for a short side and a width/height ratio, both even. */
export const fromShortSide = (short: number, ratio: number): Size =>
  ratio >= 1
    ? { width: even(short * ratio), height: even(short) }
    : { width: even(short), height: even(short / ratio) };

export interface SizeRequest {
  size: SizeChoice;
  aspect: AspectChoice;
  /** The 3D view's drawing size (what "Current view" means). */
  view: Size;
  /** Width and height typed for "Custom". */
  custom: Size;
}

/**
 * The size an output is drawn at, before the GPU cap.
 * - Custom: the typed width and height (the aspect is not used).
 * - 720p / 1080p: the short side is that number; the aspect gives the long side. Native aspect is 16:9 here,
 *   so the presets keep their former meaning (1280 × 720 and 1920 × 1080).
 * - Current view: the view's own size; with an aspect, the view's short side is kept and the aspect applied.
 */
export const outputSize = (req: SizeRequest): Size => {
  if (req.size === 'custom') return { width: even(req.custom.width), height: even(req.custom.height) };
  if (req.size === 'current') {
    if (req.aspect === 'native') return { width: even(req.view.width), height: even(req.view.height) };
    return fromShortSide(Math.min(req.view.width, req.view.height), ASPECT_RATIO[req.aspect]);
  }
  const short = SHORT_SIDE[req.size];
  const ratio = req.aspect === 'native' ? 16 / 9 : ASPECT_RATIO[req.aspect];
  return fromShortSide(short, ratio);
};

export interface GpuLimits {
  /** Largest side in pixels (the smaller of MAX_VIEWPORT_DIMS and MAX_RENDERBUFFER_SIZE). */
  maxSide: number;
  /** Largest area in pixels. */
  maxArea: number;
}

/** Default limits when the GPU cannot be queried: 8192 px a side, 40 megapixels. */
export const DEFAULT_LIMITS: GpuLimits = { maxSide: 8192, maxArea: 40_000_000 };

/**
 * Scales a size down (keeping its aspect, even dimensions) until its longest side and its area fit the limits.
 * `capped` is true when it had to shrink.
 */
export const fitToLimits = (size: Size, limits: GpuLimits): Size & { capped: boolean } => {
  const longest = Math.max(size.width, size.height);
  const scale = Math.min(1, limits.maxSide / longest, Math.sqrt(limits.maxArea / (size.width * size.height)));
  if (scale >= 1) return { width: size.width, height: size.height, capped: false };
  return {
    width: Math.max(2, 2 * Math.floor((size.width * scale) / 2)),
    height: Math.max(2, 2 * Math.floor((size.height * scale) / 2)),
    capped: true,
  };
};

export interface RenderSizePlan {
  /** The output canvas (the frame or video size). */
  out: Size;
  /** The size the 3D view is drawn at (out × supersampling, within the limits). */
  render: Size;
  /** True when the requested output or the supersampling had to shrink for the GPU. */
  capped: boolean;
  /** The supersampling actually used. */
  supersample: Supersample;
}

/**
 * The output size and the render size for a request. Supersampling draws the view at 2× the output and scales
 * it down into the output canvas (anti-aliasing); when the GPU cannot hold that, it falls back to 1×.
 */
export const planRenderSize = (desired: Size, supersample: Supersample, limits: GpuLimits): RenderSizePlan => {
  const out = fitToLimits(desired, limits);
  const wanted = fitToLimits({ width: out.width * supersample, height: out.height * supersample }, limits);
  // a supersampled render that had to shrink is not what was asked for: draw at 1x instead
  const useSuper = supersample === 2 && !wanted.capped;
  const render = useSuper ? wanted : { width: out.width, height: out.height, capped: false };
  return {
    out: { width: out.width, height: out.height },
    render: { width: render.width, height: render.height },
    capped: out.capped || (supersample === 2 && !useSuper),
    supersample: useSuper ? 2 : 1,
  };
};
