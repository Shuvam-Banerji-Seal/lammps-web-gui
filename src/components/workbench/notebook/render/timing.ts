/**
 * Timing of a clip: the playback speed factor, the frames held at the start and end, and the loops. The plan is
 * a list of recorded-frame positions, one per output frame, so the video, the GIF and the PNG sequence all play
 * the same frames in the same order. Pure.
 */

/** Playback speed factors offered in the panel (2 plays twice as fast: every other frame; 0.5 is slow motion). */
export const SPEED_CHOICES = [0.25, 0.5, 1, 2, 4] as const;

/** Output frames a render may produce: a guard against a clip too long to encode in the browser. */
export const MAX_OUTPUT_FRAMES = 6000;

/**
 * The output frame sequence over `selected` (buffer indices, already strided): output frame k shows the selected
 * frame at floor(k × speed), so a speed of 0.5 shows each frame twice and a speed of 2 skips every other one. The
 * first frame is held `holdFirst` extra times and the last `holdLast` times, then the whole clip repeats `loops`
 * times.
 */
export const playbackPlan = (
  selected: readonly number[],
  opts: { speed: number; holdFirst: number; holdLast: number; loops: number },
): number[] => {
  const n = selected.length;
  if (n === 0) return [];
  const speed = Number.isFinite(opts.speed) && opts.speed > 0 ? opts.speed : 1;
  const frames = Math.max(1, Math.round(n / speed));
  const body: number[] = [];
  for (let k = 0; k < frames; k++) {
    const pos = Math.min(n - 1, Math.floor(k * speed));
    body.push(selected[pos]);
  }
  const holdA = Math.max(0, Math.floor(opts.holdFirst) || 0);
  const holdB = Math.max(0, Math.floor(opts.holdLast) || 0);
  const once = [...new Array<number>(holdA).fill(body[0]), ...body, ...new Array<number>(holdB).fill(body[body.length - 1])];
  const loops = Math.min(20, Math.max(1, Math.floor(opts.loops) || 1));
  const out: number[] = [];
  for (let r = 0; r < loops; r++) for (const i of once) out.push(i);
  return out;
};

/** Frames per second of the clip in GIF delay units (1/100 s, at least 1). */
export const gifDelayCentiseconds = (fps: number): number =>
  Math.max(1, Math.round(100 / Math.max(1, fps)));
