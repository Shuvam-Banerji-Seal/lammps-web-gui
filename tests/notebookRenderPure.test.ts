import { describe, it, expect } from 'vitest';
import {
  even, fromShortSide, outputSize, fitToLimits, planRenderSize, DEFAULT_LIMITS,
} from '../src/components/workbench/notebook/render/size';
import {
  atomColours, colourBarLabels, displacementSpeeds, hexToRgb, heatColour, rangeOf, sameAtoms, scalarValues,
  type ColourFrame,
} from '../src/components/workbench/notebook/render/colour';
import { cameraFrame, clipFraction, orbitPose, reviveOrbitDegrees } from '../src/components/workbench/notebook/render/camera';
import { gifDelayCentiseconds, playbackPlan } from '../src/components/workbench/notebook/render/timing';
import {
  colourBarLayout, layoutTextBlock, rotateByQuat, triadArrows, triadLayout,
} from '../src/components/workbench/notebook/render/overlay';
import { backgroundHex, unmatte, RADIUS_SCALE_CHOICES } from '../src/components/workbench/notebook/render/look';
import { frameFileName, framesZip } from '../src/components/workbench/notebook/renderFrames';

const req = (size: 'current' | '720p' | '1080p' | 'custom', aspect: 'native' | '16:9' | '1:1' | '4:3' | '9:16', view = { width: 800, height: 600 }, custom = { width: 1000, height: 500 }) =>
  ({ size, aspect, view, custom });

describe('output size', () => {
  it('keeps the old presets: 720p is 1280 × 720 and 1080p is 1920 × 1080 with the native (16:9) aspect', () => {
    expect(outputSize(req('720p', 'native'))).toEqual({ width: 1280, height: 720 });
    expect(outputSize(req('1080p', 'native'))).toEqual({ width: 1920, height: 1080 });
  });

  it('gives 1:1, 4:3 and 9:16 from the short side, all even', () => {
    expect(outputSize(req('1080p', '1:1'))).toEqual({ width: 1080, height: 1080 });
    expect(outputSize(req('1080p', '4:3'))).toEqual({ width: 1440, height: 1080 });
    expect(outputSize(req('720p', '9:16'))).toEqual({ width: 720, height: 1280 });
    expect(outputSize(req('1080p', '16:9'))).toEqual({ width: 1920, height: 1080 });
  });

  it('uses the typed custom size as given (rounded to even), ignoring the aspect', () => {
    expect(outputSize(req('custom', '1:1', undefined, { width: 641, height: 359 }))).toEqual({ width: 642, height: 360 });
  });

  it('keeps the view size for Current view, and applies an aspect to its short side', () => {
    expect(outputSize(req('current', 'native', { width: 801, height: 600 }))).toEqual({ width: 802, height: 600 });
    expect(outputSize(req('current', '1:1', { width: 800, height: 600 }))).toEqual({ width: 600, height: 600 });
  });

  it('rounds to even numbers and never returns 0', () => {
    expect(even(3)).toBe(4);
    expect(even(0)).toBe(2);
    expect(fromShortSide(7, 16 / 9)).toEqual({ width: 12, height: 8 });
  });

  it('caps a size that is over the GPU limits, keeping the aspect', () => {
    const big = fitToLimits({ width: 8000, height: 4000 }, { maxSide: 4096, maxArea: 40_000_000 });
    expect(big.capped).toBe(true);
    expect(big.width).toBeLessThanOrEqual(4096);
    expect(big.width / big.height).toBeCloseTo(2, 2);
    expect(fitToLimits({ width: 1920, height: 1080 }, DEFAULT_LIMITS).capped).toBe(false);
    // the area limit applies too
    const area = fitToLimits({ width: 8000, height: 8000 }, { maxSide: 16384, maxArea: 10_000_000 });
    expect(area.capped).toBe(true);
    expect(area.width * area.height).toBeLessThanOrEqual(10_000_000);
  });

  it('supersamples at 2x when the GPU holds it, and falls back to 1x (reporting it) when it does not', () => {
    const ok = planRenderSize({ width: 960, height: 540 }, 2, DEFAULT_LIMITS);
    expect(ok.render).toEqual({ width: 1920, height: 1080 });
    expect(ok.out).toEqual({ width: 960, height: 540 });
    expect(ok.supersample).toBe(2);
    expect(ok.capped).toBe(false);

    const tight = planRenderSize({ width: 960, height: 540 }, 2, { maxSide: 1500, maxArea: 40_000_000 });
    expect(tight.supersample).toBe(1);
    expect(tight.capped).toBe(true);
    expect(tight.render).toEqual({ width: 960, height: 540 });
  });
});

describe('colour', () => {
  const frame = (step: number, xs: number[], ids: number[], v?: number[], lo = [0, 0, 0], hi = [10, 10, 10]): ColourFrame => ({
    step,
    x: Float64Array.from(xs),
    v: v ? Float32Array.from(v) : undefined,
    box: { lo, hi },
    topo: { id: Int32Array.from(ids) },
  });

  it('reports a colour range with a loop, and labels with three significant digits', () => {
    expect(rangeOf([Float64Array.from([3, -1, 7]), null, Float64Array.from([2])])).toEqual({ min: -1, max: 7 });
    expect(rangeOf([])).toBeNull();
    expect(colourBarLabels({ min: 0.123456, max: 1234.5 })).toEqual({ low: '0.123', high: '1230' });
  });

  it('maps the range onto the colour map: low is dark, high is bright, a flat range takes the middle', () => {
    const lo = heatColour(0);
    const hi = heatColour(1);
    const luma = (c: number[]) => 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    expect(luma(hi)).toBeGreaterThan(luma(lo));
    expect(heatColour(-5)).toEqual(heatColour(0));
    expect(heatColour(9)).toEqual(heatColour(1));
    const flat = atomColours(Float64Array.from([4, 4]), 2, { min: 4, max: 4 }, '#000000');
    const mid = heatColour(0.5);
    expect(flat[0]).toBeCloseTo(mid[0], 5);
  });

  it('fills a single colour with hexToRgb, and falls back to magenta for an invalid colour', () => {
    expect(hexToRgb('#ff0000')).toEqual([1, 0, 0]);
    expect(hexToRgb('nope')).toEqual([1, 0, 1]);
    const arr = atomColours(null, 2, null, '#00ff00');
    expect(Array.from(arr)).toEqual([0, 1, 0, 0, 1, 0]);
  });

  it('gives atom id, height and kinetic energy (m = 1) per atom', () => {
    const f = frame(0, [1, 2, 3, 4, 5, 6], [7, 9]);
    expect(Array.from(scalarValues('id', undefined, f, undefined, null)!)).toEqual([7, 9]);
    expect(Array.from(scalarValues('y', undefined, f, undefined, null)!)).toEqual([2, 5]);
    const fv = frame(0, [0, 0, 0, 0, 0, 0], [1, 2], [3, 4, 0, 0, 0, 0]);
    expect(Array.from(scalarValues('speed', undefined, fv, undefined, 0.1)!)).toEqual([5, 0]);
    expect(Array.from(scalarValues('ke', undefined, fv, undefined, 0.1)!)).toEqual([12.5, 0]);
  });

  it('gets speed from displacement between neighbouring frames, with the minimum image in periodic boxes', () => {
    // one atom moves 1 unit per step in +x, through the boundary of a 10-long box
    const a = frame(0, [9.5, 0, 0], [1]);
    const b = frame(1, [0.5, 0, 0], [1]); // wrapped: really moved +1
    const c = frame(2, [1.5, 0, 0], [1]);
    const sp = displacementSpeeds(a, b, c, 1);
    expect(sp).not.toBeNull();
    expect(sp![0]).toBeCloseTo(1, 6); // 9.5 -> 1.5 across the boundary is +2 in two steps: 1 per step
    expect(displacementSpeeds(undefined, b, undefined, null)).toBeNull();
    expect(displacementSpeeds(undefined, b, undefined, 0.5)).toBeNull();
  });

  it('knows when two frames hold the same atoms', () => {
    const a = frame(0, [0, 0, 0], [1]);
    const b = frame(1, [0, 0, 0], [1]);
    const c = frame(1, [0, 0, 0], [2]);
    expect(sameAtoms(a, b)).toBe(true);
    expect(sameAtoms(a, c)).toBe(false);
    expect(sameAtoms(a, frame(0, [0, 0, 0], [1, 2]))).toBe(false);
  });
});

describe('camera path', () => {
  it('is still for the static mode and at the start of a clip', () => {
    expect(cameraFrame('static', 3, 10, 360)).toEqual({ angle: 0, scale: 1 });
    expect(cameraFrame('orbit', 0, 10, 360).angle).toBe(0);
    expect(clipFraction(0, 1)).toBe(0);
  });

  it('turns exactly the chosen angle over the clip, once per frame and deterministic', () => {
    const last = cameraFrame('orbit', 9, 10, 360);
    expect(last.angle).toBeCloseTo(2 * Math.PI, 12);
    expect(cameraFrame('orbit', 4, 10, 360)).toEqual(cameraFrame('orbit', 4, 10, 360));
    expect(cameraFrame('orbit', 9, 10, 180).angle).toBeCloseTo(Math.PI, 12);
  });

  it('zooms in and then out, ending where it began', () => {
    const start = cameraFrame('zoom', 0, 11, 0).scale;
    const mid = cameraFrame('zoom', 5, 11, 0).scale;
    const end = cameraFrame('zoom', 10, 11, 0).scale;
    expect(start).toBe(1);
    expect(end).toBeCloseTo(1, 12);
    expect(mid).toBeCloseTo(0.75, 12);
  });

  it('turns a pose about the vertical axis, keeping its height and its distance from the axis', () => {
    const start: [number, number, number] = [3, 2, 4];
    const [x, y, z] = orbitPose(start, Math.PI / 2, 1);
    expect(y).toBeCloseTo(2, 12);
    expect(Math.hypot(x, z)).toBeCloseTo(Math.hypot(3, 4), 12);
    expect(orbitPose(start, 2 * Math.PI, 1)[0]).toBeCloseTo(3, 9);
    expect(orbitPose(start, 0, 0.5)).toEqual([1.5, 1, 2]);
  });

  it('reads the orbit degrees from the input text, defaulting to 360', () => {
    expect(reviveOrbitDegrees('90')).toBe(90);
    expect(reviveOrbitDegrees('')).toBe(360);
    expect(reviveOrbitDegrees('-5')).toBe(360);
    expect(reviveOrbitDegrees('abc')).toBe(360);
  });
});

describe('timing plan', () => {
  it('plays every frame once at speed 1, in order', () => {
    expect(playbackPlan([0, 2, 4], { speed: 1, holdFirst: 0, holdLast: 0, loops: 1 })).toEqual([0, 2, 4]);
  });

  it('shows each frame twice at half speed (slow motion) and skips every other frame at double speed', () => {
    expect(playbackPlan([0, 1, 2], { speed: 0.5, holdFirst: 0, holdLast: 0, loops: 1 })).toEqual([0, 0, 1, 1, 2, 2]);
    expect(playbackPlan([0, 1, 2, 3], { speed: 2, holdFirst: 0, holdLast: 0, loops: 1 })).toEqual([0, 2]);
  });

  it('holds the first and last frame, then repeats the whole clip for loops', () => {
    expect(playbackPlan([5, 6], { speed: 1, holdFirst: 2, holdLast: 1, loops: 1 })).toEqual([5, 5, 5, 6, 6]);
    expect(playbackPlan([5, 6], { speed: 1, holdFirst: 0, holdLast: 0, loops: 2 })).toEqual([5, 6, 5, 6]);
    expect(playbackPlan([], { speed: 1, holdFirst: 3, holdLast: 3, loops: 2 })).toEqual([]);
  });

  it('gives GIF delays in hundredths of a second, at least 1', () => {
    expect(gifDelayCentiseconds(25)).toBe(4);
    expect(gifDelayCentiseconds(10)).toBe(10);
    expect(gifDelayCentiseconds(1000)).toBe(1);
  });
});

describe('overlays', () => {
  const measure = (text: string, px: number, bold: boolean) => text.length * px * (bold ? 0.6 : 0.5);

  it('places the text block in the chosen corner, inside the frame', () => {
    for (const corner of ['tl', 'tr', 'bl', 'br'] as const) {
      const b = layoutTextBlock(640, 360, corner, [{ text: 'Title', bold: true, scale: 1.4 }, { text: 'Step 10 · t = 0.5 τ' }], 'm', measure)!;
      expect(b.box.x).toBeGreaterThanOrEqual(0);
      expect(b.box.y).toBeGreaterThanOrEqual(0);
      expect(b.box.x + b.box.w).toBeLessThanOrEqual(640);
      expect(b.box.y + b.box.h).toBeLessThanOrEqual(360);
      expect(corner[1] === 'l' ? b.box.x : 640 - (b.box.x + b.box.w)).toBeLessThan(b.px * 2);
      expect(corner[0] === 't' ? b.box.y : 360 - (b.box.y + b.box.h)).toBeLessThan(b.px * 2);
    }
  });

  it('scales with the font size and drops empty lines', () => {
    expect(layoutTextBlock(640, 360, 'tl', [{ text: '   ' }], 'm', measure)).toBeNull();
    const small = layoutTextBlock(640, 360, 'tl', [{ text: 'a' }], 's', measure)!;
    const large = layoutTextBlock(640, 360, 'tl', [{ text: 'a' }], 'l', measure)!;
    expect(large.px).toBeGreaterThan(small.px);
    expect(large.lines.length).toBe(1);
  });

  it('puts the colour bar at the right edge, centred vertically, inside the frame', () => {
    const cb = colourBarLayout(1280, 720);
    expect(cb.x + cb.w).toBeLessThanOrEqual(1280);
    expect(cb.y + cb.h / 2).toBeCloseTo(360, 0);
    expect(cb.labelX).toBeLessThan(cb.x);
  });

  it('rotates vectors by a quaternion, and draws the triad from the camera orientation', () => {
    const id: [number, number, number, number] = [0, 0, 0, 1];
    expect(rotateByQuat(id, [1, 2, 3])).toEqual([1, 2, 3]);
    // 180 degrees about z flips x and y
    const rz: [number, number, number, number] = [0, 0, 1, 0];
    const v = rotateByQuat(rz, [1, 0, 0]);
    expect(v[0]).toBeCloseTo(-1, 12);
    expect(v[1]).toBeCloseTo(0, 12);

    const arrows = triadArrows(id);
    const x = arrows.find((a) => a.axis === 'x')!;
    const y = arrows.find((a) => a.axis === 'y')!;
    expect(x.dx).toBeCloseTo(1, 12);
    expect(y.dy).toBeCloseTo(-1, 12); // canvas y points down, so world up is -dy
    const tl = triadLayout(640, 360);
    expect(tl.cx).toBeLessThan(320);
    expect(tl.cy).toBeGreaterThan(180);
  });
});

describe('look', () => {
  it('maps the background choice to a colour, or keeps the theme', () => {
    expect(backgroundHex('theme', '#123456')).toBeNull();
    expect(backgroundHex('black', '')).toBe('#000000');
    expect(backgroundHex('white', '')).toBe('#ffffff');
    expect(backgroundHex('custom', '#102030')).toBe('#102030');
    expect(backgroundHex('custom', 'bad')).toBe('#000000');
    expect(RADIUS_SCALE_CHOICES).toContain(1);
  });

  it('recovers alpha and colour from a black and a white pass (difference matting)', () => {
    // one pixel drawn with alpha 0.5 and colour (200, 100, 0): over black 100,50,0; over white 100+127.5...
    const black = Uint8ClampedArray.from([100, 50, 0, 255]);
    const white = Uint8ClampedArray.from([228, 178, 128, 255]);
    const out = unmatte(black, white);
    expect(out[3]).toBeCloseTo(127, -1);
    expect(out[0]).toBeCloseTo(200, -1);
    expect(out[1]).toBeCloseTo(100, -1);
    expect(out[2]).toBeCloseTo(0, -1);
  });

  it('leaves background pixels (identical in both passes, i.e. fully transparent) at zero alpha', () => {
    const b = Uint8ClampedArray.from([0, 0, 0, 255, 255, 255, 255, 255]);
    const w = Uint8ClampedArray.from([255, 255, 255, 255, 255, 255, 255, 255]);
    const out = unmatte(b, w);
    expect(Array.from(out)).toEqual([0, 0, 0, 0, 255, 255, 255, 255]);
  });
});

describe('frames zip', () => {
  it('names frames by position and step and stores one entry per frame', () => {
    expect(frameFileName(3, 250)).toBe('frame_0003_step_250.png');
    const frames = [1, 2, 3].map((p) => ({ position: p, step: p * 10, png: Uint8Array.from([p, p]) }));
    const zip = framesZip(frames, new Date(2026, 9, 11, 12, 0, 0));
    // count the local file headers (signature 0x04034b50) and the end record (0x06054b50)
    const dv = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
    let entries = 0;
    for (let i = 0; i + 4 <= zip.length; i++) if (dv.getUint32(i, true) === 0x04034b50) entries++;
    expect(entries).toBe(3);
    expect(dv.getUint32(zip.length - 22, true)).toBe(0x06054b50);
    expect(dv.getUint16(zip.length - 22 + 8, true)).toBe(3);
  });
});
