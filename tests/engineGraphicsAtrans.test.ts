/**
 * dump_modify atrans (per-type atom transparency, dump_image.cpp aopacity) and dump_modify lights (Image::keyLightColor,
 * fillLightColor, backLightColor and ambientColor, dump_image.cpp reset/restore_lighting), with the screen-door mask of
 * Image's draw loops (image.cpp: a pixel is skipped when opacity < 1 and transthresh[ix % 16][iy % 16] > opacity).
 *
 * The exact colours follow Image::draw_pixel (image.cpp at b332f28) with metallic 0, gamma 1 and shiny 1. For the
 * camera-space normal (0, 0, 1) of a face towards the viewer: sat(n . key) = 0.612372, sat(n . fill) = 0.866025,
 * sat(n . back) = 0 and the key highlight pow(0.897878, 16) = 0.178437 (computed with node). A channel is then
 * ambient * c + c * key * 0.612372 + key * 0.178437 + c * fill * 0.866025 (the highlight is added to every channel),
 * truncated after multiplying by 255. With the green surface c = (0, 1, 0), ambient 0.1, key 0.3, fill 0.2 and back 0.5:
 *   G = 0.1 + 0.3 * 0.612372 + 0.3 * 0.178437 + 0.2 * 0.866025 = 0.510448 -> 130
 *   R = 0.3 * 0.178437 = 0.053531 -> 13
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { RasterFrame } from '../src/engine/output/image_objects';
import { drawTriangle } from '../src/engine-gpl/graphics/draw';
import { drawLineAtoms, drawTriAtoms } from '../src/engine-gpl/graphics/atom_objects';
import { TRANSTHRESH } from '../src/engine-gpl/graphics/transthresh';

const frameAlongX = (w = 32, h = 32): RasterFrame => ({
  w,
  h,
  rgb: new Uint8Array(w * h * 3),
  depth: new Float64Array(w * h).fill(-Infinity),
  camRight: [0, 1, 0],
  camUp: [0, 0, 1],
  camDir: [1, 0, 0],
  ctr: [5, 5, 5],
  zdist: 20 / Math.tan(Math.PI / 6) + 10,
  pixelWidth: 20 / h,
});

const frameAlongZ = (w = 32, h = 32): RasterFrame => ({
  ...frameAlongX(w, h),
  rgb: new Uint8Array(w * h * 3),
  depth: new Float64Array(w * h).fill(-Infinity),
  camRight: [1, 0, 0],
  camUp: [0, 1, 0],
  camDir: [0, 0, 1],
});

const at = (f: RasterFrame, col: number, row: number): number[] => {
  const o = (row * f.w + col) * 3;
  return [f.rgb[o], f.rgb[o + 1], f.rgb[o + 2]];
};

/** Decode a P6 PPM written by dump image into rows from the top. */
const decodePpm = (url: string): { w: number; h: number; rgb: Uint8Array } => {
  const i0 = url.indexOf(',');
  const bin = atob(url.slice(i0 + 1));
  const b = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) b[k] = bin.charCodeAt(k);
  let i = 0;
  const tok = (): string => {
    while (i < b.length && [0x20, 0x09, 0x0a, 0x0d].includes(b[i])) i++;
    const s = i;
    while (i < b.length && ![0x20, 0x09, 0x0a, 0x0d].includes(b[i])) i++;
    return String.fromCharCode(...b.subarray(s, i));
  };
  expect(tok()).toBe('P6');
  const w = Number(tok()), h = Number(tok());
  expect(Number(tok())).toBe(255);
  i++;
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

/** Two atoms: type 1 at x = 3 (red, default colour) and type 2 at x = 7 (green), both diameter 1.5, in a 10-unit box. */
const twoTypes = (dump: string, modify = ''): string => `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 3.0 5.0 5.0
create_atoms 2 single 7.0 5.0 5.0
mass * 1.0
${dump}
${modify}
run 0
`;

const runImage = async (input: string): Promise<{ w: number; h: number; rgb: Uint8Array }> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: (_e: EngineEvent) => undefined, writeFile: (n, t) => files.set(n, t) });
  await s.execute(input);
  return decodePpm(files.get('out.0.ppm')!);
};

/** Pixels whose green channel dominates (the type 2 spheres) and whose red one dominates (type 1). */
const classify = (r: { w: number; h: number; rgb: Uint8Array }): { green: number[][]; red: number[][] } => {
  const green: number[][] = [], red: number[][] = [];
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const o = (y * r.w + x) * 3;
    const R = r.rgb[o], G = r.rgb[o + 1];
    if (G > R + 40) green.push([x, y]);
    else if (R > G + 40) red.push([x, y]);
  }
  return { green, red };
};

describe('dump_modify atrans and the screen-door mask (image.cpp transthresh)', () => {
  it('draws a triangle face only where the transthresh mask lets it through', () => {
    const f = frameAlongX();
    // the same face as the tri face test of engineGraphicsLineTri: centre pixel (16, 15), buffer row from the top
    const x: [number, number, number] = [5, 9, 1], y: [number, number, number] = [5, 1, 1], z: [number, number, number] = [5, 5, 9];
    drawTriangle(f, x, y, z, [1, 0, 0], 0.5);
    let drawn = 0;
    for (let row = 0; row < f.h; row++) for (let col = 0; col < f.w; col++) {
      if (!Number.isFinite(f.depth[row * f.w + col])) continue;
      drawn++;
      const native = f.h - 1 - row;
      expect(TRANSTHRESH[col % 16][native % 16] <= 0.5).toBe(true);
    }
    expect(drawn).toBeGreaterThan(0);
  });

  it('draws nothing for a line atom or tri particle at atrans 0', () => {
    const lines = frameAlongZ();
    drawLineAtoms(lines, [{ x: 5, y: 5, length: 4, theta: 0, color: [255, 0, 0], opacity: 0 }], 1);
    expect(lines.depth.every((d) => !Number.isFinite(d))).toBe(true);
    const corners = [2, -2, 0, -2, -2, 0, 0, 2, 0];
    const tris = frameAlongZ();
    drawTriAtoms(tris, [{ center: [5, 5, 0], quat: [1, 0, 0, 0], corners, color: [255, 0, 0], opacity: 0 }], 3, 0.2);
    expect(tris.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });

  it('gives each line atom its own opacity: an invisible segment leaves the others unchanged', () => {
    // segment A at atrans 0 and segment B at atrans 1 (both through (5, 5)): the image equals B drawn alone
    const both = frameAlongZ(), alone = frameAlongZ();
    const A = { x: 5, y: 5, length: 4, theta: 0, color: [255, 0, 0] as [number, number, number], opacity: 0 };
    const B = { x: 5, y: 5, length: 4, theta: Math.PI / 2, color: [0, 255, 0] as [number, number, number], opacity: 1 };
    drawLineAtoms(both, [A, B], 1);
    drawLineAtoms(alone, [B], 1);
    expect(Array.from(both.rgb)).toEqual(Array.from(alone.rgb));
    expect(Array.from(both.depth)).toEqual(Array.from(alone.depth));
  });

  it('keeps the atoms of a type with atrans 0 out of the image, and masks the others (dump image)', async () => {
    const r = await runImage(twoTypes('dump im all image 1 out.*.ppm type type size 120 120 box no 0.02 view 0 0 zoom 1.6', 'dump_modify im atrans 2 0'));
    const c = classify(r);
    expect(c.red.length).toBeGreaterThan(0);
    expect(c.green.length).toBe(0);
  });

  it('draws a half-transparent type through the transthresh mask only', async () => {
    const opaque = classify(await runImage(twoTypes('dump im all image 1 out.*.ppm type type size 120 120 box no 0.02 view 0 0 zoom 1.6')));
    const half = classify(await runImage(twoTypes('dump im all image 1 out.*.ppm type type size 120 120 box no 0.02 view 0 0 zoom 1.6', 'dump_modify im atrans 2 0.5')));
    expect(opaque.green.length).toBeGreaterThan(0);
    expect(half.green.length).toBeGreaterThan(0);
    expect(half.green.length).toBeLessThan(opaque.green.length);
    // the red atoms are unaffected by atrans 2
    expect(half.red.length).toBe(opaque.red.length);
    for (const [x, y] of half.green) expect(TRANSTHRESH[x % 16][(120 - 1 - y) % 16] <= 0.5).toBe(true);
  });

  it('refuses an atrans value outside 0..1 and names the keyword', async () => {
    await expect(runImage(twoTypes('dump im all image 1 out.*.ppm type type size 40 40', 'dump_modify im atrans 2 1.5'))).rejects.toThrow(/atrans/);
  });
});

describe('dump_modify lights (Image key, fill, back and ambient colours)', () => {
  it('shades a face with the lights set on the frame (exact colour)', () => {
    const f: RasterFrame = { ...frameAlongX(), ambient: 0.1, lights: { key: 0.3, fill: 0.2, back: 0.5 } };
    const x: [number, number, number] = [5, 9, 1], y: [number, number, number] = [5, 1, 1], z: [number, number, number] = [5, 5, 9];
    drawTriangle(f, x, y, z, [0, 1, 0]);
    expect(at(f, 16, 15)).toEqual([13, 130, 13]);
  });

  it('adds the ambient light 0.1 to the type 2 sphere (forestgreen, the default type 2 colour; the default lights have ambient 0)', async () => {
    const base = 'dump im all image 1 out.*.ppm type type size 120 120 box no 0.02 view 0 0 zoom 1.6';
    const lit = await runImage(twoTypes(base, 'dump_modify im lights 0.1 0.9 0.45 0.9'));
    const dflt = await runImage(twoTypes(base));
    const greenLit = classify(lit).green, greenDflt = classify(dflt).green;
    expect(greenLit.length).toBeGreaterThan(0);
    // lights 0.1 0.9 0.45 0.9 differ from the defaults only in the ambient 0.1: the type 2 colour is forestgreen
    // (0.133, 0.545, 0.133, dump_image.cpp default_colors[2]), so the green channel gains 0.1 * 0.545 * 255 = 13.9
    // before truncation. Measured with native LAMMPS (b332f28, this scene): the difference is 13 or 14
    const diffs: number[] = [];
    for (const [x, y] of greenDflt) {
      const o = (y * dflt.w + x) * 3 + 1;
      if (dflt.rgb[o] <= 229) diffs.push(lit.rgb[o] - dflt.rgb[o]);
    }
    expect(diffs.length).toBeGreaterThan(0);
    for (const d of diffs) expect([13, 14]).toContain(d);
  });

  it('refuses a light value outside 0..1 and names the keyword', async () => {
    await expect(runImage(twoTypes('dump im all image 1 out.*.ppm type type size 40 40', 'dump_modify im lights 0.1 1.5 0.2 0.3'))).rejects.toThrow(/key light/);
  });
});
