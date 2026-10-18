/**
 * Rendering of the GRAPHICS stage 2 objects (src/engine-gpl/graphics/draw.ts, a port of image.cpp's draw_sphere,
 * draw_pixmap, draw_trinorm and draw_pixel) into small images. The pixel values are worked out from the C++ by hand:
 *
 *   camera: view 90 0 (camDir = +x, camRight = +y, camUp = +z), box 10 x 10 x 10 centred on (5, 5, 5), 32 x 32 px.
 *   pixelWidth = 2 * 10 / 32 = 0.625 world units per pixel, so a world point is 1.6 px per unit from the centre
 *   pixel (native column 16, row 15: native row iy = 16 is row 15 of the buffer, which counts from the top).
 *
 *   light (draw_pixel, metallic 0, gamma 1, ambient 0, shiny 1): for a camera-space normal n and colour c,
 *     c_k = c_k * 0.9 * sat(n.key) + 0.9 * pow(sat(n.half), 16) + c_k * 0.45 * sat(n.fill) + c_k * 0.9 * sat(n.back)
 *   key = (cos30 sin(-45), sin30, cos30 cos(-45)), fill = (0, 0, 1) rotated to (sin30, 0, cos30), back = (0, sin15, -cos15),
 *   half = normalise(key + (0, 0, 1)). A surface facing the viewer has n = (0, 0, 1): key 0.61237, fill 0.86603,
 *   back 0 and pow(0.89788, 16) = 0.17814, so c_k = 0.94085 c_k + 0.16033, then int(255 c_k).
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { RasterFrame } from '../src/engine/output/image_objects';
import { drawCylinder, drawPixmapBox, drawSphere, drawTrinorm, drawPixel, scalePixmap } from '../src/engine-gpl/graphics/draw';

const frame = (w = 32, h = 32): RasterFrame => ({
  w,
  h,
  rgb: new Uint8Array(w * h * 3),
  depth: new Float64Array(w * h).fill(-Infinity),
  camRight: [0, 1, 0],
  camUp: [0, 0, 1],
  camDir: [1, 0, 0],
  ctr: [5, 5, 5],
  // view_params: zdist = (2 * 10 / tan(30 deg) + 0.5 * 20) for a 10-unit cube viewed along x
  zdist: 20 / Math.tan(Math.PI / 6) + 10,
  pixelWidth: 20 / h,
});

/** RGB of frame pixel (col, row), row 0 at the top. */
const at = (f: RasterFrame, col: number, row: number): number[] => {
  const o = (row * f.w + col) * 3;
  return [f.rgb[o], f.rgb[o + 1], f.rgb[o + 2]];
};

const b64 = (url: string): Uint8Array => {
  const i = url.indexOf(',');
  const bin = atob(url.slice(i + 1));
  const out = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) out[k] = bin.charCodeAt(k);
  return out;
};

/** Decode a P6 PPM written by dump image into rows from the top. */
const decodePpm = (b: Uint8Array): { w: number; h: number; rgb: Uint8Array } => {
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

const run = async (input: string): Promise<{ files: Map<string, string>; logs: string[] }> => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(input);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { files, logs };
};

/** A 10 x 10 x 10 box with one atom at its centre, and the input lines around the run. */
const scene = (pre: string, dump: string, atoms = 'create_atoms 1 single 5 5 5'): string => `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 2 box
${atoms}
mass * 1.0
${pre}
${dump}
run 0
`;

describe('draw_pixel shading (image.cpp draw_pixel with the dump image defaults)', () => {
  it('a surface facing the viewer with colour (0.2, 0.4, 0.6) is (88, 136, 184)', () => {
    const f = frame();
    drawPixel(f, 16, 16, 10, [0, 0, 1], [0.2, 0.4, 0.6]);
    // native row iy = 16 is buffer row 15
    expect(at(f, 16, 15)).toEqual([88, 136, 184]);
  });

  it('keeps the nearer pixel: a later pixel that is not nearer does not replace it', () => {
    const f = frame();
    drawPixel(f, 16, 16, 10, [0, 0, 1], [1, 0, 0]);
    // the red pixel (shaded: R 255, B 40 from the specular term) stays against the farther blue one
    drawPixel(f, 16, 16, 12, [0, 0, 1], [0, 0, 1]);
    expect(at(f, 16, 15)[0]).toBeGreaterThan(at(f, 16, 15)[2]);
    // a nearer blue pixel replaces it
    drawPixel(f, 16, 16, 8, [0, 0, 1], [0, 0, 1]);
    expect(at(f, 16, 15)[2]).toBeGreaterThan(at(f, 16, 15)[0]);
  });

  it('rejects depths behind the camera (depth < 0)', () => {
    const f = frame();
    drawPixel(f, 5, 5, -1, [0, 0, 1], [1, 1, 1]);
    expect(at(f, 5, 26)).toEqual([0, 0, 0]);
  });
});

describe('draw_sphere (image.cpp)', () => {
  it('a sphere of diameter 4 at the box centre covers the centre pixel and not 2.5 units away', () => {
    const f = frame();
    drawSphere(f, [5, 5, 5], [0.2, 0.4, 0.6], 4);
    expect(at(f, 16, 15)).toEqual([88, 136, 184]);
    // 4 px from the centre along the screen: radius 2 units = 3.2 px, so column 20 (4 px) is outside
    expect(at(f, 20, 15)).toEqual([0, 0, 0]);
    // column 18 is 2 px = 1.25 units out, inside the sphere
    expect(at(f, 18, 15)[0]).toBeGreaterThan(0);
  });

  it('a zero diameter draws nothing', () => {
    const f = frame();
    drawSphere(f, [5, 5, 5], [1, 1, 1], 0);
    expect(f.rgb.every((v) => v === 0)).toBe(true);
  });
});

describe('draw_cylinder (image.cpp)', () => {
  it('a cylinder along the screen x axis through the centre shows its front (normal toward the viewer) on the axis', () => {
    const f = frame();
    // from y = 3 to y = 7 (screen columns 16 - 3.2 .. 16 + 3.2), diameter 2 (radius 1 unit = 1.6 px)
    drawCylinder(f, [5, 3, 5], [5, 7, 5], [0.2, 0.4, 0.6], 2, 0);
    expect(at(f, 16, 15)).toEqual([88, 136, 184]);
    // three pixels above the axis is 1.9 units up, outside the radius 1 unit
    expect(at(f, 16, 12)).toEqual([0, 0, 0]);
    // the ends are beyond the caps: no sphere cap without flag bits, so column 16 + 8 (5 units) is empty
    expect(at(f, 16 + 8, 15)).toEqual([0, 0, 0]);
  });

  it('sflag 3 adds sphere caps at both ends', () => {
    // the end at y = 7 is 2 units = 3.2 px from the centre, column 19.2; column 20 (0.8 px past it) is past the body
    const plain = frame();
    drawCylinder(plain, [5, 3, 5], [5, 7, 5], [0.2, 0.4, 0.6], 2, 0);
    expect(at(plain, 20, 15)).toEqual([0, 0, 0]);
    const capped = frame();
    drawCylinder(capped, [5, 3, 5], [5, 7, 5], [0.2, 0.4, 0.6], 2, 3);
    // the cap sphere of radius 1 unit (1.6 px) at column 19.2 covers column 20 and not column 22 (2.8 px away)
    expect(at(capped, 20, 15)[1]).toBeGreaterThan(0);
    expect(at(capped, 22, 15)).toEqual([0, 0, 0]);
  });
});

describe('draw_trinorm (image.cpp)', () => {
  it('a triangle facing the viewer with its centre on the box centre shades like a sphere facing the viewer', () => {
    const f = frame();
    // in the plane x = 5; the screen triangle (y, z) = (2, 2), (8, 2), (5, 8) contains (5, 5)
    const n: [number, number, number] = [1, 0, 0];
    const c: [number, number, number] = [0.2, 0.4, 0.6];
    drawTrinorm(f, [5, 2, 2], [5, 8, 2], [5, 5, 8], n, n, n, c, c, c);
    expect(at(f, 16, 15)).toEqual([88, 136, 184]);
    // the bottom corner region far outside the triangle stays empty
    expect(at(f, 16, 0)).toEqual([0, 0, 0]);
  });
});

describe('draw_pixmap (image.cpp, box coordinates)', () => {
  it('a 2 x 2 pixmap centred on the box centre: rows count from the bottom, transcolor pixels are skipped', () => {
    const f = frame();
    // pixmap rows from the bottom: row 0 = [red 0.4, black], row 1 = [green, blue]
    const pix = new Uint8Array([102, 0, 0, 0, 0, 0, 0, 102, 0, 0, 0, 102]);
    drawPixmapBox(f, [5, 5, 5], 2, 2, pix, [0, 0, 0], 1);
    // native pixel (15, 15) is pixmap (row 0, column 0), red 0.4: buffer (15, 16), shaded (136, 40, 40)
    expect(at(f, 15, 16)).toEqual([136, 40, 40]);
    // native (16, 15) is pixmap (row 0, column 1), transparent: the background stays black
    expect(at(f, 16, 16)).toEqual([0, 0, 0]);
    // native (15, 16) is pixmap (row 1, column 0), green 0.4: buffer (15, 15)
    expect(at(f, 15, 15)).toEqual([40, 136, 40]);
    // native (16, 16) is pixmap (row 1, column 1), blue 0.4: buffer (16, 15)
    expect(at(f, 16, 15)).toEqual([40, 40, 136]);
  });

  it('scale 2 draws a 4 x 4 pixmap centred on the box centre; a uniform pixmap stays uniform', () => {
    const uni = new Uint8Array(2 * 2 * 3).fill(100);
    expect(scalePixmap(2, 2, uni, 4, 4).every((v) => v === 100)).toBe(true);
    const f = frame();
    drawPixmapBox(f, [5, 5, 5], 2, 2, uni, [-1, -1, -1], 2);
    // native columns 14..17 and rows 14..17 (centre 16 - 2 + j): buffer rows 14..17 and columns 14..17
    expect(at(f, 14, 17)[0]).toBeGreaterThan(0);
    expect(at(f, 17, 14)[0]).toBeGreaterThan(0);
    expect(at(f, 13, 15)).toEqual([0, 0, 0]);
  });
});

describe('engine: a graphics object reaches the image through dump image', () => {
  it('fix graphics/replica display 4 draws a sphere at the atom, with the dump colour of its fix keyword', async () => {
    const { files, logs } = await run(scene('fix rep all graphics/replica 1 display 4.0', 'dump im all image 1 out.*.ppm type type size 32 32 view 90 0 box no 0 fix rep const 0 0\ndump_modify im color mycol 0.2 0.4 0.6 fcolor rep mycol'));
    expect(logs.filter((t) => /error/i.test(t))).toEqual([]);
    const img = decodePpm(b64([...files.values()][0]));
    expect(img.w).toBe(32);
    const o = (15 * img.w + 16) * 3;
    expect([img.rgb[o], img.rgb[o + 1], img.rgb[o + 2]]).toEqual([88, 136, 184]);
  });

  it('fix graphics/objects progbar draws cylinders through dump image (non-empty image)', async () => {
    const { files } = await run(scene(
      'fix pbar all graphics/objects 1 progbar 1 2 y 5 5 5 6 1.0 0.5 3',
      'dump im all image 1 out.*.ppm type type size 32 32 view 90 0 box no 0 fix pbar const 0 0\ndump_modify im color mycol 0.2 0.4 0.6',
    ));
    const img = decodePpm(b64([...files.values()][0]));
    let lit = 0;
    for (let k = 0; k < img.rgb.length; k += 3) if (img.rgb[k] + img.rgb[k + 1] + img.rgb[k + 2] > 0) lit++;
    expect(lit).toBeGreaterThan(20);
  });

  it('fix graphics/labels draws its frame and background as a pixmap', async () => {
    const { files, logs } = await run(scene(
      'fix lab all graphics/labels 1 text "a" 5 5 5 size 8 framecolor 0/102/0 backcolor 0/0/0 transcolor none',
      'dump im all image 1 out.*.ppm type type size 32 32 view 90 0 box no 0 fix lab const 0 0',
    ));
    expect(logs.filter((t) => /error/i.test(t))).toEqual([]);
    const img = decodePpm(b64([...files.values()][0]));
    const lit = [...Array(img.w * img.h)].filter((_, p) => img.rgb[3 * p + 1] > 0).length;
    expect(lit).toBeGreaterThan(20);
  });

  it('fix graphics/chunk without a radius keyword and without a pair style draws with the default radius', async () => {
    // native: no pair style, so no sigma, so the default 0.5 * xlattice (tests/engineGraphicsChunkSigma.test.ts has the
    // native images)
    const { logs } = await run(scene(
      'compute ch all chunk/atom type\nfix hull all graphics/chunk 1 ch alpha 0.0',
      'dump im all image 1 out.*.ppm type type size 32 32 view 70 20 box no 0 fix hull const 1 0',
      'create_atoms 1 single 4.5 5 5\ncreate_atoms 1 single 5.5 5 5',
    ));
    expect(logs.filter((t) => /error/i.test(t))).toEqual([]);
  });

  it('fix graphics/chunk draws the hull of a chunk of atoms through dump image (trinorm)', async () => {
    const { files, logs } = await run(scene(
      'compute ch all chunk/atom type\nfix hull all graphics/chunk 1 ch radius 2.0',
      'dump im all image 1 out.*.ppm type type size 32 32 view 70 20 box no 0 fix hull const 1 0\ndump_modify im color mycol 0.2 0.4 0.6',
      'create_atoms 1 single 4.5 5 5\ncreate_atoms 1 single 5.5 5 5',
    ));
    expect(logs.filter((t) => /error/i.test(t))).toEqual([]);
    const img = decodePpm(b64([...files.values()][0]));
    let lit = 0;
    for (let k = 0; k < img.rgb.length; k += 3) if (img.rgb[k] + img.rgb[k + 1] + img.rgb[k + 2] > 0) lit++;
    // the lens of two radius-2 icosahedra seen from view 70 20 covers a few tens of pixels at 32 x 32
    expect(lit).toBeGreaterThan(15);
  });
});
