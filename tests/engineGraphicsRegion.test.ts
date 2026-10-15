/**
 * dump image "region" keyword for block, prism, sphere, ellipsoid and plane regions, and the points style of any region
 * (dump_image.cpp, the region branch of DumpImage::create_image; src/GRAPHICS/image_objects.cpp PlaneObj). The colours
 * follow the draw functions of image.cpp: a face towards the viewer (camera-space normal (0, 0, 1)) is shaded with the
 * default lights as the tri face test of engineGraphicsLineTri gives: red (255, 40, 40).
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { RasterFrame } from '../src/engine/output/image_objects';
import { drawSphere } from '../src/engine-gpl/graphics/draw';
import { drawEllipsoid } from '../src/engine-gpl/graphics/ellipsoid_mesh';
import { drawBoxRegion, drawEllipsoidRegion, drawPlaneRegion, drawPointsRegion, drawSphereRegion } from '../src/engine-gpl/graphics/region_box';
import { RanMars } from '../src/engine-gpl/random_mars';

/** The camera of the region tests: looking along +x, 32 x 32 px, the 10-unit box centred on (5, 5, 5). */
const frameAlongX = (): RasterFrame => ({
  w: 32,
  h: 32,
  rgb: new Uint8Array(32 * 32 * 3),
  depth: new Float64Array(32 * 32).fill(-Infinity),
  camRight: [0, 1, 0],
  camUp: [0, 0, 1],
  camDir: [1, 0, 0],
  ctr: [5, 5, 5],
  zdist: 20 / Math.tan(Math.PI / 6) + 10,
  pixelWidth: 20 / 32,
});

const at = (f: RasterFrame, col: number, row: number): number[] => {
  const o = (row * f.w + col) * 3;
  return [f.rgb[o], f.rgb[o + 1], f.rgb[o + 2]];
};

const BOX: [number, number, number] = [0, 0, 0];
const BOXHI: [number, number, number] = [10, 10, 10];
const base = { color: [255, 0, 0] as [number, number, number], drawstyle: 'filled' as const, framediam: 0.5, opacity: 1 };

describe('dump image block and prism regions (dump_image.cpp, corner branch)', () => {
  it('shades the face towards the viewer of a filled block with the default lights (exact colour)', () => {
    const f = frameAlongX();
    // the box 4..6 x 2..8 x 2..8: its face at x = 6 faces the viewer and covers the centre pixel (native row 16 = buffer row 15)
    drawBoxRegion(f, { ...base, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [] }, BOX, BOXHI);
    expect(at(f, 16, 15)).toEqual([255, 40, 40]);
  });

  it('skips the face of an open block: the centre pixel is then not the viewer-facing face colour', () => {
    const closed = frameAlongX(), open = frameAlongX();
    drawBoxRegion(closed, { ...base, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [] }, BOX, BOXHI);
    // "open 2" is the xhi face (open_faces[1] in native): its triangles are not drawn, so the centre pixel keeps the
    // depth of the farther face at x = 4 (a smaller depth-buffer value than the x = 6 face)
    drawBoxRegion(open, { ...base, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [2] }, BOX, BOXHI);
    expect(open.depth[15 * 32 + 16]).toBeLessThan(closed.depth[15 * 32 + 16]);
    expect(Number.isFinite(open.depth[15 * 32 + 16])).toBe(true);
  });

  it('draws the frame of a block as cylinders and nothing in the centre', () => {
    const f = frameAlongX();
    // a 0.25 diameter is under one pixel here (0.625 world units per pixel) and draws nothing, as native draw_cylinder does
    drawBoxRegion(f, { ...base, drawstyle: 'frame', framediam: 1, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [] }, BOX, BOXHI);
    expect(f.depth.some((d) => Number.isFinite(d))).toBe(true);
    expect(Number.isFinite(f.depth[15 * 32 + 16])).toBe(false);
  });

  it('draws a prism without tilt exactly as the block of the same bounds', () => {
    const fb = frameAlongX(), fp = frameAlongX();
    drawBoxRegion(fb, { ...base, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [] }, BOX, BOXHI);
    drawBoxRegion(fp, { ...base, kind: 'prism', b: [4, 6, 2, 8, 2, 8, 0, 0, 0], openFaces: [] }, BOX, BOXHI);
    expect(Array.from(fp.rgb)).toEqual(Array.from(fb.rgb));
    expect(Array.from(fp.depth)).toEqual(Array.from(fb.depth));
  });

  it('a transparent block at opacity 0 draws nothing', () => {
    const f = frameAlongX();
    drawBoxRegion(f, { ...base, drawstyle: 'transparent', opacity: 0, kind: 'block', b: [4, 6, 2, 8, 2, 8], openFaces: [] }, BOX, BOXHI);
    expect(f.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('dump image sphere and ellipsoid regions (dump_image.cpp, sphere and ellipsoid branches)', () => {
  it('draws a filled sphere region exactly as a sphere of diameter 2 radius', () => {
    const fa = frameAlongX(), fb = frameAlongX();
    drawSphereRegion(fa, { ...base, center: [5, 5, 5], radius: 2 });
    drawSphere(fb, [5, 5, 5], [1, 0, 0], 4, 1);
    expect(Array.from(fa.rgb)).toEqual(Array.from(fb.rgb));
    expect(Array.from(fa.depth)).toEqual(Array.from(fb.depth));
  });

  it('draws a frame sphere region as the level 4 wireframe of the ellipsoid mesh', () => {
    const fa = frameAlongX(), fb = frameAlongX();
    drawSphereRegion(fa, { ...base, drawstyle: 'frame', framediam: 0.2, center: [5, 5, 5], radius: 2 });
    drawEllipsoid(fb, 2, [1, 0, 0], [5, 5, 5], [2, 2, 2], [1, 0, 0, 0], null, 0.2, 1, 4);
    expect(Array.from(fa.rgb)).toEqual(Array.from(fb.rgb));
  });

  it('draws a filled ellipsoid region as the level 5 surface and a transparent one at opacity 0 not at all', () => {
    const fa = frameAlongX(), fb = frameAlongX(), fc = frameAlongX();
    drawEllipsoidRegion(fa, { ...base, center: [5, 5, 5], shape: [3, 2, 1] });
    drawEllipsoid(fb, 1, [1, 0, 0], [5, 5, 5], [3, 2, 1], [1, 0, 0, 0], null, 0.5, 1, 5);
    expect(Array.from(fa.rgb)).toEqual(Array.from(fb.rgb));
    drawEllipsoidRegion(fc, { ...base, drawstyle: 'transparent', opacity: 0, center: [5, 5, 5], shape: [3, 2, 1] });
    expect(fc.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('dump image plane regions (PlaneObj, dump_image.cpp plane branch)', () => {
  it('shades a filled plane facing the viewer with the default lights (exact colour at the centre)', () => {
    const f = frameAlongX();
    // the plane x = 5 with normal +x: its triangles face the viewer and share the centre vertex (5, 5, 5)
    drawPlaneRegion(f, { ...base, point: [5, 5, 5], normal: [1, 0, 0], scale: Math.hypot(10, 10, 10) }, BOX, BOXHI);
    expect(at(f, 16, 15)).toEqual([255, 40, 40]);
  });

  it('draws nothing of a plane outside the box', () => {
    const f = frameAlongX();
    drawPlaneRegion(f, { ...base, point: [50, 50, 50], normal: [1, 0, 0], scale: Math.hypot(10, 10, 10) }, BOX, BOXHI);
    expect(f.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('dump image points style (dump_image.cpp, POINTS branch, RanMars seed 19660405)', () => {
  it('draws each point inside the region as a sphere at the RanMars position (three draws per point)', () => {
    const fa = frameAlongX(), fb = frameAlongX();
    drawPointsRegion(fa, { ...base, npoints: 3, framediam: 1 }, BOX, BOXHI, () => true);
    const rand = new RanMars(19660405);
    for (let i = 0; i < 3; i++) {
      const p: [number, number, number] = [rand.uniform() * 10, rand.uniform() * 10, rand.uniform() * 10];
      drawSphere(fb, p, [1, 0, 0], 1, 1);
    }
    expect(Array.from(fa.rgb)).toEqual(Array.from(fb.rgb));
  });

  it('keeps only the points the region matches', () => {
    const f = frameAlongX();
    drawPointsRegion(f, { ...base, npoints: 50, framediam: 1 }, BOX, BOXHI, () => false);
    expect(f.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('dump image region keywords in a script', () => {
  const run = async (dump: string): Promise<void> => {
    const s = new Session({ emit: (_e: EngineEvent) => undefined, writeFile: () => undefined });
    await s.execute(`units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 5.0 5.0
mass * 1.0
region blk block 2 8 2 8 2 8
region sph sphere 5 5 5 3
region pl plane 5 5 5 1 0 0
region pri prism 0 10 0 10 0 10 0 0 0
${dump}
run 0
`);
  };
  const image = (regions: string): string => `dump im all image 1 out.*.ppm type type size 40 40 ${regions}`;

  it('accepts block, sphere, plane, prism and points regions', async () => {
    await expect(run(image('region blk red filled region sph blue transparent 0.5 region pl green frame 0.2 region pri white points 20 0.5'))).resolves.toBeUndefined();
  });

  it('names the styles it still refuses: hull_points, moving regions and compound regions', async () => {
    await expect(run(image('region blk red filled hull_points 50'))).rejects.toThrow(/hull_points/);
    await expect(run(image('region blk red points 0 0.5'))).rejects.toThrow(/number of points/);
    await expect(run('variable zero equal 0\nregion mv block 2 8 2 8 2 8 move v_zero v_zero v_zero\n' + image('region mv red filled'))).rejects.toThrow(/move or rotate/);
    await expect(run('region u union 2 blk sph\n' + image('region u red filled'))).rejects.toThrow(/not supported/);
  });
});
