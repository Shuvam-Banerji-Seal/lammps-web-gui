/**
 * dump image keywords line and tri (dump_image.cpp lineflag and triflag branches), the transparency test of dump_modify
 * ftrans and the transparency mask of Image's draw loops (image.cpp: a pixel is skipped when opacity < 1 and
 * transthresh[ix % 16][iy % 16] > opacity). The colours are worked out from the C++ by hand:
 *
 *   light (draw_pixel, metallic 0, gamma 1, shiny 1): for a camera-space normal n and colour c,
 *     c_k = c_k * ambient + c_k * key * sat(n.key) + key * pow(sat(n.half), 16) + c_k * fill * sat(n.fill) + c_k * back * sat(n.back)
 *   with the image defaults (key 0.9, fill 0.45, back 0.9, ambient 0) or the tri face lights of
 *   dump_image.cpp (reset_lighting(image, 0.3, 0.8, 0.45, 0.8)). A face towards the viewer, n = (0, 0, 1), has
 *   sat(n.key) = 0.61237, sat(n.fill) = 0.86603, sat(n.back) = 0 and pow(0.89788, 16) = 0.17844 (computed with node).
 *   Red (1, 0, 0) under the default lights is (255, 40, 40); under the tri face lights it is (255, 36, 36).
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { RasterFrame } from '../src/engine/output/image_objects';
import { drawSphere, drawTriangle } from '../src/engine-gpl/graphics/draw';
import { drawLineAtoms, drawTriAtoms } from '../src/engine-gpl/graphics/atom_objects';
import { TRANSTHRESH } from '../src/engine-gpl/graphics/transthresh';
import { drawConeRegion } from '../src/engine-gpl/graphics/region_cone';
import { drawEllipsoid, icosphereMesh } from '../src/engine-gpl/graphics/ellipsoid_mesh';
import { bodyElements, drawBodyAtoms } from '../src/engine-gpl/graphics/body_image';
import type { BodyAtom } from '../src/engine/output/image_objects';
import type { RegionConeSpec } from '../src/engine/output/image_objects';

/** The camera of the stage 2 render tests: looking along +x, 32 x 32 px, the 10-unit cube centred on (5, 5, 5). */
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

/** The same camera looking along +z: the xy plane is seen face on (the 2d line particles of atom_style line). */
const frameAlongZ = (w = 32, h = 32): RasterFrame => ({
  ...frameAlongX(w, h),
  rgb: new Uint8Array(w * h * 3),
  depth: new Float64Array(w * h).fill(-Infinity),
  camRight: [1, 0, 0],
  camUp: [0, 1, 0],
  camDir: [0, 0, 1],
});

/** RGB of frame pixel (col, row), row 0 at the top. */
const at = (f: RasterFrame, col: number, row: number): number[] => {
  const o = (row * f.w + col) * 3;
  return [f.rgb[o], f.rgb[o + 1], f.rgb[o + 2]];
};

const drawnAt = (f: RasterFrame, col: number, row: number): boolean => Number.isFinite(f.depth[row * f.w + col]);

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

describe('dump image tri faces (draw_triangle under the tri lights)', () => {
  it('shades a face towards the viewer with the tri face lights (exact colour)', () => {
    const f = frameAlongX();
    // corners relative to the box centre (5, 5, 5): (right, up) = (4, -4), (-4, -4), (0, 4); normal = +x = camDir
    const x: [number, number, number] = [5, 9, 1], y: [number, number, number] = [5, 1, 1], z: [number, number, number] = [5, 5, 9];
    drawTriangle(f, x, y, z, [1, 0, 0], 1, { ambient: 0.3, lights: { key: 0.8, fill: 0.45, back: 0.8 } });
    // native row 16 is buffer row 15; the centre pixel lies inside the face
    expect(at(f, 16, 15)).toEqual([255, 36, 36]);
  });

  it('shades the same face with the default image lights when no tri lights are given', () => {
    const f = frameAlongX();
    const x: [number, number, number] = [5, 9, 1], y: [number, number, number] = [5, 1, 1], z: [number, number, number] = [5, 5, 9];
    drawTriangle(f, x, y, z, [1, 0, 0]);
    expect(at(f, 16, 15)).toEqual([255, 40, 40]);
  });

  it('draws nothing for an opacity of zero', () => {
    const f = frameAlongX();
    drawTriangle(f, [5, 9, 1], [5, 1, 1], [5, 5, 9], [1, 0, 0], 0);
    expect(f.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('dump image transparency (image.cpp transthresh mask)', () => {
  it('drops exactly the pixels whose threshold is above the opacity', () => {
    const solid = frameAlongX();
    drawSphere(solid, [5, 5, 5], [1, 0, 0], 12);
    const half = frameAlongX();
    drawSphere(half, [5, 5, 5], [1, 0, 0], 12, 0.5);
    let kept = 0, dropped = 0;
    for (let row = 0; row < 32; row++) {
      for (let col = 0; col < 32; col++) {
        // native row iy = h - 1 - row counts from the bottom
        const iy = 31 - row;
        const shouldDraw = drawnAt(solid, col, row) && !(TRANSTHRESH[col % 16][iy % 16] > 0.5);
        expect(drawnAt(half, col, row)).toBe(shouldDraw);
        if (drawnAt(solid, col, row)) (shouldDraw ? kept++ : dropped++);
      }
    }
    // the mask removes a real share of the sphere, and not all of it
    expect(dropped).toBeGreaterThan(0);
    expect(kept).toBeGreaterThan(0);
  });

  it('keeps an opaque sphere when the opacity is 1', () => {
    const f = frameAlongX();
    drawSphere(f, [5, 5, 5], [1, 0, 0], 12, 1);
    expect(at(f, 16, 15)).toEqual([255, 40, 40]);
  });
});

describe('atom_style line segments (dump_image.cpp lineflag branch)', () => {
  it('draws a segment as a cylinder with both caps along its length', () => {
    const f = frameAlongZ();
    // a segment of length 4 centred on (5, 5, 0), along +x: end points (3, 5, 0) and (7, 5, 0), width 1
    drawLineAtoms(f, [{ x: 5, y: 5, length: 4, theta: 0, color: [255, 0, 0], opacity: 1 }], 1);
    // the centre row (native row 16 = buffer row 15) is drawn from the left end to the right end ...
    // (half the length is 2 units = 3.2 px, so columns 14 and 18 lie on the segment)
    expect(drawnAt(f, 16 - 2, 15)).toBe(true);
    expect(drawnAt(f, 16 + 2, 15)).toBe(true);
    // ... and the cylinder axis pixel at the centre of the segment has the surface normal towards the viewer
    expect(at(f, 16, 15)).toEqual([255, 40, 40]);
    // far from the segment nothing is drawn
    expect(drawnAt(f, 1, 1)).toBe(false);
  });

  it('draws nothing for a zero width', () => {
    const f = frameAlongZ();
    drawLineAtoms(f, [{ x: 5, y: 5, length: 4, theta: 0, color: [255, 0, 0], opacity: 1 }], 0);
    expect(f.depth.every((d) => !Number.isFinite(d))).toBe(true);
  });
});

describe('atom_style tri particles (dump_image.cpp triflag branch)', () => {
  // an unrotated particle (quaternion 1 0 0 0) with corners at (+2, -2, 0), (-2, -2, 0) and (0, 2, 0) in its frame
  const corners = [2, -2, 0, -2, -2, 0, 0, 2, 0];
  const quat = [1, 0, 0, 0];

  it('draws faces only for tstyle 1 and edges only for tstyle 2', () => {
    const faces = frameAlongZ();
    drawTriAtoms(faces, [{ center: [5, 5, 0], quat, corners, color: [255, 0, 0], opacity: 1 }], 1, 0.2);
    expect(faces.depth.some((d) => Number.isFinite(d))).toBe(true);
    const edges = frameAlongZ();
    drawTriAtoms(edges, [{ center: [5, 5, 0], quat, corners, color: [255, 0, 0], opacity: 1 }], 2, 0.2);
    // the centroid of the triangle is inside the face but not on an edge: only the edges are drawn there
    expect(drawnAt(edges, 16, 15)).toBe(false);
    expect(edges.depth.some((d) => Number.isFinite(d))).toBe(true);
  });

  it('draws the face of an unrotated particle with the tri face lights', () => {
    const f = frameAlongZ();
    drawTriAtoms(f, [{ center: [5, 5, 0], quat, corners, color: [255, 0, 0], opacity: 1 }], 1, 0.2);
    // the face normal is +z (towards the viewer) and the tri face lights give (255, 36, 36) at the centroid
    expect(at(f, 16, 15)).toEqual([255, 36, 36]);
  });
});

describe('dump image region cylinders and cones (dump_image.cpp cylinder and cone branch, ConeObj)', () => {
  // a cylinder of radius 3 on the axis z through (5, 5), from z = 2 to 8 in the box 0 .. 10; gray = 128 / 255
  const spec = (drawstyle: RegionConeSpec['drawstyle']): RegionConeSpec => ({
    style: 'cylinder', axis: 2, c1: 5, c2: 5, radlo: 3, radhi: 3, lo: 2, hi: 8,
    color: [128, 128, 128], drawstyle, framediam: 0.5, opacity: 1,
  });

  it('shades the top cap of a filled cylinder facing the viewer (normal (0, 0, 1), default lights)', () => {
    const f = frameAlongZ();
    drawConeRegion(f, spec('filled'), [0, 0, 0], [10, 10, 10]);
    // the cap at z = 8 is nearest the camera (looking along +z); its normal is the axis (0, 0, 1): 161 per channel
    expect(at(f, 16, 15)).toEqual([161, 161, 161]);
  });

  it('draws only the wireframe for a frame region, so the centre stays empty', () => {
    const f = frameAlongZ();
    drawConeRegion(f, spec('frame'), [0, 0, 0], [10, 10, 10]);
    expect(drawnAt(f, 16, 15)).toBe(false);
    expect(f.depth.some((d) => Number.isFinite(d))).toBe(true);
  });

  it('clips the cylinder to the box along its axis', () => {
    const f = frameAlongZ();
    // the box ends at z = 5 (the top cap then sits at z = 5, not 8): the centre pixel is still the top cap, shaded the same
    drawConeRegion(f, spec('filled'), [0, 0, 0], [10, 10, 5]);
    expect(at(f, 16, 15)).toEqual([161, 161, 161]);
  });
});

describe('dump image ellipsoid meshes (image_objects.cpp EllipsoidObj)', () => {
  it('has 20 * 4^(level - 1) triangles, all on the unit sphere', () => {
    for (const level of [1, 2, 3]) {
      const m = icosphereMesh(level);
      expect(m.length).toBe(20 * 4 ** (level - 1));
      for (const t of m) for (const v of t) expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 9);
    }
  });

  it('draws an ellipsoid with equal half-axes and no block as a sphere', () => {
    const a = frameAlongZ(64, 64), b = frameAlongZ(64, 64);
    drawEllipsoid(a, 1, [1, 0, 0], [5, 5, 5], [2, 2, 2], [1, 0, 0, 0], null, 0, 1, 4);
    drawSphere(b, [5, 5, 5], [1, 0, 0], 4);
    expect(a.rgb).toEqual(b.rgb);
  });

  it('gives a superellipsoid with block exponents 2 and 2 the same picture as the plain ellipsoid', () => {
    // with n = e = 2 the superellipsoid scale and normal reduce to the ellipsoid ones (up to rounding of the normals)
    const a = frameAlongZ(96, 96), b = frameAlongZ(96, 96);
    drawEllipsoid(a, 1, [0.8, 0.4, 0.2], [5, 5, 5], [3, 2, 1.5], [1, 0, 0, 0], null, 0, 1, 4);
    drawEllipsoid(b, 1, [0.8, 0.4, 0.2], [5, 5, 5], [3, 2, 1.5], [1, 0, 0, 0], [2, 2], 0, 1, 4);
    let worst = 0;
    for (let i = 0; i < a.rgb.length; i++) worst = Math.max(worst, Math.abs(a.rgb[i] - b.rgb[i]));
    expect(worst).toBeLessThanOrEqual(1);
    expect(a.rgb.some((v) => v > 0)).toBe(true);
  });

  it('covers the projected area of an oblate ellipsoid (half-axes 3 and 2 seen along z)', () => {
    // 128 x 128 px with pixelWidth 20 / 128: the ellipse of half-axes 3 and 2 covers pi * 3 * 2 / pw^2 pixels
    const f = frameAlongZ(128, 128);
    drawEllipsoid(f, 1, [1, 1, 1], [5, 5, 5], [3, 2, 1.5], [1, 0, 0, 0], null, 0, 1, 5);
    const drawn = f.depth.filter((d) => Number.isFinite(d)).length;
    const expected = (Math.PI * 3 * 2) / (20 / 128) ** 2;
    expect(Math.abs(drawn - expected) / expected).toBeLessThan(0.03);
  });
});

describe('dump image body particles (BodyRoundedPolyhedron::image)', () => {
  const identity = [1, 0, 0, 0];
  const body = (nsub: number, vertices: number[], edges: number[], faces: number[], nedge: number, nface: number, radius = 0.25): BodyAtom => ({
    style: 'rounded/polyhedron', center: [5, 5, 5] as [number, number, number], quat: identity, nsub, nedge, nface, vertices, edges, faces, radius, color: [255, 0, 0] as [number, number, number], opacity: 1,
  });

  it('draws a one-vertex body as a sphere of the flag1 diameter at its vertex', () => {
    const els = bodyElements(body(1, [1, 0, 0], [], [], 0, 0), 0.2, 3);
    expect(els).toEqual([{ kind: 'sphere', c: [6, 5, 5], d: 0.2 }]);
  });

  it('uses 2 * the rounded radius for the diameter when flag1 is 0', () => {
    const els = bodyElements(body(1, [0, 0, 0], [], [], 0, 0, 0.25), 0, 3);
    expect(els).toEqual([{ kind: 'sphere', c: [5, 5, 5], d: 0.5 }]);
  });

  it('makes flat triangles for flag2 4 and no edges', () => {
    const els = bodyElements(body(3, [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 2, 2, 0], [0, 1, 2, -1], 3, 1), 0.5, 4);
    expect(els.map((e) => e.kind)).toEqual(['tri']);
    const tri = els[0] as { kind: 'tri'; p: number[][] };
    expect(tri.p).toEqual([[5, 5, 5], [6, 5, 5], [5, 6, 5]]);
  });

  it('draws edges and flat triangles pushed out by half the diameter for flag2 3', () => {
    const els = bodyElements(body(3, [0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 1, 2, 2, 0], [0, 1, 2, -1], 3, 1), 0.5, 3);
    expect(els.filter((e) => e.kind === 'line')).toHaveLength(3);
    const tri = els.find((e) => e.kind === 'tri') as { kind: 'tri'; p: number[][] };
    // face centre (1/3, 1/3, 0) relative to the body centre, scaled to 0.25: a shift of 0.25 (1, 1, 0) / sqrt 2 ... / |(1/3, 1/3, 0)|
    const s = 0.25 / Math.hypot(1 / 3, 1 / 3);
    expect(tri.p[0][0]).toBeCloseTo(5 + (s * 1) / 3, 9);
    expect(tri.p[0][1]).toBeCloseTo(5 + (s * 1) / 3, 9);
    expect(tri.p[0][2]).toBeCloseTo(5, 9);
  });

  it('splits a quadrilateral face into two triangles', () => {
    const els = bodyElements(body(4, [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], [], [0, 1, 2, 3], 0, 1), 0.5, 4);
    expect(els.map((e) => e.kind)).toEqual(['tri', 'tri']);
    expect((els[0] as { p: number[][] }).p).toEqual([[5, 5, 5], [6, 5, 5], [6, 6, 5]]);
    expect((els[1] as { p: number[][] }).p).toEqual([[6, 6, 5], [5, 6, 5], [5, 5, 5]]);
  });

  it('draws a one-vertex body exactly like a sphere', () => {
    const a = frameAlongZ(64, 64), b = frameAlongZ(64, 64);
    drawBodyAtoms(a, [body(1, [0, 0, 0], [], [], 0, 0, 1)], 2, 3);
    drawSphere(b, [5, 5, 5], [1, 0, 0], 2);
    expect(a.rgb).toEqual(b.rgb);
  });
});

describe('dump image line, tri and ftrans keywords (session)', () => {
  const lineData = `line test

2 atoms
1 atom types
1 lines

0 10 xlo xhi
0 10 ylo yhi
-5 5 zlo zhi

Atoms # line

1 1 1 1 1 5 5 0 0 0 0
2 1 1 0 1 2 2 0 0 0 0

Lines

1 3 5 7 5
`;

  const run = async (dump: string): Promise<Map<string, string>> => {
    const files = new Map<string, string>();
    const events: EngineEvent[] = [];
    const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
    s.addFile('line.data', lineData);
    await s.execute(`units lj\ndimension 2\natom_style line\nread_data line.data\npair_style lj/cut 2.5\npair_coeff * * 1 1\n${dump}\nrun 0\n`);
    const errors = events.filter((e) => e.kind === 'error');
    expect(errors).toEqual([]);
    return files;
  };

  it('writes an image with the line keyword and the line particle drawn across the box', async () => {
    const files = await run('dump d all image 1 line-*.ppm type type size 32 32 zoom 1 view 0 0 box no 0.025 line type 1.0');
    const img = decodePpm(b64(files.get('line-0.ppm')!));
    expect(img.w).toBe(32);
    // count drawn (non-background) pixels: the cylinder is wide and the particle is not a sphere
    let lit = 0;
    for (let p = 0; p < img.w * img.h; p++) if (img.rgb[3 * p] || img.rgb[3 * p + 1] || img.rgb[3 * p + 2]) lit++;
    expect(lit).toBeGreaterThan(0);
  });

  it('draws the atoms as spheres without the line keyword and differs from the line image', async () => {
    const plain = await run('dump d all image 1 plain-*.ppm type type size 32 32 zoom 1 view 0 0 box no 0.025');
    const withLine = await run('dump d all image 1 line-*.ppm type type size 32 32 zoom 1 view 0 0 box no 0.025 line type 1.0');
    expect(plain.get('plain-0.ppm')).not.toEqual(withLine.get('line-0.ppm'));
  });

  it('accepts the tri keyword and refuses an unknown colour style', async () => {
    const tri = await run('dump d all image 1 tri-*.ppm type type size 16 16 tri type 1 0.2 box no 0.025');
    expect(tri.get('tri-0.ppm')).toBeDefined();
    await expect(run('dump d all image 1 bad-*.ppm type type size 16 16 line element 1.0')).rejects.toThrow(/only supports color by type/);
  });

  it('applies dump_modify ftrans to the objects of a fix (opacity 0 hides them, 1 draws them)', async () => {
    const none = await run('dump d all image 1 img-*.ppm type type size 32 32 box no 0.025 zoom 1 view 0 0');
    const fixed = (t: string): Promise<Map<string, string>> => run(
      'fix o all graphics/objects 1 arrow 1 3 5 0 7 5 0 0.3 0.2\n' +
      'dump d all image 1 img-*.ppm type type size 32 32 box no 0.025 zoom 1 view 0 0 fix o const 0 0\n' +
      'dump_modify d ftrans o ' + t);
    const hidden = await fixed('0'), shown = await fixed('1');
    expect(hidden.get('img-0.ppm')).toEqual(none.get('img-0.ppm'));
    expect(shown.get('img-0.ppm')).not.toEqual(none.get('img-0.ppm'));
  });

  it('accepts the continuous amap grammar with its delta word (dump_image.html: amap lo hi style delta N entries)', async () => {
    // the line of GRAPHICS/in.obstacle-lines: "amap 0 8 cf 0.0 5 min blue 0.25 purple 0.5 red 0.75 orange max gold"
    const files = await run('dump d all image 1 amap-*.ppm type type size 16 16 box no 0.025 zoom 1 view 0 0\n' +
      'dump_modify d amap 0 8 cf 0.0 5 min blue 0.25 purple 0.5 red 0.75 orange max gold');
    expect(files.get('amap-0.ppm')).toBeDefined();
  });

  it('draws atom_style ellipsoid particles with the ellipsoid keyword and refuses a bad refinement level', async () => {
    const head = 'units lj\natom_style ellipsoid\nregion box block 0 10 0 10 0 10\ncreate_box 1 box\ncreate_atoms 1 single 5 5 5\nset type 1 shape 2 2 1\npair_style lj/cut 2.5\npair_coeff * * 1 1\n';
    const withEll = async (dump: string): Promise<Map<string, string>> => {
      const files = new Map<string, string>();
      const events: EngineEvent[] = [];
      const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
      await s.execute(head + dump + '\nrun 0\n');
      expect(events.filter((e) => e.kind === 'error')).toEqual([]);
      return files;
    };
    const files = await withEll('dump d all image 1 ell-*.ppm type type size 64 64 zoom 1 view 0 0 box no 0.025 ellipsoid type 1 4 0.2');
    const img = decodePpm(b64(files.get('ell-0.ppm')!));
    let lit = 0;
    for (let p = 0; p < img.w * img.h; p++) if (img.rgb[3 * p] || img.rgb[3 * p + 1] || img.rgb[3 * p + 2]) lit++;
    expect(lit).toBeGreaterThan(0);
    await expect(withEll('dump d all image 1 ell-*.ppm type type size 16 16 ellipsoid type 1 7 0.2')).rejects.toThrow(/refinement level/);
  });

  it('refuses ftrans for a fix that is not in the dump, and a transparency outside 0..1', async () => {
    await expect(run('dump d all image 1 ftr-*.ppm type type size 16 16 box no 0.025\ndump_modify d ftrans nofix 0.5')).rejects.toThrow(/nofix/);
    await expect(run('dump d all image 1 ftr-*.ppm type type size 16 16 box no 0.025\ndump_modify d ftrans nofix 2')).rejects.toThrow(/transparency/);
  });
});
