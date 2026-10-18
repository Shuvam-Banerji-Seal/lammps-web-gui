/**
 * dump image "body" keyword for the body styles nparticle (BodyNparticle::image) and rounded/polygon (BodyRoundedPolygon::image)
 * in src/engine-gpl/graphics/body_image.ts, next to rounded/polyhedron. The element lists are worked out from the C++:
 *   nparticle: one sphere per sub-particle at quat * displacement + centre, diameter flag1 (dump image's first body value);
 *   rounded/polygon: vertices in the plane z = 0; one vertex is a sphere; two vertices give one rod from vertex 0 to
 *   itself; otherwise lines between neighbours (edges on unless flag2 = 1) and a fan of triangles from the centroid
 *   (triangles on unless flag2 = 2) whose outer corners and centroid have z = 0.5 * diameter when edges are drawn.
 *   The diameter is flag1, or 2 * the stored rounded radius when flag1 is 0 or less.
 */
import { describe, expect, it } from 'vitest';
import type { BodyAtom, RasterFrame } from '../src/engine/output/image_objects';
import { bodyElements, drawBodyAtoms } from '../src/engine-gpl/graphics/body_image';
import { drawSphere } from '../src/engine-gpl/graphics/draw';

const identity = [1, 0, 0, 0];
const body = (over: Partial<BodyAtom>): BodyAtom => ({
  style: 'nparticle', center: [5, 5, 5], quat: identity, nsub: 0, nedge: 0, nface: 0, vertices: [], edges: [], faces: [], radius: 0,
  color: [1, 0, 0], opacity: 1, ...over,
});

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

describe('dump image body style nparticle (BodyNparticle::image)', () => {
  it('draws one sphere per sub-particle at its displacement from the centre, with diameter flag1', () => {
    const els = bodyElements(body({ nsub: 2, vertices: [1, 0, 0, 0, 2, 0] }), 0.5, 0);
    expect(els).toEqual([{ kind: 'sphere', c: [6, 5, 5], d: 0.5 }, { kind: 'sphere', c: [5, 7, 5], d: 0.5 }]);
  });

  it('rotates the displacements by the body orientation (90 degrees about z)', () => {
    const q = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];
    const [e] = bodyElements(body({ nsub: 1, vertices: [1, 0, 0], quat: q }), 0.5, 0);
    expect(e.kind).toBe('sphere');
    if (e.kind !== 'sphere') return;
    expect(e.c[0]).toBeCloseTo(5, 12);
    expect(e.c[1]).toBeCloseTo(6, 12);
    expect(e.c[2]).toBeCloseTo(5, 12);
  });

  it('draws the spheres exactly as drawSphere does for each sub-particle', () => {
    const atom = body({ nsub: 2, vertices: [1, 0, 0, 0, 2, 0] });
    const fa = frameAlongX(), fb = frameAlongX();
    drawBodyAtoms(fa, [atom], 1, 0);
    drawSphere(fb, [6, 5, 5], [1, 0, 0], 1, 1);
    drawSphere(fb, [5, 7, 5], [1, 0, 0], 1, 1);
    expect(Array.from(fa.rgb)).toEqual(Array.from(fb.rgb));
  });
});

describe('dump image body style rounded/polygon (BodyRoundedPolygon::image)', () => {
  it('draws a one-vertex body as a sphere in the plane z = 0 with diameter 2 * radius when flag1 is 0', () => {
    const els = bodyElements(body({ style: 'rounded/polygon', nsub: 1, vertices: [1, 0, 0], radius: 0.25 }), 0, 3);
    expect(els).toEqual([{ kind: 'sphere', c: [6, 5, 0], d: 0.5 }]);
  });

  it('draws a two-vertex body as one rod from vertex 0 to itself (native n = 1 wrap)', () => {
    const els = bodyElements(body({ style: 'rounded/polygon', nsub: 2, vertices: [1, 0, 0, -1, 0, 0] }), 0.2, 3);
    expect(els).toEqual([{ kind: 'line', p1: [6, 5, 0], p2: [6, 5, 0], d: 0.2 }]);
  });

  it('draws the edges and the fan of triangles with the z = 0.5 diameter corners (flag2 3)', () => {
    const tri = body({ style: 'rounded/polygon', nsub: 3, vertices: [1, 0, 0, 0, 1, 0, -1, 0, 0], radius: 0.25 });
    const els = bodyElements(tri, 0, 3);
    // diameter 2 * 0.25 = 0.5, so the outer corners and the centroid have z = 0.25
    const lines = els.filter((e) => e.kind === 'line');
    const tris = els.filter((e) => e.kind === 'tri');
    expect(lines).toEqual([
      { kind: 'line', p1: [6, 5, 0], p2: [5, 6, 0], d: 0.5 },
      { kind: 'line', p1: [5, 6, 0], p2: [4, 5, 0], d: 0.5 },
      { kind: 'line', p1: [4, 5, 0], p2: [6, 5, 0], d: 0.5 },
    ]);
    expect(tris.length).toBe(3);
    // the first triangle is (outer 0, centroid, outer 1); the centroid of (6, 5), (5, 6), (4, 5) is (5, 16/3)
    const first = tris[0];
    if (first.kind !== 'tri') throw new Error('expected a flat triangle');
    expect(first.p[0]).toEqual([6, 5, 0.25]);
    expect(first.p[1][0]).toBeCloseTo(5, 12);
    expect(first.p[1][1]).toBeCloseTo(16 / 3, 12);
    expect(first.p[1][2]).toBeCloseTo(0.25, 12);
    expect(first.p[2]).toEqual([5, 6, 0.25]);
  });

  it('draws only the triangles for flag2 1 (edges off, centroid and corners at z = 0)', () => {
    const els = bodyElements(body({ style: 'rounded/polygon', nsub: 3, vertices: [1, 0, 0, 0, 1, 0, -1, 0, 0], radius: 0.25 }), 0.2, 1);
    expect(els.every((e) => e.kind === 'tri')).toBe(true);
    expect(els.length).toBe(3);
  });

  it('draws only the edges for flag2 2 (triangles off)', () => {
    const els = bodyElements(body({ style: 'rounded/polygon', nsub: 3, vertices: [1, 0, 0, 0, 1, 0, -1, 0, 0], radius: 0.25 }), 0.2, 2);
    expect(els.every((e) => e.kind === 'line')).toBe(true);
    expect(els.length).toBe(3);
  });
});
