/**
 * Object generation of the GRAPHICS fixes ported for the engine (graphics/labels, periodic, chunk, replica and the
 * progbar of graphics/objects). These tests check the object lists (count, coordinates, sizes, colours) against
 * values computed by hand from the LAMMPS formulas; the renderer is not involved.
 */
import { describe, expect, it } from 'vitest';
import { ScalableFont, SsfnError, createLabel, sansFontBytes } from '../src/engine-gpl/graphics/scalable_font';
import {
  FixGraphicsLabels,
  parseColor,
  parseLabelArgs,
  type LabelEnv,
} from '../src/engine-gpl/graphics/fix_graphics_labels';
import { parsePeriodicArgs, periodicObjects, type PeriodicAtom } from '../src/engine-gpl/graphics/fix_graphics_periodic';
import { parseReplicaArgs, replicaObjects, type ReplicaAtom } from '../src/engine-gpl/graphics/fix_graphics_replica';
import { FixGraphicsChunk, parseChunkArgs, type ChunkAtom } from '../src/engine-gpl/graphics/fix_graphics_chunk';
import { buildConvexHull } from '../src/engine-gpl/graphics/convex_hull';
import { parseProgbar, progbarObjects } from '../src/engine-gpl/graphics/progbar';
import type { GraphicsObject, Vec3 } from '../src/engine-gpl/graphics/objects';

const noImage = () => {
  throw new Error('no image in this test');
};
const env = (vars: Record<string, number> = {}, subst: (t: string) => string = (t) => t): LabelEnv => ({
  variable: (name) => {
    if (!(name in vars)) throw new Error(`no variable ${name}`);
    return vars[name];
  },
  substitute: subst,
});

describe('scalable font (src/GRAPHICS/scalable_font.cpp, VeraR.sfn)', () => {
  it('the bundled font is the 20847-byte SSFN file of scalable_sans_font.h', () => {
    const F = sansFontBytes();
    expect(F.length).toBe(20847);
    expect(String.fromCharCode(F[0], F[1], F[2], F[3])).toBe('SSFN');
    expect(String.fromCharCode(F[F.length - 4], F[F.length - 3], F[F.length - 2], F[F.length - 1])).toBe('NFSS');
    expect(sansFontBytes()).toBe(F);
  });

  it('accepts sizes 8..255 only, as _ssfn_select does', () => {
    const f = new ScalableFont();
    expect(() => f.selectFont(7)).toThrow(SsfnError);
    expect(() => f.selectFont(256)).toThrow(SsfnError);
    expect(() => f.selectFont(8)).not.toThrow();
    expect(() => f.selectFont(255)).not.toThrow();
  });

  it('bitmap-mode glyph height: 16 px gives 25 px (65536 * 16 / ((187 - 26) << 8) = 25.4)', () => {
    const f = new ScalableFont();
    f.selectFont(16);
    const l = f.glyph('l');
    expect(l.height).toBe(25);
    // the stem of an l stands on the baseline: its lowest point is the baseline row (+-1 for rounding)
    const ys = l.contours.flat().map((p) => p[1]);
    expect(Math.max(...ys)).toBeGreaterThanOrEqual(l.baseline - 1);
    expect(Math.max(...ys)).toBeLessThanOrEqual(l.baseline + 1);
    expect(l.contours.length).toBe(1);
    expect(l.contours[0].length).toBe(5);
  });

  it('a glyph with no outline (space) has an advance but no strokes', () => {
    const f = new ScalableFont();
    f.selectFont(16);
    const sp = f.glyph(' ');
    expect(sp.contours).toEqual([]);
    expect(sp.advX).toBeGreaterThan(0);
  });

  it('throws for a character the font does not have', () => {
    const f = new ScalableFont();
    f.selectFont(16);
    expect(() => f.glyph('中')).toThrow(SsfnError);
  });
});

describe('label layout (ScalableFont::create_label with strokes)', () => {
  it('width is the sum of the advances of the text plus "gll"; vertical swaps width and height', () => {
    const f = new ScalableFont();
    f.selectFont(16);
    const adv = (c: string) => f.glyph(c).advX;
    const h = createLabel(f, 'ab', true);
    expect(h.width).toBe(adv('a') + adv('b') + adv('g') + 2 * adv('l'));
    const v = createLabel(f, 'ab', false);
    expect(v.width).toBe(h.height);
    expect(v.height).toBe(h.width);
  });

  it('pen starts at 2 * xspace and every stroke point is inside the label', () => {
    const f = new ScalableFont();
    f.selectFont(16);
    const xspace = f.glyph(' ').advX;
    const h = createLabel(f, 'l', true);
    const xs = h.strokes.filter((_, i) => i % 2 === 0);
    const ys = h.strokes.filter((_, i) => i % 2 === 1);
    expect(Math.min(...xs)).toBeGreaterThanOrEqual(2 * xspace);
    expect(Math.max(...xs)).toBeLessThanOrEqual(h.width);
    expect(Math.min(...ys)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...ys)).toBeLessThanOrEqual(h.height);
    expect(h.frame).toBe(Math.trunc(xspace / 2));
  });

  it('a vertical label is the horizontal one rotated: (x, row) -> (height - row, x)', () => {
    const f = new ScalableFont();
    f.selectFont(20);
    const h = createLabel(f, 'Ai', true);
    const v = createLabel(f, 'Ai', false);
    expect(v.strokes.length).toBe(h.strokes.length);
    for (let k = 0; k < h.strokes.length; k += 2) {
      expect(v.strokes[k]).toBeCloseTo(h.height - h.strokes[k + 1], 9);
      expect(v.strokes[k + 1]).toBeCloseTo(h.strokes[k], 9);
    }
  });
});

describe('fix graphics/labels (src/GRAPHICS/fix_graphics_labels.cpp)', () => {
  it('parses colour names and r/g/b values and refuses out-of-range values', () => {
    expect(parseColor('white')).toEqual([255, 255, 255]);
    expect(parseColor('darkgray')).toEqual([64, 64, 64]);
    expect(parseColor('1/2/3')).toEqual([1, 2, 3]);
    expect(() => parseColor('256/0/0')).toThrow(/Invalid RGB value/);
    expect(() => parseColor('1/2')).toThrow(/Invalid RGB value/);
  });

  it('a text label with a constant size: size 10 is rendered at 20 px and scaled by 0.5', () => {
    const specs = parseLabelArgs(['text', 'hi', '1', '2', '3', 'size', '10'], noImage);
    const objs = new FixGraphicsLabels(specs).objects(env());
    expect(objs).toHaveLength(1);
    const o = objs[0];
    expect(o.kind).toBe('label');
    if (o.kind !== 'label') return;
    expect(o.pos).toEqual([1, 2, 3]);
    expect(o.scale).toBe(0.5);
    const f = new ScalableFont();
    f.selectFont(20);
    const ref = createLabel(f, 'hi', true);
    expect(o.width).toBe(ref.width);
    expect(o.height).toBe(ref.height);
    expect(o.strokes).toEqual(ref.strokes);
  });

  it('a size above 128 (after the 2x rule) is rendered at 128 and scaled by size / 256', () => {
    const objs = new FixGraphicsLabels(parseLabelArgs(['text', 'a', '0', '0', '0', 'size', '100'], noImage)).objects(env());
    const o = objs[0];
    if (o.kind !== 'label') throw new Error('expected a label');
    expect(o.scale).toBe(200 / 256);
  });

  it('defaults: size 48 rendered with scale 0.5, white font, silver back, frame and transparent colour', () => {
    const objs = new FixGraphicsLabels(parseLabelArgs(['text', 'x', '0', '0', '0'], noImage)).objects(env());
    const o = objs[0];
    if (o.kind !== 'label') throw new Error('expected a label');
    expect(o.scale).toBe(0.5);
    expect(o.fontcolor).toEqual([255, 255, 255]);
    expect(o.backcolor).toEqual([192, 192, 192]);
    expect(o.framecolor).toEqual([192, 192, 192]);
    expect(o.transcolor).toEqual([192, 192, 192]);
  });

  it('a variable size and position are evaluated per snapshot (2 * v; v = 100 gives scale 200/256)', () => {
    const specs = parseLabelArgs(['text', 'x', 'v_px', '0', '0', 'size', 'v_s'], noImage);
    const f = new FixGraphicsLabels(specs);
    const a = f.objects(env({ px: 4, s: 40 }))[0];
    const b = f.objects(env({ px: 5, s: 100 }))[0];
    if (a.kind !== 'label' || b.kind !== 'label') throw new Error('expected labels');
    expect(a.pos[0]).toBe(4);
    expect(a.scale).toBe(0.5);
    expect(b.pos[0]).toBe(5);
    expect(b.scale).toBe(200 / 256);
  });

  it('a size that is out of range is an error naming the size', () => {
    expect(() => parseLabelArgs(['text', 'x', '0', '0', '0', 'size', '1000'], noImage)).toThrow(/Invalid .*text size/);
  });

  it('images come before texts, whatever the argument order; image transcolor auto takes the first pixel', () => {
    const rgb = new Uint8Array([9, 8, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const specs = parseLabelArgs(
      ['text', 'x', '0', '0', '0', 'image', 'pic.ppm', '1', '2', 'v_z', 'transcolor', 'auto'],
      () => ({ width: 2, height: 2, rgb }),
    );
    const objs = new FixGraphicsLabels(specs).objects(env({ z: 7 }));
    expect(objs.map((o) => o.kind)).toEqual(['pixmap', 'label']);
    const p = objs[0];
    if (p.kind !== 'pixmap') throw new Error('expected a pixmap');
    expect(p.pos).toEqual([1, 2, 7]);
    expect(p.transcolor).toEqual([9, 8, 7]);
    expect(p.width).toBe(2);
    expect(p.scale).toBe(1);
  });

  it('colorscale is refused by name, and $-references are substituted only when present', () => {
    expect(() => parseLabelArgs(['colorscale', 'd', 'lbl', '0', '0', '0'], noImage)).toThrow(/colorscale/);
    let calls = 0;
    const subst = (t: string) => {
      calls++;
      return t.replace('$t', '3');
    };
    const withDollar = new FixGraphicsLabels(parseLabelArgs(['text', 'T=$t', '0', '0', '0'], noImage)).objects(env({}, subst));
    expect(calls).toBe(1);
    const plain = new FixGraphicsLabels(parseLabelArgs(['text', 'T=3', '0', '0', '0'], noImage)).objects(env({}, subst));
    expect(calls).toBe(1);
    expect(withDollar[0]).toEqual(plain[0]);
  });
});

describe('fix graphics/periodic (src/GRAPHICS/fix_graphics_periodic.cpp)', () => {
  // two atoms of a bonded pair in a 10 x 10 x 10 box, at x = 1 and x = 9
  const atoms: PeriodicAtom[] = [
    { id: 1, type: 1, x: [1, 5, 5], inGroup: true, bondPartners: [2] },
    { id: 2, type: 2, x: [9, 5, 5], inGroup: true, bondPartners: [1] },
  ];
  const box = { prd: [10, 10, 10] as Vec3, periodic: [true, true, true] as [boolean, boolean, boolean] };

  it('xlo and xhi images: per atom and per shift a sphere, then a bond to the closest image of the partner', () => {
    const opt = parsePeriodicArgs(['xlo', 'xhi'], true);
    const objs = periodicObjects(opt, atoms, box);
    expect(objs).toHaveLength(8);
    const expected: GraphicsObject[] = [
      { kind: 'sphere', type: 1, x: -9, y: 5, z: 5, diameter: -2 },
      { kind: 'bond', types: [1, 2], x1: -9, y1: 5, z1: 5, x2: -11, y2: 5, z2: 5 },
      { kind: 'sphere', type: 1, x: 11, y: 5, z: 5, diameter: -2 },
      { kind: 'bond', types: [1, 2], x1: 11, y1: 5, z1: 5, x2: 9, y2: 5, z2: 5 },
      { kind: 'sphere', type: 2, x: -1, y: 5, z: 5, diameter: -2 },
      { kind: 'bond', types: [2, 1], x1: -1, y1: 5, z1: 5, x2: 1, y2: 5, z2: 5 },
      { kind: 'sphere', type: 2, x: 19, y: 5, z: 5, diameter: -2 },
      { kind: 'bond', types: [2, 1], x1: 19, y1: 5, z1: 5, x2: 21, y2: 5, z2: 5 },
    ];
    expect(objs).toEqual(expected);
  });

  it('atoms off and a fixed radius: spheres only, diameter 2 * radius', () => {
    const opt = parsePeriodicArgs(['xlo', 'atoms', 'off', 'radius', '0.25'], true);
    const objs = periodicObjects(opt, atoms, box);
    expect(objs.every((o) => o.kind === 'bond')).toBe(true);
    const sph = periodicObjects(parsePeriodicArgs(['xhi', 'bonds', 'off', 'radius', '0.25'], true), atoms, box);
    expect(sph).toHaveLength(2);
    expect(sph[0]).toEqual({ kind: 'sphere', type: 1, x: 11, y: 5, z: 5, diameter: 0.5 });
  });

  it('bonds need a molecular system', () => {
    expect(() => parsePeriodicArgs(['xlo', 'bonds', 'on'], false)).toThrow(/non-molecular/);
  });
});

describe('fix graphics/replica (src/GRAPHICS/fix_graphics_replica.cpp, one replica)', () => {
  const box = { lo: [0, 0, 0] as Vec3, hi: [10, 10, 10] as Vec3, prd: [10, 10, 10] as Vec3, periodic: [true, true, true] as [boolean, boolean, boolean] };
  // id 2 is unwrapped to 19.5 by its image flag and remapped to 9.5; id 1 is not moved
  const atoms: ReplicaAtom[] = [
    { id: 2, type: 3, x: [9.5, 5, 5], image: [1, 0, 0], inGroup: true },
    { id: 1, type: 1, x: [2, 5, 5], image: [0, 0, 0], inGroup: true },
    { id: 4, type: 2, x: [1, 1, 1], image: [0, 0, 0], inGroup: false },
  ];

  it('display spheres in ID order, then average spheres (diameter 0 for one replica)', () => {
    const objs = replicaObjects(parseReplicaArgs(['display', '0.5', 'average', '0.5']), atoms, box);
    expect(objs).toEqual([
      { kind: 'sphere', type: 1, x: 2, y: 5, z: 5, diameter: 0.5 },
      { kind: 'sphere', type: 3, x: 9.5, y: 5, z: 5, diameter: 0.5 },
      { kind: 'sphere', type: 1, x: 2, y: 5, z: 5, diameter: 0 },
      { kind: 'sphere', type: 3, x: 9.5, y: 5, z: 5, diameter: 0 },
    ]);
  });

  it('average radius 0 gives diameter 1', () => {
    const objs = replicaObjects(parseReplicaArgs(['average', '0']), atoms, box);
    expect(objs.every((o) => o.kind === 'sphere' && o.diameter === 1)).toBe(true);
    expect(objs).toHaveLength(2);
  });
});

describe('fix graphics/objects progbar (src/GRAPHICS/fix_graphics_objects.cpp)', () => {
  it('2 + tics cylinders: background, fill at the progress, and tics spaced length / (tics - 1)', () => {
    const spec = parseProgbar(1, 2, 'x', [5, 5, 5], 4, 0.5, 3, [10, 10, 10]);
    const objs = progbarObjects(spec, 0.5);
    expect(objs).toHaveLength(2 + 3);
    expect(objs[0]).toEqual({ kind: 'cylinder', type: 1, x1: 3, y1: 5, z1: 5, x2: 7, y2: 5, z2: 5, diameter: 1 });
    // fill: first end moves to pos + (progress - 0.5) * length = 5; second end at pos - length/2 = 3; z lifted by 0.2 d
    expect(objs[1]).toEqual({ kind: 'cylinder', type: 2, x1: 5, y1: 5, z1: 5.2, x2: 3, y2: 5, z2: 5.2, diameter: 0.75 });
    const tic = (i: number) => objs[2 + i];
    expect(tic(0)).toMatchObject({ x1: 3 - 0.1, x2: 3 + 0.1, diameter: 1.1, type: 1 });
    expect(tic(1)).toMatchObject({ x1: 5 - 0.1, x2: 5 + 0.1 });
    expect(tic(2)).toMatchObject({ x1: 7 - 0.1, x2: 7 + 0.1 });
  });

  it('progress is clamped to [1e-10, 1]', () => {
    const spec = parseProgbar(1, 2, 'y', [0, 0, 0], 2, 1, 0, [10, 10, 10]);
    const high = progbarObjects(spec, 7)[1] as { y1: number };
    expect(high.y1).toBeCloseTo(1, 12);
    const low = progbarObjects(spec, -3)[1] as { y1: number };
    expect(low.y1).toBeCloseTo(-1 + 1e-10 * 2, 9);
  });

  it('refuses a length larger than twice the box and a bad tic count', () => {
    expect(() => parseProgbar(1, 2, 'x', [0, 0, 0], 25, 1, 3, [10, 10, 10])).toThrow(/length/);
    expect(() => parseProgbar(1, 2, 'x', [0, 0, 0], 4, 1, 21, [10, 10, 10])).toThrow(/tics/);
  });
});

describe('convex hull (src/GRAPHICS/image_objects.cpp, ConvexHullObj)', () => {
  it('a tetrahedron gives 4 faces, each triangle using 3 of the 4 points', () => {
    const pts: Vec3[] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const hull = buildConvexHull(pts, true, 0);
    expect(hull.triangles).toHaveLength(4);
    for (const idx of hull.colorIdx) expect(new Set(idx).size).toBe(3);
  });

  it('the 12 vertices of an icosahedron give its 20 triangular faces', () => {
    const a = 0.5257311121191336;
    const b = 0.8506508083520399;
    const ico: Vec3[] = [
      [-a, b, 0], [a, b, 0], [-a, -b, 0], [a, -b, 0], [0, -a, b], [0, a, b],
      [0, -a, -b], [0, a, -b], [b, 0, -a], [b, 0, a], [-b, 0, -a], [-b, 0, a],
    ];
    const hull = buildConvexHull(ico.map((p) => [0.5 * p[0], 0.5 * p[1], 0.5 * p[2]] as Vec3), true, 0);
    expect(hull.triangles).toHaveLength(20);
    for (const n of hull.normals) {
      for (const v of n) expect(Math.hypot(v[0], v[1], v[2])).toBeCloseTo(1, 9);
    }
  });

  it('coplanar points give the fan, both faces: 2 triangles per edge (a square gives 8)', () => {
    const sq: Vec3[] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]];
    const hull = buildConvexHull(sq, false, 0);
    expect(hull.triangles).toHaveLength(8);
    expect(hull.normals[0][0][2]).toBeCloseTo(1, 12);
    expect(hull.normals[1][0][2]).toBeCloseTo(-1, 12);
  });

  it('fewer than four points give nothing', () => {
    expect(buildConvexHull([[0, 0, 0], [1, 0, 0], [0, 1, 0]], true, 0).triangles).toHaveLength(0);
  });
});

describe('fix graphics/chunk (src/GRAPHICS/fix_graphics_chunk.cpp)', () => {
  const box = { lo: [-100, -100, -100] as Vec3, hi: [100, 100, 100] as Vec3, dimension: 3 as const };

  function chunkOpts(extra: string[] = []) {
    return parseChunkArgs(extra, () => null);
  }

  it('two atoms 10 apart in one chunk: two clusters, each replaced by 12 icosahedron points (20 triangles each)', () => {
    const atoms: ChunkAtom[] = [
      { type: 1, x: [0, 0, 0], inGroup: true, chunk: 1 },
      { type: 2, x: [10, 0, 0], inGroup: true, chunk: 1 },
    ];
    const fix = new FixGraphicsChunk({ ...chunkOpts(['radius', '0.5']), defaultRadius: 0.5 });
    const objs = fix.objects(atoms, box, 1);
    expect(objs).toHaveLength(40);
    expect(objs.every((o) => o.kind === 'trinorm')).toBe(true);
    // a cluster's triangles take the type of its first atom (vertex index / 12 is the atom index)
    const types = objs.map((o) => (o.kind === 'trinorm' ? o.types[0] : -1));
    expect(types.filter((t) => t === 1)).toHaveLength(20);
    expect(types.filter((t) => t === 2)).toHaveLength(20);
    // every vertex is an icosahedron point of radius 0.5 around its atom
    for (const o of objs) {
      if (o.kind !== 'trinorm') continue;
      for (const v of o.vertices) {
        const c = Math.min(Math.hypot(v[0], v[1], v[2]), Math.hypot(v[0] - 10, v[1], v[2]));
        expect(c).toBeCloseTo(0.5, 9);
      }
    }
  });

  it('a 4-atom cluster above maxreplace: the atoms are pushed out by the radius, and the types are the quirk of index / 12', () => {
    const atoms: ChunkAtom[] = [
      { type: 3, x: [0, 0, 0], inGroup: true, chunk: 1 },
      { type: 4, x: [1, 0, 0], inGroup: true, chunk: 1 },
      { type: 5, x: [0, 1, 0], inGroup: true, chunk: 1 },
      { type: 6, x: [0, 0, 1], inGroup: true, chunk: 1 },
    ];
    const fix = new FixGraphicsChunk({ ...chunkOpts(['radius', '0.5', 'maxreplace', '0']), defaultRadius: 0.5 });
    const objs = fix.objects(atoms, box, 1);
    expect(objs).toHaveLength(4);
    for (const o of objs) {
      if (o.kind !== 'trinorm') continue;
      expect(o.types).toEqual([3, 3, 3]);
    }
    // the pushed-out vertices lie 0.5 from their atom along (x - centre)
    const centre: Vec3 = [0.25, 0.25, 0.25];
    const allVerts = new Set<string>();
    for (const o of objs) {
      if (o.kind !== 'trinorm') continue;
      for (const v of o.vertices) allVerts.add(v.map((c) => c.toFixed(9)).join(','));
    }
    expect(allVerts.size).toBe(4);
    for (const key of allVerts) {
      const v = key.split(',').map(Number) as Vec3;
      expect(Math.min(...atoms.map((a) => Math.hypot(v[0] - a.x[0], v[1] - a.x[1], v[2] - a.x[2])))).toBeCloseTo(0.5, 6);
    }
    void centre;
  });

  it('mindist is stored once: the first snapshot fixes 4 * maxradius (2.0 here)', () => {
    const fix = new FixGraphicsChunk({ ...chunkOpts(), defaultRadius: 0.5 });
    const first: ChunkAtom[] = [
      { type: 1, x: [0, 0, 0], inGroup: true, chunk: 1, radius: 0.5 },
      { type: 1, x: [3, 0, 0], inGroup: true, chunk: 1, radius: 0.5 },
    ];
    // maxradius 0.5 gives mindist 2.0; 3 apart is more, so two clusters of 12 points: 40 triangles
    expect(fix.objects(first, box, 1)).toHaveLength(40);
    // a snapshot with maxradius 1.0 would give mindist 4.0 (one cluster); the stored 2.0 still splits them
    const later: ChunkAtom[] = first.map((a) => ({ ...a, radius: 1.0 }));
    expect(fix.objects(later, box, 1)).toHaveLength(40);
  });

  it('triangles outside the box plus mindist are dropped; the region keyword filters atoms', () => {
    const atoms: ChunkAtom[] = [
      { type: 1, x: [0, 0, 0], inGroup: true, chunk: 1 },
      { type: 2, x: [50, 0, 0], inGroup: true, chunk: 1 },
    ];
    const tight = { lo: [-5, -5, -5] as Vec3, hi: [5, 5, 5] as Vec3, dimension: 3 as const };
    const fix = new FixGraphicsChunk({ ...chunkOpts(['radius', '0.5']), defaultRadius: 0.5 });
    const objs = fix.objects(atoms, tight, 1);
    expect(objs).toHaveLength(20);
    const regionFix = new FixGraphicsChunk({ ...chunkOpts(['radius', '0.5']), defaultRadius: 0.5, region: (x) => x[0] < 20 });
    expect(regionFix.objects(atoms, box, 1)).toHaveLength(20);
  });

  it('a 2d box is refused', () => {
    const fix = new FixGraphicsChunk({ ...chunkOpts(), defaultRadius: 0.5 });
    expect(() => fix.objects([], { ...box, dimension: 2 }, 0)).toThrow(/2d/);
  });
});

describe('the image object type of fix graphics/labels (no renderer)', () => {
  it('an empty spec gives no objects', () => {
    expect(new FixGraphicsLabels([]).objects(env())).toEqual([]);
  });
});
