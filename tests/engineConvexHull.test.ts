import { describe, expect, it } from 'vitest';
import { buildConvexHull } from '../src/engine-gpl/graphics/convex_hull';

/*
 * The alpha-shape surface of fix graphics/chunk (engine-gpl/graphics/convex_hull.ts, the port of ConvexHullObj in
 * src/GRAPHICS/image_objects.cpp). The tetrahedra are flat arrays with a list of the valid ones in creation order and
 * the face map is keyed by number; the hashes below were taken from the earlier object-and-string implementation,
 * which kept the C++ data structures one for one, so the triangles, normals and their order (std::map key order) are
 * unchanged bit for bit.
 */

type V = [number, number, number];

const cloud = (n: number, s: number, f: (r: () => number, k: number) => V): V[] => {
  let seed = s;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  return Array.from({ length: n }, (_, k) => f(rnd, k));
};

/** FNV-1a of a string. */
const fnv = (t: string): number => {
  let h = 0x811c9dc5;
  for (let i = 0; i < t.length; i++) { h ^= t.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h;
};

const CASES: [string, V[], boolean, number, number, number][] = [
  ['random 400, smooth, auto alpha', cloud(400, 7, (r) => [r() * 10, r() * 7, r() * 5]), true, 0, 412, 1145091030],
  ['two clusters 300, flat, alpha 3', cloud(300, 11, (r, k) => [r() + (k % 2) * 20, r(), r()]), false, 3, 164, 584700574],
  ['sphere shell 200, smooth, alpha 6', cloud(200, 13, (r) => {
    const u = 2 * r() - 1, ph = 2 * Math.PI * r(), q = Math.sqrt(1 - u * u);
    return [5 * q * Math.cos(ph), 5 * q * Math.sin(ph), 5 * u];
  }), true, 6, 396, 4219262951],
];

describe('fix graphics/chunk alpha-shape surfaces', () => {
  for (const [name, pts, smooth, alpha, ntris, hash] of CASES) {
    it(`${name}: the triangles, normals and order of the C++ data structures`, () => {
      const h = buildConvexHull(pts, smooth, alpha);
      expect(h.triangles.length).toBe(ntris);
      expect(fnv(JSON.stringify([h.triangles, h.normals, h.colorIdx]))).toBe(hash);
    });
  }

  it('a 3000-point cloud stays under 3 s (about 0.3 s under vitest; the earlier version took several seconds)', () => {
    const pts = cloud(3000, 17, (r) => [r() * 30, r() * 20, r() * 10]);
    const t0 = performance.now();
    const h = buildConvexHull(pts, true, 0);
    expect(h.triangles.length).toBeGreaterThan(0);
    expect(performance.now() - t0).toBeLessThan(3000);
  });
});
