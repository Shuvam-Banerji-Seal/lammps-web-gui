import { describe, expect, it } from 'vitest';
import { PolyhedronPairLaw, makePBody, type PBody, type PolyParams, type V3 } from '../src/engine-gpl/body/rounded_polyhedron_law';

/*
 * The contact law of pair body/rounded/polyhedron (engine-gpl/body/rounded_polyhedron_law.ts) on two rounded cubes
 * (edge 2, rounded radius 0.25, the data.cubes vertices). Native parity of whole runs is in the oracle cases
 * tests/oracle/w121body_cubes_*.in and w121body_cube_sphere.in; this file checks the law on single pairs.
 */
const CUBE: V3[] = [[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]];
const EDGES: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
const FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
const EYE = [1, 0, 0, 0, 1, 0, 0, 0, 1];

const cube = (idx: number, c: V3, vcm: V3 = [0, 0, 0]): PBody =>
  makePBody(idx, c, vcm, [0, 0, 0], EYE, CUBE, EDGES, FACES, Math.sqrt(3), 0.25);

const params = (mu: number): PolyParams => ({ kn: 100, kna: 1, cn: 20, ct: 5, mu, Aua: 1, cut: 0.5 });

const interact = (a: PBody, b: PBody, mu = 0) => {
  const law = new PolyhedronPairLaw();
  law.f = new Float64Array(6);
  law.tq = new Float64Array(6);
  law.P = params(mu);
  law.interact(a, b);
  return { f: law.f, tq: law.tq, energy: law.energy };
};

describe('pair body/rounded/polyhedron contact law on two cubes (single pairs)', () => {
  it('gives no force or energy when the cubes are beyond the interaction range', () => {
    const r = interact(cube(0, [0, 0, 0]), cube(1, [6, 0, 0]));
    expect(Array.from(r.f)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(r.energy).toBe(0);
  });

  it('repels overlapping cubes along the face normal, with equal and opposite forces', () => {
    const r = interact(cube(0, [0, 0, 0]), cube(1, [2.4, 0, 0]));
    expect(r.f[0]).toBeLessThan(0);
    expect(r.f[3]).toBeGreaterThan(0);
    for (let k = 0; k < 3; k++) expect(Math.abs(r.f[k] + r.f[3 + k])).toBeLessThan(1e-12);
    expect(r.energy).toBeGreaterThan(0);
  });

  it('attracts cubes in the cohesive range (surface gap below cut_inner), with a negative energy', () => {
    const r = interact(cube(0, [0, 0, 0]), cube(1, [2.8, 0, 0]));
    expect(r.f[0]).toBeGreaterThan(0);
    expect(r.energy).toBeLessThan(0);
    for (let k = 0; k < 3; k++) expect(Math.abs(r.f[k] + r.f[3 + k])).toBeLessThan(1e-12);
  });

  it('keeps the forces of a sliding pair equal and opposite with friction (mu > 0)', () => {
    const a = cube(0, [0, 0, 0], [0, 0.5, 0]);
    const b = cube(1, [2.4, 0, 0], [0, 0, 0]);
    const free = interact(a, b, 0);
    const rubbed = interact(a, b, 0.5);
    for (let k = 0; k < 3; k++) expect(Math.abs(rubbed.f[k] + rubbed.f[3 + k])).toBeLessThan(1e-12);
    // the friction opposes the tangential relative velocity along y, so the y force on atom 0 differs from the frictionless one
    expect(Math.abs(rubbed.f[1] - free.f[1])).toBeGreaterThan(1e-9);
    expect(rubbed.f[1]).toBeLessThan(free.f[1]);
  });

  it('gives the same result for the pair in either order', () => {
    // the array slots follow the atom index: the first argument is always atom 0 of the array
    const r1 = interact(cube(0, [0, 0, 0]), cube(1, [2.6, 0.2, 0.1]));
    const r2 = interact(cube(0, [2.6, 0.2, 0.1]), cube(1, [0, 0, 0]));
    expect(r2.energy).toBeCloseTo(r1.energy, 12);
    // the force on the origin cube is atom 0 in r1 and atom 1 in r2: equal and opposite to the force on the offset cube
    for (let k = 0; k < 3; k++) expect(r2.f[3 + k]).toBeCloseTo(r1.f[k], 10);
  });
});
