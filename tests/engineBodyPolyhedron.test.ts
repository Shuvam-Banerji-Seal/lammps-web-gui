import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { BODY_RECORD, POLY_OFF, bodyRecordStride, exyzToQuat, parseBodyStyle, polyRecordLength, polyStoredCounts, polyhedronFrame } from '../src/engine/bodies';

/*
 * atom_style body rounded/polyhedron (3d polyhedra with rounded corners): the Bodies section (native
 * body_rounded_polyhedron.cpp data_body: "3 integers: nvertices, nedges, nfaces", then the 6 inertia values, the
 * vertices, the edge ends, the faces (4 indices, the fourth -1 for a triangle) and the rounded diameter), the record
 * layout (bodies.ts), the principal frame (native jacobi3 with decreasing moments, exyz_to_q), and the refusal of
 * the keywords of pair body/rounded/polyhedron. Native parity of the free rotation is in tests/oracle/w121body_free_polyhedra.in.
 */

// a box with edges 2, 1, 0.5 (vertices at (+-1, +-0.5, +-0.25)): the cube-body layout of examples/body/data.cubes
const BOX = `box polyhedron

1 atoms
1 bodies
1 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Atoms

1 1 1 1 0 0 0

Bodies

1 3 79
8 12 6
0.104167 0.354167 0.416667 0 0 0
1 0.5 0.25
1 -0.5 0.25
-1 -0.5 0.25
-1 0.5 0.25
1 0.5 -0.25
1 -0.5 -0.25
-1 -0.5 -0.25
-1 0.5 -0.25
0 1
1 2
2 3
3 0
4 5
5 6
6 7
7 4
0 4
1 5
2 6
3 7
0 1 2 3
4 5 6 7
0 1 5 4
1 2 6 5
2 3 7 6
3 0 4 7
0.2
`;

const HEAD = 'units lj\ndimension 3\natom_style body rounded/polyhedron 1 8\nread_data box.data\n';
const newSession = () => new Session({ emit: () => {}, writeFile: () => {} });
const run = async (script: string, files: Record<string, string>) => {
  const session = newSession();
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return session;
};

describe('atom_style body rounded/polyhedron (bodies.ts, output/data.ts)', () => {
  it('parses the style and sizes the record from the offsets of native body_rounded_polyhedron.cpp', () => {
    expect(parseBodyStyle('body rounded/polyhedron 1 8')).toEqual({ bstyle: 'rounded/polyhedron', nmin: 1, nmax: 8 });
    expect(POLY_OFF).toBe(8);
    expect(polyRecordLength(8, 12, 6)).toBe(8 + 24 + 24 + 24);
    expect(polyRecordLength(1, 0, 0)).toBe(11);
    expect(bodyRecordStride(8, 'rounded/polyhedron')).toBe(polyRecordLength(8, 24, 16));
  });

  it('stores the counts, the moments at 1..3, the radii, the body-frame vertices, edges and faces of a box', async () => {
    const s = (await run(HEAD, { 'box.data': BOX })).sys.state;
    const rec = s.custom.get(BODY_RECORD)!;
    expect(rec.data[0]).toBe(8);
    expect(rec.data[1]).toBeCloseTo(0.416667, 12);
    expect(rec.data[2]).toBeCloseTo(0.354167, 12);
    expect(rec.data[3]).toBeCloseTo(0.104167, 12);
    expect(rec.data[4]).toBe(12);
    expect(rec.data[5]).toBe(6);
    // the rounded radius is half the diameter; the enclosing radius is the largest vertex distance
    expect(rec.data[7]).toBe(0.1);
    expect(rec.data[6]).toBeCloseTo(Math.sqrt(1 + 0.25 + 0.0625), 12);
    // the axes are ordered by decreasing moment (the box's z, y, x), so the body-frame vertex is a permutation of
    // (1, 0.5, 0.25) with one sign flip for a right-handed frame: its length is kept
    const v0 = Array.from(rec.data.subarray(POLY_OFF, POLY_OFF + 3));
    expect(Math.hypot(...v0)).toBeCloseTo(Math.hypot(1, 0.5, 0.25), 12);
    expect(s.quat!.subarray(0, 4).reduce((a, b) => a + b * b, 0)).toBeCloseTo(1, 12);
    const e0 = POLY_OFF + 24;
    expect(Array.from(rec.data.subarray(e0, e0 + 4))).toEqual([0, 1, 1, 2]);
    const f0 = e0 + 24;
    expect(Array.from(rec.data.subarray(f0, f0 + 4))).toEqual([0, 1, 2, 3]);
  });

  it('stores a sphere-like body (one vertex) and a rod (two vertices) without edges or faces', async () => {
    const sphere = `one vertex

1 atoms
1 bodies
1 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Atoms

1 1 1 1 0 0 0

Bodies

1 3 10
1 0 0
0.4 0.4 0.4 0 0 0
0 0 0
0.5
`;
    const s = (await run(HEAD.replace('1 8', '1 1').replace('box.data', 'sph.data'), { 'sph.data': sphere })).sys.state;
    const rec = s.custom.get(BODY_RECORD)!;
    expect(rec.data[0]).toBe(1);
    expect(rec.data[4]).toBe(0);
    expect(rec.data[5]).toBe(0);
    expect(rec.data[6]).toBe(0.25);
    expect(rec.data[7]).toBe(0.25);
    expect(polyStoredCounts(1, 5, 3)).toEqual({ nedge: 0, nface: 0 });
    expect(polyStoredCounts(2, 0, 0)).toEqual({ nedge: 1, nface: 0 });
  });

  it('refuses a rod whose two vertices coincide, an edge with a bad vertex and a face on a line', async () => {
    const rod = `rod with coincident vertices

1 atoms
1 bodies
1 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-10 10 zlo zhi

Atoms

1 1 1 1 0 0 0

Bodies

1 3 13
2 1 0
0.09 0.16 0.25 -0.12 0 0
0.4 0.3 0
0.4 0.3 0
0.5
`;
    await expect(run(HEAD.replace('1 8', '1 2').replace('box.data', 'rod.data'), { 'rod.data': rod })).rejects.toThrow(/coincide/);
    const badEdge = BOX.replace('0 1\n1 2\n', '0 9\n1 2\n');
    await expect(run(HEAD, { 'box.data': badEdge })).rejects.toThrow(/invalid vertices or zero length/);
    const line = BOX.replace('1 0.5 0.25\n1 -0.5 0.25\n-1 -0.5 0.25\n', '1 0.5 0.25\n2 0.5 0.25\n3 0.5 0.25\n');
    await expect(run(HEAD, { 'box.data': line })).rejects.toThrow(/on a line/);
  });

  it('refuses a Bodies entry with the wrong number of doubles', async () => {
    await expect(run(HEAD, { 'box.data': BOX.replace('1 3 79', '1 3 78') })).rejects.toThrow(/Incorrect # of floating-point values/);
  });

  it('accepts the history keyword of pair body/rounded/polyhedron and refuses an unknown keyword by name', async () => {
    await expect(run(`${HEAD}pair_style body/rounded/polyhedron 20 5 0.1 1.0 0.5 history\n`, { 'box.data': BOX })).resolves.toBeDefined();
    await expect(run(`${HEAD}pair_style body/rounded/polyhedron 20 5 0.1 1.0 0.5 bogus\n`, { 'box.data': BOX })).rejects.toThrow(/bogus/);
  });
});

describe('principal frame of a polyhedron body (native jacobi3 with decreasing moments, exyz_to_q)', () => {
  it('orders the moments decreasingly and keeps a right-handed frame', () => {
    const pf = polyhedronFrame([0.1, 0.4, 0.2, 0, 0, 0]);
    expect(pf.moments).toEqual([0.4, 0.2, 0.1]);
    const E = pf.E;
    const ex = [E[0], E[3], E[6]], ey = [E[1], E[4], E[7]], ez = [E[2], E[5], E[8]];
    const cross = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
    expect(cross[0] * ez[0] + cross[1] * ez[1] + cross[2] * ez[2]).toBeCloseTo(1, 12);
  });

  it('sets a zero moment to exactly zero', () => {
    const pf = polyhedronFrame([0.25, 0.25, 0, 0, 0, 0]);
    expect(pf.moments[2]).toBe(0);
  });

  it('gives the quaternion of exyz_to_q, positive for a 90 degree turn about z', () => {
    const q = exyzToQuat([0, 1, 0], [-1, 0, 0], [0, 0, 1]);
    expect(q[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(q[3]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(exyzToQuat([1, 0, 0], [0, 1, 0], [0, 0, 1])).toEqual([1, 0, 0, 0]);
  });
});
