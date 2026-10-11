import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { BODY_RECORD, bodyRecordOffset, bodyRecordStride, parseBodyStyle, principalFrame } from '../src/engine/bodies';

/*
 * atom_style body rounded/polygon (2d polygons with rounded vertices): the Bodies section (Howto_body.rst,
 * "atom-ID 1 M", "M = 6 + 3*N + 1"), the body record (bodies.ts), and the refusals of the styles that are not
 * implemented yet, each by name. Native parity of the free motion is in tests/oracle/w49bp_free_squares.in.
 *
 * Measured with native LAMMPS (black box): the moments of the Bodies line are used as given. A square with the
 * edge-4 vertices of Howto_body.rst but Izz 1.0 (instead of 1/6 m L^2 = 2.67) gives the rotational kinetic
 * temperature 1/Izz of compute temp/body for a unit angular momentum: 1 for Izz 1.0, 0.3745 for Izz 2.67. The
 * vertices do not enter the moments.
 */

const SQUARE = `one rounded square

1 atoms
1 bodies
1 atom types

-10 10 xlo xhi
-10 10 ylo yhi
-0.5 0.5 zlo zhi

Atoms

1 1 1 1 0 0 0

Bodies

1 1 19
4
1 1 2.67 0 0 0
-2 -2 0
-2 2 0
2 2 0
2 -2 0
0.5
`;

const newSession = () => new Session({ emit: () => {}, writeFile: () => {} });

const run = async (script: string, files: Record<string, string>) => {
  const session = newSession();
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return session;
};

const HEAD = 'units lj\ndimension 2\natom_style body rounded/polygon 1 6\nread_data sq.data\n';

describe('atom_style body rounded/polygon (bodies.ts, output/data.ts)', () => {
  it('parses the style bounds and the record layout with the diameter at index 4', () => {
    expect(parseBodyStyle('body rounded/polygon 1 6')).toEqual({ bstyle: 'rounded/polygon', nmin: 1, nmax: 6 });
    expect(bodyRecordOffset('rounded/polygon')).toBe(5);
    expect(bodyRecordOffset('nparticle')).toBe(4);
    expect(bodyRecordStride(6, 'rounded/polygon')).toBe(23);
    expect(bodyRecordStride(6)).toBe(22);
  });

  it('stores the vertex count, the moments as given, the rounded diameter and the body-frame vertices', async () => {
    const s = (await run(HEAD, { 'sq.data': SQUARE })).sys.state;
    expect(s.n).toBe(1);
    expect(s.rmass![0]).toBe(1.0);
    const rec = s.custom.get(BODY_RECORD)!;
    expect(rec.cols).toBe(23);
    expect(rec.data[0]).toBe(4);
    expect(rec.data[1]).toBeCloseTo(1, 12);
    expect(rec.data[2]).toBeCloseTo(1, 12);
    expect(rec.data[3]).toBeCloseTo(2.67, 12);
    expect(rec.data[4]).toBe(0.5);
    // identity orientation: the body frame equals the box frame
    expect(Array.from(rec.data.subarray(5, 8))).toEqual([-2, -2, 0]);
    expect(Array.from(rec.data.subarray(14, 17))).toEqual([2, -2, 0]);
    expect(Array.from(s.quat!.subarray(0, 4))).toEqual([1, 0, 0, 0]);
  });

  it('reads a disk (N = 1, M = 10) and a rod (N = 2, M = 13)', async () => {
    const disk = SQUARE.replace('1 1 19\n4\n1 1 2.67 0 0 0\n-2 -2 0\n-2 2 0\n2 2 0\n2 -2 0\n0.5', '1 1 10\n1\n1 1 4.5 0 0 0\n0 0 0\n3.0');
    const sd = (await run(HEAD, { 'sq.data': disk })).sys.state;
    expect(sd.custom.get(BODY_RECORD)!.data[0]).toBe(1);
    expect(sd.custom.get(BODY_RECORD)!.data[4]).toBe(3.0);
    const rod = SQUARE.replace('1 1 19\n4\n1 1 2.67 0 0 0\n-2 -2 0\n-2 2 0\n2 2 0\n2 -2 0\n0.5', '1 1 13\n2\n1 1 1.33333 0 0 0\n-2 0 0\n2 0 0\n0.5');
    const sr = (await run(HEAD, { 'sq.data': rod })).sys.state;
    expect(sr.custom.get(BODY_RECORD)!.data[0]).toBe(2);
    expect(sr.custom.get(BODY_RECORD)!.data[4]).toBe(0.5);
  });

  it('refuses a Bodies entry with too few doubles (M = 6 + 3N + 1 = 19 for N = 4)', async () => {
    const bad = SQUARE.replace('0.5\n', '');
    await expect(run(HEAD, { 'sq.data': bad })).rejects.toThrow(/lists 18 of 19 doubles/);
  });

  it('refuses a Bodies entry whose vertex count does not give the doubles of M = 6 + 3N + 1', async () => {
    const bad = SQUARE.replace('1 1 19\n4\n', '1 1 19\n3\n');
    await expect(run(HEAD, { 'sq.data': bad })).rejects.toThrow(/6 \+ 3N \+ 1 = 16 doubles \(N = 3\), got 19/);
  });

  it('refuses vertices with z different from 0 (2d bodies)', async () => {
    const bad = SQUARE.replace('-2 -2 0', '-2 -2 0.5');
    await expect(run(HEAD, { 'sq.data': bad })).rejects.toThrow(/must have z = 0/);
  });

  it('refuses rounded/polygon in a 3d simulation', async () => {
    await expect(run(HEAD.replace('dimension 2', 'dimension 3'), { 'sq.data': SQUARE })).rejects.toThrow(/only be used in 2d/);
  });

  it('refuses atom_style body rounded/polyhedron by name', async () => {
    await expect(run('units lj\ndimension 3\natom_style body rounded/polyhedron 1 6\n', {})).rejects.toThrow(/rounded\/polyhedron/);
  });

  it('refuses pair body/nparticle on rounded/polygon bodies', async () => {
    await expect(run(`${HEAD}pair_style body/nparticle 4.0\npair_coeff * * 1.0 1.0\nrun 0\n`, { 'sq.data': SQUARE })).rejects.toThrow(/requires atom_style body nparticle/);
  });
});

describe('refused body styles are named (registry/body48.ts)', () => {
  it('pair body/rounded/polygon, fix wall/body/polygon, nvt/body and npt/body', async () => {
    await expect(run(`${HEAD}pair_style body/rounded/polygon 20.0 5.0 0.0 1.0 0.5\n`, { 'sq.data': SQUARE })).rejects.toThrow(/pair_style 'body\/rounded\/polygon' is not supported/);
    await expect(run(`${HEAD}fix 1 all wall/body/polygon 2000 50 50 yplane -5 5\n`, { 'sq.data': SQUARE })).rejects.toThrow(/wall\/body\/polygon'/);
    await expect(run(`${HEAD}fix 1 all nvt/body temp 0.5 0.5 1.0\n`, { 'sq.data': SQUARE })).rejects.toThrow(/nvt\/body'/);
    await expect(run(`${HEAD}fix 1 all npt/body temp 0.5 0.5 1.0 x 0.001 0.1 1.0\n`, { 'sq.data': SQUARE })).rejects.toThrow(/npt\/body'/);
  });
});

describe('principal moments as given', () => {
  it('keeps Izz of a rounded square as written in the Bodies line', () => {
    const pf = principalFrame([1, 1, 2.67, 0, 0, 0]);
    expect(pf.moments).toEqual([1, 1, 2.67]);
    expect(pf.quat[0]).toBeCloseTo(1, 12);
  });
});
