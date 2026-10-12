import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { LINE_BONUS, LINE_FLAG, TRI_BONUS, TRI_BONUS_COLS, TRI_FLAG } from '../src/engine/atoms';
import { lineEnds, lineFromEnds, triangleCorner } from '../src/engine/lineTri';
import { TRI_BODY_HOOK } from '../src/engine-gpl/tri/tri_bonus';

/*
 * atom_style line (2d) and tri (3d): the data layout (src/engine/lineTri.ts, output/data.ts, atoms.ts) and the
 * GPL styles that use it (src/engine-gpl/line, src/engine-gpl/tri). Native parity of the pair styles and the
 * integrators is in tests/oracle/w61line_pair.in and w61tri_pair.in (engineOracle.test.ts -t w61). The checks here:
 * the geometry conversions, the Atoms / Lines / Triangles / Velocities sections and their refusals, the write_data
 * round trip, and the refused combinations, each by name.
 */

const newSession = (written: Map<string, string> = new Map()) => new Session({
  emit: () => {},
  writeFile: (name: string, text: string) => { written.set(name, text); },
});

const run = async (script: string, files: Record<string, string> = {}, written?: Map<string, string>) => {
  const session = newSession(written);
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  await session.execute(script);
  return session;
};

// two line segments and one point particle in 2d; the Lines entries are the end points of each segment
const LINE2D = `2d lines and a point

3 atoms
2 atom types
2 lines

0 10 xlo xhi
0 10 ylo yhi
-0.5 0.5 zlo zhi

Atoms # line

1 1 1 1 2.0 2.0 2.0 0
2 2 2 1 1.5 5.0 5.0 0
3 3 1 0 1.0 8.0 8.0 0

Velocities

1 0.1 0.0 0.0 0 0 0.25
2 0.0 0.0 0.0 0 0 0
3 0.0 0.0 0.0 0 0 0

Lines

1 1.5 2.0 2.5 2.0
2 4.0 5.0 6.0 5.0
`;

const HEAD2 = 'units lj\ndimension 2\natom_style line\nread_data line2d.data\n';

describe('line and tri geometry (lineTri.ts)', () => {
  it('a segment from its end points: length, theta in (-pi, pi], and the centre and ends round trip', () => {
    const lb = lineFromEnds(0, 0, 0, 2);
    expect(lb.length).toBeCloseTo(2, 14);
    expect(lb.theta).toBeCloseTo(Math.PI / 2, 14);
    // theta is the angle of c2 - c1 from +x; a segment pointing down has a negative theta
    expect(lineFromEnds(0, 0, 1, -1).theta).toBeCloseTo(-Math.PI / 4, 14);
    const [x1, y1, x2, y2] = lineEnds(3, 4, 2, 0.3);
    expect((x1 + x2) / 2).toBeCloseTo(3, 14);
    expect((y1 + y2) / 2).toBeCloseTo(4, 14);
    expect(Math.hypot(x2 - x1, y2 - y1)).toBeCloseTo(2, 14);
  });

  it('a triangle: area, centroid, corner round trip through the principal frame, and positive moments', () => {
    const c: [[number, number, number], [number, number, number], [number, number, number]] = [
      [1.0, 2.0, 0.5], [2.2, 2.1, 0.7], [1.4, 3.1, 0.2],
    ];
    const tb = TRI_BODY_HOOK.triBody(c, 2.0);
    // area = half the length of the cross product of two edges
    const e1 = [1.2, 0.1, 0.2], e2 = [0.4, 1.1, -0.3];
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    expect(tb.area).toBeCloseTo(0.5 * Math.hypot(...n), 12);
    expect(tb.centroid[0]).toBeCloseTo((1.0 + 2.2 + 1.4) / 3, 14);
    for (let k = 0; k < 3; k++) {
      const p = triangleCorner(tb.centroid, tb.quat, tb.corners, k);
      for (let d = 0; d < 3; d++) expect(p[d]).toBeCloseTo(c[k][d], 10);
    }
    for (const m of tb.moments) expect(m).toBeGreaterThan(0);
    expect(tb.corners.length).toBe(9);
    expect(TRI_BONUS_COLS).toBe(12);
    // the unit quaternion is normalised
    expect(Math.hypot(...tb.quat)).toBeCloseTo(1, 12);
  });
});

describe('atom_style line in read_data', () => {
  it('reads Atoms (lineflag, density), Lines (end points) and Velocities (omega z)', async () => {
    const s = (await run(HEAD2, { 'line2d.data': LINE2D })).sys.state;
    const flag = s.custom.get(LINE_FLAG)!.data, bonus = s.custom.get(LINE_BONUS)!.data;
    expect(Array.from(flag.subarray(0, 3))).toEqual([1, 1, 0]);
    // the Lines entry of atom 1: end points (1.5, 2) to (2.5, 2): length 1, theta 0, centred on x
    expect(bonus[0]).toBeCloseTo(1, 12);
    expect(bonus[1]).toBeCloseTo(0, 12);
    expect(s.radius![0]).toBeCloseTo(0.5, 12);
    // rmass = density (per length) times length: 2.0 * 1
    expect(s.rmass![0]).toBeCloseTo(2.0, 12);
    // atom 2: end points (4,5) to (6,5): length 2, density 1.5 -> mass 3
    expect(bonus[2]).toBeCloseTo(2, 12);
    expect(s.rmass![1]).toBeCloseTo(3, 12);
    // point particle keeps its density as mass and has radius 0
    expect(s.rmass![2]).toBeCloseTo(1.0, 12);
    expect(s.radius![2]).toBe(0);
    // Velocities: the z component of omega is read
    expect(s.omega![2]).toBeCloseTo(0.25, 12);
    expect(s.dimension).toBe(2);
  });

  it('refuses a line atom style in 3d, a Lines entry that does not centre on the atom, and a missing Lines entry', async () => {
    await expect(run('units lj\ndimension 3\natom_style line\nread_data line2d.data\n', { 'line2d.data': LINE2D }))
      .rejects.toThrow(/Atom_style line can only be used in 2d simulations/);
    const off = LINE2D.replace('1 1.5 2.0 2.5 2.0', '1 1.5 2.0 2.5 3.0');
    await expect(run(HEAD2, { 'line2d.data': off })).rejects.toThrow(/Inconsistent line segment/);
    const none = LINE2D.replace('2 4.0 5.0 6.0 5.0\n', '');
    await expect(run(HEAD2, { 'line2d.data': none })).rejects.toThrow(/expected 2 lines|lineflag 1 but no Lines/);
    await expect(run(HEAD2, { 'line2d.data': LINE2D.replace('2 4.0 5.0 6.0 5.0', '3 4.0 5.0 6.0 5.0') })).rejects.toThrow(/is not a line segment/);
  });

  it('refuses hybrid with a line sub-style and a Masses section (read_data.html: mass per atom)', async () => {
    await expect(run('units lj\ndimension 2\natom_style hybrid line atomic\n')).rejects.toThrow(/hybrid with a line sub-style is not supported/);
    await expect(run(HEAD2, { 'line2d.data': LINE2D.replace('Atoms # line', 'Masses\n\n1 1.0\n\nAtoms # line') })).rejects.toThrow(/Cannot set mass for atom style line/);
  });

  it('write_data writes the Lines section back so a read_data round trip keeps the segments', async () => {
    const written = new Map<string, string>();
    await run(`${HEAD2}write_data line2d_out.data\n`, { 'line2d.data': LINE2D }, written);
    const text = written.get('line2d_out.data');
    expect(text).toBeDefined();
    expect(text).toMatch(/\n2 lines\n/);
    expect(text).toMatch(/\nLines\n/);
    const back = (await run('units lj\ndimension 2\natom_style line\nread_data line2d_out.data\n', { 'line2d_out.data': text! })).sys.state;
    const bonus = back.custom.get(LINE_BONUS)!.data;
    expect(bonus[0]).toBeCloseTo(1, 10);
    expect(bonus[2]).toBeCloseTo(2, 10);
    expect(back.rmass![1]).toBeCloseTo(3, 10);
  });
});

describe('write_restart and read_restart keep the line and tri data', () => {
  it('restores the line flags, lengths and angles and the triangle corners and quaternions', async () => {
    const written = new Map<string, string>();
    await run(`${HEAD2}write_restart line2d.rst\n`, { 'line2d.data': LINE2D }, written);
    const rst = written.get('line2d.rst');
    expect(rst).toBeDefined();
    const back = (await run('units lj\nread_restart line2d.rst\n', { 'line2d.rst': rst! })).sys.state;
    expect(back.atomStyle).toBe('line');
    expect(Array.from(back.custom.get(LINE_FLAG)!.data.subarray(0, 3))).toEqual([1, 1, 0]);
    expect(back.custom.get(LINE_BONUS)!.data[2]).toBeCloseTo(2, 12);
    expect(back.custom.get(LINE_BONUS)!.data[3]).toBeCloseTo(0, 12);
  });

  it('keeps the triangle quaternion, angular momentum and corner data', async () => {
    const tri = `triangle

1 atoms
1 atom types
1 triangles

0 8 xlo xhi
0 8 ylo yhi
0 8 zlo zhi

Atoms # tri

1 1 1 1 1.0 1.833333 2.333333 2.0

Velocities

1 0 0 0 0 0 0 0.1 0.2 0.3

Triangles

1 1.0 2.0 2.0 2.5 2.0 2.0 2.0 3.0 2.0
`;
    const written = new Map<string, string>();
    const src = (await run('units lj\ndimension 3\natom_style tri\nread_data tri.data\nwrite_restart tri.rst\n', { 'tri.data': tri }, written)).sys.state;
    const back = (await run('units lj\nread_restart tri.rst\n', { 'tri.rst': written.get('tri.rst')! })).sys.state;
    expect(back.atomStyle).toBe('tri');
    expect(Array.from(back.angmom!)).toEqual(Array.from(src.angmom!));
    expect(Array.from(back.quat!)).toEqual(Array.from(src.quat!));
    expect(Array.from(back.custom.get(TRI_BONUS)!.data)).toEqual(Array.from(src.custom.get(TRI_BONUS)!.data));
    expect(back.custom.get(TRI_FLAG)!.data[0]).toBe(1);
  });
});

describe('atom_style tri in read_data', () => {
  it('reads Triangles (corners) and Velocities (omega and angmom) and sets the flag and mass', async () => {
    const tri = `triangles

2 atoms
2 atom types
1 triangles

0 8 xlo xhi
0 8 ylo yhi
0 8 zlo zhi

Atoms # tri

1 1 1 1 1.0 1.833333 2.333333 2.0
2 2 1 0 1.0 5.0 5.0 5.0

Velocities

1 0.1 0 0 0 0 0 0.1 0.2 0.3
2 0 0 0 0 0 0 0 0 0

Triangles

1 1.0 2.0 2.0 2.5 2.0 2.0 2.0 3.0 2.0
`;
    const s = (await run('units lj\ndimension 3\natom_style tri\nread_data tri.data\n', { 'tri.data': tri })).sys.state;
    expect(Array.from(s.custom.get(TRI_FLAG)!.data.subarray(0, 2))).toEqual([1, 0]);
    // corners (1,2,2), (2.5,2,2), (2,3,2): area 0.5 * 1.5 * 1 * ... = 0.75; mass = density * area
    expect(s.rmass![0]).toBeCloseTo(0.75, 10);
    expect(s.angmom![0]).toBeCloseTo(0.1, 12);
    expect(s.angmom![2]).toBeCloseTo(0.3, 12);
    // the atom is moved to the centroid of its corners
    expect(s.x[0]).toBeCloseTo((1 + 2.5 + 2) / 3, 10);
    expect(s.x[1]).toBeCloseTo((2 + 2 + 3) / 3, 10);
    expect(s.quat![0] ** 2 + s.quat![1] ** 2 + s.quat![2] ** 2 + s.quat![3] ** 2).toBeCloseTo(1, 12);
    expect(s.custom.get(TRI_BONUS)!.data.length).toBe(2 * TRI_BONUS_COLS);
  });

  it('refuses tri in 2d and a degenerate Triangles entry', async () => {
    await expect(run('units lj\ndimension 2\natom_style tri\nread_data tri.data\n', { 'tri.data': 'x\n\n1 atoms\n1 atom types\n\n0 1 xlo xhi\n0 1 ylo yhi\n-0.5 0.5 zlo zhi\n\nAtoms\n\n1 1 1 1 1.0 0.5 0.5 0\n' })).rejects.toThrow(/Atom_style tri can only be used in 3d simulations/);
  });
});
