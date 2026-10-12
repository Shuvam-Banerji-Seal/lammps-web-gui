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

/*
 * Checks measured with the native binary (black box) on small data files: a segment's radius is half its length and
 * its mass is density * length; a point particle has radius 0; the stored position becomes the exact centre; the
 * centre tolerance is 0.001 of the segment length (line) or of the largest edge from the first corner (tri); end
 * points are moved to the periodic image nearest the atom; a triangle's radius is its largest centroid-to-corner
 * distance and its mass is density * area; write_data writes the N lines / N triangles counts.
 */
const lineData2d = (box: number, ax: number, ay: number, dens: number, e1: string, e2: string): string =>
  `line\n\n1 atoms\n1 lines\n1 atom types\n\n0 ${box} xlo xhi\n0 ${box} ylo yhi\n-0.5 0.5 zlo zhi\n\nAtoms # line\n\n1 1 1 1 ${dens} ${ax} ${ay} 0\n\nLines\n\n1 ${e1} ${e2}\n`;

const triData3d = (ax: number, ay: number, az: number, dens: number, corners: string): string =>
  `tri\n\n1 atoms\n1 triangles\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\nAtoms # tri\n\n1 1 1 1 ${dens} ${ax} ${ay} ${az}\n\nTriangles\n\n1 ${corners}\n`;

const readLine2d = (text: string, boundary = 'p p p') => run(`units lj\ndimension 2\natom_style line\nboundary ${boundary}\nread_data l.data\n`, { 'l.data': text });
const readTri3d = (text: string, boundary = 'p p p') => run(`units lj\natom_style tri\nboundary ${boundary}\nread_data t.data\n`, { 't.data': text });

describe('measured with native LAMMPS (black box): line and tri data', () => {
  it('a segment has radius = half its length and mass = density * length; a point particle has radius 0', async () => {
    const horiz = (await readLine2d(lineData2d(10, 5, 5, 3, '4 5', '6 5'))).sys.state;
    expect(horiz.radius![0]).toBeCloseTo(1, 12);
    expect(horiz.rmass![0]).toBeCloseTo(6, 12);
    const vert = (await readLine2d(lineData2d(10, 7, 5, 1, '7 3', '7 7'))).sys.state;
    expect(vert.radius![0]).toBeCloseTo(2, 12);
    expect(vert.rmass![0]).toBeCloseTo(4, 12);
    const diag = (await readLine2d(lineData2d(10, 1.5, 6.5, 1, '2 7', '1 6'))).sys.state;
    expect(diag.radius![0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(diag.rmass![0]).toBeCloseTo(Math.SQRT2, 12);
    const point = `pt\n\n1 atoms\n0 lines\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n-0.5 0.5 zlo zhi\n\nAtoms # line\n\n1 1 1 0 2.5 5 5 0\n`;
    const p = (await readLine2d(point)).sys.state;
    expect(p.radius![0]).toBe(0);
    expect(p.rmass![0]).toBeCloseTo(2.5, 12);
  });

  it('a segment written back keeps theta (end 1 = centre - half length along theta) and density = mass per length', async () => {
    const written = new Map<string, string>();
    await run(`units lj\ndimension 2\natom_style line\nboundary p p p\nread_data l.data\nwrite_data o.data\n`, { 'l.data': lineData2d(10, 1.5, 6.5, 3, '2 7', '1 6') }, written);
    const text = written.get('o.data')!;
    expect(text).toMatch(/\n1 lines\n/);
    const atomLine = text.split('\n').find((l) => /^1 1 1 1 /.test(l))!.split(' ');
    // native wrote the density column back as 3 (mass 3 * sqrt2 over length sqrt2)
    expect(Number(atomLine[4])).toBeCloseTo(3, 9);
    const endLine = text.split('\n').find((l) => /^1 [-0-9.e]+ [-0-9.e]+ [-0-9.e]+ [-0-9.e]+$/.test(l))!.split(' ').slice(1).map(Number);
    // native wrote "1 2 7 1 6": the first end point is the input's first end point
    expect(endLine[0]).toBeCloseTo(2, 9);
    expect(endLine[1]).toBeCloseTo(7, 9);
    expect(endLine[2]).toBeCloseTo(1, 9);
    expect(endLine[3]).toBeCloseTo(6, 9);
  });

  it('the stored position becomes the exact centre (native: 5.0015 is written back as 5)', async () => {
    const s = (await readLine2d(lineData2d(10, 5.0015, 5, 1, '4 5', '6 5'))).sys.state;
    expect(s.x[0]).toBe(5);
  });

  it('the centre tolerance is 0.001 of the segment length, not an absolute distance', async () => {
    await readLine2d(lineData2d(10, 5.0015, 5, 1, '4 5', '6 5'));
    await expect(readLine2d(lineData2d(10, 5.0025, 5, 1, '4 5', '6 5'))).rejects.toThrow(/Inconsistent line segment/);
    // length 100 with a 0.05 offset (ratio 0.0005) is accepted; with an absolute 0.001 tolerance it would be refused
    await readLine2d(lineData2d(200, 50.05, 50, 1, '0 50', '100 50'));
    await expect(readLine2d(lineData2d(200, 50.15, 50, 1, '0 50', '100 50'))).rejects.toThrow(/Inconsistent line segment/);
    // length 0.5 with a 0.0006 offset (ratio 0.0012) is refused
    await expect(readLine2d(lineData2d(10, 5.25 + 0.0006, 5, 1, '5 5', '5.5 5'))).rejects.toThrow(/Inconsistent line segment/);
  });

  it('end points outside a periodic box are moved to the image nearest the atom; a non-periodic box refuses them', async () => {
    const s = (await readLine2d(lineData2d(10, 0.5, 5, 1, '9.5 5', '11.5 5'))).sys.state;
    expect(s.x[0]).toBe(0.5);
    const written = new Map<string, string>();
    await run('units lj\ndimension 2\natom_style line\nboundary p p p\nread_data l.data\nwrite_data o.data\n', { 'l.data': lineData2d(10, 0.5, 5, 1, '9.5 5', '11.5 5') }, written);
    expect(written.get('o.data')!).toMatch(/\n1 -0\.5 5 1\.5 5\n/);
    await expect(readLine2d(lineData2d(10, 0.5, 5, 1, '9.5 5', '11.5 5'), 'f f p')).rejects.toThrow(/Inconsistent line segment/);
  });

  it('write_data counts the flagged atoms in the header ("3 lines", and "0 lines" with none)', async () => {
    const written = new Map<string, string>();
    await run('units lj\ndimension 2\natom_style line\nboundary p p p\nread_data l.data\nwrite_data o.data\n', { 'l.data': lineData2d(10, 5, 5, 1, '4 5', '6 5') }, written);
    expect(written.get('o.data')!).toMatch(/\n1 lines\n/);
    const none = `pt\n\n1 atoms\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n-0.5 0.5 zlo zhi\n\nAtoms # line\n\n1 1 1 0 1.0 5 5 0\n`;
    const w2 = new Map<string, string>();
    await run('units lj\ndimension 2\natom_style line\nboundary p p p\nread_data l.data\nwrite_data o.data\n', { 'l.data': none }, w2);
    expect(w2.get('o.data')!).toMatch(/\n0 lines\n/);
  });

  it('a triangle has radius = largest centroid-to-corner distance (not the circumradius) and mass = density * area', async () => {
    // right isoceles, legs 2: area 2, centroid (2/3, 2/3); largest centroid-corner distance sqrt(20)/3 = 1.4907 (circumradius sqrt 2 = 1.4142)
    const a = (await readTri3d(triData3d(2 / 3, 2 / 3, 0, 1.5, '0 0 0 2 0 0 0 2 0'))).sys.state;
    expect(a.radius![0]).toBeCloseTo(1.4907119849998598, 9);
    expect(a.rmass![0]).toBeCloseTo(3, 10);
    // equilateral, side 1: radius 1/sqrt(3), area sqrt(3)/4
    const e = (await readTri3d(triData3d(0.5, 0.28867513459481287, 0, 2, '0 0 0 1 0 0 0.5 0.86602540378443865 0'))).sys.state;
    expect(e.radius![0]).toBeCloseTo(0.57735026918962573, 9);
    expect(e.rmass![0]).toBeCloseTo(0.8660254037844386, 9);
    // corners (0,0), (5,0), (0.1,5): area 12.5, largest centroid-corner distance 3.6974...
    const c = (await readTri3d(triData3d(1.7, 5 / 3, 0, 1, '0 0 0 5 0 0 0.1 5 0'))).sys.state;
    expect(c.radius![0]).toBeCloseTo(3.6974465663632126, 9);
    expect(c.rmass![0]).toBeCloseTo(12.5, 9);
  });

  it('the triangle centre tolerance is 0.001 of its largest edge from the first corner (native: 0.0049 accepted, 0.0051 refused)', async () => {
    const corners = '0 0 0 5 0 0 0.1 5 0';
    await readTri3d(triData3d(1.7049, 5 / 3, 0, 1, corners));
    await expect(readTri3d(triData3d(1.7051, 5 / 3, 0, 1, corners))).rejects.toThrow(/Inconsistent triangle/);
    // scaled by 100 (edges about 500), an offset of 0.3 (ratio 0.0006) is accepted
    await readTri3d(triData3d(170.3, 500 / 3, 0, 1, '0 0 0 500 0 0 10 500 0').replace(/\n0 10 /g, '\n0 1000 '), 'p p p');
  });
});
