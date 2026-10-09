import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { parseStl } from '../src/engine/stl';

/*
 * create_atoms mesh (src/engine/commands/setup.ts, src/engine/stl.ts).
 * Native parity is in tests/oracle/w35mesh_tri.in and w35mesh_box.in; these
 * checks are the STL reader, the units/radscale/meshmode rules, and the
 * bisect and qrand placement rules quoted from
 * docs.lammps.org/create_atoms.html.
 */

const newSession = () => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
  return { session, files };
};

const TRI = `solid tri
  facet normal 0.0 1.0 0.0
    outer loop
      vertex 2.0 2.0 5.0
      vertex 4.0 2.0 5.0
      vertex 3.0 5.0 5.0
    endloop
  endfacet
endsolid tri
`;

const BOX = `units lj
atom_style sphere
region box block 0 10 0 10 0 10 units box
create_box 1 box
`;

const run = async (body: string, stl = TRI) => {
  const { session } = newSession();
  session.addFile('m.stl', stl);
  await session.execute(`${BOX}${body}\n`);
  return session.sys.state;
};

describe('stl parser', () => {
  it('reads one triangle from the vertex lines', () => {
    const tris = parseStl(TRI, 'm.stl');
    expect(tris).toHaveLength(1);
    expect(tris[0].v).toEqual([[2, 2, 5], [4, 2, 5], [3, 5, 5]]);
  });

  it('rejects a binary STL and names the conversion tool', () => {
    // 80-byte header that does not start with "solid", then a facet count
    const binary = '\u0000binary\u0000header' + 'z'.repeat(60);
    expect(() => parseStl(binary, 'part.stl')).toThrow(/binary STL.*stl_bin2txt/);
  });
});

describe('create_atoms mesh bisect', () => {
  it('places one atom at the centroid with radius = mean vertex distance', async () => {
    // create_atoms.html: "a particle is created at the center of each
    // triangle unless the average distance of the triangle vertices from its
    // center is larger than the radthresh value"
    const s = await run('create_atoms 1 mesh m.stl meshmode bisect 4.0 units box\n');
    expect(s.n).toBe(1);
    expect(s.x[0]).toBeCloseTo(3, 12);
    expect(s.x[1]).toBeCloseTo(3, 12);
    expect(s.x[2]).toBeCloseTo(5, 12);
    const avg = (Math.hypot(1, 1, 0) * 2 + 2) / 3;
    expect(s.radius![0]).toBeCloseTo(avg, 12);
  });

  it('splits along the longest side and assigns radii per native order', async () => {
    // Measured with native LAMMPS (black box): radthresh 1.0 on this triangle
    // gives four atoms, in this order, with these radii.
    const s = await run('create_atoms 1 mesh m.stl meshmode bisect 1.0 units box\n');
    expect(s.n).toBe(4);
    const want = [
      [2.66666666666667, 3, 5, 0.900240672552314],
      [3, 4, 5, 0.804737854124365],
      [2.91666666666667, 2.25, 5, 0.862999267651513],
      [3.41666666666667, 2.75, 5, 0.790476094142466],
    ];
    for (let i = 0; i < 4; i++) {
      for (let d = 0; d < 3; d++) expect(s.x[3 * i + d]).toBeCloseTo(want[i][d], 10);
      expect(s.radius![i]).toBeCloseTo(want[i][3], 10);
    }
  });

  it('scales the radius by radscale', async () => {
    const s = await run('create_atoms 1 mesh m.stl meshmode bisect 4.0 units box radscale 0.5\n');
    const avg = (Math.hypot(1, 1, 0) * 2 + 2) / 3;
    expect(s.radius![0]).toBeCloseTo(0.5 * avg, 12);
  });

  it('requires units box', async () => {
    const { session } = newSession();
    session.addFile('m.stl', TRI);
    await expect(session.execute(`${BOX}create_atoms 1 mesh m.stl meshmode bisect 4.0\n`))
      .rejects.toThrow(/units box/);
  });

  it('rejects an unknown meshmode', async () => {
    const { session } = newSession();
    session.addFile('m.stl', TRI);
    await expect(session.execute(`${BOX}create_atoms 1 mesh m.stl meshmode spiral 1.0 units box\n`))
      .rejects.toThrow(/unknown meshmode/);
  });
});

describe('create_atoms mesh qrand', () => {
  it('uses ceil(density*area) atoms with total disk area equal to the triangle', async () => {
    // create_atoms.html: "Particles are added to the triangle until the
    // minimum number density is met or exceeded such that every triangle will
    // have at least one particle." "The radius will be set so that the sum of
    // the area of the radius of the particles created in place of a triangle
    // will be equal to the area of that triangle."
    const s = await run('create_atoms 1 mesh m.stl meshmode qrand 1.0 units box\n');
    expect(s.n).toBe(3); // area 3, density 1.0 -> 3
    const r = Math.sqrt(3 / (3 * Math.PI));
    for (let i = 0; i < 3; i++) expect(s.radius![i]).toBeCloseTo(r, 12);
    // the first point of the (Roberts) R2 barycentric sequence, measured natively
    expect(s.x[0]).toBeCloseTo(2.3945583, 6);
    expect(s.x[1]).toBeCloseTo(2.7646331, 6);
    expect(s.x[2]).toBeCloseTo(5, 12);
  });

  it('always creates at least one atom per triangle', async () => {
    const s = await run('create_atoms 1 mesh m.stl meshmode qrand 0.001 units box\n');
    expect(s.n).toBe(1);
  });
});
