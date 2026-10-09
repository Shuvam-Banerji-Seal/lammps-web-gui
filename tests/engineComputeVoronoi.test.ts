import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { EngineError, type EngineEvent } from '../src/engine/types';

/*
 * compute voronoi/atom (docs.lammps.org/compute_voronoi_atom.html). A perfect
 * simple-cubic lattice has cubic Voronoi cells: volume a^3 and six faces, one
 * per nearest neighbour ("The number of faces of the Voronoi cell ... is equal to
 * the number of nearest neighbors of the central atom" for the regular case).
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

const SC = (extra = '', boundary = '') => `
units lj
atom_style atomic
lattice sc 0.5
${boundary}
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
compute v all voronoi/atom ${extra}
compute s all reduce sum c_v[1]
thermo_style custom step c_s
run 0
`;

describe('compute voronoi/atom', () => {
  it('gives volume a^3 and six faces on a perfect simple-cubic lattice', async () => {
    const s = await runScript(SC());
    const c = s.sys.compute('v');
    expect(c.style).toBe('voronoi/atom');
    expect(c.peratomFlag).toBe(true);
    expect(c.sizePeratomCols).toBe(2);
    const vals = c.peratomValues();
    const a = Math.cbrt(2); // lattice sc 0.5: a = (1/0.5)^(1/3)
    const n = s.sys.state.n;
    expect(n).toBe(27);
    for (let i = 0; i < n; i++) {
      expect(vals[2 * i]).toBeCloseTo(a ** 3, 9);
      expect(vals[2 * i + 1]).toBe(6);
    }
  });

  it('adds a surface column for the surface keyword and a histogram vector for edge_histo', async () => {
    const s = await runScript(SC('surface all edge_histo 6') + '\n');
    const c = s.sys.compute('v');
    expect(c.sizePeratomCols).toBe(3);
    const a = Math.cbrt(2);
    const vals = c.peratomValues();
    expect(vals[2]).toBeCloseTo(6 * a * a, 9); // surface of a cube: 6 a^2
    expect(c.vectorFlag).toBe(true);
    expect(c.sizeVector).toBe(7);
    const h = c.vectorValues();
    // every cell face is a square with 4 edges, counted once per cell
    // vector entry k (1-based, thermo c_v[k]) counts faces with k edges: 4 edges is h[3]
    expect(h[3]).toBe(27 * 6);
  });

  it('computes the same cells when the search has to grow past the initial guess', async () => {
    const s = await runScript(`
      units lj
      atom_style atomic
      lattice sc 0.5
      region box block 0 2 0 2 0 2
      create_box 1 box
      create_atoms 1 box
      mass 1 1.0
      pair_style lj/cut 2.5
      pair_coeff 1 1 1.0 1.0
      compute v all voronoi/atom
      run 0
    `);
    const vals = s.sys.compute('v').peratomValues();
    const a = Math.cbrt(2);
    expect(vals[0]).toBeCloseTo(a ** 3, 9);
    expect(vals[1]).toBe(6);
  });

  it('rejects unsupported keywords and bad arguments with a named error', async () => {
    const bad = async (text: string, pattern: RegExp) => {
      await expect(runScript(text)).rejects.toThrow(EngineError);
      await expect(runScript(text)).rejects.toThrow(pattern);
    };
    await bad(SC('peratom yes'), /peratom/);
    await bad(SC('frobnicate 1'), /frobnicate/);
    await bad(SC('surface nosuchgroup'), /nosuchgroup/);
    await bad(SC('radius r1'), /radius/);
    await bad(SC('edge_histo 0'), /edge_histo/);
    await bad(SC('face_threshold -1'), /face_threshold/);
    // non-periodic boundaries are supported since wave 31 (walls at the box faces, tests/engineVoronoiNp31.test.ts);
    // a non-periodic triclinic box is still refused
  });
});
