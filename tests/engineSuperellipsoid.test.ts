import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { superellipsoidInertia } from '../src/engine/superellipsoid';

/*
 * atom_style ellipsoid superellipsoid (core: atoms.ts, superellipsoid.ts, commands/setup.ts set block and set shape,
 * compute/asphere.ts, compute/peratom.ts). Inertia values and the set-order behaviour were measured with native LAMMPS
 * (black box); the contact model is tested in engineSuperellipsoidContact.test.ts.
 */

const newSession = () => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  return session;
};

const BOX = `units si
atom_style ellipsoid superellipsoid
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 5 5 5
`;

describe('superellipsoid inertia (closed form)', () => {
  it('reduces to the ellipsoid moments m/5 (b^2+c^2) etc. for block 2 2', () => {
    const [Ix, Iy, Iz] = superellipsoidInertia(1, 2, 0.5, 2, 2, 3);
    expect(Ix).toBeCloseTo(3 * (4 + 0.25) / 5, 13);
    expect(Iy).toBeCloseTo(3 * (1 + 0.25) / 5, 13);
    expect(Iz).toBeCloseTo(3 * (1 + 4) / 5, 13);
  });

  it('matches native LAMMPS (black box) for shape 2 1 1 with block 3 4 and mass 1', () => {
    // compute property/atom inertiax, inertiay, inertiaz of one atom: 0.127667540830386 0.324596661116689 0.328215200477172
    const [Ix, Iy, Iz] = superellipsoidInertia(1, 0.5, 0.5, 3, 4, 1);
    expect(Ix).toBeCloseTo(0.127667540830386, 13);
    expect(Iy).toBeCloseTo(0.324596661116689, 13);
    expect(Iz).toBeCloseTo(0.328215200477172, 13);
  });
});

describe('atom_style ellipsoid superellipsoid', () => {
  it('is accepted, and the plain ellipsoid style still refuses arguments', async () => {
    const s = newSession();
    await s.execute(BOX);
    expect(s.sys.state.atomStyle).toBe('ellipsoid superellipsoid');
    await expect(newSession().execute('atom_style ellipsoid superellipsoid extra\n')).rejects.toThrow(/usage: atom_style ellipsoid superellipsoid/);
    await expect(newSession().execute('atom_style ellipsoid foo\n')).rejects.toThrow(/takes no arguments/);
  });

  it('starts every atom as a regular ellipsoid (block 2 2) with zero inertia', async () => {
    const s = newSession();
    await s.execute(BOX);
    const st = s.sys.state;
    expect(Array.from(st.custom.get('#sblock')!.data)).toEqual([2, 2]);
    expect(Array.from(st.custom.get('#sinertia')!.data)).toEqual([0, 0, 0]);
  });

  it('stores the inertia with the mass of the atom at the time of set shape (later set mass does not change it)', async () => {
    const s = newSession();
    await s.execute(`${BOX}set type 1 shape 2 1 1\nset type 1 block 3 4\nset type 1 mass 3.0\n`);
    const I = Array.from(s.sys.state.custom.get('#sinertia')!.data);
    expect(I[0]).toBeCloseTo(0.127667540830386, 13);
    expect(I[1]).toBeCloseTo(0.324596661116689, 13);
    expect(I[2]).toBeCloseTo(0.328215200477172, 13);
  });

  it('uses the current mass for a block set after the mass (native, black box: mass 3 gives 3 times the mass-1 value)', async () => {
    const s = newSession();
    await s.execute(`${BOX}set type 1 shape 2 1 1\nset type 1 mass 3.0\nset type 1 block 3 4\n`);
    expect(s.sys.state.custom.get('#sinertia')!.data[0]).toBeCloseTo(3 * 0.127667540830386, 12);
  });

  it('set block alone gives a point particle the default diameter 1 (native, black box: shape 1 1 1)', async () => {
    const s = newSession();
    await s.execute(`${BOX}set type 1 block 2 3\n`);
    expect(Array.from(s.sys.state.shape!.subarray(0, 3))).toEqual([0.5, 0.5, 0.5]);
    expect(s.sys.state.custom.get('#sblock')!.data[0]).toBe(2);
    expect(s.sys.state.custom.get('#sblock')!.data[1]).toBe(3);
  });

  it('reports block1, block2 and inertia through compute property/atom', async () => {
    const files = new Map<string, string>();
    const s = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
    await s.execute(`${BOX}set type 1 shape 2 1 1\nset type 1 block 3 4\nset type 1 mass 1.0\ncompute q all property/atom block1 block2 inertiax shapex\ndump d all custom 1 p.dump id c_q[*]\ndump_modify d sort id format float %.15g\nrun 0\n`);
    const row = files.get('p.dump')!.trim().split('\n').at(-1)!.split(/\s+/).map(Number);
    expect(row[1]).toBe(3);
    expect(row[2]).toBe(4);
    expect(row[3]).toBeCloseTo(0.127667540830386, 13);
    expect(row[4]).toBe(2);
  });

  it('refuses set block on a plain ellipsoid atom style', async () => {
    const s = newSession();
    await expect(s.execute('units si\natom_style ellipsoid\nregion box block 0 10 0 10 0 10\ncreate_box 1 box\ncreate_atoms 1 single 5 5 5\nset type 1 block 3 4\n')).rejects.toThrow(/Cannot set attribute block for atom style ellipsoid/);
  });

  it('refuses a non-positive block exponent', async () => {
    const s = newSession();
    await expect(s.execute(`${BOX}set type 1 block 0 2\n`)).rejects.toThrow(/Invalid block/);
  });
});
