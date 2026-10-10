import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute hbond/local (docs.lammps.org/compute_hbond_local.html): argument refusals and one analytic
 * geometry. Donor D (type 1, q 0) at the origin with a bonded hydrogen H (type 2, q +0.5) at (1,0,0);
 * acceptor A (type 1, q -1.0, another molecule) at (2,2,0). The H-D-A angle is 45 degrees at D,
 * D-A = 2 sqrt(2), H-A = sqrt(5). Only the Coulomb term is non-zero (lj epsilon 0), so
 * ehb = -(E_DA + E_HA) = 332.06371 * 0.5 / sqrt(5), a positive (attractive) value in real units.
 */

const close = (got: number, want: number, rel = 1e-10) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

const DATA = (bonded: boolean) => `LAMMPS data

3 atoms
1 bonds
2 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 15.9994
2 1.00794

Atoms # full

1 1 1 0.0 0.0 0.0 0.0
2 1 2 0.5 1.0 0.0 0.0
3 2 1 -1.0 2.0 2.0 0.0

${bonded ? 'Bonds\n\n1 1 1 2\n' : 'Bonds\n\n1 1 1 3\n'}`;

const setup = async (extra: string, bonded = true) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('hb3.data', DATA(bonded));
  await session.execute(`
    units real
    atom_style full
    read_data hb3.data
    pair_style lj/cut/coul/cut 10.0
    pair_coeff * * 0.0 1.0
    bond_style harmonic
    bond_coeff 1 100.0 1.0
    group o type 1
    group h type 2
    ${extra}
    run 0
  `);
  return session;
};

describe('compute hbond/local: argument refusals', () => {
  it('rejects missing arguments, non-positive cutoffs and angles above 90 degrees', async () => {
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o')).rejects.toThrow(/missing argument/);
    await expect(setup('compute hb all hbond/local 0.0 30.0 o o h')).rejects.toThrow(/distance cutoff must be > 0/);
    await expect(setup('compute hb all hbond/local 3.3 0.0 o o h')).rejects.toThrow(/angle cutoff must be > 0/);
    await expect(setup('compute hb all hbond/local 3.3 91.0 o o h')).rejects.toThrow(/must not be > 90\.0 degrees/);
  });

  it('rejects an unknown keyword, a negative ecut, a missing ecut value and values after ecut', async () => {
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o h foo')).rejects.toThrow(/Unknown compute hbond\/local keyword foo/);
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o h dist ecut -1.0')).rejects.toThrow(/ecut must be >= 0/);
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o h dist ecut')).rejects.toThrow(/ecut command: missing/);
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o h dist ecut 1.0 ehb')).rejects.toThrow(/Unknown compute hbond\/local keyword ehb/);
  });

  it('rejects a group that does not exist', async () => {
    await expect(setup('compute hb all hbond/local 3.3 30.0 o o nope dist')).rejects.toThrow(/nope/);
  });
});

describe('compute hbond/local: analytic geometry', () => {
  it('reports one bond with the distances, the angle at the donor and the energy', async () => {
    const session = await setup('compute hb all hbond/local 3.0 50.0 o o h dist hdist angle ehb');
    const c = session.sys.compute('hb');
    expect(c.sizeLocalCols).toBe(7);
    expect(c.scalarValue()).toBe(1);
    const v = c.localValues();
    expect(c.localRows).toBe(1);
    // columns: H id, D id, A id, dist, hdist, angle, ehb
    expect(v[0]).toBe(2);
    expect(v[1]).toBe(1);
    expect(v[2]).toBe(3);
    expect(close(v[3], 2 * Math.SQRT2)).toBe(true);
    expect(close(v[4], Math.sqrt(5))).toBe(true);
    expect(close(v[5], 45)).toBe(true);
    expect(close(v[6], (332.06371 * 0.5) / Math.sqrt(5), 1e-9)).toBe(true);
  });

  it('applies the distance cutoff (2 sqrt 2 is excluded by 2.8) and the angle cutoff', async () => {
    const tooShort = await setup('compute hb all hbond/local 2.8 50.0 o o h dist');
    expect(tooShort.sys.compute('hb').scalarValue()).toBe(0);
    const tight = await setup('compute hb all hbond/local 3.0 44.0 o o h dist');
    expect(tight.sys.compute('hb').scalarValue()).toBe(0);
    // the computed angle is 45.000000000000007 degrees, so an acut of exactly 45.0 excludes it
    // (measured with native LAMMPS, black box: the same geometry gives no row with acut 45.0 and one with 45.0000001)
    const exact = await setup('compute hb all hbond/local 3.0 45.0 o o h dist');
    expect(exact.sys.compute('hb').scalarValue()).toBe(0);
    const above = await setup('compute hb all hbond/local 3.0 45.0000001 o o h dist');
    expect(above.sys.compute('hb').scalarValue()).toBe(1);
  });

  it('counts a hydrogen only when it is bonded to the donor', async () => {
    const session = await setup('compute hb all hbond/local 3.0 50.0 o o h dist', false);
    expect(session.sys.compute('hb').scalarValue()).toBe(0);
  });

  it('requires all three atoms in the compute group', async () => {
    const session = await setup('group sel id 1 2\ncompute hb sel hbond/local 3.0 50.0 o o h dist');
    expect(session.sys.compute('hb').scalarValue()).toBe(0);
  });

  it('reports three columns with no value keywords', async () => {
    const session = await setup('compute hb all hbond/local 3.0 50.0 o o h');
    const c = session.sys.compute('hb');
    expect(c.sizeLocalCols).toBe(3);
    expect(Array.from(c.localValues())).toEqual([2, 1, 3]);
  });
});
