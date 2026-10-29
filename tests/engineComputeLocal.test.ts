import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute bond/local, angle/local, dihedral/local, improper/local, property/local and the
 * dump local style: argument errors and values of a 3-atom molecule (atoms 1-2-3, bonds 1-2
 * and 2-3, one angle centred on atom 2). The reference values are worked out by hand from
 * docs.lammps.org/compute_bond_local.html, compute_angle_local.html and dump.html.
 */

const close = (got: number, want: number, rel = 1e-10) => Math.abs(got - want) <= rel * Math.max(1, Math.abs(got), Math.abs(want));

const DATA = `LAMMPS data

3 atoms
1 atom types
2 bonds
1 bond types
1 angles
1 angle types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 1.0 1.0 1.0
2 1 1 2.0 1.0 1.0
3 1 1 2.0 2.0 1.0

Velocities

1 0.5 0.0 0.0
2 0.0 0.0 0.0
3 0.0 0.5 0.0

Bonds

1 1 1 2
2 1 2 3

Angles

1 1 1 2 3
`;

const molecule = async (extra: string) => {
  const events: EngineEvent[] = [];
  const writes = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => writes.set(n, (ap ? writes.get(n) ?? '' : '') + t),
  });
  session.addFile('mol3.data', DATA);
  await session.execute(`
    units lj
    atom_style molecular
    read_data mol3.data
    bond_style harmonic
    bond_coeff 1 100.0 1.0
    angle_style harmonic
    angle_coeff 1 10.0 90.0
    pair_style lj/cut 2.5
    pair_coeff 1 1 1.0 1.0
    ${extra}
    run 0
  `);
  return { session, writes };
};

describe('local computes: argument errors', () => {
  it('rejects unknown values, v_name and set misuse with the native texts, invalid bN and omega', async () => {
    await expect(molecule('compute c all bond/local bogus')).rejects.toThrow(/Unknown compute bond\/local keyword: bogus/);
    // src/compute_bond_local.cpp, compute_angle_local.cpp, compute_dihedral_local.cpp at b332f28
    await expect(molecule('compute c all bond/local dist set dist d')).rejects.toThrow(/Compute bond\/local set used with without a variable/);
    await expect(molecule('compute c all bond/local v_d')).rejects.toThrow(/Compute bond\/local variable requires a set variable/);
    await expect(molecule('compute c all bond/local v_nope set dist d')).rejects.toThrow(/Variable name nope for compute bond\/local does not exist/);
    await expect(molecule('variable s string x\ncompute c all bond/local v_s set dist d')).rejects.toThrow(/Variable s for compute bond\/local is invalid style/);
    await expect(molecule('variable e equal 1\nvariable d equal 2\ncompute c all bond/local v_e set dist d')).rejects.toThrow(/Variable for compute bond\/local is invalid style/);
    await expect(molecule('variable e equal 1\ncompute c all bond/local v_e set theta d')).rejects.toThrow(/Unknown compute bond\/local set keyword: d/);
    await expect(molecule('variable e equal 1\ncompute c all angle/local v_e set phi t')).rejects.toThrow(/Illegal compute angle\/local command/);
    await expect(molecule('compute c all dihedral/local phi set phi p')).rejects.toThrow(/Compute dihedral\/local set with no variable/);
    await expect(molecule('compute c all improper/local v_e')).rejects.toThrow(/Unknown compute improper\/local keyword v_e/);
    // bN is a bond-style extra output (compute bond/local): b0 is an invalid keyword, b1 on angles is an unknown value
    await expect(molecule('compute c all bond/local b0')).rejects.toThrow(/Invalid keyword b0/);
    await expect(molecule('compute c all angle/local b1')).rejects.toThrow(/Illegal compute angle\/local command/);
    await expect(molecule('compute c all bond/local omega')).rejects.toThrow(/omega/);
    await expect(molecule('compute c all angle/local dist')).rejects.toThrow(/Illegal compute angle\/local command/);
  });

  it('rejects pair/local pN values of a pair style that defines none, a bad cutoff keyword and radius without radius', async () => {
    // docs.lammps.org/compute_pair_local.html: "Most pair styles do not define any additional quantities, so N = 0."
    // The message is native's (src/compute_pair_local.cpp init at b332f28), which does not name the style or the field.
    await expect(molecule('compute c all pair/local dist p1\ndump d all local 1 pl.txt c_c[1] c_c[2]')).rejects.toThrow(/Pair style does not have extra field requested by compute pair\/local/);
    await expect(molecule('compute c all pair/local dist cutoff bogus')).rejects.toThrow(/Unknown compute pair\/local cutoff keyword: bogus/);
    await expect(molecule('compute c all pair/local dist cutoff radius\ndump d all local 1 pl.txt c_c')).rejects.toThrow(/requires atom attribute radius/);
    await expect(molecule('compute c all pair/local dist p0')).rejects.toThrow(/Invalid keyword p0 in compute pair\/local command/);
    // cutoff type is the default and is accepted (the rows are checked in tests/oracle/w172pl_ljcut.in)
    await expect(molecule('compute c all pair/local dist cutoff type\ndump d all local 1 pl.txt c_c')).resolves.toBeDefined();
    // pair attributes are supported (tests/oracle/propl_*.in); kinds cannot be mixed (native's message)
    await expect(molecule('compute c all property/local patom1 patom2')).resolves.toBeDefined();
    await expect(molecule('compute c all property/local batom1 aatom1')).rejects.toThrow(/Compute property\/local cannot use these inputs together/);
  });

  it('pair granular: p1-p12 are the contact values, p13 (the mdr contact area) is refused without the mdr model', async () => {
    // docs.lammps.org/pair_granular.html: "These extra quantities can be accessed by the compute pair/local command, as p1, p2, ..., p12."
    // (the mdr contact area is p13 only with a pair_coeff that uses the mdr model)
    const run = async (extra: string) => {
      const writes = new Map<string, string>();
      const session = new Session({ emit: () => {}, writeFile: (n, t, ap) => writes.set(n, (ap ? writes.get(n) ?? '' : '') + t) });
      await session.execute(`
        units lj
        atom_style sphere
        comm_modify vel yes
        region b block -5 5 -5 5 -5 5
        create_box 1 b
        create_atoms 1 single 0 0 0
        create_atoms 1 single 0.9 0 0
        set type 1 diameter 1.0 density 1.0
        pair_style granular
        pair_coeff * * hooke 1000.0 0.5 tangential linear_history 300.0 0.0 0.5
        ${extra}
        run 0
      `);
      return writes;
    };
    await expect(run('compute c all pair/local p13 cutoff radius\ndump d all local 1 pl.txt c_c')).rejects.toThrow(/Pair style does not have extra field requested by compute pair\/local/);
    const w = await run('compute c all pair/local dist p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 cutoff radius\ndump d all local 1 pl.txt c_c[*]');
    // one pair in contact (overlap 0.1, no history yet): the dump row is dist, p1..p12 (no index column); p10..p12 are
    // x_i - x_j (+-0.9, 0, 0) and the tangential force p1..p3 is zero at the first contact step (no history, no velocity)
    const last = w.get('pl.txt')!.trim().split('\n').pop()!.trim().split(/\s+/).map(Number);
    expect(last.length).toBe(13);
    expect(close(last[0], 0.9, 1e-12)).toBe(true);
    expect(close(Math.abs(last[10]), 0.9, 1e-12)).toBe(true);
    expect(Math.abs(last[11])).toBe(0);
    expect(Math.abs(last[12])).toBe(0);
    expect(Math.abs(last[1])).toBe(0);
  });

  it('rejects a dump column that the compute cannot give', async () => {
    await expect(molecule('compute v all bond/local dist\ndump d all local 1 x.txt c_v[1]')).rejects.toThrow(/does not calculate local array/);
    await expect(molecule('compute a all bond/local dist engpot\ndump d all local 1 x.txt c_a[3]')).rejects.toThrow(/Dump local compute a vector is accessed out-of-range/);
  });
});

describe('local computes: values', () => {
  it('bond/local gives distance, separation, energy, force and vibrational energies', async () => {
    const { session } = await molecule('compute b all bond/local dist dx dy dz engpot force fx fy fz engvib engrot engtrans velvib');
    const c = session.sys.compute('b');
    expect(c.localRows).toBe(0);
    const v = c.localValues();
    expect(c.localRows).toBe(2);
    expect(c.sizeLocalCols).toBe(13);
    // bond 1 (atoms 1-2): r = 1, at r0, so no energy and no force
    expect(close(v[0], 1)).toBe(true);
    expect(close(v[4], 0)).toBe(true);
    // bond 2 (atoms 2-3): r = 1, dx dy dz = x2 - x3 = (0, -1, 0)
    expect(close(v[13 + 0], 1)).toBe(true);
    expect(close(v[13 + 2], -1)).toBe(true);
    // bond 1 velocities: v1 = (0.5,0,0), v2 = 0, m = 1 each; velvib = (v1 - v2) . (x1 - x2)/r = -0.5
    expect(close(v[12], -0.5)).toBe(true);
    // engtrans = 0.5 (m1+m2) |vcm|^2 with vcm = (0.25, 0, 0): 0.5 * 2 * 0.0625
    expect(close(v[11], 0.0625)).toBe(true);
    // engvib: along the bond the relative velocities are -0.25 and +0.25: 0.5*(0.0625+0.0625)
    expect(close(v[9], 0.0625)).toBe(true);
  });

  it('angle/local gives the angle in degrees and its energy', async () => {
    const { session } = await molecule('compute a all angle/local theta eng');
    const v = session.sys.compute('a').localValues();
    expect(session.sys.compute('a').localRows).toBe(1);
    expect(close(v[0], 90)).toBe(true);
    expect(close(v[1], 0)).toBe(true);
  });

  it('property/local gives ids and types of bonds', async () => {
    const { session } = await molecule('compute p all property/local batom1 batom2 btype');
    const v = session.sys.compute('p').localValues();
    expect(Array.from(v)).toEqual([1, 2, 1, 2, 3, 1]);
  });

  it('dump local writes the header and one line per entry with a trailing space', async () => {
    const { writes } = await molecule('compute b all bond/local dist\ncompute p all property/local batom1 batom2\ndump d all local 1 out.local index c_p[*] c_b\ndump_modify d label BONDS\nrun 0');
    const text = writes.get('out.local') ?? '';
    const lines = text.split('\n');
    expect(lines[2]).toBe('ITEM: NUMBER OF BONDS');
    expect(lines[3]).toBe('2');
    expect(lines.find((l) => l.startsWith('ITEM: BONDS'))).toBe('ITEM: BONDS index c_p[1] c_p[2] c_b');
    const rows = lines.slice(lines.findIndex((l) => l.startsWith('ITEM: BONDS')) + 1).filter((l) => l.length);
    expect(rows).toEqual(['1 1 2 1 ', '2 2 3 1 ']);
  });
});
