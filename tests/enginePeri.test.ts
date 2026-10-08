import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Peridynamics (atom_style peri, pair_style peri/pmb and peri/lps, compute damage/atom and dilatation/atom):
 * atoms.ts, output/data.ts, commands/setup.ts, force/pair/peri.ts, compute/peri.ts. Values and error texts were
 * measured with native LAMMPS (black box); the full runs are the oracle cases tests/oracle/w22peri_*.in.
 */

const newSession = () => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: (n, t) => files.set(n, t) });
  return { session, files, events };
};

const thermoRows = (events: EngineEvent[]): ThermoRow[] =>
  events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

const BOX = `units si
boundary s s s
atom_style peri
atom_modify map array
neighbor 0.0010 bin
lattice sc 0.0005
region target block 0 2 0 2 0 2 units lattice
create_box 1 target
create_atoms 1 box
`;

/** Two atoms 0.0005 apart in the data file; atom 2 is displaced along x after the read (x0 keeps the data positions). */
const TWO_ATOMS = (dx: number, extra = '') => `units si
boundary f f f
atom_style peri
atom_modify map array
neighbor 0.0010 bin
lattice sc 0.0005
read_data w22peri_two.data
group two id 2
displace_atoms two move ${dx} 0 0 units box
pair_style peri/pmb
pair_coeff * * 1.6863e22 0.0015001 0.0005 0.25
${extra}
`;

const TWO_DATA = `two peri atoms

2 atoms
1 atom types

0.0 0.01 xlo xhi
0.0 0.01 ylo yhi
0.0 0.01 zlo zhi

Atoms # peri

1 1 1.25e-10 2200 0.005 0.005 0.005
2 1 1.25e-10 2200 0.0055 0.005 0.005
`;

describe('atom_style peri', () => {
  it('creates particles with volume 1, mass (density) 1 and the reference positions of their creation', async () => {
    const { session } = newSession();
    await session.execute(BOX);
    const s = session.sys.state;
    expect(s.n).toBe(27);
    expect(Array.from(s.vfrac!.subarray(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(s.rmass!.subarray(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(s.x0!.subarray(0, 6))).toEqual(Array.from(s.x.subarray(0, 6)));
    expect(s.radius).toBeNull();
  });

  it('refuses the per-type mass command (Cannot set per-type atom mass for atom style peri)', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BOX}mass 1 2.0\n`)).rejects.toThrow('Cannot set per-type atom mass for atom style peri');
  });

  it('set density gives the mass; set volume keeps the mass (set.html: "this command does not adjust the particle mass")', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}set group all density 2200\nset group all volume 1.25e-10\n`);
    const s = session.sys.state;
    expect(s.rmass![0]).toBe(2200);
    expect(s.vfrac![26]).toBe(1.25e-10);
    expect(s.rmass![26]).toBe(2200);
  });

  it('refuses set volume on other atom styles and a zero volume (measured messages)', async () => {
    const { session } = newSession();
    await expect(session.execute('units si\natom_style atomic\nlattice sc 0.0005\nregion b block 0 2 0 2 0 2 units lattice\ncreate_box 1 b\ncreate_atoms 1 box\nset group all volume 1.0e-10\n'))
      .rejects.toThrow('Cannot set attribute volume for atom style atomic');
    const peri = newSession();
    await expect(peri.session.execute(`${BOX}set group all volume 0.0\n`)).rejects.toThrow('Invalid volume in set command');
  });

  it('writes id type volume density x y z with no Masses section (write_data)', async () => {
    const { session, files } = newSession();
    await session.execute(`${BOX}set group all density 2200\nset group all volume 1.25e-10\nwrite_data w22peri_out.data\n`);
    const text = files.get('w22peri_out.data')!;
    expect(text).toContain('Atoms # peri');
    expect(text).not.toContain('Masses');
    expect(text.split('\n').find((l) => /^1 1 /.test(l))).toBe('1 1 1.25e-10 2200 0 0 0 0 0 0');
  });

  it('reads the volume and the density (= mass) columns of an Atoms section, even with volume 0', async () => {
    const { session } = newSession();
    const data = TWO_DATA.replace('1 1 1.25e-10 2200 0.005', '1 1 0 2200 0.005');
    session.addFile('w22peri_two_zero.data', data);
    await session.execute(`units si\nboundary f f f\natom_style peri\natom_modify map array\nread_data w22peri_two_zero.data\n`);
    const s = session.sys.state;
    expect(s.rmass![0]).toBe(2200);
    expect(s.vfrac![0]).toBe(0);
    expect(s.vfrac![1]).toBe(1.25e-10);
  });
});

describe('pair_style peri/pmb and peri/lps', () => {
  it('a contact below the short-range distance gives 0.5 (c_S/delta) V (r - d)^2 (measured: 237120129.49 at r = 0.0003)', async () => {
    const { session, events } = newSession();
    session.addFile('w22peri_two.data', TWO_DATA);
    await session.execute(TWO_ATOMS(-0.0002, 'thermo_style custom step pe evdwl\nthermo_modify format float %.17g norm no\nrun 0\n'));
    const row = thermoRows(events)[0];
    expect(Math.abs(row.evdwl - 237120129.491366) / 237120129.491366).toBeLessThan(1e-9);
    expect(row.pe).toBe(row.evdwl);
  });

  it('peri/lps: a stretched bond gives (K/2) sum theta^2 (measured: 5364.00000002935 for two atoms with K = 14.9e9)', async () => {
    const { session, events } = newSession();
    session.addFile('w22peri_two.data', TWO_DATA);
    await session.execute(`units si
boundary f f f
atom_style peri
atom_modify map array
neighbor 0.0010 bin
lattice sc 0.0005
read_data w22peri_two.data
group two id 2
velocity two set 1.0 0 0 units box
pair_style peri/lps
pair_coeff * * 14.9e9 7.0e9 0.0015001 0.0005 0.25
fix 1 all nve
timestep 1e-7
thermo_style custom step pe evdwl ke
thermo_modify format float %.15g norm no
thermo 1
run 1
`);
    const last = thermoRows(events).at(-1)!;
    expect(Math.abs(last.pe - 5364.00000002935) / 5364).toBeLessThan(1e-9);
  });

  it('refuses the unknown peridynamic styles by name and a pair_coeff with missing values', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BOX}pair_style peri/ves\n`)).rejects.toThrow('peri/ves');
    const second = newSession();
    await expect(second.session.execute(`${BOX}pair_style peri/pmb\npair_coeff * * 1.6863e22 0.0015001\n`)).rejects.toThrow('usage: pair_coeff');
  });

  it('compute damage/atom needs a peridynamic pair style', async () => {
    const { session } = newSession();
    await expect(session.execute(`${BOX}compute 1 all damage/atom\ncompute 2 all reduce sum c_1\nthermo_style custom step c_2\nrun 0\n`))
      .rejects.toThrow('compute damage/atom needs a peridynamic pair style');
  });

  it('bonds are built once; damage is zero before any bond breaks and the family does not change between runs', async () => {
    const { session, events } = newSession();
    const input = `${BOX}set group all density 2200
set group all volume 1.25e-10
pair_style peri/pmb
pair_coeff * * 1.6863e22 0.0015001 0.0005 0.25
compute 1 all damage/atom
compute 2 all reduce sum c_1
thermo_style custom step pe c_2
thermo_modify format float %.15g norm no
thermo 1
run 1
run 1
`;
    await session.execute(input);
    const rows = thermoRows(events);
    expect(rows.length).toBe(4);
    expect(rows.every((r) => r.c_2 === 0)).toBe(true);
    expect(rows.every((r) => r.pe === 0)).toBe(true);
  });

  it('two bonded atoms at rest give no damage and no energy (the family is built from the reference positions)', async () => {
    const { session, events } = newSession();
    session.addFile('w22peri_two.data', TWO_DATA);
    await session.execute(TWO_ATOMS(0, 'compute 1 all damage/atom\ncompute 2 all reduce sum c_1\nthermo_style custom step pe evdwl c_2\nthermo_modify format float %.15g norm no\nrun 3\n'));
    const rows = thermoRows(events);
    expect(rows.at(-1)!.pe).toBe(0);
    expect(rows.at(-1)!.c_2).toBe(0);
  });
});
