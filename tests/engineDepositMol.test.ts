import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix deposit mol (src/engine/fix/deposit.ts): molecule insertion with its
 * orientation randomisation, type offset, molecule IDs and bonds; the rigid
 * and shake keywords; and the argument errors. The random stream itself is
 * pinned by the w26depmol_* oracle cases; here the first molecule of a seed
 * 12345 run is checked against the values recovered from native LAMMPS.
 */

const DIMER = `# dimer molecule
2 atoms
1 bonds

Coords

1 0.0 0.0 0.0
2 1.0 0.0 0.0

Types

1 2
2 3

Bonds

1 1 1 2

Special Bond Counts

1 1 0 0
2 1 0 0

Special Bonds

1 2
2 1
`;

const run = async (script: string, files: Record<string, string> = { dimer: DIMER }) => {
  const events: EngineEvent[] = [];
  const out = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => out.set(n, (ap ? out.get(n) ?? '' : '') + t),
  });
  for (const [n, t] of Object.entries(files)) session.addFile(n, t);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { error, files: out, logs };
};

/** Atom rows (id, mol, type, x, y, z) of a write_dump file. */
const dumpAtoms = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    return { id: w[0], mol: w[1], type: w[2], x: w[3], y: w[4], z: w[5] };
  });
};

const base = `units lj
atom_style bond
boundary p p f
region box block 0 10 0 10 0 10
create_box 3 box bond/types 1 extra/bond/per/atom 1
molecule dimer dimer
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
mass * 1.0
bond_style harmonic
bond_coeff 1 5.0 1.0
region slab block 0 10 0 10 8 9
`;

describe('fix deposit mol', () => {
  it('inserts a molecule with the measured position and orientation stream', async () => {
    const { error, files } = await run(`${base}fix 1 all deposit 1 0 1 12345 region slab mol dimer units box
run 1
write_dump all custom out.dump id mol type x y z modify format float %.15g sort id
`);
    expect(error).toBeNull();
    const atoms = dumpAtoms(files.get('out.dump') ?? '');
    expect(atoms.map((a) => [a.id, a.mol, a.type])).toEqual([[1, 1, 2], [2, 1, 3]]);
    // geometric center (9.10652, 3.348, 8.98221) and the dimer axis from
    // native LAMMPS for seed 12345, first deposit
    const want = [[8.68708, 3.57168, 9.13725], [9.52596, 3.12432, 8.82716]];
    atoms.forEach((a, i) => {
      expect(a.x).toBeCloseTo(want[i][0], 5);
      expect(a.y).toBeCloseTo(want[i][1], 5);
      expect(a.z).toBeCloseTo(want[i][2], 5);
    });
  });

  it('offsets the template types by the type argument and assigns one molecule ID', async () => {
    const { error, files } = await run(`units lj
atom_style bond
boundary p p f
region box block 0 10 0 10 0 10
create_box 4 box bond/types 1 extra/bond/per/atom 1
molecule dimer dimer
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
mass * 1.0
bond_style harmonic
bond_coeff 1 5.0 1.0
region slab block 0 10 0 10 8 9
fix 1 all deposit 2 1 1 12345 region slab mol dimer units box
run 2
write_dump all custom out.dump id mol type x y z modify format float %.15g sort id
`);
    expect(error).toBeNull();
    const atoms = dumpAtoms(files.get('out.dump') ?? '');
    expect(atoms.map((a) => [a.id, a.mol, a.type])).toEqual([[1, 1, 3], [2, 1, 4], [3, 2, 3], [4, 2, 4]]);
  });

  it('rejects a type offset that pushes a template type past ntypes', async () => {
    const { error } = await run(`${base}fix 1 all deposit 1 1 1 12345 region slab mol dimer units box
run 1
`);
    expect(error).toMatch(/type offset 1 plus template type 3 is larger than ntypes 3/);
  });

  it('rejects global/local with mol and a multi-molecule template', async () => {
    const multi = `# two molecules
4 atoms
2 bonds

Coords

1 0 0 0
2 1 0 0
3 0 0 0
4 0 1 0

Types

1 1
2 1
3 2
4 2

Bonds

1 1 1 2
2 1 3 4

Special Bond Counts

1 1 0 0
2 1 0 0
3 1 0 0
4 1 0 0

Special Bonds

1 2
2 1
3 4
4 3

Molecules

1 1
2 1
3 2
4 2
`;
    const glob = await run(`${base}fix 1 all deposit 1 0 1 12345 region slab mol dimer global 1.0 2.0 units box
run 1
`);
    expect(glob.error).toMatch(/global and local with mol are not supported/);
    const many = await run(`${base}molecule multi multi
fix 1 all deposit 1 0 1 12345 region slab mol multi units box
run 1
`, { dimer: DIMER, multi });
    expect(many.error).toMatch(/only a single-molecule template is supported/);
  });

  it('rigid/small mol takes the molecules added by deposit rigid', async () => {
    const { error, files } = await run(`units lj
atom_style bond
boundary p p f
region box block 0 10 0 10 0 20
create_box 3 box bond/types 1 extra/bond/per/atom 1
molecule dimer dimer
pair_style lj/cut 2.5
pair_coeff * * 0.0 1.0
mass * 1.0
bond_style harmonic
bond_coeff 1 20.0 0.5
region slab block 0 10 0 10 16 18
group addatoms empty
fix 1 addatoms rigid/small molecule mol dimer
fix 2 addatoms deposit 2 0 1 12345 region slab mol dimer rigid 1 vz -1.0 -1.0 units box
run 3
write_dump all custom out.dump id mol type x y z modify format float %.15g sort id
`);
    expect(error).toBeNull();
    const atoms = dumpAtoms(files.get('out.dump') ?? '');
    expect(atoms.map((a) => a.mol)).toEqual([1, 1, 2, 2]);
  });

  it('shake mol accepts the template and constrains the deposited molecule', async () => {
    const { error, files, logs } = await run(`units lj
atom_style bond
boundary p p f
region box block 0 10 0 10 0 20
create_box 3 box bond/types 1 extra/bond/per/atom 1
molecule dimer dimer
pair_style lj/cut 2.5
pair_coeff * * 0.0 1.0
mass * 1.0
bond_style harmonic
bond_coeff 1 20.0 1.0
region slab block 0 10 0 10 16 18
fix 1 all shake 0.0001 20 1000 b 1 mol dimer
fix 2 all deposit 2 0 1 12345 region slab mol dimer shake 1 vz -1.0 -1.0 units box
fix 3 all nve
run 3
write_dump all custom out.dump id mol type x y z modify format float %.15g sort id
`);
    expect(error).toBeNull();
    // one cluster at run setup (no atoms), one more per deposited molecule
    expect(logs.some((l) => /shake: 1 clusters/.test(l))).toBe(true);
    expect(logs.some((l) => /shake: 2 clusters/.test(l))).toBe(true);
    const atoms = dumpAtoms(files.get('out.dump') ?? '');
    expect(atoms.map((a) => a.mol)).toEqual([1, 1, 2, 2]);
  });
});
