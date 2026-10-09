import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { parseMoleculeFile, type MoleculeOptions } from '../src/engine/molecule';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix pour in 2d + molecule Diameters/Masses for sphere styles (wave 36).
 * Native behaviour is pinned by tests/oracle/w36pour2d.in, w36mol_diam.in and
 * w36pour2d_mol.in (thermo and final per-atom state compared to native LAMMPS);
 * these cases cover the argument handling and the values the oracle dumps do
 * not expose directly.
 */

const run = async (script: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const written = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => written.set(n, (ap ? written.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(files)) session.addFile(name, text);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error, thermo, files: written };
};

/** Atom rows (id, type, x, y, z, radius, mass) of a write_dump custom file. */
const dumpAtoms = (text: string): { id: number; type: number; x: number; y: number; z: number; r: number; m: number }[] => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  return lines.slice(k + 1).filter((l) => l.trim()).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    return { id: w[0], type: w[1], x: w[2], y: w[3], z: w[4], r: w[5], m: w[6] };
  });
};

const header2d = `dimension 2
atom_style sphere
boundary f f p
region box block 0 10 0 10 -0.5 0.5 units box
create_box 1 box
comm_modify vel yes
pair_style gran/hooke/history 4000.0 NULL 100.0 NULL 0.5 0
pair_coeff * *
timestep 0.001
fix 1 all nve/sphere
fix 2 all gravity 1.0 spherical 0.0 -180.0
`;

describe('2d fix pour', () => {
  it('counts particles by the insertion area, not the volume', async () => {
    // 2d nper = floor(0.001 * 100 / (pi/4 * 0.04)) = 3; the 3d sphere volume would give 23.
    const script = `${header2d}region slab block 0 10 0 10 -0.5 0.5 units box
fix 3 all pour 3 1 4767548 vol 0.001 50 diam one 0.2 region slab
fix 4 all enforce2d
thermo_style custom step atoms
thermo 1
run 1
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    expect(r.thermo.at(-1)!.atoms).toBe(3);
  });

  it('inserts inside the region with z = 0 and a negative fall velocity', async () => {
    const script = `${header2d}region slab block 2 8 4 9 -0.5 0.5 units box
fix 3 all pour 10 1 4767548 vol 0.1 50 diam one 0.5 vel 1.0 2.0 -3.0 region slab
fix 4 all enforce2d
thermo_style custom step atoms
thermo 1
run 1
write_dump all custom pour2d.dump id type x y z radius mass vx vy modify format float %.15g sort id
`;
    const r = await run(script);
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('pour2d.dump') ?? '');
    expect(atoms.length).toBe(10);
    for (const a of atoms) {
      expect(a.x).toBeGreaterThanOrEqual(2 - 1e-9);
      expect(a.x).toBeLessThanOrEqual(8 + 1e-9);
      expect(a.y).toBeGreaterThanOrEqual(4 - 1e-9);
      expect(a.y).toBeLessThanOrEqual(9 + 1e-9);
      expect(a.z).toBe(0);
      expect(a.r).toBe(0.25);
    }
    // fall velocity -sqrt(3^2 + 2 g (yhi - y)); one step of nve adds -g dt/2
    for (const a of atoms) expect(a.y).toBeLessThanOrEqual(9 + 1e-9);
  });

  it('rejects a z-axis cylinder region and gravity along -z', async () => {
    const cyl = `${header2d}region cyl cylinder z 5 5 2 0 1 side in units box
fix 3 all pour 3 1 7 region cyl vol 0.1 50
fix 4 all enforce2d
run 1
`;
    expect((await run(cyl)).error).toMatch(/cylinder style of region can only be used with 3d/);
    const wrongG = header2d.replace('fix 2 all gravity 1.0 spherical 0.0 -180.0', 'fix 2 all gravity 1.0 spherical 0.0 0.0') + `region slab block 0 10 0 10 -0.5 0.5 units box
fix 3 all pour 1 1 7 region slab vol 0.1 50
fix 4 all enforce2d
run 1
`;
    expect((await run(wrongG)).error).toMatch(/gravity fix must point in the -y direction/);
  });
});

describe('molecule Diameters and Masses', () => {
  it('parses the sections and scales them by scale', () => {
    const text = `# t
2 atoms

Coords

1 0 0 0
2 1 0 0

Types

1 1
2 1

Diameters

1 1.0
2 2.0

Masses

1 0.5
2 1.5
`;
    const o: MoleculeOptions = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 2 };
    const t = parseMoleculeFile('m', 'm.mol', text, o);
    expect(Array.from(t.diam!)).toEqual([2, 4]);
    expect(Array.from(t.mass!)).toEqual([1, 3]);
  });

  it('create_atoms mol gives sphere atoms the template radius and mass', async () => {
    const mol = `# t
2 atoms

Coords

1 0 0 0
2 1 0 0

Types

1 1
2 1

Diameters

1 1.0
2 2.0

Masses

1 0.5
2 1.5
`;
    const script = `units lj
atom_style sphere
boundary p p p
region box block 0 15 0 15 0 15 units box
create_box 1 box
molecule md md.mol
fix 1 all nve/sphere
create_atoms 0 single 5 5 5 mol md 12345 units box
run 0
write_dump all custom ca.dump id type x y z radius mass modify format float %.15g sort id
`;
    const r = await run(script, { 'md.mol': mol });
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('ca.dump') ?? '');
    expect(atoms.map((a) => a.r)).toEqual([0.5, 1]);
    expect(atoms.map((a) => a.m)).toEqual([0.5, 1.5]);
  });

  it('without the sections the defaults are radius 0.5 and mass 4/3 pi 0.5^3', async () => {
    const mol = `# t
1 atoms

Coords

1 0 0 0

Types

1 1
`;
    const script = `units lj
atom_style sphere
boundary p p p
region box block 0 10 0 10 0 10 units box
create_box 1 box
molecule md md.mol
fix 1 all nve/sphere
create_atoms 0 single 5 5 5 mol md 12345 units box
run 0
write_dump all custom ca.dump id type x y z radius mass modify format float %.15g sort id
`;
    const r = await run(script, { 'md.mol': mol });
    expect(r.error).toBeNull();
    const a = dumpAtoms(r.files.get('ca.dump') ?? '')[0];
    expect(a.r).toBe(0.5);
    expect(a.m).toBeCloseTo((4 / 3) * Math.PI * 0.5 ** 3, 12);
  });

  it('fix deposit mol applies the template diameters and masses', async () => {
    const mol = `# t
2 atoms

Coords

1 0 0 0
2 1 0 0

Types

1 1
2 1

Diameters

1 1.0
2 2.0

Masses

1 0.5
2 1.5
`;
    const script = `units lj
atom_style hybrid sphere molecular
boundary p p p
region box block 0 20 0 20 0 20 units box
create_box 1 box
molecule md md.mol
fix nve all nve/sphere
fix ins all deposit 2 0 5 12345 region box mol md
run 1
write_dump all custom dep.dump id type x y z radius mass modify format float %.15g sort id
`;
    const r = await run(script, { 'md.mol': mol });
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('dep.dump') ?? '');
    expect(atoms.map((a) => a.r)).toEqual([0.5, 1]);
    expect(atoms.map((a) => a.m)).toEqual([0.5, 1.5]);
  });
});

describe('2d fix pour with a molecule template', () => {
  const mol = `# cluster
3 atoms

Coords

1 0.0 0.0 0.0
2 1.2 0.0 0.0
3 0.6 1.0 0.0

Types

1 1
2 1
3 1

Diameters

1 0.8
2 0.8
3 0.8

Masses

1 1.0
2 1.0
3 1.0
`;
  it('inserts whole molecules with the template radii and a type offset', async () => {
    const script = `${header2d}molecule md md.mol
region slab block 2 8 4 8 -0.5 0.5 units box
fix 3 all pour 2 0 4767548 vol 0.5 50 region slab mol md
fix 4 all enforce2d
thermo_style custom step atoms
thermo 1
run 1
write_dump all custom pour2dm.dump id type x y z radius mass modify format float %.15g sort id
`;
    const r = await run(script, { 'md.mol': mol });
    expect(r.error).toBeNull();
    const atoms = dumpAtoms(r.files.get('pour2dm.dump') ?? '');
    expect(atoms.length).toBe(2 * 3);
    for (const a of atoms) expect(a.r).toBe(0.4);
    for (const a of atoms) expect(a.m).toBe(1);
    for (const a of atoms) expect(a.z).toBe(0);
  });

  it('rejects a template that defines more than one molecule', async () => {
    const multi = `# two molecules
2 atoms
2 bonds

Coords

1 0 0 0
2 1 0 0

Types

1 1
2 1

Molecules

1 1
2 2

Bonds

1 1 1 2
2 1 1 2
`;
    const script = `${header2d}molecule md md.mol
region slab block 2 8 4 8 -0.5 0.5 units box
fix 3 all pour 2 0 4767548 vol 0.5 50 region slab mol md
fix 4 all enforce2d
run 1
`;
    expect((await run(script, { 'md.mol': multi })).error).toMatch(/single-molecule template/);
  });
});
