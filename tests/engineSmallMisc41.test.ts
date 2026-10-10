import { describe, expect, it } from 'vitest';
import { parseMoleculeFile, templateDipoles, rotateDipoles, rotationMatrix } from '../src/engine/molecule';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Parsing and refusals for three small features (see the oracle cases w41sm_*):
 *  - molecule Dipoles section (molecule.rst: "This section is only allowed for atom styles that support
 *    particles with point dipoles, e.g. atom_style dipole.");
 *  - fix shake keyword store (fix_shake.rst: "The *store* keyword controls whether the fix stores the
 *    constraint (or restraint) forces as a per-atom property.");
 *  - fix wall/gran lo/hi given as v_name (fix_wall_gran.rst, versionadded 2Sep2026: "The *lo* and *hi* wall
 *    positions can also be set by an equal-style variable").
 */

const O = { toff: 0, boff: 0, aoff: 0, doff: 0, ioff: 0, scale: 1 };

const DIPOLE_MOL = `# three dipoles
3 atoms

Coords

1 0 0 0
2 1.1 0 0
3 0 1.1 0

Types

1 1
2 1
3 1

Dipoles

1 1.0 0.0 0.0
2 0.0 0.5 0.0
3 0.0 0.0 0.0
`;

const runScript = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row as unknown as Record<string, number>);
  return { rows, error };
};

describe('molecule Dipoles section', () => {
  it('reads one dipole per atom and scales it with the scale keyword', () => {
    const t = parseMoleculeFile('d', 'd.txt', DIPOLE_MOL, { ...O, scale: 2 });
    expect(t.mu).not.toBeNull();
    expect(Array.from(t.mu!)).toEqual([2, 0, 0, 0, 1, 0, 0, 0, 0]);
  });

  it('has no dipoles without the section, and the storage is length-augmented (mux, muy, muz, |mu|)', () => {
    const plain = parseMoleculeFile('p', 'p.txt', DIPOLE_MOL.replace(/\nDipoles[\s\S]*$/, '\n'), O);
    expect(plain.mu).toBeNull();
    expect(templateDipoles(plain, 1)).toBeUndefined();
    const t = parseMoleculeFile('d', 'd.txt', DIPOLE_MOL, O);
    const out = templateDipoles(t, 2)!;
    expect(out.length).toBe(4 * 3 * 2);
    expect(out[3]).toBeCloseTo(1, 15);
    expect(out[7]).toBeCloseTo(0.5, 15);
    expect(out[11]).toBeCloseTo(0, 15);
  });

  it('turns the dipoles with a rotation (R mu), as the coordinates are turned', () => {
    const t = parseMoleculeFile('d', 'd.txt', DIPOLE_MOL, O);
    const R = rotationMatrix(Math.PI / 2, 0, 0, 1);
    const r = rotateDipoles(t, R)!;
    // (1,0,0) about z by 90 degrees is (0,1,0); (0,0.5,0) becomes (-0.5,0,0)
    expect(r[0]).toBeCloseTo(0, 12);
    expect(r[1]).toBeCloseTo(1, 12);
    expect(r[3]).toBeCloseTo(-0.5, 12);
    expect(r[4]).toBeCloseTo(0, 12);
  });

  it('rejects an atom ID outside the template and a short section', () => {
    expect(() => parseMoleculeFile('d', 'd.txt', DIPOLE_MOL.replace('3 0.0 0.0 0.0', '7 0 0 0'), O)).toThrow(/outside 1\.\.3/);
    expect(() => parseMoleculeFile('d', 'd.txt', DIPOLE_MOL.replace('2 0.0 0.5 0.0\n', ''), O)).toThrow(/has 2 of 3 lines/);
  });

  it('create_atoms mol refuses dipoles for an atom style without point dipoles', async () => {
    const { error } = await runScript(`units lj
atom_style sphere
region box block 0 10 0 10 0 10 units box
create_box 1 box
molecule dm dipoles.txt
create_atoms 0 single 2 2 2 mol dm 1 units box`, { 'dipoles.txt': DIPOLE_MOL });
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/dipoles/);
  });
});

describe('fix shake keyword store', () => {
  const BASE = `units real
atom_style full
region box block 0 10 0 10 0 10 units box
create_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 2
mass 1 12.0
mass 2 1.0
create_atoms 1 single 5 5 5 units box
create_atoms 2 single 6.1 5 5 units box
bond_style harmonic
bond_coeff 1 100.0 1.0
pair_style zero 3.0
pair_coeff * *
create_bonds single/bond 1 1 2
fix 1 all nve
`;

  it('store yes gives a per-atom array of three columns; reduce over it works', async () => {
    const { error, rows } = await runScript(`${BASE}fix 2 all shake 1.0e-10 100 0 b 1 store yes
compute fr all reduce sumsq f_2[1] f_2[2] f_2[3]
thermo_style custom step c_fr[1] c_fr[2] c_fr[3]
run 0`);
    expect(error).toBeNull();
    // the stretched bond (1.1 vs 1.0) gives a non-zero setup correction on both atoms
    expect(rows[0]['c_fr[1]'] + rows[0]['c_fr[2]'] + rows[0]['c_fr[3]']).toBeGreaterThan(0);
  });

  it('store no makes the per-atom reference an error', async () => {
    const { error } = await runScript(`${BASE}fix 2 all shake 1.0e-10 100 0 b 1
compute fr all reduce sumsq f_2[1] f_2[2] f_2[3]
thermo_style custom step c_fr[1]
run 0`);
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/does not calculate per-atom values/);
  });

  it('a store value other than yes or no is refused', async () => {
    const { error } = await runScript(`${BASE}fix 2 all shake 1.0e-10 100 0 b 1 store maybe
run 0`);
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/store must be yes or no/);
  });
});

describe('fix wall/gran lo/hi as v_name', () => {
  const BASE = `units lj
boundary f f f
atom_style sphere
atom_modify map array
comm_modify vel yes
region box block -5 5 -5 5 0 4 units box
create_box 1 box
create_atoms 1 single 0 0 1.7
set atom 1 diameter 1.0 density 1.0
set atom 1 vz 0.0
pair_style none
timestep 0.001
fix 1 all nve
`;

  it('a moving hi wall gives the same thermo as the native probe (ke after one step, v = -0.78057477705)', async () => {
    const { error, rows } = await runScript(`${BASE}variable zhi equal 2.0-0.001*step
fix 2 all wall/gran hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 v_zhi
compute r all reduce sum vz z
thermo_style custom step c_r[1] c_r[2]
thermo 1
run 2`);
    expect(error).toBeNull();
    // the sphere falls from z = 1.7 under the moving wall at 2.0 - 0.001 step (velocity -1): the wall force
    // at step 1 is -417.416 and vz = -0.78057477705 after it (native dump, step 1; step 2 vz = -1.5575082544)
    expect(rows[1]['c_r[1]']).toBeCloseTo(-0.78057477705, 9);
    expect(rows[2]['c_r[1]']).toBeCloseTo(-1.5575082544, 9);
  });

  it('a variable wall cannot be combined with wiggle or shear', async () => {
    const { error } = await runScript(`${BASE}variable zhi equal 2.0
fix 2 all wall/gran hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 v_zhi wiggle z 0.1 2.0
run 0`);
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/cannot be combined with wiggle or shear/);
  });

  it('lo must stay below hi during the run', async () => {
    const { error } = await runScript(`${BASE}variable zlo equal 3.0
fix 2 all wall/gran hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane v_zlo 2.0
run 1`);
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/lo wall position must remain below/);
  });

  it('an unknown variable is refused', async () => {
    const { error } = await runScript(`${BASE}fix 2 all wall/gran hooke 2000.0 NULL 50.0 NULL 0.5 0 zplane 0.0 v_nope
run 0`);
    expect(error).not.toBeNull();
  });
});

describe('rigid bodies with point dipoles', () => {
  it('fix rigid/small refuses a body whose atoms carry dipoles', async () => {
    const { error } = await runScript(`units lj
atom_style hybrid sphere dipole
region box block 0 10 0 10 0 10 units box
create_box 1 box
molecule dm dipoles.txt
create_atoms 0 single 2 2 2 mol dm 1 units box
fix 1 all rigid/small molecule
run 0`, { 'dipoles.txt': DIPOLE_MOL });
    expect(error).not.toBeNull();
    expect(error!.message).toMatch(/point dipoles/);
  });
});
