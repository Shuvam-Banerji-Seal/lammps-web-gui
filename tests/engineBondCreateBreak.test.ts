import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix bond/create and fix bond/break (src/MC, GPL port in src/engine-gpl/mc): small checks of the topology edits
 * and the error messages. Parity with native LAMMPS is in tests/oracle/w139bc_*.in; these tests use four atoms with
 * no forces (zero coefficients), so the expected counts follow from the geometry alone.
 * docs.lammps.org/fix_bond_break.html and docs.lammps.org/fix_bond_create.html.
 */

const runScript = async (text: string, data: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  session.addFile('chain.data', data);
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, message: error instanceof Error ? error.message : String(error ?? '') };
};

const header = `units lj
boundary f f f
atom_style molecular
special_bonds lj/coul 0.0 1.0 1.0
`;

// the coefficients follow read_data (a style command needs the box in the engine)
const coeffs = `pair_style zero 2.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 0.0 1.0
angle_style harmonic
angle_coeff 1 0.0 90.0
dihedral_style harmonic
dihedral_coeff 1 0.0 1 1
`;

// bond 2-3 is stretched to 1.6 (fix bond/break cuts at 1.5): its angles 1-2-3 and 2-3-4 and the dihedral 1-2-3-4 go with it
const BREAK_DATA = `chain
4 atoms
3 bonds
2 angles
1 dihedrals
1 atom types
1 bond types
1 angle types
1 dihedral types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 2.0 2.0 2.0
2 1 1 3.0 2.0 2.0
3 1 1 4.6 2.0 2.0
4 1 1 5.6 2.0 2.0

Bonds

1 1 1 2
2 1 2 3
3 1 3 4

Angles

1 1 1 2 3
2 1 2 3 4

Dihedrals

1 1 1 2 3 4
`;

// atoms 1 and 4 are 1.118 apart (within 1.2) and no other pair is: the new bond 1-4 induces the angles 2-1-4 and 1-4-3
// (native LAMMPS: 4 bonds and 2 angles after the step, checked with the native binary)
const CREATE_DATA = BREAK_DATA
  .replace('2 angles', '0 angles')
  .replace('1 dihedrals', '0 dihedrals')
  .replace(/Atoms # molecular[\s\S]*?\n\nBonds/, `Atoms # molecular

1 1 1 2.0 2.0 2.0
2 1 1 3.0 2.0 2.0
3 1 1 3.0 3.0 2.0
4 1 1 1.0 2.5 2.0

Bonds`)
  .replace(/\n\nAngles[\s\S]*$/, '\n');

describe('fix bond/break', () => {
  it('breaks a stretched bond and removes the angles and dihedrals that contain it', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}velocity all set 0 0 0
fix 1 all nve
fix b all bond/break 1 1 1.5
thermo_style custom step bonds angles dihedrals f_b[1] f_b[2]
thermo_modify format float %.15g
run 1`, BREAK_DATA);
    expect(r.message).toBe('');
    const last = r.thermo[r.thermo.length - 1];
    expect(last.bonds).toBe(2);
    expect(last.angles).toBe(0);
    expect(last.dihedrals).toBe(0);
    expect(last['f_b[1]']).toBe(1);
    expect(last['f_b[2]']).toBe(1);
  });

  it('leaves the topology alone when no bond is past the cutoff', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}velocity all set 0 0 0
fix 1 all nve
fix b all bond/break 1 1 2.0
thermo_style custom step bonds angles dihedrals f_b[1] f_b[2]
run 1`, BREAK_DATA);
    expect(r.message).toBe('');
    const last = r.thermo[r.thermo.length - 1];
    expect([last.bonds, last.angles, last.dihedrals, last['f_b[1]']]).toEqual([3, 2, 1, 0]);
  });

  it('refuses keywords it does not support, naming them', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}fix b all bond/break 1 1 1.5 vizsteps 5
run 0`, BREAK_DATA);
    expect(r.message).toMatch(/vizsteps/);
  });
});

describe('fix bond/create', () => {
  it('creates a bond between atoms within the cutoff and induces the angle through it', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}velocity all set 0 0 0
fix 1 all nve
fix c all bond/create 1 1 1 1.2 1 atype 1
thermo_style custom step bonds angles dihedrals f_c[1] f_c[2]
thermo_modify format float %.15g
run 1`, CREATE_DATA);
    expect(r.message).toBe('');
    const last = r.thermo[r.thermo.length - 1];
    expect([last.bonds, last.angles, last.dihedrals, last['f_c[1]'], last['f_c[2]']]).toEqual([4, 2, 0, 1, 1]);
  });

  it('refuses aconstrain, which belongs to fix bond/create/angle', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}fix c all bond/create 1 1 1 1.2 1 aconstrain 90 120
run 0`, CREATE_DATA);
    expect(r.message).toMatch(/aconstrain/);
  });

  it('rejects an iparam/jparam pair that disagrees for one atom type', async () => {
    const r = await runScript(`${header}read_data chain.data
${coeffs}fix c all bond/create 1 1 1 1.2 1 iparam 1 1 jparam 2 1
run 0`, CREATE_DATA);
    expect(r.message).toMatch(/Inconsistent iparam\/jparam/);
  });
});

describe('fix bond/create and bond/break argument checks', () => {
  it('needs a probability in [0, 1] and a positive seed', async () => {
    const bad = await runScript(`${header}read_data chain.data
${coeffs}fix c all bond/create 1 1 1 1.2 1 prob 1.5 1234
run 0`, CREATE_DATA);
    expect(bad.message).toMatch(/prob fraction/);
    const brk = await runScript(`${header}read_data chain.data
${coeffs}fix b all bond/break 1 1 1.5 prob 0.5 0
run 0`, BREAK_DATA);
    expect(brk.message).toMatch(/seed/);
  });
});
