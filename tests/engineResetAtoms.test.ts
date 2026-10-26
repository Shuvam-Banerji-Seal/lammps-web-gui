import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * reset_atoms (engine/commands/reset_atoms.ts; docs.lammps.org/reset_atoms.html). The results are pinned by the
 * oracle cases tests/oracle/w104cf_reset_*.in and w143cf_reset_sort.in. These tests cover the argument checks, which the docs describe as "reset_atoms id sort yes" and "offset value = Noffset >= -1".
 */

const DATA = `refusals

4 atoms
2 bonds
1 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # bond

1 1 1 1.0 1.0 1.0
2 1 1 2.0 1.0 1.0
3 2 1 3.0 1.0 1.0
4 2 1 4.0 1.0 1.0

Bonds

1 1 1 2
2 1 3 4
`;

const runScript = async (script: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('refusals.data', DATA);
  try {
    await session.execute(`units lj\natom_style bond\nread_data refusals.data\n${script}\n`);
  } catch {
    // reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('reset_atoms: refusals', () => {
  it('sort yes runs the native spatial order (the oracle case w143cf_reset_sort pins the IDs)', async () => {
    expect(await runScript('reset_atoms id sort yes')).toBe('');
  });

  it('accepts sort no and refuses other keywords by name', async () => {
    expect(await runScript('reset_atoms id sort no')).toBe('');
    expect(await runScript('reset_atoms id frob 1')).toContain('Unknown reset_atoms id keyword: frob');
  });

  it('refuses an unknown property and a negative offset below -1', async () => {
    expect(await runScript('reset_atoms foo all')).toMatch(/Unknown command: reset_atoms foo all/);
    expect(await runScript('reset_atoms mol all offset -2')).toMatch(/Illegal reset_atoms mol offset: -2/);
  });

  // As native LAMMPS checks them (src/reset_atoms_id.cpp, reset_atoms_image.cpp and reset_atoms_mol.cpp at b332f28):
  // the box, the fixes that store atom IDs (id), bonds allowed (image) or a molecular system (mol), then the words in
  // order and the group. Measured with native LAMMPS: each line below gives the message expected.
  it('checks its arguments in native order with native messages', async () => {
    const cases: [string, RegExp][] = [
      ['reset_atoms id sort', /Illegal reset_atoms id command: missing argument\(s\)/],
      ['reset_atoms id sort maybe', /Expected boolean parameter instead of 'maybe'/],
      ['reset_atoms mol', /Illegal reset_atoms mol command: missing argument\(s\)/],
      ['reset_atoms mol nogroup', /Could not find reset_atoms mol group nogroup/],
      ['reset_atoms mol all bogus 1', /Unknown reset_atoms mol keyword: bogus/],
      ['reset_atoms mol all compress', /Illegal reset_atoms mol compress command: missing argument\(s\)/],
      ['reset_atoms image', /Illegal reset_atoms image command: missing argument\(s\)/],
      ['reset_atoms image all extra', /Unknown reset_atoms image keyword: extra/],
      ['reset_atoms image nogroup', /Could not find reset_atoms image group nogroup/],
      ['bond_style harmonic\nbond_coeff 1 100 1.0\nfix 1 all shake 1e-4 20 0 b 1\nreset_atoms id', /Cannot use reset_atoms id with a fix shake storing atom IDs/],
    ];
    for (const [line, want] of cases) expect(await runScript(line), line).toMatch(want);
    const atomic = async (line: string): Promise<string> => {
      const session = new Session({ emit: () => {}, writeFile: () => {} });
      try {
        await session.execute(`units lj\natom_style atomic\nregion b block 0 4 0 4 0 4\ncreate_box 1 b\n${line}\n`);
      } catch (e) { return String((e as Error).message); }
      return '';
    };
    expect(await atomic('reset_mol_ids')).toMatch(/Can only use reset_atoms mol on molecular systems/);
    expect(await atomic('reset_atoms image all')).toMatch(/Cannot use reset_atoms image used when bonds are not allowed/);
    const before = new Session({ emit: () => {}, writeFile: () => {} });
    await expect(before.execute('units lj\nreset_atoms id\n')).rejects.toThrow(/Reset_atoms id command before simulation box is defined/);
  });

  it('refuses bonds that are turned off (type 0 or below) for image and mol', async () => {
    expect(await runScript('delete_bonds all bond 1\nreset_atoms image all')).toMatch(/bonds of type -1 \(turned off or broken\) are not supported/);
    expect(await runScript('delete_bonds all bond 1\nreset_atoms mol all')).toMatch(/bonds of type -1/);
  });
});
