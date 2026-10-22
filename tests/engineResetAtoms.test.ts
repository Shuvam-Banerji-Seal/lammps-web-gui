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
    expect(await runScript('reset_atoms foo all')).toMatch(/property must be id, image or mol/);
    expect(await runScript('reset_atoms mol all offset -2')).toMatch(/offset must be >= -1/);
  });

  it('refuses bonds that are turned off (type 0 or below) for image and mol', async () => {
    expect(await runScript('delete_bonds all bond 1\nreset_atoms image all')).toMatch(/bonds of type -1 \(turned off or broken\) are not supported/);
    expect(await runScript('delete_bonds all bond 1\nreset_atoms mol all')).toMatch(/bonds of type -1/);
  });
});
