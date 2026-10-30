import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * The bonds an atom can store (native atom->bond_per_atom): read_data stores the largest per-atom count of the file
 * plus extra/bond/per/atom (src/read_data.cpp at b332f28, "maxall += atom->extra_bond_per_atom"), create_box the extra
 * alone, and create_bonds stops when an atom's storage is full. Messages measured with native LAMMPS.
 */

const CHAIN = readFileSync(join(__dirname, 'oracle', 'w220xp_chain.data'), 'utf8');

const run = async (script: string): Promise<Session> => {
  const s = new Session({ emit: () => undefined, writeFile: () => undefined } as never);
  s.addFile('chain.data', CHAIN);
  await s.execute(script);
  return s;
};

const HEAD = (extra: number) => `units lj\natom_style full\nread_data chain.data extra/bond/per/atom ${extra} extra/special/per/atom 40
mass * 1.0\nbond_style harmonic\nbond_coeff 1 50.0 1.2\npair_style lj/cut 1.5\npair_coeff * * 1.0 1.0\ngroup two type 2\n`;

describe('bonds per atom', () => {
  it('create_bonds many stops at read_data\'s storage: one bond per atom in the file plus 6 extra is a limit of 7', async () => {
    // Measured with native LAMMPS: ERROR on proc 0: New bond exceeded bonds per atom limit of 7 in create_bonds
    await expect(run(`${HEAD(6)}create_bonds many two two 1 1.6 1.8\n`)).rejects.toThrow('New bond exceeded bonds per atom limit of 7 in create_bonds');
    await expect(run(`${HEAD(20)}create_bonds many two two 1 1.6 1.8\n`)).resolves.toBeDefined();
  });

  it('create_bonds single/bond stores the bond at its first atom', async () => {
    // Measured with native LAMMPS (create_box extra/bond/per/atom 1, a bond from atom 1 to 2, then from 1 to 3): ERROR on
    // proc 0: New bond exceeded bonds per atom in create_bonds
    const box = 'units real\natom_style full\nregion b block 0 10 0 10 0 10\ncreate_box 2 b bond/types 1 extra/bond/per/atom 1 extra/special/per/atom 2\nmass * 1.0\n'
      + 'create_atoms 1 single 1 1 1\ncreate_atoms 2 single 2 2 2\ncreate_atoms 2 single 3 3 3\nbond_style harmonic\nbond_coeff 1 100.0 1.0\ncreate_bonds single/bond 1 1 2\n';
    await expect(run(`${box}create_bonds single/bond 1 1 3\n`)).rejects.toThrow('New bond exceeded bonds per atom in create_bonds');
    // the bond from atom 3 to atom 1 is stored at atom 3, which has room
    await expect(run(`${box}create_bonds single/bond 1 3 1\n`)).resolves.toBeDefined();
  });
});
