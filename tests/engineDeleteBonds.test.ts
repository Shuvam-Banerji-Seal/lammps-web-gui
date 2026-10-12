import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// 4 atoms in a chain (atom 2 of type 2), bonds 1-2 (type 1), 2-3 (type 2), 3-4 (type 1), angles 1-2-3 and 2-3-4
const DATA = `LAMMPS data

4 atoms
3 bonds
2 angles
2 atom types
2 bond types
1 angle types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1
2 1

Atoms # molecular

1 1 1 1 1 1
2 1 2 2 1 1
3 1 1 3 1 1
4 1 1 4 1 1

Bonds

1 1 1 2
2 2 2 3
3 1 3 4

Angles

1 1 1 2 3
2 1 2 3 4
`;

const runWith = async (cmds: string): Promise<number[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  session.addFile('data.mol', DATA);
  await session.execute(`units lj
atom_style molecular
read_data data.mol
bond_style harmonic
bond_coeff 1 1 0.5
bond_coeff 2 10 0.5
angle_style harmonic
angle_coeff * 1 90
pair_style zero 2
pair_coeff * *
group g id 1 2 3
${cmds}
thermo_style custom step ebond eangle bonds angles
run 0
print "VALUES $(ebond:%.12g) $(eangle:%.12g) $(bonds)"
`);
  const line = events.flatMap((e) => (e.kind === 'log' ? [e.text] : [])).join('\n').split('\n').find((l) => l.startsWith('VALUES'));
  return line!.split(/\s+/).slice(1).map(Number);
};

describe('delete_bonds', () => {
  // Measured with native LAMMPS (black box), same system: ebond eangle bonds angles after each command
  const cases: [string, number, number, number][] = [
    ['', 0.75, 1.2337006, 3],
    ['delete_bonds g bond 1', 0.6875, 1.2337006, 3],
    ['delete_bonds g bond 1 any', 0.625, 1.2337006, 3],
    ['delete_bonds g bond 2 any', 0.125, 1.2337006, 3],
    ['delete_bonds g atom 2', 0.0625, 0.61685028, 3],
    ['delete_bonds g atom 1', 0.0625, 0.61685028, 3],
    ['delete_bonds g multi', 0.0625, 0.61685028, 3],
    ['delete_bonds g multi any', 0, 0, 3],
    ['delete_bonds g bond 1\ndelete_bonds g bond 1 undo', 0.75, 1.2337006, 3],
    ['delete_bonds g bond 1*2', 0.0625, 1.2337006, 3],
    ['delete_bonds g bond 1 undo remove', 0.75, 1.2337006, 3],
    ['delete_bonds g bond 1 remove special', 0.6875, 1.2337006, 2],
    ['delete_bonds g stats', 0.75, 1.2337006, 3],
    ['delete_bonds g angle 1 any', 0.75, 0, 3],
  ];
  for (const [cmds, ebond, eangle, nbonds] of cases) {
    it(`matches native: ${cmds.replace(/\n/g, ' | ') || '(none)'}`, async () => {
      const [eb, ea, nb] = await runWith(cmds);
      expect(eb).toBeCloseTo(ebond, 7);
      expect(ea).toBeCloseTo(eangle, 7);
      expect(nb).toBe(nbonds);
    });
  }
  it('stops on a bare * range, as native', async () => {
    await expect(runWith('delete_bonds g bond *')).rejects.toThrow('Invalid range string: *');
  });
  it('refuses remove of a turned-off bond without special (native would keep the old special lists)', async () => {
    await expect(runWith('delete_bonds g bond 1\ndelete_bonds g angle 1 remove')).rejects.toThrow('add the special keyword');
  });
  it('refuses an unknown keyword', async () => {
    await expect(runWith('delete_bonds g bond 1 sometimes')).rejects.toThrow("unknown keyword 'sometimes'");
  });
});
