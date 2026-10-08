import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * special_bonds and periodic images (src/engine/neighbor.ts encode): a bonded partner seen through
 * an image more than half a box edge away, in any periodic dimension, is an ordinary neighbor.
 * Expected energies were measured with native LAMMPS (black box), thermo pe in lj units (per atom).
 */

const data = (tilt: string, x2: string): string => `dimer

2 atoms
1 bonds
1 atom types
1 bond types

0 5 xlo xhi
0 5 ylo yhi
0 5 zlo zhi
${tilt}
Masses

1 1.0

Atoms # molecular

1 1 1 ${tilt ? '1.0 1.0' : '1.0 2.5'} 2.5
2 1 1 ${x2} 2.5

Bonds

1 1 1 2
`;

const pe = async (file: string, sb: string): Promise<number> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  s.addFile('d.data', file);
  await s.execute(`units lj
atom_style molecular
boundary p p p
read_data d.data
pair_style lj/cut 4.0
pair_coeff 1 1 1.0 1.0
bond_style zero
bond_coeff 1
special_bonds lj ${sb}
thermo_style custom step pe
run 0
`);
  const row = events.find((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')!;
  return row.row.pe;
};

describe('special bonds through periodic images', () => {
  const ortho = data('', '2.2 2.5');
  const tri = data('2.0 0 0 xy xz yz\n', '3.7 3.4');
  const cases: [string, string, string, number][] = [
    ['orthogonal, lj 0', ortho, '0.0 0.0 0.0', -0.000664024575689433],
    ['orthogonal, lj 0.5', ortho, '0.5 0.0 0.0', -0.223405346471458],
    ['orthogonal, lj 1', ortho, '1.0 1.0 1.0', -0.446146668367227],
    ['triclinic, lj 0', tri, '0.0 0.0 0.0', -0.00613399930820693],
    ['triclinic, lj 1', tri, '1.0 1.0 1.0', -0.0076152247387935],
  ];
  for (const [name, file, sb, want] of cases) {
    it(name, async () => {
      const got = await pe(file, sb);
      expect(Math.abs(got - want)).toBeLessThan(1e-13);
    });
  }
});
