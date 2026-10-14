import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { UNIT_SYSTEMS } from '../src/engine/units';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix shake / rattle (src/engine/fix/shake.ts). Agreement with native LAMMPS:
 * oracle cases shake_clusters and rattle_clusters.
 */

const STAR = `# central atom with four bonds: too large a cluster
5 atoms
4 bonds

Coords

1 0 0 0
2 1 0 0
3 -1 0 0
4 0 1 0
5 0 -1 0

Types

1 1
2 1
3 1
4 1
5 1

Bonds

1 1 1 2
2 1 1 3
3 1 1 4
4 1 1 5
`;
const CHAIN4 = `# four-atom chain: two central atoms joined by a constrained bond
4 atoms
3 bonds

Coords

1 0 0 0
2 1 0 0
3 2 0.3 0
4 3 0.3 0.2

Types

1 1
2 1
3 1
4 1

Bonds

1 1 1 2
2 1 2 3
3 1 3 4
`;

const run = async (mol: string, extra = 'run 20') => {
  const rows: ThermoRow[] = [];
  const s = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); }, writeFile: () => {} });
  s.addFile('m.txt', mol);
  try {
    await s.execute(`units real
atom_style full
region box block 0 12 0 12 0 12 units box
create_box 1 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 8
mass 1 12.0
molecule m m.txt
create_atoms 0 single 6 6 6 mol m 1 rotate 10 1 1 1 units box
pair_style lj/cut 4.0
pair_coeff * * 0.0 1.0
bond_style harmonic
bond_coeff 1 300.0 1.1
variable vx atom 0.003*sin(x)
velocity all set v_vx 0.001 0.0 units box
fix 1 all nve
fix 2 all shake 1e-12 100 0 b 1
thermo_style custom step temp ebond
thermo 10
${extra}`);
    return { rows, error: null as EngineError | null };
  } catch (e) { return { rows, error: e as EngineError }; }
};

describe('fix shake', () => {
  it('rejects clusters the docs do not allow', async () => {
    expect((await run(STAR)).error?.message).toMatch(/Shake cluster of more than 4 atoms/);
    expect((await run(CHAIN4)).error?.message).toMatch(/Shake clusters are connected/);
  });

  it('allows only one shake or rattle fix', async () => {
    const { error } = await run(STAR.replace('5 atoms\n4 bonds', '2 atoms\n1 bonds').replace(/\n3 -1 0 0\n4 0 1 0\n5 0 -1 0/, '').replace(/\n3 1\n4 1\n5 1/, '').replace(/\n2 1 1 3\n3 1 1 4\n4 1 1 5/, ''), 'fix 3 all rattle 1e-8 10 0 b 1\nrun 0');
    expect(error?.message).toMatch(/only be one shake or rattle fix/);
  });

  describe('minimization restraints (fix_shake.cpp min_post_force)', () => {
    // a bond of length 1.2 with r0 = 1.0 (bond_coeff 1 100 1.0); the bond is constrained, so it has no bond energy
    const DIMER_DATA = `LAMMPS data

2 atoms
1 bonds
1 atom types
1 bond types

0 20 xlo xhi
0 20 ylo yhi
0 20 zlo zhi

Masses

1 12.0

Atoms # bond

1 1 1 5.0 5.0 5.0
2 1 1 6.2 5.0 5.0

Bonds

1 1 1 2
`;
    const pe = async (fix: string, body: string): Promise<{ pe: number[]; error: string | null }> => {
      const pes: number[] = [];
      const s = new Session({
        emit: (e: EngineEvent) => { if (e.kind === 'thermo') pes.push(e.row.pe); },
        writeFile: () => {},
      });
      s.addFile('dimer.data', DIMER_DATA);
      try {
        await s.execute(`units real
atom_style bond
read_data dimer.data
pair_style zero 3.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 100.0 1.0
${fix}
thermo_style custom step pe
${body}
`);
        return { pe: pes, error: null };
      } catch (e) {
        return { pe: pes, error: String(e) };
      }
    };

    it('uses kbond = 1e9 * boltz by default: step-0 pe is kbond (r - r0)^2 and the bond has no energy in dynamics', async () => {
      const k = 1e9 * UNIT_SYSTEMS.real.boltz;
      const min = await pe('fix 2 all shake 1e-4 20 0 b 1', 'minimize 0 0 1 10');
      expect(min.error).toBeNull();
      expect(Math.abs(min.pe[0] - 0.04 * k) / (0.04 * k)).toBeLessThan(1e-12);
      const md = await pe('fix 2 all shake 1e-4 20 0 b 1', 'run 0');
      expect(md.error).toBeNull();
      expect(Math.abs(md.pe[0])).toBeLessThan(1e-9);
    });

    it('takes the kbond keyword as the restraint constant (pe = 100 * 0.04 = 4)', async () => {
      const r = await pe('fix 2 all shake 1e-4 20 0 b 1 kbond 100', 'minimize 0 0 1 10');
      expect(r.error).toBeNull();
      expect(r.pe[0]).toBeCloseTo(4, 12);
    });

    it('refuses a negative kbond by name', async () => {
      const r = await pe('fix 2 all shake 1e-4 20 0 b 1 kbond -1', 'run 0');
      expect(r.error).toMatch(/Illegal fix shake kbond value -1/);
    });
  });
});
