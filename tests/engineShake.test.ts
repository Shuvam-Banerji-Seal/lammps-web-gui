import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
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

  it('is an explicit error during minimization', async () => {
    const dimer = '# dimer\n2 atoms\n1 bonds\n\nCoords\n\n1 0 0 0\n2 1.3 0 0\n\nTypes\n\n1 1\n2 1\n\nBonds\n\n1 1 1 2\n';
    const { error } = await run(dimer, 'minimize 1e-8 1e-8 10 10');
    expect(error?.message).toMatch(/fix shake during minimization .* is not supported/);
  });
});
