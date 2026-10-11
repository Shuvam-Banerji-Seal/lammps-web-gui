import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * w47ls: min_modify line quadratic (the default). Native counts measured with LAMMPS (black box) on the 108-atom
 * perturbed lj crystal tests/oracle/w47ls_lj.data (the summary line prints "Iterations, force evaluations"):
 * Measured with native LAMMPS (black box): cg with line quadratic, 30 iterations: 30 iterations, 59 force evaluations;
 * sd, 30 iterations: 30 and 59; cg to etol 1e-10: 15 iterations, 29 force evaluations.
 */
const lj = readFileSync(join(__dirname, 'oracle', 'w47ls_lj.data'), 'utf8');
const head = `units lj
atom_style atomic
read_data w47ls_lj.data
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
neighbor 0.3 bin
thermo_style custom step pe fnorm
thermo_modify format float %.15g
thermo 1
`;

const run = async (tail: string): Promise<string[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('w47ls_lj.data', lj);
  await session.execute(head + tail);
  return events.map((e) => JSON.stringify(e)).filter((s) => s.includes('force evaluations'));
};

describe('w47ls line search counts (native-measured)', () => {
  it('cg line quadratic: 30 iterations, 59 force evaluations', async () => {
    const lines = await run('min_style cg\nminimize 0 0 30 1000\n');
    expect(lines.join('\n')).toContain('Iterations, force evaluations = 30 59');
  }, 120000);

  it('sd line quadratic: 30 iterations, 59 force evaluations', async () => {
    const lines = await run('min_style sd\nminimize 0 0 30 1000\n');
    expect(lines.join('\n')).toContain('Iterations, force evaluations = 30 59');
  }, 120000);

  it('cg line quadratic to convergence: 15 iterations, 29 force evaluations', async () => {
    const lines = await run('min_style cg\nminimize 1e-10 0 200 2000\n');
    expect(lines.join('\n')).toContain('Iterations, force evaluations = 15 29');
  }, 120000);
});
