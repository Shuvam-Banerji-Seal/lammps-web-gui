import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * min16: line search of min_style sd/cg. Expected values were measured with native LAMMPS (black box) on the
 * 8-atom chain tests/oracle/w16min_chain.data: one iteration of min_style sd, then the printed energy.
 * Measured with native LAMMPS (black box): line backtrack, dmax 0.02 accepts the first trial (1 force evaluation);
 * dmax 0.1 halves three times (4 evaluations); dmax 0.5 halves five times (6 evaluations).
 */
const dir = join(__dirname, 'oracle');
const chain = readFileSync(join(dir, 'w16min_chain.data'), 'utf8');
const head = `units real
atom_style full
boundary f f f
read_data w16min_chain.data
bond_style harmonic
angle_style harmonic
dihedral_style harmonic
pair_style zero 10.0
pair_coeff * *
bond_coeff 1 300.0 1.53
angle_coeff 1 60.0 111.0
dihedral_coeff 1 2.0 1 3
thermo_style custom step pe
thermo_modify format float %.15g
thermo 1
`;

const run = async (tail: string): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile('w16min_chain.data', chain);
  await session.execute(head + tail);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

describe('min16 line search (native-measured)', () => {
  it('backtrack: first trial accepted when dmax is small', async () => {
    const rows = await run('min_style sd\nmin_modify dmax 0.02 line backtrack\nminimize 1.0e-10 1.0e-10 1 50\n');
    expect(rows[rows.length - 1].pe).toBeCloseTo(13.4798531883798, 9);
  });

  it('backtrack: halving to alpha0/8 then accepted (dmax 0.1)', async () => {
    const rows = await run('min_style sd\nmin_modify dmax 0.1 line backtrack\nminimize 1.0e-10 1.0e-10 1 50\n');
    expect(rows[rows.length - 1].pe).toBeCloseTo(13.5105358410885, 9);
  });

  it('backtrack: halving to alpha0/32 (dmax 0.5)', async () => {
    const rows = await run('min_style sd\nmin_modify dmax 0.5 line backtrack\nminimize 1.0e-10 1.0e-10 1 50\n');
    expect(rows[rows.length - 1].pe).toBeCloseTo(13.4877808078577, 9);
  });

  it('forcezero is refused with a named StyleError (native differs from quadratic)', async () => {
    await expect(run('min_style sd\nmin_modify line forcezero\nminimize 1.0e-10 1.0e-10 1 50\n')).rejects.toThrow(/forcezero/);
  });

  it('sanity: the chain minimizes monotonically under backtracking', async () => {
    const rows = await run('min_style sd\nmin_modify line backtrack\nminimize 1.0e-10 1.0e-10 5 50\n');
    for (let i = 1; i < rows.length; i++) expect(rows[i].pe).toBeLessThanOrEqual(rows[i - 1].pe + 1e-12);
  });
});
