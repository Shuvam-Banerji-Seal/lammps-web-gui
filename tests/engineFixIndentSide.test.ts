import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Regression tests for fix indent side in / side out (fix/indent.ts): side in acts on atoms outside the
 * indenter, side out on atoms inside it.
 * The expected values are measured with native LAMMPS (black box); the oracle
 * cases tests/oracle/w60fix_*.in hold the full per-step comparisons.
 */

/** Runs a LAMMPS input and returns its thermo rows. */
const thermo = async (text: string): Promise<ThermoRow[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  await session.execute(text);
  return events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
};

describe('fix indent side in and side out (single atom, sphere K = 1, R = 1)', () => {
  // Measured with native LAMMPS (black box). Atom at the origin; "fix 2 all indent 1.0 sphere X 0 0 1.0 units box side S".
  const one = async (cx: number, side: 'in' | 'out') => {
    const rows = await thermo(`
units lj
dimension 3
boundary f f f
atom_style atomic
region box block -10 10 -10 10 -10 10 units box
create_box 1 box
create_atoms 1 single 0.0 0.0 0.0 units box
mass * 1.0
pair_style zero 1.0
pair_coeff * *
velocity all set 0.0 0.0 0.0 sum no units box
fix 1 all nve
fix 2 all indent 1.0 sphere ${cx} 0.0 0.0 1.0 units box side ${side}
fix_modify 2 energy yes
timestep 0.001
thermo_style custom step pe f_2 f_2[1]
thermo_modify format float %.15g
thermo_modify lost ignore
run 1
`);
    return rows[0];
  };

  it('side in acts on an atom outside the sphere: energy K/3 (r-R)^3 = 1/3, force toward the centre', async () => {
    const r = await one(2.0, 'in');
    expect(r.pe).toBeCloseTo(1 / 3, 6);
    // the indenter feels minus the force on the atom: force on the atom is +1 along x
    expect(r['f_2[1]'] as number).toBeCloseTo(-1, 5);
  });

  it('side in ignores an atom inside the sphere', async () => {
    const r = await one(0.5, 'in');
    expect(r.pe).toBeCloseTo(0, 12);
    expect(r['f_2[1]'] as number).toBeCloseTo(0, 12);
  });

  it('side out acts on an atom inside the sphere: energy K/3 (R-r)^3 = 1/24, force away from the centre', async () => {
    const r = await one(0.5, 'out');
    expect(r.pe).toBeCloseTo(1 / 24, 6);
    expect(r['f_2[1]'] as number).toBeCloseTo(0.25, 5);
  });

  it('side out ignores an atom outside the sphere', async () => {
    const r = await one(2.0, 'out');
    expect(r.pe).toBeCloseTo(0, 12);
  });
});
