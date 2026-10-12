import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Regression tests for two engine-vs-native discrepancies (w60fix):
 *  - bonded partners use the closest periodic ghost image (bonded_util.ts bondDelta);
 *  - fix indent side in acts on atoms outside the indenter (fix/indent.ts).
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

/** Two atoms at x = 0.5 and x = x2 in a box periodic in x (length 12), bonded harmonic K = 1, r0 = 1. */
const pairInput = (x2: number, order: string): string => `
units lj
atom_style bond
boundary p p p
region box block 0 12 0 12 0 12
create_box 1 box bond/types 1 extra/bond/per/atom 2
create_atoms 1 single 0.5 5 5 units box
create_atoms 1 single ${x2} 5 5 units box
mass * 1.0
pair_style zero 2.0
pair_coeff * *
bond_style harmonic
bond_coeff 1 1.0 1.0
create_bonds single/bond 1 ${order}
thermo_style custom step pe ebond
thermo_modify format float %.15g norm no
run 0
`;

describe('bond partner image (closest ghost copy)', () => {
  // Measured with native LAMMPS (black box), comm cutoff 2.3 (pair cutoff 2.0 + skin 0.3):
  // partner 9.0 (image -3.0 is not a ghost): 56.25 = (8.5 - 1)^2; partner 10.0 (image -2.0 is a ghost): 2.25
  it.each([
    [11.0, '1 2', 0.25],
    [10.0, '1 2', 2.25],
    [9.0, '1 2', 56.25],
    [8.0, '1 2', 42.25],
    [7.0, '1 2', 30.25],
    [9.0, '2 1', 6.25],
  ])('partner at %s, bond %s: energy %s', async (x2, order, e) => {
    const rows = await thermo(pairInput(x2, order));
    expect(rows[0].pe).toBeCloseTo(e, 9);
  });
});

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
