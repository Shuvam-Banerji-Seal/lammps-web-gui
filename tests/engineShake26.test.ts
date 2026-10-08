import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix shake / rattle: the mol keyword and the constraint-force virial.
 *
 * mol — docs.lammps.org/fix_shake.html: "The mol keyword should be used when
 * other commands, such as fix deposit or fix pour, add molecules on-the-fly
 * during a simulation, and you wish to constrain the new molecules via
 * SHAKE." It only affects molecules added by those commands; a molecule that
 * already exists when the fix is defined is constrained only by the b/a/t/m
 * lists. Measured with native LAMMPS (black box): with a two-atom molecule
 * whose template carries Shake Flags 2, `shake ... mol t` leaves the existing
 * bond stretched (ebond 27.6957113756821), while `shake ... b 1 mol t`
 * constrains it (ebond 0). The template must exist: `mol nosuch` fails with
 * "Molecule template ID fix shake for nosuch does not exist".
 *
 * virial — the constraint force is solved per cluster and its virial is the
 * sum of lam_j r_j (x) r_j (docs.lammps.org/fix_shake.html: "the SHAKE
 * contribution to the pressure of the system (virial) is also accounted
 * for"). The oracle case w26shake_press compares it with native at rel 1e-9.
 */

const MOL2 = `# two-atom molecule with SHAKE info: one bond, flag 2
2 atoms
1 bonds

Coords

1 0.0 0.0 0.0
2 1.3 0.1 0.0

Types

1 1
2 1

Bonds

1 1 1 2

Shake Flags

1 2
2 2

Shake Atoms

1 1 2
2 1 2

Shake Bond Types

1 1
2 1
`;

const run = async (fixLine: string) => {
  const rows: ThermoRow[] = [];
  const s = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); }, writeFile: () => {} });
  s.addFile('m.txt', MOL2);
  try {
    await s.execute(`units real
atom_style full
region box block 0 20 0 20 0 20 units box
create_box 1 box bond/types 1 extra/bond/per/atom 1 extra/special/per/atom 2
mass 1 12.0
molecule t m.txt
create_atoms 0 single 10 10 10 mol t 1 units box
pair_style lj/cut 5.0
pair_coeff * * 0.0 1.0
bond_style harmonic
bond_coeff 1 300.0 1.0
fix 1 all nve
${fixLine}
thermo_style custom step temp pe ebond
thermo_modify format float %.15g
thermo 1
run 0`);
    return { rows, error: null as EngineError | null };
  } catch (e) { return { rows, error: e as EngineError }; }
};

describe('fix shake mol keyword', () => {
  it('accepts a valid template but does not constrain pre-existing molecules', async () => {
    const { rows, error } = await run('fix 2 all shake 1e-12 200 0 mol t');
    expect(error).toBeNull();
    // native: ebond 27.6957113756821 (bond not constrained)
    expect(rows[0].ebond).toBeCloseTo(27.6957113756821, 9);
  });

  it('constrains through b when combined with mol', async () => {
    const { rows, error } = await run('fix 2 all shake 1e-12 200 0 b 1 mol t');
    expect(error).toBeNull();
    expect(rows[0].ebond).toBe(0);
  });

  it('names a missing molecule template', async () => {
    const { error } = await run('fix 2 all shake 1e-12 200 0 mol nosuch');
    expect(error?.message).toMatch(/molecule template 'nosuch' does not exist/);
  });
});
