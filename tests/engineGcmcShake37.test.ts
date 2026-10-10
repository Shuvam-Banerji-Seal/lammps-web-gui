import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix gcmc shake keyword (src/engine/fix/gcmc.ts): inserted molecules are
 * registered with fix shake after they are accepted, deleted molecules are
 * dropped, and the keyword's documented restrictions raise StyleErrors. The
 * draw-by-draw and energy checks are the w37gcmc_* oracle cases; these tests
 * pin the bookkeeping and the error texts.
 */

const MOL = readFileSync(join(__dirname, 'oracle', 'w37gcmc_dimer_shake.mol'), 'utf8');

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  session.addFile('w37gcmc_dimer_shake.mol', MOL);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { error, rows };
};

/** A bonded pair (r0 1.0, stretched to 1.2 in the template) in a 10 A box, one gcmc fix. */
const script = (gcmc: string, shake = 'fix wshake pair shake 0.0001 50 0 b 1 mol dpair', steps = 3) => `units real
atom_style molecular
boundary p p p
pair_style lj/cut 8.0
pair_modify mix arithmetic
bond_style harmonic
region box block 0 10 0 10 0 10 units box
create_box 2 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 2
molecule dpair w37gcmc_dimer_shake.mol
pair_coeff 1 1 0.15535 3.166
pair_coeff * 2 0.0 0.0
bond_coeff 1 1000 1.0
mass 1 15.9994
mass 2 1.0
group pair type 1 2
${shake}
fix 2 pair nve
${gcmc}
thermo_style custom step atoms pe ebond f_mygcmc[1] f_mygcmc[2] f_mygcmc[3] f_mygcmc[4] f_mygcmc[5] f_mygcmc[6]
thermo 1
timestep 1.0
run ${steps}
`;

describe('fix gcmc shake keyword', () => {
  it('accepts shake with mol and M = 0, and holds the inserted bond from the next step on', async () => {
    const { error, rows } = await run(script('fix mygcmc pair gcmc 1000 1 0 0 54341 338 5.0 0.5 mol dpair tfac_insert 1.6667 group pair shake wshake'));
    expect(error).toBeNull();
    // the inserting step still counts the stretched bond (0.2^2 * 1000 = 40 kcal/mol)
    expect(rows[1].atoms).toBe(2);
    expect(rows[1].ebond).toBeCloseTo(40, 6);
    // from the next step on the bond is a SHAKE constraint
    expect(rows[2].ebond).toBeLessThan(1e-9);
    expect(rows[3].ebond).toBeLessThan(1e-9);
  });

  it('keeps every molecule constrained through insertions and deletions', async () => {
    const { error, rows } = await run(script('fix mygcmc pair gcmc 2 1 0 0 54341 338 2.0 0.5 mol dpair tfac_insert 1.6667 group pair shake wshake', undefined, 40));
    expect(error).toBeNull();
    const last = rows[rows.length - 1] as ThermoRow;
    expect(last.atoms % 2).toBe(0);
    expect(last.ebond).toBeLessThan(1e-6);
  });

  it('rejects a trial insertion whose stretched bond makes the energy too high, leaving no atoms', async () => {
    const { error, rows } = await run(script('fix mygcmc pair gcmc 1000 1 0 0 54341 338 0.0 0.5 mol dpair full_energy tfac_insert 1.6667 group pair shake wshake'));
    expect(error).toBeNull();
    for (const r of rows) expect(r.atoms).toBe(0);
  });

  it('throws a StyleError naming the shake restriction when M is not zero', async () => {
    const { error } = await run(script('fix mygcmc pair gcmc 1000 1 1 0 54341 338 5.0 0.5 mol dpair tfac_insert 1.6667 group pair shake wshake'));
    expect(error).toContain('Cannot use fix gcmc shake with MC moves');
  });

  it('throws when shake is given without the mol keyword', async () => {
    const { error } = await run(script('fix mygcmc pair gcmc 1000 1 0 0 54341 338 5.0 0.5 tfac_insert 1.6667 group pair shake wshake'));
    expect(error).toContain('Cannot use fix gcmc shake and not molecule');
  });

  it('throws when the fix shake has a different molecule template', async () => {
    const { error } = await run(script('fix mygcmc pair gcmc 1000 1 0 0 54341 338 5.0 0.5 mol dpair shake wshake', 'fix wshake pair shake 0.0001 50 0 b 1'));
    expect(error).toContain('not using same molecule template ID');
  });

  it('throws when the shake ID does not exist', async () => {
    const { error } = await run(script('fix mygcmc pair gcmc 1000 1 0 0 54341 338 5.0 0.5 mol dpair shake nosuch'));
    expect(error).toContain('nosuch');
  });
});
