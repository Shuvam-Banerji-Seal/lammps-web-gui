import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineError, EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix rigid langevin (src/engine/fix/rigid.ts): the keyword parses, names bad
 * arguments, and reproduces the same stream for the same seed. The dynamics
 * against native LAMMPS are the oracle cases w36rigid_lgv1 and w36rigid_lgv_mol.
 */

const body = (fix: string) => `
units lj
atom_style molecular
lattice sc 1.5
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
set atom 1*14 mol 1
set atom 15*27 mol 2
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
${fix}
thermo_style custom step temp ke
thermo_modify format float %.15g
run 10`;

const run = async (fix: string): Promise<{ rows: ThermoRow[] } | EngineError> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await s.execute(body(fix));
    return { rows: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row) };
  } catch (e) { return e as EngineError; }
};

describe('fix rigid langevin', () => {
  it('accepts langevin on the NVE rigid styles', async () => {
    for (const style of ['rigid', 'rigid/nve', 'rigid/small', 'rigid/nve/small']) {
      const r = await run(`fix 1 all ${style} molecule langevin 1.0 1.0 0.5 4567`);
      expect(r, style).not.toBeInstanceOf(Error);
    }
  });

  it('names bad langevin arguments', async () => {
    expect((await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5') as EngineError).message).toMatch(/expected Tstart Tstop Tperiod seed/);
    expect((await run('fix 1 all rigid molecule langevin 1.0 1.0 0.0 4567') as EngineError).message).toMatch(/Tperiod must be > 0/);
    expect((await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 0') as EngineError).message).toMatch(/seed must be a positive integer/);
    expect((await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 1.5') as EngineError).message).toMatch(/seed must be a positive integer/);
    expect((await run('fix 1 all rigid molecule langevin 1.0 x 0.5 4567') as EngineError).message).toMatch(/Tstop must be a number/);
  });

  it('is deterministic for a given seed and differs across seeds', async () => {
    const a = (await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 4567')) as { rows: ThermoRow[] };
    const b = (await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 4567')) as { rows: ThermoRow[] };
    const c = (await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 7654')) as { rows: ThermoRow[] };
    expect(a.rows.map((r) => r.temp)).toEqual(b.rows.map((r) => r.temp));
    expect(a.rows.map((r) => r.temp)).not.toEqual(c.rows.map((r) => r.temp));
  });

  it('heats a body at rest (random force) and damps it at T = 0 (drag)', async () => {
    const cold = (await run('fix 1 all rigid molecule langevin 1.0 1.0 0.5 4567')) as { rows: ThermoRow[] };
    expect(cold.rows[0].temp).toBe(0);
    expect(cold.rows[cold.rows.length - 1].temp).toBeGreaterThan(0);
    const damp = (await run('fix 1 all rigid molecule langevin 0.0 0.0 0.5 4567')) as { rows: ThermoRow[] };
    expect(damp.rows[damp.rows.length - 1].ke).toBeLessThanOrEqual(damp.rows[0].ke + 1e-12);
  });
});
