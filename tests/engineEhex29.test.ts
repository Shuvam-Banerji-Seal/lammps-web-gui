import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix ehex with the constrain keyword (docs.lammps.org/fix_ehex.html):
 * "If either of these constraining algorithms is specified in the input script
 * and the keyword *constrain* is set, the bond distances will be corrected a
 * second time at the end of the integration step." The numerical agreement
 * with native LAMMPS is in tests/oracle/w29ehex_*.in; here the argument checks
 * and the shake/rattle requirement are exercised.
 */

const runScript = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  for (const [name, content] of Object.entries(files)) session.addFile(name, content);
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { thermo, error };
};

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

const WATER = readFileSync(join(__dirname, 'oracle', 'w29ehex_water.data'), 'utf8');

/** A 24-atom rigid-water system: one hot and one cold reservoir. */
const BODY = `
units           real
atom_style      full
boundary        p p p
read_data       w29ehex_water.data
pair_style      lj/cut 8.0
pair_coeff      * * 0.05 3.0
pair_coeff      1 1 0.1553 3.166
bond_style      harmonic
bond_coeff      1 450.0 1.0
angle_style     harmonic
angle_coeff     1 55.0 109.47
region          hot block INF INF INF INF 0 5.0 units box
region          cold block INF INF INF INF 15.0 20.0 units box
velocity        all create 400.0 4321 mom yes rot yes dist gaussian
timestep        1.0
fix             1 all nve
`;

describe('fix ehex constrain', () => {
  it('requires a fix shake or rattle', async () => {
    const { error } = await runScript(`${BODY}
fix             3 all ehex 1 0.05 region hot constrain
run             1
`, { 'w29ehex_water.data': WATER });
    expect(error).not.toBeNull();
    expect(errorText(error)).toMatch(/constrain.*shake|rattle/i);
  });

  it('rejects com (not implemented)', async () => {
    const { error } = await runScript(`${BODY}
fix             2 all shake 1.0e-10 400 0 b 1 a 1
fix             3 all ehex 1 0.05 region hot constrain com
run             1
`, { 'w29ehex_water.data': WATER });
    expect(errorText(error)).toMatch(/com.*not implemented/);
  });

  it('rejects com without constrain', async () => {
    const { error } = await runScript(`${BODY}
fix             2 all shake 1.0e-10 400 0 b 1 a 1
fix             3 all ehex 1 0.05 region hot com
run             1
`, { 'w29ehex_water.data': WATER });
    expect(errorText(error)).toMatch(/together with the keyword 'constrain'/);
  });

  for (const style of ['shake', 'rattle'] as const) {
    it(`runs with fix ${style} and reports the ehex scalar`, async () => {
      const { thermo, error } = await runScript(`${BODY}
fix             2 all ${style} 1.0e-12 400 0 b 1 a 1
fix             3 all ehex 1 0.05 region hot constrain
fix             4 all ehex 1 -0.05 region cold constrain
thermo_style    custom step temp pe ke etotal press f_3 f_4
thermo_modify   format float %.15g
thermo          1
run             20
`, { 'w29ehex_water.data': WATER });
      expect(error).toBeNull();
      expect(thermo.length).toBe(21);
      expect(thermo[0].f_3).toBe(1);
      // the hot reservoir is heated (scalar > 1), the cold one cooled (scalar < 1)
      expect(Number(thermo[1].f_3)).toBeGreaterThan(1);
      expect(Number(thermo[1].f_4)).toBeLessThan(1);
      for (const r of thermo) expect(Number.isFinite(r.temp as number)).toBe(true);
    });
  }

  it('hex constrain also re-applies the constraint', async () => {
    const { thermo, error } = await runScript(`${BODY}
fix             2 all shake 1.0e-12 400 0 b 1 a 1
fix             3 all ehex 1 0.05 region hot constrain hex
thermo_style    custom step f_3
thermo_modify   format float %.15g
thermo          1
run             5
`, { 'w29ehex_water.data': WATER });
    expect(error).toBeNull();
    expect(thermo.length).toBe(6);
  });

  it('constrain is only applied on the ehex nevery steps', async () => {
    // nevery 3: the ehex scalar is 1 on the steps in between and changes on multiples of 3
    const { thermo, error } = await runScript(`${BODY}
fix             2 all shake 1.0e-12 400 0 b 1 a 1
fix             3 all ehex 3 0.1 region hot constrain
thermo_style    custom step f_3
thermo_modify   format float %.15g
thermo          1
run             7
`, { 'w29ehex_water.data': WATER });
    expect(error).toBeNull();
    expect(thermo[0].f_3).toBe(1);
    expect(thermo[1].f_3).toBe(1);
    expect(thermo[2].f_3).toBe(1);
    expect(Number(thermo[3].f_3)).toBeGreaterThan(1);
    expect(thermo[4].f_3).toBe(Number(thermo[3].f_3));
    expect(Number(thermo[6].f_3)).toBeGreaterThan(1);
  });
});
