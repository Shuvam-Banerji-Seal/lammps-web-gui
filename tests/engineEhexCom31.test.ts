import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix ehex with the com keyword (docs.lammps.org/fix_ehex.html): "With this
 * option all sites of a constrained cluster are rescaled, if its center of mass
 * is located inside the region." The numerical agreement with native LAMMPS is
 * in tests/oracle/w31ehexcom_*.in; here the argument checks and the input
 * rejections are exercised.
 */

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
const DATA = readFileSync(join(CASES, 'w29ehex_water.data'), 'utf8');

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
region          hot block 0.0 5.5 INF INF INF INF units box
region          cold block 9.5 15.5 INF INF INF INF units box
velocity        all create 400.0 4321 mom yes rot yes dist gaussian
timestep        1.0
fix             1 all nve
`;

describe('fix ehex com', () => {
  it('requires constrain together with com', async () => {
    const { error } = await runScript(`${BODY}
fix             2 all shake 1.0e-10 400 0 b 1 a 1
fix             3 all ehex 1 0.05 region hot com
run             1
`, { 'w29ehex_water.data': DATA });
    expect(errorText(error)).toMatch(/together with the keyword 'constrain'/);
  });

  it('requires a fix shake or rattle with com+constrain', async () => {
    const { error } = await runScript(`${BODY}
fix             3 all ehex 1 0.05 region hot constrain com
run             1
`, { 'w29ehex_water.data': DATA });
    expect(error).not.toBeNull();
    expect(errorText(error)).toMatch(/constrain.*shake|rattle/i);
  });

  // Diagnostics for the oracle cases: engine vs native fixture, first divergent column.
  for (const name of ['w31ehexcom_shake', 'w31ehexcom_rattle', 'w31ehexcom_hex', 'w31ehexcom_nevery']) {
    it(`matches native: ${name}`, async () => {
      const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
      const fx = JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')) as { thermo: ThermoRow[] };
      const { thermo: rows, error } = await runScript(text, { 'w29ehex_water.data': DATA });
      expect(error).toBeNull();
      expect(rows.length).toBe(fx.thermo.length);
      const close = (a: number, b: number) => (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= 1e-10 + 1e-8 * Math.max(Math.abs(a), Math.abs(b));
      for (let r = 0; r < fx.thermo.length; r++) {
        for (const [k, v] of Object.entries(fx.thermo[r])) {
          const g = Number(rows[r][k]);
          if (!close(g, Number(v))) {
            throw new Error(`${name}: row ${r} (step ${fx.thermo[r].step}) ${k}: engine ${g} vs LAMMPS ${v}`);
          }
        }
      }
    });
  }
});
