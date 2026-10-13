import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

// The native G-ewald and mesh of kspace_style pppm in triclinic boxes, and of kspace_modify diff ad (src/engine-gpl/pppm),
// pinned to the values native LAMMPS prints (the "G vector" and "grid" lines of its log, tests/oracle/w101pppm_*.in).
// The full thermo and per-atom parity is in tests/engineOracle.test.ts (w101pppm_*).

const ORACLE = join(__dirname, 'oracle');

/** Runs a case up to its first run and returns the kspace log lines and the errors. */
const runCase = async (text: string): Promise<{ log: string[]; errors: string[] }> => {
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never, undefined, 0, 1);
  try {
    await s.execute(text);
  } catch (e) {
    events.push({ kind: 'error', message: e instanceof Error ? e.message : String(e) } as never);
  }
  const log = events.filter((e) => e.kind === 'log').map((e) => String((e as { text?: string }).text ?? ''));
  const errors = events.filter((e) => e.kind === 'error').map((e) => String((e as { message?: string }).message ?? ''));
  return { log, errors };
};

const caseText = (name: string): string => readFileSync(join(ORACLE, `${name}.in`), 'utf8').replace(/^#.*$/gm, '');

const gridLine = (log: string[]): string => log.find((l) => /grid =/.test(l)) ?? '';
const gLine = (log: string[]): number => {
  const m = log.join('\n').match(/G vector \(1\/distance\) = ([0-9.eE+-]+)/);
  return m ? Number(m[1]) : NaN;
};

describe('kspace_style pppm: native G-ewald and mesh in triclinic boxes', () => {
  // native (black box, OMP_NUM_THREADS=1): "G vector (1/distance) = <G>", "grid = <n>" for each case
  const cases: Array<[string, number, string]> = [
    ['w101pppm_tri_auto_a', 0.32991775, '8 9 9'],
    ['w101pppm_tri_auto_b', 0.3742342, '10 12 15'],
    ['w101pppm_tri_auto_c', 0.41818585, '15 20 15'],
    ['w101pppm_tri_auto_d', 0.32807317, '8 8 10'],
  ];
  for (const [name, g, grid] of cases) {
    it(`${name}: G and grid match the native log`, async () => {
      const { log, errors } = await runCase(caseText(name));
      expect(errors).toEqual([]);
      expect(gridLine(log)).toContain(`grid = ${grid}`);
      // the native line prints 8 significant digits; the engine solves the same Newton problem
      expect(Math.abs(gLine(log) - g) / g).toBeLessThan(1e-7);
    });
  }

  it('w101pppm_tri_fixed: kspace_modify gewald and mesh are used as given', async () => {
    const { log, errors } = await runCase(caseText('w101pppm_tri_fixed'));
    expect(errors).toEqual([]);
    expect(gLine(log)).toBe(0.33);
    expect(gridLine(log)).toContain('grid = 12 12 16');
  });
});

describe('kspace_modify diff ad (src/engine-gpl/pppm)', () => {
  // native: ad on an orthogonal box with automatic G and grid
  const cases: Array<[string, number, string]> = [
    ['w101pppm_ad_auto_a', 0.29307262, '6 6 6'],
    ['w101pppm_ad_auto_b', 0.35984748, '10 12 15'],
  ];
  for (const [name, g, grid] of cases) {
    it(`${name}: G and grid match the native log`, async () => {
      const { log, errors } = await runCase(caseText(name));
      expect(errors).toEqual([]);
      expect(gridLine(log)).toContain(`grid = ${grid}`);
      expect(Math.abs(gLine(log) - g) / g).toBeLessThan(1e-7);
    });
  }

  it('w101pppm_ad_fixed: gewald and mesh given, as native', async () => {
    const { log, errors } = await runCase(caseText('w101pppm_ad_fixed'));
    expect(errors).toEqual([]);
    expect(gLine(log)).toBe(0.3);
    expect(gridLine(log)).toContain('grid = 16 16 16');
  });

  it('native refuses diff ad with a triclinic box; so does the engine, by the same message', async () => {
    const text = caseText('w101pppm_ad_auto_a').replace('region          box block 0 12 0 12 0 12 units box', 'region          box prism 0 12 0 12 0 12 1 0 0 units box');
    const { errors } = await runCase(text);
    expect(errors.join('\n')).toMatch(/Cannot \(yet\) use PPPM with triclinic box and kspace_modify diff ad/);
  });

  it('diff ad is refused for the styles the port does not build (ewald, pppm/cg)', async () => {
    const ew = await runCase(caseText('w101pppm_ad_auto_a').replace('kspace_style    pppm 1.0e-4', 'kspace_style    ewald 1.0e-4'));
    expect(ew.errors.join('\n')).toMatch(/diff ad is not supported by kspace_style ewald/);
    const cg = await runCase(caseText('w101pppm_ad_auto_a').replace('kspace_style    pppm 1.0e-4', 'kspace_style    pppm/cg 1.0e-4'));
    expect(cg.errors.join('\n')).toMatch(/diff ad is not supported by kspace_style pppm\/cg/);
  });
});
