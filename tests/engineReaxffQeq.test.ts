import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { calcHValue, halfFlag, taperFix } from '../src/engine-gpl/reaxff/charge';
import { parsePertypeFile } from '../src/engine-gpl/reaxff/qeq';

/*
 * The charge models of pair_style reaxff ported with the QEq family (src/engine-gpl/reaxff/charge.ts,
 * qtpie.ts and acks2.ts, from LAMMPS src/REAXFF at b332f28): the refusals (fix efield, dual, the
 * parameter-file errors) and the shared helpers. The numerical agreement with native LAMMPS is in the
 * w65reaxq oracle cases (tests/engineOracle.test.ts).
 */

const CASES = join(__dirname, 'oracle');

const runScript = async (text: string, files: Record<string, string>): Promise<EngineEvent[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, 1);
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  await session.execute(text);
  return events;
};

const water = readFileSync(join(CASES, 'w65reaxq_water.data'), 'utf8');
const files = {
  'w65reaxq_water.data': water,
  'w65reaxq_acks2_ff.water': readFileSync(join(CASES, 'w65reaxq_acks2_ff.water'), 'utf8'),
  'w65reaxq_qeq_ff.water': readFileSync(join(CASES, 'w65reaxq_qeq_ff.water'), 'utf8'),
  'w65reaxq_gauss_exp.txt': readFileSync(join(CASES, 'w65reaxq_gauss_exp.txt'), 'utf8'),
};
const head = 'units real\natom_style charge\nboundary p p p\nread_data w65reaxq_water.data\npair_style reaxff NULL safezone 3.0 mincap 150\npair_coeff * * w65reaxq_qeq_ff.water O H\n';

describe('ReaxFF charge models: refusals', () => {
  it('fix efield: qeq/reaxff and acks2/reaxff refuse a field along a periodic direction (boundary p p p)', async () => {
    await expect(runScript(`${head}fix 2 all acks2/reaxff 1 0.0 10.0 1e-6 reaxff\nfix 3 all efield 0.0 0.0 0.05\nrun 0\n`, files))
      .rejects.toThrow(/Must not have electric field component in direction of periodic boundary/);
  });

  it('fix efield: qtpie/reaxff and qeq/rel/reaxff accept a field along a periodic direction (no such check in their init)', async () => {
    await expect(runScript(`${head}fix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff w65reaxq_gauss_exp.txt\nfix 3 all efield 0.0 0.0 0.05\nrun 0\n`, files)).resolves.toBeDefined();
    await expect(runScript(`${head}fix 2 all qeq/rel/reaxff 1 0.0 10.0 1e-6 reaxff w65reaxq_gauss_exp.txt\nfix 3 all efield 0.0 0.0 0.05\nrun 0\n`, files)).resolves.toBeDefined();
  });

  it('fix efield: a non-periodic z (boundary p p s) lets acks2/reaxff take the field', async () => {
    await expect(runScript(`${head.replace('boundary p p p', 'boundary p p s')}fix 2 all acks2/reaxff 1 0.0 10.0 1e-6 reaxff\nfix 3 all efield 0.0 0.0 1.0\nrun 0\n`, files)).resolves.toBeDefined();
  });

  it('fix efield: region and atom-style field variables are refused by name', async () => {
    await expect(runScript(`${head}region r block 0 31 0 31 0 31 units box\nfix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff w65reaxq_gauss_exp.txt\nfix 3 all efield 0.0 0.0 0.05 region r\nrun 0\n`, files))
      .rejects.toThrow(/region/);
    await expect(runScript(`${head}variable ez atom 0.05\nfix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff w65reaxq_gauss_exp.txt\nfix 3 all efield 0.0 0.0 v_ez\nrun 0\n`, files))
      .rejects.toThrow(/Atom-style/);
  });

  it('refuses the dual and matfree variants of fix acks2/reaxff by name', async () => {
    await expect(runScript(`${head}fix 2 all acks2/reaxff 1 0.0 10.0 1e-6 reaxff dual\nrun 0\n`, files)).rejects.toThrow(/dual/);
    await expect(runScript(`${head}fix 2 all acks2/reaxff 1 0.0 10.0 1e-6 reaxff matfree\nrun 0\n`, files)).rejects.toThrow(/matfree/);
  });

  it('refuses the obsolete reax/c keyword of fix qtpie/reaxff and an unknown keyword', async () => {
    await expect(runScript(`${head}fix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reax/c w65reaxq_gauss_exp.txt\nrun 0\n`, files)).rejects.toThrow(/reax\/c/);
    await expect(runScript(`${head}fix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff w65reaxq_gauss_exp.txt frobnicate\nrun 0\n`, files)).rejects.toThrow(/frobnicate/);
  });

  it('reads the Gaussian exponents of the gauss file and refuses a bad one', async () => {
    const bad = { ...files, 'bad_gauss.txt': '1 0.2240\n2 -1.0\n' };
    await expect(runScript(`${head}fix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff bad_gauss.txt\nrun 0\n`, bad))
      .rejects.toThrow(/Invalid orbital exponent/);
    const short = { ...files, 'short_gauss.txt': '1 0.2240\n' };
    await expect(runScript(`${head}fix 2 all qtpie/reaxff 1 0.0 10.0 1e-6 reaxff short_gauss.txt\nrun 0\n`, short))
      .rejects.toThrow(/Incorrect number of atom types/);
  });
});

describe('ReaxFF charge models: shared helpers', () => {
  it('parses the per-type parameter file of fix qeq/reaxff and rejects a short one', () => {
    const p = parsePertypeFile('1 4.0 7.0 0.6\n2 4.5 6.5 0.8\n', 2, 'qeq/reaxff');
    expect(p.chi[2]).toBe(4.5);
    expect(p.eta[1]).toBe(7.0);
    expect(p.gamma[2]).toBe(0.8);
    expect(() => parsePertypeFile('1 4.0 7.0 0.6\n', 2, 'qeq/reaxff')).toThrow(/Invalid param file format/);
  });

  it('the shielded H value at zero distance is the taper constant over gamma^(1/3)', () => {
    const Tap = taperFix(0.0, 10.0);
    const gamma = 1.7;
    expect(calcHValue(Tap, 0.0, gamma)).toBeCloseTo((Tap[0] * 14.4) / Math.cbrt(gamma), 12);
    // the taper vanishes at the upper cutoff
    expect(Math.abs(calcHValue(Tap, 10.0, gamma))).toBeLessThan(1e-12);
  });

  it('the half-list flag takes exactly one of two mirrored ghost images', () => {
    const orig = new Int32Array([7, 7]);
    // owned atom 0 and the ghost image 1 of the same atom, displaced by +x: only one of (0,1) and (1,0) passes
    const forward = halfFlag(1.0, 0, 0, 1.0, 4.0, 0, 1, 1, orig);
    const backward = halfFlag(-1.0, 0, 0, 1.0, 4.0, 0, 1, 1, orig);
    expect(forward !== backward).toBe(true);
    // an owned partner is always taken
    expect(halfFlag(3, 0, 0, 9, 16, 0, 1, 2, new Int32Array([1, 2]))).toBe(true);
    // beyond the cutoff nothing passes
    expect(halfFlag(5, 0, 0, 25, 16, 0, 1, 2, new Int32Array([1, 2]))).toBe(false);
  });
});
