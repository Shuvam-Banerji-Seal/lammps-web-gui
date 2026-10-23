import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { slaterS, slaterShort, pointUval, pointU } from '../src/engine/force/pair/coul_streitz';

/*
 * Wave 43: pair_style coul/streitz dsf (taper width 0), the qeq/slater dsf pairing and the refusals of the
 * taper widths and of coul/ctip. Bit-level agreement with native LAMMPS is covered by the w43st_* oracle cases
 * (engineOracle.test.ts); this file checks the error paths and the allocation-free helpers.
 */

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const streitzFile = readFileSync(join(POT, 'AlO.streitz'), 'utf8');

const run = async (text: string, files: Record<string, string> = {}) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  for (const [name, body] of Object.entries(files)) session.addFile(name, body);
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const errEvent = events.find((e) => e.kind === 'error');
  return { error: error ?? (errEvent && 'message' in errEvent ? String((errEvent as { message: unknown }).message) : null) };
};

const head = `units metal
atom_style charge
boundary p p p
region box block 0 40 0 40 0 40
create_box 2 box
mass * 26.98
create_atoms 1 single 10 10 10
create_atoms 2 single 12 10 10
set atom 1 charge 1.0
set atom 2 charge -1.0
`;

describe('pair coul/streitz dsf keyword', () => {
  it('dsf takes alpha and an optional taper width', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 dsf 0.3\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toBeFalsy();
  });

  it('a taper width with dsf runs (the Slater part is tapered); wolf with a taper width is refused by name', async () => {
    const dsf = await run(`${head}pair_style coul/streitz 12.0 dsf 0.3 2.0\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(dsf.error).toBeFalsy();
    const wolf = await run(`${head}pair_style coul/streitz 12.0 wolf 0.3 2.0\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(wolf.error).toMatch(/wolf taper width 2 is not reproduced/);
  });

  it('dsf without its damping parameter names it', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 dsf\npair_coeff * * AlO.streitz Al O\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toContain('Illegal pair_style coul/streitz dsf command: missing argument(s)');
  });

  it('qeq/slater with dsf and a taper width runs', async () => {
    const out = await run(`${head}pair_style coul/streitz 12.0 dsf 0.3\npair_coeff * * AlO.streitz Al O\nfix 1 all qeq/slater 1 12.0 1.0e-6 100 coul/streitz alpha 0.3 dsf 2.0\nrun 0\n`, { 'AlO.streitz': streitzFile });
    expect(out.error).toBeFalsy();
  });
});

describe('Slater helpers', () => {
  it('slaterS and pointUval equal the values of slaterShort and pointU', () => {
    for (const [R, za, zb] of [[1.5, 0.968438, 2.143957], [3.2, 2.143957, 2.143957], [0.4, 0.968438, 0.968438], [11.0, 2.143957, 0.968438]] as const) {
      // the same formula with a different operation order: equal to the last few bits
      const s0 = slaterShort(R, za, zb).s, u0 = pointU(R, za).u;
      expect(Math.abs(slaterS(R, za, zb) - s0)).toBeLessThanOrEqual(1e-14 * Math.abs(s0));
      expect(Math.abs(pointUval(R, za) - u0)).toBeLessThanOrEqual(1e-14 * Math.abs(u0));
    }
  });
});
