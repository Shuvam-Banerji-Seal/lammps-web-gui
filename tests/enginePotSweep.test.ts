import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Reader checks found by the potential sweep (tests/oracle/w22pot_*.in compare the real files with
 * native LAMMPS; these tests pin the reader decisions that the oracle cases rely on, on small inputs).
 */

const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const pot = (name: string): string => readFileSync(join(POTENTIALS, name), 'utf8');

/** Runs a two-atom cell with the given potential text; returns the first error message, or null. */
const runWith = async (style: string, coeff: string, fileName: string, text: string, units = 'metal'): Promise<string | null> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  session.addFile(fileName, text);
  const script = `units ${units}
atom_style atomic
lattice diamond 5.431
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 28.06
pair_style ${style}
pair_coeff ${coeff}
run 0
`;
  try {
    await session.execute(script);
  } catch (e) {
    return String(e);
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? err.message : null;
};

describe('sw potential file reader', () => {
  it('accepts entries with sigma = 0 (GaN.sw carries them for three-body-only triplets)', async () => {
    expect(await runWith('sw', '* * GaN.sw Ga', 'GaN.sw', pot('GaN.sw'))).toBeNull();
  });

  it('refuses a negative sigma', async () => {
    const bad = pot('Si.sw').replace(/^(Si Si Si\s+\S+\s+)(\S+)/m, '$1-2.0');
    const err = await runWith('sw', '* * bad.sw Si', 'bad.sw', bad);
    expect(err).toMatch(/sigma and a must be >= 0/);
  });
});

describe('tersoff/mod potential file reader', () => {
  it('accepts a UNITS tag that matches the simulation units', async () => {
    expect(await runWith('tersoff/mod', '* * Si.tersoff.mod Si', 'Si.tersoff.mod', pot('Si.tersoff.mod'))).toBeNull();
  });

  it('refuses a UNITS tag that differs from the simulation units', async () => {
    const err = await runWith('tersoff/mod', '* * Si.tersoff.mod Si', 'Si.tersoff.mod', pot('Si.tersoff.mod').replace('UNITS: metal', 'UNITS: real'));
    expect(err).toMatch(/does not convert potential units/);
  });

  it('refuses beta values other than 1 or 3 (native stops with an illegal Tersoff parameter)', async () => {
    const bad = pot('Si.tersoff.mod').replace(/^Si\s+Si\s+Si\s+1\.0\s+/m, 'Si Si Si 2.0 ');
    const err = await runWith('tersoff/mod', '* * bad.mod Si', 'bad.mod', bad);
    expect(err).toMatch(/beta must be 1 or 3/);
  });
});
