/*
 * pair_style sw: the GPL port (src/engine-gpl/manybody3/pair_sw.ts) is the registered 'sw'; sw/mod stays in the core.
 * The native reference numbers below were measured with native LAMMPS (b332f28) on the step-0 configuration of
 * tests/oracle/w22pot_sw_cdtezn.in (CdTeZnSeHgS0.sw, Cd Te Zn, three types); the full cases are in the oracle suite.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { PAIRS as GPL_PAIRS } from '../src/engine-gpl/registry/sw';
import { PAIRS as CORE_PAIRS } from '../src/engine/registry/pair_sw';

const POT = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const CASE = join(__dirname, 'oracle', 'w22pot_sw_cdtezn.in');

/** Native LAMMPS step-0 PotEng of the cdtezn configuration (pair_style sw, threebody on and threebody off). */
const NATIVE_PE_ON = -127.729453062231;
const NATIVE_PE_OFF = -155.931111308207;

/** Runs the cdtezn setup with the given pair_style and pair_coeff lines and returns the step-0 events. */
const setup = async (pairStyle: string, pairCoeff: string): Promise<EngineEvent[]> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  session.addFile('CdTeZnSeHgS0.sw', readFileSync(join(POT, 'CdTeZnSeHgS0.sw'), 'utf8'));
  const head = readFileSync(CASE, 'utf8').split('\n')
    .filter((l) => !/^(run|thermo|timestep|fix|velocity|pair_style|pair_coeff|#)/.test(l));
  const script = [
    ...head,
    `pair_style ${pairStyle}`,
    pairCoeff,
    'thermo_style custom step pe',
    'thermo_modify format float %.15g norm no',
    'run 0',
  ].join('\n');
  await session.execute(`${script}\n`);
  return events;
};

const pe = (events: EngineEvent[]): number => {
  const row = events.find((e) => e.kind === 'thermo');
  if (!row || row.kind !== 'thermo') throw new Error(`no thermo row: ${JSON.stringify(events.filter((e) => e.kind === 'error'))}`);
  return row.row.pe as number;
};

/** The message of the error the setup raises (the engine throws EngineError), or undefined when it runs. */
const errorOf = async (pairStyle: string, pairCoeff: string): Promise<string | undefined> => {
  try {
    await setup(pairStyle, pairCoeff);
    return undefined;
  } catch (e) {
    return (e as Error).message;
  }
};

describe('pair_style sw registry', () => {
  it('registers plain sw from the GPL port and keeps sw/mod in the core', () => {
    expect(Object.keys(GPL_PAIRS)).toEqual(['sw']);
    expect(Object.keys(CORE_PAIRS)).toEqual(['sw/mod']);
  });
});

describe('pair_style sw native parity (step 0 of w22pot_sw_cdtezn)', () => {
  it('matches the native PotEng with and without the three-body term', async () => {
    const on = pe(await setup('sw', 'pair_coeff * * CdTeZnSeHgS0.sw Cd Te Zn'));
    const off = pe(await setup('sw threebody off', 'pair_coeff * * CdTeZnSeHgS0.sw Cd Te Zn'));
    expect(Math.abs(on - NATIVE_PE_ON)).toBeLessThanOrEqual(1e-10 * Math.abs(NATIVE_PE_ON));
    expect(Math.abs(off - NATIVE_PE_OFF)).toBeLessThanOrEqual(1e-12 * Math.abs(NATIVE_PE_OFF));
  });
});

describe('pair_style sw keywords and pair_coeff rules', () => {
  it('accepts threebody on|off and refuses other values', async () => {
    expect(await errorOf('sw threebody maybe', 'pair_coeff * * CdTeZnSeHgS0.sw Cd Te Zn')).toMatch(/'maybe'/);
  });

  it('refuses pair_coeff with threebody on unless the first two arguments are * *', async () => {
    expect(await errorOf('sw', 'pair_coeff 1 * CdTeZnSeHgS0.sw Cd Te Zn')).toMatch(/Incorrect args/);
  });

  it('refuses an unknown element name', async () => {
    expect(await errorOf('sw', 'pair_coeff * * CdTeZnSeHgS0.sw Cd Te Hg2')).toMatch(/Hg2/);
  });
});
