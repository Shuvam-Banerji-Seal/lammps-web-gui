/**
 * fix external (src/engine/fix/external.ts, the library-interface fix; the external-force fixes are in
 * engineFixExternal.test.ts): the argument checks and the init error of a pf/callback fix without a
 * library caller, with the messages native LAMMPS b332f28 gives (src/fix_external.cpp). The pf/array run is the oracle
 * case fix_external_array.
 */
import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

const errorOf = async (fixLine: string): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e: EngineEvent) => events.push(e), writeFile: () => {} } as never);
  const script = ['units lj', 'atom_style atomic', 'lattice fcc 0.8442', 'region box block 0 3 0 3 0 3', 'create_box 1 box',
    'create_atoms 1 box', 'mass 1 1.0', 'pair_style lj/cut 2.5', 'pair_coeff 1 1 1.0 1.0 2.5', fixLine, 'run 0'].join('\n');
  try {
    await session.execute(script);
  } catch (e) {
    const err = events.find((ev) => ev.kind === 'error');
    return (err && 'message' in err ? String(err.message) : String(e)).replace(/^line \d+: /, '');
  }
  const err = events.find((ev) => ev.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('fix external (messages measured with native LAMMPS)', () => {
  it.each([
    ['fix e all external', 'Illegal fix external command: missing argument(s)'],
    ['fix e all external pf/callback 1', 'Incorrect number of args for fix external pf/callback'],
    ['fix e all external pf/callback x 1', "Expected integer parameter instead of 'x' in input script or data file"],
    ['fix e all external pf/callback 0 1', 'Illegal value for ncall in fix external pf/callback'],
    ['fix e all external pf/callback 1 0', 'Illegal value for napply in fix external pf/callback'],
    ['fix e all external pf/array', 'Incorrect number of args for fix external pf/array'],
    ['fix e all external pf/array 0', 'Illegal value for napply (0) in fix external pf/callback'],
    ['fix e all external pf/array 1 2', 'Incorrect number of args for fix external pf/array'],
    ['fix e all external bogus 1', 'Unknown fix external mode bogus'],
    // no library caller registers a callback, so a run stops at init
    ['fix e all external pf/callback 1 1', 'Fix external callback function not set'],
    ['fix e all external pf/array 1', ''],
  ])('%s', async (line, msg) => {
    expect((await errorOf(line)).split('\n')[0]).toBe(msg);
  });
});
