import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix store/state history and compute property/atom history[I][J] (fix/store_state.ts, compute/peratom.ts): the
 * argument checks. Measured with native LAMMPS: each input below stops with the message in the table (the values are
 * stored and read as in the oracle cases w269ss_history_cont and w269ss_history_nfreq).
 */

const BASE = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
`;

const H = 'fix h all store/state 0 vx history 5 4 0';

const CASES: [string, string][] = [
  ['fix h all store/state 5 vx history 5 4 0', 'Fix store/state N = 0 required with history keyword'],
  ['fix h all store/state 0 vx history 0 4 0', 'Invalid fix store/state history Nevery 0'],
  ['fix h all store/state 0 vx history 5 0 0', 'Invalid fix store/state history Nrepeat 0'],
  ['fix h all store/state 0 vx history 5 4 7', 'Invalid fix store/state history Nfreq 7'],
  ['fix h all store/state 0 vx history 5 4', 'Illegal fix store/state history command: missing argument(s)'],
  [`${H}\ncompute c all property/atom history[1][1]`, 'Compute property/atom history attribute requires history option'],
  [`${H}\ncompute c all property/atom vx history h`, 'Compute property/atom history option cannot be used without history attribute'],
  [`${H}\ncompute c all property/atom history[5][1] history h`, 'Compute property/atom history references invalid history frame 5 from fix store/state'],
  [`${H}\ncompute c all property/atom history[1][2] history h`, 'Compute property/atom history references invalid attribute 2 from fix store/state'],
  ['fix h all store/state 0 vx\ncompute c all property/atom history[1][1] history h', 'Compute property/atom history fix does not store history'],
  [`${H}\ncompute c all property/atom history[1][1] history nope`, 'Could not find compute property/atom history fix ID: nope'],
  [`${H}\ncompute c all property/atom history[1][1] history h\ndump d all custom 3 x.dump id c_c\nrun 3`, 'Compute property/atom is accessing stored history on an incompatible timestep'],
  [`${H}\ncompute c all property/atom history[0][1] history h`, 'Compute property/atom history references invalid history frame 0 from fix store/state'],
  [`${H}\ncompute c all property/atom history[a][1] history h`, 'Inconsistent request for history keyword'],
];

const errorOf = async (text: string): Promise<string> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(text);
  } catch (e) {
    return String((e as Error).message);
  }
  const err = events.find((e) => e.kind === 'error');
  return err && 'message' in err ? String(err.message) : '';
};

describe('fix store/state history argument checks', () => {
  it.each(CASES)('%s', async (tail, message) => {
    expect(await errorOf(BASE + tail)).toContain(message);
  });
});
