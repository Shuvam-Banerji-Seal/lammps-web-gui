import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Fix state in the browser restart file (output/restart.ts, fix/restartRecord.ts). Parity with native LAMMPS is in the
 * oracle cases w176rs_* (tests/oracle). Here: the refusal of fix styles whose native class keeps state the engine does
 * not store yet, the record that a fix with the same ID and style takes back, and the record that another style leaves.
 * Native (src/modify.cpp Modify::add_fix at b332f28): the global list matches the ID and the style (suffix stripped);
 * the per-atom list matches the ID and the style.
 */

const newSession = (files = new Map<string, string>()) => {
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  return { session, events, files };
};

const BOX = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.2 4321 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
timestep        0.002
fix             1 all nve
`;

const AFTER_READ = 'pair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5\nfix 1 all nve\n';
const MSST = 'fix m all msst z 1.5 q 10.0 mu 0.5 tscale 0.1';

/** Runs a script; returns the error it stopped with (thrown or emitted), or '' when it ran. */
const errorOf = async (session: Session, events: EngineEvent[], text: string): Promise<string> => {
  let error = '';
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e instanceof Error ? e.message : e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  return error;
};

describe('fix state in write_restart', () => {
  it('writes a marker for a fix whose state is not stored, with one warning naming it', async () => {
    const { session, events, files } = newSession();
    const error = await errorOf(session, events, `${BOX}fix t all store/state 0 x\nrun 1\nwrite_restart t.restart\n`);
    expect(error).toBe('');
    expect(files.get('t.restart')).toContain('"stateNotStored":true');
    expect(JSON.stringify(events)).toContain('state of fix t (store/state) is not stored in the browser restart file');
  });

  it('refuses, by name, a fix with the same ID and style defined after read_restart', async () => {
    const { session, events } = newSession();
    const error = await errorOf(session, events, `${BOX}fix t all store/state 0 x\nrun 1\nwrite_restart t.restart\nclear\nread_restart t.restart\n${AFTER_READ}fix t all store/state 0 x\n`);
    expect(error).toContain('fix t (store/state): native LAMMPS would restore its saved state');
  });

  it('accepts another style or another ID with the same name after read_restart (as native does)', async () => {
    const { session, events } = newSession();
    const error = await errorOf(session, events, `${BOX}fix t all store/state 0 x\nrun 1\nwrite_restart t.restart\nclear\nread_restart t.restart\n${AFTER_READ}fix t all nvt temp 1.0 1.0 0.5\nfix u all store/state 0 x\n`);
    expect(error).toBe('');
  });

  it('stores the msst state and restores it when the fix is re-specified with the same ID and style', async () => {
    const { session, files } = newSession();
    await session.execute(`${BOX}${MSST}\nrun 5\nwrite_restart m.restart\n`);
    expect(files.get('m.restart')).toContain('"style":"msst"');
    const written = (session.sys.fixes.find((f) => f.id === 'm') as unknown as { restartState(): unknown }).restartState();

    await session.execute(`clear\nread_restart m.restart\n${AFTER_READ}${MSST}\n`);
    expect(session.sys.pendingFixData.has('m')).toBe(false);
    const restored = (session.sys.fixes.find((f) => f.id === 'm') as unknown as { restartState(): unknown }).restartState();
    expect(restored).toEqual(written);
  });

  it('leaves a record of another style in place (native matches the style too)', async () => {
    const { session } = newSession();
    await session.execute(`${BOX}${MSST}\nrun 2\nwrite_restart m.restart\nclear\nread_restart m.restart\n${AFTER_READ}fix m all nvt temp 1.0 1.0 0.5\n`);
    expect(session.sys.pendingFixData.has('m')).toBe(true);
  });
  it.each([
    ['temp/csvr', 'fix t all temp/csvr 1.0 1.0 0.5 77', 't'],
    ['gcmc', 'fix t all gcmc 1 2 0 1 77 2.0 -1.25 0.5', 't'],
    ['temp/csld', 'fix t all temp/csld 1.0 1.0 0.5 77', 't'],
    ['gjf', 'fix t all gjf 1.0 1.0 0.5 77', 't'],
    ['brownian', 'fix t all brownian 1.0 77 gamma_t 1.0 rng uniform', 't'],
    ['baoab', 'fix t all baoab 1.0 1.0 0.5 77', 't'],
  ])('stores the generator and counters of %s and restores them when re-specified', async (style, def, id) => {
    const { session, files } = newSession();
    await session.execute(`${BOX}${def}\nrun 4\nwrite_restart r.restart\n`);
    expect(files.get('r.restart')).toContain(`"style":"${style}"`);
    const written = (session.sys.fixes.find((f) => f.id === id) as unknown as { restartState(): unknown }).restartState();
    await session.execute(`clear\nread_restart r.restart\n${AFTER_READ}${def}\n`);
    expect(session.sys.pendingFixData.has(id)).toBe(false);
    const restored = (session.sys.fixes.find((f) => f.id === id) as unknown as { restartState(): unknown }).restartState();
    expect(restored).toEqual(written);
  });
});
