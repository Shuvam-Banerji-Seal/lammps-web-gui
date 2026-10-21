import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

const BASE = `units lj
lattice sc 0.5
region box block 0 4 0 4 0 4
create_box 1 box
`;

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(BASE + text);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  return err?.message ?? error;
};

// change_box.html: "The volume style can only be used following a keyword that changed the volume, which is any of
// the x, y, z keywords." Measured with native LAMMPS (b332f28): each of these stops with this text
// (src/change_box.cpp: volume not right after an x, y or z operation, on the operation's own dimension, or a third one).
describe('change_box volume misuse (native text)', () => {
  for (const cmd of [
    'change_box all y volume',
    'change_box all x scale 1.1 x volume',
    'change_box all x scale 1.1 y volume z volume y volume',
    'change_box all boundary p p p y volume',
  ]) {
    it(cmd, async () => {
      expect(await run(cmd)).toContain('Change_box volume used incorrectly');
    });
  }
  it('x scale with y and z volume keeps the volume', async () => {
    expect(await run('change_box all x scale 1.1 y volume z volume units box\nvariable v equal vol\nprint "${v}"')).toBeNull();
  });
});
