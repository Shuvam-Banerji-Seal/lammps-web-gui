import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * The partition command (src/engine/session.ts partitionCommand): the argument checks with native's messages
 * (checked against native LAMMPS b332f28 on the same one-partition inputs) and the selection of the partitions.
 * Native parity of a two-partition run is tests/oracle/w248pt_partition.
 */

async function run(script: string, partitions = 1) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, undefined, 0, partitions);
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { error: err && 'message' in err ? String(err.message) : undefined, logs, session };
}

describe('partition command', () => {
  it('checks its arguments as native does', async () => {
    const cases: [string, RegExp][] = [
      ['partition maybe 1 print hi', /Expected boolean parameter instead of 'maybe' in input script or data file/],
      ['partition yes 3 print hi', /Numeric index 3 is out of bounds \(1-1\)/],
      ['partition yes 1 partition yes 1 print hi', /Illegal partition command argument/],
      ['partition yes 0 print hi', /Invalid range string: 0/],
      ['partition yes 1', /Illegal partition command: missing argument\(s\)/],
    ];
    for (const [line, msg] of cases) expect((await run(`${line}\n`)).error, line).toMatch(msg);
  });

  it('runs the command on the selected partitions only, quoted words kept', async () => {
    const r = await run('partition yes 2 variable who string "second partition"\npartition no 2 variable who string first\n', 2);
    expect(r.error).toBeUndefined();
    const sys = (r.session as unknown as { replicaSystems: { vars: { text(n: string): string } }[] }).replicaSystems;
    expect(sys[0].vars.text('who')).toBe('first');
    expect(sys[1].vars.text('who')).toBe('second partition');
  });
});
