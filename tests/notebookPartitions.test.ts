/**
 * Partitions in the notebook's engine protocol (reset { partitions }), as lammps -partition N: the host builds the
 * session with that many replicas, reports them in 'ready', and a world-style variable takes each replica's value.
 * Without partitions, neb is refused with the message the notebook turns into a "set Partitions" hint.
 */
import { describe, expect, it } from 'vitest';
import { EngineHost } from '../src/engine/host';
import type { FromEngine } from '../src/engine/protocol';

const run = async (partitions: number | undefined, text: string) => {
  const out: FromEngine[] = [];
  const host = new EngineHost((m) => out.push(m));
  await host.handle({ type: 'reset', backend: 'cpu', threads: 1, frameEvery: 0, partitions });
  await host.handle({ type: 'exec', id: 1, text, firstLine: 1 });
  const ready = out.find((m): m is Extract<FromEngine, { type: 'ready' }> => m.type === 'ready')!;
  const events = out.filter((m): m is Extract<FromEngine, { type: 'event' }> => m.type === 'event').map((m) => m.event);
  return { ready, events };
};

describe('notebook partitions', () => {
  it('reset with partitions builds that many replicas and reports them', async () => {
    const { ready, events } = await run(3, 'variable w world a b c\nprint "replica $w"\n');
    expect(ready.partitions).toBe(3);
    const logs = events.filter((e) => e.kind === 'log').map((e) => (e as { text: string }).text);
    expect(logs.some((t) => t.includes('replica a'))).toBe(true);
    expect(events.some((e) => e.kind === 'error')).toBe(false);
  });

  it('defaults to one partition, where a multi-replica command says it needs partitions', async () => {
    const { ready, events } = await run(undefined, 'variable w world a b c\n');
    expect(ready.partitions).toBe(1);
    const err = events.find((e) => e.kind === 'error') as { message: string } | undefined;
    expect(err?.message ?? '').toMatch(/partition|world/i);
  });
});
