import { describe, expect, it } from 'vitest';
import { EngineClient } from '../src/engine/client';
import type { EngineEvent } from '../src/engine/types';

const SMALL = `units lj
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.0 1234
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
fix 1 all nve
thermo 10`;

describe('EngineClient (main-thread fallback, same protocol as the worker)', () => {
  it('runs cells in one persistent session and streams events and files', async () => {
    const client = new EngineClient(false);
    expect(client.onMainThread).toBe(true);
    const ready = await client.reset('cpu', 0);
    expect(ready.backend).toBe('CPU · fp64');

    const ev1: EngineEvent[] = [];
    const r1 = await client.exec(SMALL, 1, { onEvent: (e) => ev1.push(e) });
    expect(r1).toEqual({ ok: true, cancelled: false });
    expect(ev1.some((e) => e.kind === 'log' && e.text === 'Created 108 atoms')).toBe(true);

    const ev2: EngineEvent[] = [];
    const files: Record<string, string> = {};
    const r2 = await client.exec('run 20\nwrite_data out.data', 12, {
      onEvent: (e) => ev2.push(e),
      onFile: (n, t) => { files[n] = t; },
    });
    expect(r2.ok).toBe(true);
    expect(ev2.filter((e) => e.kind === 'thermo').map((e) => (e as { row: { step?: number } }).row.step)).toEqual([0, 10, 20]);
    expect(files['out.data']).toMatch(/^108 atoms$/m);
    client.dispose();
  });

  it('reports errors with the cell line offset and recovers', async () => {
    const client = new EngineClient(false);
    await client.reset('cpu', 0);
    const ev: EngineEvent[] = [];
    const r = await client.exec('units lj\nbogus_command 1', 40, { onEvent: (e) => ev.push(e) });
    expect(r).toEqual({ ok: false, cancelled: false });
    const err = ev.find((e) => e.kind === 'error') as Extract<EngineEvent, { kind: 'error' }>;
    expect(err.line).toBe(41);
    expect(err.command).toBe('bogus_command');
    const again = await client.exec('units real', 50, { onEvent: () => {} });
    expect(again.ok).toBe(true);
  });

  it('cancels a long run and keeps the session usable', async () => {
    const client = new EngineClient(false);
    await client.reset('cpu', 0);
    await client.exec(SMALL, 1, { onEvent: () => {} });
    const p = client.exec('run 1000000', 20, { onEvent: () => {} });
    setTimeout(() => client.cancel(), 50);
    const r = await p;
    expect(r).toEqual({ ok: false, cancelled: true });
    const ev: EngineEvent[] = [];
    const after = await client.exec('run 5', 30, { onEvent: (e) => ev.push(e) });
    expect(after.ok).toBe(true);
    expect(ev.some((e) => e.kind === 'done')).toBe(true);
  });

  it('a webgpu request without navigator.gpu (jsdom) falls back to the CPU and says so', async () => {
    const client = new EngineClient(false);
    const ready = await client.reset('webgpu', 0);
    expect(ready.backend).toBe('CPU · fp64');
    expect(ready.note).toMatch(/no WebGPU adapter/);
  });
});
