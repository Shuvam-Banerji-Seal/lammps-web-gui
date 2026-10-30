import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { MIN_THREADED_ATOMS, SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';

/*
 * The resource monitor's report of where a run computes its forces (EngineEvent 'perf' path / threads / why, and the
 * "Loop time" line): on the pair threads, or on one thread with the reason. The workers run in-process, as in
 * enginePairThreads.test.ts.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker(THREADED_PAIRS);
  postMessage(msg: unknown): void { this.worker.handle(msg as PairThreadMessage); }
  terminate(): void {}
}

type Perf = Extract<EngineEvent, { kind: 'perf' }>;

const run = async (pair: string, opts: { threads?: number; minAtoms?: number; cells?: number } = {}) => {
  const backend = opts.threads ? new SharedThreadsBackend(opts.threads, () => new InProcessWorker()) : undefined;
  if (backend && opts.minAtoms !== undefined) backend.pairThreads.minAtoms = opts.minAtoms;
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} }, backend);
  const c = opts.cells ?? 5;
  await session.execute(`units lj
lattice fcc 0.8442
region box block 0 ${c} 0 ${c} 0 ${c}
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287 loop geom
${pair}
fix 1 all nve
run 20
`);
  backend?.dispose();
  const perf = events.filter((e): e is Perf => e.kind === 'perf');
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { last: perf[perf.length - 1], loop: logs.find((l) => l.startsWith('Loop time')) ?? '' };
};

const LJ = 'pair_style lj/cut 2.5\npair_coeff 1 1 1.0 1.0 2.5';

describe('where a run computes its forces', () => {
  it('a system below the threads threshold runs on one thread and says so', async () => {
    const { last, loop } = await run(LJ, { threads: 3 });
    expect(last).toMatchObject({ path: 'serial', threads: 1, atoms: 500 });
    expect(last.why).toBe(`500 atoms; the pair threads start at ${MIN_THREADED_ATOMS} atoms, below that they cost more than they save`);
    expect(loop).toContain('(CPU · fp64; one thread: 500 atoms; the pair threads start at');
  });

  it('a threaded style above the threshold runs on the pair threads', async () => {
    const { last, loop } = await run(LJ, { threads: 3, minAtoms: 1 });
    expect(last).toMatchObject({ path: 'threads', threads: 3, why: null });
    expect(loop).toContain('(CPU · fp64 · 3 threads (shared memory))');
  });

  it('a style without a threaded kernel names itself', async () => {
    const { last } = await run('pair_style soft 1.0\npair_coeff 1 1 10.0', { threads: 3, minAtoms: 1 });
    expect(last).toMatchObject({ path: 'serial', why: 'pair style soft has no threaded kernel; it runs on the engine thread' });
  });

  it('without shared memory (not cross-origin isolated) only plain lj/cut can use threads', async () => {
    const { last, loop } = await run(LJ);
    expect(last.path).toBe('serial');
    expect(last.why).toMatch(/^the page is not cross-origin isolated/);
    // the plain CPU backend has no threads to explain
    expect(loop).toMatch(/\(CPU · fp64\)$/);
  });
});
