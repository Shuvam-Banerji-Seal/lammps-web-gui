import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { SharedThreadsBackend } from '../src/engine/cpu/pairThreads';
import { PairThreadWorker, type PairThreadMessage, type PairWorkerLike } from '../src/engine/cpu/pairThreadsCore';
import { THREADED_PAIRS } from '../src/engine/cpu/threadedPairFactories';
import { THREADED as GPL_THREADED } from '../src/engine-gpl/registry/threaded_pairs';

/*
 * pair reaxff on the engine thread with its bond-order loop on the pair threads (Pair.threadKernel 'bop',
 * PairCompute.kernels): the threads compute the uncorrected sigma, pi and pi-pi bond orders of every far pair within the
 * bond cutoff (core.ts bondOrderTerms) into shared arrays, and BOp reads them. They are the values the engine thread
 * computes, so a run on 4 threads gives the one-thread run's thermo and positions bit for bit. The workers run
 * in-process here and build the style from the GPL registry's factories.
 */

class InProcessWorker implements PairWorkerLike {
  onmessage: ((ev: MessageEvent) => void) | null = null;
  readonly worker = new PairThreadWorker({ ...THREADED_PAIRS, ...GPL_THREADED });
  kernels = 0;
  postMessage(msg: unknown): void {
    const m = msg as PairThreadMessage;
    if (m.type === 'kernel') this.kernels++;
    this.worker.handle(m);
  }
  terminate(): void {}
}

const CASES = join(__dirname, 'oracle');

// the 384-atom TATB cell of examples/reaxff with fix qeq/reaxff (w55reax_in_tatb), 10 steps
const INPUT = readFileSync(join(CASES, 'w55reax_in_tatb.in'), 'utf8')
  .split('\n').filter((l) => !l.startsWith('#')).join('\n')
  .replace(/^run\s+\d+/m, 'run 10');

const run = async (threads: number) => {
  const workers: InProcessWorker[] = [];
  const backend = threads > 1 ? new SharedThreadsBackend(threads, () => { const w = new InProcessWorker(); workers.push(w); return w; }) : undefined;
  if (backend) backend.pairThreads.minAtoms = 1;
  const rows: ThermoRow[] = [], logs: string[] = [];
  const s = new Session({ emit: (e: EngineEvent) => { if (e.kind === 'thermo') rows.push(e.row); else if (e.kind === 'log') logs.push(e.text); }, writeFile: () => {} }, backend);
  for (const f of ['w55reax_tatb.data', 'w55reax_tatb.control', 'w55reax_ffield.reax']) s.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  await s.execute(INPUT);
  const sys = (s as unknown as { sys: { state: { n: number; x: Float64Array; f: Float64Array } } }).sys;
  const out = { rows, loop: logs.filter((l) => l.startsWith('Loop time')), x: Array.from(sys.state.x.subarray(0, 3 * sys.state.n)), f: Array.from(sys.state.f.subarray(0, 3 * sys.state.n)),
    kernels: workers.reduce((n, w) => n + w.kernels, 0), kernelCalls: backend ? backend.pairThreads.kernelCalls : 0 };
  backend?.dispose();
  return out;
};

describe('pair reaxff: the bond-order loop on the pair threads', () => {
  it('a 4-thread run of the TATB cell gives the one-thread run bit for bit', async () => {
    const serial = await run(1);
    const threaded = await run(4);
    // the threads computed the bond orders of most evaluations (the first may come before the workers have the style)
    expect(threaded.kernelCalls).toBeGreaterThan(5);
    expect(threaded.kernels).toBe(3 * threaded.kernelCalls);
    expect(threaded.rows.length).toBe(serial.rows.length);
    expect(threaded.rows).toEqual(serial.rows);
    expect(threaded.x).toEqual(serial.x);
    expect(threaded.f).toEqual(serial.f);
    // the run says where its work went
    expect(threaded.loop.at(-1)).toContain('pair style reaxff runs on the engine thread; some of its loops ran on the 4 threads');
    expect(serial.loop.at(-1)).not.toContain('threads');
  });
});
