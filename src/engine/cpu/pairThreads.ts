import type { Pair, PairCompute } from '../force/types';
import type { NeighList } from '../neighbor';
import { THREADED_PAIRS } from './threadedPairs';
import { CpuForceBackend } from './forces';
import {
  ACC_LEN, DONE, FAILED, cloneableFields, pairFromFields, restrictCounts, sameShape, splitRanges, type PairThreadMessage, type PairWorkerLike,
} from './pairThreadsCore';

/*
 * Shared-memory threads for the pair term of the general engine. The engine thread copies the
 * owned+ghost coordinates, types and charges into SharedArrayBuffers every force evaluation and the
 * half neighbor list after every rebuild; each force worker (src/workers/pair.worker.ts) holds a
 * copy of the pair style, made from the engine's object when its version changes, and runs it on
 * its own range of owned atoms into its own force array. The engine thread runs the first range
 * itself, waits on an Atomics counter, then adds the workers' forces, energies and virials. Every
 * other part of the step (neighbor lists, bonded terms, kspace, fixes) stays on the engine thread,
 * so any input whose pair style is in THREADED_PAIRS gets the threads.
 *
 * This needs SharedArrayBuffer (a cross-origin isolated page, see src/coi.ts) and Atomics.wait,
 * which browsers allow in workers only: the engine runs in a worker. The pair forces equal the
 * single-thread ones up to the order of floating-point additions.
 */

/** Below this many owned atoms the threads cost more than they save. */
export const MIN_THREADED_ATOMS = 2000;

const shared = (bytes: number): SharedArrayBuffer => new SharedArrayBuffer(Math.max(8, bytes));

/**
 * Shared-memory threads are possible here: a cross-origin isolated context with SharedArrayBuffer,
 * nested workers, and a worker global scope (browsers forbid Atomics.wait on the main thread).
 */
export const sharedThreadsAvailable = (): boolean =>
  typeof SharedArrayBuffer !== 'undefined' && typeof Atomics !== 'undefined' && typeof Worker !== 'undefined'
  && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true
  && typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined';

export const spawnPairWorker = (): PairWorkerLike =>
  new Worker(new URL('../../workers/pair.worker.ts', import.meta.url), { type: 'module' });

/** The engine side: owns the workers and the shared buffers. */
export class SharedPairThreads {
  /** Force evaluations that ran on the threads. */
  calls = 0;
  private workers: PairWorkerLike[] = [];
  private readonly ctlBuf = shared(16);
  private readonly ctl = new Int32Array(this.ctlBuf);
  private pairRef: Pair | null = null;
  private pairVersion = -1;
  private pairOk = false;
  private listRef: NeighList | null = null;
  private listArrays: Int32Array | null = null;
  private listGen = 0;
  private ranges: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private own: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private cap = 0;
  private capL = 0;
  private capN = 0;
  private x: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private t: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private q: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private numneigh: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private firstneigh: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private neighbors: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private fs: Float64Array<ArrayBufferLike>[] = [];
  private accs: Float64Array<ArrayBufferLike>[] = [];

  constructor(readonly threads: number, spawn: () => PairWorkerLike = spawnPairWorker, public minAtoms = MIN_THREADED_ATOMS) {
    for (let k = 0; k < threads - 1; k++) {
      const w = spawn();
      w.postMessage({ type: 'init', ctl: this.ctlBuf, index: k } satisfies PairThreadMessage);
      this.workers.push(w);
    }
  }

  /** Runs pair.compute(pc) on the threads; false (nothing done) when this style or call cannot use them. */
  run(pair: Pair, pc: PairCompute): boolean {
    const W = this.workers.length;
    const list = pc.half;
    if (W === 0 || !list || pc.full || pc.eatom || pc.vatom || pair.needsFull || !pair.needsHalf) return false;
    if (!THREADED_PAIRS[pair.name] || list.inum < this.minAtoms) return false;
    if (pair !== this.pairRef || pair.version !== this.pairVersion) this.sendPair(pair);
    if (!this.pairOk) return false;
    const nall = pc.nall, nlocal = list.inum;
    let resend = false;
    if (nall > this.cap) { this.cap = Math.ceil(nall * 1.25); resend = true; }
    if (list !== this.listRef || list.neighbors !== this.listArrays) {
      const nnb = list.inum > 0 ? list.firstneigh[nlocal - 1] + list.numneigh[nlocal - 1] : 0;
      if (nlocal > this.capL) { this.capL = Math.ceil(nlocal * 1.25); resend = true; }
      if (nnb > this.capN) { this.capN = Math.ceil(nnb * 1.25); resend = true; }
      if (resend) this.allocate();
      this.numneigh.set(list.numneigh.subarray(0, nlocal));
      this.firstneigh.set(list.firstneigh.subarray(0, nlocal));
      this.neighbors.set(list.neighbors.subarray(0, nnb));
      this.ranges = splitRanges(list.numneigh, nlocal, W + 1);
      if (this.own.length < nlocal) this.own = new Int32Array(this.capL);
      restrictCounts(list.numneigh, nlocal, this.ranges[0], this.ranges[1], this.own);
      this.listRef = list;
      this.listArrays = list.neighbors;
      this.listGen++;
    } else if (resend) {
      this.allocate();
      this.copyList(list);
    }
    this.x.set(pc.x.subarray(0, 3 * nall));
    this.t.set(pc.type.subarray(0, nall));
    this.q.set(pc.q.subarray(0, nall));
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    const specialLJ = Array.from(pc.specialLJ), specialCoul = Array.from(pc.specialCoul);
    for (let w = 0; w < W; w++) {
      this.workers[w].postMessage({
        type: 'compute', nlocal, nall, qqrd2e: pc.qqrd2e, specialLJ, specialCoul, listGen: this.listGen,
        i0: this.ranges[w + 1], i1: this.ranges[w + 2],
      } satisfies PairThreadMessage);
    }
    // the engine thread's own range, straight into the real arrays
    pair.compute({ ...pc, half: { inum: nlocal, numneigh: this.own, firstneigh: list.firstneigh, neighbors: list.neighbors } });
    for (;;) {
      const done = Atomics.load(this.ctl, DONE);
      if (done >= W) break;
      if (Atomics.wait(this.ctl, DONE, done, 60_000) === 'timed-out') throw new Error('pair force threads did not answer within 60 s');
    }
    if (Atomics.load(this.ctl, FAILED) > 0) throw new Error(`pair style ${pair.name} failed in a force thread`);
    const f = pc.f, acc = pc.acc;
    for (let w = 0; w < W; w++) {
      const fw = this.fs[w];
      for (let k = 0; k < 3 * nall; k++) f[k] += fw[k];
      const a = this.accs[w];
      acc.evdwl += a[0];
      acc.ecoul += a[1];
      for (let c = 0; c < 6; c++) acc.virial[c] += a[2 + c];
    }
    this.calls++;
    return true;
  }

  private sendPair(pair: Pair): void {
    this.pairRef = pair;
    this.pairVersion = pair.version;
    const fields = cloneableFields(pair);
    // the worker must be able to rebuild exactly this style (same classes for every member that is
    // not a cache it rebuilds itself); otherwise the style runs on the engine thread
    const copy = pairFromFields(pair.name, structuredClone(fields));
    this.pairOk = copy !== null && sameShape(pair, copy);
    if (this.pairOk) for (const w of this.workers) w.postMessage({ type: 'pair', name: pair.name, fields } satisfies PairThreadMessage);
  }

  private copyList(list: NeighList): void {
    const nlocal = list.inum;
    const nnb = nlocal > 0 ? list.firstneigh[nlocal - 1] + list.numneigh[nlocal - 1] : 0;
    this.numneigh.set(list.numneigh.subarray(0, nlocal));
    this.firstneigh.set(list.firstneigh.subarray(0, nlocal));
    this.neighbors.set(list.neighbors.subarray(0, nnb));
  }

  /** New shared buffers at the current capacities, announced to every worker. */
  private allocate(): void {
    const xb = shared(24 * this.cap), tb = shared(4 * this.cap), qb = shared(8 * this.cap);
    const nb = shared(4 * this.capL), fb = shared(4 * this.capL), lb = shared(4 * this.capN);
    this.x = new Float64Array(xb); this.t = new Int32Array(tb); this.q = new Float64Array(qb);
    this.numneigh = new Int32Array(nb); this.firstneigh = new Int32Array(fb); this.neighbors = new Int32Array(lb);
    this.fs = []; this.accs = [];
    this.workers.forEach((w) => {
      const f = shared(24 * this.cap), acc = shared(8 * ACC_LEN);
      this.fs.push(new Float64Array(f));
      this.accs.push(new Float64Array(acc));
      w.postMessage({ type: 'buffers', x: xb, t: tb, q: qb, numneigh: nb, firstneigh: fb, neighbors: lb, f, acc } satisfies PairThreadMessage);
    });
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

/**
 * The general fp64 engine with its pair term on shared-memory threads. It is a CpuForceBackend, so
 * runs take the general engine path (run/accel.ts) with every style and fix; the session hands
 * pairThreads to the force field.
 */
export class SharedThreadsBackend extends CpuForceBackend {
  readonly label: string;
  readonly pairThreads: SharedPairThreads;

  constructor(threads: number, spawn?: () => PairWorkerLike) {
    super();
    this.pairThreads = new SharedPairThreads(threads, spawn);
    this.label = `CPU · fp64 · ${threads} threads (shared memory)`;
  }

  dispose(): void {
    this.pairThreads.dispose();
    super.dispose();
  }
}
