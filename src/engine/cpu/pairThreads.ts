import type { Pair, PairCompute } from '../force/types';
import type { NeighList } from '../neighbor';
import { THREADED_PAIRS } from './threadedPairs';
import { CpuForceBackend } from './forces';
import {
  ACC_LEN, CHUNKS_PER_THREAD, CTL_WORDS, DONE, DONE2, FAILED, NEXT, NEXT2, NOT_DONE, PHASE, READY, cloneableFields, sumForces, pairFromFields, sameShape, splitRanges, type PairThreadMessage, type PairWorkerLike,
} from './pairThreadsCore';
import { binCandidates, halfBinsPairs, type HalfBins, type NeighborThreads } from './neighborKernel';

/*
 * Shared-memory threads for the pair term of the general engine. The engine thread copies the
 * owned+ghost coordinates, types and charges into SharedArrayBuffers every force evaluation and the
 * half neighbor list after every rebuild; each force worker (src/workers/pair.worker.ts) holds a
 * copy of the pair style, made from the engine's object when its version changes. The half list is
 * cut into chunks of about equal work (CHUNKS_PER_THREAD per thread); every thread, the engine thread
 * included, claims chunks from a shared Atomics counter until none are left, each into its own force
 * array, so slow threads take fewer chunks. The engine thread then waits on an Atomics counter and
 * adds the workers' forces, energies and virials. Every
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

/** Shared copies of a neighbor search's binned atoms and chunk table, and each worker's entry arrays. */
interface NeighborShared {
  bL: Int32Array; bG: Int32Array; rLo: Int32Array; rHi: Int32Array;
  lk: Int32Array; lt: Int32Array; lxs: Float64Array; gk: Int32Array; gt: Int32Array; gxs: Float64Array; cutsq: Float64Array;
  chunkBins: Int32Array; chunkOut: Int32Array;
  pa: Int32Array[]; pj: Int32Array[];
  /** capacities: bins, rows, owned, ghost atoms, type pairs, chunks, entries per worker */
  caps: number[];
}

/** The engine side: owns the workers and the shared buffers. */
export class SharedPairThreads implements NeighborThreads {
  /** Force evaluations that ran on the threads. */
  calls = 0;
  private workers: PairWorkerLike[] = [];
  private readonly ctlBuf = shared(4 * CTL_WORDS);
  private readonly ctl = new Int32Array(this.ctlBuf);
  private pairRef: Pair | null = null;
  private pairVersion = -1;
  private pairOk = false;
  private listRef: NeighList | null = null;
  private listArrays: Int32Array | null = null;
  /** Chunk boundaries of the current half list (nchunks + 1 entries), shared with the workers. */
  private readonly chunksBuf: SharedArrayBuffer;
  private readonly chunks: Int32Array;
  private nchunks = 0;
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

  /**
   * The threads sum their forces into a shared force array in parallel (a second phase in which every thread blocks):
   * real workers only, as an in-process stand-in (tests) handles a message at once and could not wait for the others.
   */
  private readonly parallelSum: boolean;

  constructor(readonly threads: number, spawn: () => PairWorkerLike = spawnPairWorker, public minAtoms = MIN_THREADED_ATOMS) {
    this.parallelSum = spawn === spawnPairWorker;
    this.chunksBuf = shared(4 * (CHUNKS_PER_THREAD * Math.max(1, threads) + 1));
    this.chunks = new Int32Array(this.chunksBuf);
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
    // until every worker has started (READY, see pairThreadsCore.ts) the pair term runs on the engine thread: blocking
    // in Atomics.wait on a worker that has not started yet would deadlock
    if (Atomics.load(this.ctl, READY) < W) return false;
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
      this.splitChunks(list, nlocal, W + 1);
      this.listRef = list;
      this.listArrays = list.neighbors;
    } else if (resend) {
      this.allocate();
      this.copyList(list);
    }
    this.x.set(pc.x.subarray(0, 3 * nall));
    this.t.set(pc.type.subarray(0, nall));
    this.q.set(pc.q.subarray(0, nall));
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    Atomics.store(this.ctl, NEXT, 0);
    const nchunks = this.nchunks;
    const specialLJ = Array.from(pc.specialLJ), specialCoul = Array.from(pc.specialCoul);
    // a shared force array is summed by all the threads (sumForces), else by the engine thread below
    const F = this.parallelSum && pc.f.buffer instanceof SharedArrayBuffer && pc.f.byteOffset === 0 ? pc.f.buffer : null;
    const phase = Atomics.load(this.ctl, PHASE), nslices = CHUNKS_PER_THREAD * (W + 1);
    for (let w = 0; w < W; w++) {
      this.workers[w].postMessage({
        type: 'compute', nlocal, nall, qqrd2e: pc.qqrd2e, specialLJ, specialCoul, nchunks, F, phase, nslices,
      } satisfies PairThreadMessage);
    }
    // the engine thread claims chunks too, straight into the real arrays
    for (;;) {
      const c = Atomics.add(this.ctl, NEXT, 1);
      if (c >= nchunks) break;
      pair.compute({ ...pc, half: { ilo: this.chunks[c], inum: this.chunks[c + 1], numneigh: list.numneigh, firstneigh: list.firstneigh, neighbors: list.neighbors } });
    }
    for (;;) {
      const done = Atomics.load(this.ctl, DONE);
      if (done >= W) break;
      if (Atomics.wait(this.ctl, DONE, done, 60_000) === 'timed-out') throw new Error('pair force threads did not answer within 60 s');
    }
    const failed = Atomics.load(this.ctl, FAILED) > 0;
    const f = pc.f, acc = pc.acc;
    if (F) {
      // start the parallel sum (no slices after a failure, so the waiting workers return), take slices, wait
      Atomics.store(this.ctl, NEXT2, failed ? nslices : 0);
      Atomics.store(this.ctl, DONE2, 0);
      Atomics.add(this.ctl, PHASE, 1);
      Atomics.notify(this.ctl, PHASE);
      const len = Math.ceil((3 * nall) / nslices);
      for (;;) {
        const sl = Atomics.add(this.ctl, NEXT2, 1);
        if (sl >= nslices) break;
        sumForces(f, this.fs, sl * len, Math.min(3 * nall, (sl + 1) * len));
      }
      for (;;) {
        const done = Atomics.load(this.ctl, DONE2);
        if (done >= W) break;
        if (Atomics.wait(this.ctl, DONE2, done, 60_000) === 'timed-out') throw new Error('pair force threads did not answer within 60 s');
      }
    }
    if (failed) throw new Error(`pair style ${pair.name} failed in a force thread`);
    if (!F) sumForces(f, this.fs, 0, 3 * nall);
    for (let w = 0; w < W; w++) {
      const a = this.accs[w];
      acc.evdwl += a[0];
      acc.ecoul += a[1];
      for (let c = 0; c < 6; c++) acc.virial[c] += a[2 + c];
    }
    this.calls++;
    return true;
  }

  /** Cuts the half list into chunks of about equal work, written where the workers read them. */
  private splitChunks(list: NeighList, nlocal: number, threads: number): void {
    const k = Math.max(1, Math.min(nlocal, CHUNKS_PER_THREAD * threads));
    this.chunks.set(splitRanges(list.numneigh, nlocal, k));
    this.nchunks = k;
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
    const fbufs = this.workers.map(() => shared(24 * this.cap));
    this.fs = fbufs.map((b) => new Float64Array(b));
    this.accs = [];
    this.workers.forEach((w, k) => {
      const acc = shared(8 * ACC_LEN);
      this.accs.push(new Float64Array(acc));
      w.postMessage({ type: 'buffers', x: xb, t: tb, q: qb, numneigh: nb, firstneigh: fb, neighbors: lb, f: fbufs[k], fs: fbufs, acc, chunks: this.chunksBuf } satisfies PairThreadMessage);
    });
  }

  /** Neighbor searches that ran on the threads. */
  neighborCalls = 0;
  private nshared: NeighborShared | null = null;
  /** Entries of the last threaded search (sizes the workers' entry arrays). */
  private lastEntries = 0;
  /** The engine thread's own entries of a threaded search (grown when needed). */
  private ownPa = new Int32Array(0);
  private ownPj = new Int32Array(0);

  /**
   * The pair search of a half list (cpu/neighborKernel.ts) on the threads: the bins are cut into chunks of about equal
   * work (candidate pairs), every thread claims chunks from the shared counter as in run(), and the entries are joined
   * in chunk order, which is the order of the search on one thread.
   */
  halfList(B: HalfBins, nbins: number, nlocal: number, pa: Int32Array<ArrayBuffer>, pj: Int32Array<ArrayBuffer>): { pa: Int32Array<ArrayBuffer>; pj: Int32Array<ArrayBuffer>; np: number } | null {
    const W = this.workers.length;
    if (W === 0 || nlocal < this.minAtoms || Atomics.load(this.ctl, READY) < W) return null;
    // chunks of about equal candidate counts
    const cand = new Float64Array(nbins + 1);
    let maxBin = 0;
    for (let b = 0; b < nbins; b++) {
      const c = binCandidates(B, b);
      if (c > maxBin) maxBin = c;
      cand[b + 1] = cand[b] + c;
    }
    const total = cand[nbins];
    if (total === 0) return null;
    const K = Math.max(1, Math.min(nbins, CHUNKS_PER_THREAD * (W + 1)));
    const nghost = B.gk.length;
    // each worker's arrays hold twice its share of the entries of the last search (a fourth of the candidates at
    // first) and the largest bin's candidates, which are written before their test
    const share = Math.ceil((2 * (this.lastEntries > 0 ? this.lastEntries : total / 4)) / (W + 1)) + maxBin;
    const S = this.neighborShared(nbins + 1, B.nrows, nlocal, nghost, B.nt * B.nt, K + 1, share);
    S.bL.set(B.bL.subarray(0, nbins + 1)); S.bG.set(B.bG.subarray(0, nbins + 1));
    S.rLo.set(B.rLo.subarray(0, B.nrows)); S.rHi.set(B.rHi.subarray(0, B.nrows));
    S.lk.set(B.lk.subarray(0, nlocal)); S.lt.set(B.lt.subarray(0, nlocal)); S.lxs.set(B.lxs.subarray(0, 3 * nlocal));
    S.gk.set(B.gk.subarray(0, nghost)); S.gt.set(B.gt.subarray(0, nghost)); S.gxs.set(B.gxs.subarray(0, 3 * nghost));
    S.cutsq.set(B.cutsq);
    let k = 1;
    S.chunkBins[0] = 0;
    for (let b = 1; b < nbins && k < K; b++) if (cand[b] >= (total * k) / K) S.chunkBins[k++] = b;
    for (; k <= K; k++) S.chunkBins[k] = nbins;
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    Atomics.store(this.ctl, NEXT, 0);
    for (const w of this.workers) w.postMessage({ type: 'neigh', nrows: B.nrows, nt: B.nt, nchunks: K } satisfies PairThreadMessage);
    // the engine thread claims chunks too, into its own arrays
    const out = new Int32Array(2);
    let own = 0;
    const searchOwn = (c: number): void => {
      let b = S.chunkBins[c];
      const b1 = S.chunkBins[c + 1];
      const start = own;
      while (b < b1) {
        halfBinsPairs(B, b, b1, this.ownPa, this.ownPj, own, this.ownPa.length, out);
        b = out[0]; own = out[1];
        if (b < b1) {
          const cap = Math.max(2 * this.ownPa.length, own + binCandidates(B, b));
          const na = new Int32Array(cap); na.set(this.ownPa.subarray(0, own)); this.ownPa = na;
          const nj = new Int32Array(cap); nj.set(this.ownPj.subarray(0, own)); this.ownPj = nj;
        }
      }
      S.chunkOut[3 * c] = W;
      S.chunkOut[3 * c + 1] = start;
      S.chunkOut[3 * c + 2] = own - start;
    };
    for (;;) {
      const c = Atomics.add(this.ctl, NEXT, 1);
      if (c >= K) break;
      searchOwn(c);
    }
    for (;;) {
      const done = Atomics.load(this.ctl, DONE);
      if (done >= W) break;
      if (Atomics.wait(this.ctl, DONE, done, 60_000) === 'timed-out') throw new Error('neighbor threads did not answer within 60 s');
    }
    if (Atomics.load(this.ctl, FAILED) > 0) return null;
    // chunks no worker had room for: searched here, and the workers get more room next time
    let grow = false;
    for (let c = 0; c < K; c++) if (S.chunkOut[3 * c] === NOT_DONE) { searchOwn(c); grow = true; }
    let np = 0;
    for (let c = 0; c < K; c++) np += S.chunkOut[3 * c + 2];
    if (np > Math.min(pa.length, pj.length)) { pa = new Int32Array(Math.ceil(np * 1.25)); pj = new Int32Array(pa.length); }
    let e = 0;
    for (let c = 0; c < K; c++) {
      const t = S.chunkOut[3 * c], o = S.chunkOut[3 * c + 1], n = S.chunkOut[3 * c + 2];
      const srcA = t === W ? this.ownPa : S.pa[t], srcJ = t === W ? this.ownPj : S.pj[t];
      pa.set(srcA.subarray(o, o + n), e);
      pj.set(srcJ.subarray(o, o + n), e);
      e += n;
    }
    this.lastEntries = grow ? 2 * np : np;
    this.neighborCalls++;
    return { pa, pj, np };
  }

  /** The shared buffers of a neighbor search, reallocated (and announced to the workers) when they are too small. */
  private neighborShared(nb: number, nrows: number, nlocal: number, nghost: number, ntt: number, nchunk: number, share: number): NeighborShared {
    const want = [nb, nrows, nlocal, nghost, ntt, nchunk, share];
    const S = this.nshared;
    if (S && want.every((v, i) => v <= S.caps[i])) return S;
    const caps = want.map((v, i) => Math.max(S?.caps[i] ?? 0, Math.ceil(v * 1.25)));
    const [cb, cr, cl, cg, ct, cc, ce] = caps;
    const bufs = {
      bL: shared(4 * cb), bG: shared(4 * cb), rLo: shared(4 * cr), rHi: shared(4 * cr),
      lk: shared(4 * cl), lt: shared(4 * cl), lxs: shared(24 * cl), gk: shared(4 * cg), gt: shared(4 * cg), gxs: shared(24 * cg),
      cutsq: shared(8 * ct), chunkBins: shared(4 * cc), chunkOut: shared(12 * cc),
    };
    const pa: Int32Array[] = [], pj: Int32Array[] = [];
    this.workers.forEach((w) => {
      const a = shared(4 * ce), j = shared(4 * ce);
      pa.push(new Int32Array(a)); pj.push(new Int32Array(j));
      w.postMessage({ type: 'nbuffers', ...bufs, pa: a, pj: j } satisfies PairThreadMessage);
    });
    this.nshared = {
      bL: new Int32Array(bufs.bL), bG: new Int32Array(bufs.bG), rLo: new Int32Array(bufs.rLo), rHi: new Int32Array(bufs.rHi),
      lk: new Int32Array(bufs.lk), lt: new Int32Array(bufs.lt), lxs: new Float64Array(bufs.lxs),
      gk: new Int32Array(bufs.gk), gt: new Int32Array(bufs.gt), gxs: new Float64Array(bufs.gxs), cutsq: new Float64Array(bufs.cutsq),
      chunkBins: new Int32Array(bufs.chunkBins), chunkOut: new Int32Array(bufs.chunkOut), pa, pj, caps,
    };
    return this.nshared;
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
