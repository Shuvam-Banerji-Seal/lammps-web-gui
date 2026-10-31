import type { Pair, PairCompute } from '../force/types';
import type { NeighList } from '../neighbor';
import { THREADED_NAMES, isTwoPass } from './threadedPairs';
import { GPL_THREADED_PAIRS, loadGplThreadedPair } from '../lazyStyles';
import { CpuForceBackend } from './forces';
import {
  ACC_LEN, CHUNKS_PER_THREAD, CTL_WORDS, atomChunk, DONE, DONE2, FAILED, NEXT, NEXT2, NOT_DONE, MAX_PAIR_SLOTS, PHASE, READY, readyIndex, cloneableFields, sumForces, pairFromFactory, sameShape, splitRanges, type PairThreadMessage, type PairWorkerLike,
} from './pairThreadsCore';
import { halfBinsPairs, halfBinsPairsSpecial, type HalfBins, type ListSegment, type NeighborThreads, type SpecialBins } from './neighborKernel';
import { PPPM_PARAMS, pppmField, pppmWeights, weightScratch, writePppmParams, type PppmGrid } from '../force/kspace/pppmKernel';

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
 *
 * A two-pass style (threadedPairs.ts TwoPassPair, the EAM styles) runs its first loop the same way, each thread summing
 * per-atom values into its own array; the engine thread adds the arrays, turns the sums into the values the second loop
 * reads (shared with the workers), and the threads run the second loop as a plain pair style.
 */

/**
 * Below this many owned atoms the threads cost more than they save. Measured in Chromium 1243 (24 logical cores,
 * cross-origin isolated, LJ melt, atom-steps/s at 1/2/4/8 threads, threshold off): 108 atoms 525k/595k/620k,
 * 256 atoms 568k/793k/533k, 500 atoms 604k/926k/1.12M/1.11M, 1372 atoms 621k/1.03M/1.45M/1.74M, 2048 atoms
 * 637k/1.04M/1.55M/1.65M. From 500 atoms every thread count beats one thread (the neighbor search and the force sum
 * run on the threads too since #575/#576, which moved the break-even point down from about 2000 atoms).
 */
export const MIN_THREADED_ATOMS = 500;

/**
 * Why the last pair evaluation ran on the engine thread, for the resource monitor: the system is below minAtoms, the
 * pair style has no threaded kernel, per-atom energy or virial was asked for, the style needs a list the threads do not
 * take, or the workers were still starting. null when it ran on the threads.
 */
export type ThreadSkip = 'small' | 'style' | 'peratom' | 'list' | 'starting' | null;

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
  ltag: Int32Array; gtag: Int32Array;
  chunkBins: Int32Array; chunkOut: Int32Array;
  /** the special bonds of the owned atoms, their IDs and the owner of each owned or ghost atom */
  spOffset: Int32Array; spPartner: Int32Array; spOrder: Int8Array; id: Int32Array; owner: Int32Array;
  pa: Int32Array[]; pj: Int32Array[];
  /** capacities: bins, rows, owned, ghost atoms, type pairs, chunks, entries per worker, special partners, all atoms */
  caps: number[];
}

/** The engine side: owns the workers and the shared buffers. */
export class SharedPairThreads implements NeighborThreads {
  /** Force evaluations that ran on the threads. */
  calls = 0;
  /** Why the last call did not run on the threads (null when it did). */
  skip: ThreadSkip = null;
  private workers: PairWorkerLike[] = [];
  private readonly ctlBuf = shared(4 * CTL_WORDS);
  private readonly ctl = new Int32Array(this.ctlBuf);
  /** per pair slot (pairThreadsCore.ts MAX_PAIR_SLOTS): the style last sent, its version, and whether it can run here */
  private readonly slotRef: (Pair | null)[] = [];
  private readonly slotVersion: number[] = [];
  private readonly slotOk: boolean[] = [];
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
  /** The atom IDs of the owned and ghost atoms (Pair.threadTags), and the list they were filled for. */
  private tags: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private tagsFor: Int32Array | null = null;
  /** The module URL of each GPL-ported style's registry, once loaded (lazyStyles.ts loadGplThreadedPair). */
  private readonly gplUrls = new Map<string, string>();
  private fs: Float64Array<ArrayBufferLike>[] = [];
  private accs: Float64Array<ArrayBufferLike>[] = [];
  /** A two-pass style's per-atom sums (engine thread) and the values its second loop reads (shared). */
  private sums = new Float64Array(0);
  private auxBuf: SharedArrayBuffer = shared(8);
  private aux: Float64Array<ArrayBufferLike> = new Float64Array(0);

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
  run(pair: Pair, pc: PairCompute, slot = 0): boolean {
    const W = this.workers.length;
    // a half-list style, or a full-list one whose rows are independent (the three-body styles of THREADED_PAIRS)
    const fullStyle = pair.needsFull && !pair.needsHalf;
    const list = fullStyle ? pc.full : pc.half;
    if (pc.eatom || pc.vatom) { this.skip = 'peratom'; return false; }
    if (W === 0 || !list || (!fullStyle && (pc.full || pair.needsFull)) || !(pair.needsHalf || fullStyle)) { this.skip = 'list'; return false; }
    if (!THREADED_NAMES.has(pair.name)) { this.skip = 'style'; return false; }
    if (list.inum < this.minAtoms) { this.skip = 'small'; return false; }
    const two = isTwoPass(pair) ? pair : null;
    if (two && !two.twoPassReady()) { this.skip = 'list'; return false; }
    // until every worker has started (READY, see pairThreadsCore.ts) the pair term runs on the engine thread: blocking
    // in Atomics.wait on a worker that has not started yet would deadlock
    if (Atomics.load(this.ctl, READY) < W) { this.skip = 'starting'; return false; }
    if (slot >= MAX_PAIR_SLOTS) { this.skip = 'style'; return false; }
    if (pair !== this.slotRef[slot] || pair.version !== this.slotVersion[slot]) this.sendPair(pair, slot);
    if (!this.slotOk[slot]) { this.skip = 'style'; return false; }
    // until every worker has built the style (it may load its module first) the pair term runs on the engine thread
    if (Atomics.load(this.ctl, readyIndex(slot)) < W) { this.skip = 'starting'; return false; }
    this.skip = null;
    const nall = pc.nall, nlocal = list.inum;
    let resend = false;
    if (nall > this.cap) { this.cap = Math.ceil(nall * 1.25); resend = true; }
    // a style that reads its neighbors in another order (native order) gives the array the threads read; it is made
    // once per neighbor build (the style caches it), so the copy below happens on rebuilds only
    const ordered = pair.threadNeighbors ? pair.threadNeighbors(pc, list) : null;
    const nbArr = ordered ?? list.neighbors;
    if (list !== this.listRef || nbArr !== this.listArrays) {
      const nnb = list.inum > 0 ? list.firstneigh[nlocal - 1] + list.numneigh[nlocal - 1] : 0;
      if (nlocal > this.capL) { this.capL = Math.ceil(nlocal * 1.25); resend = true; }
      if (nnb > this.capN) { this.capN = Math.ceil(nnb * 1.25); resend = true; }
      if (resend) this.allocate();
      this.numneigh.set(list.numneigh.subarray(0, nlocal));
      this.firstneigh.set(list.firstneigh.subarray(0, nlocal));
      this.neighbors.set(nbArr.subarray(0, nnb));
      this.splitChunks(list, nlocal, W + 1);
      this.listRef = list;
      this.listArrays = nbArr;
      this.tagsFor = null;
    } else if (resend) {
      this.allocate();
      this.copyList(list, nbArr);
      this.tagsFor = null;
    }
    // the IDs of the owned and ghost atoms, for a style that reads them (they change with the lists only)
    if (pair.threadTags && this.tagsFor !== nbArr) {
      const id = pc.s.id, owner = pc.nb.owner;
      for (let k = 0; k < nall; k++) this.tags[k] = id[owner[k]];
      this.tagsFor = nbArr;
    }
    this.x.set(pc.x.subarray(0, 3 * nall));
    this.t.set(pc.type.subarray(0, nall));
    this.q.set(pc.q.subarray(0, nall));
    const specialLJ = Array.from(pc.specialLJ), specialCoul = Array.from(pc.specialCoul);
    // a shared force array is summed by all the threads (sumForces), else by the engine thread below
    const F = this.parallelSum && pc.f.buffer instanceof SharedArrayBuffer && pc.f.byteOffset === 0 ? pc.f.buffer : null;
    const phase = Atomics.load(this.ctl, PHASE), nslices = CHUNKS_PER_THREAD * (W + 1);
    const chunk = (c: number): PairCompute => ({
      ...pc,
      [fullStyle ? 'full' : 'half']: { ilo: this.chunks[c], inum: this.chunks[c + 1], numneigh: list.numneigh, firstneigh: list.firstneigh, neighbors: nbArr, nativeOrdered: ordered !== null },
    });
    if (two) {
      // the first loop: every thread sums into its own array, the engine thread adds them in worker order
      const sums = this.sums;
      sums.fill(0, 0, nall);
      const firstFailed = this.dispatch({ type: 'compute', full: fullStyle, nlocal, nall, qqrd2e: pc.qqrd2e, specialLJ, specialCoul, nchunks: this.nchunks, F: null, phase, nslices, pass: 1, aux: null, ordered: ordered !== null, tags: !!pair.threadTags, slot }, (c) => two.densities(chunk(c), sums));
      if (firstFailed) throw new Error(`pair style ${pair.name} failed in a force thread`);
      for (let w = 0; w < W; w++) {
        const a = this.fs[w];
        for (let k = 0; k < nall; k++) sums[k] += a[k];
      }
      two.embed(pc, sums, this.aux as Float64Array);
    }
    const failed = this.dispatch(
      { type: 'compute', full: fullStyle, nlocal, nall, qqrd2e: pc.qqrd2e, specialLJ, specialCoul, nchunks: this.nchunks, F, phase, nslices, pass: two ? 2 : 0, aux: two ? this.auxBuf : null, ordered: ordered !== null, tags: !!pair.threadTags, slot },
      two ? (c) => two.forces(chunk(c), this.aux as Float64Array) : (c) => pair.compute(chunk(c)),
    );
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

  /**
   * One loop over the half list on every thread: the workers get `msg`, the engine thread claims chunks too (`own`, into
   * the real arrays), then waits for the workers. True when a worker failed.
   */
  private dispatch(msg: Extract<PairThreadMessage, { type: 'compute' }>, own: (c: number) => void): boolean {
    const W = this.workers.length;
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    Atomics.store(this.ctl, NEXT, 0);
    for (const w of this.workers) w.postMessage(msg);
    for (;;) {
      const c = Atomics.add(this.ctl, NEXT, 1);
      if (c >= msg.nchunks) break;
      own(c);
    }
    for (;;) {
      const done = Atomics.load(this.ctl, DONE);
      if (done >= W) break;
      if (Atomics.wait(this.ctl, DONE, done, 60_000) === 'timed-out') throw new Error('pair force threads did not answer within 60 s');
    }
    return Atomics.load(this.ctl, FAILED) > 0;
  }

  /** Cuts the half list into chunks of about equal work, written where the workers read them. */
  private splitChunks(list: NeighList, nlocal: number, threads: number): void {
    const k = Math.max(1, Math.min(nlocal, CHUNKS_PER_THREAD * threads));
    this.chunks.set(splitRanges(list.numneigh, nlocal, k));
    this.nchunks = k;
  }

  private sendPair(pair: Pair, slot: number): void {
    this.slotRef[slot] = pair;
    this.slotVersion[slot] = pair.version;
    const fields = cloneableFields(pair);
    // the worker must be able to rebuild exactly this style (same classes for every member that is
    // not a cache it rebuilds itself); otherwise the style runs on the engine thread
    // the engine's own class rebuilt from the cloned fields (the worker builds its copy from threadedPairFactories.ts)
    const own = pair.constructor as new () => Pair;
    const copy = pairFromFactory(() => new own(), structuredClone(fields));
    this.slotOk[slot] = copy !== null && sameShape(pair, copy);
    if (!this.slotOk[slot]) return;
    Atomics.store(this.ctl, readyIndex(slot), 0);
    if (!GPL_THREADED_PAIRS.includes(pair.name)) {
      for (const w of this.workers) w.postMessage({ type: 'pair', name: pair.name, fields, slot } satisfies PairThreadMessage);
      return;
    }
    // a GPL-ported style: its registry module is loaded here once (lazyStyles.ts), then the workers import it by URL;
    // until they have built it (PAIR_READY) the pair term runs on the engine thread. Each run's init sends the style
    // again, at once with the URL known.
    const known = this.gplUrls.get(pair.name);
    if (known !== undefined) {
      for (const w of this.workers) w.postMessage({ type: 'pair', name: pair.name, fields, url: known, slot } satisfies PairThreadMessage);
      return;
    }
    const version = this.slotVersion[slot];
    void loadGplThreadedPair(pair.name).then((g) => {
      if (!g) return;
      this.gplUrls.set(pair.name, g.url);
      if (this.slotRef[slot] !== pair || this.slotVersion[slot] !== version) return;
      for (const w of this.workers) w.postMessage({ type: 'pair', name: pair.name, fields, url: g.url, slot } satisfies PairThreadMessage);
    });
  }

  private copyList(list: NeighList, neighbors: Int32Array = list.neighbors): void {
    const nlocal = list.inum;
    const nnb = nlocal > 0 ? list.firstneigh[nlocal - 1] + list.numneigh[nlocal - 1] : 0;
    this.numneigh.set(list.numneigh.subarray(0, nlocal));
    this.firstneigh.set(list.firstneigh.subarray(0, nlocal));
    this.neighbors.set(neighbors.subarray(0, nnb));
  }

  /** New shared buffers at the current capacities, announced to every worker. */
  private allocate(): void {
    const xb = shared(24 * this.cap), tb = shared(4 * this.cap), qb = shared(8 * this.cap), idb = shared(4 * this.cap);
    this.tags = new Int32Array(idb);
    const nb = shared(4 * this.capL), fb = shared(4 * this.capL), lb = shared(4 * this.capN);
    this.x = new Float64Array(xb); this.t = new Int32Array(tb); this.q = new Float64Array(qb);
    this.numneigh = new Int32Array(nb); this.firstneigh = new Int32Array(fb); this.neighbors = new Int32Array(lb);
    const fbufs = this.workers.map(() => shared(24 * this.cap));
    this.sums = new Float64Array(this.cap);
    this.auxBuf = shared(8 * this.cap);
    this.aux = new Float64Array(this.auxBuf);
    this.fs = fbufs.map((b) => new Float64Array(b));
    this.accs = [];
    this.workers.forEach((w, k) => {
      const acc = shared(8 * ACC_LEN);
      this.accs.push(new Float64Array(acc));
      w.postMessage({ type: 'buffers', x: xb, t: tb, q: qb, numneigh: nb, firstneigh: fb, neighbors: lb, f: fbufs[k], fs: fbufs, acc, chunks: this.chunksBuf, tags: idb } satisfies PairThreadMessage);
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
   * work (candidate pairs), every thread claims chunks from the shared counter as in run(), and the chunks' entries are
   * returned in chunk order, which is the order of the search on one thread.
   */
  halfList(B: HalfBins, sp: SpecialBins | null, nbins: number, nlocal: number, binMax: number): { segs: ListSegment[]; np: number } | null {
    const W = this.workers.length;
    if (W === 0 || nlocal < this.minAtoms || Atomics.load(this.ctl, READY) < W) return null;
    // chunks of about equal work: a bin's owned atoms meet its own and the stencil's atoms, its ghosts only owned ones
    const weight = new Float64Array(nbins + 1);
    for (let b = 0; b < nbins; b++) weight[b + 1] = weight[b] + 2 * (B.bL[b + 1] - B.bL[b]) + (B.bG[b + 1] - B.bG[b]);
    const total = weight[nbins];
    if (total === 0) return null;
    const K = Math.max(1, Math.min(nbins, CHUNKS_PER_THREAD * (W + 1)));
    const nghost = B.gk.length, nall = nlocal + nghost;
    const npart = sp ? sp.offset[nlocal] : 0;
    // each worker's arrays hold twice its share of the entries of the last search (48 per atom at first) and one bin's
    // candidates (binMax), which are written before their test
    const share = Math.ceil((2 * (this.lastEntries > 0 ? this.lastEntries : 48 * nlocal)) / (W + 1)) + binMax;
    const S = this.neighborShared(nbins + 1, B.nrows, nlocal, nghost, B.nt * B.nt, K + 1, share, Math.max(1, npart), nall);
    S.bL.set(B.bL.subarray(0, nbins + 1)); S.bG.set(B.bG.subarray(0, nbins + 1));
    S.rLo.set(B.rLo.subarray(0, B.nrows)); S.rHi.set(B.rHi.subarray(0, B.nrows));
    S.lk.set(B.lk.subarray(0, nlocal)); S.lt.set(B.lt.subarray(0, nlocal)); S.lxs.set(B.lxs.subarray(0, 3 * nlocal));
    S.gk.set(B.gk.subarray(0, nghost)); S.gt.set(B.gt.subarray(0, nghost)); S.gxs.set(B.gxs.subarray(0, 3 * nghost));
    S.cutsq.set(B.cutsq);
    if (sp) {
      S.spOffset.set(sp.offset.subarray(0, nlocal + 1)); S.spPartner.set(sp.partner.subarray(0, npart)); S.spOrder.set(sp.order.subarray(0, npart));
      S.id.set(sp.id.subarray(0, nlocal)); S.owner.set(sp.owner.subarray(0, nall));
    }
    // the atom IDs in bin order of the ID-parity rule (neighborKernel.ts keepGhostPair)
    if (B.rule) { S.ltag.set(B.ltag!.subarray(0, nlocal)); S.gtag.set(B.gtag!.subarray(0, nghost)); }
    let k = 1;
    S.chunkBins[0] = 0;
    for (let b = 1; b < nbins && k < K; b++) if (weight[b] >= (total * k) / K) S.chunkBins[k++] = b;
    for (; k <= K; k++) S.chunkBins[k] = nbins;
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    Atomics.store(this.ctl, NEXT, 0);
    const special = sp ? { lj: Array.from(sp.lj), coul: Array.from(sp.coul), keep: sp.keep, hx: sp.hx, hy: sp.hy, hz: sp.hz, sbbits: sp.sbbits } : null;
    for (const w of this.workers) w.postMessage({ type: 'neigh', nrows: B.nrows, nt: B.nt, nchunks: K, binMax, special, rule: B.rule ?? 0, delta: B.delta ?? 0 } satisfies PairThreadMessage);
    // the engine thread claims chunks too, into its own arrays
    const out = new Int32Array(2);
    let own = 0;
    const searchOwn = (c: number): void => {
      let b = S.chunkBins[c];
      const b1 = S.chunkBins[c + 1];
      const start = own;
      while (b < b1) {
        if (sp) halfBinsPairsSpecial(B, sp, b, b1, this.ownPa, this.ownPj, own, this.ownPa.length, binMax, out);
        else halfBinsPairs(B, b, b1, this.ownPa, this.ownPj, own, this.ownPa.length, binMax, out);
        b = out[0]; own = out[1];
        if (b < b1) {
          const cap = Math.max(2 * this.ownPa.length, own + binMax);
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
    const segs: ListSegment[] = [];
    for (let c = 0; c < K; c++) {
      const t = S.chunkOut[3 * c], n = S.chunkOut[3 * c + 2];
      segs.push({ a: t === W ? this.ownPa : S.pa[t], j: t === W ? this.ownPj : S.pj[t], o: S.chunkOut[3 * c + 1], n });
      np += n;
    }
    this.lastEntries = grow ? 2 * np : np;
    this.neighborCalls++;
    return { segs, np };
  }

  /** The shared buffers of a neighbor search, reallocated (and announced to the workers) when they are too small. */
  private neighborShared(nb: number, nrows: number, nlocal: number, nghost: number, ntt: number, nchunk: number, share: number, npart: number, nall: number): NeighborShared {
    const want = [nb, nrows, nlocal, nghost, ntt, nchunk, share, npart, nall];
    const S = this.nshared;
    if (S && want.every((v, i) => v <= S.caps[i])) return S;
    const caps = want.map((v, i) => Math.max(S?.caps[i] ?? 0, Math.ceil(v * 1.25)));
    const [cb, cr, cl, cg, ct, cc, ce, cp, ca] = caps;
    const bufs = {
      bL: shared(4 * cb), bG: shared(4 * cb), rLo: shared(4 * cr), rHi: shared(4 * cr),
      lk: shared(4 * cl), lt: shared(4 * cl), lxs: shared(24 * cl), gk: shared(4 * cg), gt: shared(4 * cg), gxs: shared(24 * cg),
      ltag: shared(4 * cl), gtag: shared(4 * cg),
      cutsq: shared(8 * ct), chunkBins: shared(4 * cc), chunkOut: shared(12 * cc),
      spOffset: shared(4 * (cl + 1)), spPartner: shared(4 * cp), spOrder: shared(cp), id: shared(4 * cl), owner: shared(4 * ca),
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
      ltag: new Int32Array(bufs.ltag), gtag: new Int32Array(bufs.gtag),
      chunkBins: new Int32Array(bufs.chunkBins), chunkOut: new Int32Array(bufs.chunkOut),
      spOffset: new Int32Array(bufs.spOffset), spPartner: new Int32Array(bufs.spPartner), spOrder: new Int8Array(bufs.spOrder),
      id: new Int32Array(bufs.id), owner: new Int32Array(bufs.owner), pa, pj, caps,
    };
    return this.nshared;
  }

  // ---- PPPM's per-atom loops (force/kspace/pppmKernel.ts), on the same threads as the pair term

  private pX: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private pQ: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private pFirst: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private pWts: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private pMesh: Float64Array<ArrayBufferLike>[] = [];
  private pOut: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private pCapN = 0;
  private pCapG = 0;
  private pOrder = 0;
  private readonly pParams = new Float64Array(shared(8 * PPPM_PARAMS));

  /** The threads can take a loop now (every worker has started). */
  private threadsUp(): boolean {
    const W = this.workers.length;
    return W > 0 && Atomics.load(this.ctl, READY) >= W;
  }

  /** PPPM's shared arrays for n atoms of order P and meshes of len values, announced to the workers when they grow. */
  private pppmBuffers(n: number, P: number, len: number): void {
    let resend = false;
    if (n > this.pCapN || P !== this.pOrder) {
      this.pCapN = Math.ceil(Math.max(n, this.pCapN) * 1.25);
      this.pOrder = P;
      const N = this.pCapN;
      this.pX = new Float64Array(shared(24 * N)); this.pQ = new Float64Array(shared(8 * N));
      this.pFirst = new Int32Array(shared(12 * N)); this.pWts = new Float64Array(shared(24 * N * P));
      this.pOut = new Float64Array(shared(24 * N));
      resend = true;
    }
    if (len > this.pCapG) {
      this.pCapG = Math.ceil(len * 1.25);
      this.pMesh = [0, 1, 2].map(() => new Float64Array(shared(8 * this.pCapG)));
      resend = true;
    }
    if (!this.pMesh.length) { this.pMesh = [0, 1, 2].map(() => new Float64Array(shared(8))); resend = true; }
    if (!resend) return;
    const b = (a: { buffer: ArrayBufferLike }) => a.buffer as SharedArrayBuffer;
    const msg: PairThreadMessage = {
      type: 'pbuffers', x: b(this.pX), q: b(this.pQ), first: b(this.pFirst), wts: b(this.pWts),
      ex: b(this.pMesh[0]), ey: b(this.pMesh[1]), ez: b(this.pMesh[2]), out: b(this.pOut), params: b(this.pParams),
    };
    for (const w of this.workers) w.postMessage(msg);
  }

  /** One PPPM loop over chunks of atoms on every thread, the engine thread claiming chunks too. */
  private pppmLoop(phase: 1 | 2, g: PppmGrid, n: number, nchunks: number, C: number, V: number, zForce: boolean, own: (i0: number, i1: number) => void): void {
    const W = this.workers.length;
    writePppmParams(this.pParams, g, n, nchunks, C, V, zForce);
    const msg: PairThreadMessage = { type: 'pppm', phase };
    Atomics.store(this.ctl, DONE, 0);
    Atomics.store(this.ctl, FAILED, 0);
    Atomics.store(this.ctl, NEXT, 0);
    for (const w of this.workers) w.postMessage(msg);
    for (;;) {
      const c = Atomics.add(this.ctl, NEXT, 1);
      if (c >= nchunks) break;
      const [i0, i1] = atomChunk(n, nchunks, c);
      own(i0, i1);
    }
    for (;;) {
      const done = Atomics.load(this.ctl, DONE);
      if (done >= W) break;
      if (Atomics.wait(this.ctl, DONE, done, 60_000) === 'timed-out') throw new Error('PPPM threads did not answer within 60 s');
    }
    if (Atomics.load(this.ctl, FAILED) > 0) throw new Error('PPPM failed in a force thread');
  }

  pppmWeights(g: PppmGrid, x: Float64Array, n: number): { first: Int32Array; wts: Float64Array } | null {
    if (!this.threadsUp() || n < this.minAtoms) return null;
    const P = g.order;
    this.pppmBuffers(n, P, 0);
    this.pX.set(x.subarray(0, 3 * n));
    const nchunks = Math.min(n, CHUNKS_PER_THREAD * (this.workers.length + 1));
    const sc = weightScratch(P);
    this.pppmLoop(1, g, n, nchunks, 0, 1, false,
      (i0, i1) => pppmWeights(g, this.pX, i0, i1, this.pFirst, this.pWts, sc));
    return { first: this.pFirst.subarray(0, 3 * n), wts: this.pWts.subarray(0, 3 * n * P) };
  }

  pppmMeshes(len: number): [Float64Array, Float64Array, Float64Array] | null {
    if (!this.threadsUp() || this.pCapN === 0) return null;
    this.pppmBuffers(0, this.pOrder, len);
    const m = this.pMesh.map((a) => { const v = a.subarray(0, len); v.fill(0); return v; });
    return [m[0], m[1], m[2]];
  }

  pppmField(g: PppmGrid, n: number, q: ArrayLike<number>, first: Int32Array, wts: Float64Array,
    ex: Float64Array, ey: Float64Array, ez: Float64Array, C: number, V: number, zForce: boolean): Float64Array | null {
    // only the arrays these threads share: the weights of pppmWeights and the meshes of pppmMeshes
    if (!this.threadsUp() || first.buffer !== this.pFirst.buffer || wts.buffer !== this.pWts.buffer
      || ex.buffer !== this.pMesh[0].buffer || ey.buffer !== this.pMesh[1].buffer || ez.buffer !== this.pMesh[2].buffer || n > this.pCapN) return null;
    for (let i = 0; i < n; i++) this.pQ[i] = q[i];
    const nchunks = Math.min(n, CHUNKS_PER_THREAD * (this.workers.length + 1));
    const xs = new Int32Array(3 * g.order);
    this.pppmLoop(2, g, n, nchunks, C, V, zForce,
      (i0, i1) => pppmField(g, i0, i1, this.pQ, this.pFirst, this.pWts, this.pMesh[0], this.pMesh[1], this.pMesh[2], C, V, zForce, this.pOut, xs));
    return this.pOut.subarray(0, 3 * n);
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
