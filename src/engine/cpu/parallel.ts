import type { ForceBackend, ForceResult, PairTable, SimState } from '../types';
import { pairArrays, type PairArrays } from '../pairs';
import { CpuForceBackend } from './forces';
import { cellGrid, sortIntoCells, type CellSort } from './cells';
import { computeCellRange, type RangeResult, type RangeTask } from './rangeKernel';

/*
 * Multi-threaded fp64 CPU forces: the engine worker sorts the atoms into
 * cells, splits the cells into `threads` contiguous ranges holding about the
 * same number of atoms, and sends each range to its own force worker
 * (src/workers/force.worker.ts, running computeCellRange); the engine thread
 * computes the first range itself instead of waiting, so N threads = the
 * engine thread + N-1 workers. Each pair is evaluated once (half stencil);
 * a thread returns the forces on its own atoms plus a short list of reaction
 * forces on other threads' atoms, summed here. Nothing is shared, so this
 * needs no SharedArrayBuffer / cross-origin isolation.
 * Small boxes (< 3 cells in a dimension) use the single-thread backend.
 */

/** The part of a Worker this backend uses — lets tests run it in-process. */
export interface ForceWorkerLike {
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  onmessage: ((ev: MessageEvent) => void) | null;
  terminate(): void;
}

/**
 * The atoms one range's thread needs, copied into small local arrays: the
 * range's own z-layers of cells, the next layer (half-stencil neighbours
 * reach one layer up), and layer 0 when that next layer wraps around the
 * periodic box. In-layer neighbours (+y, +x, with their wraps) stay in the
 * same layer. Local slot = global slot - w1Start + w0Len inside the main
 * window, = global slot inside the wrapped layer 0. Sending this instead of
 * all positions cuts the per-step copy from N to ~N/threads + 2 layers.
 */
export interface RangeSlice {
  task: RangeTask;
  /** Global slot of local slot `l`. */
  toGlobal: (l: number) => number;
}

export const sliceRange = (cs: CellSort, cell0: number, cell1: number, two: boolean, coef: RangeTask['coef']): RangeSlice => {
  const [ncx, ncy, ncz] = cs.nc;
  const nxy = ncx * ncy;
  const start = cs.start;
  const firstLayer = Math.floor(cell0 / nxy);
  const lastLayer = Math.floor((cell1 - 1) / nxy);
  const topLayer = two ? lastLayer : Math.min(ncz - 1, lastLayer + 1);
  const wraps = !two && lastLayer + 1 >= ncz && firstLayer > 0;
  const w1Start = start[firstLayer * nxy];
  const w1End = start[(topLayer + 1) * nxy];
  const w0Len = wraps ? start[nxy] : 0;
  const len = w0Len + (w1End - w1Start);
  const xs = new Float64Array(3 * len);
  const ts = new Int32Array(len);
  if (wraps) {
    xs.set(cs.xs.subarray(0, 3 * w0Len), 0);
    ts.set(cs.ts.subarray(0, w0Len), 0);
  }
  xs.set(cs.xs.subarray(3 * w1Start, 3 * w1End), 3 * w0Len);
  ts.set(cs.ts.subarray(w1Start, w1End), w0Len);
  // local cell offsets for every cell the thread can touch
  const local = new Int32Array(cs.ncell + 1);
  if (wraps) for (let c = 0; c <= nxy; c++) local[c] = start[c];
  for (let c = firstLayer * nxy; c <= (topLayer + 1) * nxy; c++) local[c] = start[c] - w1Start + w0Len;
  return {
    task: { xs, ts, start: local, nc: cs.nc, L: cs.L, two, cell0, cell1, coef },
    toGlobal: (l) => (l < w0Len ? l : l - w0Len + w1Start),
  };
};

export const spawnForceWorker = (): ForceWorkerLike =>
  new Worker(new URL('../../workers/force.worker.ts', import.meta.url), { type: 'module' });

export class ParallelCpuForceBackend implements ForceBackend {
  readonly kind = 'cpu' as const;
  readonly label: string;
  private workers: ForceWorkerLike[] = [];
  private serial = new CpuForceBackend();
  private sortBuf: CellSort | undefined;
  private cache: { key: string; arrays: PairArrays } | null = null;
  private nextId = 1;
  private pending = new Map<number, (r: RangeResult) => void>();

  constructor(readonly threads: number, spawn: () => ForceWorkerLike = spawnForceWorker) {
    this.label = `CPU · fp64 · ${threads} thread${threads === 1 ? '' : 's'}`;
    for (let k = 0; k < threads - 1; k++) {
      const w = spawn();
      w.onmessage = (ev: MessageEvent<RangeResult & { id: number }>) => {
        const done = this.pending.get(ev.data.id);
        this.pending.delete(ev.data.id);
        done?.(ev.data);
      };
      this.workers.push(w);
    }
  }

  private arraysFor(table: PairTable): PairArrays {
    const key = JSON.stringify([table.pairs, table.shift]);
    if (!this.cache || this.cache.key !== key) this.cache = { key, arrays: pairArrays(table) };
    return this.cache.arrays;
  }

  async compute(state: SimState, table: PairTable): Promise<ForceResult> {
    const pa = this.arraysFor(table);
    if (state.n === 0 || pa.maxCutoff <= 0) {
      state.f.fill(0);
      return { pe: 0, virial: 0 };
    }
    const grid = cellGrid(state, pa.maxCutoff);
    if (!grid || this.threads <= 1) return this.serial.compute(state, table);
    const cs = (this.sortBuf = sortIntoCells(state, grid, this.sortBuf));
    const n = state.n;
    const xs = cs.xs.subarray(0, 3 * n);
    const ts = cs.ts.subarray(0, n);
    const start = cs.start.subarray(0, cs.ncell + 1);

    // contiguous cell ranges with ~n/threads atoms each
    const bounds = [0];
    const per = n / this.threads;
    for (let c = 0, k = 1; c < cs.ncell && k < this.threads; c++) {
      if (start[c + 1] >= per * k) { bounds.push(c + 1); k++; }
    }
    bounds.push(cs.ncell);
    const coef = { stride: pa.stride, cutsq: pa.cutsq, e12: pa.e12, e6: pa.e6, f12: pa.f12, f6: pa.f6, eshift: pa.eshift };
    const two = state.dimension === 2;
    const jobs: Promise<{ r: RangeResult; toGlobal: (l: number) => number }>[] = [];
    // ranges 1.. go to the workers, each with only the slab of atoms it needs
    for (let k = 1; k + 1 < bounds.length; k++) {
      if (bounds[k] === bounds[k + 1]) continue;
      const slice = sliceRange(cs, bounds[k], bounds[k + 1], two, coef);
      const id = this.nextId++;
      jobs.push(new Promise((resolve) => this.pending.set(id, (r) => resolve({ r, toGlobal: slice.toGlobal }))));
      const t = slice.task;
      this.workers[(k - 1) % this.workers.length].postMessage({ id, task: t }, [t.xs.buffer, t.ts.buffer, t.start.buffer]);
    }
    // the engine thread computes range 0 on the full arrays meanwhile
    const own = computeCellRange({ xs, ts, start, nc: cs.nc, L: cs.L, two, cell0: bounds[0], cell1: bounds[1], coef });
    const results = [{ r: own, toGlobal: (l: number) => l }, ...(await Promise.all(jobs))];
    const f = state.f;
    const sorted = cs.sorted;
    let pe = 0;
    let virial = 0;
    // own ranges partition all slots: set them first, then add the reactions
    for (const { r, toGlobal } of results) {
      pe += r.pe;
      virial += r.virial;
      const m = r.f.length / 3;
      const g0 = toGlobal(r.slot0);   // own slots are contiguous in both numberings
      for (let j = 0; j < m; j++) {
        const i = sorted[g0 + j];
        f[3 * i] = r.f[3 * j]; f[3 * i + 1] = r.f[3 * j + 1]; f[3 * i + 2] = r.f[3 * j + 2];
      }
    }
    for (const { r, toGlobal } of results) {
      const g = r.ghostSlots;
      for (let j = 0; j < g.length; j++) {
        const i = sorted[toGlobal(g[j])];
        f[3 * i] += r.ghostF[3 * j]; f[3 * i + 1] += r.ghostF[3 * j + 1]; f[3 * i + 2] += r.ghostF[3 * j + 2];
      }
    }
    return { pe, virial };
  }

  dispose(): void {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    for (const [, done] of this.pending) {
      done({ slot0: 0, f: new Float64Array(0), ghostSlots: new Int32Array(0), ghostF: new Float64Array(0), pe: Number.NaN, virial: Number.NaN });
    }
    this.pending.clear();
  }
}
