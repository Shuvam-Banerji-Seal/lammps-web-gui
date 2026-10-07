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
  postMessage(msg: unknown): void;
  onmessage: ((ev: MessageEvent) => void) | null;
  terminate(): void;
}

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
    const jobs: Promise<RangeResult>[] = [];
    const task = (k: number): RangeTask => ({
      xs, ts, start, nc: cs.nc, L: cs.L, two: state.dimension === 2,
      cell0: bounds[k], cell1: bounds[k + 1], coef,
    });
    // ranges 1.. go to the workers; the engine thread computes range 0
    for (let k = 1; k + 1 < bounds.length; k++) {
      if (bounds[k] === bounds[k + 1]) continue;
      const id = this.nextId++;
      jobs.push(new Promise((resolve) => this.pending.set(id, resolve)));
      this.workers[(k - 1) % this.workers.length].postMessage({ id, task: task(k) });
    }
    const own = computeCellRange(task(0));
    const results = [own, ...(await Promise.all(jobs))];
    const f = state.f;
    const sorted = cs.sorted;
    let pe = 0;
    let virial = 0;
    // own ranges partition all slots: set them first, then add the reactions
    for (const r of results) {
      pe += r.pe;
      virial += r.virial;
      const m = r.f.length / 3;
      for (let j = 0; j < m; j++) {
        const i = sorted[r.slot0 + j];
        f[3 * i] = r.f[3 * j]; f[3 * i + 1] = r.f[3 * j + 1]; f[3 * i + 2] = r.f[3 * j + 2];
      }
    }
    for (const r of results) {
      const g = r.ghostSlots;
      for (let j = 0; j < g.length; j++) {
        const i = sorted[g[j]];
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
