import { describe, expect, it } from 'vitest';
import { ParallelCpuForceBackend, sliceRange, type ForceWorkerLike } from '../src/engine/cpu/parallel';
import { cellGrid, sortIntoCells } from '../src/engine/cpu/cells';
import { pairArrays } from '../src/engine/pairs';
import { computeCellRange, type RangeTask } from '../src/engine/cpu/rangeKernel';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import { UNIT_SYSTEMS } from '../src/engine/units';
import { Rng } from '../src/engine/rng';
import { addAtoms, emptyState } from '../src/engine/md';
import { latticePoints, makeLattice } from '../src/engine/lattice';
import { newPairTable, resolvePairs, setPairCoeff } from '../src/engine/pairs';
import type { SimState } from '../src/engine/types';

/**
 * In-process stand-in for src/workers/force.worker.ts: same kernel, same
 * message shapes, answered asynchronously like a real worker. The real
 * threads are exercised in Chromium by the notebook live check.
 */
const fakeWorker = (log: number[]): ForceWorkerLike => {
  const w: ForceWorkerLike = {
    onmessage: null,
    postMessage(msg: unknown) {
      const { id, task } = structuredClone(msg) as { id: number; task: RangeTask };
      setTimeout(() => {
        const r = computeCellRange(task);
        log.push(task.cell1 - task.cell0);
        w.onmessage?.({ data: { id, ...r } } as MessageEvent);
      }, 0);
    },
    terminate() { /* nothing */ },
  };
  return w;
};

const lj = UNIT_SYSTEMS.lj;

const jitteredFcc = (cells: number, seed: number): SimState => {
  const lat = makeLattice('fcc', 0.8442, lj, 3);
  const hi: [number, number, number] = [cells * lat.spacing[0], cells * lat.spacing[1], cells * lat.spacing[2]];
  const s = emptyState(lj, 3, { lo: [0, 0, 0], hi, periodic: [true, true, true] }, 2);
  const pts = latticePoints(lat, [0, 0, 0], hi, 3);
  const rng = new Rng(seed);
  for (let k = 0; k < pts.length; k++) pts[k] = Math.min(hi[k % 3] - 1e-9, Math.max(0, pts[k] + 0.25 * (rng.uniform() - 0.5)));
  const half = (pts.length / 3) >> 1;
  addAtoms(s, pts.slice(0, 3 * half), 1);
  addAtoms(s, pts.slice(3 * half), 2);
  s.massByType[1] = 1; s.massByType[2] = 2;
  return s;
};

const twoTypes = () => {
  const t = newPairTable(2, 2.5);
  setPairCoeff(t, '1', '1', 1.0, 1.0);
  setPairCoeff(t, '2', '2', 0.7, 1.1, 3.0);
  resolvePairs(t);
  return t;
};

const compare = async (s: SimState, threads: number) => {
  const t = twoTypes();
  const serial = new CpuForceBackend().compute(s, t);
  const fs = Float64Array.from(s.f);
  s.f.fill(99);
  const log: number[] = [];
  const par = new ParallelCpuForceBackend(threads, () => fakeWorker(log));
  const pr = await par.compute(s, t);
  let worst = 0;
  let scale = 0;
  for (const v of fs) scale = Math.max(scale, Math.abs(v));
  for (let k = 0; k < fs.length; k++) worst = Math.max(worst, Math.abs(s.f[k] - fs[k]));
  par.dispose();
  return { worst: worst / scale, pe: Math.abs(pr.pe - serial.pe) / Math.abs(serial.pe), vir: Math.abs(pr.virial - serial.virial) / Math.abs(serial.virial), log, label: par.label };
};

describe('multi-threaded CPU forces', () => {
  it.each([1, 2, 3, 4, 7, 16])('%i thread(s) reproduce the single-thread forces, energy and virial', async (threads) => {
    const r = await compare(jitteredFcc(6, 11), threads);
    expect(r.worst).toBeLessThan(1e-12);
    expect(r.pe).toBeLessThan(1e-12);
    expect(r.vir).toBeLessThan(1e-12);
    expect(r.label).toBe(`CPU · fp64 · ${threads} thread${threads === 1 ? '' : 's'}`);
  });

  it('the engine thread takes one range and the workers the rest, covering every cell once', async () => {
    const r = await compare(jitteredFcc(8, 3), 4);
    // box 8 x 1.6796 = 13.44 over the largest cutoff 3.0 -> 4 cells per side, 64 cells;
    // 3 worker ranges plus the engine thread's own range
    expect(r.log.length).toBe(3);
    expect(r.log.reduce((a, b) => a + b, 0)).toBeLessThan(64);
    expect(r.log.reduce((a, b) => a + b, 0)).toBeGreaterThan(32);
  });

  it('2D systems match too', async () => {
    const lat = makeLattice('sq2', 0.7, lj, 2);
    const hi: [number, number, number] = [20 * lat.spacing[0], 20 * lat.spacing[1], 0.5];
    const s = emptyState(lj, 2, { lo: [0, 0, -0.5], hi, periodic: [true, true, true] }, 2);
    const pts = latticePoints(lat, [0, 0, -0.5], hi, 2);
    const rng = new Rng(5);
    for (let k = 0; k < pts.length; k += 3) { pts[k] = Math.min(hi[0] - 1e-9, Math.max(0, pts[k] + 0.2 * (rng.uniform() - 0.5))); }
    addAtoms(s, pts.slice(0, 300), 1);
    addAtoms(s, pts.slice(300), 2);
    s.massByType[1] = 1; s.massByType[2] = 1;
    const r = await compare(s, 5);
    expect(r.worst).toBeLessThan(1e-12);
    expect(r.pe).toBeLessThan(1e-12);
  });

  it.each([3, 5, 8, 12])('a box of 8x8x8 cells on %i threads (slabs + periodic wrap) matches too', async (threads) => {
    const r = await compare(jitteredFcc(16, 21), threads);
    expect(r.worst).toBeLessThan(1e-12);
    expect(r.pe).toBeLessThan(1e-12);
    expect(r.vir).toBeLessThan(1e-12);
  });

  it('each worker gets only its slab: own layers + one layer up (+ layer 0 when it wraps)', () => {
    const s = jitteredFcc(16, 21);
    const pa = pairArrays(twoTypes());
    const cs = sortIntoCells(s, cellGrid(s, pa.maxCutoff)!);
    const nxy = cs.nc[0] * cs.nc[1];
    const coef = { stride: pa.stride, cutsq: pa.cutsq, e12: pa.e12, e6: pa.e6, f12: pa.f12, f6: pa.f6, eshift: pa.eshift };
    // a middle range of 2 layers: 3 layers of atoms, no wrap
    const mid = sliceRange(cs, 3 * nxy, 5 * nxy, false, coef);
    expect(mid.task.ts.length).toBe(cs.start[6 * nxy] - cs.start[3 * nxy]);
    expect(mid.toGlobal(0)).toBe(cs.start[3 * nxy]);
    // the last 2 layers: those + layer 0 (the wrap), not the 5 layers between
    const last = sliceRange(cs, 6 * nxy, 8 * nxy, false, coef);
    expect(last.task.ts.length).toBe((cs.start[8 * nxy] - cs.start[6 * nxy]) + cs.start[nxy]);
    expect(last.toGlobal(0)).toBe(0);
    expect(last.toGlobal(cs.start[nxy])).toBe(cs.start[6 * nxy]);
    expect(last.task.ts.length).toBeLessThan(s.n / 2);
  });

  it('small boxes (< 3 cells) fall back to the single-thread path', async () => {
    const r = await compare(jitteredFcc(2, 1), 4);
    expect(r.log).toEqual([]);
    expect(r.worst).toBeLessThan(1e-14);
  });
});
