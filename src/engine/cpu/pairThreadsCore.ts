import { newAccum, type Pair } from '../force/types';
import type { NeighList } from '../neighbor';
import { THREADED_PAIRS } from './threadedPairs';

/*
 * The worker side of the shared-memory pair threads (pairThreads.ts) and the helpers both sides
 * share. Kept apart from pairThreads.ts so the worker bundle (src/workers/pair.worker.ts) does not
 * contain the code that spawns it.
 */

/** The part of a Worker the pool uses (tests pass in-process stand-ins). */
export interface PairWorkerLike {
  postMessage(msg: unknown): void;
  onmessage: ((ev: MessageEvent) => void) | null;
  terminate(): void;
}

export type PairThreadMessage =
  | { type: 'init'; ctl: SharedArrayBuffer; index: number }
  | { type: 'buffers'; x: SharedArrayBuffer; t: SharedArrayBuffer; q: SharedArrayBuffer; numneigh: SharedArrayBuffer; firstneigh: SharedArrayBuffer; neighbors: SharedArrayBuffer; f: SharedArrayBuffer; acc: SharedArrayBuffer; chunks: SharedArrayBuffer }
  | { type: 'pair'; name: string; fields: Record<string, unknown> }
  | { type: 'compute'; nlocal: number; nall: number; qqrd2e: number; specialLJ: number[]; specialCoul: number[]; nchunks: number };

/** Words of the control buffer: finished workers, failed workers. */
/**
 * Slots of the shared control array: DONE counts finished ranges, FAILED counts failed ones, READY counts workers that
 * have started and handled their init message. A worker created from inside the engine worker may only start once the
 * engine thread returns to its event loop, so the engine must not block in Atomics.wait on a worker before it is READY.
 */
export const DONE = 0, FAILED = 1, READY = 2;
/**
 * NEXT is the next unclaimed chunk of the neighbor list. Every thread (the engine thread too) claims chunks with
 * Atomics.add until none are left, so a thread that runs slower (a busy core, an efficiency core, a late wake-up)
 * simply takes fewer chunks. Measured in Chromium before this (8 threads, 16k-atom LJ, fixed equal ranges): the
 * engine thread finished its range in 2.1 ms and then waited 2.4 ms for the slowest worker.
 */
export const NEXT = 3;
/** Chunks per thread: enough to even out the load, few enough that the per-call overhead stays small. */
export const CHUNKS_PER_THREAD = 8;
/** Doubles of a worker's result: evdwl, ecoul, virial[6]. */
export const ACC_LEN = 8;

/** Splits [0, inum) into k ranges with about the same number of list entries (plus one per atom). */
export const splitRanges = (numneigh: Int32Array, inum: number, k: number): Int32Array => {
  const b = new Int32Array(k + 1);
  let total = 0;
  for (let i = 0; i < inum; i++) total += numneigh[i] + 1;
  let acc = 0, r = 1;
  for (let i = 0; i < inum && r < k; i++) {
    acc += numneigh[i] + 1;
    while (r < k && acc >= (total * r) / k) b[r++] = i + 1;
  }
  for (; r <= k; r++) b[r] = inum;
  return b;
};

/** numneigh with the atoms outside [i0, i1) given no neighbors, written into `out`. */
export const restrictCounts = (numneigh: Int32Array, inum: number, i0: number, i1: number, out: Int32Array): Int32Array => {
  out.fill(0, 0, inum);
  out.set(numneigh.subarray(i0, i1), i0);
  return out;
};

/** The pair style's fields that survive structured cloning (caches holding functions are left out). */
export const cloneableFields = (pair: Pair): Record<string, unknown> => {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(pair)) {
    if (typeof v === 'function') continue;
    try {
      out[k] = structuredClone(v);
    } catch {
      // e.g. an erfc table cache (closures): the worker's fresh instance builds its own
    }
  }
  return out;
};

const plainProto = (o: object): boolean => {
  const p = Object.getPrototypeOf(o);
  return p === Object.prototype || p === null;
};

/** Copies cloned fields into a fresh instance, keeping the instance's own class-typed members (their methods). */
export const adoptFields = (target: Record<string, unknown>, src: Record<string, unknown>): void => {
  for (const k of Object.keys(src)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    const v = src[k], cur = target[k];
    if (v && typeof v === 'object' && cur && typeof cur === 'object' && !ArrayBuffer.isView(cur) && !Array.isArray(cur)
      && !(cur instanceof Map) && !(cur instanceof Set) && !plainProto(cur) && plainProto(v)) {
      adoptFields(cur as Record<string, unknown>, v as Record<string, unknown>);
    } else {
      target[k] = v;
    }
  }
};

/** A pair style rebuilt in a worker from the engine's fields (null when the style is not threaded). */
export const pairFromFields = (name: string, fields: Record<string, unknown>): Pair | null => {
  if (!Object.hasOwn(THREADED_PAIRS, name)) return null;
  const make = THREADED_PAIRS[name];
  const p = make();
  // allocate() creates the class-typed members (coefficient tables) the cloned fields go into
  if (typeof fields.ntypes === 'number' && fields.ntypes > 0) p.allocate(fields.ntypes);
  adoptFields(p as unknown as Record<string, unknown>, fields);
  return p;
};

/**
 * Every class-typed member of the engine's style has the same class in the rebuilt copy (a member
 * that came back as a plain object would have lost its methods).
 */
export const sameShape = (orig: object, copy: object, depth = 0): boolean => {
  if (depth > 4) return true;
  const o = orig as Record<string, unknown>, c = copy as Record<string, unknown>;
  for (const [k, v] of Object.entries(o)) {
    if (!v || typeof v !== 'object' || ArrayBuffer.isView(v) || Array.isArray(v) || v instanceof Map || v instanceof Set) continue;
    if (plainProto(v)) continue;
    const w = c[k];
    if (!w || typeof w !== 'object' || Object.getPrototypeOf(w) !== Object.getPrototypeOf(v)) return false;
    if (!sameShape(v, w, depth + 1)) return false;
  }
  return true;
};

/** One worker's side of the protocol, also run in-process by tests. */
export class PairThreadWorker {
  private ctl: Int32Array | null = null;
  private pair: Pair | null = null;
  private x: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private t: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private q: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private numneigh: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private firstneigh: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private neighbors: Int32Array<ArrayBufferLike> = new Int32Array(0);
  private f: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private acc: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private chunks: Int32Array<ArrayBufferLike> = new Int32Array(0);

  handle(m: PairThreadMessage): void {
    if (m.type === 'init') {
      this.ctl = new Int32Array(m.ctl);
      Atomics.add(this.ctl, READY, 1);
      return;
    }
    if (m.type === 'buffers') {
      this.x = new Float64Array(m.x); this.t = new Int32Array(m.t); this.q = new Float64Array(m.q);
      this.numneigh = new Int32Array(m.numneigh); this.firstneigh = new Int32Array(m.firstneigh); this.neighbors = new Int32Array(m.neighbors);
      this.f = new Float64Array(m.f); this.acc = new Float64Array(m.acc); this.chunks = new Int32Array(m.chunks);
      return;
    }
    if (m.type === 'pair') { this.pair = pairFromFields(m.name, m.fields); return; }
    const ctl = this.ctl!;
    try {
      const pair = this.pair;
      if (!pair) throw new Error('no pair style');
      this.f.fill(0, 0, 3 * m.nall);
      const a = newAccum();
      const specialLJ = Float64Array.from(m.specialLJ), specialCoul = Float64Array.from(m.specialCoul);
      for (;;) {
        const c = Atomics.add(ctl, NEXT, 1);
        if (c >= m.nchunks) break;
        const half: NeighList = { ilo: this.chunks[c], inum: this.chunks[c + 1], numneigh: this.numneigh, firstneigh: this.firstneigh, neighbors: this.neighbors };
        pair.compute({
          x: this.x, f: this.f, type: this.t, q: this.q, nlocal: m.nlocal, nall: m.nall, half, full: null,
          specialLJ, specialCoul, qqrd2e: m.qqrd2e,
          acc: a, eatom: null, vatom: null,
          // THREADED_PAIRS styles do not read these
          s: undefined as never, nb: undefined as never, geom: undefined as never,
        });
      }
      this.acc[0] = a.evdwl; this.acc[1] = a.ecoul;
      for (let c = 0; c < 6; c++) this.acc[2 + c] = a.virial[c];
    } catch {
      Atomics.add(ctl, FAILED, 1);
    }
    Atomics.add(ctl, DONE, 1);
    Atomics.notify(ctl, DONE);
  }
}

