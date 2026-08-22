import type { FileFormat, MoleculeData } from '../types';

/**
 * Client side of the parser worker. Routes responses by request id so
 * concurrent parses cannot cross wires. Falls back to synchronous parsing
 * when Workers are unavailable (tests, very old browsers).
 */
export interface ParseRequest {
  id: number;
  text: string;
  format?: FileFormat;
}

export interface ParseResponse {
  id: number;
  ok: boolean;
  data?: MoleculeData;
  error?: string;
}

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, { resolve: (d: MoleculeData) => void; reject: (e: Error) => void }>();

const ensureWorker = (): Worker | null => {
  if (worker) return worker;
  try {
    worker = new Worker(new URL('../workers/parser.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (ev: MessageEvent<ParseResponse>) => {
      const entry = pending.get(ev.data.id);
      if (!entry) return;
      pending.delete(ev.data.id);
      if (ev.data.ok && ev.data.data) entry.resolve(ev.data.data);
      else entry.reject(new Error(ev.data.error ?? 'Worker parse failed'));
    };
    worker.onerror = () => {
      // Fail every in-flight request; callers fall back to sync parsing.
      for (const [, entry] of pending) entry.reject(new Error('worker-crashed'));
      pending.clear();
    };
    return worker;
  } catch {
    worker = null;
    return null;
  }
};

/** Parse off the main thread. Rejects with 'no-worker' when unavailable. */
export const parseInWorker = (text: string, format?: FileFormat): Promise<MoleculeData> =>
  new Promise((resolve, reject) => {
    const w = ensureWorker();
    if (!w) {
      reject(new Error('no-worker'));
      return;
    }
    const id = nextId++;
    pending.set(id, { resolve, reject });
    w.postMessage({ id, text, format } satisfies ParseRequest);
  });
