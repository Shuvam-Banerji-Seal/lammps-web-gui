import { EngineHost } from './host';
import type { BackendChoice, FromEngine, ToEngine } from './protocol';
import type { EngineEvent } from './types';

/**
 * UI side of the notebook engine. Runs the engine in a Web Worker when
 * possible and on the calling thread otherwise (tests, locked-down embeds),
 * with the same message protocol either way.
 */

export interface ExecHandlers {
  onEvent: (event: EngineEvent) => void;
  onFile?: (name: string, text: string, append: boolean) => void;
}

export interface ExecResult { ok: boolean; cancelled: boolean }

export class EngineClient {
  private worker: Worker | null = null;
  private host: EngineHost | null = null;
  private nextId = 1;
  private pending = new Map<number, { handlers: ExecHandlers; resolve: (r: ExecResult) => void }>();
  private readyWaiters: ((info: Extract<FromEngine, { type: 'ready' }>) => void)[] = [];
  /** Where the engine runs, for display. */
  readonly onMainThread: boolean;

  constructor(useWorker = typeof Worker !== 'undefined') {
    if (useWorker) {
      try {
        this.worker = new Worker(new URL('../workers/engine.worker.ts', import.meta.url), { type: 'module' });
        this.worker.onmessage = (ev: MessageEvent<FromEngine>) => this.receive(ev.data);
      } catch {
        this.worker = null;
      }
    }
    if (!this.worker) this.host = new EngineHost((msg) => this.receive(msg));
    this.onMainThread = !this.worker;
  }

  private send(msg: ToEngine): void {
    if (this.worker) this.worker.postMessage(msg);
    else void this.host!.handle(msg);
  }

  private receive(msg: FromEngine): void {
    if (msg.type === 'ready') {
      const waiters = this.readyWaiters;
      this.readyWaiters = [];
      waiters.forEach((w) => w(msg));
      return;
    }
    const entry = this.pending.get(msg.id);
    if (!entry) return;
    if (msg.type === 'event') entry.handlers.onEvent(msg.event);
    else if (msg.type === 'file') entry.handlers.onFile?.(msg.name, msg.text, msg.append);
    else if (msg.type === 'finished') {
      this.pending.delete(msg.id);
      entry.resolve({ ok: msg.ok, cancelled: msg.cancelled });
    }
  }

  /** Starts a fresh session; resolves with the backend actually in use. */
  reset(backend: BackendChoice, frameEvery = 0, threads = 1): Promise<Extract<FromEngine, { type: 'ready' }>> {
    const p = new Promise<Extract<FromEngine, { type: 'ready' }>>((r) => this.readyWaiters.push(r));
    this.send({ type: 'reset', backend, threads, frameEvery });
    return p;
  }

  /** Changes CPU/GPU or the CPU thread count, keeping the current system. */
  setBackend(backend: BackendChoice, threads = 1): Promise<Extract<FromEngine, { type: 'ready' }>> {
    const p = new Promise<Extract<FromEngine, { type: 'ready' }>>((r) => this.readyWaiters.push(r));
    this.send({ type: 'backend', backend, threads });
    return p;
  }

  /** Runs one cell; events for it stream to `handlers` until it finishes. */
  exec(text: string, firstLine: number, handlers: ExecHandlers): Promise<ExecResult> {
    const id = this.nextId++;
    const p = new Promise<ExecResult>((resolve) => this.pending.set(id, { handlers, resolve }));
    this.send({ type: 'exec', id, text, firstLine });
    return p;
  }

  cancel(): void {
    this.send({ type: 'cancel' });
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    for (const [, e] of this.pending) e.resolve({ ok: false, cancelled: true });
    this.pending.clear();
  }
}
