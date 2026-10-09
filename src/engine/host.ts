import { Session, RunCancelled, SUPPORTED_COMMANDS } from './interpreter';
import { styleNames } from './styles';
import { CpuForceBackend } from './cpu/forces';
import { ParallelCpuForceBackend } from './cpu/parallel';
import { SharedThreadsBackend, SharedPairThreads, sharedThreadsAvailable } from './cpu/pairThreads';
import { probeDevice, autoPlan, autoThreads, type AutoPlan, type DeviceProfile } from './device';
import type { ForceBackend } from './types';
import type { BackendChoice, FromEngine, ToEngine } from './protocol';

/**
 * The engine side of the notebook protocol, independent of where it runs:
 * the Web Worker wraps it, and the client uses it directly on the main
 * thread when Workers are unavailable (tests, locked-down embeds).
 */

const cores = (): number => {
  const n = (globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator?.hardwareConcurrency;
  return n && n > 0 ? n : 1;
};

const hasWorkers = (): boolean => typeof Worker !== 'undefined';

export class EngineHost {
  private session: Session | null = null;
  private backend: ForceBackend | null = null;
  private current = 0;
  private frameEvery = 0;
  /** Files added by the notebook; kept across session resets. */
  private files = new Map<string, string>();
  private device: DeviceProfile | null = null;
  private plan: AutoPlan | undefined;
  private threadsInUse = 1;

  /** The device profile (probed once: the WebGPU adapter request is not free). */
  private async profile(): Promise<DeviceProfile> {
    this.device ??= await probeDevice();
    return this.device;
  }

  constructor(private post: (msg: FromEngine, transfer?: Transferable[]) => void) {}

  async handle(msg: ToEngine): Promise<void> {
    if (msg.type === 'reset') return this.reset(msg.backend, msg.threads, msg.frameEvery);
    if (msg.type === 'backend') return this.switchBackend(msg.backend, msg.threads);
    if (msg.type === 'cancel') { this.session?.cancel(); return; }
    if (msg.type === 'exec') return this.exec(msg.id, msg.text, msg.firstLine);
    if (msg.type === 'file') {
      if (msg.text === null) { this.files.delete(msg.name); this.session?.removeFile(msg.name); return; }
      this.files.set(msg.name, msg.text);
      this.session?.addFile(msg.name, msg.text);
    }
  }

  private cpuBackend(threads: number): ForceBackend {
    const t = Math.max(1, Math.min(Math.floor(threads) || 1, cores()));
    this.threadsInUse = t <= 1 || !hasWorkers() ? 1 : t;
    // threads need (nested) Web Workers; with shared memory (a cross-origin isolated page) the
    // general engine runs its pair term on them, otherwise only plain lj/cut runs threaded
    if (t <= 1 || !hasWorkers()) return new CpuForceBackend();
    return sharedThreadsAvailable() ? new SharedThreadsBackend(t) : new ParallelCpuForceBackend(t);
  }

  private async makeBackend(choice: BackendChoice, threads: number): Promise<{ backend: ForceBackend; note?: string }> {
    const device = await this.profile();
    this.plan = undefined;
    if (choice === 'auto') {
      this.plan = autoPlan(device);
      choice = this.plan.backend;
      if (threads <= 0) threads = this.plan.threads;
    }
    if (threads <= 0) {
      threads = autoThreads(device);
      this.plan ??= { backend: choice, threads, why: `${threads} CPU threads (half the ${device.cores} cores, at most 8)` };
    }
    if (choice === 'webgpu') {
      try {
        // loaded on demand: the WGSL backend is only needed when chosen
        const { createWebGpuBackend, webgpuAdapterKind } = await import('./gpu/webgpuForces');
        const gpu = await createWebGpuBackend();
        if (gpu) return { backend: this.withFallbackThreads(gpu, threads) };
        const kind = await webgpuAdapterKind();
        return {
          backend: this.cpuBackend(threads),
          note: kind === 'software'
            ? 'this browser offers only a software WebGPU adapter (SwiftShader), which is slower than the CPU engine; using the CPU'
            : 'no WebGPU adapter in this browser; using the CPU',
        };
      } catch (e) {
        const why = e instanceof Error ? e.message : String(e);
        return { backend: this.cpuBackend(threads), note: `WebGPU failed to start (${why}); using the CPU` };
      }
    }
    return { backend: this.cpuBackend(threads) };
  }

  /**
   * A GPU backend runs plain lj/cut only (run/accel.ts); every other run takes the general fp64 engine,
   * which used to run on one thread under a GPU backend. With shared memory it now gets the pair threads.
   */
  private withFallbackThreads(gpu: ForceBackend, threads: number): ForceBackend {
    const t = Math.max(1, Math.min(Math.floor(threads) || 1, cores()));
    this.threadsInUse = 1;
    if (t <= 1 || !hasWorkers() || !sharedThreadsAvailable()) return gpu;
    const pairThreads = new SharedPairThreads(t);
    this.threadsInUse = t;
    const dispose = gpu.dispose.bind(gpu);
    return Object.assign(gpu, { pairThreads, dispose: () => { pairThreads.dispose(); dispose(); } });
  }

  private ready(backend: ForceBackend, note?: string): void {
    const webgpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;
    this.post({
      type: 'ready', backend: backend.label, kind: backend.kind, webgpuAvailable, cores: cores(), note,
      device: this.device!, auto: this.plan, threads: this.threadsInUse,
      commands: [...SUPPORTED_COMMANDS], styles: styleNames(),
    });
  }

  private async reset(choice: BackendChoice, threads: number, frameEvery: number): Promise<void> {
    this.backend?.dispose();
    this.frameEvery = frameEvery;
    const { backend, note } = await this.makeBackend(choice, threads);
    this.backend = backend;
    this.session = new Session({
      emit: (event) => {
        if (event.kind === 'frame') {
          this.post({ type: 'event', id: this.current, event }, [event.x.buffer, event.image.buffer, event.type.buffer, event.id.buffer]);
        } else {
          this.post({ type: 'event', id: this.current, event });
        }
      },
      writeFile: (name, text, append) => this.post({ type: 'file', id: this.current, name, text, append }),
    }, backend, this.frameEvery);
    for (const [name, text] of this.files) this.session.addFile(name, text);
    this.ready(backend, note);
  }

  private async switchBackend(choice: BackendChoice, threads: number): Promise<void> {
    if (!this.session) return this.reset(choice, threads, this.frameEvery);
    const { backend, note } = await this.makeBackend(choice, threads);
    const old = this.backend;
    this.backend = backend;
    this.session.setBackend(backend);
    old?.dispose();
    this.ready(backend, note);
  }

  private async exec(id: number, text: string, firstLine: number): Promise<void> {
    if (!this.session) await this.reset('cpu', 1, 0);
    this.current = id;
    try {
      await this.session!.execute(text, firstLine);
      this.post({ type: 'finished', id, ok: true, cancelled: false });
    } catch (e) {
      // EngineErrors were already emitted as 'error' events by the session
      this.post({ type: 'finished', id, ok: false, cancelled: e instanceof RunCancelled });
    }
  }
}
