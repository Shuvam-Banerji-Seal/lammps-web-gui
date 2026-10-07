import { Session, RunCancelled } from './interpreter';
import { CpuForceBackend } from './cpu/forces';
import type { ForceBackend } from './types';
import type { BackendChoice, FromEngine, ToEngine } from './protocol';

/**
 * The engine side of the notebook protocol, independent of where it runs:
 * the Web Worker wraps it, and the client uses it directly on the main
 * thread when Workers are unavailable (tests, locked-down embeds).
 */
export class EngineHost {
  private session: Session | null = null;
  private backend: ForceBackend | null = null;
  private current = 0;

  constructor(private post: (msg: FromEngine, transfer?: Transferable[]) => void) {}

  async handle(msg: ToEngine): Promise<void> {
    if (msg.type === 'reset') return this.reset(msg.backend, msg.frameEvery);
    if (msg.type === 'cancel') { this.session?.cancel(); return; }
    if (msg.type === 'exec') return this.exec(msg.id, msg.text, msg.firstLine);
  }

  private async makeBackend(choice: BackendChoice): Promise<{ backend: ForceBackend; note?: string }> {
    if (choice === 'webgpu') {
      try {
        // loaded on demand: the WGSL backend is only needed when chosen
        const { createWebGpuBackend } = await import('./gpu/webgpuForces');
        const gpu = await createWebGpuBackend();
        if (gpu) return { backend: gpu };
        return { backend: new CpuForceBackend(), note: 'no WebGPU adapter in this browser; using the CPU' };
      } catch (e) {
        const why = e instanceof Error ? e.message : String(e);
        return { backend: new CpuForceBackend(), note: `WebGPU failed to start (${why}); using the CPU` };
      }
    }
    return { backend: new CpuForceBackend() };
  }

  private async reset(choice: BackendChoice, frameEvery: number): Promise<void> {
    this.backend?.dispose();
    const { backend, note } = await this.makeBackend(choice);
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
    }, backend, frameEvery);
    const webgpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;
    this.post({ type: 'ready', backend: backend.label, webgpuAvailable, note });
  }

  private async exec(id: number, text: string, firstLine: number): Promise<void> {
    if (!this.session) await this.reset('cpu', 0);
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
