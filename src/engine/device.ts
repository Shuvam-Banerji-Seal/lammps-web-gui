/**
 * What this device offers the engine, and the settings "Auto" picks from it.
 *
 * The probe runs where the engine runs (the Web Worker, or the page when Workers are missing), so it
 * reports what the engine can really use: logical cores, memory, Web Workers, shared memory (a
 * cross-origin isolated page, needed for the general engine's pair threads, see cpu/pairThreads.ts)
 * and the WebGPU adapter.
 */
import type { WebGpuAdapterInfo } from './gpu/webgpuForces';

/**
 * The same test as cpu/pairThreads.ts sharedThreadsAvailable, repeated here so that the notebook's
 * resource monitor (which imports this module) does not pull the threaded pair code into the page bundle.
 */
const sharedMemoryAvailable = (): boolean =>
  typeof SharedArrayBuffer !== 'undefined' && typeof Atomics !== 'undefined' && typeof Worker !== 'undefined'
  && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true
  && typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined';

export interface DeviceProfile {
  /** navigator.hardwareConcurrency (at least 1). */
  cores: number;
  /** navigator.deviceMemory in GB (Chromium; rounded by the browser), or null. */
  memoryGB: number | null;
  workers: boolean;
  /** SharedArrayBuffer + Atomics in a cross-origin isolated context: every threaded pair style can run. */
  sharedMemory: boolean;
  gpu: WebGpuAdapterInfo;
}

const nav = (): { hardwareConcurrency?: number; deviceMemory?: number; gpu?: unknown } =>
  (globalThis as { navigator?: { hardwareConcurrency?: number; deviceMemory?: number } }).navigator ?? {};

export const probeDevice = async (): Promise<DeviceProfile> => {
  const n = nav();
  let gpu: WebGpuAdapterInfo = { kind: 'none', name: '', compatibility: false, maxStorageBufferMB: 0 };
  if (n.gpu) {
    try {
      // loaded on demand, like the WebGPU backend itself
      const { webgpuAdapterInfo } = await import('./gpu/webgpuForces');
      gpu = await webgpuAdapterInfo();
    } catch { /* no adapter */ }
  }
  return {
    cores: n.hardwareConcurrency && n.hardwareConcurrency > 0 ? n.hardwareConcurrency : 1,
    memoryGB: typeof n.deviceMemory === 'number' && n.deviceMemory > 0 ? n.deviceMemory : null,
    workers: typeof Worker !== 'undefined',
    sharedMemory: sharedMemoryAvailable(),
    gpu,
  };
};

/**
 * CPU threads for "Auto": half the logical cores, at most 8 (1 without Web Workers).
 * Measured in Chromium (24 logical cores, cross-origin isolated, LJ melt, steps/s at 1/2/4/8/12/16
 * threads): 2048 atoms 373/437/491/499/384/306, 6912 atoms 117/153/149/152/134/119, 16384 atoms
 * 46/60/69/71/71/53, 42592 atoms 18/20/27/28/26/27. Throughput peaks at 4-8 threads and falls beyond
 * 8 (the threads' synchronisation and the serial force reduction grow with the thread count), and
 * half the cores leaves the rest to the page, the 3D view and the OS.
 */
export const autoThreads = (d: Pick<DeviceProfile, 'cores' | 'workers'>): number =>
  d.workers ? Math.max(1, Math.min(8, Math.floor(d.cores / 2))) : 1;

export interface AutoPlan {
  backend: 'cpu' | 'webgpu';
  threads: number;
  /** One line for the monitor: why this choice. */
  why: string;
}

/** Auto: a hardware GPU when there is one (the CPU threads still run what the GPU path cannot), else the CPU. */
export const autoPlan = (d: DeviceProfile): AutoPlan => {
  const threads = autoThreads(d);
  const cpu = `${threads} CPU thread${threads === 1 ? '' : 's'}`;
  if (d.gpu.kind === 'hardware') {
    return { backend: 'webgpu', threads, why: `hardware GPU${d.gpu.name ? ` (${d.gpu.name})` : ''} for lj/cut runs; ${cpu} for everything else` };
  }
  const gpuNote = d.gpu.kind === 'software'
    ? 'the only WebGPU adapter is a software one (slower than the CPU)'
    : 'no WebGPU adapter';
  return { backend: 'cpu', threads, why: `${cpu} of ${d.cores} cores; ${gpuNote}` };
};

/** Short hints about what limits this device, for the monitor. */
export const deviceHints = (d: DeviceProfile): string[] => {
  const out: string[] = [];
  if (!d.workers) out.push('Web Workers are unavailable: the engine runs on the page thread, on one core.');
  else if (!d.sharedMemory) out.push('The page is not cross-origin isolated, so shared-memory threads are off: only plain lj/cut runs on several threads.');
  if (d.cores <= 2) out.push(`This device reports ${d.cores} logical core${d.cores === 1 ? '' : 's'}: keep systems small (a few thousand atoms).`);
  if (d.memoryGB !== null && d.memoryGB <= 2) out.push(`This device reports about ${d.memoryGB} GB of memory: very large systems may not fit.`);
  if (d.gpu.kind === 'software') out.push('WebGPU is software-only here (SwiftShader); the CPU engine is faster, so Auto uses it.');
  return out;
};
