import type { EngineEvent } from './types';
import type { AutoPlan, DeviceProfile } from './device';

/** Messages between the notebook UI and the engine worker. */

/** 'auto' lets the engine choose from the device (device.ts autoPlan). */
export type BackendChoice = 'auto' | 'cpu' | 'webgpu';

export type ToEngine =
  /** (Re)creates the session; drops every atom, fix and variable. threads 0 = the engine's choice (device.ts autoThreads). */
  | { type: 'reset'; backend: BackendChoice; threads: number; frameEvery: number }
  /** Switches the force backend of the current session, keeping its system. */
  | { type: 'backend'; backend: BackendChoice; threads: number }
  /** Runs one cell's text in the current session. */
  | { type: 'exec'; id: number; text: string; firstLine: number }
  /** Stops the running `run` after its current step. */
  | { type: 'cancel' }
  /** Adds or replaces a file the session can read (data, include, potential files); null text removes it. */
  | { type: 'file'; name: string; text: string | null };

export type FromEngine =
  | {
    type: 'ready';
    /** Label of the backend actually in use, e.g. "CPU · fp64 · 8 threads". */
    backend: string;
    kind: 'cpu' | 'webgpu';
    webgpuAvailable: boolean;
    /** Logical cores the browser reports (navigator.hardwareConcurrency). */
    cores: number;
    /** What the engine found on this device (cores, memory, shared memory, WebGPU adapter). */
    device: DeviceProfile;
    /** The plan behind backend 'auto' or threads 0 (what was chosen and why). */
    auto?: AutoPlan;
    /** CPU threads the backend runs (pair term; for a GPU backend, the runs the GPU path cannot take). */
    threads: number;
    note?: string;
    /** Commands the engine accepts (for the notebook's help panel). */
    commands: string[];
    /** Styles per style command (pair_style, fix, compute, ...), for the help panel. */
    styles: Record<string, string[]>;
  }
  | { type: 'event'; id: number; event: EngineEvent }
  | { type: 'file'; id: number; name: string; text: string; append: boolean }
  | { type: 'finished'; id: number; ok: boolean; cancelled: boolean };
