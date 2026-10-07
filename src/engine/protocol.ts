import type { EngineEvent } from './types';

/** Messages between the notebook UI and the engine worker. */

export type BackendChoice = 'cpu' | 'webgpu';

export type ToEngine =
  /** (Re)creates the session; drops every atom, fix and variable. */
  | { type: 'reset'; backend: BackendChoice; threads: number; frameEvery: number }
  /** Switches the force backend of the current session, keeping its system. */
  | { type: 'backend'; backend: BackendChoice; threads: number }
  /** Runs one cell's text in the current session. */
  | { type: 'exec'; id: number; text: string; firstLine: number }
  /** Stops the running `run` after its current step. */
  | { type: 'cancel' };

export type FromEngine =
  | {
    type: 'ready';
    /** Label of the backend actually in use, e.g. "CPU · fp64 · 8 threads". */
    backend: string;
    kind: 'cpu' | 'webgpu';
    webgpuAvailable: boolean;
    /** Logical cores the browser reports (navigator.hardwareConcurrency). */
    cores: number;
    note?: string;
  }
  | { type: 'event'; id: number; event: EngineEvent }
  | { type: 'file'; id: number; name: string; text: string; append: boolean }
  | { type: 'finished'; id: number; ok: boolean; cancelled: boolean };
