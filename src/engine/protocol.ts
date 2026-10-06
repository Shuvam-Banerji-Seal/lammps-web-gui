import type { EngineEvent } from './types';

/** Messages between the notebook UI and the engine worker. */

export type BackendChoice = 'cpu' | 'webgpu';

export type ToEngine =
  /** (Re)creates the session; drops every atom, fix and variable. */
  | { type: 'reset'; backend: BackendChoice; frameEvery: number }
  /** Runs one cell's text in the current session. */
  | { type: 'exec'; id: number; text: string; firstLine: number }
  /** Stops the running `run` after its current step. */
  | { type: 'cancel' };

export type FromEngine =
  | { type: 'ready'; backend: string; webgpuAvailable: boolean; note?: string }
  | { type: 'event'; id: number; event: EngineEvent }
  | { type: 'file'; id: number; name: string; text: string; append: boolean }
  | { type: 'finished'; id: number; ok: boolean; cancelled: boolean };
