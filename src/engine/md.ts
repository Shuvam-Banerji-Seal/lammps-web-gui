import type { ForceBackend, ForceResult, PairTable, SimBox, SimState, ThermoKeyword, ThermoRow, UnitSystem } from './types';
import type { Fix, RunContext } from './integrate';
import { wrapPositions } from './integrate';
import { thermoRow } from './observables';

/*
 * The run loop. docs.lammps.org/thermo.html: thermo output happens "on
 * timesteps that are a multiple of N and at the beginning and end of a
 * simulation"; "thermo 0" (the default) prints only the beginning and end.
 * docs.lammps.org/run.html: "A value of N = 0 is acceptable; only the
 * thermodynamics of the system are computed and printed without taking a
 * timestep."
 */

export { emptyState, addAtoms } from './atoms';

/**
 * A backend that can run whole velocity-Verlet steps itself (the WebGPU
 * backend, gpu/resident.ts), reading state back only when asked to.
 */
export interface ResidentBackend extends ForceBackend {
  canAdvance(state: SimState, table: PairTable): boolean;
  advance(state: SimState, table: PairTable, nsteps: number, opts: { enforce2d: boolean }): Promise<ForceResult>;
}

export const isResident = (b: ForceBackend): b is ResidentBackend =>
  typeof (b as Partial<ResidentBackend>).advance === 'function' && typeof (b as Partial<ResidentBackend>).canAdvance === 'function';

/** Longest stretch run on the GPU without coming back (keeps Stop responsive). */
export const MAX_CHUNK = 200;

export interface RunOptions {
  /** thermo N; 0 = only first and last step. */
  thermoEvery: number;
  keywords: readonly ThermoKeyword[];
  norm?: boolean;
  onThermo?: (row: ThermoRow) => void;
  /** Called once after the setup force evaluation (e.g. the step-0 dump). */
  onSetup?: (s: SimState) => void;
  /** Called after every step (e.g. dumps); return false to stop early. */
  onStep?: (s: SimState) => boolean | void;
  /** Lets a caller yield to the event loop every `yieldEvery` steps. */
  yieldEvery?: number;
  /**
   * Steps at which the caller needs the state on the host (dumps, frames).
   * With a resident backend only those steps, thermo steps and the last step
   * come back from the GPU, and onStep is called only for them.
   */
  hostStep?: (step: number) => boolean;
}

/** Forces + fix setup at the start of a run; returns the step-0 forces. */
export const setupRun = async (
  s: SimState, table: PairTable, backend: ForceBackend, fixes: readonly Fix[], ctx: RunContext,
): Promise<ForceResult> => {
  wrapPositions(s);
  const res = await backend.compute(s, table);
  for (const fx of fixes) fx.setup?.(s, ctx);
  return res;
};

/** Runs n steps of velocity Verlet with the given fixes; returns the final forces. */
export const run = async (
  s: SimState, table: PairTable, backend: ForceBackend, fixes: readonly Fix[], nsteps: number, opts: RunOptions,
): Promise<ForceResult> => {
  const ctx: RunContext = { runStart: s.step, runStop: s.step + nsteps };
  let res = await setupRun(s, table, backend, fixes, ctx);
  opts.onSetup?.(s);
  const emit = () => opts.onThermo?.(thermoRow(s, opts.keywords, res, { norm: opts.norm, runStart: ctx.runStart }));
  emit();
  // GPU-resident path: plain NVE (optionally 2d) on a backend that can step itself
  const nve = fixes.filter((f) => f.style === 'nve').length === 1
    && fixes.every((f) => f.style === 'nve' || f.style === 'enforce2d');
  if (nsteps > 0 && nve && isResident(backend) && backend.canAdvance(s, table)) {
    const enforce2d = fixes.some((f) => f.style === 'enforce2d');
    const due = (step: number) => step === ctx.runStop
      || (opts.thermoEvery > 0 && step % opts.thermoEvery === 0)
      || (opts.hostStep?.(step) ?? false);
    while (s.step < ctx.runStop) {
      let m = 1;
      while (m < MAX_CHUNK && !due(s.step + m)) m++;
      res = await backend.advance(s, table, m, { enforce2d });
      s.step += m;
      if (s.step === ctx.runStop || (opts.thermoEvery > 0 && s.step % opts.thermoEvery === 0)) emit();
      if (opts.onStep?.(s) === false) break;
      await new Promise((r) => setTimeout(r, 0));
    }
    return res;
  }
  for (let k = 0; k < nsteps; k++) {
    for (const fx of fixes) fx.initialIntegrate?.(s, ctx);
    wrapPositions(s);
    res = await backend.compute(s, table);
    for (const fx of fixes) fx.postForce?.(s, ctx);
    for (const fx of fixes) fx.finalIntegrate?.(s, ctx);
    s.step++;
    for (const fx of fixes) fx.endOfStep?.(s, ctx);
    const last = k === nsteps - 1;
    if (last || (opts.thermoEvery > 0 && s.step % opts.thermoEvery === 0)) emit();
    if (opts.onStep?.(s) === false) break;
    if (opts.yieldEvery && (k + 1) % opts.yieldEvery === 0) await new Promise((r) => setTimeout(r, 0));
  }
  return res;
};
