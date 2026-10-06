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

export const emptyState = (units: UnitSystem, dimension: 2 | 3, box: SimBox, ntypes: number): SimState => ({
  n: 0,
  dimension,
  box,
  units,
  ntypes,
  type: new Int32Array(0),
  massByType: new Float64Array(ntypes + 1).fill(Number.NaN),
  x: new Float64Array(0),
  v: new Float64Array(0),
  f: new Float64Array(0),
  image: new Int32Array(0),
  id: new Int32Array(0),
  step: 0,
  dt: units.dt,
});

/** Appends atoms (flat 3N positions, already inside the box) of one type. */
export const addAtoms = (s: SimState, positions: Float64Array, type: number): number => {
  const add = positions.length / 3;
  if (add === 0) return 0;
  const n = s.n + add;
  const grow3 = (a: Float64Array) => { const b = new Float64Array(3 * n); b.set(a); return b; };
  const x = grow3(s.x);
  x.set(positions, 3 * s.n);
  const t = new Int32Array(n); t.set(s.type); t.fill(type, s.n);
  const id = new Int32Array(n); id.set(s.id);
  let next = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > next) next = s.id[i];
  for (let i = s.n; i < n; i++) id[i] = ++next;
  const image = new Int32Array(3 * n); image.set(s.image);
  s.x = x;
  s.v = grow3(s.v);
  s.f = grow3(s.f);
  s.image = image;
  s.type = t;
  s.id = id;
  s.n = n;
  return add;
};

export interface RunOptions {
  /** thermo N; 0 = only first and last step. */
  thermoEvery: number;
  keywords: readonly ThermoKeyword[];
  norm?: boolean;
  onThermo?: (row: ThermoRow) => void;
  /** Called after every step (e.g. dumps); return false to stop early. */
  onStep?: (s: SimState) => boolean | void;
  /** Lets a caller yield to the event loop every `yieldEvery` steps. */
  yieldEvery?: number;
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
  const emit = () => opts.onThermo?.(thermoRow(s, opts.keywords, res, { norm: opts.norm, runStart: ctx.runStart }));
  emit();
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
