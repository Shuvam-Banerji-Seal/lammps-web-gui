import type { System } from '../system';
import type { Compute } from '../compute/compute';
import { COMPUTE_STYLES } from '../styles';
import { StyleError } from '../force/types';

/*
 * Shared helpers for thermostat and barostat fixes.
 *
 * Ramps: docs.lammps.org/run.html — "The start or stop keywords can be used
 * if multiple runs are being performed and you want a fix command that
 * changes some value over time (e.g. temperature) to make the change across
 * the entire set of runs and not just a single run." "The option defaults
 * are start = the current timestep, stop = current timestep + N". A fix's
 * target is start + (stop - start) * (step - beginstep) / (endstep -
 * beginstep).
 */

export const rampFraction = (sys: System): number => {
  const r = sys.run;
  const span = r.endStep - r.beginStep;
  return span > 0 ? (sys.state.step - r.beginStep) / span : 0;
};

export const ramp = (sys: System, start: number, stop: number): number => start + (stop - start) * rampFraction(sys);

/** A number or v_name (equal-style variable evaluated now). */
export type NumOrVar = number | { variable: string };

export const parseNumOrVar = (w: string | undefined, what: string): NumOrVar => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  if (w.startsWith('v_')) return { variable: w.slice(2) };
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`expected a number for ${what}, got '${w}'`);
  return v;
};

export const valueOf = (sys: System, v: NumOrVar): number => (typeof v === 'number' ? v : sys.equalVariable(v.variable));

/**
 * Creates the fix's own compute (e.g. "compute fix-ID_temp group-ID temp"),
 * as the fix pages document, and registers it so users can reference it.
 */
export const ownCompute = (sys: System, id: string, group: string, style: string, args: string[]): Compute => {
  const old = sys.computes.findIndex((c) => c.id === id);
  if (old >= 0) sys.computes.splice(old, 1);
  const c = COMPUTE_STYLES[style](sys, id, group, args);
  sys.computes.push(c);
  return c;
};

export const removeCompute = (sys: System, id: string): void => {
  const k = sys.computes.findIndex((c) => c.id === id);
  if (k >= 0) sys.computes.splice(k, 1);
};

/** Current temperature from a compute (fresh value). */
export const temperatureOf = (sys: System, c: Compute): number => {
  sys.refreshComputes();
  return c.scalarValue();
};
