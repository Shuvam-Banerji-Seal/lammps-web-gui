import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Session } from '../session';

/** What a command handler sees. */
export interface Ctx {
  sys: System;
  session: Session;
  frame: unknown;
  fail(message: string): never;
}

export type Handler = (ctx: Ctx, args: string[]) => void | Promise<void>;

export const num = (w: string | undefined, what: string): number => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`expected a number for ${what}, got '${w}'`);
  return v;
};

export const int = (w: string | undefined, what: string): number => {
  const v = num(w, what);
  if (!Number.isInteger(v)) throw new StyleError(`expected an integer for ${what}, got '${w}'`);
  return v;
};

export const yesno = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`${what} must be yes or no, got '${w ?? ''}'`);
};

/** A number, or v_name evaluated as an equal-style variable now. */
export const numOrVar = (sys: System, w: string | undefined, what: string): number => {
  if (w?.startsWith('v_')) return sys.equalVariable(w.slice(2));
  return num(w, what);
};

/** Lattice spacings for "units lattice" (error if no lattice was defined). */
export const latticeScale = (sys: System, units: string, what: string): [number, number, number] => {
  if (units === 'box') return [1, 1, 1];
  if (units !== 'lattice') throw new StyleError(`${what}: units must be lattice or box, got '${units}'`);
  // "By default, a "lattice none 1.0" is defined" (lattice.html)
  return sys.lattice ? [...sys.lattice.spacing] as [number, number, number] : [1, 1, 1];
};

/** Splits key value pairs, requiring each key to be in `allowed` (value counts per key). */
export const keywords = (words: string[], allowed: Record<string, number>, what: string): Map<string, string[]> => {
  const out = new Map<string, string[]>();
  for (let k = 0; k < words.length;) {
    const key = words[k];
    const n = allowed[key];
    if (n === undefined) throw new StyleError(`${what}: unknown keyword '${key}' (supported: ${Object.keys(allowed).join(', ') || 'none'})`);
    const vals = words.slice(k + 1, k + 1 + n);
    if (vals.length < n) throw new StyleError(`${what}: keyword '${key}' needs ${n} value(s)`);
    out.set(key, vals);
    k += 1 + n;
  }
  return out;
};
