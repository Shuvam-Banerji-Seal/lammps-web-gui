import { StyleError } from '../force/types';
import type { System } from '../system';
import type { Session } from '../session';
import type { RunHooks } from '../run/verlet';
import type { MinHooks } from '../run/min';
import type { NebHost } from './neb';
import type { Fix } from '../fix/fix';

/**
 * The run-loop helpers a command handler can call. Lazily loaded commands (src/engine-gpl) take them from the
 * context instead of importing the run modules, so their chunks do not pull the run machinery into shared chunks.
 */
export interface RunApi {
  emitThermoRow(sys: System): number;
  writeDumps(sys: System, firstOfRun: boolean): boolean;
  emitFrame(sys: System): void;
  ensureThermoComputes(sys: System): void;
  markRunSetup(sys: System): void;
  startRestarts(sys: System): void;
  writeRestarts(sys: System, step: number, force?: boolean): void;
  initRun(sys: System, opts?: { verlet?: boolean }): void;
  runVerlet(sys: System, nsteps: number, hooks: RunHooks): Promise<number>;
  /** minimize (run/min.ts): the quench of prd and tad; the minimizer of the session's min_style */
  minimize(sys: System, p: { etol: number; ftol: number; maxiter: number; maxeval: number }, hooks: MinHooks): Promise<{ reason: string }>;
  /** the neb command's calculation on the replicas (commands/neb.ts runNeb); returns the barrier EBF (native neb->ebf) */
  runNeb(host: NebHost, a: string[]): Promise<number>;
  /** fix neb Kspring of a replica, as tad adds one to every replica (fix/neb.ts) */
  makeNebFix(sys: System, id: string, group: string, args: string[]): Fix;
  /** a new minimizer of a style with the default min_modify settings, as min_style and tad make one (run.ts) */
  createMinimize(sys: System, style: string): void;
}

/** What a command handler sees. */
export interface Ctx {
  sys: System;
  session: Session;
  frame: unknown;
  fail(message: string): never;
  run: RunApi;
  /** The command's words as typed, before an accelerator alias was replaced by its base style (Session.baseStyleNames). */
  typed?: string[];
}

export type Handler = (ctx: Ctx, args: string[]) => void | Promise<void>;

/**
 * A command that initializes the whole system as a run does (native lmp->init(), at b332f28: balance, create_bonds,
 * delete_bonds, delete_atoms overlap, set of bond, angle, dihedral or improper types); afterwards variable formulas may
 * read the computes (System.markComputesInitialized). `when` picks the forms of the command that do.
 */
export const initializing = (h: Handler, when: (a: string[]) => boolean = () => true): Handler => (ctx, a) => {
  const r = h(ctx, a);
  if (!when(a)) return r;
  if (r instanceof Promise) return r.then(() => ctx.sys.markComputesInitialized());
  ctx.sys.markComputesInitialized();
};

/**
 * A number of a command's arguments, read as native LAMMPS reads one (utils::numeric, src/utils.cpp at b332f28: its
 * messages do not name the argument; `what` documents the call site).
 */
export const num = (w: string | undefined, _what: string): number => numeric(w);

/** An integer argument, read as native LAMMPS reads one (utils::inumeric). */
export const int = (w: string | undefined, _what: string): number => inumeric(w);

export const yesno = (w: string | undefined, what: string): boolean => {
  if (w === 'yes') return true;
  if (w === 'no') return false;
  throw new StyleError(`${what} must be yes or no, got '${w ?? ''}'`);
};

// the native-text argument checks live with parseNum (force/util.ts), where the styles reach them too
import { inumeric, numeric } from '../force/util';
export { missingArgs, logical, numeric, inumeric, bnumeric, bounds } from '../force/util';

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
