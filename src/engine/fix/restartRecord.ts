import type { System } from '../system';
import { StyleError } from '../force/types';
import type { PropertyAtomRestart } from './property_atom';
import type { CmapRestart } from './cmap';
import type { NHRestart } from './nh';

/**
 * One fix's record in the browser restart file (output/restart.ts). A record holds what the fix's class
 * stores: the property/atom values, the CMAP cross-terms, the NH chain and barostat variables, or a generic
 * `state` object for the other fixes that keep state natively (restartState / restoreFromRestart).
 */
export type FixRecord = PropertyAtomRestart & { style?: string; cmap?: CmapRestart; nh?: NHRestart; state?: unknown; stateNotStored?: boolean };

/**
 * The record of a fix re-specified after read_restart. fix_property_atom.html: "LAMMPS will only check whether a
 * fix is of the same style and has the same fix ID and in case of a match will then try to initialize the fix
 * with the data stored in the binary restart file." Modify::add_fix (src/modify.cpp at b332f28) matches the ID
 * and the style the same way for the per-atom list, and the global list strips the accelerator suffix from the
 * style first. A record of another style stays in the map (native warns about it and leaves it unused). A
 * record written without a style (an older file) matches by ID alone.
 */
export const takeFixRecord = (sys: System, id: string, style: string): FixRecord | undefined => {
  const rec = sys.pendingFixData.get(id);
  if (!rec || (rec.style !== undefined && rec.style !== style)) return undefined;
  sys.pendingFixData.delete(id);
  return rec;
};

/**
 * A record written for a fix whose native state the browser restart file does not hold (FIX_STATE_NOT_STORED in
 * output/restart.ts). Native restores the saved state when a fix with the same ID and style is defined after
 * read_restart (Modify::add_fix); the engine cannot, so that definition is refused by name here. Another ID or
 * another style is not affected, as in native.
 */
export const checkFixStateNotStored = (sys: System, id: string, style: string): void => {
  const rec = sys.pendingFixData.get(id);
  if (!rec || !rec.stateNotStored || rec.style !== style) return;
  throw new StyleError(`fix ${id} (${style}): native LAMMPS would restore its saved state, which the browser restart file does not hold; re-create the system instead`);
};

/**
 * A RanMars stream (the 105 values of RanMars::get_state) and the energy a thermostat has exchanged: fix temp/csvr and
 * fix temp/csld write_restart store the energy and the stream of the one processor (random_mars.cpp).
 */
export interface MarsEnergyRestart { energy: number; rng: number[] }

/** A RanMars stream alone: fix gjf and the fix brownian family store nprocs and the stream (write_restart). */
export interface MarsRestart { rng: number[] }

/** A fix whose native class keeps global state in the restart file (fix.h restart_global). */
export interface RestartableFix {
  /** The state the restart file stores, as plain JSON data (the fix's write_restart list). */
  restartState(): unknown;
}

export const isRestartableFix = (f: unknown): f is RestartableFix =>
  typeof f === 'object' && f !== null && typeof (f as Partial<RestartableFix>).restartState === 'function';
