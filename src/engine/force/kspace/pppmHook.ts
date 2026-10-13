/*
 * The slot of the lazily loaded GPL PPPM helper (PppmGplHook in pppm.ts). src/engine/lazyStyles.ts loads the
 * group 'gpl_pppm' (HOOKS 'pppm') before a command that names a PPPM style, and styles.ts loadStyleGroup stores the
 * helper here, so the core PPPM can reach it without importing src/engine-gpl (engineGplBoundary test).
 */
import { StyleError } from '../types';
import type { PppmGplHook } from './pppm';

let loaded: PppmGplHook | null = null;

/** Called once the 'gpl_pppm' group is loaded. */
export const setPppmGplHook = (hook: PppmGplHook): void => {
  loaded = hook;
};

/** The helper; throws naming the style when the session did not load it (run the command through the session). */
export const pppmGplHook = (): PppmGplHook => {
  if (!loaded) throw new StyleError('kspace_style pppm: the native PPPM helper is not loaded (run the command through the session)');
  return loaded;
};
