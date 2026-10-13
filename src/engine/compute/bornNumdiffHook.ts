/*
 * The slot of the lazily loaded numdiff helper of compute born/matrix (BornNumdiffHook in born_matrix.ts). The helper
 * is the GPL port in src/engine-gpl/extra_compute/born_matrix.ts; src/engine/lazyStyles.ts loads the group
 * 'gpl_born_matrix' (HOOKS 'born/matrix') before a command that names born/matrix, and styles.ts loadStyleGroup stores
 * the helper here, so the core compute can reach it without importing src/engine-gpl (engineGplBoundary test).
 */
import { StyleError } from '../force/types';
import type { BornNumdiffHook } from './born_matrix';

let loaded: BornNumdiffHook | null = null;

/** Called once the 'gpl_born_matrix' group is loaded. */
export const setBornNumdiffHook = (hook: BornNumdiffHook): void => {
  loaded = hook;
};

/** The helper; throws naming the style when the session did not load it (run the command through the session). */
export const bornNumdiffHook = (): BornNumdiffHook => {
  if (!loaded) throw new StyleError('compute born/matrix numdiff: the numdiff helper is not loaded (run the command through the session)');
  return loaded;
};
