/*
 * The slots of the lazily loaded helpers (HOOKS entries of src/engine/lazyStyles.ts). A lazy group's HOOKS names are
 * command words; styles.ts loadStyleGroup stores each exported HOOKS helper under its word here, and core code reads it
 * with gplHook(word). The core never imports src/engine-gpl (tests/engineGplBoundary.test.ts), and a caller outside
 * Session.dispatch gets a StyleError naming the word, never a different algorithm.
 */
import { StyleError } from './force/types';

const slots = new Map<string, unknown>();

/** Called once per helper when its lazy group is loaded (styles.ts loadStyleGroup). */
export const setGplHook = (word: string, hook: unknown): void => {
  slots.set(word, hook);
};

/** The helper stored under a command word; throws naming the word when the session did not load it. */
export const gplHook = <T>(word: string): T => {
  if (!slots.has(word)) throw new StyleError(`${word}: the lazily loaded helper for this command is not loaded (run the command through the session)`);
  return slots.get(word) as T;
};
