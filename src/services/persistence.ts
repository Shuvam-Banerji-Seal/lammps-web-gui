/**
 * Tiny JSON persistence layer over a pluggable key/value store.
 *
 * Design goals:
 *  - Pure & unit-testable: storage is injected (localStorage in the app,
 *    an in-memory map in tests).
 *  - Never throws: quota errors, private-mode failures and corrupt payloads
 *    degrade to "no saved state" instead of breaking the app.
 *  - Forward compatible: callers pass a `revive` validator that merges a raw
 *    payload onto current defaults, so old/half-shaped data still loads.
 */

export interface JsonStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** Browser localStorage wrapped defensively; null when unavailable. */
export const browserStore = (): JsonStore | null => {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return null;
    return window.localStorage;
  } catch {
    return null; // privacy mode / disabled storage
  }
};

/** Read + parse + validate. Returns null for anything unusable. */
export const loadJson = <T>(
  store: JsonStore | null,
  key: string,
  revive?: (raw: unknown) => T | null
): T | null => {
  if (!store) return null;
  try {
    const text = store.getItem(key);
    if (!text) return null;
    const parsed = JSON.parse(text) as unknown;
    if (revive) return revive(parsed);
    return parsed as T;
  } catch {
    return null;
  }
};

/** Serialize + write. Returns false when the store refused (quota etc.). */
export const saveJson = <T>(store: JsonStore | null, key: string, value: T): boolean => {
  if (!store) return false;
  try {
    store.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
};

export const removeJson = (store: JsonStore | null, key: string): void => {
  if (!store) return;
  try {
    store.removeItem(key);
  } catch {
    /* non-fatal */
  }
};
