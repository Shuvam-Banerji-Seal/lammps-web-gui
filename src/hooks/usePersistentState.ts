import { useCallback, useEffect, useRef, useState } from 'react';
import { browserStore, loadJson, saveJson } from '../services/persistence';

/**
 * useState that transparently persists to localStorage (JSON).
 *
 * - Hydrates once on mount from `key`; falls back to `initial` when absent
 *   or corrupt.
 * - Writes are debounced to the next microtask-batched effect tick; storage
 *   failures are swallowed (persistence is best-effort by design).
 * - `revive` lets callers merge stored payloads onto fresh defaults so
 *   schema evolution never crashes on stale data.
 */
export function usePersistentState<T>(
  key: string,
  initial: T | (() => T),
  revive?: (raw: unknown) => T | null
): [T, React.Dispatch<React.SetStateAction<T>>] {
  const storeRef = useRef(browserStore());

  const [value, setValue] = useState<T>(() => {
    const initialResolved =
      typeof initial === 'function' ? (initial as () => T)() : initial;
    const loaded = loadJson<T>(storeRef.current, key, revive);
    return loaded ?? initialResolved;
  });

  useEffect(() => {
    saveJson(storeRef.current, key, value);
  }, [key, value]);

  const setBoth = useCallback<React.Dispatch<React.SetStateAction<T>>>(
    action => setValue(action),
    []
  );

  return [value, setBoth];
}
