import { useCallback, useRef, useState } from 'react';
import { createHistory, HistoryState } from '../services/history';

/**
 * useState with bounded undo/redo. Persistence stays with the plain value
 * (callers persist `value` separately); history itself is in-memory.
 * `replace` sets the present WITHOUT pushing history — for UI-only state
 * changes (e.g. switching tabs in a multi-document workspace).
 */
export function useUndoableState<T>(
  initial: T | (() => T),
  capacity = 50,
): {
  value: T;
  set: (next: T | ((prev: T) => T)) => void;
  replace: (next: T | ((prev: T) => T)) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
} {
  const historyRef = useRef(createHistory<T>(
    typeof initial === 'function' ? (initial as () => T)() : initial,
    capacity,
  ));
  const [, forceRender] = useState(0);
  const snapshot: HistoryState<T> = historyRef.current.get();

  const commit = useCallback(() => forceRender(n => n + 1), []);

  const resolve = useCallback(
    (next: T | ((prev: T) => T)) =>
      typeof next === 'function' ? (next as (prev: T) => T)(historyRef.current.get().present) : next,
    [],
  );

  const set = useCallback(
    (next: T | ((prev: T) => T)) => {
      historyRef.current.push(resolve(next));
      commit();
    },
    [commit, resolve],
  );

  const replace = useCallback(
    (next: T | ((prev: T) => T)) => {
      historyRef.current.reset(resolve(next));
      commit();
    },
    [commit, resolve],
  );

  const undo = useCallback(() => {
    historyRef.current.undo();
    commit();
  }, [commit]);

  const redo = useCallback(() => {
    historyRef.current.redo();
    commit();
  }, [commit]);

  return {
    value: snapshot.present,
    set,
    replace,
    undo,
    redo,
    canUndo: snapshot.past.length > 0,
    canRedo: snapshot.future.length > 0,
  };
}
