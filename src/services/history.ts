/**
 * Bounded undo/redo history for immutable state values.
 * Pure data structure — no React dependency — so it is trivially testable;
 * `useUndoableState` wraps it for components.
 *
 * Semantics:
 *  - `push` commits a new present, clearing the redo stack.
 *  - `undo`/`redo` move the present pointer; capacity is bounded by dropping
 *    the OLDEST entries (present is always kept).
 *  - `canUndo`/`canRedo` drive UI enablement.
 */

export interface HistoryState<T> {
  past: T[];
  present: T;
  future: T[];
}

export interface HistoryApi<T> {
  get(): Readonly<HistoryState<T>>;
  push(next: T): HistoryState<T>;
  undo(): HistoryState<T>;
  redo(): HistoryState<T>;
  canUndo(): boolean;
  canRedo(): boolean;
  reset(initial: T): HistoryState<T>;
}

export const createHistory = <T>(initial: T, capacity = 50): HistoryApi<T> => {
  let state: HistoryState<T> = { past: [], present: initial, future: [] };

  const trimPast = (past: T[]): T[] =>
    past.length > capacity ? past.slice(past.length - capacity) : past;

  return {
    get: () => state,
    push: (next: T) => {
      state = {
        past: trimPast([...state.past, state.present]),
        present: next,
        future: [],
      };
      return state;
    },
    undo: () => {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      state = {
        past: state.past.slice(0, -1),
        present: previous,
        future: [state.present, ...state.future],
      };
      return state;
    },
    redo: () => {
      if (state.future.length === 0) return state;
      const [next, ...rest] = state.future;
      state = {
        past: [...state.past, state.present],
        present: next,
        future: rest,
      };
      return state;
    },
    canUndo: () => state.past.length > 0,
    canRedo: () => state.future.length > 0,
    reset: (initial: T) => {
      state = { past: [], present: initial, future: [] };
      return state;
    },
  };
};
