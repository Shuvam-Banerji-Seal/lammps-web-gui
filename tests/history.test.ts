import { describe, it, expect } from 'vitest';
import { createHistory } from '../src/services/history';

describe('bounded undo/redo history', () => {
  it('push/undo/redo round-trips immutably', () => {
    const h = createHistory({ steps: 1 });
    h.push({ steps: 2 });
    h.push({ steps: 3 });
    expect(h.get().present).toEqual({ steps: 3 });
    expect(h.canUndo()).toBe(true);
    expect(h.canRedo()).toBe(false);

    h.undo();
    expect(h.get().present).toEqual({ steps: 2 });
    expect(h.canRedo()).toBe(true);
    h.undo();
    expect(h.get().present).toEqual({ steps: 1 });
    expect(h.canUndo()).toBe(false);

    h.redo();
    h.redo();
    expect(h.get().present).toEqual({ steps: 3 });
    expect(h.canRedo()).toBe(false);
  });

  it('push clears the redo branch', () => {
    const h = createHistory('a');
    h.push('b');
    h.undo();
    expect(h.canRedo()).toBe(true);
    h.push('c');
    expect(h.canRedo()).toBe(false);
    expect(h.get().present).toBe('c');
    h.undo();
    expect(h.get().present).toBe('a');
  });

  it('bounds memory by dropping the oldest entries', () => {
    const h = createHistory(0, 3);
    for (const v of [1, 2, 3, 4, 5]) h.push(v);
    expect(h.get().present).toBe(5);
    expect(h.get().past).toEqual([2, 3, 4]);
    h.undo(); h.undo(); h.undo();
    expect(h.get().present).toBe(2);
    expect(h.canUndo()).toBe(false);
  });

  it('undo/redo at the boundaries are no-ops', () => {
    const h = createHistory('x');
    expect(h.undo().present).toBe('x');
    expect(h.redo().present).toBe('x');
  });

  it('reset clears everything', () => {
    const h = createHistory('a');
    h.push('b');
    h.reset('z');
    expect(h.get()).toEqual({ past: [], present: 'z', future: [] });
    expect(h.canUndo()).toBe(false);
  });
});
