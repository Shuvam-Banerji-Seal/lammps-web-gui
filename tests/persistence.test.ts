import { describe, it, expect } from 'vitest';
import {
  JsonStore,
  loadJson,
  saveJson,
  removeJson,
} from '../src/services/persistence';

const memoryStore = (): JsonStore & { data: Map<string, string> } => {
  const data = new Map<string, string>();
  return {
    data,
    getItem: k => (data.has(k) ? (data.get(k) as string) : null),
    setItem: (k, v) => void data.set(k, v),
    removeItem: k => void data.delete(k),
  };
};

describe('persistence layer', () => {
  it('round-trips JSON values', () => {
    const store = memoryStore();
    expect(saveJson(store, 'k', { a: 1, b: ['x'] })).toBe(true);
    expect(loadJson(store, 'k')).toEqual({ a: 1, b: ['x'] });
  });

  it('returns null for missing keys and a null store', () => {
    const store = memoryStore();
    expect(loadJson(store, 'missing')).toBeNull();
    expect(loadJson(null, 'k')).toBeNull();
    expect(saveJson(null, 'k', {})).toBe(false);
  });

  it('returns null on corrupt payloads instead of throwing', () => {
    const store = memoryStore();
    store.setItem('bad', '{not json!!');
    expect(loadJson(store, 'bad')).toBeNull();
  });

  it('revive validator can reshape or reject stored data', () => {
    const store = memoryStore();
    saveJson(store, 'opts', { presetId: 'most', jobs: 4 });
    const revive = (raw: unknown) => {
      if (typeof raw !== 'object' || raw === null) return null;
      const r = raw as Record<string, unknown>;
      return { presetId: typeof r.presetId === 'string' ? r.presetId : '', jobs: Number(r.jobs) || 8 };
    };
    expect(loadJson(store, 'opts', revive)).toEqual({ presetId: 'most', jobs: 4 });
    saveJson(store, 'garbage', 42);
    // revive rejects non-objects -> loadJson yields null (caller falls back)
    expect(loadJson(store, 'garbage', revive)).toBeNull();
    saveJson(store, 'nullish', null);
    expect(loadJson(store, 'nullish', revive)).toBeNull();
  });

  it('swallows storage write failures (quota)', () => {
    const failing: JsonStore = {
      getItem: () => null,
      setItem: () => {
        throw new Error('QuotaExceededError');
      },
      removeItem: () => {},
    };
    expect(saveJson(failing, 'k', { x: 1 })).toBe(false);
  });

  it('removeJson is safe on any store state', () => {
    const store = memoryStore();
    saveJson(store, 'k', 1);
    removeJson(store, 'k');
    expect(loadJson(store, 'k')).toBeNull();
    expect(() => removeJson(null, 'k')).not.toThrow();
  });
});
