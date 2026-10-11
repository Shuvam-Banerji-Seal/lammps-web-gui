import { describe, expect, it } from 'vitest';
import {
  createIndexedDbBackend, createMemoryBackend, createNotebookLibrary, fingerprint, readLegacyCells, type LibraryBackend,
} from '../src/components/workbench/notebook/notebookLibrary';
import type { Cell } from '../src/components/workbench/notebook/cells';

const cellsA: Cell[] = [{ id: 'a1', text: 'units lj' }, { id: 'a2', kind: 'markdown', text: '# md' }];

const counter = () => {
  let n = 0;
  return () => `id${++n}`;
};
const clock = () => {
  let t = Date.parse('2026-10-11T00:00:00.000Z');
  return () => new Date((t += 1000)).toISOString();
};

/** A backend whose every call throws, as a broken storage does. */
const brokenBackend: LibraryBackend = {
  kind: 'indexeddb',
  all: async () => { throw new Error('boom'); },
  get: async () => { throw new Error('boom'); },
  put: async () => { throw new Error('boom'); },
  remove: async () => { throw new Error('boom'); },
  getMeta: async () => { throw new Error('boom'); },
  setMeta: async () => { throw new Error('boom'); },
};

/** A minimal IndexedDB double: object stores with keyPath or out-of-line keys, enough for the backend. */
const fakeIndexedDb = () => {
  type Store = Map<string, unknown>;
  const dbs = new Map<string, Map<string, Store>>();
  const factory = {
    open(name: string) {
      const req: { result?: unknown; onsuccess?: () => void; onerror?: () => void; onupgradeneeded?: () => void; onblocked?: () => void; error?: unknown } = {};
      queueMicrotask(() => {
        let db = dbs.get(name);
        const fresh = !db;
        if (!db) { db = new Map(); dbs.set(name, db); }
        const stores = db;
        const mkDb = {
          objectStoreNames: { contains: (s: string) => stores.has(s) },
          createObjectStore: (s: string) => { stores.set(s, new Map()); },
          transaction: (s: string) => ({
            objectStore: () => {
              const st = stores.get(s)!;
              const reqOf = (fn: () => unknown) => {
                const r: { result?: unknown; onsuccess?: () => void; onerror?: () => void } = {};
                queueMicrotask(() => {
                  try { r.result = fn(); r.onsuccess?.(); } catch (e) { r.onerror?.(); void e; }
                });
                return r;
              };
              return {
                getAll: () => reqOf(() => [...st.values()]),
                get: (k: string) => reqOf(() => st.get(k)),
                put: (v: { id?: string }, key?: string) => reqOf(() => { st.set(key ?? v.id!, v); return key ?? v.id; }),
                delete: (k: string) => reqOf(() => { st.delete(k); }),
              };
            },
          }),
        };
        req.result = mkDb;
        if (fresh) req.onupgradeneeded?.();
        req.onsuccess?.();
      });
      return req;
    },
  };
  return { factory: factory as unknown as IDBFactory, dbs };
};

describe('notebook library (memory backend)', () => {
  it('creates, lists (newest first), updates, renames by update, duplicates and removes', async () => {
    const lib = createNotebookLibrary(createMemoryBackend(), clock(), counter());
    const a = await lib.create('First', cellsA);
    const b = await lib.create('Second', []);
    expect((await lib.list()).map((e) => e.name)).toEqual(['Second', 'First']);
    expect((await lib.list())[1].cellCount).toBe(2);

    const up = await lib.update(a.id, { cells: [{ id: 'z', text: 'run 1' }] });
    expect(up?.cells).toEqual([{ id: 'z', text: 'run 1' }]);
    expect(up?.created).toBe(a.created);
    expect(up && up.modified > a.modified).toBe(true);
    expect((await lib.update(a.id, { name: '  ' }))?.name).toBe('Untitled notebook');
    await lib.update(a.id, { name: 'First' });
    expect(await lib.update('missing', { name: 'x' })).toBeNull();

    const copy = await lib.duplicate(a.id, 'First (copy)');
    expect(copy?.name).toBe('First (copy)');
    expect(copy?.cells).toEqual([{ id: 'z', text: 'run 1' }]);
    await lib.remove(b.id);
    expect((await lib.list()).map((e) => e.name)).toEqual(['First (copy)', 'First']);
    expect(await lib.get(b.id)).toBeUndefined();
  });

  it('returns copies, so a caller cannot change a stored notebook by mutating it', async () => {
    const lib = createNotebookLibrary(createMemoryBackend(), clock(), counter());
    const a = await lib.create('N', cellsA);
    a.cells[0].text = 'changed';
    expect((await lib.get(a.id))?.cells[0].text).toBe('units lj');
  });

  it('migrates the earlier single notebook once, into the library', async () => {
    const lib = createNotebookLibrary(createMemoryBackend(), clock(), counter());
    const rec = await lib.migrateLegacy(cellsA);
    expect(rec?.name).toBe('Untitled notebook');
    expect(rec?.cells).toEqual(cellsA);
    expect(await lib.migrateLegacy(cellsA)).toBeNull();
    expect(await lib.migrateLegacy(null)).toBeNull();
    expect(await lib.list()).toHaveLength(1);
  });

  it('keeps the working copy across a restart (same backend)', async () => {
    const backend = createMemoryBackend();
    const lib = createNotebookLibrary(backend, clock(), counter());
    const saved = await lib.create('Kept', cellsA);
    await lib.saveWorking({ baseId: saved.id, name: 'Kept', cells: [{ id: 'w', text: 'run 2' }], savedFingerprint: fingerprint('Kept', cellsA) });
    const again = createNotebookLibrary(backend, clock(), counter());
    const w = await again.loadWorking();
    expect(w?.baseId).toBe(saved.id);
    expect(w?.cells).toEqual([{ id: 'w', text: 'run 2' }]);
    expect(w?.savedFingerprint).toBe(fingerprint('Kept', cellsA));
  });

  it('ignores a stored working copy that is malformed', async () => {
    const backend = createMemoryBackend();
    await backend.setMeta('working', { name: 3, cells: 'x' });
    expect(await createNotebookLibrary(backend).loadWorking()).toBeNull();
  });

  it('fingerprints the name and the cells, so an edit or a rename makes the notebook unsaved', () => {
    const base = fingerprint('N', cellsA);
    expect(fingerprint('N', cellsA)).toBe(base);
    expect(fingerprint('M', cellsA)).not.toBe(base);
    expect(fingerprint('N', [{ id: 'a1', text: 'units lj!' }, cellsA[1]])).not.toBe(base);
    expect(fingerprint('N', [{ id: 'a1', text: 'units lj', kind: 'code' }, cellsA[1]])).toBe(base);
  });

  it('reads the earlier version storage, and gives null for missing or broken values', () => {
    const store = (v: string | null) => ({ getItem: () => v });
    expect(readLegacyCells(store(JSON.stringify(cellsA)))).toEqual(cellsA);
    expect(readLegacyCells(store(null))).toBeNull();
    expect(readLegacyCells(store('{not json'))).toBeNull();
    expect(readLegacyCells({ getItem: () => { throw new Error('blocked'); } })).toBeNull();
    expect(readLegacyCells(null)).toBeNull();
  });
});

describe('notebook library (IndexedDB backend)', () => {
  it('stores notebooks and meta values in the database and reads them back', async () => {
    const { factory, dbs } = fakeIndexedDb();
    const backend = createIndexedDbBackend(factory);
    expect(backend.kind).toBe('indexeddb');
    const lib = createNotebookLibrary(backend, clock(), counter());
    const rec = await lib.create('In IDB', cellsA);
    await lib.saveWorking({ baseId: rec.id, name: 'In IDB', cells: cellsA, savedFingerprint: null });
    expect(dbs.get('lammps-notebook-library')?.get('notebooks')?.size).toBe(1);
    const again = createNotebookLibrary(createIndexedDbBackend(factory), clock(), counter());
    expect((await again.list())[0].name).toBe('In IDB');
    expect((await again.loadWorking())?.baseId).toBe(rec.id);
  });

  it('falls back to memory when IndexedDB is missing, and says so', async () => {
    const backend = createIndexedDbBackend(null);
    expect(backend.kind).toBe('memory');
    const lib = createNotebookLibrary(backend, clock(), counter());
    await lib.create('Private', cellsA);
    expect((await lib.list())[0].name).toBe('Private');
  });

  it('falls back to memory when an operation fails, and keeps working', async () => {
    const broken = {
      open() {
        const req: { result?: unknown; onsuccess?: () => void; onerror?: () => void; error?: unknown } = {};
        queueMicrotask(() => { req.onerror?.(); });
        return req;
      },
    } as unknown as IDBFactory;
    const backend = createIndexedDbBackend(broken);
    const lib = createNotebookLibrary(backend, clock(), counter());
    await lib.create('After failure', cellsA);
    expect(backend.kind).toBe('memory');
    expect((await lib.list()).map((e) => e.name)).toEqual(['After failure']);
  });

  it('a backend that throws on every call is not caught by the library layer itself', async () => {
    // the session hook catches these and shows a notice; the library reports them to its caller
    const lib = createNotebookLibrary(brokenBackend, clock(), counter());
    await expect(lib.list()).rejects.toThrow('boom');
  });
});
