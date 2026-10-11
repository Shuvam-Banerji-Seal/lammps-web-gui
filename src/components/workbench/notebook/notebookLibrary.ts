/**
 * The notebook library: several saved notebooks in browser storage, and the working copy of the
 * notebook on screen (autosaved, so a reload brings it back). IndexedDB holds them (sizes are large);
 * where IndexedDB is missing or fails (private modes) the library keeps them in memory for the page's
 * lifetime. Every storage call is guarded: a failure never throws into the UI.
 *
 * Pure apart from the storage handed in, so it is tested with a fake backend (tests/notebookLibrary.test.ts).
 */
import type { Cell } from './cells';
import { reviveCells } from './cells';
import { UNTITLED } from './notebookFile';

/** A saved notebook. `inputFiles` is present only when the notebook was imported with its files. */
export interface NotebookRecord {
  id: string;
  name: string;
  created: string;
  modified: string;
  cells: Cell[];
  inputFiles?: Record<string, string>;
}

/** What the Open list shows (no cell text). */
export interface LibraryEntry {
  id: string;
  name: string;
  created: string;
  modified: string;
  cellCount: number;
}

/** The notebook on screen. `baseId` is its library entry (null: never saved). `savedFingerprint` is what was last saved. */
export interface WorkingCopy {
  baseId: string | null;
  name: string;
  cells: Cell[];
  inputFiles?: Record<string, string>;
  savedFingerprint: string | null;
}

/** Storage under the library: notebooks by id, and a few named values. Implementations may throw. */
export interface LibraryBackend {
  readonly kind: 'indexeddb' | 'memory';
  all(): Promise<NotebookRecord[]>;
  get(id: string): Promise<NotebookRecord | undefined>;
  put(record: NotebookRecord): Promise<void>;
  remove(id: string): Promise<void>;
  getMeta(key: string): Promise<unknown>;
  setMeta(key: string, value: unknown): Promise<void>;
}

/** The part of the content that decides "unsaved changes": the name and the cells, by id, kind and text. */
export const fingerprint = (name: string, cells: readonly Cell[]): string =>
  JSON.stringify([name, cells.map((c) => [c.id, c.kind === 'markdown' ? 'markdown' : 'code', c.text])]);

export const createMemoryBackend = (): LibraryBackend => {
  const notebooks = new Map<string, NotebookRecord>();
  const meta = new Map<string, unknown>();
  return {
    kind: 'memory',
    async all() {
      return [...notebooks.values()].map((r) => structuredClone(r));
    },
    async get(id) {
      const r = notebooks.get(id);
      return r && structuredClone(r);
    },
    async put(record) {
      notebooks.set(record.id, structuredClone(record));
    },
    async remove(id) {
      notebooks.delete(id);
    },
    async getMeta(key) {
      return meta.has(key) ? structuredClone(meta.get(key)) : undefined;
    },
    async setMeta(key, value) {
      meta.set(key, structuredClone(value));
    },
  };
};

const DB_NAME = 'lammps-notebook-library';
const NOTEBOOKS = 'notebooks';
const META = 'meta';

const openDb = (factory: IDBFactory): Promise<IDBDatabase | null> =>
  new Promise((resolve) => {
    try {
      const req = factory.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(NOTEBOOKS)) db.createObjectStore(NOTEBOOKS, { keyPath: 'id' });
        if (!db.objectStoreNames.contains(META)) db.createObjectStore(META);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch {
      resolve(null);
    }
  });

const request = <T>(req: IDBRequest<T>): Promise<T> =>
  new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

/**
 * IndexedDB backend. If the database cannot be opened, or an operation fails, the backend switches to an
 * in-memory copy for the rest of the page's life (`kind` then reads 'memory'). Notebooks written to
 * IndexedDB before a failure are not visible after it; the UI says which storage is in use.
 */
export const createIndexedDbBackend = (factory: IDBFactory | null | undefined): LibraryBackend => {
  const memory = createMemoryBackend();
  let fallback = !factory;
  let dbPromise: Promise<IDBDatabase | null> | null = null;
  const connect = (): Promise<IDBDatabase | null> => {
    if (!factory) return Promise.resolve(null);
    if (!dbPromise) dbPromise = openDb(factory);
    return dbPromise;
  };
  const run = async <T>(
    onDb: (db: IDBDatabase) => Promise<T>,
    onMemory: () => Promise<T>,
  ): Promise<T> => {
    if (fallback) return onMemory();
    try {
      const db = await connect();
      if (!db) {
        fallback = true;
        return onMemory();
      }
      return await onDb(db);
    } catch {
      fallback = true;
      return onMemory();
    }
  };
  const store = (db: IDBDatabase, name: string, mode: IDBTransactionMode) => db.transaction(name, mode).objectStore(name);
  return {
    get kind() {
      return fallback ? 'memory' : 'indexeddb';
    },
    all: () => run(
      async (db) => (await request(store(db, NOTEBOOKS, 'readonly').getAll() as IDBRequest<NotebookRecord[]>)) ?? [],
      () => memory.all(),
    ),
    get: (id) => run(
      async (db) => (await request(store(db, NOTEBOOKS, 'readonly').get(id) as IDBRequest<NotebookRecord | undefined>)) ?? undefined,
      () => memory.get(id),
    ),
    put: (record) => run(
      async (db) => { await request(store(db, NOTEBOOKS, 'readwrite').put(structuredClone(record))); },
      () => memory.put(record),
    ),
    remove: (id) => run(
      async (db) => { await request(store(db, NOTEBOOKS, 'readwrite').delete(id)); },
      () => memory.remove(id),
    ),
    getMeta: (key) => run(
      async (db) => request(store(db, META, 'readonly').get(key) as IDBRequest<unknown>),
      () => memory.getMeta(key),
    ),
    setMeta: (key, value) => run(
      async (db) => { await request(store(db, META, 'readwrite').put(structuredClone(value), key)); },
      () => memory.setMeta(key, value),
    ),
  };
};

/** The default backend in this browser: IndexedDB when it exists, else memory. */
export const defaultLibraryBackend = (): LibraryBackend =>
  createIndexedDbBackend(typeof indexedDB !== 'undefined' ? indexedDB : null);

/** Reads the notebook the earlier version kept in localStorage (`m3d.notebook.v1`); null when none or unreadable. */
export const readLegacyCells = (storage: Pick<Storage, 'getItem'> | null | undefined, key = 'm3d.notebook.v1'): Cell[] | null => {
  try {
    const raw = storage?.getItem(key);
    if (!raw) return null;
    return reviveCells(JSON.parse(raw));
  } catch {
    return null;
  }
};

const LEGACY_META = 'legacyMigrated';
const WORKING_META = 'working';

export const emptyCells = (newId: () => string): Cell[] => [{ id: newId(), kind: 'code', text: '' }];

/** Creates the library operations over a backend. `now` and `newId` are injected for tests. */
export const createNotebookLibrary = (
  backend: LibraryBackend,
  now: () => string = () => new Date().toISOString(),
  newId: () => string = () => `nb${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`,
) => {
  const entryOf = (r: NotebookRecord): LibraryEntry => ({
    id: r.id, name: r.name, created: r.created, modified: r.modified, cellCount: r.cells.length,
  });

  const list = async (): Promise<LibraryEntry[]> => {
    const all = await backend.all();
    return all.map(entryOf).sort((a, b) => (a.modified < b.modified ? 1 : a.modified > b.modified ? -1 : 0));
  };

  /** Creates a saved notebook (new id). */
  const create = async (name: string, cells: Cell[], inputFiles?: Record<string, string>): Promise<NotebookRecord> => {
    const t = now();
    const rec: NotebookRecord = {
      id: newId(), name: name.trim() || UNTITLED, created: t, modified: t, cells: structuredClone(cells),
      ...(inputFiles ? { inputFiles: structuredClone(inputFiles) } : {}),
    };
    await backend.put(rec);
    return rec;
  };

  /** Writes the content of an existing notebook. Returns the stored record, or null when the id is unknown. */
  const update = async (id: string, patch: { name?: string; cells?: Cell[] }): Promise<NotebookRecord | null> => {
    const old = await backend.get(id);
    if (!old) return null;
    const rec: NotebookRecord = {
      ...old,
      name: patch.name !== undefined ? (patch.name.trim() || UNTITLED) : old.name,
      cells: patch.cells ?? old.cells,
      modified: now(),
    };
    await backend.put(rec);
    return rec;
  };

  const get = (id: string) => backend.get(id);

  const remove = (id: string) => backend.remove(id);

  const duplicate = async (id: string, copyName: string): Promise<NotebookRecord | null> => {
    const old = await backend.get(id);
    if (!old) return null;
    return create(copyName, structuredClone(old.cells), old.inputFiles);
  };

  /**
   * The first-run migration: the notebook kept by the earlier version becomes a saved entry, once.
   * Returns the new record, or null when there was nothing to migrate (or it was done before).
   */
  const migrateLegacy = async (cells: Cell[] | null): Promise<NotebookRecord | null> => {
    if (!cells) return null;
    if ((await backend.getMeta(LEGACY_META)) === true) return null;
    const rec = await create(UNTITLED, cells);
    await backend.setMeta(LEGACY_META, true);
    return rec;
  };

  const loadWorking = async (): Promise<WorkingCopy | null> => {
    const raw = await backend.getMeta(WORKING_META);
    if (!raw || typeof raw !== 'object') return null;
    const w = raw as Partial<WorkingCopy>;
    const cells = reviveCells(w.cells);
    if (!cells || typeof w.name !== 'string') return null;
    return {
      baseId: typeof w.baseId === 'string' ? w.baseId : null,
      name: w.name,
      cells,
      inputFiles: w.inputFiles,
      savedFingerprint: typeof w.savedFingerprint === 'string' ? w.savedFingerprint : null,
    };
  };

  const saveWorking = (w: WorkingCopy): Promise<void> => backend.setMeta(WORKING_META, w);

  return { list, create, update, get, remove, duplicate, migrateLegacy, loadWorking, saveWorking, entryOf, storageKind: () => backend.kind };
};

export type NotebookLibrary = ReturnType<typeof createNotebookLibrary>;
