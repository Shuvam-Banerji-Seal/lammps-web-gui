/**
 * Persistent store for the input files the notebook reads (data, include and potential files,
 * and converted structures). Files live in IndexedDB so they survive a page reload; when
 * IndexedDB is missing or cannot be opened (jsdom, some private modes) the store keeps them
 * in memory for the page's lifetime instead.
 *
 * Text is stored as the engine receives it: binary input (gzip) as one character per byte
 * (latin1), the same encoding the notebook sends to the engine.
 */
import { FILESTORE_EVENT } from './fileStoreEvent';

export type StoredKind = 'text' | 'gzip' | 'structure';

export interface StoredFile {
  name: string;
  /** File content; binary as latin1 (one character per byte). */
  text: string;
  /** Size in bytes of the file as added. */
  size: number;
  /** Milliseconds since the epoch, when it was stored. */
  mtime: number;
  kind: StoredKind;
}

export type FileStoreErrorCode = 'quota' | 'storage';

export class FileStoreError extends Error {
  constructor(readonly code: FileStoreErrorCode, message: string) {
    super(message);
    this.name = 'FileStoreError';
  }
}

export interface FileStore {
  /** Stores (or replaces) a file. Throws FileStoreError('quota') when the browser is full. */
  put(file: Omit<StoredFile, 'mtime' | 'kind'> & { kind?: StoredKind; mtime?: number }): Promise<void>;
  get(name: string): Promise<StoredFile | undefined>;
  /** All stored files, sorted by name. */
  list(): Promise<StoredFile[]>;
  remove(name: string): Promise<void>;
  clear(): Promise<void>;
  /** Which backend holds the files. */
  readonly backend: 'indexeddb' | 'memory';
}

/** The file kind a name implies: gzip for .gz, structure for PDB, CIF, XYZ and mmCIF, text otherwise. */
export const kindOf = (name: string): StoredKind => {
  const lower = name.toLowerCase();
  if (lower.endsWith('.gz')) return 'gzip';
  if (/\.(pdb|ent|cif|mmcif|xyz)$/.test(lower)) return 'structure';
  return 'text';
};

/** Window event dispatched after every successful put, remove or clear (both backends), so open views re-list. */
export { FILESTORE_EVENT };

const notifyChange = (): void => {
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(FILESTORE_EVENT));
};

const DB_NAME = 'lammps-notebook-files';
const STORE = 'files';

const isQuota = (e: unknown): boolean => {
  const name = (e as { name?: string } | null)?.name;
  return name === 'QuotaExceededError' || name === 'NS_ERROR_DOM_QUOTA_REACHED';
};

const errorMessage = (e: unknown): string => (e instanceof Error ? e.message : String(e));

const normalize = (f: Parameters<FileStore['put']>[0]): StoredFile => ({
  name: f.name,
  text: f.text,
  size: f.size,
  mtime: f.mtime ?? Date.now(),
  kind: f.kind ?? kindOf(f.name),
});

/**
 * In-memory store. `limitBytes` (optional) makes put throw FileStoreError('quota') once the
 * total size of the stored text would exceed it, as a full browser quota does.
 */
export const createMemoryFileStore = (limitBytes?: number): FileStore => {
  const files = new Map<string, StoredFile>();
  const total = (skip?: string): number => {
    let t = 0;
    for (const [k, f] of files) if (k !== skip) t += f.size;
    return t;
  };
  return {
    backend: 'memory',
    async put(input) {
      const rec = normalize(input);
      if (limitBytes !== undefined && total(rec.name) + rec.size > limitBytes) {
        throw new FileStoreError('quota', `${rec.name} does not fit: the store is limited to ${limitBytes} bytes.`);
      }
      files.set(rec.name, rec);
      notifyChange();
    },
    async get(name) {
      return files.get(name);
    },
    async list() {
      return [...files.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    },
    async remove(name) {
      files.delete(name);
      notifyChange();
    },
    async clear() {
      files.clear();
      notifyChange();
    },
  };
};

/** Opens the IndexedDB database, or resolves null when it cannot be opened. */
const openDb = (factory: IDBFactory): Promise<IDBDatabase | null> =>
  new Promise((resolve) => {
    try {
      const req = factory.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE, { keyPath: 'name' });
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
 * File store over IndexedDB. If the database cannot be opened it falls back to an in-memory
 * store for this page. Quota errors from IndexedDB are reported as FileStoreError('quota').
 */
export const createIndexedDbFileStore = (factory: IDBFactory | null | undefined): FileStore => {
  const memory = createMemoryFileStore();
  let dbPromise: Promise<IDBDatabase | null> | null = null;
  let fallback = false;
  const db = (): Promise<IDBDatabase | null> => {
    if (!factory) return Promise.resolve(null);
    if (!dbPromise) dbPromise = openDb(factory);
    return dbPromise;
  };
  const withStore = async <T>(mode: IDBTransactionMode, run: (s: IDBObjectStore) => Promise<T>): Promise<T | undefined> => {
    if (fallback) return undefined;
    const conn = await db();
    if (!conn) {
      fallback = true;
      return undefined;
    }
    const tx = conn.transaction(STORE, mode);
    return run(tx.objectStore(STORE));
  };
  const store: FileStore = {
    get backend() {
      return fallback || !factory ? 'memory' : 'indexeddb';
    },
    async put(input) {
      const rec = normalize(input);
      const done = await withStore('readwrite', async (s) => {
        try {
          await request(s.put(rec));
        } catch (e) {
          if (isQuota(e)) throw new FileStoreError('quota', `${rec.name} does not fit in the browser's file storage. Remove some files first.`);
          throw new FileStoreError('storage', `Could not store ${rec.name}: ${errorMessage(e)}`);
        }
        return true;
      });
      if (done === undefined) await memory.put(rec);
      notifyChange();
    },
    async get(name) {
      const found = await withStore('readonly', (s) => request(s.get(name)) as Promise<StoredFile | undefined>);
      return found === undefined ? memory.get(name) : found;
    },
    async list() {
      const all = await withStore('readonly', (s) => request(s.getAll()) as Promise<StoredFile[]>);
      if (all === undefined) return memory.list();
      return all.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    },
    async remove(name) {
      const done = await withStore('readwrite', (s) => request(s.delete(name)).then(() => true));
      if (done === undefined) await memory.remove(name);
      notifyChange();
    },
    async clear() {
      const done = await withStore('readwrite', (s) => request(s.clear()).then(() => true));
      if (done === undefined) await memory.clear();
      notifyChange();
    },
  };
  return store;
};

let shared: FileStore | null = null;

/** The page's file store: IndexedDB when available, otherwise in memory. */
export const getFileStore = (): FileStore => {
  if (!shared) {
    const factory = typeof indexedDB !== 'undefined' ? indexedDB : null;
    shared = factory ? createIndexedDbFileStore(factory) : createMemoryFileStore();
  }
  return shared;
};

/** Total bytes of a listing. */
export const totalBytes = (files: { size: number }[]): number => {
  let t = 0;
  for (const f of files) t += f.size;
  return t;
};

/**
 * Text of a picked File as the engine expects it: gzip bytes one character per byte, other
 * files as UTF-8 text. A file that is not valid UTF-8 (a binary STL, for one) keeps its bytes
 * one character per byte too, so the engine reads the same bytes the file holds.
 */
export const readInputText = async (file: File): Promise<string> => {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const latin1 = (): string => {
    let out = '';
    for (let k = 0; k < bytes.length; k += 0x8000) out += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
    return out;
  };
  if (file.name.toLowerCase().endsWith('.gz')) return latin1();
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return latin1();
  }
};
