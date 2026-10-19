import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { EngineClient } from '../../../engine/client';
import { getFileStore, readInputText, FileStoreError, type FileStore, type StoredFile } from '../../../services/fileStore';
import { loadPotentialText, resolveMissing, type PotentialEntry } from './potentialLibrary';

/** A file the user may convert with the structure import dialog. */
export interface ImportSource {
  name: string;
  text: string;
}

export interface InputFiles {
  /** Files the notebook can read by name: name -> size in bytes. */
  inputs: Record<string, number>;
  /** Total bytes in the persistent store. */
  totalBytes: number;
  /** The last failure (quota, storage), for the panel; null when none. */
  error: string | null;
  /**
   * Adds picked or dropped files (a FileList, or a File[] from a drop) (stored and sent to the engine).
   * Returns the first structure or data file, for the import dialog.
   */
  addPicked: (list: FileList | readonly File[] | null) => Promise<ImportSource | null>;
  /** Adds one file given as text (used by the import dialog for its converted output). */
  addText: (name: string, text: string) => Promise<void>;
  /** Reads a picked file for the import dialog without storing it. */
  readForImport: (file: File) => Promise<ImportSource>;
  remove: (name: string) => Promise<void>;
  clearAll: () => Promise<void>;
  /** Sends every stored file to the engine (after a reset). */
  sendToEngine: () => void;
  dismissError: () => void;
  /** Names of the stored files that came from the LAMMPS potential library (GPL-2.0). */
  libraryNames: ReadonlySet<string>;
  /** Adds one file of the potential library to the inputs (a user file of the same name is replaced). */
  addLibrary: (name: string) => Promise<void>;
  /**
   * Adds the library files a script names that the inputs do not have yet (a user file always wins).
   * Resolves to the names added and the names whose fetch failed.
   */
  fetchLibraryFor: (script: string) => Promise<{ added: string[]; failed: string[] }>;
}

/** Per-viewer record of which stored names came from the library (a convenience; the files live in the store). */
const LIBRARY_KEY = 'lammps-notebook-library-files';
const readLibraryNames = (): Set<string> => {
  try {
    const v: unknown = JSON.parse(localStorage.getItem(LIBRARY_KEY) ?? '[]');
    return new Set(Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []);
  } catch {
    return new Set();
  }
};
const writeLibraryNames = (names: ReadonlySet<string>): void => {
  try {
    localStorage.setItem(LIBRARY_KEY, JSON.stringify([...names]));
  } catch {
    // the mark is lost on reload when storage is blocked; the file itself is unaffected
  }
};

/** Names the import dialog offers for: PDB, mmCIF, XYZ, and LAMMPS data files. */
export const isImportable = (name: string): boolean =>
  /\.(pdb|ent|cif|mmcif|xyz|data|lmp|lammps)$/i.test(name);

const describeError = (e: unknown): string =>
  e instanceof FileStoreError ? e.message : `The file could not be stored: ${e instanceof Error ? e.message : String(e)}`;

/**
 * Input files of the notebook, kept in the persistent store (IndexedDB, or memory where
 * IndexedDB is missing) and mirrored into the engine. Restores the stored files on mount.
 */
export const useInputFiles = (getClient: () => EngineClient, store: FileStore = getFileStore()): InputFiles => {
  const [stored, setStored] = useState<StoredFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  /** The stored texts by name, for sending to the engine without waiting on the store. */
  const texts = useRef(new Map<string, string>());
  const [libraryNames, setLibraryNames] = useState<ReadonlySet<string>>(() => readLibraryNames());
  /** Marks (or unmarks) a stored name as a library file. */
  const markLibrary = useCallback((name: string, fromLibrary: boolean) => {
    setLibraryNames((prev) => {
      if (prev.has(name) === fromLibrary) return prev;
      const next = new Set(prev);
      if (fromLibrary) next.add(name);
      else next.delete(name);
      writeLibraryNames(next);
      return next;
    });
  }, []);

  const reload = useCallback(async () => {
    const list = await store.list();
    texts.current = new Map(list.map((f) => [f.name, f.text]));
    setStored(list);
  }, [store]);

  const sendAll = useCallback(() => {
    const client = getClient();
    for (const [name, text] of texts.current) client.addFile(name, text);
  }, [getClient]);

  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        await reload();
        if (alive) sendAll();
      } catch (e) {
        if (alive) setError(describeError(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [reload, sendAll]);

  const storeFile = useCallback(async (name: string, text: string, size: number, fromLibrary = false) => {
    try {
      await store.put({ name, text, size });
      texts.current.set(name, text);
      getClient().addFile(name, text);
      markLibrary(name, fromLibrary);
      await reload();
      setError(null);
    } catch (e) {
      setError(describeError(e));
      throw e;
    }
  }, [store, getClient, reload, markLibrary]);

  const addLibrary = useCallback(async (name: string) => {
    const text = await loadPotentialText(name);
    try {
      await storeFile(name, text, text.length, true);
    } catch {
      // the store refused it (quota, shown by storeFile): the engine still gets the text for this session
      texts.current.set(name, text);
      getClient().addFile(name, text);
      markLibrary(name, true);
    }
  }, [storeFile, getClient, markLibrary]);

  const fetchLibraryFor = useCallback(async (script: string): Promise<{ added: string[]; failed: string[] }> => {
    const missing: PotentialEntry[] = await resolveMissing(script, new Set(texts.current.keys()));
    const added: string[] = [];
    const failed: string[] = [];
    for (const f of missing) {
      try {
        await addLibrary(f.name);
        added.push(f.name);
      } catch {
        // the library could not be fetched (network); the run goes on without the file
        failed.push(f.name);
      }
    }
    return { added, failed };
  }, [addLibrary]);

  const addText = useCallback(async (name: string, text: string) => {
    await storeFile(name, text, text.length);
  }, [storeFile]);

  const addPicked = useCallback(async (list: FileList | readonly File[] | null): Promise<ImportSource | null> => {
    if (!list) return null;
    let candidate: ImportSource | null = null;
    for (const file of Array.from(list)) {
      try {
        const text = await readInputText(file);
        await storeFile(file.name, text, file.size);
        if (!candidate && isImportable(file.name)) candidate = { name: file.name, text };
      } catch {
        // error already shown
      }
    }
    return candidate;
  }, [storeFile]);

  const readForImport = useCallback(async (file: File): Promise<ImportSource> => ({
    name: file.name,
    text: await readInputText(file),
  }), []);

  const remove = useCallback(async (name: string) => {
    getClient().removeFile(name);
    texts.current.delete(name);
    markLibrary(name, false);
    try {
      await store.remove(name);
    } catch (e) {
      setError(describeError(e));
    }
    await reload();
  }, [getClient, store, reload]);

  const clearAll = useCallback(async () => {
    for (const name of texts.current.keys()) getClient().removeFile(name);
    texts.current.clear();
    setLibraryNames(new Set());
    writeLibraryNames(new Set());
    try {
      await store.clear();
    } catch (e) {
      setError(describeError(e));
    }
    await reload();
  }, [getClient, store, reload]);

  const inputs = useMemo(() => {
    const out: Record<string, number> = {};
    for (const f of stored) out[f.name] = f.size;
    return out;
  }, [stored]);

  const totalBytes = useMemo(() => stored.reduce((t, f) => t + f.size, 0), [stored]);

  const dismissError = useCallback(() => setError(null), []);

  return {
    inputs, totalBytes, error, addPicked, addText, readForImport, remove, clearAll, sendToEngine: sendAll, dismissError,
    libraryNames, addLibrary, fetchLibraryFor,
  };
};
