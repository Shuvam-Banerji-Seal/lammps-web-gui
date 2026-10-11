import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Cell } from './cells';
import {
  createNotebookLibrary, defaultLibraryBackend, emptyCells, fingerprint, readLegacyCells,
  type LibraryEntry, type NotebookLibrary, type WorkingCopy,
} from './notebookLibrary';
import { UNTITLED, type CellOutput, type ImportResult } from './notebookFile';

/** A notebook replaced by New, Open, Import or Close: the engine session is reset and the outputs shown. */
export interface LoadedOutputs {
  generation: number;
  outputs: Record<string, CellOutput>;
}

export interface NotebookSession {
  /** False until the library and the working copy are read from storage. */
  ready: boolean;
  cells: Cell[];
  /** The React state setter for the cells (same shape as useState's). */
  setCells: (action: Cell[] | ((prev: Cell[]) => Cell[])) => void;
  name: string;
  /** Library id of the open notebook; null when it was never saved. */
  baseId: string | null;
  /** True when the open notebook is in the library and matches what was last saved. */
  saved: boolean;
  /** Counts the replacements of the open notebook (New, Open, Import, Close). The page resets its engine on a change. */
  generation: number;
  /** Outputs of an imported notebook, to show once (matched by generation). */
  loadedOutputs: LoadedOutputs | null;
  entries: LibraryEntry[];
  storage: 'indexeddb' | 'memory';
  /** A message for the person (storage fallback, a failed save); null when none. */
  notice: string | null;
  dismissNotice: () => void;
  library: NotebookLibrary;
  newNotebook: () => void;
  openEntry: (id: string) => Promise<void>;
  /** Saves the open notebook under its name. Returns false when it was never saved (use saveAs). */
  save: () => Promise<boolean>;
  /** Saves the open notebook under a new library entry. Returns false when the save failed (see notice). */
  saveAs: (name: string) => Promise<boolean>;
  /** Sets the name of the open notebook (it is unsaved until saved). */
  setName: (name: string) => void;
  rename: (id: string, name: string) => Promise<void>;
  duplicate: (id: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  /** Adds an imported notebook to the library and opens it. */
  importNotebook: (imp: ImportResult) => Promise<void>;
  refresh: () => Promise<void>;
}

const AUTOSAVE_MS = 300;

/**
 * The notebook's cells and their library. The open notebook (the working copy) is autosaved to browser
 * storage on every change, so a reload brings it back; "saved" means it matches its library entry.
 * `starter` is the first-run content (used when nothing is stored yet).
 */
export const useNotebookSession = (starter: Cell[], library: NotebookLibrary = defaultLibrary()): NotebookSession => {
  const [work, setWork] = useState<WorkingCopy>(() => ({ baseId: null, name: UNTITLED, cells: starter, savedFingerprint: null }));
  const [ready, setReady] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [loadedOutputs, setLoadedOutputs] = useState<LoadedOutputs | null>(null);
  const [entries, setEntries] = useState<LibraryEntry[]>([]);
  const [storage, setStorage] = useState<'indexeddb' | 'memory'>('indexeddb');
  const [notice, setNotice] = useState<string | null>(null);
  const newId = useCallback(() => `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, []);
  const workRef = useRef(work);
  workRef.current = work;
  /** Set when a notebook is replaced: its working copy is written at once, not after the autosave delay. */
  const saveNow = useRef(false);

  const note = useCallback((e: unknown, what: string) => {
    setNotice(`${what}: ${e instanceof Error ? e.message : String(e)}`);
  }, []);

  const refresh = useCallback(async () => {
    try {
      setEntries(await library.list());
    } catch (e) {
      note(e, 'The notebook list could not be read');
    }
    setStorage(library.storageKind());
  }, [library, note]);

  // first load: the working copy (or the notebook the earlier version kept, or the starter)
  useEffect(() => {
    let alive = true;
    void (async () => {
      try {
        const working = await library.loadWorking();
        const legacy = await library.migrateLegacy(readLegacyCells(safeStorage()));
        if (!alive) return;
        if (working) {
          setWork(working);
        } else if (legacy) {
          setWork({ baseId: legacy.id, name: legacy.name, cells: legacy.cells, savedFingerprint: fingerprint(legacy.name, legacy.cells) });
        }
      } catch (e) {
        if (alive) note(e, 'The notebook storage could not be read; this notebook is kept for this page only');
      }
      if (!alive) return;
      setReady(true);
      await refresh();
    })();
    return () => {
      alive = false;
    };
  }, [library, note, refresh]);

  // autosave of the working copy
  useEffect(() => {
    if (!ready) return;
    const save = () => {
      library.saveWorking(workRef.current).catch((e) => note(e, 'The open notebook could not be autosaved'));
    };
    const t = window.setTimeout(save, saveNow.current ? 0 : AUTOSAVE_MS);
    saveNow.current = false;
    const flush = () => {
      window.clearTimeout(t);
      save();
    };
    window.addEventListener('pagehide', flush);
    return () => {
      window.clearTimeout(t);
      window.removeEventListener('pagehide', flush);
    };
  }, [work, ready, library, note]);

  const saved = work.baseId !== null && work.savedFingerprint === fingerprint(work.name, work.cells);

  const setCells = useCallback((action: Cell[] | ((prev: Cell[]) => Cell[])) => {
    setWork((w) => ({ ...w, cells: typeof action === 'function' ? action(w.cells) : action }));
  }, []);

  const replace = useCallback((next: WorkingCopy, outputs?: Record<string, CellOutput>) => {
    saveNow.current = true;
    setWork(next);
    setGeneration((g) => g + 1);
    setLoadedOutputs(outputs ? { generation: Date.now(), outputs } : null);
  }, []);

  const newNotebook = useCallback(() => {
    replace({ baseId: null, name: UNTITLED, cells: emptyCells(newId), savedFingerprint: null });
  }, [replace, newId]);

  const openEntry = useCallback(async (id: string) => {
    try {
      const rec = await library.get(id);
      if (!rec) {
        setNotice('That notebook is no longer in the library.');
        await refresh();
        return;
      }
      replace({ baseId: rec.id, name: rec.name, cells: rec.cells, inputFiles: rec.inputFiles, savedFingerprint: fingerprint(rec.name, rec.cells) });
    } catch (e) {
      note(e, 'The notebook could not be opened');
    }
  }, [library, replace, refresh, note]);

  const save = useCallback(async (): Promise<boolean> => {
    const w = workRef.current;
    if (w.baseId === null) return false;
    try {
      const rec = await library.update(w.baseId, { name: w.name, cells: w.cells });
      if (!rec) {
        setNotice('The library entry for this notebook is gone. Use Save as to keep it.');
        setWork({ ...w, baseId: null, savedFingerprint: null });
        return false;
      }
      // the fingerprint of what was written: edits made while the save was in flight stay unsaved
      setWork((cur) => ({ ...cur, savedFingerprint: fingerprint(rec.name, w.cells) }));
      await refresh();
      return true;
    } catch (e) {
      note(e, 'The notebook could not be saved');
      return false;
    }
  }, [library, refresh, note]);

  const saveAs = useCallback(async (name: string): Promise<boolean> => {
    const w = workRef.current;
    try {
      const rec = await library.create(name, w.cells, w.inputFiles);
      // the fingerprint of the snapshot that was written: edits made while the save was in flight stay unsaved
      setWork((cur) => ({ ...cur, baseId: rec.id, name: rec.name, savedFingerprint: fingerprint(rec.name, w.cells) }));
      await refresh();
      return true;
    } catch (e) {
      note(e, 'The notebook could not be saved');
      return false;
    }
  }, [library, refresh, note]);

  const setName = useCallback((name: string) => {
    setWork((cur) => ({ ...cur, name: name.trim() || UNTITLED }));
  }, []);

  const rename = useCallback(async (id: string, name: string) => {
    try {
      const rec = await library.update(id, { name });
      if (rec && id === workRef.current.baseId) {
        setWork((cur) => {
          const wasSaved = cur.savedFingerprint === fingerprint(cur.name, cur.cells);
          return { ...cur, name: rec.name, savedFingerprint: wasSaved ? fingerprint(rec.name, cur.cells) : cur.savedFingerprint };
        });
      }
      await refresh();
    } catch (e) {
      note(e, 'The notebook could not be renamed');
    }
  }, [library, refresh, note]);

  const duplicate = useCallback(async (id: string) => {
    try {
      const old = await library.get(id);
      if (old) await library.create(`${old.name} (copy)`, structuredClone(old.cells), old.inputFiles);
      await refresh();
    } catch (e) {
      note(e, 'The notebook could not be copied');
    }
  }, [library, refresh, note]);

  const remove = useCallback(async (id: string) => {
    try {
      await library.remove(id);
      if (id === workRef.current.baseId) {
        setWork((cur) => ({ ...cur, baseId: null, savedFingerprint: null }));
        setNotice('The saved notebook was deleted. The open cells are kept, unsaved.');
      }
      await refresh();
    } catch (e) {
      note(e, 'The notebook could not be deleted');
    }
  }, [library, refresh, note]);

  const importNotebook = useCallback(async (imp: ImportResult) => {
    try {
      const rec = await library.create(imp.name, imp.cells, imp.inputFiles);
      replace({ baseId: rec.id, name: rec.name, cells: rec.cells, inputFiles: rec.inputFiles, savedFingerprint: fingerprint(rec.name, rec.cells) },
        imp.outputs);
      await refresh();
    } catch (e) {
      note(e, 'The notebook could not be imported');
    }
  }, [library, replace, refresh, note]);

  const dismissNotice = useCallback(() => setNotice(null), []);

  return useMemo(() => ({
    ready, cells: work.cells, setCells, name: work.name, baseId: work.baseId, saved, generation, loadedOutputs,
    entries, storage, notice, dismissNotice, library, newNotebook, openEntry, save, saveAs, setName, rename, duplicate, remove,
    importNotebook, refresh,
  }), [ready, work, setCells, saved, generation, loadedOutputs, entries, storage, notice, dismissNotice, library,
    newNotebook, openEntry, save, saveAs, setName, rename, duplicate, remove, importNotebook, refresh]);
};

/** localStorage, or null where it cannot be reached (private modes, blocked storage). */
const safeStorage = (): Storage | null => {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
};

/** One library per page: created on first use, shared by every session hook. */
let sharedLibrary: NotebookLibrary | null = null;
const defaultLibrary = (): NotebookLibrary => {
  if (!sharedLibrary) sharedLibrary = createNotebookLibrary(defaultLibraryBackend());
  return sharedLibrary;
};
