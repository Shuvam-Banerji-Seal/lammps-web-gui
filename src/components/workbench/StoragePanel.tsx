import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Copy, Download, Search, ShieldCheck, Trash2, Upload, X } from 'lucide-react';
import { getThemeTokens, type Theme } from '../../theme';
import { isLammpsScript } from '../../lammps/scriptDetect';
import {
  FILESTORE_EVENT,
  FileStoreError,
  getFileStore,
  readInputText,
  totalBytes,
  type FileStore,
  type StoredFile,
  type StoredKind,
} from '../../services/fileStore';

/** The browser-storage side panel: the files kept in IndexedDB, and how the workbench modules can use them. */

export { isLammpsScript };

/**
 * True for a structure, data, dump or trajectory file the Structure Viewer reads; a gzipped one (the viewer
 * decompresses it) when the name without .gz is one.
 */
export const isViewable = (file: { name: string; kind: StoredKind }): boolean => {
  if (file.kind === 'structure') return true;
  const name = file.kind === 'gzip' ? file.name.replace(/\.gz$/i, '') : file.name;
  return /^data\./i.test(name) || /\.(data|dump|lammpstrj|xyz|pdb|cif)$/i.test(name);
};

const formatSize = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

/** The first `n` lines of a text, found without splitting the whole (possibly large) file. */
export const headLines = (text: string, n: number): string => {
  let end = -1;
  for (let k = 0; k < n; k++) {
    end = text.indexOf('\n', end + 1);
    if (end < 0) return text;
  }
  return text.slice(0, end);
};

const formatMb = (n: number): string => `${(n / (1024 * 1024)).toFixed(1)} MB`;

const formatDate = (ms: number): string => new Date(ms).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });

/** Bytes of a latin1 string (one character per byte), as gzip files are stored. */
const latin1Bytes = (text: string): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(new ArrayBuffer(text.length));
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
  return out;
};

/** Saves a blob under a file name (an object URL, clicked and revoked). */
const saveBlob = (name: string, blob: Blob): void => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const readEstimate = async (): Promise<{ usage: number; quota: number } | null> => {
  try {
    const est = await navigator.storage?.estimate?.();
    if (est && typeof est.usage === 'number' && typeof est.quota === 'number') return { usage: est.usage, quota: est.quota };
  } catch {
    /* unavailable: the header just omits the quota line */
  }
  return null;
};

const readPersisted = async (): Promise<boolean | null> => {
  try {
    if (typeof navigator.storage?.persisted !== 'function') return null;
    return await navigator.storage.persisted();
  } catch {
    return null;
  }
};

export interface StoragePanelProps {
  theme: Theme;
  open: boolean;
  onClose: () => void;
  /** A LAMMPS input script to open as a new MD Notebook. */
  onOpenInNotebook: (script: string) => void;
  /** A script to open as a new Script Builder tab (its flowchart view shows it). */
  onOpenInBuilder: (file: { text: string; name: string }) => void;
  /** A structure / data / dump / trajectory file to show in the Structure Viewer. */
  onOpenInViewer: (file: { name: string; text: string }) => void;
  /** Injected for tests; defaults to getFileStore(). */
  store?: FileStore;
}

/** Drawer listing the files kept in browser storage, with download, delete, preview and hand-off actions. */
const StoragePanel: React.FC<StoragePanelProps> = ({
  theme, open, onClose, onOpenInNotebook, onOpenInBuilder, onOpenInViewer, store: injected,
}) => {
  const ct = getThemeTokens(theme);
  const store = useMemo(() => injected ?? getFileStore(), [injected]);
  const [files, setFiles] = useState<StoredFile[]>([]);
  const [backend, setBackend] = useState<FileStore['backend']>(store.backend);
  const [estimate, setEstimate] = useState<{ usage: number; quota: number } | null>(null);
  const [persisted, setPersisted] = useState<boolean | null>(null);
  const [query, setQuery] = useState('');
  const [previewName, setPreviewName] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const alive = useRef(true);
  const closeRef = useRef<HTMLButtonElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [list, est, pers] = await Promise.all([store.list(), readEstimate(), readPersisted()]);
      if (!alive.current) return;
      setFiles(list);
      setBackend(store.backend);
      setEstimate(est);
      setPersisted(pers);
    } catch (e) {
      if (alive.current) setError(`Could not read browser storage: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, [store]);

  useEffect(() => {
    if (!open) return;
    void refresh();
    closeRef.current?.focus();
    const onChange = () => void refresh();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCloseRef.current();
    };
    window.addEventListener(FILESTORE_EVENT, onChange);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener(FILESTORE_EVENT, onChange);
      window.removeEventListener('keydown', onKey);
    };
  }, [open, refresh]);

  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(null), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);

  if (!open) return null;

  const needle = query.trim().toLowerCase();
  const shown = needle ? files.filter((f) => f.name.toLowerCase().includes(needle)) : files;
  const hasNavStorage = typeof navigator !== 'undefined' && navigator.storage !== undefined;
  const canPersist = hasNavStorage && typeof navigator.storage.persist === 'function';

  const addFiles = async (list: File[]): Promise<void> => {
    if (list.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      for (const f of list) {
        const text = await readInputText(f);
        await store.put({ name: f.name, text, size: f.size });
      }
    } catch (e) {
      if (e instanceof FileStoreError && e.code === 'quota') setError('Browser storage is full. Remove some files first.');
      else setError(`Could not store the file: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      if (alive.current) setBusy(false);
      await refresh();
    }
  };

  const removeFile = async (name: string): Promise<void> => {
    setConfirmDelete(null);
    try {
      await store.remove(name);
    } catch (e) {
      setError(`Could not delete ${name}: ${e instanceof Error ? e.message : String(e)}`);
    }
    await refresh();
  };

  const clearAll = async (): Promise<void> => {
    setConfirmClear(false);
    try {
      await store.clear();
    } catch (e) {
      setError(`Could not clear storage: ${e instanceof Error ? e.message : String(e)}`);
    }
    await refresh();
  };

  const download = (f: StoredFile): void => {
    const blob = f.kind === 'gzip'
      ? new Blob([latin1Bytes(f.text)], { type: 'application/gzip' })
      : new Blob([f.text], { type: 'text/plain;charset=utf-8' });
    saveBlob(f.name, blob);
  };

  const copyName = (name: string): void => {
    try {
      const p = navigator.clipboard?.writeText(name);
      p?.then(() => setCopied(name), () => undefined);
    } catch {
      /* no clipboard: skipped silently */
    }
  };

  const keepFiles = async (): Promise<void> => {
    try {
      await navigator.storage.persist();
    } catch {
      /* the browser refused: the status below says so */
    }
    const pers = await readPersisted();
    if (alive.current) setPersisted(pers);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setDragOver(false);
    void addFiles(Array.from(e.dataTransfer?.files ?? []));
  };

  const btn = `inline-flex min-h-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;
  const iconBtn = `inline-flex h-7 w-7 shrink-0 items-center justify-center rounded transition-colors disabled:opacity-50 ${ct.hoverSurface}`;
  const total = totalBytes(files);

  return (
    <div
      role="dialog"
      aria-label="Browser storage"
      className={`fixed bottom-0 right-0 top-0 z-[65] flex w-full min-w-0 flex-col overflow-hidden border-l shadow-xl sm:w-[24rem] ${ct.panel} ${ct.text} ${dragOver ? `ring-2 ring-inset ${ct.focusRing}` : ''}`}
      onDragOver={(e) => {
        if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) {
          e.preventDefault();
          setDragOver(true);
        }
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragOver(false);
      }}
      onDrop={onDrop}
    >
      <header className={`flex items-start justify-between gap-2 border-b p-3 ${ct.divider}`}>
        <div className="min-w-0">
          <h2 className={`text-sm font-semibold ${ct.headerText}`}>Browser storage</h2>
          <p className={`text-[11px] ${ct.muted}`}>
            {files.length} {files.length === 1 ? 'file' : 'files'}, {formatSize(total)} stored
          </p>
          <p className={`text-[11px] ${ct.muted}`}>
            {backend === 'indexeddb' ? 'IndexedDB' : 'memory only: files are lost on reload'}
          </p>
          {estimate && (
            <p className={`text-[11px] ${ct.muted}`}>
              {formatMb(estimate.usage)} of {formatMb(estimate.quota)} used by this site
            </p>
          )}
          {hasNavStorage && (
            <p className={`mt-0.5 text-[11px] ${ct.muted}`}>
              {persisted === true
                ? 'Persistent: the browser keeps these files.'
                : persisted === false
                  ? 'Not persistent: the browser may clear these files when short of space.'
                  : null}
            </p>
          )}
          {canPersist && persisted !== true && (
            <button type="button" className={`${btn} ${ct.button} mt-1`} onClick={() => void keepFiles()}>
              <ShieldCheck size={12} aria-hidden="true" /> Keep files
            </button>
          )}
        </div>
        <button ref={closeRef} type="button" className={iconBtn} aria-label="Close browser storage" onClick={onClose}>
          <X size={14} aria-hidden="true" />
        </button>
      </header>

      <div className={`space-y-2 border-b p-3 ${ct.divider}`}>
        <div className="flex flex-wrap items-center gap-2">
          <button type="button" className={`${btn} ${ct.accent}`} disabled={busy} onClick={() => fileInput.current?.click()}>
            <Upload size={12} aria-hidden="true" /> Add files
          </button>
          <input
            ref={fileInput}
            type="file"
            multiple
            className="hidden"
            aria-label="Add files to browser storage"
            onChange={(e) => {
              const picked = Array.from(e.target.files ?? []);
              e.target.value = '';
              void addFiles(picked);
            }}
          />
          {files.length > 0 && !confirmClear && (
            <button type="button" className={`${btn} ${ct.button}`} disabled={busy} onClick={() => setConfirmClear(true)}>
              Clear all
            </button>
          )}
          {confirmClear && (
            <p className={`flex flex-wrap items-center gap-1 text-xs ${ct.danger}`}>
              <span>Remove all {files.length} files?</span>
              <button type="button" className={`${btn} ${ct.danger}`} onClick={() => void clearAll()}>Yes</button>
              <button type="button" className={`${btn} ${ct.button}`} onClick={() => setConfirmClear(false)}>No</button>
            </p>
          )}
        </div>
        <p className={`text-[11px] ${ct.muted}`}>Or drop files anywhere on this panel.</p>
        {dragOver && <p className={`rounded border border-dashed p-2 text-center text-xs ${ct.accentText}`}>Drop files to store them</p>}
        {error && (
          <p role="alert" className={`flex items-start justify-between gap-2 rounded p-2 text-xs ${ct.errorBox}`}>
            <span>{error}</span>
            <button type="button" className={btn} aria-label="Dismiss storage error" onClick={() => setError(null)}>
              <X size={12} aria-hidden="true" />
            </button>
          </p>
        )}
        <label className={`flex items-center gap-2 rounded border px-2 ${ct.input}`}>
          <Search size={12} aria-hidden="true" className={ct.muted} />
          <input
            type="search"
            aria-label="Search stored files"
            placeholder="Search by name"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-w-0 flex-1 bg-transparent py-1.5 text-xs outline-none"
          />
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-3">
        {files.length === 0 && <p className={`text-xs ${ct.muted}`}>No files are stored yet.</p>}
        {files.length > 0 && shown.length === 0 && <p className={`text-xs ${ct.muted}`}>No file matches “{query.trim()}”.</p>}
        <ul className="space-y-1" aria-label="Stored files">
          {shown.map((f) => {
            const script = f.kind !== 'gzip' && isLammpsScript(f.name, f.text);
            const viewable = isViewable(f);
            const previewOpen = previewName === f.name;
            return (
              <li key={f.name} className={`min-w-0 border-b pb-2 ${ct.divider}`}>
                <div className="flex items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <button
                      type="button"
                      aria-expanded={previewOpen}
                      className="block max-w-full break-all text-left font-mono text-xs underline-offset-2 hover:underline"
                      onClick={() => setPreviewName(previewOpen ? null : f.name)}
                    >
                      {f.name}
                    </button>
                    <div className={`mt-0.5 flex flex-wrap items-center gap-x-2 text-[11px] ${ct.muted}`}>
                      <span className={`rounded px-1 ${ct.chip}`}>{f.kind}</span>
                      <span>{formatSize(f.size)}</span>
                      <span>{formatDate(f.mtime)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {script && (
                        <button type="button" className={`${btn} ${ct.button}`} onClick={() => { onOpenInNotebook(f.text); onClose(); }}>
                          Open in Notebook
                        </button>
                      )}
                      {script && (
                        <button type="button" className={`${btn} ${ct.button}`} onClick={() => { onOpenInBuilder({ text: f.text, name: f.name }); onClose(); }}>
                          Open in Script Builder
                        </button>
                      )}
                      {viewable && (
                        <button type="button" className={`${btn} ${ct.button}`} onClick={() => { onOpenInViewer({ name: f.name, text: f.text }); onClose(); }}>
                          Open in Viewer
                        </button>
                      )}
                      <button type="button" className={`${btn} ${ct.button}`} onClick={() => copyName(f.name)}>
                        <Copy size={12} aria-hidden="true" /> {copied === f.name ? 'Copied' : 'Copy name'}
                      </button>
                    </div>
                    {previewOpen && (
                      <div className="mt-1 max-w-full">
                        {f.kind === 'gzip' ? (
                          <p className={`text-[11px] ${ct.muted}`}>binary (gzip)</p>
                        ) : (
                          <pre className={`max-h-60 max-w-full overflow-auto whitespace-pre rounded p-2 font-mono text-[11px] ${ct.card}`}>{headLines(f.text, 40)}</pre>
                        )}
                      </div>
                    )}
                    {confirmDelete === f.name && (
                      <p className={`mt-1 flex flex-wrap items-center gap-1 text-xs ${ct.danger}`}>
                        <span className="break-all">Delete {f.name}?</span>
                        <button type="button" className={`${btn} ${ct.danger}`} onClick={() => void removeFile(f.name)}>Yes</button>
                        <button type="button" className={`${btn} ${ct.button}`} onClick={() => setConfirmDelete(null)}>No</button>
                      </p>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center">
                    <button type="button" className={iconBtn} aria-label={`Download ${f.name}`} title="Download" onClick={() => download(f)}>
                      <Download size={14} aria-hidden="true" />
                    </button>
                    <button type="button" className={`${iconBtn} ${ct.danger}`} aria-label={`Delete ${f.name}`} title="Delete" onClick={() => setConfirmDelete(f.name)}>
                      <Trash2 size={14} aria-hidden="true" />
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
};

export default StoragePanel;
