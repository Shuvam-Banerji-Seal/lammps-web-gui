import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Copy, Download, FilePlus, FolderOpen, Pencil, Power, Save, Search, Trash2, Upload, X } from 'lucide-react';
import { getThemeTokens } from '../../../theme';
import { getFileStore } from '../../../services/fileStore';
import type { NotebookSession } from './useNotebookSession';
import type { InputFiles } from './useInputFiles';
import {
  exportFileName, exportLammps, exportMarkdown, importNotebookFile, outputsOf, serializeNotebookFile, UNTITLED,
  type ImportResult, type OutputTable,
} from './notebookFile';
import type { Cell } from './cells';

/** The cell output the menu saves with a notebook (the notebook's per-cell run state has this shape). */
export interface RunOutput {
  logs: string[];
  error: string | null;
  tables: OutputTable[];
}

type ExportFormat = 'lmpnb' | 'lammps' | 'markdown';

type Dialog =
  | { kind: 'open' }
  /** `back`: Rename opened from the library list, which is shown again afterwards. */
  | { kind: 'name'; mode: 'saveAs' | 'rename'; id: string | null; back?: boolean }
  | { kind: 'export' }
  | { kind: 'unsaved' }
  | { kind: 'delete'; id: string; name: string }
  | { kind: 'message'; title: string; lines: string[]; error: boolean };

interface NotebookMenuProps {
  theme: Parameters<typeof getThemeTokens>[0];
  /** Class names of the toolbar buttons (the notebook's own). */
  btn: string;
  session: NotebookSession;
  /** The notebook's input files (read through its hook; this menu only adds to them on import). */
  inputFiles: InputFiles;
  /** A run is in progress. Replacing the notebook stops it first, then does the action. */
  busy: boolean;
  onStop: () => void;
  /** Cell output on screen, saved with the notebook when the export asks for it. */
  runs: Record<string, RunOutput>;
}

const fmtBytes = (n: number): string => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`);
const fmtDate = (iso: string): string => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString();
};
/** Above this many bytes of input files, the export warns (a notebook file is a single JSON text). */
const LARGE_INPUTS = 5 * 1024 * 1024;

const downloadText = (fileName: string, text: string, type: string): void => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = fileName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const isBlank = (cells: readonly Cell[]): boolean => cells.every((c) => c.text.trim() === '');

const NotebookMenu: React.FC<NotebookMenuProps> = ({ theme, btn, session, inputFiles, busy, onStop, runs }) => {
  const ct = getThemeTokens(theme);
  const [open, setOpen] = useState(false);
  const [dialog, setDialog] = useState<Dialog | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const menuButton = useRef<HTMLButtonElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  /** The action to run after an unsaved-changes dialog is answered with Save (or Discard). */
  const pendingAction = useRef<(() => void) | null>(null);
  /** An action asked for while a run was in progress: it runs once the run has stopped. */
  const queuedAction = useRef<(() => void) | null>(null);
  const toastTimer = useRef<number | null>(null);

  const say = (text: string) => {
    setToast(text);
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 4000);
  };
  useEffect(() => () => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
  }, []);

  /** Runs `fn` when no unsaved changes would be lost (asks first otherwise). */
  const guard = (fn: () => void) => {
    if (session.saved || (session.baseId === null && isBlank(session.cells))) {
      fn();
      return;
    }
    pendingAction.current = fn;
    setDialog({ kind: 'unsaved' });
  };

  /** Runs `fn` after any unsaved-changes check, stopping a run in progress first. */
  const act = (fn: () => void) => {
    setOpen(false);
    if (!busy) {
      guard(fn);
      return;
    }
    queuedAction.current = () => guard(fn);
    onStop();
  };

  useEffect(() => {
    if (busy || !queuedAction.current) return;
    const f = queuedAction.current;
    queuedAction.current = null;
    f();
  }, [busy]);

  const runPending = () => {
    const f = pendingAction.current;
    pendingAction.current = null;
    setDialog(null);
    f?.();
  };

  const openName = (mode: 'saveAs' | 'rename', id: string | null) => {
    setOpen(false);
    setDialog({ kind: 'name', mode, id });
  };

  const doSave = async () => {
    setOpen(false);
    if (session.baseId === null) {
      openName('saveAs', null);
      return;
    }
    if (await session.save()) say(`Saved "${session.name}".`);
  };

  /** Ctrl/Cmd+S saves and Ctrl/Cmd+O opens, anywhere in the page while the notebook is mounted. */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === 's' && !e.shiftKey) {
        e.preventDefault();
        void doSave();
      } else if (key === 'o' && !e.shiftKey) {
        e.preventDefault();
        setOpen(false);
        setDialog({ kind: 'open' });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // doSave reads the session through its closure; re-register when the session changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session]);

  // Drag and drop: files dropped on the notebook import as a new notebook.
  useEffect(() => {
    const inside = (t: EventTarget | null) => t instanceof Element && t.closest('[data-notebook-root]') !== null;
    let depth = 0;
    const onEnter = (e: DragEvent) => {
      if (!inside(e.target) || !e.dataTransfer?.types.includes('Files')) return;
      depth++;
      setDragging(true);
    };
    const onOver = (e: DragEvent) => {
      if (!inside(e.target) || !e.dataTransfer?.types.includes('Files')) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    };
    const onLeave = (e: DragEvent) => {
      if (!inside(e.target)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setDragging(false);
    };
    const onDrop = (e: DragEvent) => {
      depth = 0;
      setDragging(false);
      if (!inside(e.target) || !e.dataTransfer?.files.length) return;
      e.preventDefault();
      const files = Array.from(e.dataTransfer.files);
      if (files.length > 1) say('Only the first file was imported. Drop one notebook or script at a time.');
      void importFile(files[0]);
    };
    window.addEventListener('dragenter', onEnter);
    window.addEventListener('dragover', onOver);
    window.addEventListener('dragleave', onLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragenter', onEnter);
      window.removeEventListener('dragover', onOver);
      window.removeEventListener('dragleave', onLeave);
      window.removeEventListener('drop', onDrop);
    };
    // importFile reads the session and the guard through closures; re-register when they change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, busy, inputFiles]);

  /** Imports a file: parse first (a bad file never touches the notebook), then replace it as a new library notebook. */
  const importFile = async (file: File) => {
    let imp: ImportResult;
    try {
      imp = importNotebookFile(file.name, await file.text());
    } catch (e) {
      setDialog({ kind: 'message', title: `Could not import ${file.name}`, lines: [e instanceof Error ? e.message : String(e)], error: true });
      return;
    }
    act(() => void doImport(imp));
  };

  const doImport = async (imp: ImportResult) => {
    await session.importNotebook(imp);
    const added = Object.entries(imp.inputFiles ?? {});
    for (const [name, text] of added) {
      try {
        await inputFiles.addText(name, text);
      } catch {
        // the input files panel shows the storage error
      }
    }
    const what = imp.kind === 'lmpnb' ? 'notebook' : imp.kind === 'ipynb' ? 'Jupyter notebook' : 'LAMMPS script';
    const extra = added.length ? ` ${added.length} input file${added.length === 1 ? '' : 's'} added.` : '';
    say(`Imported the ${what} "${imp.name}" (${imp.cells.length} cells) into the library.${extra}`);
    if (imp.notes.length) setDialog({ kind: 'message', title: 'Imported with notes', lines: imp.notes, error: false });
  };

  const doExport = async (format: ExportFormat, withInputs: boolean, withOutputs: boolean) => {
    const name = session.name;
    const modified = new Date().toISOString();
    const created = session.entries.find((e) => e.id === session.baseId)?.created ?? modified;
    let text: string;
    let type: string;
    if (format === 'lmpnb') {
      let files: Record<string, string> | undefined;
      if (withInputs) {
        files = {};
        for (const f of await getFileStore().list()) files[f.name] = f.text;
      }
      text = serializeNotebookFile({
        name, created, modified, cells: session.cells, inputFiles: files,
        outputs: withOutputs ? outputsOf(runs) : undefined,
      });
      type = 'application/json';
    } else if (format === 'lammps') {
      text = exportLammps(session.cells);
      type = 'text/plain';
    } else {
      text = exportMarkdown(name, session.cells);
      type = 'text/markdown';
    }
    const file = exportFileName(name, format);
    downloadText(file, text, type);
    setDialog(null);
    say(`Downloaded ${file}.`);
  };

  const [query, setQuery] = useState('');
  const [draft, setDraft] = useState('');
  const [exportFormat, setExportFormat] = useState<ExportFormat>('lmpnb');
  const [exportInputs, setExportInputs] = useState(false);
  const [exportOutputs, setExportOutputs] = useState(false);

  useEffect(() => {
    if (!dialog) return;
    if (dialog.kind === 'open') {
      setQuery('');
      void session.refresh();
    }
    if (dialog.kind === 'name') {
      const current = dialog.mode === 'rename' && dialog.id !== null
        ? session.entries.find((e) => e.id === dialog.id)?.name ?? session.name
        : session.name;
      setDraft(dialog.mode === 'saveAs' && session.baseId === null ? (session.name || UNTITLED) : current);
    }
    if (dialog.kind === 'export') {
      setExportFormat('lmpnb');
      setExportInputs(false);
      setExportOutputs(false);
    }
    // focus the first field of the dialog
    window.setTimeout(() => {
      const el = document.querySelector<HTMLElement>('[data-nb-dialog] [autofocus]');
      el?.focus();
    }, 0);
  }, [dialog]); // eslint-disable-line react-hooks/exhaustive-deps

  const closeDialog = () => setDialog(null);

  const submitName = async () => {
    if (dialog?.kind !== 'name') return;
    const name = draft.trim() || UNTITLED;
    if (dialog.mode === 'rename') {
      if (dialog.id !== null) await session.rename(dialog.id, name);
      else session.setName(name);
      setDialog(dialog.back ? { kind: 'open' } : null);
      return;
    }
    const ok = await session.saveAs(name);
    if (ok) {
      say(`Saved "${name}" to the library.`);
      runPending();
    } else {
      setDialog(null);
    }
  };

  const entries = useMemo(() => {
    const q = query.trim().toLowerCase();
    return q ? session.entries.filter((e) => e.name.toLowerCase().includes(q)) : session.entries;
  }, [session.entries, query]);

  const inputBytes = inputFiles.totalBytes;
  const itemCls = `flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-xs ${ct.hoverSurface} disabled:opacity-50`;
  const dialogCls = `fixed inset-x-3 top-16 z-50 mx-auto max-h-[80vh] max-w-lg overflow-y-auto rounded-lg border p-4 shadow-xl ${ct.card} ${ct.text}`;

  const dialogBody = (): React.ReactNode => {
    if (!dialog) return null;
    switch (dialog.kind) {
      case 'open':
        return (
          <>
            <h3 className="mb-2 text-sm font-semibold">Open a notebook</h3>
            {session.storage === 'memory' && (
              <p role="note" className={`mb-2 text-xs ${ct.muted}`}>
                Browser storage is not available here, so notebooks last only until this page is closed.
              </p>
            )}
            <label className="mb-2 flex items-center gap-2 text-xs">
              <Search size={13} aria-hidden="true" />
              <span className="sr-only">Search notebooks</span>
              <input autoFocus type="search" value={query} onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by name" className={`min-h-8 min-w-0 flex-1 rounded border px-2 py-1 ${ct.input}`} />
            </label>
            {session.entries.length === 0 && (
              <p className={`text-xs ${ct.muted}`}>No saved notebooks yet. Use Save as to add the open one.</p>
            )}
            {session.entries.length > 0 && entries.length === 0 && <p className={`text-xs ${ct.muted}`}>No notebook matches “{query}”.</p>}
            <ul className="space-y-1" aria-label="Saved notebooks">
              {entries.map((e) => {
                const isOpen = e.id === session.baseId;
                return (
                  <li key={e.id} className={`rounded border p-2 ${ct.divider}`}>
                    <div className="flex flex-wrap items-center gap-2">
                      <button className="min-w-0 flex-1 text-left text-sm font-medium underline-offset-2 hover:underline"
                        disabled={isOpen} onClick={() => act(() => { setDialog(null); void session.openEntry(e.id); })}
                        aria-label={`Open ${e.name}`}>
                        {e.name}{isOpen ? ' (open)' : ''}
                      </button>
                      <span className={`text-[11px] ${ct.muted}`}>{e.cellCount} cells · {fmtDate(e.modified)}</span>
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      <button className={`${btn} ${ct.button}`} onClick={() => setDialog({ kind: 'name', mode: 'rename', id: e.id, back: true })}>
                        <Pencil size={12} aria-hidden="true" />Rename
                      </button>
                      <button className={`${btn} ${ct.button}`} onClick={() => void session.duplicate(e.id)}>
                        <Copy size={12} aria-hidden="true" />Duplicate
                      </button>
                      <button className={`${btn} ${ct.button}`} onClick={() => setDialog({ kind: 'delete', id: e.id, name: e.name })}>
                        <Trash2 size={12} aria-hidden="true" />Delete
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
            <div className="mt-3 flex justify-end">
              <button className={`${btn} ${ct.button}`} onClick={closeDialog}>Close</button>
            </div>
          </>
        );
      case 'name': {
        const rename = dialog.mode === 'rename';
        return (
          <form onSubmit={(e) => { e.preventDefault(); void submitName(); }}>
            <h3 className="mb-2 text-sm font-semibold">{rename ? 'Rename notebook' : 'Save as'}</h3>
            <label className="block text-xs">
              <span className={ct.muted}>Name</span>
              <input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={120}
                className={`mt-1 block min-h-8 w-full rounded border px-2 py-1 text-sm ${ct.input}`} />
            </label>
            <div className="mt-3 flex justify-end gap-2">
              <button type="button" className={`${btn} ${ct.button}`} onClick={() => { pendingAction.current = null; setDialog(dialog.back ? { kind: 'open' } : null); }}>Cancel</button>
              <button type="submit" className={`${btn} ${ct.accent}`}>{rename ? 'Rename' : 'Save'}</button>
            </div>
          </form>
        );
      }
      case 'export':
        return (
          <form onSubmit={(e) => { e.preventDefault(); void doExport(exportFormat, exportInputs, exportOutputs); }}>
            <h3 className="mb-2 text-sm font-semibold">Export “{session.name}”</h3>
            <fieldset className="space-y-1 text-xs">
              <legend className={`mb-1 ${ct.muted}`}>Format</legend>
              {([
                ['lmpnb', 'Notebook file (.lmpnb): cells, and the options below. Opens here again.'],
                ['lammps', 'LAMMPS input (.in): code cells in order, markdown as # comments.'],
                ['markdown', 'Markdown (.md): markdown cells, code in lammps blocks.'],
              ] as const).map(([value, label]) => (
                <label key={value} className="flex items-start gap-2">
                  <input type="radio" name="export-format" value={value} checked={exportFormat === value}
                    onChange={() => setExportFormat(value)} className="mt-0.5" />
                  <span>{label}</span>
                </label>
              ))}
            </fieldset>
            {exportFormat === 'lmpnb' && (
              <fieldset className="mt-3 space-y-1 text-xs">
                <legend className={`mb-1 ${ct.muted}`}>Include</legend>
                <label className="flex items-start gap-2">
                  <input type="checkbox" checked={exportInputs} onChange={(e) => setExportInputs(e.target.checked)} className="mt-0.5" />
                  <span>
                    Input files ({inputFiles.inputs ? Object.keys(inputFiles.inputs).length : 0} files, {fmtBytes(inputBytes)})
                    {exportInputs && inputBytes > LARGE_INPUTS && (
                      <span role="note" className="mt-0.5 block text-amber-600 dark:text-amber-300">
                        Large: the file will be {fmtBytes(inputBytes)} of JSON. Export without them if you only need the cells.
                      </span>
                    )}
                  </span>
                </label>
                <label className="flex items-start gap-2">
                  <input type="checkbox" checked={exportOutputs} onChange={(e) => setExportOutputs(e.target.checked)} className="mt-0.5" />
                  <span>Output on screen (log lines and thermo tables)</span>
                </label>
              </fieldset>
            )}
            <div className="mt-4 flex justify-end gap-2">
              <button type="button" className={`${btn} ${ct.button}`} onClick={closeDialog}>Cancel</button>
              <button type="submit" className={`${btn} ${ct.accent}`}><Download size={12} aria-hidden="true" />Download</button>
            </div>
          </form>
        );
      case 'unsaved':
        return (
          <>
            <h3 className="mb-2 text-sm font-semibold">Unsaved changes</h3>
            <p className="text-xs">“{session.name}” has changes that are not saved to the library.</p>
            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <button className={`${btn} ${ct.button}`} onClick={() => { pendingAction.current = null; closeDialog(); }}>Cancel</button>
              <button className={`${btn} ${ct.button}`} onClick={() => runPending()}>Discard changes</button>
              <button className={`${btn} ${ct.accent}`} autoFocus onClick={() => {
                if (session.baseId === null) {
                  setDialog({ kind: 'name', mode: 'saveAs', id: null });
                  return;
                }
                void session.save().then((ok) => { if (ok) runPending(); else setDialog(null); });
              }}>Save and continue</button>
            </div>
          </>
        );
      case 'delete':
        return (
          <>
            <h3 className="mb-2 text-sm font-semibold">Delete “{dialog.name}”?</h3>
            <p className="text-xs">It is removed from the library. This cannot be undone.</p>
            <div className="mt-3 flex justify-end gap-2">
              <button className={`${btn} ${ct.button}`} onClick={() => setDialog({ kind: 'open' })}>Cancel</button>
              <button className={`${btn} ${ct.accent}`} autoFocus onClick={() => {
                const id = dialog.id;
                setDialog({ kind: 'open' });
                void session.remove(id);
              }}><Trash2 size={12} aria-hidden="true" />Delete</button>
            </div>
          </>
        );
      case 'message':
        return (
          <>
            <h3 className="mb-2 text-sm font-semibold">{dialog.title}</h3>
            {dialog.lines.map((l, i) => <p key={i} className={`mt-1 text-xs ${dialog.error ? 'text-red-600 dark:text-red-300' : ''}`}>{l}</p>)}
            <div className="mt-3 flex justify-end">
              <button className={`${btn} ${ct.button}`} autoFocus onClick={closeDialog}>OK</button>
            </div>
          </>
        );
    }
  };

  const dialogNode = dialog && typeof document !== 'undefined' ? createPortal(
    <div className="fixed inset-0 z-40 bg-black/40" onMouseDown={(e) => { if (e.target === e.currentTarget && dialog.kind !== 'unsaved') closeDialog(); }}
      onKeyDown={(e) => { if (e.key === 'Escape' && dialog.kind !== 'unsaved') closeDialog(); }}>
      <div role="dialog" aria-modal="true" data-nb-dialog="" aria-label={dialog.kind === 'open' ? 'Open a notebook' : 'Notebook dialog'}
        className={dialogCls} onMouseDown={(e) => e.stopPropagation()}>
        {dialog.kind === 'name' && (
          <button className="absolute right-2 top-2 rounded p-1" aria-label="Close" onClick={closeDialog}><X size={14} aria-hidden="true" /></button>
        )}
        {dialogBody()}
      </div>
    </div>,
    document.body,
  ) : null;

  const dragNode = dragging && typeof document !== 'undefined' ? createPortal(
    <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-black/30">
      <p className={`rounded-lg border-2 border-dashed px-6 py-4 text-sm ${ct.card}`}>Drop a notebook (.lmpnb, .ipynb) or a LAMMPS script to import it</p>
    </div>,
    document.body,
  ) : null;

  const toastNode = toast ? (
    <div role="status" className={`pointer-events-none fixed bottom-4 left-1/2 z-50 max-w-[90vw] -translate-x-1/2 rounded-md px-3 py-2 text-xs shadow ${ct.card} ${ct.text}`}>
      {toast}
    </div>
  ) : null;

  const unsavedMark = !session.saved;
  const label = session.name || UNTITLED;

  return (
    <div className="relative flex items-center gap-1">
      <button ref={menuButton} className={`${btn} ${ct.button} max-w-[14rem]`} aria-haspopup="true" aria-expanded={open}
        onClick={() => setOpen((v) => !v)} title={`Notebook: ${label}${unsavedMark ? ' (unsaved changes)' : ''}`}>
        <span className="truncate" data-testid="notebook-name">{label}</span>
        {unsavedMark && <span aria-label="unsaved changes" title="Unsaved changes" className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />}
        <ChevronDown size={13} aria-hidden="true" />
      </button>
      {open && (
        <div role="menu" aria-label="Notebook actions" className={`absolute left-0 top-full z-30 mt-1 w-60 rounded-lg border p-1 shadow-lg ${ct.card} ${ct.text}`}
          onKeyDown={(e) => { if (e.key === 'Escape') { setOpen(false); menuButton.current?.focus(); } }}>
          <button role="menuitem" className={itemCls} onClick={() => act(() => session.newNotebook())}>
            <FilePlus size={13} aria-hidden="true" />New notebook
          </button>
          <button role="menuitem" className={itemCls} onClick={() => { setOpen(false); setDialog({ kind: 'open' }); }}>
            <FolderOpen size={13} aria-hidden="true" />Open… <span className={`ml-auto text-[10px] ${ct.muted}`}>Ctrl+O</span>
          </button>
          <button role="menuitem" className={itemCls} onClick={() => void doSave()}>
            <Save size={13} aria-hidden="true" />Save <span className={`ml-auto text-[10px] ${ct.muted}`}>Ctrl+S</span>
          </button>
          <button role="menuitem" className={itemCls} onClick={() => openName('saveAs', null)}>
            <Save size={13} aria-hidden="true" />Save as…
          </button>
          <button role="menuitem" className={itemCls} onClick={() => openName('rename', session.baseId)}>
            <Pencil size={13} aria-hidden="true" />Rename…
          </button>
          <div className={`my-1 border-t ${ct.divider}`} />
          <button role="menuitem" className={itemCls} onClick={() => { setOpen(false); fileInput.current?.click(); }}>
            <Upload size={13} aria-hidden="true" />Import file…
          </button>
          <button role="menuitem" className={itemCls} onClick={() => { setOpen(false); setDialog({ kind: 'export' }); }}>
            <Download size={13} aria-hidden="true" />Export…
          </button>
          <div className={`my-1 border-t ${ct.divider}`} />
          <button role="menuitem" className={itemCls} onClick={() => act(() => session.newNotebook())}>
            <Power size={13} aria-hidden="true" />Close notebook
            <span className={`ml-auto text-[10px] ${ct.muted}`}>{busy ? 'stops the run' : 'empty notebook'}</span>
          </button>
          {busy && (
            <button role="menuitem" className={itemCls} onClick={() => { setOpen(false); onStop(); }}>
              Stop the run
            </button>
          )}
        </div>
      )}
      <input ref={fileInput} type="file" hidden aria-label="Import a notebook or LAMMPS script"
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (f) void importFile(f);
        }} />
      {session.notice && (
        <button role="status" className={`max-w-[16rem] truncate rounded px-2 py-1 text-left text-[11px] ${ct.errorBox}`}
          onClick={session.dismissNotice} title="Dismiss">
          {session.notice}
        </button>
      )}
      {dialogNode}
      {dragNode}
      {toastNode}
    </div>
  );
};

export default NotebookMenu;
