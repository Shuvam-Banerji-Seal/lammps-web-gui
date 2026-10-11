import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Square, RotateCcw, Cpu, Gpu, Gauge, HelpCircle, FileUp, X, Workflow, Atom } from 'lucide-react';
import ResourceMonitor from './ResourceMonitor';
import { cellsToScript } from '../../lammps/notebookBridge';
import { Cell, duplicateCell, insertAfter, isMarkdownCell, moveCell, reviveCells, runnableCells, withKind } from './notebook/cells';
import MarkdownView from './notebook/MarkdownView';
import CellToolbar, { AddCellRow } from './notebook/CellToolbar';
import CellFlowchart from './notebook/CellFlowchart';
import { explainEngineError } from './engineError';
import { thermoUnit, unitsCaption } from '../../lammps/thermoUnits';
import { getThemeTokens, isDarkTheme, Theme } from '../../theme';
import { usePersistentState } from '../../hooks/usePersistentState';
import MoleculeCanvas from '../MoleculeCanvas';
import { LineChart } from '../charts/SimpleChart';
import RunFiles from './notebook/RunFiles';
import type { VisualizationConfig } from '../../types';
import { EngineClient } from '../../engine/client';
import type { BackendChoice, FromEngine } from '../../engine/protocol';
import type { EngineEvent, ThermoKeyword, ThermoRow } from '../../engine/types';
import { frameToMoleculeData, typeColors, type FrameEvent } from '../../engine/view';
import { filesReadBy, formatNumber } from '../../engine/script';
import { useInputFiles, type ImportSource } from './notebook/useInputFiles';
import { InputFilesPanel } from './notebook/InputFilesPanel';
import { ImportStructureDialog } from './notebook/ImportStructureDialog';
import { TrajectoryBuffer, dumpToRecords, type TrajectoryRecord } from './notebook/trajectory';
import TrajectoryPlayer from './notebook/TrajectoryPlayer';
import RenderPanel from './notebook/RenderPanel';
import ViewControls, { DEFAULT_RECORD_EVERY, reviveBool, reviveRadius, reviveRecordEvery } from './notebook/ViewControls';

/**
 * MD Notebook: LAMMPS-style input in cells, run by the in-browser engine
 * (src/engine) in a Web Worker, with a live 3D view and thermo output.
 * It runs a documented subset of LAMMPS input; it is not LAMMPS.
 */

/** Per-cell view state: collapsed output, flowchart shown, Markdown in the editor. */
interface CellUi { collapsed?: boolean; flow?: boolean; editing?: boolean }

type Status = 'idle' | 'running' | 'ok' | 'error' | 'cancelled';

interface ThermoTable { keywords: ThermoKeyword[]; labels?: string[]; units?: string; rows: ThermoRow[] }

interface CellRun {
  status: Status;
  logs: string[];
  error: string | null;
  tables: ThermoTable[];
}

const STARTER: Cell[] = [
  {
    id: 'c1',
    text: [
      '# 3d Lennard-Jones melt: a small version of the LAMMPS examples/melt setup',
      'units           lj',
      'atom_style      atomic',
      'lattice         fcc 0.8442',
      'region          box block 0 5 0 5 0 5',
      'create_box      1 box',
      'create_atoms    1 box',
      'mass            1 1.0',
    ].join('\n'),
  },
  {
    id: 'c2',
    text: [
      'velocity        all create 3.0 87287',
      'pair_style      lj/cut 2.5',
      'pair_coeff      1 1 1.0 1.0 2.5',
      'fix             1 all nve',
      'thermo          50',
    ].join('\n'),
  },
  { id: 'c3', text: 'run             250' },
];

const STORAGE_KEY = 'm3d.notebook.v1';
const BACKEND_KEY = 'm3d.notebook.backend';
const THREADS_KEY = 'm3d.notebook.threads';
const RECORD_KEY = 'm3d.notebook.recordEvery';
const RADIUS_KEY = 'm3d.notebook.atomRadius';
const BOX_KEY = 'm3d.notebook.showBox';

/** Logical cores the browser reports (at least 1). */
const browserCores = (): number =>
  typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 1;
/** Threads 0 = Auto: the engine picks (engine/device.ts autoThreads, from measurements). */
const AUTO_THREADS = 0;
const reviveThreads = (raw: unknown): number | null =>
  typeof raw === 'number' && Number.isInteger(raw) && raw >= 0 ? Math.min(raw, browserCores()) : null;

const reviveBackend = (raw: unknown): BackendChoice | null => (raw === 'auto' || raw === 'cpu' || raw === 'webgpu' ? raw : null);

const emptyRun = (): CellRun => ({ status: 'idle', logs: [], error: null, tables: [] });

const newId = () => `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

const INTEGER_KEYS = new Set<ThermoKeyword>(['step', 'elapsed', 'atoms']);
const fmtCell = (k: ThermoKeyword, v: number | undefined) =>
  v === undefined ? '' : INTEGER_KEYS.has(k) ? String(v) : formatNumber(v, '%.6g');

const vizConfig = (spacing: number, ntypes: number, dark: boolean): VisualizationConfig => ({
  atomScale: spacing,
  bondScale: 1,
  materialType: 'realistic',
  backgroundColor: dark ? '#151515' : '#f4f5f7',
  showBonds: false,
  customColors: typeColors(ntypes),
  visualizationMode: 'ball-and-stick',
  lightingPreset: 'studio',
  showBox: true,
  showAxes: false,
  showLabels: false,
  shadowsEnabled: false,
  autoRotateSpeed: 0.5,
  fov: 40,
});

interface NotebookProps {
  theme: Theme;
  /** A script sent from the Script Builder, waiting for the user to place it. */
  incoming?: string | null;
  onIncomingTaken?: () => void;
  /** Opens the notebook's cells in the Script Builder (as a new tab). */
  onOpenInBuilder?: (script: string) => void;
  /** Opens a file written by the run, or the recorded trajectory as a dump, in the Molecule 3D Viewer. */
  onOpenInViewer?: (file: { name: string; text: string }) => void;
}

const Notebook: React.FC<NotebookProps> = ({ theme, incoming = null, onIncomingTaken, onOpenInBuilder, onOpenInViewer }) => {
  const ct = getThemeTokens(theme);
  const [cells, setCells] = usePersistentState<Cell[]>(STORAGE_KEY, STARTER, reviveCells);
  const [backend, setBackend] = usePersistentState<BackendChoice>(BACKEND_KEY, 'auto', reviveBackend);
  const [threads, setThreads] = usePersistentState<number>(THREADS_KEY, AUTO_THREADS, reviveThreads);
  /** Run speed (engine 'perf' events) for the resource monitor, and the run's timestep/units. */
  const [perf, setPerf] = useState<Extract<EngineEvent, { kind: 'perf' }> | null>(null);
  const [perfHistory, setPerfHistory] = useState<number[]>([]);
  const [runInfo, setRunInfo] = useState<{ dt: number; units: string } | null>(null);
  const [runs, setRuns] = useState<Record<string, CellRun>>({});
  const [ready, setReady] = useState<Extract<FromEngine, { type: 'ready' }> | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [frame, setFrame] = useState<FrameEvent | null>(null);
  /** Engine frames kept during runs (cleared on reset), the recorded frame shown (null = the live frame), view settings. */
  const [traj] = useState(() => new TrajectoryBuffer());
  const [trajVersion, setTrajVersion] = useState(0);
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const [videoActive, setVideoActive] = useState(false);
  /** Render panel: open, the dump file loaded into the player for it (null = the run's frames), and a load error. */
  const [renderOpen, setRenderOpen] = useState(false);
  const [renderSource, setRenderSource] = useState<{ name: string } | null>(null);
  const [renderNotice, setRenderNotice] = useState<string | null>(null);
  /** The run's own frames, set aside while a file is loaded for rendering. */
  const renderBackup = useRef<{ records: TrajectoryRecord[]; stride: number } | null>(null);
  const [recordEvery, setRecordEvery] = usePersistentState<number>(RECORD_KEY, DEFAULT_RECORD_EVERY, reviveRecordEvery);
  const [atomRadius, setAtomRadius] = usePersistentState<number>(RADIUS_KEY, 1, reviveRadius);
  const [showBox, setShowBox] = usePersistentState<boolean>(BOX_KEY, true, reviveBool);
  const canvasHost = useRef<HTMLDivElement>(null);
  /** The MD run in progress (engine 'run' event): its cell and its first and last step, for the progress bar. */
  const [runSpan, setRunSpan] = useState<{ cell: string; from: number; to: number } | null>(null);
  const [files, setFiles] = useState<Record<string, string>>({});
  const fileInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  /** The file the structure import dialog converts (null when closed). */
  const [importFrom, setImportFrom] = useState<ImportSource | null>(null);
  const [series, setSeries] = useState<ThermoRow[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const helpButton = useRef<HTMLButtonElement>(null);
  /** Reset asks for a second click (within 4 s) when it would drop cell output. */
  const [confirmReset, setConfirmReset] = useState(false);
  /** The last deleted cell and its place, for Undo (offered for 10 s). */
  const [deleted, setDeleted] = useState<{ cell: Cell; index: number } | null>(null);
  /** View state per cell id (see CellUi). */
  const [cellUi, setCellUi] = useState<Record<string, CellUi>>({});
  const patchUi = (id: string, patch: CellUi) => setCellUi((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  useEffect(() => {
    if (!confirmReset) return;
    const t = setTimeout(() => setConfirmReset(false), 4000);
    return () => clearTimeout(t);
  }, [confirmReset]);
  useEffect(() => {
    if (!deleted) return;
    const t = setTimeout(() => setDeleted(null), 10000);
    return () => clearTimeout(t);
  }, [deleted]);
  const clientRef = useRef<EngineClient | null>(null);
  const pendingFrame = useRef<FrameEvent | null>(null);
  const frameTimer = useRef<number | null>(null);
  const stopAll = useRef(false);

  const client = useCallback((): EngineClient => {
    if (!clientRef.current) clientRef.current = new EngineClient();
    return clientRef.current;
  }, []);
  const inputFiles = useInputFiles(client);
  const { inputs } = inputFiles;
  const fetchLibraryFor = inputFiles.fetchLibraryFor;
  const sendInputs = inputFiles.sendToEngine;

  const resetSession = useCallback(async (choice: BackendChoice, nThreads: number) => {
    setReady(null);
    setFrame(null);
    traj.clear();
    renderBackup.current = null;
    setRenderSource(null);
    setRenderNotice(null);
    setTrajVersion((v) => v + 1);
    setViewIndex(null);
    setFiles({});
    setSeries([]);
    setRuns({});
    const info = await client().reset(choice, recordEvery, nThreads);
    sendInputs();
    setReady(info);
  }, [client, recordEvery, traj, sendInputs]);

  useEffect(() => {
    void resetSession(backend, threads);
    return () => {
      clientRef.current?.dispose();
      clientRef.current = null;
      if (frameTimer.current !== null) window.clearTimeout(frameTimer.current);
    };
    // the session is created once; backend changes go through changeBackend
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // frames can arrive every few steps; repaint the 3D view at most ~8x a second
  const showFrame = useCallback((f: FrameEvent) => {
    pendingFrame.current = f;
    if (frameTimer.current !== null) return;
    frameTimer.current = window.setTimeout(() => {
      frameTimer.current = null;
      if (pendingFrame.current) setFrame(pendingFrame.current);
      setTrajVersion((v) => v + 1);
    }, 120);
  }, []);

  const patchRun = (id: string, fn: (r: CellRun) => CellRun) =>
    setRuns((prev) => ({ ...prev, [id]: fn(prev[id] ?? emptyRun()) }));

  const runCell = useCallback(async (cell: Cell): Promise<boolean> => {
    if (cell.text.trim() === '') {
      setRuns((prev) => ({ ...prev, [cell.id]: { ...emptyRun(), status: 'ok', logs: ['Nothing to run: this cell is empty.'] } }));
      return true;
    }
    setRunning(cell.id);
    setRuns((prev) => ({ ...prev, [cell.id]: { ...emptyRun(), status: 'running' } }));
    // potential files the cell names (pair_coeff etc.) that the inputs lack come from the LAMMPS potential library
    const libraryFetch = await fetchLibraryFor(cell.text);
    if (libraryFetch.added.length > 0) {
      patchRun(cell.id, (r) => ({ ...r, logs: [...r.logs, `Added from the LAMMPS potential library (GPL-2.0): ${libraryFetch.added.join(', ')}`] }));
    }
    if (libraryFetch.failed.length > 0) {
      patchRun(cell.id, (r) => ({ ...r, logs: [...r.logs, `Could not fetch from the LAMMPS potential library: ${libraryFetch.failed.join(', ')}`] }));
    }
    const onEvent = (ev: EngineEvent) => {
      switch (ev.kind) {
        case 'log':
          patchRun(cell.id, (r) => ({ ...r, logs: [...r.logs, ev.text] }));
          break;
        case 'error':
          patchRun(cell.id, (r) => ({ ...r, error: ev.message }));
          break;
        case 'thermo-header':
          patchRun(cell.id, (r) => ({ ...r, tables: [...r.tables, { keywords: ev.keywords, labels: ev.labels, units: ev.units, rows: [] }] }));
          break;
        case 'thermo':
          patchRun(cell.id, (r) => {
            if (!r.tables.length) return r;
            const tables = r.tables.slice();
            const last = tables[tables.length - 1];
            tables[tables.length - 1] = { ...last, rows: [...last.rows, ev.row] };
            return { ...r, tables };
          });
          setSeries((s) => [...s, ev.row]);
          break;
        case 'frame':
          traj.append(ev);
          showFrame(ev);
          break;
        case 'run':
          setRunSpan({ cell: cell.id, from: ev.from, to: ev.to });
          // a run records into the player: a loaded file's frames make way for it
          if (renderBackup.current) {
            renderBackup.current = null;
            traj.clear();
            setRenderSource(null);
            setTrajVersion((v) => v + 1);
          }
          setViewIndex(null);
          setRunInfo({ dt: ev.dt, units: ev.units });
          setPerfHistory([]);
          break;
        case 'perf':
          setPerf(ev);
          setPerfHistory((h) => [...h.slice(-59), ev.stepsPerSec]);
          break;
      }
    };
    const onFile = (name: string, text: string, append: boolean) =>
      setFiles((prev) => ({ ...prev, [name]: (append ? prev[name] ?? '' : '') + text }));
    const result = await client().exec(cell.text, 1, { onEvent, onFile });
    patchRun(cell.id, (r) => ({ ...r, status: result.cancelled ? 'cancelled' : result.ok ? 'ok' : 'error' }));
    setRunning(null);
    setRunSpan(null);
    return result.ok;
  }, [client, showFrame, traj, fetchLibraryFor]);

  const runAll = useCallback(async () => {
    stopAll.current = false;
    await resetSession(backend, threads);
    for (const cell of runnableCells(cells)) {
      if (stopAll.current) break;
      const ok = await runCell(cell);
      if (!ok) break;
    }
  }, [backend, threads, cells, resetSession, runCell]);

  const stop = () => {
    stopAll.current = true;
    clientRef.current?.cancel();
  };

  // Switching CPU/GPU or the thread count keeps the system already built.
  const changeBackend = async (choice: BackendChoice, nThreads = threads) => {
    setBackend(choice);
    setThreads(nThreads);
    setReady(null);
    setReady(await client().setBackend(choice, nThreads));
  };

  const updateText = (id: string, text: string) =>
    setCells((prev) => prev.map((c) => (c.id === id ? { ...c, text } : c)));

  /** Insert an empty cell of `kind` after the cell at `index`. */
  const addBelow = (index: number, kind: 'code' | 'markdown') => {
    const cell = withKind({ id: newId(), text: '' }, kind);
    setCells((prev) => insertAfter(prev, index, cell));
    if (kind === 'markdown') patchUi(cell.id, { editing: true });
  };

  const moveBy = (id: string, delta: number) => {
    setCells((prev) => moveCell(prev, prev.findIndex((c) => c.id === id), delta));
    // keep keyboard focus on the moved cell
    window.requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-cell-id="${id}"]`)?.focus());
  };

  const duplicateAt = (index: number) => setCells((prev) => duplicateCell(prev, index, newId()));

  const toggleKind = (id: string) =>
    setCells((prev) => prev.map((c) => (c.id === id ? withKind(c, isMarkdownCell(c) ? 'code' : 'markdown') : c)));

  const removeCell = (index: number) => {
    if (cells.length <= 1) return;
    const cell = cells[index];
    setDeleted({ cell, index });
    setCells((prev) => prev.filter((c) => c.id !== cell.id));
  };
  const undoDelete = () => {
    if (!deleted) return;
    const { cell, index } = deleted;
    setCells((prev) => [...prev.slice(0, index), cell, ...prev.slice(index)]);
    setDeleted(null);
  };

  /** Loads a dump the run wrote into the player (the run's frames are set aside) and opens the Render panel on it. */
  const loadFileForRender = useCallback((file: { name: string; text: string }) => {
    setRenderOpen(true);
    // a run writes its dump as it goes: a partial file would render a partial trajectory
    if (running !== null) {
      setRenderNotice(`${file.name} is still being written. Render it when the run finishes.`);
      return;
    }
    let records: TrajectoryRecord[];
    try {
      records = dumpToRecords(file.text);
    } catch (e) {
      setRenderNotice(`Could not read ${file.name}: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    if (records.length === 0) { setRenderNotice(`${file.name} has no frames.`); return; }
    if (!renderBackup.current) renderBackup.current = { records: traj.list(), stride: traj.stride };
    traj.restore(records);
    setTrajVersion((v) => v + 1);
    setViewIndex(null);
    setRenderSource({ name: file.name });
    setRenderNotice(null);
  }, [traj, running]);

  const restoreRunFrames = () => {
    const backup = renderBackup.current;
    if (!backup) return;
    renderBackup.current = null;
    traj.restore(backup.records, backup.stride);
    setTrajVersion((v) => v + 1);
    setViewIndex(null);
    setRenderSource(null);
    setRenderNotice(null);
  };

  /** The frame the view draws: the recorded frame picked in the player, else the live frame. */
  const shownFrame = useMemo(
    () => (viewIndex !== null ? traj.toFrameEvent(viewIndex) : frame),
    // trajVersion: the buffer is mutable, and its version changes with every recorded frame
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [viewIndex, trajVersion, frame, traj],
  );
  const data = useMemo(() => (shownFrame ? frameToMoleculeData(shownFrame) : null), [shownFrame]);
  const config = useMemo(() => {
    if (!shownFrame || !data) return vizConfig(1, 1, isDarkTheme(theme));
    const { lo, hi } = shownFrame.box;
    const dims = hi[2] - lo[2] > 0 && shownFrame.x.some((_, k) => k % 3 === 2 && shownFrame.x[k] !== 0) ? 3 : 2;
    const vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (dims === 3 ? hi[2] - lo[2] : 1);
    const spacing = Math.pow(vol / Math.max(1, data.atoms.length), 1 / dims);
    return { ...vizConfig(spacing * atomRadius, Math.max(1, ...Object.keys(data.atomTypes).map(Number)), isDarkTheme(theme)), showBox };
  }, [shownFrame, data, theme, atomRadius, showBox]);

  const chart = useMemo(() => {
    const pick = (k: ThermoKeyword) => series.filter((r) => r.step !== undefined && r[k] !== undefined)
      .map((r) => ({ x: r.step as number, y: r[k] as number }));
    return { temp: pick('temp'), etotal: pick('etotal') };
  }, [series]);

  const busy = running !== null;

  const takeIncoming = (mode: 'replace' | 'append') => {
    if (incoming === null) return;
    const cell = { id: newId(), text: incoming };
    setCells((prev) => (mode === 'replace' ? [cell] : [...prev, cell]));
    onIncomingTaken?.();
  };
  // A script sent from the Script Builder is placed at once when nothing would be lost: the notebook is empty or
  // still holds only the untouched starter cells, and no run is going. Otherwise the banner below asks first.
  const [placedFromBuilder, setPlacedFromBuilder] = useState(false);
  useEffect(() => {
    if (incoming === null || busy) return;
    const untouched = cells.every((c) => !c.text.trim())
      || (cells.length === STARTER.length && cells.every((c, i) => c.text === STARTER[i].text && isMarkdownCell(c) === isMarkdownCell(STARTER[i])));
    if (!untouched) return;
    takeIncoming('replace');
    setPlacedFromBuilder(true);
  }, [incoming, busy, cells]);
  const incomingNeeds = incoming === null ? [] : filesReadBy(incoming).filter((f) => !(f in inputs));

  const addInputFiles = async (list: FileList | null) => {
    const candidate = await inputFiles.addPicked(list);
    if (candidate) setImportFrom(candidate);
  };
  const removeInputFile = (name: string) => void inputFiles.remove(name);
  /** Appends a code cell (the suggestion of the structure import dialog). */
  const insertCell = (text: string) => setCells((prev) => [...prev, { id: newId(), text }]);
  const webgpuAvailable = ready?.webgpuAvailable ?? (typeof navigator !== 'undefined' && 'gpu' in navigator);
  const cores = ready?.cores ?? browserCores();
  const btn = `inline-flex min-h-6 min-w-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;

  return (
    <div className={`flex h-full min-h-0 flex-col ${ct.bg} ${ct.text}`}>
      {/* toolbar */}
      <div className={`flex shrink-0 flex-wrap items-center gap-2 border-b px-3 py-2 ${ct.divider}`}>
        <h2 className={`mr-1 text-sm font-semibold ${ct.headerText}`}>MD Notebook</h2>
        <button className={`${btn} ${ct.accent}`} onClick={() => void runAll()} disabled={busy}
          title="Restart the session and run every cell in order">
          <Play size={13} aria-hidden="true" />Run all
        </button>
        <button className={`${btn} ${ct.button}`} onClick={stop} disabled={!busy} aria-label="Stop the run" title="Stop the run">
          <Square size={13} aria-hidden="true" />Stop
        </button>
        <button className={`${btn} ${confirmReset ? ct.accent : ct.button}`} disabled={busy}
          onClick={() => {
            const hasOutput = Object.values(runs).some((r) => r.status !== 'idle');
            if (hasOutput && !confirmReset) { setConfirmReset(true); return; }
            setConfirmReset(false);
            void resetSession(backend, threads);
          }}
          title="Drop every atom, fix and variable and start over (cell output is cleared; the cells are kept)">
          <RotateCcw size={13} aria-hidden="true" />{confirmReset ? 'Click again to reset' : 'Reset session'}
        </button>
        <div role="radiogroup" aria-label="Compute device" className={`flex overflow-hidden rounded border ${ct.divider}`}>
          {([['auto', 'Auto', <Gauge key="i" size={13} aria-hidden="true" />], ['cpu', 'CPU', <Cpu key="i" size={13} aria-hidden="true" />], ['webgpu', 'GPU', <Gpu key="i" size={13} aria-hidden="true" />]] as const)
            .map(([value, text, icon]) => (
              <button key={value} role="radio" aria-checked={backend === value}
                disabled={busy || (value === 'webgpu' && !webgpuAvailable)}
                title={value === 'auto'
                  ? 'Let the engine choose from this device: a hardware GPU if there is one, else the CPU on several threads'
                  : value === 'webgpu'
                    ? (webgpuAvailable ? 'Compute forces on the GPU with WebGPU' : 'This browser has no WebGPU')
                    : 'Compute forces on the CPU (fp64), on the chosen number of threads'}
                onClick={() => { if (backend !== value) void changeBackend(value); }}
                className={`${btn} rounded-none ${backend === value ? ct.accent : ct.button}`}>
                {icon}{text}
              </button>
            ))}
        </div>
        <label className="flex items-center gap-1 text-xs"
          title="CPU threads used for force computation (with the GPU: for the runs the GPU path cannot take). Your browser reports this many logical cores">
          <span className={ct.muted}>Threads</span>
          <select aria-label="CPU threads" value={threads === AUTO_THREADS ? AUTO_THREADS : Math.min(threads, cores)} disabled={busy}
            onChange={(e) => void changeBackend(backend, Number(e.target.value))}
            className={`min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`}>
            <option value={AUTO_THREADS}>Auto{ready && threads === AUTO_THREADS ? ` (${ready.threads})` : ''}</option>
            {Array.from({ length: cores }, (_, k) => k + 1).map((k) => (
              <option key={k} value={k}>{k}{k === cores ? ' (all)' : ''}</option>
            ))}
          </select>
          <span className={ct.muted}>of {cores}</span>
        </label>
        <span className={`text-xs ${ct.muted}`} aria-live="polite">
          {ready ? `${ready.auto && backend === 'auto' ? 'Auto: ' : ''}${ready.backend}${ready.note ? ` — ${ready.note}` : ''}` : 'starting engine…'}
        </span>
        <button className={`${btn} ${ct.button}`} onClick={() => fileInput.current?.click()}
          title="Add data, include or potential files; scripts refer to them by file name">
          <FileUp size={13} aria-hidden="true" />Add files
        </button>
        <input ref={fileInput} type="file" multiple hidden aria-label="Add input files"
          onChange={(e) => { void addInputFiles(e.target.files); e.target.value = ''; }} />
        <button className={`${btn} ${ct.button}`} onClick={() => importInput.current?.click()}
          title="Convert a PDB, CIF, XYZ or LAMMPS data file into LAMMPS input files (data file or molecule template)">
          <Atom size={13} aria-hidden="true" />Import structure…
        </button>
        <input ref={importInput} type="file" hidden aria-label="Import structure file"
          accept=".pdb,.ent,.cif,.mmcif,.xyz,.data,.lmp,.lammps"
          onChange={async (e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (f) setImportFrom(await inputFiles.readForImport(f));
          }} />
        {onOpenInBuilder && (
          <button className={`${btn} ${ct.button}`} disabled={cells.every((c) => isMarkdownCell(c) || c.text.trim() === '')}
            onClick={() => onOpenInBuilder(cellsToScript(cells))}
            title="Open these cells in the Script Builder as a new tab (flowchart, validator, export)"
            aria-label="Open these cells in the Script Builder">
            <Workflow size={13} aria-hidden="true" />Open in Script Builder
          </button>
        )}
        <button ref={helpButton} className={`${btn} ml-auto ${ct.button}`} onClick={() => setShowHelp((v) => !v)} aria-expanded={showHelp}
          aria-label="What the notebook supports">
          <HelpCircle size={13} aria-hidden="true" />
        </button>
      </div>
      {incoming !== null && (
        <div role="region" aria-label="Script from the Script Builder"
          className={`flex shrink-0 flex-wrap items-center gap-2 border-b px-3 py-2 text-xs ${ct.divider} ${ct.panel}`}>
          <span className="font-medium">Script from the Script Builder ({incoming.split('\n').length} lines), not placed yet: your cells would change</span>
          <button className={`${btn} ${ct.accent}`} disabled={busy} onClick={() => takeIncoming('replace')}
            title="Replace every cell, and its output, with this script">
            Replace {cells.length === 1 ? 'the cell' : `all ${cells.length} cells`}
          </button>
          <button className={`${btn} ${ct.button}`} disabled={busy} onClick={() => takeIncoming('append')}
            title="Add the script as a new cell after the current ones">
            Add as a new cell
          </button>
          <button className={`${btn} ${ct.button}`} onClick={() => onIncomingTaken?.()}>Dismiss</button>
          {incomingNeeds.length > 0 && (
            <span className={ct.muted}>
              It reads {incomingNeeds.map((f) => <code key={f} className="font-mono">{f}</code>).reduce<React.ReactNode[]>((a, el, i) => (i ? [...a, ', ', el] : [el]), [])}
              {' '}— add {incomingNeeds.length === 1 ? 'it' : 'them'} with Add files.
            </span>
          )}
        </div>
      )}
      {placedFromBuilder && incoming === null && (
        <div role="status" className={`flex shrink-0 items-center gap-2 border-b px-3 py-1.5 text-xs ${ct.divider} ${ct.panel}`}>
          <span>The script from the Script Builder is in the notebook. It has not run yet: press Run all.</span>
          <button className={`${btn} ${ct.button}`} aria-label="Dismiss" onClick={() => setPlacedFromBuilder(false)}><X size={12} aria-hidden="true" /></button>
        </div>
      )}
      {deleted && (
        <div role="status" className={`flex shrink-0 items-center gap-2 border-b px-3 py-1.5 text-xs ${ct.divider} ${ct.panel}`}>
          <span>Cell {deleted.index + 1} deleted.</span>
          <button className={`${btn} ${ct.button}`} onClick={undoDelete}>Undo</button>
          <button className={`${btn} ${ct.button}`} aria-label="Dismiss" onClick={() => setDeleted(null)}><X size={12} aria-hidden="true" /></button>
        </div>
      )}
      {showHelp && (
        <div role="region" aria-label="What the notebook supports"
          onKeyDown={(e) => { if (e.key === 'Escape') { setShowHelp(false); helpButton.current?.focus(); } }}
          className={`relative max-h-[40vh] shrink-0 overflow-y-auto border-b px-3 py-2 pr-9 text-xs leading-relaxed ${ct.divider} ${ct.panel}`}>
          <button className={`${btn} absolute right-2 top-2 ${ct.button}`} aria-label="Close help"
            onClick={() => { setShowHelp(false); helpButton.current?.focus(); }}>
            <X size={12} aria-hidden="true" />
          </button>
          <p>
            This notebook runs a documented subset of LAMMPS input in your browser with an independent engine,
            checked against native LAMMPS. It is not LAMMPS; a command or style outside the subset stops with an
            error that names it.
          </p>
          <p className={`mt-1 font-mono ${ct.muted}`}>{ready ? ready.commands.join(' · ') : 'starting engine…'}</p>
          {ready && (
            <dl className="mt-1 grid grid-cols-[max-content_minmax(0,1fr)] gap-x-2 font-mono">
              {Object.entries(ready.styles).filter(([, names]) => names.length > 0).map(([cmd, names]) => (
                <React.Fragment key={cmd}>
                  <dt className={ct.muted}>{cmd}</dt>
                  <dd className="break-words">{names.join(' · ')}</dd>
                </React.Fragment>
              ))}
            </dl>
          )}
          <p className={`mt-1 ${ct.muted}`}>
            Shift+Enter runs a cell. Run all restarts the session first. Add files makes data, include and
            potential files readable by name (read_data, include, pair_coeff). Open in Script Builder opens the
            cells as a new Script Builder tab; its Run in Notebook sends a script back here.
          </p>
        </div>
      )}

      {/* phone: one scrolling column (view, cells, charts); desktop: cells left, view + charts right */}
      <div className="min-h-0 flex-1 overflow-y-auto lg:grid lg:grid-cols-[minmax(0,1fr)_42%] lg:grid-rows-[minmax(0,55%)_minmax(0,1fr)] lg:overflow-hidden">
        <div className={`flex flex-col border-b lg:col-start-2 lg:row-start-1 lg:min-h-0 lg:border-b-0 lg:border-l ${ct.divider}`}>
          <div ref={canvasHost} className="relative h-56 lg:h-auto lg:min-h-0 lg:flex-1">
            {data ? (
              <MoleculeCanvas data={data} autoRotate={false} config={config} forceContinuousRender={videoActive} />
            ) : (
              <div className={`flex h-full items-center justify-center p-4 text-center text-xs ${ct.muted}`}>
                Run a cell with create_atoms to see the system here.
              </div>
            )}
            {shownFrame && (
              <span className={`absolute left-2 top-2 rounded px-1.5 py-0.5 text-[11px] ${ct.chip}`}>
                {shownFrame.id.length} atoms · step {shownFrame.step}
              </span>
            )}
          </div>
          <div className="shrink-0 px-2 pt-2">
            <ViewControls ct={ct} recordEvery={recordEvery} onRecordEvery={setRecordEvery} radius={atomRadius} onRadius={setAtomRadius}
              showBox={showBox} onShowBox={setShowBox} />
          </div>
          <TrajectoryPlayer ct={ct} btn={btn} buffer={traj} version={trajVersion} index={viewIndex} live={running !== null}
            onShow={setViewIndex} canvasHost={canvasHost} onVideoActive={setVideoActive} onOpenInViewer={onOpenInViewer}
            onRender={() => setRenderOpen(true)} />
          {renderOpen && (
            <RenderPanel ct={ct} btn={btn} buffer={traj} version={trajVersion} index={viewIndex}
              dt={runInfo?.dt ?? null} units={runInfo?.units} live={running !== null}
              source={renderSource} canRestore={renderSource !== null} onRestore={restoreRunFrames}
              notice={renderNotice} canvasHost={canvasHost} onShow={setViewIndex} onVideoActive={setVideoActive}
              onClose={() => setRenderOpen(false)} />
          )}
        </div>

        {/* cells */}
        <div className="space-y-3 p-3 lg:col-start-1 lg:row-span-2 lg:row-start-1 lg:overflow-y-auto">
          {cells.map((cell, i) => {
            const run = runs[cell.id] ?? emptyRun();
            const n = i + 1;
            const ui = cellUi[cell.id] ?? {};
            const md = isMarkdownCell(cell);
            // a Markdown cell is rendered until Edit (or double-click); new empty ones open in the editor
            const editing = md && ui.editing === true;
            const hasOutput = run.logs.length > 0 || run.error !== null || run.tables.length > 0;
            return (
              <section key={cell.id} data-cell="" data-cell-id={cell.id} data-kind={md ? 'markdown' : 'code'} data-status={run.status}
                tabIndex={-1}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
                    e.preventDefault();
                    moveBy(cell.id, e.key === 'ArrowUp' ? -1 : 1);
                  }
                }}
                className={`rounded-lg border p-2 ${ct.card} ${run.status === 'error' ? ct.errorAccent : ''}`} aria-label={`Cell ${n}`}>
                <CellToolbar ct={ct} n={n} kind={md ? 'markdown' : 'code'} busy={busy} isFirst={i === 0} isLast={i === cells.length - 1}
                  canDelete={cells.length > 1} status={run.status === 'idle' ? '' : run.status} statusError={run.status === 'error'}
                  collapsed={!!ui.collapsed} flowOpen={!!ui.flow} editing={editing}
                  onRun={() => void runCell(cell)}
                  onToggleKind={() => {
                    toggleKind(cell.id);
                    if (!md && cell.text.trim() === '') patchUi(cell.id, { editing: true });
                  }}
                  onToggleEdit={() => patchUi(cell.id, { editing: !editing })}
                  onMoveUp={() => moveBy(cell.id, -1)}
                  onMoveDown={() => moveBy(cell.id, 1)}
                  onDuplicate={() => duplicateAt(i)}
                  onToggleCollapse={() => patchUi(cell.id, { collapsed: !ui.collapsed })}
                  onToggleFlow={() => patchUi(cell.id, { flow: !ui.flow })}
                  onDelete={() => removeCell(i)}>
                  {running === cell.id && runSpan?.cell === cell.id && runSpan.to > runSpan.from && (
                    <RunProgress ct={ct} from={runSpan.from} to={runSpan.to} step={frame?.step ?? runSpan.from} />
                  )}
                </CellToolbar>
                {md ? (
                  editing ? (
                    <textarea
                      aria-label={`Cell ${n} text`}
                      placeholder="Markdown: # heading, **bold**, - list, [link](https://...). Shift+Enter renders."
                      value={cell.text}
                      spellCheck
                      rows={Math.max(3, cell.text.split('\n').length + 1)}
                      onChange={(e) => updateText(cell.id, e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && e.shiftKey) {
                          e.preventDefault();
                          patchUi(cell.id, { editing: false });
                        }
                      }}
                      className={`block w-full resize-y rounded border px-2 py-1 font-mono text-xs leading-relaxed ${ct.input}`}
                    />
                  ) : (
                    <div onDoubleClick={() => patchUi(cell.id, { editing: true })} aria-label={`Cell ${n} rendered text`}
                      className="cursor-text px-1 py-1">
                      <MarkdownView ct={ct} source={cell.text} />
                    </div>
                  )
                ) : (
                  <>
                    <textarea
                    aria-label={`Cell ${n} input`}
                    placeholder="LAMMPS input, e.g. run 100 (Shift+Enter runs this cell)"
                    value={cell.text}
                    spellCheck={false}
                    rows={Math.max(2, cell.text.split('\n').length + 1)}
                    onChange={(e) => updateText(cell.id, e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && e.shiftKey) {
                        e.preventDefault();
                        if (!busy) void runCell(cell);
                      }
                    }}
                    className={`block w-full resize-y rounded border px-2 py-1 font-mono text-xs leading-relaxed ${ct.input}`}
                  />
                  {ui.flow && <CellFlowchart theme={theme} n={n} text={cell.text} />}
                  {ui.collapsed ? (
                    hasOutput && <p className={`mt-1 font-sans text-[11px] ${ct.muted}`}>Output hidden.</p>
                  ) : (
                  <div role="log" aria-label={`Cell ${n} output`} className="mt-1 space-y-1 font-mono text-[11px]">
                    {run.logs.map((l, k) => <p key={k} className={ct.muted}>{l}</p>)}
                    {run.error && <EngineErrorText ct={ct} message={run.error} />}
                    {run.tables.map((t, k) => (
                      <div key={k} className="max-h-64 overflow-auto">
                        {unitsCaption(t.units) && <p className={`mb-0.5 font-sans ${ct.muted}`}>{unitsCaption(t.units)}</p>}
                        <table className="border-collapse text-right">
                          <thead>
                            <tr>{t.keywords.map((kw, c) => <th key={kw} title={thermoUnit(t.units, kw) ?? undefined} className={`px-2 font-semibold ${ct.headerText}`}>{t.labels?.[c] ?? kw}</th>)}</tr>
                          </thead>
                          <ThermoRows ct={ct} keywords={t.keywords} rows={t.rows} />
                        </table>
                      </div>
                    ))}
                  </div>
                  )}
                  </>
                )}
                <AddCellRow ct={ct} n={n} onAdd={(kind) => addBelow(i, kind)} />
              </section>
            );
          })}
        </div>

        <div className={`space-y-2 border-t p-3 lg:col-start-2 lg:row-start-2 lg:overflow-y-auto lg:border-l lg:border-t-0 ${ct.divider}`}>
          <ResourceMonitor ct={ct} ready={ready} perf={perf} history={perfHistory} runInfo={runInfo} running={running !== null} />
          {chart.temp.length > 1 && (
            <div>
              <p className={`text-[11px] font-semibold ${ct.muted}`}>Temperature</p>
              <LineChart data={chart.temp} xLabel="step" yLabel="temp" theme={theme} height={120} />
            </div>
          )}
          {chart.etotal.length > 1 && (
            <div>
              <p className={`text-[11px] font-semibold ${ct.muted}`}>Total energy</p>
              <LineChart data={chart.etotal} xLabel="step" yLabel="etotal" theme={theme} height={120} />
            </div>
          )}
          <InputFilesPanel theme={theme} inputs={inputs} totalBytes={inputFiles.totalBytes} error={inputFiles.error}
            busy={busy} onRemove={removeInputFile} onClearAll={() => void inputFiles.clearAll()}
            onDismissError={inputFiles.dismissError} libraryNames={inputFiles.libraryNames} onAddLibrary={inputFiles.addLibrary}
            pairStyles={ready?.styles.pair_style ?? null} />
          <RunFiles ct={ct} files={files} cells={cells} runs={runs} onOpenInViewer={onOpenInViewer} onRenderFile={loadFileForRender} />
          {chart.temp.length <= 1 && Object.keys(files).length === 0 && Object.keys(inputs).length === 0 && (
            <p className={`text-xs ${ct.muted}`}>
              Thermo charts, files written by dump / write_data, and the files you add with Add files appear here.
            </p>
          )}
          {importFrom && (
            <ImportStructureDialog theme={theme} source={importFrom} onClose={() => setImportFrom(null)}
              onAddText={inputFiles.addText} onInsertCell={insertCell} />
          )}
        </div>
      </div>
    </div>
  );
};

/** Rows kept in the page per thermo table; long runs show the first row and the latest ones, the rest on demand. */
const THERMO_ROWS_SHOWN = 200;

/**
 * Thermo table body. A long run (thermo every few steps over many steps) used to put every row in the page: a
 * 20,000-step run with thermo 50 made 398 rows and a 143,000-character accessibility tree. Now the first row and the
 * latest THERMO_ROWS_SHOWN - 1 are shown, with a row that says how many are hidden and shows them all on request.
 */

export const ThermoRows: React.FC<{ ct: ReturnType<typeof getThemeTokens>; keywords: ThermoKeyword[]; rows: ThermoRow[] }> = ({ ct, keywords, rows }) => {
  const [all, setAll] = useState(false);
  const hidden = all ? 0 : Math.max(0, rows.length - THERMO_ROWS_SHOWN);
  const row = (r: ThermoRow, j: number) => (
    <tr key={j}>{keywords.map((kw) => <td key={kw} className="px-2">{fmtCell(kw, r[kw])}</td>)}</tr>
  );
  return (
    <tbody>
      {hidden > 0 ? (
        <>
          {row(rows[0], 0)}
          <tr>
            <td colSpan={keywords.length} className={`px-2 py-0.5 text-center font-sans ${ct.muted}`}>
              … {hidden} earlier rows hidden{' '}
              <button className="underline" onClick={() => setAll(true)}>show all {rows.length}</button>
            </td>
          </tr>
          {rows.slice(hidden + 1).map((r, j) => row(r, hidden + 1 + j))}
        </>
      ) : rows.map(row)}
    </tbody>
  );
};

/** Progress of the MD run in a cell: a bar and "step S of T (P%)" (the step of the latest frame shown). */
const RunProgress: React.FC<{ ct: ReturnType<typeof getThemeTokens>; from: number; to: number; step: number }> = ({ ct, from, to, step }) => {
  const s = Math.min(to, Math.max(from, step));
  const pct = Math.round((100 * (s - from)) / (to - from));
  return (
    <span className="flex items-center gap-1.5">
      <span role="progressbar" aria-label="Run progress" aria-valuemin={from} aria-valuemax={to} aria-valuenow={s}
        className={`h-1.5 w-24 overflow-hidden rounded ${ct.track}`}>
        <span className={`block h-full ${ct.trackFill}`} style={{ width: `${pct}%` }} />
      </span>
      <span className={`font-mono text-[11px] ${ct.muted}`}>step {s} of {to} ({pct}%)</span>
    </span>
  );
};

/** An engine error: its head, a "did you mean" for an unknown name, and the supported list folded away. */
const EngineErrorText: React.FC<{ ct: ReturnType<typeof getThemeTokens>; message: string }> = ({ ct, message }) => {
  const v = explainEngineError(message);
  return (
    <div role="alert" className={`rounded px-1.5 py-1 ${ct.errorBox}`}>
      <p>{v.head}</p>
      {v.suggestion && <p className="mt-0.5 font-semibold">Did you mean <code>{v.suggestion}</code>?</p>}
      {v.supported.length > 0 && (
        <details className="mt-0.5">
          <summary className="cursor-pointer">Supported ({v.supported.length})</summary>
          <p className="mt-0.5 break-words">{v.supported.join(' · ')}</p>
        </details>
      )}
    </div>
  );
};

export default Notebook;
