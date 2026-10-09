import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, Square, Plus, Trash2, RotateCcw, Download, Cpu, Gpu, Gauge, HelpCircle, FileUp, X, Workflow } from 'lucide-react';
import ResourceMonitor from './ResourceMonitor';
import { cellsToScript } from '../../lammps/notebookBridge';
import { explainEngineError } from './engineError';
import { thermoUnit, unitsCaption } from '../../lammps/thermoUnits';
import { getThemeTokens, isDarkTheme, Theme } from '../../theme';
import { usePersistentState } from '../../hooks/usePersistentState';
import MoleculeCanvas from '../MoleculeCanvas';
import { LineChart } from '../charts/SimpleChart';
import { downloadFile, isImageDataUrl } from '../../lammps/exporter';
import type { VisualizationConfig } from '../../types';
import { EngineClient } from '../../engine/client';
import type { BackendChoice, FromEngine } from '../../engine/protocol';
import type { EngineEvent, ThermoKeyword, ThermoRow } from '../../engine/types';
import { frameToMoleculeData, typeColors, type FrameEvent } from '../../engine/view';
import { filesReadBy, formatNumber } from '../../engine/script';

/**
 * MD Notebook: LAMMPS-style input in cells, run by the in-browser engine
 * (src/engine) in a Web Worker, with a live 3D view and thermo output.
 * It runs a documented subset of LAMMPS input; it is not LAMMPS.
 */

interface Cell { id: string; text: string }

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

/** Logical cores the browser reports (at least 1). */
const browserCores = (): number =>
  typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 0 ? navigator.hardwareConcurrency : 1;
/** Threads 0 = Auto: the engine picks (engine/device.ts autoThreads, from measurements). */
const AUTO_THREADS = 0;
const reviveThreads = (raw: unknown): number | null =>
  typeof raw === 'number' && Number.isInteger(raw) && raw >= 0 ? Math.min(raw, browserCores()) : null;

const reviveCells = (raw: unknown): Cell[] | null => {
  if (!Array.isArray(raw)) return null;
  const cells = raw.filter((c): c is Cell =>
    !!c && typeof (c as Cell).id === 'string' && typeof (c as Cell).text === 'string');
  return cells.length ? cells : null;
};

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

const formatBytes = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

interface NotebookProps {
  theme: Theme;
  /** A script sent from the Script Builder, waiting for the user to place it. */
  incoming?: string | null;
  onIncomingTaken?: () => void;
  /** Opens the notebook's cells in the Script Builder (as a new tab). */
  onOpenInBuilder?: (script: string) => void;
}

const Notebook: React.FC<NotebookProps> = ({ theme, incoming = null, onIncomingTaken, onOpenInBuilder }) => {
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
  /** The MD run in progress (engine 'run' event): its cell and its first and last step, for the progress bar. */
  const [runSpan, setRunSpan] = useState<{ cell: string; from: number; to: number } | null>(null);
  const [files, setFiles] = useState<Record<string, string>>({});
  /** Images written by dump image (data URLs), in write order; the preview follows the newest unless one is picked. */
  const imageNames = useMemo(() => Object.keys(files).filter((n) => isImageDataUrl(files[n])), [files]);
  const [imagePick, setImagePick] = useState<number | null>(null);
  /** Files the user added for read_data / include / potential files: name -> size in bytes. */
  const [inputs, setInputs] = useState<Record<string, number>>({});
  const fileInput = useRef<HTMLInputElement>(null);
  const [series, setSeries] = useState<ThermoRow[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const helpButton = useRef<HTMLButtonElement>(null);
  /** Reset asks for a second click (within 4 s) when it would drop cell output. */
  const [confirmReset, setConfirmReset] = useState(false);
  /** The last deleted cell and its place, for Undo (offered for 10 s). */
  const [deleted, setDeleted] = useState<{ cell: Cell; index: number } | null>(null);
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

  const resetSession = useCallback(async (choice: BackendChoice, nThreads: number) => {
    setReady(null);
    setFrame(null);
    setFiles({});
    setSeries([]);
    setRuns({});
    const info = await client().reset(choice, 25, nThreads);
    setReady(info);
  }, [client]);

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
          showFrame(ev);
          break;
        case 'run':
          setRunSpan({ cell: cell.id, from: ev.from, to: ev.to });
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
  }, [client, showFrame]);

  const runAll = useCallback(async () => {
    stopAll.current = false;
    await resetSession(backend, threads);
    for (const cell of cells) {
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

  const addBelow = (index: number) =>
    setCells((prev) => [...prev.slice(0, index + 1), { id: newId(), text: '' }, ...prev.slice(index + 1)]);

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

  const data = useMemo(() => (frame ? frameToMoleculeData(frame) : null), [frame]);
  const config = useMemo(() => {
    if (!frame || !data) return vizConfig(1, 1, isDarkTheme(theme));
    const { lo, hi } = frame.box;
    const dims = hi[2] - lo[2] > 0 && frame.x.some((_, k) => k % 3 === 2 && frame.x[k] !== 0) ? 3 : 2;
    const vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (dims === 3 ? hi[2] - lo[2] : 1);
    const spacing = Math.pow(vol / Math.max(1, data.atoms.length), 1 / dims);
    return vizConfig(spacing, Math.max(1, ...Object.keys(data.atomTypes).map(Number)), isDarkTheme(theme));
  }, [frame, data, theme]);

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
  const incomingNeeds = incoming === null ? [] : filesReadBy(incoming).filter((f) => !(f in inputs));

  const addInputFiles = async (list: FileList | null) => {
    if (!list) return;
    for (const file of Array.from(list)) {
      const text = await file.text();
      client().addFile(file.name, text);
      setInputs((prev) => ({ ...prev, [file.name]: file.size }));
    }
  };
  const removeInputFile = (name: string) => {
    client().removeFile(name);
    setInputs((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
  };
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
        {onOpenInBuilder && (
          <button className={`${btn} ${ct.button}`} disabled={cells.every((c) => c.text.trim() === '')}
            onClick={() => onOpenInBuilder(cellsToScript(cells.map((c) => c.text)))}
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
          <span className="font-medium">Script from the Script Builder ({incoming.split('\n').length} lines)</span>
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
        <div className={`relative h-56 border-b lg:col-start-2 lg:row-start-1 lg:h-auto lg:border-b-0 lg:border-l ${ct.divider}`}>
          {data ? (
            <MoleculeCanvas data={data} autoRotate={false} config={config} />
          ) : (
            <div className={`flex h-full items-center justify-center p-4 text-center text-xs ${ct.muted}`}>
              Run a cell with create_atoms to see the system here.
            </div>
          )}
          {frame && (
            <span className={`absolute left-2 top-2 rounded px-1.5 py-0.5 text-[11px] ${ct.chip}`}>
              {frame.id.length} atoms · step {frame.step}
            </span>
          )}
        </div>

        {/* cells */}
        <div className="space-y-3 p-3 lg:col-start-1 lg:row-span-2 lg:row-start-1 lg:overflow-y-auto">
          {cells.map((cell, i) => {
            const run = runs[cell.id] ?? emptyRun();
            const n = i + 1;
            return (
              <section key={cell.id} data-cell="" data-status={run.status}
                className={`rounded-lg border p-2 ${ct.card} ${run.status === 'error' ? ct.errorAccent : ''}`} aria-label={`Cell ${n}`}>
                <div className="mb-1 flex items-center gap-1">
                  <span className={`font-mono text-[11px] ${ct.muted}`}>In [{n}]</span>
                  <button className={`${btn} ${ct.button}`} aria-label={`Run cell ${n}`} title="Run this cell (Shift+Enter)"
                    disabled={busy} onClick={() => void runCell(cell)}>
                    <Play size={12} aria-hidden="true" />
                  </button>
                  <span aria-live="polite" className={`text-[11px] ${run.status === 'error' ? ct.danger : ct.muted}`}>
                    {run.status === 'idle' ? '' : run.status}
                  </span>
                  {running === cell.id && runSpan?.cell === cell.id && runSpan.to > runSpan.from && (
                    <RunProgress ct={ct} from={runSpan.from} to={runSpan.to} step={frame?.step ?? runSpan.from} />
                  )}
                  <button className={`${btn} ml-auto ${ct.button}`} aria-label={`Delete cell ${n}`} title="Delete this cell"
                    disabled={busy || cells.length <= 1} onClick={() => removeCell(i)}>
                    <Trash2 size={12} aria-hidden="true" />
                  </button>
                </div>
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
                        <tbody>
                          {t.rows.map((r, j) => (
                            <tr key={j}>{t.keywords.map((kw) => <td key={kw} className="px-2">{fmtCell(kw, r[kw])}</td>)}</tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
                <div className="mt-1 flex">
                  <button className={`${btn} ${ct.button}`} aria-label={`Add a cell below cell ${n}`} onClick={() => addBelow(i)}>
                    <Plus size={12} aria-hidden="true" />cell
                  </button>
                </div>
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
          {Object.keys(inputs).length > 0 && (
            <div>
              <p className={`text-[11px] font-semibold ${ct.muted}`}>Input files</p>
              <ul className="mt-1 flex flex-wrap gap-1" aria-label="Input files">
                {Object.entries(inputs).map(([name, size]) => (
                  <li key={name} className={`inline-flex items-center rounded text-xs ${ct.chip}`}>
                    <span className="px-2 py-1 font-mono">{name}</span>
                    <span className={`pr-1 ${ct.muted}`}>{formatBytes(size)}</span>
                    <button className={`${btn} ${ct.hoverSurface}`} aria-label={`Remove ${name}`} title={`Remove ${name}`}
                      disabled={busy} onClick={() => removeInputFile(name)}>
                      <X size={12} aria-hidden="true" />
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {imageNames.length > 0 && (() => {
            const i = Math.min(imagePick ?? imageNames.length - 1, imageNames.length - 1);
            const name = imageNames[i];
            return (
              <div>
                <p className={`text-[11px] font-semibold ${ct.muted}`}>Images written ({imageNames.length})</p>
                <img src={files[name]} alt={`dump image ${name}`} className={`mt-1 max-h-80 max-w-full rounded border ${ct.divider}`} />
                {imageNames.length > 1 && (
                  <input type="range" min={0} max={imageNames.length - 1} value={i} className="mt-1 w-full"
                    aria-label={`Image ${i + 1} of ${imageNames.length}`}
                    onChange={(e) => { const k = Number(e.target.value); setImagePick(k === imageNames.length - 1 ? null : k); }} />
                )}
                <div className="mt-1 flex items-center gap-2 text-[11px]">
                  <span className="font-mono">{name}</span>
                  <button className={`${btn} ${ct.button}`} aria-label={`Download ${name}`} onClick={() => downloadFile(name, files[name])}>
                    <Download size={12} aria-hidden="true" />Download
                  </button>
                </div>
              </div>
            );
          })()}
          {Object.keys(files).some((n) => !isImageDataUrl(files[n])) && (
            <div>
              <p className={`text-[11px] font-semibold ${ct.muted}`}>Files written</p>
              <ul className="mt-1 flex flex-wrap gap-1">
                {Object.entries(files).filter(([, text]) => !isImageDataUrl(text)).map(([name, text]) => (
                  <li key={name}>
                    <button className={`${btn} ${ct.button}`} aria-label={`Download ${name}`}
                      onClick={() => downloadFile(name, text)}>
                      <Download size={12} aria-hidden="true" />{name}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {chart.temp.length <= 1 && Object.keys(files).length === 0 && Object.keys(inputs).length === 0 && (
            <p className={`text-xs ${ct.muted}`}>
              Thermo charts, files written by dump / write_data, and the files you add with Add files appear here.
            </p>
          )}
        </div>
      </div>
    </div>
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
