import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  ALL_COMMANDS,
  COMMAND_BY_ID,
  SECTION_LABELS,
  SECTION_ORDER,
  ScriptModel,
  ScriptStep,
  ScriptTab,
  ScriptWorkspace,
  defaultParams,
  emptyScriptModel,
  newTabId,
  ParamDef,
} from '../../lammps/catalog';
import { generateScript, deriveFlowchart, FlowGraph } from '../../lammps/generator';
import { parseScript, ImportResult } from '../../lammps/scriptParser';
import { downloadTextFile } from '../../lammps/exporter';
import { SCRIPT_TEMPLATES, buildTemplate } from '../../lammps/templates';
import { downloadFlowchart } from '../../lammps/flowchartSvg';
import { useUndoableState } from '../../hooks/useUndoableState';
import { browserStore, loadJson, saveJson } from '../../services/persistence';
import { getThemeTokens, ThemeTokens, Theme } from '../../theme';
import {
  Plus, Trash2, Copy, Download, Eye, EyeOff, Upload,
  FileCode2, Workflow, ChevronDown, ChevronRight, ChevronUp, ChevronLeft, Search,
  Atom as AtomIcon, PencilLine, X, GripVertical, Link2, Undo2, Redo2,
  LayoutTemplate, ZoomIn, ZoomOut, Maximize2, FileInput, ImageDown, FileImage,
} from 'lucide-react';

interface ScriptBuilderProps {
  theme: Theme;
  onOpenViewer?: () => void;
}

let uidCounter = 1;
const newUid = () => `step-${uidCounter++}`;

/** Merge a stored payload onto fresh defaults; drop unknown commands. */
const reviveModel = (raw: unknown): ScriptModel | null => {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Partial<ScriptModel>;
  if (!Array.isArray(r.steps)) return null;
  const steps: ScriptStep[] = [];
  for (const s of r.steps) {
    if (!s || typeof s !== 'object') continue;
    const rec = s as Partial<ScriptStep>;
    const def = rec.defId ? COMMAND_BY_ID[rec.defId] : undefined;
    if (!def) continue;
    steps.push({
      uid: typeof rec.uid === 'string' ? rec.uid : newUid(),
      defId: def.id,
      params: { ...defaultParams(def), ...(rec.params ?? {}) },
      enabled: rec.enabled !== false,
      note: typeof rec.note === 'string' ? rec.note : undefined,
    });
  }
  return {
    title: typeof r.title === 'string' ? r.title : 'My LAMMPS Simulation',
    steps,
    manualText: typeof r.manualText === 'string' ? r.manualText : undefined,
    manualBase: typeof r.manualBase === 'string' ? r.manualBase : undefined,
  };
};

const WORKSPACE_KEY = 'm3d.scriptTabs.v1';
const LEGACY_MODEL_KEY = 'm3d.scriptModel.v1';

const DEFAULT_MODEL: ScriptModel = emptyScriptModel('My LAMMPS Simulation');

/** Revive the multi-tab workspace; migrate the legacy single-model key. */
const reviveWorkspace = (raw: unknown): ScriptWorkspace | null => {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Partial<ScriptWorkspace>;
  if (!Array.isArray(r.tabs) || r.tabs.length === 0) return null;
  const tabs: ScriptTab[] = [];
  for (const t of r.tabs) {
    if (!t || typeof t !== 'object') continue;
    const rec = t as Partial<ScriptTab>;
    const model = reviveModel(rec.model);
    if (!model) continue;
    tabs.push({ id: typeof rec.id === 'string' ? rec.id : newTabId(), model });
  }
  if (tabs.length === 0) return null;
  const activeId =
    typeof r.activeId === 'string' && tabs.some(t => t.id === r.activeId)
      ? r.activeId
      : tabs[0].id;
  return { tabs, activeId };
};

const loadWorkspace = (): ScriptWorkspace => {
  const stored = reviveWorkspace(loadJson(browserStore(), WORKSPACE_KEY));
  if (stored) return stored;
  // Migrate the pre-tabs single model so nobody loses work on upgrade.
  const legacy = reviveModel(loadJson(browserStore(), LEGACY_MODEL_KEY));
  const first: ScriptTab = {
    id: newTabId(),
    model: legacy ?? { title: 'My LAMMPS Simulation', steps: [] },
  };
  return { tabs: [first], activeId: first.id };
};
const ZOOM_MIN = 0.35;
const ZOOM_MAX = 2.5;

interface Transform {
  x: number;
  y: number;
  k: number;
}

const ScriptBuilder: React.FC<ScriptBuilderProps> = ({ theme, onOpenViewer }) => {
  const ct = getThemeTokens(theme);
  // Undoable WORKSPACE (multi-tab); persists to localStorage for durability.
  const [workspace, setWorkspace, replaceWorkspace, undo, redo, canUndo, canRedo] = (() => {
    const api = useUndoableState<ScriptWorkspace>(loadWorkspace);
    return [api.value, api.set, api.replace, api.undo, api.redo, api.canUndo, api.canRedo] as const;
  })();

  const activeTab: ScriptTab =
    workspace.tabs.find(t => t.id === workspace.activeId) ?? workspace.tabs[0];
  const model = activeTab.model;

  /** Update the ACTIVE tab's model (pushes undo history). */
  const setModel = useCallback(
    (next: ScriptModel | ((prev: ScriptModel) => ScriptModel)) => {
      setWorkspace(ws => ({
        ...ws,
        tabs: ws.tabs.map(t =>
          t.id === ws.activeId
            ? { ...t, model: typeof next === 'function' ? next(t.model) : next }
            : t
        ),
      }));
    },
    [setWorkspace],
  );

  /** Switch tabs WITHOUT pushing undo history. */
  const switchTab = useCallback(
    (id: string) => replaceWorkspace(ws => ({ ...ws, activeId: id })),
    [replaceWorkspace],
  );

  const addTab = useCallback(() => {
    const tab: ScriptTab = { id: newTabId(), model: emptyScriptModel(`Untitled ${workspace.tabs.length + 1}`) };
    setWorkspace(ws => ({ tabs: [...ws.tabs, tab], activeId: tab.id }));
  }, [setWorkspace, workspace.tabs.length]);

  const closeTab = useCallback(
    (id: string) => {
      setWorkspace(ws => {
        const idx = ws.tabs.findIndex(t => t.id === id);
        if (idx < 0) return ws;
        const tabs = ws.tabs.filter(t => t.id !== id);
        if (tabs.length === 0) {
          const fresh: ScriptTab = { id: newTabId(), model: emptyScriptModel('Untitled') };
          return { tabs: [fresh], activeId: fresh.id };
        }
        const activeId = id === ws.activeId ? tabs[Math.min(idx, tabs.length - 1)].id : ws.activeId;
        return { tabs, activeId };
      });
    },
    [setWorkspace],
  );

  /** Clear the ACTIVE flowchart (undoable). */
  const clearActive = useCallback(() => {
    setModel(prev => ({ ...prev, steps: [], manualText: undefined, manualBase: undefined }));
    setSelectedUid(null);
    setImportStats(null);
  }, [setModel]);
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 768 : true);
  const [view, setView] = useState<'flow' | 'script'>('flow');
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);

  // Script import (in.* → flowchart)
  const importInputRef = useRef<HTMLInputElement>(null);
  const [importStats, setImportStats] = useState<ImportResult['stats'] | null>(null);
  const [exporting, setExporting] = useState<'svg' | 'png' | null>(null);

  const exportFlowchart = useCallback(async (format: 'svg' | 'png') => {
    setExporting(format);
    try {
      await downloadFlowchart(model, format, { theme });
    } catch {
      /* download failed — non-fatal */
    } finally {
      setExporting(null);
    }
  }, [model, theme]);

  // Pointer drag-to-reorder (grab a card, drop between two others)
  const [drag, setDrag] = useState<{ uid: string; label: string; x: number; y: number } | null>(null);
  const [overIndex, setOverIndex] = useState<number | null>(null);
  const nodeRefs = useRef<Map<string, HTMLElement>>(new Map());
  const dragState = useRef<{ uid: string; startX: number; startY: number; active: boolean } | null>(null);

  // Flowchart canvas pan/zoom
  const canvasRef = useRef<HTMLDivElement>(null);
  const [viewTf, setViewTf] = useState<Transform>({ x: 0, y: 0, k: 1 });
  const panRef = useRef<{ startX: number; startY: number; ox: number; oy: number } | null>(null);
  const [panning, setPanning] = useState(false);

  // Connection-editing menu + insert-at-edge picker
  const [edgeMenuIndex, setEdgeMenuIndex] = useState<number | null>(null);
  const [insertAt, setInsertAt] = useState<number | null>(null);
  const [insertSearch, setInsertSearch] = useState('');

  // Mobile: palette/editor become overlay drawers
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < 768 : false);
  const [editorOpenMobile, setEditorOpenMobile] = useState(false);
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);
  // Close the mobile editor drawer whenever no step is selected
  useEffect(() => {
    if (!selectedUid) setEditorOpenMobile(false);
  }, [selectedUid]);

  const isManual = model.manualText !== undefined;
  const manualStale =
    isManual && model.manualBase !== undefined && JSON.stringify(model.steps) !== model.manualBase;

  const generated = useMemo(() => generateScript(model), [model]);
  const flow = useMemo(() => deriveFlowchart(model), [model]);
  /** The text currently shown/copied/downloaded. */
  const activeText = isManual ? (model.manualText ?? '') : generated.text;

  // Persist the workspace whenever it changes (history itself is in-memory).
  useEffect(() => {
    saveJson(browserStore(), WORKSPACE_KEY, workspace);
  }, [workspace]);

  // Undo/redo + Delete-selected keyboard layer (builder-local; the manual
  // textarea keeps native undo because typing targets are skipped).
  const removeStepRef = useRef<(uid: string) => void>(() => {});
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && !e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
      if ((mod && e.shiftKey && e.key.toLowerCase() === 'z') || (mod && e.key.toLowerCase() === 'y')) { e.preventDefault(); redo(); return; }
      if (!mod && e.key === 'Delete' && selectedUid) { e.preventDefault(); removeStepRef.current(selectedUid); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo, selectedUid]);

  // ---- model mutations -------------------------------------------------
  const insertStepAt = useCallback((defId: string, index: number) => {
    const def = COMMAND_BY_ID[defId];
    if (!def) return;
    const step: ScriptStep = {
      uid: newUid(), defId, params: defaultParams(def), enabled: true,
    };
    setModel(prev => {
      const steps = [...prev.steps];
      const at = Math.max(0, Math.min(index, steps.length));
      steps.splice(at, 0, step);
      return { ...prev, steps };
    });
    setSelectedUid(step.uid);
  }, [setModel]);

  const addStep = useCallback((defId: string) => {
    const def = COMMAND_BY_ID[defId];
    if (!def) return;
    const step: ScriptStep = {
      uid: newUid(), defId, params: defaultParams(def), enabled: true,
    };
    setModel(prev => {
      // Insert after the last step in the same section, else at end
      const sectionSteps = prev.steps.filter(s => COMMAND_BY_ID[s.defId]?.section === def.section);
      const insertAfter = sectionSteps.length > 0 ? sectionSteps[sectionSteps.length - 1] : null;
      if (!insertAfter) return { ...prev, steps: [...prev.steps, step] };
      const idx = prev.steps.indexOf(insertAfter);
      const steps = [...prev.steps];
      steps.splice(idx + 1, 0, step);
      return { ...prev, steps };
    });
    setSelectedUid(step.uid);
  }, [setModel]);

  const updateStep = useCallback((uid: string, patch: Partial<ScriptStep>) => {
    setModel(prev => ({
      ...prev,
      steps: prev.steps.map(s => (s.uid === uid ? { ...s, ...patch } : s)),
    }));
  }, [setModel]);

  const updateParam = useCallback((uid: string, key: string, value: string) => {
    setModel(prev => ({
      ...prev,
      steps: prev.steps.map(s =>
        s.uid === uid ? { ...s, params: { ...s.params, [key]: value } } : s
      ),
    }));
  }, [setModel]);

  const removeStep = useCallback((uid: string) => {
    setModel(prev => ({ ...prev, steps: prev.steps.filter(s => s.uid !== uid) }));
    setSelectedUid(prev => (prev === uid ? null : prev));
  }, [setModel]);
  removeStepRef.current = removeStep;

  /** Chevron move: swap with the neighbour. */
  const moveStep = useCallback((uid: string, dir: -1 | 1) => {
    setModel(prev => {
      const steps = [...prev.steps];
      const i = steps.findIndex(s => s.uid === uid);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= steps.length) return prev;
      [steps[i], steps[j]] = [steps[j], steps[i]];
      return { ...prev, steps };
    });
  }, [setModel]);

  /** Move a dragged step so that it lands at insertion position `toIndex`. */
  const moveStepToIndex = useCallback((uid: string, toIndex: number) => {
    setModel(prev => {
      const steps = [...prev.steps];
      const from = steps.findIndex(s => s.uid === uid);
      if (from < 0) return prev;
      const clamped = Math.max(0, Math.min(toIndex, steps.length));
      if (clamped === from || clamped === from + 1) return prev;
      const [dragged] = steps.splice(from, 1);
      const at = clamped > from ? clamped - 1 : clamped;
      steps.splice(at, 0, dragged);
      return { ...prev, steps };
    });
  }, [setModel]);

  /** Duplicate a step (same command + params) right below it. */
  const duplicateStep = useCallback((uid: string) => {
    setModel(prev => {
      const idx = prev.steps.findIndex(s => s.uid === uid);
      if (idx < 0) return prev;
      const src = prev.steps[idx];
      const clone: ScriptStep = { ...src, uid: newUid(), params: { ...src.params } };
      const steps = [...prev.steps];
      steps.splice(idx + 1, 0, clone);
      return { ...prev, steps };
    });
  }, [setModel]);

  // ---- script import ----------------------------------------------------
  const handleImportFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = e => {
      const text = (e.target?.result as string) ?? '';
      const result = parseScript(text, file.name.replace(/\.[^.]+$/, ''));
      setModel(result.model);
      setImportStats(result.stats);
      setSelectedUid(null);
      setView('flow');
    };
    reader.readAsText(file);
  }, [setModel]);

  // ---- manual script editing -------------------------------------------
  const enterManualMode = useCallback(() => {
    setModel(prev => ({
      ...prev,
      manualBase: JSON.stringify(prev.steps),
      manualText: prev.manualText ?? generateScript({ ...prev }).text,
    }));
    setView('script');
  }, [setModel]);

  const exitManualMode = useCallback(() => {
    setModel(prev => {
      const { manualText: _drop, manualBase: _base, ...rest } = prev;
      void _drop;
      void _base;
      return rest;
    });
  }, [setModel]);

  const setManualText = useCallback((text: string) => {
    setModel(prev => ({ ...prev, manualText: text }));
  }, [setModel]);

  const copyScript = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(activeText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch { /* clipboard blocked */ }
  }, [activeText]);

  const filteredCommands = useMemo(() => {
    if (!search.trim()) return ALL_COMMANDS;
    const q = search.toLowerCase();
    return ALL_COMMANDS.filter(d =>
      d.label.toLowerCase().includes(q) ||
      d.command.toLowerCase().includes(q) ||
      d.category.toLowerCase().includes(q)
    );
  }, [search]);

  const insertCandidates = useMemo(() => {
    if (!insertSearch.trim()) return ALL_COMMANDS;
    const q = insertSearch.toLowerCase();
    return ALL_COMMANDS.filter(d =>
      d.label.toLowerCase().includes(q) ||
      d.command.toLowerCase().includes(q)
    );
  }, [insertSearch]);

  const selectedStep = model.steps.find(s => s.uid === selectedUid) ?? null;
  const selectedDef = selectedStep ? COMMAND_BY_ID[selectedStep.defId] : null;

  // Group palette commands by section
  const paletteBySection = useMemo(() => {
    const map = new Map<string, typeof ALL_COMMANDS>();
    for (const cmd of filteredCommands) {
      const list = map.get(cmd.section) ?? [];
      list.push(cmd);
      map.set(cmd.section, list);
    }
    return map;
  }, [filteredCommands]);

  // ---- pointer drag-to-reorder ------------------------------------------
  const computeInsertIndex = useCallback((clientY: number): number => {
    let idx = 0;
    for (const s of model.steps) {
      const el = nodeRefs.current.get(s.uid);
      if (!el) continue;
      const r = el.getBoundingClientRect();
      if (clientY > r.top + r.height / 2) idx++;
    }
    return idx;
  }, [model.steps]);

  const cardPointerDown = useCallback((uid: string) => (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.stopPropagation(); // keep the canvas pan handler out of card drags
    dragState.current = { uid, startX: e.clientX, startY: e.clientY, active: false };
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const cardPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const st = dragState.current;
    if (!st) return;
    if (!st.active) {
      if (Math.hypot(e.clientX - st.startX, e.clientY - st.startY) < 6) return;
      st.active = true;
      const step = model.steps.find(s => s.uid === st.uid);
      setDrag({
        uid: st.uid,
        label: COMMAND_BY_ID[step?.defId ?? '']?.command ?? step?.defId ?? '',
        x: e.clientX, y: e.clientY,
      });
    }
    setDrag(d => (d ? { ...d, x: e.clientX, y: e.clientY } : d));
    setOverIndex(computeInsertIndex(e.clientY));
  }, [model.steps, computeInsertIndex]);

  const cardPointerUp = useCallback(() => {
    const st = dragState.current;
    dragState.current = null;
    if (st?.active && overIndex !== null) {
      moveStepToIndex(st.uid, overIndex);
    }
    setDrag(null);
    setOverIndex(null);
  }, [overIndex, moveStepToIndex]);

  // ---- flowchart canvas pan/zoom ----------------------------------------
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      setViewTf(t => {
        const factor = Math.exp(-e.deltaY * 0.0015);
        const k = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, t.k * factor));
        const f = k / t.k;
        return { k, x: cx - (cx - t.x) * f, y: cy - (cy - t.y) * f };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [view]);

  const bgPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0 && e.button !== 1) return;
    panRef.current = { startX: e.clientX, startY: e.clientY, ox: viewTf.x, oy: viewTf.y };
    setPanning(true);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, [viewTf]);

  const bgPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    const p = panRef.current;
    if (!p) return;
    setViewTf(t => ({ ...t, x: p.ox + (e.clientX - p.startX), y: p.oy + (e.clientY - p.startY) }));
  }, []);

  const bgPointerUp = useCallback(() => {
    panRef.current = null;
    setPanning(false);
  }, []);

  const zoomBy = useCallback((factor: number) => {
    const el = canvasRef.current;
    const cx = (el?.clientWidth ?? 800) / 2;
    const cy = (el?.clientHeight ?? 600) / 2;
    setViewTf(t => {
      const k = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, t.k * factor));
      const f = k / t.k;
      return { k, x: cx - (cx - t.x) * f, y: cy - (cy - t.y) * f };
    });
  }, []);

  const resetView = useCallback(() => setViewTf({ x: 0, y: 0, k: 1 }), []);

  return (
    <div className="flex h-full min-h-0">
      {/* Left: palette (overlay drawer on mobile) */}
      <div
        className={`flex-col border-r transition-all ${ct.panel} ${
          isMobile
            ? `fixed inset-y-0 left-0 z-40 flex w-72 ${paletteOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'}`
            : `${paletteOpen ? 'flex w-72' : 'hidden w-0 overflow-hidden'}`
        }`}
      >
        <div className={`flex items-center justify-between px-3 py-2 border-b ${ct.divider}`}>
          <span className={`text-[13px] font-bold tracking-tight ${ct.headerText}`}>Commands</span>
          <button onClick={() => setPaletteOpen(false)} className={`p-1 rounded ${ct.muted} ${ct.hoverSurface}`}>
            <ChevronLeft size={14} />
          </button>
          {isMobile && (
            <button onClick={() => setPaletteOpen(false)} className={`rounded p-1 ${ct.muted} ${ct.hoverSurface}`} title="Close">
              <X size={14} />
            </button>
          )}
        </div>
        <div className={`border-b px-3 py-2.5 ${ct.divider}`}>
          <div className={`flex items-center gap-2 rounded-lg border px-2.5 py-2 ${ct.input}`}>
            <Search size={14} className="shrink-0 opacity-50" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder={`Search ${ALL_COMMANDS.length} commands…`}
              className={`w-full bg-transparent text-[13px] ${ct.text} placeholder:text-[#6f6353] focus:outline-none`}
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-3">
          {SECTION_ORDER.map(section => {
            const cmds = paletteBySection.get(section);
            if (!cmds || cmds.length === 0) return null;
            return (
              <div key={section} className="space-y-0.5">
                <div className={`mb-1 flex items-center gap-2 px-1 pt-2`}>
                  <div className="h-3.5 w-0.5 rounded-full bg-[#7fa66b] opacity-60" />
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${ct.headerText}`}>
                    {SECTION_LABELS[section]}
                  </span>
                </div>
                {cmds.map(cmd => (
                  <button
                    key={cmd.id}
                    onClick={() => addStep(cmd.id)}
                    title={cmd.doc ?? cmd.label}
                    className={`group flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors ${ct.muted} ${ct.hoverSurface}`}
                  >
                    <Plus size={12} className="shrink-0 opacity-40 transition-opacity group-hover:opacity-80" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-[12px] leading-tight">{cmd.label}</div>
                      <code className={`text-[10px] font-mono opacity-50 ${ct.accentCode}`}>{cmd.command}</code>
                    </div>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Center: tabs + pipeline + step editor */}
      <div className={`flex min-w-0 flex-1 flex-col ${ct.bg}`}>
        {/* Tab strip */}
        <div className={`flex h-9 shrink-0 items-center gap-1 border-b px-2 ${ct.divider}`}>
          <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
            {workspace.tabs.map(tab => {
              const active = tab.id === workspace.activeId;
              return (
                <div
                  key={tab.id}
                  onClick={() => switchTab(tab.id)}
                  className={`group flex shrink-0 cursor-pointer items-center gap-1 rounded-t-lg border-b-2 px-1.5 py-1.5 text-[11px] font-medium transition-colors sm:gap-1.5 sm:px-2.5 ${
                    active ? ct.active : `${ct.muted} border-transparent ${ct.hoverSurface}`
                  }`}
                  title={tab.model.title || 'Untitled'}
                >
                  <FileCode2 size={11} className="shrink-0 opacity-60" />
                  <span className="max-w-[72px] truncate sm:max-w-[110px]">{tab.model.title || 'Untitled'}</span>
                  <span className={`hidden text-[9px] tabular-nums opacity-50 sm:inline`}>{tab.model.steps.length}</span>
                  <button
                    onClick={e => { e.stopPropagation(); closeTab(tab.id); }}
                    className={`rounded p-0.5 opacity-0 transition-opacity hover:text-[#cf8b76] group-hover:opacity-100 ${
                      active ? 'opacity-70' : ''
                    }`}
                    title="Close this flowchart (Ctrl+Z restores)"
                    aria-label={`Close ${tab.model.title || 'tab'}`}
                  >
                    <X size={11} />
                  </button>
                </div>
              );
            })}
            <button
              onClick={addTab}
              className={`shrink-0 rounded p-1.5 ${ct.muted} ${ct.hoverSurface}`}
              title="New flowchart tab"
              aria-label="New flowchart tab"
            >
              <Plus size={13} />
            </button>
          </div>
          <button
            onClick={clearActive}
            disabled={model.steps.length === 0}
            className={`flex shrink-0 items-center gap-1 rounded px-2 py-1 text-[10px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface} disabled:opacity-30 disabled:pointer-events-none`}
            title="Clear this flowchart (Ctrl+Z restores it)"
          >
            <Trash2 size={12} />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>

        {/* Toolbar (scrollable on mobile, two groups on desktop) */}
        <div className={`flex min-h-10 shrink-0 items-center justify-between gap-x-2 overflow-x-auto border-b px-3 py-1 ${ct.divider}`}>
          <div className="flex min-w-0 shrink-0 items-center gap-1.5">
            {!paletteOpen && (
              <button onClick={() => setPaletteOpen(true)} className={`shrink-0 rounded p-1 ${ct.muted} ${ct.hoverSurface}`} title="Open palette">
                <ChevronRight size={16} />
              </button>
            )}
            <input
              value={model.title}
              onChange={e => setModel(prev => ({ ...prev, title: e.target.value }))}
              className={`w-24 min-w-0 flex-shrink rounded bg-transparent px-1 py-0.5 text-sm font-medium focus:outline-none focus:ring-1 focus:ring-[#7fa66b] sm:w-40 sm:px-1.5 ${ct.headerText}`}
              aria-label="Script title"
            />
            <span className={`hidden whitespace-nowrap text-[10px] md:block ${ct.muted}`}>
              {isManual ? 'manual' : `${model.steps.length} steps · ${generated.emitted.length} lines`}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-0.5 sm:gap-1.5">
            <button
              onClick={undo}
              disabled={!canUndo}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface} disabled:opacity-30 disabled:pointer-events-none`}
              title="Undo (Ctrl+Z)"
              aria-label="Undo"
            >
              <Undo2 size={13} />
            </button>
            <button
              onClick={redo}
              disabled={!canRedo}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface} disabled:opacity-30 disabled:pointer-events-none`}
              title="Redo (Ctrl+Shift+Z)"
              aria-label="Redo"
            >
              <Redo2 size={13} />
            </button>
            <div className="relative">
              <button
                onClick={() => setTemplatesOpen(v => !v)}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
                title="Start from a ready-made pipeline"
              >
                <LayoutTemplate size={13} />
                <span className="hidden md:inline">Templates</span>
              </button>
              {templatesOpen && (
                <div
                  className={`absolute right-0 top-full z-40 mt-1 w-72 rounded-lg border p-1.5 shadow-2xl ${ct.card}`}
                  onClick={e => e.stopPropagation()}
                >
                  <p className={`px-2 pb-1 pt-0.5 text-[9px] uppercase tracking-wide ${ct.muted}`}>
                    Starter pipelines (replaces the current steps — undo works)
                  </p>
                  {SCRIPT_TEMPLATES.map(tpl => (
                    <button
                      key={tpl.id}
                      onClick={() => {
                        setModel(buildTemplate(tpl));
                        setSelectedUid(null);
                        setTemplatesOpen(false);
                      }}
                      className={`flex w-full flex-col rounded px-2 py-1.5 text-left transition-colors ${ct.hoverSurface}`}
                      title={tpl.description}
                    >
                      <span className="text-[11px] font-semibold">{tpl.label}</span>
                      <span className={`text-[10px] ${ct.muted}`}>{tpl.description}</span>
                    </button>
                  ))}
                  <button
                    onClick={() => setTemplatesOpen(false)}
                    className={`mt-0.5 w-full rounded px-2 py-1 text-left text-[10px] ${ct.muted} ${ct.hoverSurface}`}
                  >
                    Close
                  </button>
                </div>
              )}
            </div>
            <button
              onClick={() => importInputRef.current?.click()}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
              title="Import a LAMMPS input script — builds the flowchart"
            >
              <FileInput size={13} />
              <span className="hidden lg:inline">Import</span>
            </button>
            <input
              ref={importInputRef}
              type="file"
              accept=".in,.lammps,.lmp,.txt,.script,.mod"
              className="hidden"
              onChange={e => {
                const f = e.target.files?.[0];
                if (f) handleImportFile(f);
                e.target.value = '';
              }}
            />
            {view === 'flow' && !isManual && (
              <>
                <button
                  onClick={() => exportFlowchart('svg')}
                  disabled={exporting !== null || model.steps.length === 0}
                  className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface} disabled:opacity-30 disabled:pointer-events-none`}
                  title="Export the flowchart as a presentable SVG"
                >
                  <FileImage size={13} />
                  <span className="hidden md:inline">SVG</span>
                </button>
                <button
                  onClick={() => exportFlowchart('png')}
                  disabled={exporting !== null || model.steps.length === 0}
                  className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface} disabled:opacity-30 disabled:pointer-events-none`}
                  title="Export the flowchart as a 2x PNG image"
                >
                  <ImageDown size={13} />
                  <span className="hidden md:inline">PNG</span>
                </button>
              </>
            )}
            <button
              onClick={() => setView(v => v === 'flow' ? 'script' : 'flow')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                view === 'script' ? ct.active : `${ct.muted} ${ct.hoverSurface}`
              }`}
              title={view === 'flow' ? 'View the generated script' : 'Back to the flowchart'}
            >
              {view === 'flow' ? <FileCode2 size={13} /> : <Workflow size={13} />}
              {view === 'flow' ? 'Script' : 'Flowchart'}
              {isManual && <span className="h-1.5 w-1.5 rounded-full bg-[#d9a05b]" title="Manual override active" />}
            </button>
            {isManual ? (
              <button
                onClick={exitManualMode}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium ${ct.warnAction}`}
                title="Discard manual edits and regenerate from your steps"
              >
                <X size={13} />
                <span className="hidden 2xl:inline">Exit manual</span>
              </button>
            ) : (
              <button
                onClick={() => { enterManualMode(); setView('script'); }}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
                title="Override the generated script with hand-edited text"
              >
                <PencilLine size={13} />
                <span className="hidden 2xl:inline">Override</span>
              </button>
            )}
            <button
              onClick={copyScript}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
              title="Copy script to clipboard"
            >
              {copied ? <span className={ct.accentText}>✓ Copied</span> : <><Copy size={13} /><span className="hidden lg:inline">Copy</span></>}
            </button>
            <button
              onClick={() => downloadTextFile('in.lammps', activeText)}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
              title="Download in.lammps"
            >
              <Download size={13} />
              <span className="hidden 2xl:inline">Download</span>
            </button>
            {onOpenViewer && (
              <button
                onClick={onOpenViewer}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
                title="Open 3D structure viewer"
              >
                <AtomIcon size={13} />
                <span className="hidden 2xl:inline">Viewer</span>
              </button>
            )}
          </div>
        </div>

        {/* Import / manual notices */}
        {importStats && (
          <div className={`flex shrink-0 items-center justify-between border-b px-3 py-1.5 text-[11px] ${ct.card}`}>
            <span className={ct.text}>
              <FileInput size={11} className="mr-1 inline" />
              Imported {importStats.total} statements — {importStats.recognized} recognized
              {importStats.raw > 0 && `, ${importStats.raw} kept as verbatim raw lines`}
              .
            </span>
            <button onClick={() => setImportStats(null)} className={`rounded p-0.5 ${ct.muted}`} title="Dismiss">
              <X size={12} />
            </button>
          </div>
        )}
        {isManual && (
          <div className={`flex shrink-0 flex-wrap items-center justify-between gap-2 border-b px-3 py-1.5 text-[11px] ${ct.warn}`}>
            <span>
              ✎ Manual override — this text is emitted verbatim; builder changes do
              <b> not </b>update it.
            </span>
            <button
              onClick={exitManualMode}
              className={`rounded px-2 py-0.5 font-semibold ${ct.button}`}
              title="Discard the manual text and follow the builder steps again"
            >
              Discard & follow builder
            </button>
          </div>
        )}
        {manualStale && (
          <div className={`flex shrink-0 flex-wrap items-center justify-between gap-2 border-b px-3 py-1.5 text-[11px] ${ct.warn}`}>
            <span>⚠ Builder steps changed since your manual edit — the script text is stale.</span>
            <button
              onClick={exitManualMode}
              className={`rounded px-2 py-0.5 font-semibold ${ct.button}`}
              title="Discard the manual text and regenerate from the current steps"
            >
              Regenerate from steps
            </button>
          </div>
        )}
        {!isManual && generated.warnings.length > 0 && (
          <div className={`shrink-0 border-t px-3 py-2 text-[11px] ${ct.warn}`}>
            {generated.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}

        {/* Flowchart canvas (pan + zoom) or Script view */}
        {view === 'flow' ? (
          <div
            ref={canvasRef}
            onPointerDown={bgPointerDown}
            onPointerMove={bgPointerMove}
            onPointerUp={bgPointerUp}
            onPointerCancel={bgPointerUp}
            className={`relative min-h-0 flex-1 overflow-hidden ${panning ? 'cursor-grabbing' : ''}`}
            style={{ touchAction: 'none' }}
          >
            <div
              className="absolute left-0 top-0 origin-top-left"
              style={{ transform: `translate(${viewTf.x}px, ${viewTf.y}px) scale(${viewTf.k})` }}
            >
              <div style={{ width: 640 }}>
                <FlowchartView
                  ct={ct}
                  flow={flow}
                  steps={model.steps}
                  selectedUid={selectedUid}
                  dragUid={drag?.uid ?? null}
                  overIndex={overIndex}
                  edgeMenuIndex={edgeMenuIndex}
                  onSelect={(uid) => { setSelectedUid(uid); if (isMobile) setEditorOpenMobile(true); }}
                  onToggle={(uid) => updateStep(uid, { enabled: !model.steps.find(s => s.uid === uid)?.enabled })}
                  onRemove={removeStep}
                  onDuplicate={duplicateStep}
                  onMove={moveStep}
                  onCardPointerDown={cardPointerDown}
                  onCardPointerMove={cardPointerMove}
                  onCardPointerUp={cardPointerUp}
                  registerNodeRef={(uid, el) => {
                    if (el) nodeRefs.current.set(uid, el);
                    else nodeRefs.current.delete(uid);
                  }}
                  onEdgeClick={(i) => setEdgeMenuIndex(prev => (prev === i ? null : i))}
                  onEdgeInsertHere={(i) => { setEdgeMenuIndex(null); setInsertAt(i); }}
                  onEdgeDisableNext={(i) => {
                    const s = model.steps[i];
                    if (s) updateStep(s.uid, { enabled: !s.enabled });
                    setEdgeMenuIndex(null);
                  }}
                  onEdgeRemoveNext={(i) => {
                    const s = model.steps[i];
                    if (s) removeStep(s.uid);
                    setEdgeMenuIndex(null);
                  }}
                  onCloseEdgeMenu={() => setEdgeMenuIndex(null)}
                />
              </div>
            </div>

            {/* Zoom controls */}
            <div
              className={`absolute bottom-3 right-3 z-20 flex items-center gap-0.5 rounded-full border px-1 py-0.5 shadow-lg sm:gap-1 sm:px-1.5 sm:py-1 ${ct.card}`}
              onPointerDown={e => e.stopPropagation()}
              onWheel={e => e.stopPropagation()}
            >
              <button
                onClick={() => zoomBy(1 / 1.2)}
                className={`rounded-full p-1.5 ${ct.muted} ${ct.hoverSurface}`}
                title="Zoom out (mouse wheel)"
                aria-label="Zoom out"
              >
                <ZoomOut size={14} />
              </button>
              <button
                onClick={resetView}
                className={`rounded-full px-2 py-1 text-[10px] font-mono tabular-nums ${ct.muted} ${ct.hoverSurface}`}
                title="Reset view"
              >
                {Math.round(viewTf.k * 100)}%
              </button>
              <button
                onClick={() => zoomBy(1.2)}
                className={`rounded-full p-1.5 ${ct.muted} ${ct.hoverSurface}`}
                title="Zoom in (mouse wheel)"
                aria-label="Zoom in"
              >
                <ZoomIn size={14} />
              </button>
              <div className={`mx-0.5 h-4 w-px ${ct.divider.split(' ')[0]}`} />
              <button
                onClick={resetView}
                className={`rounded-full p-1.5 ${ct.muted} ${ct.hoverSurface}`}
                title="Fit / reset pan & zoom"
                aria-label="Reset view"
              >
                <Maximize2 size={14} />
              </button>
            </div>
            <p className={`pointer-events-none absolute bottom-3 left-3 hidden text-[10px] md:block ${ct.muted}`}>
              wheel = zoom · drag background = pan · grab cards to reorder
            </p>
          </div>
        ) : isManual ? (
          <textarea
            value={model.manualText ?? ''}
            onChange={e => setManualText(e.target.value)}
            spellCheck={false}
            className={`h-full min-h-full w-full resize-none p-4 text-[11px] leading-relaxed font-mono focus:outline-none ${ct.bg} ${ct.text}`}
            aria-label="Manual LAMMPS script editor"
          />
        ) : (
          <div className="min-h-0 flex-1 overflow-auto">
            <p className={`sticky top-0 z-10 px-4 py-1.5 text-[10px] ${ct.muted} ${ct.bg}`}>
              Auto-generated from your steps — edits in the flowchart update this live.{' '}
              <button onClick={enterManualMode} className={`font-semibold ${ct.accentText} hover:underline`}>
                Override by hand
              </button>
            </p>
            <pre className={`px-4 pb-4 text-[11px] leading-relaxed font-mono whitespace-pre-wrap ${ct.text}`}>
              {generated.text}
            </pre>
          </div>
        )}
      </div>

      {/* Right: step editor (overlay drawer on mobile) */}
      <div
        className={`overflow-y-auto border-l ${ct.panel} ${
          isMobile
            ? `fixed inset-y-0 right-0 z-40 w-80 max-w-[85vw] shadow-2xl transition-transform ${
                selectedStep && editorOpenMobile ? 'translate-x-0' : 'translate-x-full'
              }`
            : 'w-80 shrink-0'
        }`}
      >
        {isMobile && selectedStep && editorOpenMobile && (
          <div className={`flex items-center justify-between border-b px-3 py-2 ${ct.divider}`}>
            <span className={`text-xs font-semibold ${ct.headerText}`}>Step editor</span>
            <button
              onClick={() => { setEditorOpenMobile(false); setSelectedUid(null); }}
              className={`rounded p-1 ${ct.muted} ${ct.hoverSurface}`}
              title="Close editor"
            >
              <X size={14} />
            </button>
          </div>
        )}
        {selectedStep && selectedDef ? (
          <StepEditor
            ct={ct}
            step={selectedStep}
            def={selectedDef}
            onUpdateParam={(k, v) => updateParam(selectedStep.uid, k, v)}
            onUpdateNote={(note) => updateStep(selectedStep.uid, { note })}
            onToggle={() => updateStep(selectedStep.uid, { enabled: !selectedStep.enabled })}
            onDuplicate={() => duplicateStep(selectedStep.uid)}
            onRemove={() => removeStep(selectedStep.uid)}
            onMoveUp={() => moveStep(selectedStep.uid, -1)}
            onMoveDown={() => moveStep(selectedStep.uid, 1)}
          />
        ) : (
          <div className={`flex h-full items-center justify-center p-6 text-center text-xs ${ct.muted}`}>
            Select a step from the flowchart to edit its parameters.
          </div>
        )}
      </div>

      {/* Mobile drawer backdrop */}
      {isMobile && (paletteOpen || (selectedStep && editorOpenMobile)) && (
        <div
          className="fixed inset-0 z-30 bg-black/50"
          onClick={() => { setPaletteOpen(false); setEditorOpenMobile(false); if (editorOpenMobile) setSelectedUid(null); }}
        />
      )}

      {/* Floating drag ghost */}
      {drag && (
        <div
          className={`pointer-events-none fixed z-[60] rounded-lg border px-3 py-1.5 text-xs font-bold shadow-2xl ${ct.active}`}
          style={{ left: drag.x + 12, top: drag.y + 10 }}
        >
          <GripVertical size={11} className="mr-1 inline opacity-60" />
          {drag.label}
        </div>
      )}

      {/* Insert-at-edge command picker */}
      {insertAt !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
          onClick={() => setInsertAt(null)}
          role="dialog"
          aria-modal="true"
          aria-label="Add command at connection"
        >
          <div
            className={`flex max-h-[70vh] w-full max-w-md flex-col rounded-xl border shadow-2xl ${ct.card}`}
            onClick={e => e.stopPropagation()}
          >
            <div className={`flex items-center justify-between border-b px-4 py-2.5 ${ct.divider}`}>
              <span className="text-xs font-semibold">Add command at position {(insertAt ?? 0) + 1}</span>
              <button onClick={() => setInsertAt(null)} className={`rounded p-1 ${ct.muted}`}><X size={14} /></button>
            </div>
            <div className={`border-b px-3 py-2 ${ct.divider}`}>
              <input
                autoFocus
                value={insertSearch}
                onChange={e => setInsertSearch(e.target.value)}
                placeholder="Filter commands…"
                className={`w-full rounded border px-2 py-1.5 text-xs focus:outline-none ${ct.input}`}
              />
            </div>
            <div className="overflow-y-auto p-2">
              {insertCandidates.map(cmd => (
                <button
                  key={cmd.id}
                  onClick={() => { insertStepAt(cmd.id, insertAt ?? model.steps.length); setInsertAt(null); setInsertSearch(''); }}
                  title={cmd.doc ?? cmd.label}
                  className={`flex w-full items-center justify-between rounded px-2 py-1.5 text-left text-[11px] transition-colors ${ct.muted} ${ct.hoverSurface}`}
                >
                  <span className="truncate">{cmd.label}</span>
                  <span className={`ml-2 shrink-0 rounded px-1 py-0.5 text-[9px] uppercase opacity-70 ${ct.chip}`}>
                    {SECTION_LABELS[cmd.section].split('·')[0]}
                  </span>
                </button>
              ))}
              {insertCandidates.length === 0 && (
                <p className={`p-3 text-center text-xs ${ct.muted}`}>No matching commands.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Flowchart view                                                      */
/* ------------------------------------------------------------------ */

interface FlowchartViewProps {
  ct: ThemeTokens;
  flow: FlowGraph;
  steps: ScriptStep[];
  selectedUid: string | null;
  dragUid: string | null;
  overIndex: number | null;
  edgeMenuIndex: number | null;
  onSelect: (uid: string) => void;
  onToggle: (uid: string) => void;
  onRemove: (uid: string) => void;
  onDuplicate: (uid: string) => void;
  onMove: (uid: string, dir: -1 | 1) => void;
  onCardPointerDown: (uid: string) => (e: React.PointerEvent<HTMLDivElement>) => void;
  onCardPointerMove: (e: React.PointerEvent<HTMLDivElement>) => void;
  onCardPointerUp: () => void;
  registerNodeRef: (uid: string, el: HTMLDivElement | null) => void;
  onEdgeClick: (index: number) => void;
  onEdgeInsertHere: (index: number) => void;
  onEdgeDisableNext: (index: number) => void;
  onEdgeRemoveNext: (index: number) => void;
  onCloseEdgeMenu: () => void;
}

const DropLine: React.FC<{ ct: ThemeTokens; active: boolean }> = ({ ct, active }) =>
  active ? (
    <div className="my-1 flex items-center" data-drop-active>
      <div className="h-1 w-full rounded-full bg-[#7fa66b] shadow-[0_0_8px_rgba(127,166,107,0.7)]" />
    </div>
  ) : null;

const FlowchartView: React.FC<FlowchartViewProps> = ({
  ct, flow, steps, selectedUid, dragUid, overIndex, edgeMenuIndex,
  onSelect, onToggle, onRemove, onDuplicate, onMove,
  onCardPointerDown, onCardPointerMove, onCardPointerUp, registerNodeRef,
  onEdgeClick, onEdgeInsertHere, onEdgeDisableNext, onEdgeRemoveNext, onCloseEdgeMenu,
}) => {
  const shortLabel = (s: ScriptStep): string => COMMAND_BY_ID[s.defId]?.command ?? s.defId;

  const edgeRow = (i: number) => {
    const menuOpen = edgeMenuIndex === i;
    return (
      <div key={`edge-${i}`} className="relative flex flex-col items-center">
        <div className={`h-3 w-px bg-[#453a2b]`} />
        <button
          onClick={(e) => { e.stopPropagation(); onEdgeClick(i); }}
          title="Edit this connection — insert or rewire steps here"
          className={`flex h-5 items-center gap-1 rounded-full border px-2 text-[9px] font-medium transition-colors ${
            menuOpen
              ? ct.edgeActive
              : ct.edgePill
          }`}
          aria-label={`Connection ${i + 1} actions`}
        >
          <Link2 size={10} />
          connect
        </button>
        <div className={`h-3 w-px bg-[#453a2b]`} />

        {menuOpen && (
          <div
            className={`absolute left-1/2 top-full z-30 w-56 -translate-x-1/2 rounded-lg border p-1.5 shadow-2xl ${ct.card}`}
            onClick={e => e.stopPropagation()}
          >
            <p className={`px-2 pb-1 pt-0.5 text-[9px] uppercase tracking-wide ${ct.muted}`}>Edit pipeline here</p>
            <MenuItem ct={ct} icon={<Plus size={12} />} label="Add command at this point…" onClick={() => onEdgeInsertHere(i)} />
            {steps[i] && (
              <>
                <MenuItem
                  ct={ct}
                  icon={steps[i].enabled ? <EyeOff size={12} /> : <Eye size={12} />}
                  label={steps[i].enabled ? `Disable “${shortLabel(steps[i])}”` : `Enable “${shortLabel(steps[i])}”`}
                  onClick={() => onEdgeDisableNext(i)}
                />
                <MenuItem ct={ct} icon={<Trash2 size={12} />} danger label={`Remove “${shortLabel(steps[i])}”`} onClick={() => onEdgeRemoveNext(i)} />
              </>
            )}
            <button
              onClick={onCloseEdgeMenu}
              className={`mt-0.5 w-full rounded px-2 py-1 text-left text-[10px] ${ct.muted} ${ct.hoverSurface}`}
            >
              Close
            </button>
          </div>
        )}
      </div>
    );
  };

  if (flow.nodes.length === 0) {
    return (
      <div className={`flex h-full items-center justify-center text-sm ${ct.muted}`}>
        <div className="text-center">
          <Workflow size={32} className="mx-auto mb-3 opacity-40" />
          <p>Empty pipeline — add commands, import a script, or pick a template.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center gap-0 p-6 ${dragUid ? 'select-none' : ''}`}>
      {/* START */}
      <div className="flex items-center gap-2 text-[10px] text-[#a3937f]">
        <div className={`h-px w-8 ${ct.edgeLine.replace('bg-', 'bg-')}`} />
        <span className={`rounded-full px-3 py-0.5 ${ct.startBadge}`}>
          START
        </span>
      </div>
      {edgeRow(0)}

      {flow.nodes.map((node, i) => {
        const step = steps.find(s => s.uid === node.uid);
        const isSel = selectedUid === node.uid;
        const isDragging = dragUid === node.uid;
        return (
          <React.Fragment key={node.uid}>
            <DropLine ct={ct} active={dragUid !== null && overIndex === i} />
            <div
              ref={el => registerNodeRef(node.uid, el)}
              onPointerDown={onCardPointerDown(node.uid)}
              onPointerMove={onCardPointerMove}
              onPointerUp={onCardPointerUp}
              onPointerCancel={onCardPointerUp}
              onClick={() => onSelect(node.uid)}
              style={{ touchAction: 'none' }}
              className={`group relative w-full cursor-grab rounded-xl border p-3 transition-shadow active:cursor-grabbing ${
                node.defId === 'raw_line'
                  ? `border-dashed border-[#6b5124]/70 bg-[#332612]/25 ${isSel ? 'ring-1 ring-[#d9a05b]/60' : ''}`
                  : isSel
                    ? `${ct.active} ring-1 ring-[#7fa66b]/40`
                    : node.enabled
                      ? ct.nodeCard
                      : `${ct.nodeDisabled} opacity-50`
              } ${isDragging ? 'opacity-30' : ''}`}
            >
              <div className="mb-1 flex items-center justify-between">
                <span className={`text-[9px] uppercase tracking-wide ${ct.muted}`}>
                  <GripVertical size={9} className="mr-1 inline opacity-50" />
                  {SECTION_LABELS[node.section as keyof typeof SECTION_LABELS]?.split('·')[1]?.trim() ?? node.section}
                </span>
                <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    onClick={(e) => { e.stopPropagation(); onMove(node.uid, -1); }}
                    className={`rounded p-0.5 ${ct.muted} ${ct.hoverSurface}`}
                    title="Move up"
                  >
                    <ChevronUp size={12} />
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onMove(node.uid, 1); }}
                    className={`rounded p-0.5 ${ct.muted} ${ct.hoverSurface}`}
                    title="Move down"
                  >
                    <ChevronDown size={12} />
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onToggle(node.uid); }}
                    className={`rounded p-0.5 ${ct.muted} ${ct.hoverSurface}`}
                    title={node.enabled ? 'Disable' : 'Enable'}
                  >
                    {node.enabled ? <Eye size={12} /> : <EyeOff size={12} />}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDuplicate(node.uid); }}
                    className={`rounded p-0.5 ${ct.muted} ${ct.hoverSurface}`}
                    title="Duplicate step"
                  >
                    <Copy size={12} />
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onRemove(node.uid); }}
                    className={`rounded p-0.5 ${ct.dangerItem}`}
                    title="Remove"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <code className={`text-xs font-bold ${node.defId === 'raw_line' ? 'text-[#e4b877]' : ct.accentCode}`}>
                  {node.defId === 'raw_line' ? 'raw' : node.label}
                </code>
                {node.defId === 'raw_line' && (
                  <span className={`rounded px-1 py-0.5 text-[8px] font-semibold uppercase tracking-wide ${ct.warn.split(' ').slice(-1)}`}>
                    not in library
                  </span>
                )}
              </div>
              {node.sublabel && (
                <div className={`mt-0.5 truncate text-[10px] font-mono ${ct.muted}`}>{node.sublabel}</div>
              )}
            </div>
            {i === flow.nodes.length - 1 && <DropLine ct={ct} active={dragUid !== null && overIndex === flow.nodes.length} />}
            {i < flow.nodes.length - 1 && edgeRow(i + 1)}
          </React.Fragment>
        );
      })}

      <div className="flex items-center gap-2 text-[10px] text-[#a3937f]">
        <div className={`h-px w-8 ${ct.edgeLine}`} />
        <span className={`rounded-full px-3 py-0.5 ${ct.endBadge}`}>
          END
        </span>
      </div>
      <p className={`mt-3 text-center text-[10px] ${ct.muted}`}>
        Grab a card and drop it between two others — it locks into place · click any <span className={ct.accentText}>connect</span> pill to insert there.
      </p>
    </div>
  );
};

const MenuItem: React.FC<{ ct: ThemeTokens; icon: React.ReactNode; label: string; onClick: () => void; danger?: boolean }> = ({
  ct, icon, label, onClick, danger,
}) => (
  <button
    onClick={onClick}
    className={`flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-[11px] transition-colors ${
      danger ? ct.dangerItem : `${ct.text} ${ct.hoverSurface}`
    }`}
  >
    {icon} {label}
  </button>
);

/* ------------------------------------------------------------------ */
/* Step editor panel                                                   */
/* ------------------------------------------------------------------ */

interface StepEditorProps {
  ct: ThemeTokens;
  step: ScriptStep;
  def: import('../../lammps/catalog').CommandDef;
  onUpdateParam: (key: string, value: string) => void;
  onUpdateNote: (note?: string) => void;
  onToggle: () => void;
  onDuplicate: () => void;
  onRemove: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}

const StepEditor: React.FC<StepEditorProps> = ({
  ct, step, def, onUpdateParam, onUpdateNote, onToggle, onDuplicate, onRemove, onMoveUp, onMoveDown,
}) => {
  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h3 className={`text-sm font-bold ${ct.headerText}`}>{def.label}</h3>
        <div className="flex items-center gap-1">
          <button onClick={onMoveUp} className={`rounded p-1 ${ct.muted} ${ct.hoverSurface}`} title="Move up">
            <ChevronUp size={14} />
          </button>
          <button onClick={onMoveDown} className={`rounded p-1 ${ct.muted} ${ct.hoverSurface}`} title="Move down">
            <ChevronDown size={14} />
          </button>
        </div>
      </div>

      {def.doc && (
        <a href={def.doc} target="_blank" rel="noopener noreferrer" className={`block text-[10px] ${ct.accentText} hover:underline`}>
          📖 LAMMPS docs ↗
        </a>
      )}

      <div className="space-y-1">
        <label className={`text-[10px] font-semibold ${ct.muted}`}>Comment (optional)</label>
        <input
          value={step.note ?? ''}
          onChange={e => onUpdateNote(e.target.value || undefined)}
          placeholder="# your note…"
          className={`w-full rounded border px-2 py-1.5 text-xs focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
        />
      </div>

      <div className="space-y-3">
        {def.params.map(pd => (
          <ParamControl key={pd.key} ct={ct} def={pd} value={step.params[pd.key] ?? ''} onChange={v => onUpdateParam(pd.key, v)} />
        ))}
      </div>

      <div className={`flex flex-wrap items-center gap-2 border-t pt-3 ${ct.divider}`}>
        <button
          onClick={onToggle}
          className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
            step.enabled ? ct.enabledBtn : ct.disabledBtn
          }`}
        >
          {step.enabled ? <Eye size={13} /> : <EyeOff size={13} />}
          {step.enabled ? 'Enabled' : 'Disabled'}
        </button>
        <button
          onClick={onDuplicate}
          className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium ${ct.muted} ${ct.hoverSurface}`}
          title="Duplicate this step with its parameters"
        >
          <Copy size={13} /> Duplicate
        </button>
        <button
          onClick={onRemove}
          className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium ${ct.removeBtn}`}
        >
          <Trash2 size={13} /> Remove
        </button>
      </div>
    </div>
  );
};

const ParamControl: React.FC<{ ct: ThemeTokens; def: ParamDef; value: string; onChange: (v: string) => void }> = ({
  ct, def, value, onChange,
}) => {
  const label = (
    <label className={`text-[10px] font-semibold ${ct.muted}`}>
      {def.label}
      {def.help && <span className={`ml-1 font-normal opacity-60`}>— {def.help}</span>}
    </label>
  );

  if (def.type === 'enum' && def.options) {
    const known = def.options.some(o => o.value === value);
    return (
      <div className="space-y-1">
        {label}
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className={`w-full rounded border px-2 py-1.5 text-xs focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
        >
          {!known && value !== '' && <option value={value}>{value} (imported)</option>}
          {def.options.map(o => (
            <option key={o.value} value={o.value}>{o.label ?? o.value}</option>
          ))}
        </select>
      </div>
    );
  }

  if (def.type === 'flag') {
    return (
      <div className="flex items-center justify-between">
        {label}
        <button
          onClick={() => onChange(value === 'yes' ? 'no' : 'yes')}
          role="switch"
          aria-checked={value === 'yes'}
          className={`relative h-5 w-9 rounded-full transition-colors ${value === 'yes' ? ct.toggleOn : ct.toggleOff}`}
        >
          <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${value === 'yes' ? 'left-4' : 'left-0.5'}`} />
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {label}
      <input
        type={def.type === 'number' ? 'number' : 'text'}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={def.placeholder}
        className={`w-full rounded border px-2 py-1.5 text-xs focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
      />
    </div>
  );
};

export default ScriptBuilder;
