import React, { useState, useCallback, useMemo, useRef } from 'react';
import {
  ALL_COMMANDS,
  COMMAND_BY_ID,
  SECTION_LABELS,
  SECTION_ORDER,
  ScriptModel,
  ScriptStep,
  defaultParams,
  ParamDef,
} from '../../lammps/catalog';
import { generateScript, deriveFlowchart, FlowGraph } from '../../lammps/generator';
import { downloadTextFile } from '../../lammps/exporter';
import { SCRIPT_TEMPLATES, buildTemplate } from '../../lammps/templates';
import { useUndoableState } from '../../hooks/useUndoableState';
import { browserStore, loadJson, saveJson } from '../../services/persistence';
import { getThemeTokens, ThemeTokens, Theme } from '../../theme';
import {
  Plus, Trash2, Copy, Download, Eye, EyeOff,
  FileCode2, Workflow, ChevronDown, ChevronRight, ChevronUp, ChevronLeft, Search,
  Atom as AtomIcon, PencilLine, X, GripVertical, Link2, Undo2, Redo2, LayoutTemplate,
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
  };
};

const DEFAULT_MODEL: ScriptModel = { title: 'My LAMMPS Simulation', steps: [] };

const MODEL_KEY = 'm3d.scriptModel.v1';

const ScriptBuilder: React.FC<ScriptBuilderProps> = ({ theme, onOpenViewer }) => {
  const ct = getThemeTokens(theme);
  // Undoable model; the plain value persists to localStorage for durability.
  const [model, setModel, undo, redo, canUndo, canRedo] = (() => {
    const api = useUndoableState<ScriptModel>(() => {
      const stored = loadJson<ScriptModel>(browserStore(), MODEL_KEY, reviveModel);
      return stored ?? DEFAULT_MODEL;
    });
    return [api.value, api.set, api.undo, api.redo, api.canUndo, api.canRedo] as const;
  })();
  const [templatesOpen, setTemplatesOpen] = useState(false);
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(true);
  const [view, setView] = useState<'flow' | 'script'>('flow');
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);

  // Persist the model whenever it changes (history itself is in-memory).
  React.useEffect(() => {
    saveJson(browserStore(), MODEL_KEY, model);
  }, [model]);

  // Undo/redo + Delete-selected keyboard layer (builder-local; the manual
  // textarea keeps native undo because typing targets are skipped).
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && !e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
      if ((mod && e.shiftKey && e.key.toLowerCase() === 'z') || (mod && e.key.toLowerCase() === 'y')) { e.preventDefault(); redo(); return; }
      if (!mod && (e.key === 'Delete') && selectedUid) { e.preventDefault(); removeStepRef.current(selectedUid); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [undo, redo, selectedUid]);

  // Drag-to-reorder state (HTML5 DnD)
  const [dragUid, setDragUid] = useState<string | null>(null);
  const [overIndex, setOverIndex] = useState<number | null>(null);

  // Connection-editing menu: which edge (insertion index) is open
  const [edgeMenuIndex, setEdgeMenuIndex] = useState<number | null>(null);

  // Insert-command picker: insertion index it will place into
  const [insertAt, setInsertAt] = useState<number | null>(null);
  const [insertSearch, setInsertSearch] = useState('');

  const isManual = model.manualText !== undefined;

  const generated = useMemo(() => generateScript(model), [model]);
  const flow = useMemo(() => deriveFlowchart(model), [model]);
  /** The text currently shown/copied/downloaded. */
  const activeText = isManual ? (model.manualText ?? '') : generated.text;

  // ---- model mutations -------------------------------------------------
  const removeStepRef = useRef<(uid: string) => void>(() => {});

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

  /** Duplicate a step (same command + params) right below it. */
  const duplicateStep = useCallback((uid: string) => {
    setModel(prev => {
      const idx = prev.steps.findIndex(s => s.uid === uid);
      if (idx < 0) return prev;
      const src = prev.steps[idx];
      const clone: ScriptStep = {
        ...src,
        uid: newUid(),
        params: { ...src.params },
        note: src.note,
      };
      const steps = [...prev.steps];
      steps.splice(idx + 1, 0, clone);
      return { ...prev, steps };
    });
  }, [setModel]);

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
      const insertAt2 = clamped > from ? clamped - 1 : clamped;
      steps.splice(insertAt2, 0, dragged);
      return { ...prev, steps };
    });
  }, [setModel]);

  // ---- manual script editing -------------------------------------------
  const enterManualMode = useCallback(() => {
    setModel(prev => ({
      ...prev,
      manualText: prev.manualText ?? generateScript({ ...prev }).text,
    }));
    setView('script');
  }, [setModel]);

  const exitManualMode = useCallback(() => {
    setModel(prev => {
      const { manualText: _drop, ...rest } = prev;
      void _drop;
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

  // ---- drag handlers ----------------------------------------------------
  const handleDrop = useCallback((targetIndex: number) => {
    if (dragUid) moveStepToIndex(dragUid, targetIndex);
    setDragUid(null);
    setOverIndex(null);
  }, [dragUid, moveStepToIndex]);

  const dropProps = (index: number) => ({
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      setOverIndex(index);
    },
    onDragLeave: () => setOverIndex(prev => (prev === index ? null : prev)),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      handleDrop(index);
    },
  });

  return (
    <div className="flex h-full min-h-0">
      {/* Left: palette */}
      <div className={`flex flex-col border-r transition-all ${ct.panel} ${paletteOpen ? 'w-64' : 'w-0 overflow-hidden'}`}>
        <div className={`flex items-center justify-between px-3 py-2 border-b ${ct.divider}`}>
          <span className={`text-xs font-semibold ${ct.headerText}`}>Commands</span>
          <button onClick={() => setPaletteOpen(false)} className={`p-1 rounded ${ct.muted} ${ct.hoverSurface}`}>
            <ChevronLeft size={14} />
          </button>
        </div>
        <div className={`px-3 py-2 border-b ${ct.divider}`}>
          <div className="flex items-center gap-1.5">
            <Search size={13} className={ct.muted} />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder={`Search ${ALL_COMMANDS.length} commands…`}
              className={`flex-1 bg-transparent text-xs ${ct.text} placeholder:text-[#6f6353] focus:outline-none`}
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-3">
          {SECTION_ORDER.map(section => {
            const cmds = paletteBySection.get(section);
            if (!cmds || cmds.length === 0) return null;
            return (
              <div key={section} className="space-y-0.5">
                <div className={`text-[10px] font-semibold uppercase px-1 pb-0.5 ${ct.muted}`}>
                  {SECTION_LABELS[section]}
                </div>
                {cmds.map(cmd => (
                  <button
                    key={cmd.id}
                    onClick={() => addStep(cmd.id)}
                    title={cmd.doc ?? cmd.label}
                    className={`flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-[11px] transition-colors ${ct.muted} ${ct.hoverSurface}`}
                  >
                    <Plus size={11} className="shrink-0 opacity-50" />
                    <span className="truncate">{cmd.label}</span>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Center: pipeline + step editor */}
      <div className={`flex min-w-0 flex-1 flex-col ${ct.bg}`}>
        {/* Toolbar */}
        <div className={`flex h-10 shrink-0 items-center justify-between border-b px-3 ${ct.divider}`}>
          <div className="flex items-center gap-2">
            {!paletteOpen && (
              <button onClick={() => setPaletteOpen(true)} className={`p-1 rounded ${ct.muted} ${ct.hoverSurface}`} title="Open palette">
                <ChevronRight size={16} />
              </button>
            )}
            <input
              value={model.title}
              onChange={e => setModel(prev => ({ ...prev, title: e.target.value }))}
              className={`w-32 min-w-0 flex-shrink rounded bg-transparent px-1.5 py-0.5 text-sm font-medium focus:outline-none focus:ring-1 focus:ring-[#7fa66b] sm:w-40 ${ct.headerText}`}
            />
            <span className={`hidden whitespace-nowrap text-[10px] md:block ${ct.muted}`}>
              {isManual ? 'manual' : `${model.steps.length} steps · ${generated.emitted.length} lines`}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
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
              onClick={() => setView(v => v === 'flow' ? 'script' : 'flow')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                view === 'flow' ? ct.active : `${ct.muted} ${ct.hoverSurface}`
              }`}
              title="Toggle view"
            >
              {view === 'flow' ? <Workflow size={13} /> : <FileCode2 size={13} />}
              {view === 'flow' ? 'Flowchart' : 'Script'}
            </button>
            {isManual ? (
              <button
                onClick={exitManualMode}
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium ${ct.warnAction}"
                title="Discard manual edits and regenerate from your steps"
              >
                <X size={13} />
                <span className="hidden 2xl:inline">Exit manual</span>
              </button>
            ) : (
              <button
                onClick={enterManualMode}
                className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
                title="Edit the generated script by hand"
              >
                <PencilLine size={13} />
                <span className="hidden 2xl:inline">Edit script</span>
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
                <AtomIcon size={13} /> Viewer
              </button>
            )}
          </div>
        </div>

        {/* Manual-mode notice */}
        {isManual && (
          <div className={`shrink-0 border-b px-3 py-1.5 text-[11px] ${ct.warn}`}>
            ✎ Manual mode — this text is emitted verbatim. Your builder steps are kept safe below the flowchart; press “Exit manual” to regenerate.
          </div>
        )}
        {!isManual && generated.warnings.length > 0 && (
          <div className={`shrink-0 border-t px-3 py-2 text-[11px] ${ct.warn}`}>
            {generated.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}

        {/* Flowchart or Script view */}
        <div className="min-h-0 flex-1 overflow-auto">
          {view === 'flow' ? (
            <FlowchartView
              ct={ct}
              flow={flow}
              steps={model.steps}
              selectedUid={selectedUid}
              dragUid={dragUid}
              overIndex={overIndex}
              edgeMenuIndex={edgeMenuIndex}
              onSelect={setSelectedUid}
              onToggle={(uid) => updateStep(uid, { enabled: !model.steps.find(s => s.uid === uid)?.enabled })}
              onRemove={removeStep}
              onDuplicate={duplicateStep}
              onMove={moveStep}
              onDragStart={(uid) => setDragUid(uid)}
              onDragEnd={() => { setDragUid(null); setOverIndex(null); }}
              onDropAt={handleDrop}
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
          ) : isManual ? (
            <textarea
              value={model.manualText ?? ''}
              onChange={e => setManualText(e.target.value)}
              spellCheck={false}
              className={`h-full min-h-full w-full resize-none p-4 text-[11px] leading-relaxed font-mono focus:outline-none ${ct.bg} ${ct.text}`}
              aria-label="Manual LAMMPS script editor"
            />
          ) : (
            <pre className={`p-4 text-[11px] leading-relaxed font-mono whitespace-pre-wrap ${ct.muted}`}>
              {generated.text}
            </pre>
          )}
        </div>
      </div>

      {/* Right: step editor */}
      <div className={`w-80 shrink-0 overflow-y-auto border-l ${ct.panel}`}>
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
  onDragStart: (uid: string) => void;
  onDragEnd: () => void;
  onDropAt: (index: number) => void;
  onEdgeClick: (index: number) => void;
  onEdgeInsertHere: (index: number) => void;
  onEdgeDisableNext: (index: number) => void;
  onEdgeRemoveNext: (index: number) => void;
  onCloseEdgeMenu: () => void;
}

const FlowchartView: React.FC<FlowchartViewProps> = ({
  ct, flow, steps, selectedUid, dragUid, overIndex, edgeMenuIndex,
  onSelect, onToggle, onRemove, onDuplicate, onMove,
  onDragStart, onDragEnd, onDropAt,
  onEdgeClick, onEdgeInsertHere, onEdgeDisableNext, onEdgeRemoveNext, onCloseEdgeMenu,
}) => {
  if (flow.nodes.length === 0) {
    return (
      <div className={`flex h-full items-center justify-center text-sm ${ct.muted}`}>
        <div className="text-center">
          <Workflow size={32} className="mx-auto mb-3 opacity-40" />
          <p>Empty pipeline — add commands from the palette on the left.</p>
        </div>
      </div>
    );
  }

  const edgeRow = (i: number) => {
    const isActive = overIndex === i && dragUid !== null;
    const menuOpen = edgeMenuIndex === i;
    return (
      <div key={`edge-${i}`} className="relative flex flex-col items-center" {...dropPropsFor(i)}>
        {/* connector line */}
        <div className={`h-3 w-px ${isActive ? 'w-1 bg-[#7fa66b]' : ct.edgeLine}`} />
        {/* connection pill */}
        <button
          onClick={(e) => { e.stopPropagation(); onEdgeClick(i); }}
          title="Edit this connection — insert or rewire steps here"
          className={`group flex h-5 items-center gap-1 rounded-full border px-2 text-[9px] font-medium transition-colors ${
            menuOpen || isActive
              ? ct.edgeActive
              : ct.edgePill
          }`}
          aria-label={`Connection ${i + 1} actions`}
        >
          <Link2 size={10} />
          connect
        </button>
        <div className={`h-3 w-px ${isActive ? 'w-1 bg-[#7fa66b]' : ct.edgeLine}`} />

        {/* connection action menu */}
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

  const dropPropsFor = (index: number) => ({
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      onDropAt(index);
    },
  });

  const shortLabel = (s: ScriptStep): string => COMMAND_BY_ID[s.defId]?.command ?? s.defId;

  return (
    <div className="flex flex-col items-center gap-0 p-6">
      {/* Start node */}
      <div className="flex items-center gap-2 text-[10px] text-[#a3937f]">
        <div className={`h-px w-8 ${ct.edgeLine}`} />
        <span className="rounded-full px-3 py-0.5 ${ct.startBadge}">
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
            <div
              draggable
              onDragStart={(e) => { e.dataTransfer.effectAllowed = 'move'; onDragStart(node.uid); }}
              onDragEnd={onDragEnd}
              {...dropPropsFor(i)}
              onClick={() => onSelect(node.uid)}
              className={`group relative w-full max-w-md cursor-grab rounded-xl border p-3 transition-all active:cursor-grabbing ${
                isSel
                  ? ct.active + ' ring-1 ring-[#7fa66b]/40'
                  : node.enabled
                    ? ct.nodeCard
                    : ct.nodeDisabled + ' opacity-50'
              } ${isDragging ? 'opacity-30' : ''}`}
            >
              {/* Section badge + controls */}
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
                    className={`rounded p-0.5 ${ct.muted} hover:text-[#ede5d8]`}
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
              {/* Command + params */}
              <div className="flex items-center gap-2">
                <code className="text-xs font-bold ${ct.accentCode}">{node.label}</code>
              </div>
              {node.sublabel && (
                <div className={`mt-0.5 truncate text-[10px] font-mono ${ct.muted}`}>{node.sublabel}</div>
              )}
            </div>
            {edgeRow(i + 1)}
          </React.Fragment>
        );
      })}

      <div className="flex items-center gap-2 text-[10px] text-[#a3937f]">
        <div className={`h-px w-8 ${ct.edgeLine}`} />
        <span className="rounded-full px-3 py-0.5 ${ct.endBadge}">
          END
        </span>
      </div>
      <p className={`mt-3 text-center text-[10px] ${ct.muted}`}>
        Drag cards to reorder · click any <span className={ct.accentText}>connect</span> pill to insert or edit at that spot.
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

      {/* Comment / note */}
      <div className="space-y-1">
        <label className={`text-[10px] font-semibold ${ct.muted}`}>Comment (optional)</label>
        <input
          value={step.note ?? ''}
          onChange={e => onUpdateNote(e.target.value || undefined)}
          placeholder="# your note…"
          className={`w-full rounded border px-2 py-1.5 text-xs focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
        />
      </div>

      {/* Parameters */}
      <div className="space-y-3">
        {def.params.map(pd => (
          <ParamControl key={pd.key} ct={ct} def={pd} value={step.params[pd.key] ?? ''} onChange={v => onUpdateParam(pd.key, v)} />
        ))}
      </div>

      {/* Toggle + remove */}
      <div className={`flex items-center gap-2 border-t pt-3 ${ct.divider}`}>
        <button
          onClick={onToggle}
          className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
            step.enabled ? ct.enabledBtn : ct.disabledBtn}`}
        >
          {step.enabled ? <Eye size={13} /> : <EyeOff size={13} />}
          {step.enabled ? 'Enabled' : 'Disabled'}
        </button>
        <button
          onClick={onRemove}
          className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium ${ct.removeBtn}"
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
    return (
      <div className="space-y-1">
        {label}
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className={`w-full rounded border px-2 py-1.5 text-xs focus:border-[#7fa66b] focus:outline-none ${ct.input}`}
        >
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
