import React, { useState, useCallback, useMemo } from 'react';
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
import {
  Plus, Trash2, Copy, Download, Eye, EyeOff,
  FileCode2, Workflow, ChevronDown, ChevronRight, ChevronUp, ChevronLeft, Search,
  Atom as AtomIcon,
} from 'lucide-react';

interface ScriptBuilderProps {
  onOpenViewer?: () => void;
}

let uidCounter = 1;
const newUid = () => `step-${uidCounter++}`;

const ScriptBuilder: React.FC<ScriptBuilderProps> = ({ onOpenViewer }) => {
  const [model, setModel] = useState<ScriptModel>({
    title: 'My LAMMPS Simulation',
    steps: [],
  });
  const [selectedUid, setSelectedUid] = useState<string | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(true);
  const [view, setView] = useState<'flow' | 'script'>('flow');
  const [search, setSearch] = useState('');
  const [copied, setCopied] = useState(false);

  const generated = useMemo(() => generateScript(model), [model]);
  const flow = useMemo(() => deriveFlowchart(model), [model]);

  const addStep = useCallback((defId: string) => {
    const def = COMMAND_BY_ID[defId];
    if (!def) return;
    const step: ScriptStep = {
      uid: newUid(),
      defId,
      params: defaultParams(def),
      enabled: true,
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
  }, []);

  const updateStep = useCallback((uid: string, patch: Partial<ScriptStep>) => {
    setModel(prev => ({
      ...prev,
      steps: prev.steps.map(s => (s.uid === uid ? { ...s, ...patch } : s)),
    }));
  }, []);

  const updateParam = useCallback((uid: string, key: string, value: string) => {
    setModel(prev => ({
      ...prev,
      steps: prev.steps.map(s =>
        s.uid === uid ? { ...s, params: { ...s.params, [key]: value } } : s
      ),
    }));
  }, []);

  const removeStep = useCallback((uid: string) => {
    setModel(prev => ({ ...prev, steps: prev.steps.filter(s => s.uid !== uid) }));
    setSelectedUid(prev => (prev === uid ? null : prev));
  }, []);

  const moveStep = useCallback((uid: string, dir: -1 | 1) => {
    setModel(prev => {
      const steps = [...prev.steps];
      const i = steps.findIndex(s => s.uid === uid);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= steps.length) return prev;
      [steps[i], steps[j]] = [steps[j], steps[i]];
      return { ...prev, steps };
    });
  }, []);

  const copyScript = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(generated.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch { /* clipboard blocked */ }
  }, [generated.text]);

  const filteredCommands = useMemo(() => {
    if (!search.trim()) return ALL_COMMANDS;
    const q = search.toLowerCase();
    return ALL_COMMANDS.filter(d =>
      d.label.toLowerCase().includes(q) ||
      d.command.toLowerCase().includes(q) ||
      d.category.toLowerCase().includes(q)
    );
  }, [search]);

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

  return (
    <div className="flex h-full min-h-0">
      {/* Left: palette */}
      <div className={`flex flex-col border-r border-gray-800 bg-[#16191d] transition-all ${paletteOpen ? 'w-64' : 'w-0 overflow-hidden'}`}>
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
          <span className="text-xs font-semibold text-gray-300">Commands</span>
          <button onClick={() => setPaletteOpen(false)} className="p-1 rounded text-gray-500 hover:text-gray-300">
            <ChevronLeft size={14} />
          </button>
        </div>
        <div className="px-3 py-2 border-b border-gray-800">
          <div className="flex items-center gap-1.5">
            <Search size={13} className="text-gray-500" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search commands…"
              className="flex-1 bg-transparent text-xs text-gray-300 placeholder:text-gray-600 focus:outline-none"
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-3">
          {SECTION_ORDER.map(section => {
            const cmds = paletteBySection.get(section);
            if (!cmds || cmds.length === 0) return null;
            return (
              <div key={section} className="space-y-0.5">
                <div className="text-[10px] font-semibold uppercase text-gray-600 px-1 pb-0.5">
                  {SECTION_LABELS[section]}
                </div>
                {cmds.map(cmd => (
                  <button
                    key={cmd.id}
                    onClick={() => addStep(cmd.id)}
                    title={cmd.doc ?? cmd.label}
                    className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left text-[11px] text-gray-400 hover:bg-gray-800/60 hover:text-gray-200 transition-colors"
                  >
                    <Plus size={11} className="shrink-0 text-gray-600" />
                    <span className="truncate">{cmd.label}</span>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      </div>

      {/* Center: pipeline + step editor */}
      <div className="flex min-w-0 flex-1 flex-col bg-[#101214]">
        {/* Toolbar */}
        <div className="flex h-10 shrink-0 items-center justify-between border-b border-gray-800 px-3">
          <div className="flex items-center gap-2">
            {!paletteOpen && (
              <button onClick={() => setPaletteOpen(true)} className="p-1 rounded text-gray-500 hover:text-gray-300" title="Open palette">
                <ChevronRight size={16} />
              </button>
            )}
            <input
              value={model.title}
              onChange={e => setModel(prev => ({ ...prev, title: e.target.value }))}
              className="bg-transparent text-sm font-medium text-gray-200 focus:outline-none focus:ring-1 focus:ring-blue-500 rounded px-1.5 py-0.5"
              aria-label="Script title"
            />
            <span className="text-[10px] text-gray-600">
              {model.steps.length} steps · {generated.emitted.length} lines
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setView(v => v === 'flow' ? 'script' : 'flow')}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                view === 'flow' ? 'bg-blue-600/20 text-blue-300' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}
              title="Toggle view"
            >
              {view === 'flow' ? <Workflow size={13} /> : <FileCode2 size={13} />}
              {view === 'flow' ? 'Flowchart' : 'Script'}
            </button>
            <button
              onClick={copyScript}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
              title="Copy script to clipboard"
            >
              {copied ? <span className="text-emerald-400">✓ Copied</span> : <><Copy size={13} /> Copy</>}
            </button>
            <button
              onClick={() => downloadTextFile('in.lammps', generated.text)}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
              title="Download in.lammps"
            >
              <Download size={13} /> Download
            </button>
            {onOpenViewer && (
              <button
                onClick={onOpenViewer}
                className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
                title="Open 3D structure viewer"
              >
                <AtomIcon size={13} /> Viewer
              </button>
            )}
          </div>
        </div>

        {/* Flowchart or Script view */}
        <div className="min-h-0 flex-1 overflow-auto">
          {view === 'flow' ? (
            <FlowchartView
              flow={flow}
              steps={model.steps}
              selectedUid={selectedUid}
              onSelect={setSelectedUid}
              onToggle={(uid) => updateStep(uid, { enabled: !model.steps.find(s => s.uid === uid)?.enabled })}
              onRemove={removeStep}
              onMove={moveStep}
            />
          ) : (
            <pre className="p-4 text-[11px] leading-relaxed font-mono text-gray-300 whitespace-pre-wrap">
              {generated.text}
            </pre>
          )}
        </div>

        {/* Warnings */}
        {generated.warnings.length > 0 && (
          <div className="shrink-0 border-t border-amber-800/50 bg-amber-950/30 px-3 py-2 text-[11px] text-amber-300">
            {generated.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}
      </div>

      {/* Right: step editor */}
      <div className="w-80 shrink-0 border-l border-gray-800 bg-[#16191d] overflow-y-auto">
        {selectedStep && selectedDef ? (
          <StepEditor
            step={selectedStep}
            def={selectedDef}
            onUpdateParam={(k, v) => updateParam(selectedStep.uid, k, v)}
            onUpdateNote={(note) => updateStep(selectedStep.uid, { note })}
            onToggle={() => updateStep(selectedStep.uid, { enabled: !selectedStep.enabled })}
            onRemove={() => removeStep(selectedStep.uid)}
            onMoveUp={() => moveStep(selectedStep.uid, -1)}
            onMoveDown={() => moveStep(selectedStep.uid, 1)}
          />
        ) : (
          <div className="flex h-full items-center justify-center p-6 text-center text-xs text-gray-600">
            Select a step from the flowchart to edit its parameters.
          </div>
        )}
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Flowchart view                                                      */
/* ------------------------------------------------------------------ */

interface FlowchartViewProps {
  flow: FlowGraph;
  steps: ScriptStep[];
  selectedUid: string | null;
  onSelect: (uid: string) => void;
  onToggle: (uid: string) => void;
  onRemove: (uid: string) => void;
  onMove: (uid: string, dir: -1 | 1) => void;
}

const FlowchartView: React.FC<FlowchartViewProps> = ({
  flow, steps, selectedUid, onSelect, onToggle, onRemove, onMove,
}) => {
  if (flow.nodes.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-gray-600">
        <div className="text-center">
          <Workflow size={32} className="mx-auto mb-3 opacity-40" />
          <p>Empty pipeline — add commands from the palette on the left.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2 p-6">
      {/* Start node */}
      <div className="flex items-center gap-2 text-[10px] text-gray-600">
        <div className="h-px w-8 bg-gray-700" />
        <span className="rounded-full border border-emerald-700/50 bg-emerald-950/30 px-3 py-0.5 text-emerald-400">
          START
        </span>
      </div>
      <div className="h-4 w-px bg-gray-700" />

      {flow.nodes.map((node, i) => {
        const step = steps.find(s => s.uid === node.uid);
        const isSel = selectedUid === node.uid;
        return (
          <React.Fragment key={node.uid}>
            <div
              onClick={() => onSelect(node.uid)}
              className={`group relative w-full max-w-md cursor-pointer rounded-xl border p-3 transition-all ${
                isSel
                  ? 'border-blue-500 bg-blue-950/30 ring-1 ring-blue-500/30'
                  : node.enabled
                    ? 'border-gray-700 bg-gray-900/40 hover:border-gray-600'
                    : 'border-gray-800 bg-gray-900/20 opacity-50'
              }`}
            >
              {/* Section badge */}
              <div className="mb-1 flex items-center justify-between">
                <span className="text-[9px] uppercase tracking-wide text-gray-600">
                  {SECTION_LABELS[node.section as keyof typeof SECTION_LABELS]?.split('·')[1]?.trim() ?? node.section}
                </span>
                <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    onClick={(e) => { e.stopPropagation(); onMove(node.uid, -1); }}
                    className="rounded p-0.5 text-gray-600 hover:text-gray-300"
                    title="Move up"
                  >
                    <ChevronUp size={12} />
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onMove(node.uid, 1); }}
                    className="rounded p-0.5 text-gray-600 hover:text-gray-300"
                    title="Move down"
                  >
                    <ChevronDown size={12} />
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onToggle(node.uid); }}
                    className="rounded p-0.5 text-gray-600 hover:text-gray-300"
                    title={node.enabled ? 'Disable' : 'Enable'}
                  >
                    {node.enabled ? <Eye size={12} /> : <EyeOff size={12} />}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onRemove(node.uid); }}
                    className="rounded p-0.5 text-red-700 hover:text-red-400"
                    title="Remove"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              </div>
              {/* Command + params */}
              <div className="flex items-center gap-2">
                <code className="text-xs font-bold text-blue-300">{node.label}</code>
              </div>
              {node.sublabel && (
                <div className="mt-0.5 text-[10px] font-mono text-gray-500 truncate">{node.sublabel}</div>
              )}
            </div>
            {i < flow.nodes.length - 1 && (
              <div className="h-4 w-px bg-gray-700" />
            )}
          </React.Fragment>
        );
      })}

      <div className="h-4 w-px bg-gray-700" />
      <div className="flex items-center gap-2 text-[10px] text-gray-600">
        <div className="h-px w-8 bg-gray-700" />
        <span className="rounded-full border border-red-700/50 bg-red-950/30 px-3 py-0.5 text-red-400">
          END
        </span>
      </div>
    </div>
  );
};

/* ------------------------------------------------------------------ */
/* Step editor panel                                                   */
/* ------------------------------------------------------------------ */

interface StepEditorProps {
  step: ScriptStep;
  def: import('../../lammps/catalog').CommandDef;
  onUpdateParam: (key: string, value: string) => void;
  onUpdateNote: (note?: string) => void;
  onToggle: () => void;
  onRemove: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}

const StepEditor: React.FC<StepEditorProps> = ({
  step, def, onUpdateParam, onUpdateNote, onToggle, onRemove, onMoveUp, onMoveDown,
}) => {
  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-gray-200">{def.label}</h3>
        <div className="flex items-center gap-1">
          <button onClick={onMoveUp} className="rounded p-1 text-gray-500 hover:text-gray-300" title="Move up">
            <ChevronUp size={14} />
          </button>
          <button onClick={onMoveDown} className="rounded p-1 text-gray-500 hover:text-gray-300" title="Move down">
            <ChevronDown size={14} />
          </button>
        </div>
      </div>

      {def.doc && (
        <a href={def.doc} target="_blank" rel="noopener noreferrer" className="block text-[10px] text-blue-400 hover:underline">
          📖 LAMMPS docs ↗
        </a>
      )}

      {/* Comment / note */}
      <div className="space-y-1">
        <label className="text-[10px] font-semibold text-gray-500">Comment (optional)</label>
        <input
          value={step.note ?? ''}
          onChange={e => onUpdateNote(e.target.value || undefined)}
          placeholder="# your note…"
          className="w-full rounded border border-gray-700 bg-[#12151a] px-2 py-1.5 text-xs text-gray-300 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none"
        />
      </div>

      {/* Parameters */}
      <div className="space-y-3">
        {def.params.map(pd => (
          <ParamControl key={pd.key} def={pd} value={step.params[pd.key] ?? ''} onChange={v => onUpdateParam(pd.key, v)} />
        ))}
      </div>

      {/* Toggle + remove */}
      <div className="flex items-center gap-2 border-t border-gray-800 pt-3">
        <button
          onClick={onToggle}
          className={`flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium transition-colors ${
            step.enabled ? 'text-emerald-400' : 'text-gray-500'
          } bg-gray-800/60 hover:bg-gray-800`}
        >
          {step.enabled ? <Eye size={13} /> : <EyeOff size={13} />}
          {step.enabled ? 'Enabled' : 'Disabled'}
        </button>
        <button
          onClick={onRemove}
          className="flex items-center gap-1.5 rounded px-2 py-1 text-[11px] font-medium text-red-400 hover:bg-red-950/30"
        >
          <Trash2 size={13} /> Remove
        </button>
      </div>
    </div>
  );
};

const ParamControl: React.FC<{ def: ParamDef; value: string; onChange: (v: string) => void }> = ({
  def, value, onChange,
}) => {
  const label = (
    <label className="text-[10px] font-semibold text-gray-500">
      {def.label}
      {def.help && <span className="ml-1 text-gray-600 font-normal">— {def.help}</span>}
    </label>
  );

  if (def.type === 'enum' && def.options) {
    return (
      <div className="space-y-1">
        {label}
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full rounded border border-gray-700 bg-[#12151a] px-2 py-1.5 text-xs text-gray-300 focus:border-blue-500 focus:outline-none"
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
          className={`relative h-5 w-9 rounded-full transition-colors ${value === 'yes' ? 'bg-blue-600' : 'bg-gray-700'}`}
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
        className="w-full rounded border border-gray-700 bg-[#12151a] px-2 py-1.5 text-xs text-gray-300 placeholder:text-gray-600 focus:border-blue-500 focus:outline-none"
      />
    </div>
  );
};

export default ScriptBuilder;
