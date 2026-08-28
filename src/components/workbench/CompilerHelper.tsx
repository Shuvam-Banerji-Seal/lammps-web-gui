import React, { useState, useMemo, useEffect } from 'react';
import {
  LMP_PACKAGES,
  PACKAGE_CATEGORIES,
  ACCELERATORS,
  PRESETS,
  BUILD_OPTIONS,
  generateBuildScript,
  DEFAULT_COMPILER_OPTIONS,
  CompilerOptions,
  FlagDetail,
  OsTarget,
} from '../../lammps/compiler';
import { downloadTextFile } from '../../lammps/exporter';
import { usePersistentState } from '../../hooks/usePersistentState';
import { getThemeTokens, ThemeTokens, Theme } from '../../theme';
import { Copy, Download, Terminal, Monitor, Cpu, Package, Zap, Settings, ChevronDown, ChevronUp, X, Info, SlidersHorizontal } from 'lucide-react';

const CATEGORY_LABELS: Record<string, string> = {
  'accel': 'Accelerators',
  'force-fields': 'Force fields',
  'molecular': 'Molecular',
  'methods': 'Methods & analysis',
  'ml': 'Machine-learning potentials',
  'mesoscale': 'Mesoscale & CG',
  'io': 'I/O & compression',
};

const GROUP_LABELS: Record<FlagDetail['group'], string> = {
  package: 'Package',
  accelerator: 'Accelerator',
  option: 'Build option',
  build: 'Build type',
  mpi: 'MPI',
};

/** Merge stored options onto current defaults (schema-evolution safe). */
const reviveOptions = (raw: unknown): CompilerOptions | null => {
  if (typeof raw !== 'object' || raw === null) return null;
  const r = raw as Partial<CompilerOptions>;
  if (typeof r.os !== 'string' && typeof r.presetId !== 'string' && !Array.isArray(r.manualPackages)) {
    return null;
  }
  const knownAccel = ACCELERATORS.some(a => a.id === r.accelerator);
  return {
    ...DEFAULT_COMPILER_OPTIONS,
    ...r,
    os: r.os === 'windows' ? 'windows' : 'linux',
    presetId: typeof r.presetId === 'string' ? r.presetId : DEFAULT_COMPILER_OPTIONS.presetId,
    manualPackages: Array.isArray(r.manualPackages)
      ? r.manualPackages.filter((p): p is string => typeof p === 'string')
      : [],
    accelerator: knownAccel ? r.accelerator! : DEFAULT_COMPILER_OPTIONS.accelerator,
    buildType: r.buildType === 'Debug' || r.buildType === 'RelWithDebInfo' ? r.buildType : 'Release',
    withMpi: r.withMpi !== false,
    jobs: Number.isFinite(r.jobs) && (r.jobs as number) >= 1 ? Math.min(256, Math.round(r.jobs as number)) : 8,
    options: { ...DEFAULT_COMPILER_OPTIONS.options, ...(typeof r.options === 'object' && r.options ? r.options : {}) },
  };
};

const COLLAPSED_CHIPS = 10;

const CompilerHelper: React.FC<{ theme: Theme }> = ({ theme }) => {
  const ct = getThemeTokens(theme);
  const [opts, setOpts] = usePersistentState<CompilerOptions>(
    'm3d.compilerOpts.v1', DEFAULT_COMPILER_OPTIONS, reviveOptions,
  );
  const [copied, setCopied] = useState(false);
  const [showAllFlags, setShowAllFlags] = useState(false);
  const [selectedFlag, setSelectedFlag] = useState<FlagDetail | null>(null);
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < 768 : false);
  const [optionsOpen, setOptionsOpen] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 768 : true);
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  const result = useMemo(() => generateBuildScript(opts), [opts]);

  // Keep the selection valid when the option set changes.
  const selectedDetail = useMemo(
    () => (selectedFlag ? result.flagDetails.find(d => d.flag === selectedFlag.flag) ?? null : null),
    [selectedFlag, result.flagDetails],
  );

  const update = <K extends keyof CompilerOptions>(key: K, value: CompilerOptions[K]) =>
    setOpts(prev => ({ ...prev, [key]: value }));

  const togglePackage = (pkg: string) => {
    setOpts(prev => {
      const set = new Set(prev.manualPackages);
      if (set.has(pkg)) set.delete(pkg); else set.add(pkg);
      return { ...prev, manualPackages: Array.from(set).sort() };
    });
  };

  const copyScript = async () => {
    try {
      await navigator.clipboard.writeText(result.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch { /* blocked */ }
  };

  const selectedPackages = useMemo(() => {
    const preset = PRESETS.find(p => p.id === opts.presetId);
    if (preset) {
      const acc = ACCELERATORS.find(a => a.id === opts.accelerator);
      return Array.from(new Set([...preset.packages, ...(acc?.packages ?? [])])).sort();
    }
    return [...opts.manualPackages].sort();
  }, [opts.presetId, opts.manualPackages, opts.accelerator]);

  const visibleFlags = showAllFlags
    ? result.flagDetails
    : result.flagDetails.slice(0, COLLAPSED_CHIPS);
  const hiddenCount = result.flagDetails.length - visibleFlags.length;

  const chipClass = (d: FlagDetail) =>
    `cursor-pointer rounded px-1.5 py-0.5 text-[9px] transition-colors ${
      selectedDetail?.flag === d.flag
        ? ct.accentSoft
        : ct.chipIdle
    }`;

  return (
    <div className="flex h-full min-h-0">
      {/* Left: options (overlay drawer on mobile) */}
      <div
        className={`overflow-y-auto border-r transition-transform duration-300 ease-in-out ${ct.panel} ${
          isMobile
            ? `fixed inset-y-0 left-0 z-40 w-80 max-w-[92vw] shadow-2xl sm:w-96 ${optionsOpen ? 'translate-x-0' : '-translate-x-full'}`
            : 'w-96 shrink-0'
        }`}
      >
        {isMobile && optionsOpen && (
          <div className={`flex items-center justify-between border-b px-3 py-2 ${ct.divider}`}>
            <span className={`text-xs font-semibold ${ct.headerText}`}>Build options</span>
            <button onClick={() => setOptionsOpen(false)} className={`rounded p-1 ${ct.muted}`} title="Close">
              <X size={14} />
            </button>
          </div>
        )}
        <div className="space-y-6 p-4 antialiased">
          {/* OS target */}
          <section className="space-y-3">
            <h3 className={`flex items-center gap-2 text-sm font-bold tracking-tight ${ct.headerText}`}>
              <Monitor size={13} /> Target OS
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {(['linux', 'windows'] as OsTarget[]).map(os => (
                <button
                  key={os}
                  onClick={() => update('os', os)}
                  className={`py-2 text-xs font-medium rounded-lg border capitalize transition-colors ${
                    opts.os === os ? ct.active : `${ct.muted} ${ct.hoverSurface}`
                  }`}
                >
                  {os === 'linux' ? '🐧 Linux (bash)' : '🪟 Windows (PS)'}
                </button>
              ))}
            </div>
          </section>

          {/* Preset */}
          <section className="space-y-3">
            <h3 className={`flex items-center gap-2 text-sm font-bold tracking-tight ${ct.headerText}`}>
              <Package size={14} /> Package preset
            </h3>
            <div className="grid grid-cols-1 gap-1.5">
              <button
                onClick={() => update('presetId', '')}
                className={`text-left p-2 rounded-lg border text-xs transition-colors ${
                  opts.presetId === '' ? ct.active : `${ct.muted} ${ct.borderStrong} ${ct.hoverSurface}`
                }`}
              >
                <span className="font-medium">Manual selection</span>
                <span className={`block text-[10px] ${ct.muted}`}>Pick packages individually</span>
              </button>
              {PRESETS.map(p => (
                <button
                  key={p.id}
                  onClick={() => update('presetId', p.id)}
                  className={`text-left p-2 rounded-lg border text-xs transition-colors ${
                    opts.presetId === p.id ? ct.active : `${ct.muted} ${ct.borderStrong} ${ct.hoverSurface}`
                  }`}
                >
                  <span className="font-medium">{p.label}</span>
                  <span className={`block text-[10px] ${ct.muted}`}>{p.description}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Accelerator */}
          <section className="space-y-3">
            <h3 className={`flex items-center gap-2 text-sm font-bold tracking-tight ${ct.headerText}`}>
              <Zap size={14} /> Accelerator backend
            </h3>
            <div className="space-y-1">
              {ACCELERATORS.map(acc => (
                <button
                  key={acc.id}
                  onClick={() => update('accelerator', acc.id)}
                  className={`flex w-full items-center justify-between p-2 rounded-lg border text-xs transition-colors ${
                    opts.accelerator === acc.id ? ct.active : `${ct.muted} ${ct.borderStrong} ${ct.hoverSurface}`
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <Cpu size={12} />
                    <span className="font-medium">{acc.label}</span>
                  </span>
                  {acc.vendor !== 'any' && (
                    <span className={`text-[9px] uppercase ${ct.muted}`}>{acc.vendor}</span>
                  )}
                </button>
              ))}
            </div>
          </section>

          {/* Manual packages (when presetId === '') */}
          {opts.presetId === '' && (
            <section className="space-y-3">
              <h3 className={`flex items-center gap-2 text-sm font-bold tracking-tight ${ct.headerText}`}>
                <Package size={14} /> Packages ({opts.manualPackages.length} selected)
              </h3>
              <div className="max-h-64 space-y-3 overflow-y-auto pr-1">
                {PACKAGE_CATEGORIES.map(cat => {
                  const pkgs = LMP_PACKAGES.filter(p => p.category === cat);
                  if (pkgs.length === 0) return null;
                  return (
                    <div key={cat} className="space-y-1">
                      <div className={`text-[10px] font-semibold uppercase ${ct.muted}`}>{CATEGORY_LABELS[cat]}</div>
                      {pkgs.map(pkg => (
                        <label key={pkg.name} className={`flex cursor-pointer items-start gap-2 rounded p-1 ${ct.hoverSurface}`}>
                          <input
                            type="checkbox"
                            checked={opts.manualPackages.includes(pkg.name)}
                            onChange={() => togglePackage(pkg.name)}
                            className={theme === 'dark' ? "mt-0.5 accent-[#7fa66b]" : "mt-0.5 accent-[#4e7a41]"}
                          />
                          <div>
                            <span className={`font-mono text-[11px] ${ct.text}`}>{pkg.name}</span>
                            <span className={`block text-[9px] ${ct.muted}`}>{pkg.description}</span>
                          </div>
                        </label>
                      ))}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Build options */}
          <section className="space-y-3">
            <h3 className={`flex items-center gap-2 text-sm font-bold tracking-tight ${ct.headerText}`}>
              <Settings size={14} /> Build options
            </h3>
            <div className="space-y-3">
              <label className={`flex items-center justify-between text-sm`}>
                <span className={`${ct.muted} font-medium`}>MPI parallel</span>
                <button
                  onClick={() => update('withMpi', !opts.withMpi)}
                  role="switch"
                  aria-checked={opts.withMpi}
                  className={`relative h-5 w-9 rounded-full transition-colors ${opts.withMpi ? ct.toggleOn : ct.toggleOff}`}
                >
                  <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${opts.withMpi ? 'left-4' : 'left-0.5'}`} />
                </button>
              </label>
              <label className={`flex items-center justify-between text-[11px]`}>
                <span className={ct.muted}>Build type</span>
                <select
                  value={opts.buildType}
                  onChange={e => update('buildType', e.target.value as CompilerOptions['buildType'])}
                  className={`rounded border px-1.5 py-0.5 text-[11px] focus:outline-none ${ct.input}`}
                >
                  {['Release', 'Debug', 'RelWithDebInfo'].map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <label className={`flex items-center justify-between text-[11px]`}>
                <span className={ct.muted}>Parallel jobs (-j)</span>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={opts.jobs}
                  onChange={e => update('jobs', parseInt(e.target.value, 10) || 1)}
                  className={`w-16 rounded border px-1.5 py-0.5 text-[11px] focus:outline-none ${ct.input}`}
                />
              </label>
              {BUILD_OPTIONS.map(bo => (
                <label key={bo.key} className={`flex items-center justify-between text-[11px]`} title={bo.help}>
                  <span className={ct.muted}>{bo.label}</span>
                  <select
                    value={opts.options[bo.key] ?? bo.default}
                    onChange={e => setOpts(prev => ({ ...prev, options: { ...prev.options, [bo.key]: e.target.value } }))}
                    className={`rounded border px-1.5 py-0.5 text-[11px] focus:outline-none ${ct.input}`}
                  >
                    {bo.values.map(v => <option key={v} value={v}>{v}</option>)}
                  </select>
                </label>
              ))}
            </div>
          </section>
        </div>
      </div>

      {/* Right: generated script */}
      <div className={`flex min-w-0 flex-1 flex-col ${ct.bg}`}>
        <div className={`flex min-h-10 shrink-0 items-center justify-between gap-x-2 overflow-x-auto border-b px-3 py-1 ${ct.divider}`}>
          <div className="flex items-center gap-2">
            {isMobile && (
              <button
                onClick={() => setOptionsOpen(v => !v)}
                className={`rounded p-1.5 ${ct.muted} ${ct.hoverSurface}`}
                title={optionsOpen ? 'Hide build options' : 'Show build options'}
                aria-label="Toggle build options"
              >
                <SlidersHorizontal size={14} />
              </button>
            )}
            <Terminal size={14} className={ct.muted} />
            <span className={`text-xs font-medium ${ct.headerText}`}>
              {opts.os === 'linux' ? 'build.sh' : 'build.ps1'}
            </span>
            <span className={`text-[10px] ${ct.muted}`}>
              {selectedPackages.length} packages · {result.flags.length} flags
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={copyScript}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
            >
              {copied ? <span className={ct.accentText}>✓ Copied</span> : <><Copy size={13} /> Copy</>}
            </button>
            <button
              onClick={() => downloadTextFile(opts.os === 'linux' ? 'build-lammps.sh' : 'build-lammps.ps1', result.text)}
              className={`flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium transition-colors ${ct.muted} ${ct.hoverSurface}`}
            >
              <Download size={13} /> Download
            </button>
          </div>
        </div>

        {/* Warnings */}
        {result.warnings.length > 0 && (
          <div className={`shrink-0 border-b px-3 py-1.5 text-[11px] ${ct.warn}`}>
            {result.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}

        {/* Flags summary — expandable, click any chip for its description */}
        <div className={`shrink-0 border-b px-3 py-2 ${ct.divider}`}>
          <div className="mb-1 flex items-center justify-between">
            <span className={`text-[10px] font-semibold uppercase tracking-wide ${ct.muted}`}>
              CMake flags · click to inspect
            </span>
            {result.flagDetails.length > COLLAPSED_CHIPS && (
              <button
                onClick={() => setShowAllFlags(v => !v)}
                className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${ct.chip} ${ct.muted} ${ct.hoverSurface}`}
                title={showAllFlags ? 'Collapse the flag list' : 'Show every flag that will be added'}
              >
                {showAllFlags ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                {showAllFlags ? 'Show less' : `+${hiddenCount} more`}
              </button>
            )}
          </div>
          <div className="flex flex-wrap gap-1">
            {visibleFlags.map(d => (
              <button
                key={d.flag}
                onClick={() => setSelectedFlag(prev => (prev?.flag === d.flag ? null : d))}
                className={chipClass(d)}
                title={`${GROUP_LABELS[d.group]} — ${d.description}`}
              >
                {d.flag.replace('-D ', '')}
              </button>
            ))}
          </div>

          {/* Flag detail card */}
          {selectedDetail && (
            <div className={`relative mt-2 rounded-lg border p-3 pr-8 text-xs shadow-lg ${ct.card}`} role="status">
              <button
                onClick={() => setSelectedFlag(null)}
                className={`absolute right-2 top-2 rounded p-0.5 ${ct.muted}`}
                title="Close"
                aria-label="Close flag details"
              >
                <X size={12} />
              </button>
              <code className={`block break-all font-mono text-[11px] font-bold ${ct.accentCode}`}>
                {selectedDetail.flag}
              </code>
              <p className={`mt-1.5 leading-relaxed ${ct.text}`}>{selectedDetail.description}</p>
              <p className={`mt-1.5 flex items-center gap-1 text-[10px] ${ct.muted}`}>
                <Info size={10} />
                <span className={`rounded px-1 py-0.5 uppercase ${ct.chipIdle}`}>{GROUP_LABELS[selectedDetail.group]}</span>
                change under “{selectedDetail.source}”
              </p>
            </div>
          )}
        </div>

        {/* Script text */}
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <pre className={`text-[10px] leading-relaxed font-mono whitespace-pre-wrap sm:text-[11px] ${ct.muted}`}>
            {result.text}
          </pre>
        </div>
      </div>
      {isMobile && optionsOpen && (
        <div className="fixed inset-0 z-30 bg-black/50" onClick={() => setOptionsOpen(false)} />
      )}
    </div>
  );
};

export default CompilerHelper;
