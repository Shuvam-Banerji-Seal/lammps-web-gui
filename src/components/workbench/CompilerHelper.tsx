import React, { useState, useMemo } from 'react';
import {
  LMP_PACKAGES,
  PACKAGE_CATEGORIES,
  ACCELERATORS,
  PRESETS,
  BUILD_OPTIONS,
  generateBuildScript,
  DEFAULT_COMPILER_OPTIONS,
  CompilerOptions,
  OsTarget,
} from '../../lammps/compiler';
import { downloadTextFile } from '../../lammps/exporter';
import { Copy, Download, Terminal, Monitor, Cpu, Package, Zap, Settings } from 'lucide-react';

const CATEGORY_LABELS: Record<string, string> = {
  'accel': 'Accelerators',
  'force-fields': 'Force fields',
  'molecular': 'Molecular',
  'methods': 'Methods & analysis',
  'ml': 'Machine-learning potentials',
  'mesoscale': 'Mesoscale & CG',
  'io': 'I/O & compression',
};

const CompilerHelper: React.FC = () => {
  const [opts, setOpts] = useState<CompilerOptions>(DEFAULT_COMPILER_OPTIONS);
  const [copied, setCopied] = useState(false);

  const result = useMemo(() => generateBuildScript(opts), [opts]);

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

  return (
    <div className="flex h-full min-h-0">
      {/* Left: options */}
      <div className="w-96 shrink-0 overflow-y-auto border-r border-gray-800 bg-[#16191d]">
        <div className="space-y-5 p-4">
          {/* OS target */}
          <section className="space-y-2">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
              <Monitor size={13} /> Target OS
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {(['linux', 'windows'] as OsTarget[]).map(os => (
                <button
                  key={os}
                  onClick={() => update('os', os)}
                  className={`py-2 text-xs font-medium rounded-lg border capitalize transition-colors ${
                    opts.os === os ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'border-gray-700 text-gray-400 hover:bg-gray-800'
                  }`}
                >
                  {os === 'linux' ? '🐧 Linux (bash)' : '🪟 Windows (PS)'}
                </button>
              ))}
            </div>
          </section>

          {/* Preset */}
          <section className="space-y-2">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
              <Package size={13} /> Package preset
            </h3>
            <div className="grid grid-cols-1 gap-1.5">
              <button
                onClick={() => update('presetId', '')}
                className={`text-left p-2 rounded-lg border text-xs transition-colors ${
                  opts.presetId === '' ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'border-gray-700 text-gray-400 hover:bg-gray-800'
                }`}
              >
                <span className="font-medium">Manual selection</span>
                <span className="block text-[10px] text-gray-600">Pick packages individually</span>
              </button>
              {PRESETS.map(p => (
                <button
                  key={p.id}
                  onClick={() => update('presetId', p.id)}
                  className={`text-left p-2 rounded-lg border text-xs transition-colors ${
                    opts.presetId === p.id ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'border-gray-700 text-gray-400 hover:bg-gray-800'
                  }`}
                >
                  <span className="font-medium">{p.label}</span>
                  <span className="block text-[10px] text-gray-600">{p.description}</span>
                </button>
              ))}
            </div>
          </section>

          {/* Accelerator */}
          <section className="space-y-2">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
              <Zap size={13} /> Accelerator backend
            </h3>
            <div className="space-y-1">
              {ACCELERATORS.map(acc => (
                <button
                  key={acc.id}
                  onClick={() => update('accelerator', acc.id)}
                  className={`flex w-full items-center justify-between p-2 rounded-lg border text-xs transition-colors ${
                    opts.accelerator === acc.id ? 'bg-blue-600/20 border-blue-500 text-blue-300' : 'border-gray-700 text-gray-400 hover:bg-gray-800'
                  }`}
                >
                  <span className="flex items-center gap-1.5">
                    <Cpu size={12} />
                    <span className="font-medium">{acc.label}</span>
                  </span>
                  {acc.vendor !== 'any' && (
                    <span className="text-[9px] uppercase text-gray-600">{acc.vendor}</span>
                  )}
                </button>
              ))}
            </div>
          </section>

          {/* Manual packages (when presetId === '') */}
          {opts.presetId === '' && (
            <section className="space-y-2">
              <h3 className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
                <Package size={13} /> Packages ({opts.manualPackages.length} selected)
              </h3>
              <div className="space-y-3 max-h-64 overflow-y-auto pr-1">
                {PACKAGE_CATEGORIES.map(cat => {
                  const pkgs = LMP_PACKAGES.filter(p => p.category === cat);
                  if (pkgs.length === 0) return null;
                  return (
                    <div key={cat} className="space-y-1">
                      <div className="text-[10px] font-semibold uppercase text-gray-600">{CATEGORY_LABELS[cat]}</div>
                      {pkgs.map(pkg => (
                        <label key={pkg.name} className="flex items-start gap-2 cursor-pointer p-1 rounded hover:bg-gray-800/40">
                          <input
                            type="checkbox"
                            checked={opts.manualPackages.includes(pkg.name)}
                            onChange={() => togglePackage(pkg.name)}
                            className="mt-0.5 accent-blue-500"
                          />
                          <div>
                            <span className="text-[11px] font-mono text-gray-300">{pkg.name}</span>
                            <span className="block text-[9px] text-gray-600">{pkg.description}</span>
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
          <section className="space-y-2">
            <h3 className="flex items-center gap-1.5 text-xs font-semibold text-gray-300">
              <Settings size={13} /> Build options
            </h3>
            <div className="space-y-2">
              <label className="flex items-center justify-between text-[11px]">
                <span className="text-gray-400">MPI parallel</span>
                <button
                  onClick={() => update('withMpi', !opts.withMpi)}
                  className={`relative h-5 w-9 rounded-full transition-colors ${opts.withMpi ? 'bg-blue-600' : 'bg-gray-700'}`}
                >
                  <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${opts.withMpi ? 'left-4' : 'left-0.5'}`} />
                </button>
              </label>
              <label className="flex items-center justify-between text-[11px]">
                <span className="text-gray-400">Build type</span>
                <select
                  value={opts.buildType}
                  onChange={e => update('buildType', e.target.value as CompilerOptions['buildType'])}
                  className="rounded border border-gray-700 bg-[#12151a] px-1.5 py-0.5 text-[11px] text-gray-300 focus:outline-none"
                >
                  {['Release', 'Debug', 'RelWithDebInfo'].map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <label className="flex items-center justify-between text-[11px]">
                <span className="text-gray-400">Parallel jobs (-j)</span>
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={opts.jobs}
                  onChange={e => update('jobs', parseInt(e.target.value, 10) || 1)}
                  className="w-16 rounded border border-gray-700 bg-[#12151a] px-1.5 py-0.5 text-[11px] text-gray-300 focus:outline-none"
                />
              </label>
              {BUILD_OPTIONS.map(bo => (
                <label key={bo.key} className="flex items-center justify-between text-[11px]">
                  <span className="text-gray-400" title={bo.help}>{bo.label}</span>
                  <select
                    value={opts.options[bo.key] ?? bo.default}
                    onChange={e => setOpts(prev => ({ ...prev, options: { ...prev.options, [bo.key]: e.target.value } }))}
                    className="rounded border border-gray-700 bg-[#12151a] px-1.5 py-0.5 text-[11px] text-gray-300 focus:outline-none"
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
      <div className="flex min-w-0 flex-1 flex-col bg-[#101214]">
        <div className="flex h-10 shrink-0 items-center justify-between border-b border-gray-800 px-3">
          <div className="flex items-center gap-2">
            <Terminal size={14} className="text-gray-500" />
            <span className="text-xs font-medium text-gray-300">
              {opts.os === 'linux' ? 'build.sh' : 'build.ps1'}
            </span>
            <span className="text-[10px] text-gray-600">
              {selectedPackages.length} packages · {result.flags.length} flags
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              onClick={copyScript}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
            >
              {copied ? <span className="text-emerald-400">✓ Copied</span> : <><Copy size={13} /> Copy</>}
            </button>
            <button
              onClick={() => downloadTextFile(opts.os === 'linux' ? 'build-lammps.sh' : 'build-lammps.ps1', result.text)}
              className="flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition-colors"
            >
              <Download size={13} /> Download
            </button>
          </div>
        </div>

        {/* Warnings */}
        {result.warnings.length > 0 && (
          <div className="shrink-0 border-b border-amber-800/50 bg-amber-950/30 px-3 py-1.5 text-[11px] text-amber-300">
            {result.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}
          </div>
        )}

        {/* Flags summary */}
        <div className="shrink-0 border-b border-gray-800 px-3 py-2">
          <div className="flex flex-wrap gap-1">
            {result.flags.slice(0, 20).map(f => (
              <code key={f} className="rounded bg-gray-800/60 px-1.5 py-0.5 text-[9px] text-blue-300">{f}</code>
            ))}
            {result.flags.length > 20 && (
              <span className="text-[9px] text-gray-600">+{result.flags.length - 20} more…</span>
            )}
          </div>
        </div>

        {/* Script text */}
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <pre className="text-[11px] leading-relaxed font-mono text-gray-300 whitespace-pre-wrap">
            {result.text}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default CompilerHelper;
