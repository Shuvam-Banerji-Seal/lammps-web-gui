import React, { useState } from 'react';
import ViewerModule from './components/workbench/ViewerModule';
import ScriptBuilder from './components/workbench/ScriptBuilder';
import CompilerHelper from './components/workbench/CompilerHelper';
import { FlaskConical, FileCode2, Hammer, Atom as AtomIcon, Sun, Moon } from 'lucide-react';
import { getThemeTokens, initialTheme, Theme, THEME_STORAGE_KEY } from './theme';
import { browserStore } from './services/persistence';

type Module = 'builder' | 'compiler' | 'viewer';

const MODULES: { id: Module; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: 'builder', label: 'Script Builder', icon: <FileCode2 size={15} />, hint: 'Build LAMMPS input scripts visually' },
  { id: 'compiler', label: 'Compiler Helper', icon: <Hammer size={15} />, hint: 'Generate clone + CMake build commands' },
  { id: 'viewer', label: 'Structure Viewer', icon: <AtomIcon size={15} />, hint: '3D visualization of LAMMPS/XYZ/PDB/CIF files' },
];

const MODULE_KEY = 'm3d.activeModule';
const loadLastModule = (): Module => {
  try {
    const v = localStorage.getItem(MODULE_KEY);
    if (v === 'builder' || v === 'compiler' || v === 'viewer') return v;
  } catch { /* storage unavailable */ }
  return 'builder';
};

/**
 * Molecule3D Workbench — three modules:
 *  1. Script Builder (primary): visual LAMMPS input construction + flowchart
 *  2. Compiler Helper: package/accelerator selection → build commands
 *  3. Structure Viewer: the original 3D visualizer
 *
 * Global light/dark theme (warm coffee-green dark) is owned here and passed
 * to every module so switching modules never loses your look. The active
 * module and theme persist across reloads.
 */
const App: React.FC = () => {
  const [module, setModule] = useState<Module>(loadLastModule);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const ct = getThemeTokens(theme);

  const switchModule = (m: Module) => {
    setModule(m);
    try { localStorage.setItem(MODULE_KEY, m); } catch { /* non-fatal */ }
  };

  const toggleTheme = () => {
    setTheme(t => {
      const next: Theme = t === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch { /* non-fatal */ }
      return next;
    });
  };

  return (
    <div className={`flex h-screen w-screen flex-col overflow-hidden font-sans ${ct.bg} ${ct.text}`}>
      {/* Top-level module switcher */}
      <header className={`flex h-12 shrink-0 items-center justify-between border-b px-4 ${ct.panel}`}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className="text-[#9dc487]" />
            <span className="text-sm font-bold tracking-tight">LAMMPS Workbench</span>
            <span className={`rounded px-1.5 py-0.5 text-[9px] ${ct.chip} ${ct.muted}`}>
              by Shuvam Banerji Seal
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <nav className="flex items-center gap-1" aria-label="Modules">
            {MODULES.map(m => (
              <button
                key={m.id}
                onClick={() => switchModule(m.id)}
                title={m.hint}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  module === m.id
                    ? ct.active
                    : `${ct.muted} ${ct.hoverSurface} border border-transparent`
                }`}
              >
                {m.icon}
                {m.label}
              </button>
            ))}
          </nav>
          <button
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle color theme"
            className={`rounded-lg p-2 transition-colors ${ct.button}`}
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </header>

      <main className="min-h-0 flex-1">
        {module === 'builder' && <ScriptBuilder theme={theme} onOpenViewer={() => switchModule('viewer')} />}
        {module === 'compiler' && <CompilerHelper theme={theme} />}
        {module === 'viewer' && <ViewerModule theme={theme} onToggleTheme={toggleTheme} />}
      </main>
    </div>
  );
};

export default App;
