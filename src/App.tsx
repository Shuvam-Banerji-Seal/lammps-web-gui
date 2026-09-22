import React, { useState } from 'react';
import ViewerModule from './components/workbench/ViewerModule';
import ScriptBuilder from './components/workbench/ScriptBuilder';
import CompilerHelper from './components/workbench/CompilerHelper';
import { FlaskConical, FileCode2, Hammer, Atom as AtomIcon, Sun, Moon, Info, X, ExternalLink } from 'lucide-react';
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
const REPO_URL = 'https://github.com/Shuvam-Banerji-Seal/lammps-web-gui';

const App: React.FC = () => {
  const [module, setModule] = useState<Module>(loadLastModule);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [aboutOpen, setAboutOpen] = useState(false);
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
    <div className={`flex h-dvh w-full flex-col overflow-hidden font-sans ${ct.bg} ${ct.text}`}>
      {/* Top-level module switcher */}
      <header className={`flex h-12 shrink-0 items-center justify-between border-b px-4 ${ct.panel}`}>
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex min-w-0 items-center gap-1.5">
            <FlaskConical size={18} className="shrink-0 text-[#9dc487]" />
            <span className="hidden truncate text-sm font-bold tracking-tight sm:inline">LAMMPS Workbench</span>
            {/*
              Author credit. LICENSE §3.2 requires a legible, permanently
              reachable attribution in any deployed build — do not remove,
              shrink or hide this, and keep the About dialog reachable.
            */}
            <button
              onClick={() => setAboutOpen(true)}
              title="About, credits and licence"
              className={`flex shrink-0 items-center gap-1 rounded-md px-1.5 py-1 text-[11px] transition-colors ${ct.muted} ${ct.hoverSurface}`}
            >
              <Info size={12} />
              <span className="hidden whitespace-nowrap md:inline">by Shuvam Banerji Seal</span>
            </button>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <nav className="flex items-center gap-1" aria-label="Modules">
            {MODULES.map(m => (
              <button
                key={m.id}
                onClick={() => switchModule(m.id)}
                title={m.hint}
                className={`flex items-center gap-1 rounded-lg px-2 py-1.5 text-xs font-medium transition-colors sm:gap-1.5 sm:px-3 ${
                  module === m.id
                    ? ct.active
                    : `${ct.muted} ${ct.hoverSurface} border border-transparent`
                }`}
              >
                {m.icon}
                <span className="hidden sm:inline">{m.label}</span>
              </button>
            ))}
          </nav>
          <button
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label="Toggle color theme"
            className={`rounded-lg p-1.5 transition-colors sm:p-2 ${ct.button}`}
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

      {aboutOpen && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4"
          onClick={() => setAboutOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label="About LAMMPS Workbench"
        >
          <div
            className={`max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl border p-6 shadow-2xl ${ct.card}`}
            onClick={e => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <FlaskConical size={22} className="shrink-0 text-[#9dc487]" />
                <div>
                  <h2 className="text-base font-bold tracking-tight">LAMMPS Workbench · Molecule3D</h2>
                  <p className={`text-[11px] ${ct.muted}`}>
                    Created by <span className="font-semibold">Shuvam Banerji Seal</span>
                  </p>
                </div>
              </div>
              <button
                onClick={() => setAboutOpen(false)}
                className={`shrink-0 rounded-lg p-1.5 ${ct.button}`}
                aria-label="Close about dialog"
              >
                <X size={16} />
              </button>
            </div>

            <p className={`text-xs leading-relaxed ${ct.muted}`}>
              A browser-based LAMMPS workbench: a branching visual script builder that
              validates its output against the official command-ordering rules, a CMake
              compiler helper, and a GPU-accelerated structure and trajectory viewer.
              Everything runs locally — no file you open ever leaves your browser.
            </p>

            <div className={`mt-4 rounded-xl border p-3 text-xs leading-relaxed ${ct.stat}`}>
              <p className="font-semibold">Free for education and non-commercial research.</p>
              <p className={`mt-1 ${ct.muted}`}>
                Attribution to Shuvam Banerji Seal is required, including a citation in
                academic work. Commercial use needs a separate written licence and carries
                a revenue share.
              </p>
            </div>

            <div className="mt-4 grid gap-1.5">
              {[
                { label: 'Source repository', href: REPO_URL },
                { label: 'Licence (educational / non-commercial)', href: `${REPO_URL}/blob/main/LICENSE` },
                { label: 'Commercial licensing', href: `${REPO_URL}/blob/main/COMMERCIAL.md` },
                { label: 'How to cite', href: `${REPO_URL}/blob/main/CITATION.cff` },
                { label: 'LAMMPS documentation (docs.lammps.org)', href: 'https://docs.lammps.org' },
              ].map(l => (
                <a
                  key={l.href}
                  href={l.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`flex items-center justify-between rounded-lg px-3 py-2 text-xs font-medium transition-colors ${ct.button}`}
                >
                  {l.label}
                  <ExternalLink size={12} className="shrink-0 opacity-60" />
                </a>
              ))}
            </div>

            <p className={`mt-4 text-[10px] leading-relaxed ${ct.muted}`}>
              LAMMPS is a separate project of Sandia National Laboratories and the LAMMPS
              Developers. This tool is independent and is not affiliated with or endorsed
              by it. Generated scripts and build commands are aids — always verify them
              against the official documentation before running production work.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
