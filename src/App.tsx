import React, { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react';
import ViewerModule from './components/workbench/ViewerModule';
import ScriptBuilder from './components/workbench/ScriptBuilder';
import CompilerHelper from './components/workbench/CompilerHelper';
import { FlaskConical, FileCode2, Hammer, Atom as AtomIcon, NotebookPen, Info, X, ExternalLink } from 'lucide-react';
import { getThemeTokens, initialTheme, isDarkTheme, nextTheme, THEMES, THEME_STORAGE_KEY, themeColor, Theme } from './theme';
import { browserStore } from './services/persistence';
import type { BuilderIncoming } from './lammps/notebookBridge';

// The notebook carries the MD engine; load it only when it is opened.
const Notebook = lazy(() => import('./components/workbench/Notebook'));

type Module = 'builder' | 'compiler' | 'viewer' | 'notebook';

const MODULES: { id: Module; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: 'builder', label: 'Script Builder', icon: <FileCode2 size={15} />, hint: 'Build LAMMPS input scripts visually' },
  { id: 'compiler', label: 'Compiler Helper', icon: <Hammer size={15} />, hint: 'Generate clone + CMake build commands' },
  { id: 'viewer', label: 'Structure Viewer', icon: <AtomIcon size={15} />, hint: '3D visualization of LAMMPS/XYZ/PDB/CIF files' },
  { id: 'notebook', label: 'MD Notebook', icon: <NotebookPen size={15} />, hint: 'Run small LAMMPS-style simulations in the browser (WebGPU or CPU)' },
];

const MODULE_KEY = 'm3d.activeModule';
const loadLastModule = (): Module => {
  try {
    const v = localStorage.getItem(MODULE_KEY);
    if (v === 'builder' || v === 'compiler' || v === 'viewer' || v === 'notebook') return v;
  } catch { /* storage unavailable */ }
  return 'builder';
};

/**
 * Molecule3D Workbench — four modules:
 *  1. Script Builder (primary): visual LAMMPS input construction + flowchart
 *  2. Compiler Helper: package/accelerator selection → build commands
 *  3. Structure Viewer: the original 3D visualizer
 *  4. MD Notebook: runs a documented LAMMPS input subset in the browser
 *
 * Global light/dark theme (warm coffee-green dark) is owned here and passed
 * to every module so switching modules never loses your look. The active
 * module and theme persist across reloads.
 */
const REPO_URL = 'https://github.com/Shuvam-Banerji-Seal/lammps-web-gui';

const App: React.FC = () => {
  const [module, setModule] = useState<Module>(loadLastModule);
  /** A script the Script Builder sent to the notebook, until the notebook places it. */
  const [notebookInbox, setNotebookInbox] = useState<string | null>(null);
  const [builderInbox, setBuilderInbox] = useState<BuilderIncoming | null>(null);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [aboutOpen, setAboutOpen] = useState(false);
  const aboutDialogRef = useRef<HTMLDivElement>(null);
  const aboutCloseRef = useRef<HTMLButtonElement>(null);
  /** Element that had focus when the dialog opened — focus returns there. */
  const aboutReturnFocus = useRef<HTMLElement | null>(null);

  const openAbout = useCallback(() => {
    aboutReturnFocus.current = document.activeElement as HTMLElement | null;
    setAboutOpen(true);
  }, []);
  const closeAbout = useCallback(() => setAboutOpen(false), []);

  // ARIA APG modal dialog: focus moves in on open, Escape closes, Tab stays
  // inside, and focus returns to the trigger on close (WCAG 2.4.3).
  useEffect(() => {
    if (!aboutOpen) return;
    aboutCloseRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); setAboutOpen(false); return; }
      if (e.key !== 'Tab' || !aboutDialogRef.current) return;
      const focusable = aboutDialogRef.current.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      aboutReturnFocus.current?.focus?.();
    };
  }, [aboutOpen]);
  const ct = getThemeTokens(theme);

  const switchModule = (m: Module) => {
    setModule(m);
    try { localStorage.setItem(MODULE_KEY, m); } catch { /* non-fatal */ }
  };

  const chooseTheme = useCallback((next: Theme) => {
    setTheme(next);
    try { localStorage.setItem(THEME_STORAGE_KEY, next); } catch { /* non-fatal */ }
  }, []);

  /** Next theme in THEMES order (the ViewerModule shortcut and sidebar item). */
  const cycleTheme = useCallback(() => chooseTheme(nextTheme(theme)), [theme, chooseTheme]);

  // Keep the browser UI colour (mobile address bar, etc.) in step with the page.
  useEffect(() => {
    try {
      document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.setAttribute('content', themeColor(theme));
      // the page behind the app (seen on overscroll / mobile bounce) follows the theme too
      document.body.style.backgroundColor = themeColor(theme);
      document.documentElement.style.colorScheme = isDarkTheme(theme) ? 'dark' : 'light';
    } catch { /* non-fatal */ }
  }, [theme]);

  return (
    <div className={`flex h-dvh w-full flex-col overflow-hidden font-sans ${ct.bg} ${ct.text}`}>
      {/* Top-level module switcher */}
      <header className={`flex h-12 shrink-0 items-center justify-between border-b px-4 ${ct.panel}`}>
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex min-w-0 items-center gap-1.5">
            <FlaskConical size={18} aria-hidden="true" className={`shrink-0 ${ct.accentText}`} />
            <span className="hidden truncate text-sm font-bold tracking-tight sm:inline">LAMMPS Workbench</span>
            {/*
              Author credit. LICENSE §3.2 requires a legible, permanently
              reachable attribution in any deployed build — do not remove,
              shrink or hide this, and keep the About dialog reachable.
            */}
            <button
              onClick={openAbout}
              title="About, credits and licence"
              aria-label="About, credits and licence"
              className={`flex min-h-6 min-w-6 shrink-0 items-center justify-center gap-1 rounded-md px-1.5 py-1 text-[11px] transition-colors ${ct.muted} ${ct.hoverSurface}`}
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
                aria-label={m.label}
                aria-current={module === m.id ? 'page' : undefined}
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
          <select
            aria-label="Color theme"
            title="Color theme"
            value={theme}
            onChange={e => chooseTheme(e.target.value as Theme)}
            className={`max-w-[8.5rem] cursor-pointer rounded-lg px-2 py-1.5 text-xs transition-colors sm:max-w-none ${ct.button} ${ct.text}`}
          >
            {THEMES.map(t => (
              <option key={t.id} value={t.id}>{t.label}</option>
            ))}
          </select>
        </div>
      </header>

      <main className="min-h-0 flex-1">
        {module === 'builder' && (
          <ScriptBuilder theme={theme} onOpenViewer={() => switchModule('viewer')}
            onRunInNotebook={(script) => { setNotebookInbox(script); switchModule('notebook'); }}
            incoming={builderInbox} onIncomingTaken={() => setBuilderInbox(null)} />
        )}
        {module === 'compiler' && <CompilerHelper theme={theme} />}
        {module === 'viewer' && <ViewerModule theme={theme} onToggleTheme={cycleTheme} />}
        {module === 'notebook' && (
          <Suspense fallback={<div className={`p-6 text-sm ${ct.muted}`}>Loading the notebook…</div>}>
            <Notebook theme={theme} incoming={notebookInbox} onIncomingTaken={() => setNotebookInbox(null)}
              onOpenInBuilder={(script) => { setBuilderInbox({ text: script, name: 'From MD Notebook' }); switchModule('builder'); }} />
          </Suspense>
        )}
      </main>

      {aboutOpen && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4"
          onClick={closeAbout}
          role="dialog"
          aria-modal="true"
          aria-label="About LAMMPS Workbench"
        >
          <div
            ref={aboutDialogRef}
            className={`max-h-[85vh] w-full max-w-lg overflow-y-auto rounded-2xl border p-6 shadow-2xl ${ct.card}`}
            onClick={e => e.stopPropagation()}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div className="flex items-center gap-2.5">
                <FlaskConical size={22} aria-hidden="true" className={`shrink-0 ${ct.accentText}`} />
                <div>
                  <h2 className="text-base font-bold tracking-tight">LAMMPS Workbench · Molecule3D</h2>
                  <p className={`text-[11px] ${ct.muted}`}>
                    Created by <span className="font-semibold">Shuvam Banerji Seal</span>
                  </p>
                </div>
              </div>
              <button
                ref={aboutCloseRef}
                onClick={closeAbout}
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
