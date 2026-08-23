import React, { useState } from 'react';
import ViewerModule from './components/workbench/ViewerModule';
import ScriptBuilder from './components/workbench/ScriptBuilder';
import CompilerHelper from './components/workbench/CompilerHelper';
import { FlaskConical, FileCode2, Hammer, Atom as AtomIcon } from 'lucide-react';

type Module = 'builder' | 'compiler' | 'viewer';

const MODULES: { id: Module; label: string; icon: React.ReactNode; hint: string }[] = [
  { id: 'builder', label: 'Script Builder', icon: <FileCode2 size={15} />, hint: 'Build LAMMPS input scripts visually' },
  { id: 'compiler', label: 'Compiler Helper', icon: <Hammer size={15} />, hint: 'Generate clone + CMake build commands' },
  { id: 'viewer', label: 'Structure Viewer', icon: <AtomIcon size={15} />, hint: '3D visualization of LAMMPS/XYZ/PDB/CIF files' },
];

/**
 * Molecule3D Workbench — three modules:
 *  1. Script Builder (primary): visual LAMMPS input construction + flowchart
 *  2. Compiler Helper: package/accelerator selection → build commands
 *  3. Structure Viewer: the original 3D visualizer
 */
const App: React.FC = () => {
  const [module, setModule] = useState<Module>('builder');

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#101214] font-sans text-gray-100">
      {/* Top-level module switcher */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-gray-800 bg-[#16191d] px-4">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <FlaskConical size={18} className="text-blue-400" />
            <span className="text-sm font-bold tracking-tight">LAMMPS Workbench</span>
            <span className="rounded bg-gray-800 px-1.5 py-0.5 text-[9px] text-gray-400">
              by Molecule3D
            </span>
          </div>
        </div>
        <nav className="flex items-center gap-1" aria-label="Modules">
          {MODULES.map(m => (
            <button
              key={m.id}
              onClick={() => setModule(m.id)}
              title={m.hint}
              className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                module === m.id
                  ? 'bg-blue-600/20 text-blue-300 border border-blue-500/50'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/60 border border-transparent'
              }`}
            >
              {m.icon}
              {m.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="min-h-0 flex-1">
        {module === 'builder' && <ScriptBuilder onOpenViewer={() => setModule('viewer')} />}
        {module === 'compiler' && <CompilerHelper />}
        {module === 'viewer' && <ViewerModule />}
      </main>
    </div>
  );
};

export default App;
