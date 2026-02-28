import React, { useState, useEffect, useCallback } from 'react';
import MoleculeCanvas from './components/MoleculeCanvas';
import { parseFile, detectFileFormat } from './services/fileParser';
import { MoleculeData, VisualizationConfig, VisualizationMode, FileFormat } from './types';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR, ELEMENT_DATA } from './constants';
import { Upload, RotateCw, Play, Pause, AlertCircle, Info, Settings, Eye, EyeOff, Palette, Box, Sun, Moon, Menu, X, Camera, Atom } from 'lucide-react';

type Theme = 'light' | 'dark';

const App: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [moleculeData, setMoleculeData] = useState<MoleculeData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [theme, setTheme] = useState<Theme>('dark');
  const [fileFormat, setFileFormat] = useState<FileFormat>('lammps');
  
  const [vizConfig, setVizConfig] = useState<VisualizationConfig>({
    atomScale: 1.0,
    bondScale: 1.0,
    materialType: 'realistic',
    backgroundColor: '#151515',
    showBonds: true,
    customColors: {},
    visualizationMode: 'ball-and-stick',
  });

  const [activeTab, setActiveTab] = useState<'data' | 'settings'>('data');

  // Responsive detection
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) setIsSidebarOpen(false);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Load initial data
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}c60.data`);
        const text = await response.text();
        handleVisualize(text, 'lammps');
      } catch {
        setError("Failed to fetch initial data.");
      }
    };
    fetchInitialData();
  }, []);

  const handleVisualize = useCallback((text: string, format?: FileFormat) => {
    try {
      setError(null);
      const data = parseFile(text, format);
      if (data.atoms.length === 0) {
        throw new Error("No atoms found in data. Check the format.");
      }
      
      const newCustomColors: Record<number, string> = {};
      Object.values(data.atomTypes).forEach(info => {
        let defaultColor = DEFAULT_ATOM_COLOR;
        const elementMeta = ELEMENT_DATA.find(e => e.symbol === info.element);
        if (elementMeta && ATOM_COLORS[elementMeta.number]) {
          defaultColor = ATOM_COLORS[elementMeta.number];
        } else if (ATOM_COLORS[info.id]) {
          defaultColor = ATOM_COLORS[info.id];
        }
        newCustomColors[info.id] = defaultColor;
      });

      setVizConfig(prev => ({ ...prev, customColors: newCustomColors }));
      setMoleculeData(data);
      setInputText(text);
      if (isMobile) setIsSidebarOpen(false);
    } catch (e: any) {
      setError(e.message || "Failed to parse data file.");
      setMoleculeData(null);
    }
  }, [isMobile]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const detectedFormat = detectFileFormat(file.name);
      setFileFormat(detectedFormat);
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setInputText(content);
        handleVisualize(content, detectedFormat);
      };
      reader.readAsText(file);
    }
  };

  const handleLoadExample = async () => {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}c60.data`);
      const text = await response.text();
      setFileFormat('lammps');
      handleVisualize(text, 'lammps');
    } catch {
      setError("Failed to fetch example data.");
    }
  };

  const updateConfig = (key: keyof VisualizationConfig, value: any) => {
    setVizConfig(prev => ({ ...prev, [key]: value }));
  };

  const updateCustomColor = (typeId: number, color: string) => {
    setVizConfig(prev => ({
      ...prev,
      customColors: { ...prev.customColors, [typeId]: color }
    }));
  };

  const toggleTheme = () => {
    setTheme(currentTheme => {
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      updateConfig('backgroundColor', newTheme === 'dark' ? '#151515' : '#ffffff');
      return newTheme;
    });
  };

  const themeClasses = {
    dark: {
      bg: 'bg-black',
      text: 'text-gray-100',
      sidebar: 'bg-gray-950 border-gray-800',
      tabActive: 'text-blue-400 border-b-2 border-blue-400 bg-gray-900/50',
      tabInactive: 'text-gray-400 hover:text-white hover:bg-gray-900',
      card: 'bg-gray-900 border-gray-800',
      input: 'bg-gray-900 border-gray-700 text-gray-300 focus:ring-blue-500',
      button: 'bg-gray-800 hover:bg-gray-700',
      textMuted: 'text-gray-400',
      textHeader: 'text-gray-300',
      statsCard: 'bg-gray-800',
    },
    light: {
      bg: 'bg-gray-100',
      text: 'text-gray-800',
      sidebar: 'bg-white border-gray-200',
      tabActive: 'text-blue-600 border-b-2 border-blue-600 bg-gray-100',
      tabInactive: 'text-gray-500 hover:text-black hover:bg-gray-200',
      card: 'bg-white border-gray-200',
      input: 'bg-white border-gray-300 text-gray-800 focus:ring-blue-500',
      button: 'bg-gray-200 hover:bg-gray-300',
      textMuted: 'text-gray-500',
      textHeader: 'text-gray-700',
      statsCard: 'bg-gray-200',
    }
  };

  const ct = themeClasses[theme];

  const vizModes: { value: VisualizationMode; label: string; desc: string }[] = [
    { value: 'ball-and-stick', label: 'Ball & Stick', desc: 'Classic molecular view' },
    { value: 'space-fill', label: 'Space Fill', desc: 'Van der Waals radii' },
    { value: 'wireframe', label: 'Wireframe', desc: 'Minimal bond view' },
  ];

  return (
    <div className={`flex h-screen w-screen font-sans ${ct.bg} ${ct.text}`}>
      
      {/* Mobile overlay backdrop */}
      {isMobile && isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-20"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`
        flex flex-col border-r transition-all duration-300 ease-in-out z-30
        ${isMobile ? 'fixed inset-y-0 left-0 w-80 max-w-[85vw]' : 'w-96'}
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        ${ct.sidebar}
      `}>
        
        {/* Header */}
        <div className={`p-4 border-b flex items-center justify-between ${ct.sidebar}`}>
          <div className="flex items-center gap-2">
            <img src={`${import.meta.env.BASE_URL}logo.svg`} alt="Molecule3D" className="w-8 h-8" />
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Molecule3D</h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-gray-500">v2.0</span>
            {isMobile && (
              <button onClick={() => setIsSidebarOpen(false)} className={`p-1 rounded ${ct.button}`}>
                <X size={18} />
              </button>
            )}
          </div>
        </div>

        {/* Tab Switcher */}
        <div className={`flex border-b ${ct.sidebar}`}>
          <button 
            onClick={() => setActiveTab('data')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'data' ? ct.tabActive : ct.tabInactive}`}
          >
            Data Source
          </button>
          <button 
            onClick={() => setActiveTab('settings')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'settings' ? ct.tabActive : ct.tabInactive}`}
          >
            Appearance
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          
          {activeTab === 'data' ? (
            <>
              {/* Format Guide */}
              <div className={`rounded-lg p-3 text-xs border ${ct.card} ${ct.textMuted}`}>
                <div className="flex items-center gap-2 mb-2 text-blue-500 font-semibold">
                  <Info size={14} /> 
                  <span>Supported Formats</span>
                </div>
                <div className="space-y-1">
                  <div><span className="font-medium text-blue-400">.data/.lmp</span> — LAMMPS data files</div>
                  <div><span className="font-medium text-green-400">.xyz</span> — XYZ molecular files</div>
                  <div><span className="font-medium text-purple-400">.pdb</span> — Protein Data Bank files</div>
                </div>
              </div>

              {/* File Operations */}
              <div className="space-y-2">
                <label className={`text-xs font-medium ${ct.textHeader}`}>File Operations</label>
                <div className="flex gap-2">
                  <button 
                    onClick={handleLoadExample}
                    className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded text-xs font-medium transition-colors ${ct.button}`}
                  >
                    <RotateCw size={12} /> Load C60
                  </button>
                  <label className="flex-1 cursor-pointer flex items-center justify-center gap-1.5 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs font-medium transition-colors">
                    <Upload size={12} /> Upload File
                    <input type="file" onChange={handleFileUpload} className="hidden" accept=".data,.txt,.lmp,.lammps,.xyz,.pdb,.ent" />
                  </label>
                </div>
                
                {/* Format selector */}
                <div className="flex gap-1">
                  {(['lammps', 'xyz', 'pdb'] as FileFormat[]).map(fmt => (
                    <button
                      key={fmt}
                      onClick={() => setFileFormat(fmt)}
                      className={`flex-1 py-1.5 text-[10px] uppercase font-bold rounded transition-all ${
                        fileFormat === fmt
                          ? 'bg-blue-600 text-white'
                          : `${ct.button} ${ct.textMuted}`
                      }`}
                    >
                      {fmt}
                    </button>
                  ))}
                </div>

                <textarea 
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="# Paste molecular data here..."
                  className={`w-full h-48 border rounded p-3 text-xs font-mono resize-y ${ct.input}`}
                  spellCheck={false}
                />
                <button 
                  onClick={() => handleVisualize(inputText, fileFormat)}
                  className="w-full py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-bold text-white transition-colors shadow-lg shadow-green-900/20"
                >
                  Visualize Data
                </button>
              </div>

              {/* Statistics */}
              {moleculeData && (
                <div className="space-y-2 pt-3 border-t border-gray-800">
                  <h3 className={`text-xs font-semibold ${ct.textHeader}`}>Statistics</h3>
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className={`p-2 rounded text-center ${ct.statsCard}`}>
                      <div className={`text-[10px] ${ct.textMuted}`}>Atoms</div>
                      <div className="font-mono font-bold">{moleculeData.atoms.length}</div>
                    </div>
                    <div className={`p-2 rounded text-center ${ct.statsCard}`}>
                      <div className={`text-[10px] ${ct.textMuted}`}>Bonds</div>
                      <div className="font-mono font-bold">{moleculeData.bonds.length}</div>
                    </div>
                    <div className={`p-2 rounded text-center ${ct.statsCard}`}>
                      <div className={`text-[10px] ${ct.textMuted}`}>Types</div>
                      <div className="font-mono font-bold">{Object.keys(moleculeData.atomTypes).length}</div>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="p-3 bg-red-900/30 border border-red-800 rounded text-red-200 text-xs flex gap-2 items-start">
                  <AlertCircle size={14} className="shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-6">
              {/* Theme Toggle */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Palette size={14} /> Theme
                </div>
                <button
                  onClick={toggleTheme}
                  className={`w-full flex items-center justify-center gap-2 py-2 rounded text-xs font-medium transition-colors ${ct.button}`}
                >
                  {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                  Switch to {theme === 'dark' ? 'Light' : 'Dark'} Mode
                </button>
              </div>

              {/* Visualization Mode */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Atom size={14} /> Visualization Mode
                </div>
                <div className="space-y-1.5">
                  {vizModes.map(mode => (
                    <button
                      key={mode.value}
                      onClick={() => updateConfig('visualizationMode', mode.value)}
                      className={`w-full text-left p-2 rounded border text-xs transition-all ${
                        vizConfig.visualizationMode === mode.value
                          ? 'bg-blue-600/20 border-blue-500 text-blue-300'
                          : `${ct.button} border-transparent`
                      }`}
                    >
                      <div className="font-medium">{mode.label}</div>
                      <div className={`text-[10px] ${ct.textMuted}`}>{mode.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Render Style */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Settings size={14} /> Material
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {['realistic', 'plastic', 'toon'].map((type) => (
                    <button
                      key={type}
                      onClick={() => updateConfig('materialType', type)}
                      className={`py-2 px-1 text-xs capitalize rounded border transition-all ${
                        vizConfig.materialType === type 
                          ? 'bg-blue-600 border-blue-500 text-white' 
                          : `${ct.button} border-gray-700`
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              {/* Scaling */}
              <div className="space-y-3">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Box size={14} /> Geometry Scale
                </div>
                
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-gray-400">
                    <span>Atom Size</span>
                    <span>{(vizConfig.atomScale * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0.1" max="3.0" step="0.1"
                    value={vizConfig.atomScale}
                    onChange={(e) => updateConfig('atomScale', parseFloat(e.target.value))}
                    className={`w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 ${ct.button}`}
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-[10px] text-gray-400">
                    <span>Bond Thickness</span>
                    <span>{(vizConfig.bondScale * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range" min="0.1" max="3.0" step="0.1"
                    value={vizConfig.bondScale}
                    onChange={(e) => updateConfig('bondScale', parseFloat(e.target.value))}
                    className={`w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 ${ct.button}`}
                  />
                </div>
              </div>
              
              {/* Atom Colors */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Palette size={14} /> Atom Colors
                </div>
                
                {moleculeData && Object.values(moleculeData.atomTypes).length > 0 ? (
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {Object.values(moleculeData.atomTypes).map((typeInfo) => (
                      <div key={typeInfo.id} className={`flex items-center justify-between p-2 rounded border ${ct.statsCard} border-gray-700`}>
                        <div className="flex items-center gap-2">
                          <div className="relative w-7 h-7 rounded-full overflow-hidden border border-gray-600">
                            <input 
                              type="color" 
                              value={vizConfig.customColors[typeInfo.id] || DEFAULT_ATOM_COLOR}
                              onChange={(e) => updateCustomColor(typeInfo.id, e.target.value)}
                              className="absolute -top-2 -left-2 w-12 h-12 cursor-pointer p-0 border-0"
                            />
                          </div>
                          <div>
                            <div className={`text-[11px] font-bold ${ct.textHeader}`}>{typeInfo.label}</div>
                            <div className={`text-[9px] ${ct.textMuted}`}>
                              {typeInfo.element !== 'X' ? `${typeInfo.element} · Mass: ${typeInfo.mass.toFixed(2)}` : `ID: ${typeInfo.id}`}
                            </div>
                          </div>
                        </div>
                        <div className="bg-gray-700 px-2 py-0.5 rounded text-[10px] font-mono text-gray-300">
                          {typeInfo.count}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={`text-xs italic ${ct.textMuted}`}>Load data to customize atom colors.</div>
                )}
              </div>

              {/* Visibility */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Eye size={14} /> Visibility
                </div>
                <button 
                  onClick={() => updateConfig('showBonds', !vizConfig.showBonds)}
                  className={`flex items-center justify-between w-full p-2 rounded text-xs transition-colors ${ct.button}`}
                >
                  <span>Show Bonds</span>
                  {vizConfig.showBonds ? <Eye size={14} className="text-green-400"/> : <EyeOff size={14} className="text-gray-500"/>}
                </button>
              </div>

              {/* Background */}
              <div className="space-y-2">
                <div className={`flex items-center gap-2 text-xs font-semibold border-b pb-2 ${ct.textHeader} ${ct.sidebar}`}>
                  <Box size={14} /> Background
                </div>
                <div className="flex gap-2 flex-wrap">
                  {['#151515', '#000000', '#1a202c', '#ffffff', '#e2e8f0'].map((color) => (
                    <button
                      key={color}
                      onClick={() => updateConfig('backgroundColor', color)}
                      className={`w-7 h-7 rounded-full border-2 transition-transform hover:scale-110 ${
                        vizConfig.backgroundColor === color ? 'border-blue-500 scale-110' : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color }}
                      title={color}
                    />
                  ))}
                  <div className="relative w-7 h-7 rounded-full overflow-hidden border-2 border-gray-600">
                    <input 
                      type="color" 
                      value={vizConfig.backgroundColor}
                      onChange={(e) => updateConfig('backgroundColor', e.target.value)}
                      className="absolute -top-2 -left-2 w-12 h-12 cursor-pointer p-0 border-0"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className={`p-3 border-t text-[10px] flex flex-col items-center text-center ${ct.sidebar} ${ct.textMuted}`}>
          <span>Created by <span className="text-gray-300 font-medium">Shuvam Banerji Seal</span></span>
          <a href="https://shuvam-banerji-seal.github.io/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-400 mt-0.5">
            shuvam-banerji-seal.github.io
          </a>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className={`flex-1 relative ${ct.bg}`}>
        {/* Top controls bar */}
        <div className="absolute top-3 left-3 right-3 z-10 flex items-center justify-between pointer-events-none">
          <div className="pointer-events-auto">
            {!isSidebarOpen && (
              <button 
                onClick={() => setIsSidebarOpen(true)}
                className={`p-2 rounded-lg shadow-lg transition-colors ${
                  theme === 'dark' ? 'bg-gray-800/90 hover:bg-gray-700 text-white' : 'bg-white/90 hover:bg-gray-100 text-gray-800'
                } backdrop-blur`}
                title="Open sidebar"
              >
                <Menu size={20} />
              </button>
            )}
          </div>
          <div className="pointer-events-auto flex gap-2">
            {isSidebarOpen && !isMobile && (
              <button 
                onClick={() => setIsSidebarOpen(false)}
                className={`p-2 rounded-lg shadow-lg transition-colors ${
                  theme === 'dark' ? 'bg-gray-800/90 hover:bg-gray-700 text-white' : 'bg-white/90 hover:bg-gray-100 text-gray-800'
                } backdrop-blur`}
                title="Hide sidebar"
              >
                <X size={20} />
              </button>
            )}
          </div>
        </div>

        {/* Bottom controls */}
        <div className={`absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex gap-3 px-5 py-2.5 rounded-full border shadow-2xl backdrop-blur ${
          theme === 'dark' ? 'bg-gray-900/80 border-gray-700/50' : 'bg-white/80 border-gray-200'
        }`}>
          <button 
            onClick={() => setAutoRotate(!autoRotate)}
            className={`flex items-center gap-1.5 text-xs font-medium transition-colors ${autoRotate ? 'text-blue-400' : `${ct.textMuted} hover:text-blue-300`}`}
          >
            {autoRotate ? <Pause size={16} /> : <Play size={16} />}
            <span className="hidden sm:inline">Auto-Rotate</span>
          </button>
          <div className={`w-px ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={() => {
              const canvas = document.querySelector('canvas');
              if (canvas) {
                const link = document.createElement('a');
                link.download = 'molecule3d-screenshot.png';
                link.href = canvas.toDataURL('image/png');
                link.click();
              }
            }}
            className={`flex items-center gap-1.5 text-xs font-medium transition-colors ${ct.textMuted} hover:text-blue-300`}
          >
            <Camera size={16} />
            <span className="hidden sm:inline">Screenshot</span>
          </button>
        </div>

        {moleculeData ? (
          <MoleculeCanvas data={moleculeData} autoRotate={autoRotate} config={vizConfig} />
        ) : (
          <div className={`w-full h-full flex flex-col items-center justify-center ${ct.textMuted}`}>
            <div className="w-16 h-16 border-4 border-gray-700 border-t-blue-500 rounded-full animate-spin mb-4"></div>
            <p>Waiting for data...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;