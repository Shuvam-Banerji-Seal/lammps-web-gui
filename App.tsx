import React, { useState, useEffect } from 'react';
import MoleculeCanvas from './components/MoleculeCanvas';
import { parseDataFile } from './services/parser';
import { MoleculeData, VisualizationConfig } from './types';
import { ATOM_COLORS, DEFAULT_ATOM_COLOR, ELEMENT_DATA } from './constants';
import { Upload, FileText, RotateCw, Play, Pause, AlertCircle, Info, Settings, Eye, EyeOff, Palette, Box, Sun, Moon } from 'lucide-react';

type Theme = 'light' | 'dark';

const App: React.FC = () => {
  const [inputText, setInputText] = useState<string>('');
  const [moleculeData, setMoleculeData] = useState<MoleculeData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [theme, setTheme] = useState<Theme>('dark');
  
  // Visualization State
  const [vizConfig, setVizConfig] = useState<VisualizationConfig>({
    atomScale: 1.0,
    bondScale: 1.0,
    materialType: 'realistic',
    backgroundColor: '#151515',
    showBonds: true,
    customColors: {}
  });

  const [activeTab, setActiveTab] = useState<'data' | 'settings'>('data');

  // Load example on mount
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await fetch('/c60.data');
        const text = await response.text();
        handleVisualize(text);
      } catch (error) {
        setError("Failed to fetch initial data.");
      }
    };

    fetchInitialData();
  }, []);


  const handleVisualize = (text: string) => {
    try {
      setError(null);
      const data = parseDataFile(text);
      if (data.atoms.length === 0) {
        throw new Error("No atoms found in data. Check the format.");
      }
      
      // Initialize custom colors for the new data
      const newCustomColors: Record<number, string> = {};
      Object.values(data.atomTypes).forEach(info => {
        // Try to find a default color based on inferred element
        let defaultColor = DEFAULT_ATOM_COLOR;
        
        // 1. Check if Element matches a known atomic number in constants
        const elementMeta = ELEMENT_DATA.find(e => e.symbol === info.element);
        if (elementMeta && ATOM_COLORS[elementMeta.number]) {
          defaultColor = ATOM_COLORS[elementMeta.number];
        } 
        // 2. Check if ID itself is a key in ATOM_COLORS
        else if (ATOM_COLORS[info.id]) {
          defaultColor = ATOM_COLORS[info.id];
        }

        newCustomColors[info.id] = defaultColor;
      });

      setVizConfig(prev => ({
        ...prev,
        customColors: newCustomColors
      }));

      setMoleculeData(data);
      setInputText(text); // Sync input if loaded from example
    } catch (e: any) {
      setError(e.message || "Failed to parse data file.");
      setMoleculeData(null);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setInputText(content);
        handleVisualize(content);
      };
      reader.readAsText(file);
    }
  };

  const handleLoadExample = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/data');
      const data = await response.json();
      handleVisualize(data.data);
    } catch (error) {
      setError("Failed to fetch example data.");
    }
  };

  const updateConfig = (key: keyof VisualizationConfig, value: any) => {
    setVizConfig(prev => ({ ...prev, [key]: value }));
  };

  const updateCustomColor = (typeId: number, color: string) => {
    setVizConfig(prev => ({
      ...prev,
      customColors: {
        ...prev.customColors,
        [typeId]: color
      }
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

  const currentTheme = themeClasses[theme];

  return (
    <div className={`flex h-screen w-screen font-sans ${currentTheme.bg} ${currentTheme.text}`}>
      
      {/* Sidebar */}
      <div className={`flex flex-col border-r transition-all duration-300 ease-in-out ${isSidebarOpen ? 'w-96 translate-x-0' : 'w-0 -translate-x-full overflow-hidden opacity-0'} z-20 ${currentTheme.sidebar}`}>
        
        <div className={`p-5 border-b flex items-center justify-between ${currentTheme.sidebar}`}>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Molecule3D</h1>
            <div className="text-xs text-gray-500">v1.2</div>
        </div>

        {/* Tab Switcher */}
        <div className={`flex border-b ${currentTheme.sidebar}`}>
          <button 
            onClick={() => setActiveTab('data')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'data' ? currentTheme.tabActive : currentTheme.tabInactive}`}
          >
            Data Source
          </button>
          <button 
            onClick={() => setActiveTab('settings')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${activeTab === 'settings' ? currentTheme.tabActive : currentTheme.tabInactive}`}
          >
            Appearance
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          
          {activeTab === 'data' ? (
            <>
              {/* Instructions */}
              <div className={`rounded-lg p-3 text-sm border ${currentTheme.card} ${currentTheme.textMuted}`}>
                 <div className="flex items-center gap-2 mb-2 text-blue-500 font-semibold">
                   <Info size={16} /> 
                   <span>Format Guide</span>
                 </div>
                 Paste LAMMPS-style .data content or upload a file. The parser looks for <code>Atoms</code> and <code>Bonds</code> sections.
              </div>

              {/* Input Area */}
              <div className="space-y-2">
                <label className={`text-sm font-medium ${currentTheme.textHeader}`}>File Operations</label>
                <div className="flex gap-2">
                  <button 
                    onClick={handleLoadExample}
                    className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded text-xs font-medium transition-colors ${currentTheme.button}`}
                  >
                    <RotateCw size={14} /> Load C60 Example
                  </button>
                  <label className="flex-1 cursor-pointer flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white text-xs font-medium transition-colors">
                    <Upload size={14} /> Upload File
                    <input type="file" onChange={handleFileUpload} className="hidden" accept=".data,.txt" />
                  </label>
                </div>
                
                <textarea 
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="# Paste .data content here..."
                  className={`w-full h-64 border rounded p-3 text-xs font-mono resize-y ${currentTheme.input}`}
                  spellCheck={false}
                />
                <button 
                  onClick={() => handleVisualize(inputText)}
                  className="w-full py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-bold text-white transition-colors shadow-lg shadow-green-900/20"
                >
                  Visualize Data
                </button>
              </div>

              {/* Stats */}
              {moleculeData && (
                <div className="space-y-2 pt-4 border-t border-gray-800">
                  <h3 className={`text-sm font-semibold ${currentTheme.textHeader}`}>Statistics</h3>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className={`p-2 rounded flex justify-between ${currentTheme.statsCard}`}>
                      <span className={`${currentTheme.textMuted}`}>Atoms</span>
                      <span className="font-mono">{moleculeData.atoms.length}</span>
                    </div>
                    <div className={`p-2 rounded flex justify-between ${currentTheme.statsCard}`}>
                      <span className={`${currentTheme.textMuted}`}>Bonds</span>
                      <span className="font-mono">{moleculeData.bonds.length}</span>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div className="p-3 bg-red-900/30 border border-red-800 rounded text-red-200 text-xs flex gap-2 items-start">
                  <AlertCircle size={16} className="shrink-0 mt-0.5" />
                  <span>{error}</span>
                </div>
              )}
            </>
          ) : (
            <div className="space-y-8 animate-fadeIn">
                {/* Theme Toggle */}
                <div className="space-y-3">
                  <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                    <Palette size={16} /> Theme
                  </div>
                  <button
                    onClick={toggleTheme}
                    className={`w-full flex items-center justify-center gap-2 py-2 rounded text-xs font-medium transition-colors ${currentTheme.button}`}
                  >
                    {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                    Switch to {theme === 'dark' ? 'Light' : 'Dark'} Mode
                  </button>
                </div>

               {/* Appearance Controls */}
               
               {/* Render Style */}
               <div className="space-y-3">
                 <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                   <Settings size={16} /> Rendering Style
                 </div>
                 <div className="grid grid-cols-3 gap-2">
                    {['realistic', 'plastic', 'toon'].map((type) => (
                      <button
                        key={type}
                        onClick={() => updateConfig('materialType', type)}
                        className={`py-2 px-1 text-xs capitalize rounded border transition-all ${vizConfig.materialType === type 
                          ? 'bg-blue-600 border-blue-500 text-white shadow-lg shadow-blue-900/50' 
                          : `${currentTheme.button} border-gray-700`}`}
                      >
                        {type}
                      </button>
                    ))}
                 </div>
               </div>

               {/* Scaling */}
               <div className="space-y-4">
                 <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                   <Box size={16} /> Geometry Scale
                 </div>
                 
                 <div className="space-y-1">
                   <div className="flex justify-between text-xs text-gray-400">
                     <span>Atom Size</span>
                     <span>{(vizConfig.atomScale * 100).toFixed(0)}%</span>
                   </div>
                   <input 
                     type="range" 
                     min="0.1" 
                     max="3.0" 
                     step="0.1"
                     value={vizConfig.atomScale}
                     onChange={(e) => updateConfig('atomScale', parseFloat(e.target.value))}
                     className={`w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 ${currentTheme.button}`}
                   />
                 </div>

                 <div className="space-y-1">
                   <div className="flex justify-between text-xs text-gray-400">
                     <span>Bond Thickness</span>
                     <span>{(vizConfig.bondScale * 100).toFixed(0)}%</span>
                   </div>
                   <input 
                     type="range" 
                     min="0.1" 
                     max="3.0" 
                     step="0.1"
                     value={vizConfig.bondScale}
                     onChange={(e) => updateConfig('bondScale', parseFloat(e.target.value))}
                     className={`w-full h-2 rounded-lg appearance-none cursor-pointer accent-blue-500 ${currentTheme.button}`}
                   />
                 </div>
               </div>
               
               {/* Atom Colors Section */}
               <div className="space-y-3">
                 <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                   <Palette size={16} /> Atom Colors
                 </div>
                 
                 {moleculeData && Object.values(moleculeData.atomTypes).length > 0 ? (
                   <div className="space-y-2 max-h-60 overflow-y-auto pr-1 custom-scrollbar">
                     {Object.values(moleculeData.atomTypes).map((typeInfo) => (
                       <div key={typeInfo.id} className={`flex items-center justify-between p-2 rounded border ${currentTheme.statsCard} border-gray-700`}>
                         <div className="flex items-center gap-3">
                            <div className="relative w-8 h-8 rounded-full overflow-hidden border border-gray-600">
                              <input 
                                type="color" 
                                value={vizConfig.customColors[typeInfo.id] || DEFAULT_ATOM_COLOR}
                                onChange={(e) => updateCustomColor(typeInfo.id, e.target.value)}
                                className="absolute -top-2 -left-2 w-12 h-12 cursor-pointer p-0 border-0"
                              />
                            </div>
                            <div>
                               <div className={`text-xs font-bold ${currentTheme.textHeader}`}>{typeInfo.label}</div>
                               <div className={`text-[10px] ${currentTheme.textMuted}`}>{typeInfo.element !== 'X' ? `Element: ${typeInfo.element}` : `ID: ${typeInfo.id}`}</div>
                            </div>
                         </div>
                         <div className="bg-gray-700 px-2 py-0.5 rounded text-[10px] font-mono text-gray-300">
                           {typeInfo.count}
                         </div>
                       </div>
                     ))}
                   </div>
                 ) : (
                    <div className={`text-xs italic ${currentTheme.textMuted}`}>Load data to customize atom colors.</div>
                 )}
               </div>

               {/* Visibility */}
               <div className="space-y-3">
                 <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                   <Eye size={16} /> Visibility
                 </div>
                 <button 
                   onClick={() => updateConfig('showBonds', !vizConfig.showBonds)}
                   className={`flex items-center justify-between w-full p-2 rounded text-xs transition-colors ${currentTheme.button}`}
                 >
                   <span>Show Bonds</span>
                   {vizConfig.showBonds ? <Eye size={14} className="text-green-400"/> : <EyeOff size={14} className="text-gray-500"/>}
                 </button>
               </div>

               {/* Background */}
               <div className="space-y-3">
                 <div className={`flex items-center gap-2 text-sm font-semibold border-b pb-2 ${currentTheme.textHeader} ${currentTheme.sidebar}`}>
                   <Box size={16} /> Background
                 </div>
                 <div className="flex gap-2">
                    {['#151515', '#000000', '#1a202c', '#ffffff', '#e2e8f0'].map((color) => (
                      <button
                        key={color}
                        onClick={() => updateConfig('backgroundColor', color)}
                        className={`w-8 h-8 rounded-full border-2 transition-transform hover:scale-110 ${vizConfig.backgroundColor === color ? 'border-blue-500 scale-110' : 'border-transparent'}`}
                        style={{ backgroundColor: color }}
                        title={color}
                      />
                    ))}
                    <div className="relative w-8 h-8 rounded-full overflow-hidden border-2 border-gray-600 hover:border-gray-400">
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
        <div className={`p-4 border-t text-[10px] flex flex-col items-center text-center ${currentTheme.sidebar} ${currentTheme.textMuted}`}>
            <span>Created by <span className="text-gray-300 font-medium">Shuvam Banerji Seal</span></span>
            <a href="https://shuvam-banerji-seal.github.io/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-400 mt-1">
               shuvam-banerji-seal.github.io
            </a>
        </div>
      </div>

      {/* Toggle Sidebar Button (visible when sidebar closed) */}
      {!isSidebarOpen && (
        <button 
          onClick={() => setIsSidebarOpen(true)}
          className="absolute top-4 left-4 z-30 p-2 bg-gray-800 rounded-md hover:bg-gray-700 text-white"
        >
          <FileText size={20} />
        </button>
      )}

      {/* Main Canvas Area */}
      <div className={`flex-1 relative ${currentTheme.bg}`}>
        {/* Toggle Sidebar Button (Inside sidebar area logic handled by layout, this is overlay controls) */}
        <div className="absolute top-4 right-4 z-10 flex gap-2">
             {isSidebarOpen && (
                 <button 
                    onClick={() => setIsSidebarOpen(false)}
                    className="p-2 bg-gray-800/80 backdrop-blur rounded hover:bg-gray-700 text-white transition-colors"
                    title="Hide Sidebar"
                 >
                   <FileText size={20} />
                 </button>
             )}
        </div>

        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 flex gap-4 bg-gray-900/80 backdrop-blur px-6 py-3 rounded-full border border-gray-700/50 shadow-2xl">
           <button 
             onClick={() => setAutoRotate(!autoRotate)}
             className={`flex items-center gap-2 text-sm font-medium transition-colors ${autoRotate ? 'text-blue-400' : 'text-gray-400 hover:text-white'}`}
           >
             {autoRotate ? <Pause size={18} /> : <Play size={18} />}
             Auto-Rotate
           </button>
        </div>

        {moleculeData ? (
          <MoleculeCanvas data={moleculeData} autoRotate={autoRotate} config={vizConfig} />
        ) : (
          <div className={`w-full h-full flex flex-col items-center justify-center ${currentTheme.textMuted}`}>
            <div className="w-16 h-16 border-4 border-gray-700 border-t-blue-500 rounded-full animate-spin mb-4"></div>
            <p>Waiting for data...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;