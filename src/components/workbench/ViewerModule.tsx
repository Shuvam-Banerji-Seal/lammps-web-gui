import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import MoleculeCanvas from '../../components/MoleculeCanvas';
import { parseFile, detectFileFormat } from '../../services/fileParser';
import { parseInWorker } from '../../services/parserClient';
import {
  MoleculeData, VisualizationConfig, VisualizationMode, FileFormat,
  LightingPreset, CameraPreset,
} from '../../types';
import { DEFAULT_ATOM_COLOR, ELEMENT_DATA, ATOM_COLORS } from '../../constants';
import { emitCameraCommand } from '../../services/cameraBus';
import { useKeyboardShortcuts, SHORTCUT_CATALOG } from '../../hooks/useKeyboardShortcuts';
import { encodeViewState, viewStateFromSearch } from '../../services/viewState';
import { measureSelection, measurementGlyph, MeasurementResult } from '../../services/measure';
import { startCanvasRecording, RecordingHandle } from '../../services/recorder';
import { captureActiveCanvas } from '../../services/glRegistry';
import { getThemeTokens, initialTheme, Theme as UITheme } from '../../theme';
import {
  Upload, RotateCw, AlertCircle, Info, Settings, Eye, EyeOff, Palette, Box,
  Sun, Moon, Menu, X, Camera, Atom, Keyboard, Layers, Lightbulb,
  Grid3x3, Maximize2, Play, Pause, ZoomIn, ZoomOut, FileText, HelpCircle,
  Link2, Check, ChevronLeft, ChevronRight, Ruler, Circle, Loader2, BarChart3, Activity, TrendingUp, Download, Sparkles,
} from 'lucide-react';
import { LineChart, Histogram } from '../charts/SimpleChart';
import { computeRDF, computeMSD, computeDensityProfile, computeSpeedDistribution, trajStats } from '../../services/trajectoryAnalysis';
import { downloadTextFile } from '../../lammps/exporter';

/** GitHub mark as inline SVG — lucide 1.x removed brand icons. */
const GithubIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
);

type Theme = 'light' | 'dark';
type Tab = 'data' | 'view' | 'scene' | 'elements' | 'analysis';

const VIZ_MODES: { value: VisualizationMode; label: string; desc: string; key: string }[] = [
  { value: 'ball-and-stick', label: 'Ball & Stick', desc: 'Classic molecular view', key: '1' },
  { value: 'space-fill', label: 'Space Fill', desc: 'Van der Waals radii', key: '2' },
  { value: 'wireframe', label: 'Wireframe', desc: 'Small point atoms', key: '3' },
  { value: 'licorice', label: 'Licorice', desc: 'Bond-centric sticks', key: '4' },
];

const MATERIALS: { value: VisualizationConfig['materialType']; label: string }[] = [
  { value: 'realistic', label: 'Realistic' },
  { value: 'plastic', label: 'Plastic' },
  { value: 'metallic', label: 'Metallic' },
  { value: 'toon', label: 'Toon' },
];

const LIGHTING_PRESETS: LightingPreset[] = ['studio', 'lab', 'outdoor', 'space', 'soft'];

const CAMERA_VIEWS: { preset: CameraPreset; label: string }[] = [
  { preset: 'iso', label: 'Iso' },
  { preset: 'front', label: 'Front' },
  { preset: 'back', label: 'Back' },
  { preset: 'left', label: 'Left' },
  { preset: 'right', label: 'Right' },
  { preset: 'top', label: 'Top' },
  { preset: 'bottom', label: 'Bottom' },
];

const EXAMPLES: { file: string; format: FileFormat; label: string }[] = [
  { file: 'c60.data', format: 'lammps', label: 'C60 · LAMMPS' },
  { file: 'examples/benzene.pdb', format: 'pdb', label: 'Benzene · PDB' },
  { file: 'examples/nacl.cif', format: 'cif', label: 'NaCl · CIF' },
  { file: 'examples/water.xyz', format: 'xyz', label: 'Water · XYZ' },
  { file: 'examples/water-traj.xyz', format: 'xyz', label: 'Trajectory · XYZ' },
  { file: 'examples/water-dump.lammpstrj', format: 'lammpsdump', label: 'Trajectory · Dump' },
  { file: 'examples/stress-12k.xyz', format: 'xyz', label: 'Stress 12k · XYZ' },
  { file: 'examples/stress-60k.xyz', format: 'xyz', label: 'Stress 60k · XYZ' },
];

const prefersLightScheme = (): boolean =>
  typeof window !== 'undefined' &&
  !!window.matchMedia?.('(prefers-color-scheme: light)').matches;

/** Base defaults, optionally overridden by a shared-view ?s= token (P3). */
const initialConfig = (): VisualizationConfig => ({
  atomScale: 1.0,
  bondScale: 1.0,
  materialType: 'realistic',
  backgroundColor: prefersLightScheme() ? '#f4f5f7' : '#151515',
  showBonds: true,
  customColors: {},
  visualizationMode: 'ball-and-stick',
  lightingPreset: 'studio',
  showBox: false,
  showAxes: false,
  showLabels: false,
  shadowsEnabled: true,
  autoRotateSpeed: 0.5,
  fov: 40,
});

const ViewerModule: React.FC<{
  embedded?: boolean;
  /** Owned by App when mounted in the workbench; standalone otherwise. */
  theme?: UITheme;
  onToggleTheme?: () => void;
}> = ({ embedded = false, theme: themeProp, onToggleTheme }) => {
  const [inputText, setInputText] = useState<string>('');
  const [moleculeData, setMoleculeData] = useState<MoleculeData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  // Theme: controlled by App when provided, else local (standalone/embedded).
  const [localTheme, setLocalTheme] = useState<UITheme>(initialTheme);
  const theme: UITheme = themeProp ?? localTheme;
  const toggleTheme = useCallback(() => {
    if (themeProp !== undefined && onToggleTheme) {
      onToggleTheme();
      return;
    }
    setLocalTheme(currentTheme => {
      const next = currentTheme === 'dark' ? 'light' : 'dark';
      updateConfig('backgroundColor', next === 'dark' ? '#151515' : '#f4f5f7');
      return next;
    });
  }, [themeProp, onToggleTheme]);
  const [fileFormat, setFileFormat] = useState<FileFormat>('lammps');
  const [activeTab, setActiveTab] = useState<Tab>('data');
  const [showHelp, setShowHelp] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  /** Type ids the user manually recolored — survive re-parsing. */
  const userEditedTypes = useRef<Set<number>>(new Set());

  // Shared view state (P3): ?s=<token> overrides defaults before first paint.
  const [vizConfig, setVizConfig] = useState<VisualizationConfig>(() => ({
    ...initialConfig(),
    ...(typeof window !== 'undefined'
      ? viewStateFromSearch(window.location.search) ?? {}
      : {}),
  }));

  // Measurement tool (P4)
  const [selectedIds, setSelectedIds] = useState<number[]>([]);

  // Trajectory playback (P5)
  const frameCount = moleculeData?.frames?.length ?? 1;
  const [frameIdx, setFrameIdx] = useState(0);
  const [trajPlaying, setTrajPlaying] = useState(false);
  const [trajFps, setTrajFps] = useState(10);

  // Share-link feedback
  const [linkCopied, setLinkCopied] = useState(false);

  // Video recording (P8)
  const recorderRef = useRef<RecordingHandle | null>(null);
  const autoRotateBeforeRecording = useRef(true);
  const [isRecording, setIsRecording] = useState(false);
  const [savingVideo, setSavingVideo] = useState(false);
  const [recordingMs, setRecordingMs] = useState(0);

  // Resizable desktop sidebar (P9)
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    try {
      const stored = Number(localStorage.getItem('m3d.sidebarWidth'));
      return Number.isFinite(stored) && stored >= 280 && stored <= 560 ? stored : 384;
    } catch {
      return 384;
    }
  });
  const sidebarWidthRef = useRef(sidebarWidth);
  sidebarWidthRef.current = sidebarWidth;
  const resizingRef = useRef(false);

  // New structure → reset transient view state (P4/P5)
  useEffect(() => {
    setSelectedIds([]);
    setFrameIdx(0);
    setTrajPlaying(false);
  }, [moleculeData]);

  // Theme flips: keep the canvas default background coherent when the user
  // has NOT customized it (custom colors and share-link values are untouched).
  const lastThemeRef = useRef(theme);
  useEffect(() => {
    if (lastThemeRef.current === theme) return;
    const defaultFor = (t: UITheme) => (t === 'dark' ? '#151515' : '#f4f5f7');
    const prevDefault = defaultFor(lastThemeRef.current);
    lastThemeRef.current = theme;
    setVizConfig(prev =>
      prev.backgroundColor === prevDefault
        ? { ...prev, backgroundColor: defaultFor(theme) }
        : prev
    );
  }, [theme]);

  // Trajectory playback clock (P5)
  useEffect(() => {
    if (!trajPlaying || frameCount <= 1) return;
    const id = window.setInterval(
      () => setFrameIdx(i => (i + 1) % frameCount),
      Math.max(16, Math.round(1000 / trajFps))
    );
    return () => window.clearInterval(id);
  }, [trajPlaying, trajFps, frameCount]);

  // Shareable URL sync (P3): debounced replaceState, never reloads.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const id = window.setTimeout(() => {
      try {
        const url = new URL(window.location.href);
        url.searchParams.set('s', encodeViewState(vizConfig));
        window.history.replaceState(null, '', url.toString());
      } catch {
        /* history unavailable (sandboxed iframe etc.) — non-fatal */
      }
    }, 300);
    return () => window.clearTimeout(id);
  }, [vizConfig]);

  const toggleSelectAtom = useCallback((id: number) => {
    setSelectedIds(prev =>
      prev.includes(id)
        ? prev.filter(x => x !== id)
        : [...prev, id].slice(-4) // keep the last four picks
    );
  }, []);

  const copyShareLink = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setLinkCopied(true);
      window.setTimeout(() => setLinkCopied(false), 1600);
    } catch {
      setError('Clipboard unavailable — copy the address bar URL instead.');
    }
  }, []);

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

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const response = await fetch(`${import.meta.env.BASE_URL}c60.data`);
        if (!response.ok) throw new Error(String(response.status));
        const text = await response.text();
        handleVisualize(text, 'lammps');
      } catch {
        setError('Failed to load the built-in C60 example.');
      }
    };
    fetchInitialData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const [isParsing, setIsParsing] = useState(false);

  const applyParsed = useCallback((data: MoleculeData) => {
    if (data.atoms.length === 0) throw new Error('No atoms found in data. Check the format.');

    // Color assignment: user-edited colors survive; others get CPK by element symbol.
    const newCustomColors: Record<number, string> = {};
    for (const info of Object.values(data.atomTypes)) {
      const edited = userEditedTypes.current.has(info.id);
      if (edited && vizConfig.customColors[info.id]) {
        newCustomColors[info.id] = vizConfig.customColors[info.id];
        continue;
      }
      const meta = ELEMENT_DATA.find(e => e.symbol === info.element);
      newCustomColors[info.id] = meta
        ? ATOM_COLORS[meta.number] ?? DEFAULT_ATOM_COLOR
        : DEFAULT_ATOM_COLOR;
    }

    setVizConfig(prev => ({ ...prev, customColors: newCustomColors }));
    setMoleculeData(data);
    setError(null);
    // Very large systems render one static frame instead of spinning —
    // continuous rotation on huge instanced scenes wastes GPU and can
    // saturate low-end/software renderers. Users can re-enable rotation.
    if (data.atoms.length > 20000) setAutoRotate(false);
    if (isMobile) setIsSidebarOpen(false);
  }, [isMobile, vizConfig.customColors]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Parse via Web Worker when available (keeps huge files from freezing the
   * UI); falls back to synchronous parsing otherwise.
   */
  const handleVisualize = useCallback((text: string, format?: FileFormat) => {
    setError(null);
    setIsParsing(true);
    const fmt = format ?? detectFileFormat('pasted.txt');
    parseInWorker(text, fmt)
      .catch(err => {
        if (!(err instanceof Error && /no-worker|worker-crashed/.test(err.message))) throw err;
        return parseFile(text, fmt); // synchronous fallback
      })
      .then(applyParsed)
      .then(() => setInputText(text))
      .catch(e => {
        setError(e instanceof Error ? e.message : 'Failed to parse data file.');
        setMoleculeData(null);
      })
      .finally(() => setIsParsing(false));
  }, [applyParsed]);

  const loadExample = useCallback(async (file: string, format: FileFormat) => {
    try {
      const response = await fetch(`${import.meta.env.BASE_URL}${file}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const text = await response.text();
      setFileFormat(format);
      handleVisualize(text, format);
    } catch {
      setError(`Failed to fetch example: ${file}`);
    }
  }, [handleVisualize]);

  const handleFileUpload = useCallback((file: File) => {
    const detectedFormat = detectFileFormat(file.name);
    setFileFormat(detectedFormat);
    const reader = new FileReader();
    reader.onload = e => {
      const content = (e.target?.result as string) ?? '';
      setInputText(content);
      handleVisualize(content, detectedFormat);
    };
    reader.readAsText(file);
  }, [handleVisualize]);

  // Drag & drop anywhere on the window
  useEffect(() => {
    const onDragOver = (e: DragEvent) => { e.preventDefault(); setDragActive(true); };
    const onDragLeave = (e: DragEvent) => {
      if (e.relatedTarget === null) setDragActive(false);
    };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFileUpload(file);
    };
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);
    return () => {
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, [handleFileUpload]);

  const updateConfig = (key: keyof VisualizationConfig, value: unknown) => {
    setVizConfig(prev => ({ ...prev, [key]: value }));
  };

  const updateCustomColor = (typeId: number, color: string) => {
    userEditedTypes.current.add(typeId);
    setVizConfig(prev => ({
      ...prev,
      customColors: { ...prev.customColors, [typeId]: color },
    }));
  };

  const doScreenshot = useCallback(() => {
    // Explicit render-before-capture (no preserveDrawingBuffer tax).
    const dataUrl = captureActiveCanvas();
    if (!dataUrl) {
      setError('Screenshot failed — canvas not ready.');
      return;
    }
    try {
      const link = document.createElement('a');
      link.download = `molecule3d-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch {
      setError('Screenshot failed — canvas not ready.');
    }
  }, []);

  // --- Video recording (P8) ---
  const startRecording = useCallback(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) {
      setError('No canvas to record yet.');
      return;
    }
    try {
      autoRotateBeforeRecording.current = autoRotate;
      setAutoRotate(true); // guarantee motion in the captured video
      recorderRef.current = startCanvasRecording(canvas as HTMLCanvasElement, { fps: 60 });
      setRecordingMs(0);
      setIsRecording(true);
    } catch (e) {
      setAutoRotate(autoRotateBeforeRecording.current);
      setError(e instanceof Error ? e.message : 'Recording failed to start.');
    }
  }, [autoRotate]);

  const stopRecording = useCallback(async () => {
    const rec = recorderRef.current;
    if (!rec) return;
    recorderRef.current = null;
    setIsRecording(false);
    setSavingVideo(true);
    try {
      const result = await rec.stop();
      const url = URL.createObjectURL(result.blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `molecule3d-${new Date().toISOString().replace(/[:.]/g, '-')}.${result.ext}`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save recording.');
    } finally {
      setSavingVideo(false);
      setAutoRotate(autoRotateBeforeRecording.current);
    }
  }, []);

  useEffect(() => {
    if (!isRecording) return;
    const id = window.setInterval(() => setRecordingMs(ms => ms + 250), 250);
    return () => window.clearInterval(id);
  }, [isRecording]);

  // --- Desktop sidebar resize (P9) ---
  const startSidebarResize = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    resizingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const moveSidebarResize = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current) return;
    setSidebarWidth(Math.min(560, Math.max(280, Math.round(e.clientX))));
  };
  const endSidebarResize = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!resizingRef.current) return;
    resizingRef.current = false;
    try {
      localStorage.setItem('m3d.sidebarWidth', String(sidebarWidthRef.current));
    } catch { /* storage unavailable — non-fatal */ }
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const cycleLighting = useCallback(() => {
    setVizConfig(prev => {
      const idx = LIGHTING_PRESETS.indexOf(prev.lightingPreset);
      return { ...prev, lightingPreset: LIGHTING_PRESETS[(idx + 1) % LIGHTING_PRESETS.length] };
    });
  }, []);

  useKeyboardShortcuts({
    onToggleRotate: () => setAutoRotate(v => !v),
    onFit: () => emitCameraCommand({ type: 'fit' }),
    onViewPreset: preset => emitCameraCommand({ type: 'preset', preset }),
    onZoom: delta => emitCameraCommand({ type: 'zoom', delta }),
    onOrbit: (dx, dy) => emitCameraCommand({ type: 'orbit', dx, dy }),
    onVisualizationMode: idx =>
      updateConfig('visualizationMode', VIZ_MODES[idx].value),
    onToggleBonds: () => setVizConfig(p => ({ ...p, showBonds: !p.showBonds })),
    onToggleBox: () => setVizConfig(p => ({ ...p, showBox: !p.showBox })),
    onToggleLabels: () => setVizConfig(p => ({ ...p, showLabels: !p.showLabels })),
    onCycleLighting: cycleLighting,
    onToggleTheme: toggleTheme,
    onToggleSidebar: () => setIsSidebarOpen(v => !v),
    onScreenshot: doScreenshot,
    onHelp: () => setShowHelp(v => !v),
    onEscape: () => {
      setShowHelp(false);
      setSelectedIds([]);
      if (isMobile) setIsSidebarOpen(false);
    },
    onClearSelection: () => setSelectedIds([]),
    onPrevFrame: () => setFrameIdx(i => (i - 1 + frameCount) % frameCount),
    onNextFrame: () => setFrameIdx(i => (i + 1) % frameCount),
    onToggleTrajectoryPlay: () => { if (frameCount > 1) setTrajPlaying(v => !v); },
  });

  // ---- theme tokens (shared warm coffee-green palette, flat, no gradients) ----
  const shared = getThemeTokens(theme);
  const ct = {
    ...shared,
    sidebar: shared.panel,
    header: shared.headerText,
  };

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'data', label: 'Data', icon: <FileText size={14} /> },
    { id: 'view', label: 'View', icon: <Atom size={14} /> },
    { id: 'scene', label: 'Scene', icon: <Lightbulb size={14} /> },
    { id: 'elements', label: 'Elements', icon: <Palette size={14} /> },
    { id: 'analysis', label: 'Analysis', icon: <BarChart3 size={14} /> },
  ];

  const atomTypeList = moleculeData ? Object.values(moleculeData.atomTypes) : [];

  // Active frame for trajectory playback (P5): swaps atoms, keeps topology.
  const activeData: MoleculeData | null = useMemo(() => {
    if (!moleculeData || !moleculeData.frames || frameCount <= 1) return moleculeData;
    const frame = moleculeData.frames[Math.min(frameIdx, frameCount - 1)];
    return frame ? { ...moleculeData, atoms: frame.atoms } : moleculeData;
  }, [moleculeData, frameIdx, frameCount]);

  const measurement: MeasurementResult | null = useMemo(() => {
    if (!activeData) return null;
    const picked = selectedIds
      .map(id => activeData.atoms.find(a => a.id === id))
      .filter((a): a is NonNullable<typeof a> => !!a);
    return measureSelection(picked);
  }, [activeData, selectedIds]);

  const measurementHint = useMemo(() => {
    if (!selectedIds.length) return 'Click 2–4 atoms to measure';
    return `${selectedIds.length} picked — ${4 - selectedIds.length > 0 ? `pick ${4 - selectedIds.length} more` : 'measurement ready'}`;
  }, [selectedIds]);

  return (
    <div className={`flex h-full w-full font-sans overflow-hidden ${ct.bg} ${ct.text}`}>
      {/* Mobile backdrop */}
      {isMobile && isSidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-20" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* Drag overlay */}
      {dragActive && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-[#22301c]/60 border-4 border-dashed border-[#7fa66b]">
          <div className="px-6 py-4 rounded-xl bg-[#1e1913] text-[#ede5d8] text-sm font-medium shadow-2xl">
            Drop a structure file to visualize (.data .xyz .pdb .cif)
          </div>
        </div>
      )}

      <aside
        className={`
        flex flex-col border-r transition-transform duration-300 ease-in-out z-30
        ${isMobile ? 'fixed inset-y-0 left-0 w-80 max-w-[85vw] shadow-2xl' : 'relative shrink-0'}
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
        ${ct.sidebar}
      `}
        style={isMobile ? undefined : { width: sidebarWidth }}
      >
        {/* Desktop resize handle (P9) */}
        {!isMobile && isSidebarOpen && (
          <div
            onPointerDown={startSidebarResize}
            onPointerMove={moveSidebarResize}
            onPointerUp={endSidebarResize}
            className="absolute top-0 right-[-3px] h-full w-1.5 cursor-col-resize z-40 hover:bg-[#7fa66b]/40 transition-colors touch-none"
            title="Drag to resize sidebar"
            aria-label="Resize sidebar"
          />
        )}
        {/* Header */}
        <div className={`flex items-center justify-between px-4 h-14 border-b ${ct.divider}`}>
          <div className="flex items-center gap-2.5">
            <img src={`${import.meta.env.BASE_URL}logo.svg`} alt="Molecule3D logo" className="w-8 h-8" />
            <div className="min-w-0">
              <h1 className="text-base font-bold tracking-tight">Molecule3D</h1>
              <p className={`hidden text-[10px] leading-none sm:block ${ct.muted}`}>Molecular Structure Viewer</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <a
              href="https://github.com/Shuvam-Banerji-Seal/lammps-web-gui"
              target="_blank" rel="noopener noreferrer"
              className={`p-2 rounded-lg ${ct.button}`}
              title="GitHub repository"
            >
              <GithubIcon size={16} />
            </a>
            {isMobile && (
              <button onClick={() => setIsSidebarOpen(false)} className={`p-2 rounded-lg ${ct.button}`} title="Close sidebar">
                <X size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Tabs */}
        <nav className={`grid grid-cols-5 border-b ${ct.divider}`} aria-label="Sidebar sections">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex flex-col items-center gap-1.5 py-3 text-xs font-semibold tracking-wide transition-colors ${
                activeTab === tab.id
                  ? `${ct.active} border-b-2`
                  : `${ct.muted} hover:${theme === 'dark' ? 'text-gray-200' : 'text-gray-700'}`
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-6 antialiased">
          {/* Improved readability: slightly larger base, better line-height */}
          {/* ============================== DATA TAB */}
          {activeTab === 'data' && (
            <>
              <section className={`rounded-xl p-4 text-sm border shadow-sm ${ct.card} ${ct.muted} leading-relaxed`}>
                <div className={`flex items-center gap-2 mb-2.5 font-bold tracking-tight ${ct.accentText}`}>
                  <Info size={15} /> Supported formats
                </div>
                <ul className="space-y-1.5 leading-relaxed">
                  <li><span className={`font-semibold ${ct.accentText}`}>.data / .lmp</span> — LAMMPS (atomic·charge·molecular·full)</li>
                  <li><span className={`font-semibold ${theme === 'dark' ? 'text-[#e4b877]' : 'text-[#7a5716]'}`}>.xyz</span> — XYZ trajectories (first frame)</li>
                  <li><span className={`font-semibold ${theme === 'dark' ? 'text-[#e4b877]' : 'text-[#7a5716]'}`}>.lammpstrj / .dump</span> — LAMMPS dump trajectories (playback)</li>
                  <li><span className={`font-semibold ${theme === 'dark' ? "text-[#c9a9d4]" : "text-[#7d5a8c]"}`}>.pdb</span> — Protein Data Bank (+CONECT, CRYST1)</li>
                  <li><span className={`font-semibold ${theme === 'dark' ? "text-[#cf8b76]" : "text-[#a4502f]"}`}>.cif</span> — Crystallographic Information Framework</li>
                </ul>
              </section>

              <section className="space-y-3">
                <h3 className={`text-sm font-bold tracking-tight ${ct.header}`}>Load structure</h3>
                <label className={`flex cursor-pointer items-center justify-center gap-2 px-3 py-3 rounded-xl text-sm font-semibold transition-colors shadow-sm hover:shadow ${ct.accent}`}>
                  <Upload size={14} /> Upload file
                  <input
                    ref={fileInputRef}
                    type="file"
                    className="hidden"
                    accept=".data,.lmp,.lammps,.xyz,.pdb,.ent,.cif,.mmcif,.lammpstrj,.dump,.txt"
                    onChange={e => {
                      const f = e.target.files?.[0];
                      if (f) handleFileUpload(f);
                      e.target.value = '';
                    }}
                  />
                </label>
                <div className="grid grid-cols-2 gap-2.5">
                  {EXAMPLES.map(ex => (
                    <button
                      key={ex.file}
                      onClick={() => loadExample(ex.file, ex.format)}
                      className={`px-3 py-2.5 rounded-xl text-xs font-semibold transition-colors shadow-sm hover:shadow ${ct.button}`}
                      title={`Load ${ex.label} example`}
                    >
                      {ex.label}
                    </button>
                  ))}
                </div>
              </section>

              <section className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className={`text-sm font-bold tracking-tight ${ct.header}`}>Paste data</h3>
                  <div className="flex gap-1">
                    {( ['lammps', 'xyz', 'pdb', 'cif', 'lammpsdump'] as FileFormat[]).map(fmt => (
                      <button
                        key={fmt}
                        onClick={() => setFileFormat(fmt)}
                        className={`px-2 py-1 rounded text-[10px] uppercase font-bold transition-colors ${
                          fileFormat === fmt ? ct.accent : ct.chip + ' ' + ct.muted
                        }`}
                        title={`Parse pasted text as ${fmt === 'lammpsdump' ? 'LAMMPS dump' : fmt.toUpperCase()}`}
                      >
                        {fmt === 'lammps' ? 'lmp' : fmt === 'lammpsdump' ? 'dump' : fmt}
                      </button>
                    ))}
                  </div>
                </div>
                <textarea
                  value={inputText}
                  onChange={e => setInputText(e.target.value)}
                  placeholder="# Paste LAMMPS / XYZ / PDB / CIF content here…"
                  className={`w-full h-44 border rounded-xl p-3.5 text-sm font-mono resize-y focus:outline-none leading-relaxed ${ct.input}`}
                  spellCheck={false}
                />
                <button
                  onClick={() => handleVisualize(inputText, fileFormat)}
                  className={`w-full py-3 rounded-xl text-sm font-bold tracking-wide transition-colors shadow-sm hover:shadow ${ct.go}`}
                >
                  Visualize
                </button>
              </section>

              {moleculeData && (
                <section className="space-y-2 pt-1">
                  <h3 className={`text-xs font-semibold ${ct.header}`}>Statistics</h3>
                  <div className="grid grid-cols-4 gap-2 text-center">
                    {[
                      { label: 'Atoms', value: moleculeData.atoms.length },
                      { label: 'Bonds', value: moleculeData.bonds.length },
                      { label: 'Types', value: atomTypeList.length },
                      { label: 'Box', value: moleculeData.box ? 'Yes' : '—' },
                    ].map(s => (
                      <div key={s.label} className={`p-2 rounded-lg ${ct.stat}`}>
                        <div className={`text-[10px] ${ct.muted}`}>{s.label}</div>
                        <div className="text-sm font-mono font-bold">{s.value}</div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {error && (
                <div className={`p-3 rounded-lg border text-xs flex gap-2 items-start ${ct.errorBox}`} role="alert">
                  <AlertCircle size={14} className="shrink-0 mt-0.5" />
                  <span className="leading-relaxed">{error}</span>
                </div>
              )}
            </>
          )}

          {/* ============================== VIEW TAB */}
          {activeTab === 'view' && (
            <>
              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Representation</h3>
                <div className="grid grid-cols-2 gap-2">
                  {VIZ_MODES.map(mode => (
                    <button
                      key={mode.value}
                      onClick={() => updateConfig('visualizationMode', mode.value)}
                      className={`text-left p-2.5 rounded-lg border text-xs transition-colors ${
                        vizConfig.visualizationMode === mode.value ? ct.active : ct.button
                      }`}
                    >
                      <div className="font-semibold">{mode.label}</div>
                      <div className={`text-[10px] mt-0.5 ${ct.muted}`}>{mode.desc}</div>
                      <kbd className={`inline-block mt-1 px-1 rounded text-[9px] ${ct.chip}`}>{mode.key}</kbd>
                    </button>
                  ))}
                </div>
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Material</h3>
                <div className="grid grid-cols-4 gap-2">
                  {MATERIALS.map(m => (
                    <button
                      key={m.value}
                      onClick={() => updateConfig('materialType', m.value)}
                      className={`py-2 text-[11px] font-medium rounded-lg border capitalize transition-colors ${
                        vizConfig.materialType === m.value ? ct.active : ct.button
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </section>

              <section className="space-y-3">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Geometry</h3>
                {([
                  { label: 'Atom size', key: 'atomScale' as const },
                  { label: 'Bond thickness', key: 'bondScale' as const },
                ]).map(sl => (
                  <div key={sl.key} className="space-y-1">
                    <div className={`flex justify-between text-[11px] ${ct.muted}`}>
                      <span>{sl.label}</span>
                      <span className="font-mono">{(vizConfig[sl.key] * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range" min="0.1" max="3" step="0.05"
                      value={vizConfig[sl.key]}
                      onChange={e => updateConfig(sl.key, parseFloat(e.target.value))}
                      className={`w-full ${theme === 'dark' ? "accent-[#7fa66b]" : "accent-[#4e7a41]"}`}
                    />
                  </div>
                ))}
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Visibility</h3>
                {([
                  { key: 'showBonds' as const, label: 'Bonds', hint: 'B' },
                  { key: 'showLabels' as const, label: `Element labels${moleculeData && moleculeData.atoms.length > 400 ? ' (≤400 atoms)' : ''}`, hint: 'L' },
                ]).map(row => (
                  <button
                    key={row.key}
                    onClick={() => updateConfig(row.key, !vizConfig[row.key])}
                    className={`flex items-center justify-between w-full p-2.5 rounded-lg text-xs transition-colors ${ct.button}`}
                  >
                    <span>{row.label} <kbd className={`ml-1 px-1 rounded text-[9px] ${ct.chip}`}>{row.hint}</kbd></span>
                    {vizConfig[row.key]
                      ? <Eye size={14} className={ct.accentText} />
                      : <EyeOff size={14} className={ct.muted} />}
                  </button>
                ))}
              </section>
            </>
          )}

          {/* ============================== SCENE TAB */}
          {activeTab === 'scene' && (
            <>
              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Lighting</h3>
                <div className="grid grid-cols-3 gap-2">
                  {LIGHTING_PRESETS.map(p => (
                    <button
                      key={p}
                      onClick={() => updateConfig('lightingPreset', p)}
                      className={`py-2 text-[11px] font-medium rounded-lg border capitalize transition-colors ${
                        vizConfig.lightingPreset === p ? ct.active : ct.button
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
                <p className={`text-[10px] ${ct.muted}`}>
                  Shortcut <kbd className={`px-1 rounded text-[9px] ${ct.chip}`}>G</kbd> cycles presets.
                </p>
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Camera views</h3>
                <div className="grid grid-cols-4 gap-2">
                  {CAMERA_VIEWS.map(v => (
                    <button
                      key={v.preset}
                      onClick={() => emitCameraCommand({ type: 'preset', preset: v.preset })}
                      className={`py-2 text-[11px] font-medium rounded-lg border transition-colors ${ct.button}`}
                      title={`View from ${v.label.toLowerCase()}`}
                    >
                      {v.label}
                    </button>
                  ))}
                  <button
                    onClick={() => emitCameraCommand({ type: 'fit' })}
                    className={`py-2 text-[11px] font-medium rounded-lg border flex items-center justify-center gap-1 transition-colors ${ct.button}`}
                    title="Fit to scene (R)"
                  >
                    <Maximize2 size={12} /> Fit
                  </button>
                </div>
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Simulation box & axes</h3>
                <button
                  onClick={() => updateConfig('showBox', !vizConfig.showBox)}
                  disabled={!moleculeData?.box}
                  className={`flex items-center justify-between w-full p-2.5 rounded-lg text-xs transition-colors disabled:opacity-40 ${ct.button}`}
                  title="X"
                >
                  <span className="flex items-center gap-2"><Grid3x3 size={14} /> Show simulation box <kbd className={`px-1 rounded text-[9px] ${ct.chip}`}>X</kbd></span>
                  <span className={`w-8 h-4 rounded-full relative transition-colors ${vizConfig.showBox ? ct.toggleOn : ct.toggleOff}`}>
                    <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${vizConfig.showBox ? 'left-4' : 'left-0.5'}`} />
                  </span>
                </button>
                <button
                  onClick={() => updateConfig('showAxes', !vizConfig.showAxes)}
                  className={`flex items-center justify-between w-full p-2.5 rounded-lg text-xs transition-colors ${ct.button}`}
                >
                  <span className="flex items-center gap-2"><Grid3x3 size={14} /> Show XYZ axes</span>
                  <span className={`w-8 h-4 rounded-full relative transition-colors ${vizConfig.showAxes ? ct.toggleOn : ct.toggleOff}`}>
                    <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${vizConfig.showAxes ? 'left-4' : 'left-0.5'}`} />
                  </span>
                </button>
                {!moleculeData?.box && (
                  <p className={`text-[10px] ${ct.muted}`}>This structure has no cell information (load a LAMMPS/PDB-CRYST1/CIF file).</p>
                )}
              </section>

              <section className="space-y-3">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Motion & optics</h3>
                <div className="space-y-1">
                  <div className={`flex justify-between text-[11px] ${ct.muted}`}>
                    <span>Auto-rotate speed</span><span className="font-mono">{vizConfig.autoRotateSpeed.toFixed(1)}</span>
                  </div>
                  <input
                    type="range" min="0.1" max="6" step="0.1"
                    value={vizConfig.autoRotateSpeed}
                    onChange={e => updateConfig('autoRotateSpeed', parseFloat(e.target.value))}
                    className={`w-full ${theme === 'dark' ? "accent-[#7fa66b]" : "accent-[#4e7a41]"}`}
                  />
                </div>
                <div className="space-y-1">
                  <div className={`flex justify-between text-[11px] ${ct.muted}`}>
                    <span>Field of view</span><span className="font-mono">{vizConfig.fov.toFixed(0)}°</span>
                  </div>
                  <input
                    type="range" min="15" max="90" step="1"
                    value={vizConfig.fov}
                    onChange={e => updateConfig('fov', parseInt(e.target.value, 10))}
                    className={`w-full ${theme === 'dark' ? "accent-[#7fa66b]" : "accent-[#4e7a41]"}`}
                  />
                </div>
                <button
                  onClick={() => updateConfig('shadowsEnabled', !vizConfig.shadowsEnabled)}
                  className={`flex items-center justify-between w-full p-2.5 rounded-lg text-xs transition-colors ${ct.button}`}
                >
                  <span>Shadows (≤8k atoms)</span>
                  <span className={`w-8 h-4 rounded-full relative transition-colors ${vizConfig.shadowsEnabled ? ct.toggleOn : ct.toggleOff}`}>
                    <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${vizConfig.shadowsEnabled ? 'left-4' : 'left-0.5'}`} />
                  </span>
                </button>
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Appearance</h3>
                <button
                  onClick={toggleTheme}
                  className={`flex items-center justify-center gap-2 w-full py-2.5 rounded-lg text-xs font-medium transition-colors ${ct.button}`}
                >
                  {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                  Switch to {theme === 'dark' ? 'light' : 'dark'} mode <kbd className={`px-1 rounded text-[9px] ${ct.chip}`}>T</kbd>
                </button>
                <div className="flex gap-2 flex-wrap pt-1">
                  {(theme === 'dark'
                    ? ['#151515', '#000000', '#10141c', '#1a1e26']
                    : ['#f4f5f7', '#ffffff', '#eef2f7', '#e8ecef']
                  ).map(color => (
                    <button
                      key={color}
                      onClick={() => updateConfig('backgroundColor', color)}
                      className={`w-7 h-7 rounded-full border-2 transition-transform hover:scale-110 ${
                        vizConfig.backgroundColor === color ? (theme === 'dark' ? 'border-[#7fa66b] scale-110' : 'border-[#4e7a41] scale-110') : 'border-transparent'
                      }`}
                      style={{ backgroundColor: color }}
                      title={color}
                    />
                  ))}
                  <label className={`relative w-7 h-7 rounded-full overflow-hidden border-2 ${ct.chip} cursor-pointer`}>
                    <input
                      type="color"
                      value={vizConfig.backgroundColor}
                      onChange={e => updateConfig('backgroundColor', e.target.value)}
                      className="absolute -top-2 -left-2 w-12 h-12 cursor-pointer p-0 border-0"
                      title="Custom background"
                    />
                  </label>
                </div>
              </section>
            </>
          )}

          {/* ============================== ELEMENTS TAB */}
          {activeTab === 'elements' && (
            <>
              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>Element types in structure</h3>
                {atomTypeList.length > 0 ? (
                  <div className="space-y-1.5">
                    {atomTypeList.map(typeInfo => {
                      const meta = ELEMENT_DATA.find(el => el.symbol === typeInfo.element);
                      return (
                        <div key={typeInfo.id} className={`flex items-center justify-between p-2.5 rounded-lg border ${ct.card}`}>
                          <div className="flex items-center gap-2.5">
                            <div className="relative w-8 h-8 rounded-full overflow-hidden border border-gray-600 shrink-0">
                              <input
                                type="color"
                                value={vizConfig.customColors[typeInfo.id] || DEFAULT_ATOM_COLOR}
                                onChange={e => updateCustomColor(typeInfo.id, e.target.value)}
                                className="absolute -top-2 -left-2 w-14 h-14 cursor-pointer p-0 border-0"
                                title={`Pick color for type ${typeInfo.id}`}
                              />
                            </div>
                            <div>
                              <div className="text-[11px] font-bold">
                                {meta ? `${meta.name} (${meta.symbol})` : typeInfo.label}
                              </div>
                              <div className={`text-[9px] ${ct.muted}`}>
                                {meta
                                  ? `Z=${meta.number} · ${meta.mass.toFixed(2)} u`
                                  : `ID ${typeInfo.id}${typeInfo.mass ? ` · ${typeInfo.mass.toFixed(2)} u` : ''}`}
                              </div>
                            </div>
                          </div>
                          <div className={`px-2 py-0.5 rounded text-[10px] font-mono ${ct.chip}`}>
                            ×{typeInfo.count}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className={`text-xs italic ${ct.muted}`}>Load a structure to edit its element colors.</p>
                )}
              </section>

              <section className="space-y-2">
                <h3 className={`text-xs font-semibold ${ct.header}`}>CPK/Jmol reference</h3>
                <p className={`text-[10px] ${ct.muted}`}>
                  All 118 elements carry standard CPK/Jmol colors and are resolved by symbol from any
                  supported format — LAMMPS mass tables, XYZ symbols, PDB element columns and CIF type symbols.
                </p>
              </section>
            </>
          )}

          {/* ============================== ANALYSIS TAB */}
          {activeTab === 'analysis' && (
            <>
              {!moleculeData || !moleculeData.frames || moleculeData.frames.length <= 1 ? (
                <section className={`rounded-xl p-4 text-sm border shadow-sm ${ct.card} ${ct.muted} leading-relaxed`}>
                  <div className={`flex items-center gap-2 mb-2 font-bold ${ct.accentText}`}>
                    <BarChart3 size={15} /> Trajectory analysis
                  </div>
                  <p className="leading-relaxed">
                    Load a <span className="font-semibold">LAMMPS dump</span> (`.lammpstrj`/`.dump`) or multi-frame <span className="font-semibold">XYZ</span> trajectory to unlock analysis.
                    Try the bundled <em>Trajectory · Dump</em> or <em>Trajectory · XYZ</em> examples in the Data tab.
                  </p>
                  <p className={`mt-2 text-xs ${ct.muted}`}>
                    Once loaded, this tab shows RDF, MSD, density profiles and velocity histograms — all computed locally, with CSV export.
                  </p>
                </section>
              ) : (
                <>
                  {/* Stats */}
                  {(() => {
                    const stats = trajStats(moleculeData);
                    const is2D = stats.box ? (stats.box.zhi - stats.box.zlo) < 2 : false;
                    return (
                      <section className={`rounded-xl p-3 border ${ct.card} space-y-2`}>
                        <h3 className={`text-xs font-bold flex items-center gap-1.5 ${ct.header}`}>
                          <Activity size={13} className={ct.accentText} /> Trajectory stats
                        </h3>
                        <div className="grid grid-cols-3 gap-2 text-center">
                          {[
                            { label: 'Frames', value: stats.frames },
                            { label: 'Atoms', value: stats.atoms },
                            { label: is2D ? 'Area ρ' : 'Density', value: (is2D ? stats.areaDensity : stats.density)?.toFixed(3) ?? '—' },
                          ].map(s => (
                            <div key={s.label} className={`p-2 rounded-lg ${ct.stat}`}>
                              <div className={`text-[10px] ${ct.muted}`}>{s.label}</div>
                              <div className="text-sm font-mono font-bold">{s.value}</div>
                            </div>
                          ))}
                        </div>
                        {stats.box && (
                          <div className={`text-[10px] font-mono ${ct.muted} leading-relaxed`}>
                            Box {stats.box.xlo.toFixed(1)}→{stats.box.xhi.toFixed(1)} × {stats.box.ylo.toFixed(1)}→{stats.box.yhi.toFixed(1)} × {stats.box.zlo.toFixed(1)}→{stats.box.zhi.toFixed(1)}
                            {is2D && ' · 2D (thin z)'}
                          </div>
                        )}
                      </section>
                    );
                  })()}

                  {/* RDF */}
                  <section className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className={`text-xs font-semibold flex items-center gap-1.5 ${ct.header}`}>
                        <TrendingUp size={12} className={ct.accentText} /> Radial distribution g(r)
                      </h3>
                      <button
                        onClick={() => {
                          const frames = moleculeData.frames!.slice(0, Math.min(20, moleculeData.frames!.length));
                          const pts = computeRDF(frames, moleculeData.box, { rMax: 10, bins: 80 });
                          const csv = 'r,g(r),count\n' + pts.map(p => `${p.r.toFixed(3)},${p.g.toFixed(4)},${p.count.toFixed(1)}`).join('\n');
                          downloadTextFile('rdf.csv', csv);
                        }}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium ${ct.button}`}
                        title="Export RDF as CSV"
                      >
                        <Download size={11} /> CSV
                      </button>
                    </div>
                    {(() => {
                      const frames = moleculeData.frames!.filter((_, i) => i % Math.ceil(moleculeData.frames!.length / 15) === 0);
                      const pts = computeRDF(frames, moleculeData.box, { rMax: 10, bins: 80 });
                      const data = pts.map(p => ({ x: p.r, y: p.g }));
                      const hasPeaks = data.some(d => d.y > 1.5);
                      return (
                        <>
                          <div className={`rounded-lg border p-2 ${ct.card}`}>
                            <LineChart data={data} xLabel="r (Å / LJ σ)" yLabel="g(r)" theme={theme} height={150} yMin={0} />
                          </div>
                          <p className={`text-[10px] leading-relaxed ${ct.muted}`}>
                            {hasPeaks ? 'Peaks indicate local order (crystal). Flat ~1 = ideal gas / liquid.' : 'Flat g(r)≈1 — disordered / ideal gas.'}
                            {' '}Averaged over {frames.length} sampled frames. First peak ≈ nearest-neighbour.
                          </p>
                        </>
                      );
                    })()}
                  </section>

                  {/* MSD */}
                  <section className="space-y-2">
                    <div className="flex items-center justify-between">
                      <h3 className={`text-xs font-semibold flex items-center gap-1.5 ${ct.header}`}>
                        <Activity size={12} className={ct.accentText} /> Mean squared displacement
                      </h3>
                      <button
                        onClick={() => {
                          const pts = computeMSD(moleculeData.frames!, moleculeData.box);
                          const csv = 't,msd\n' + pts.map(p => `${p.t},${p.msd.toFixed(4)}`).join('\n');
                          downloadTextFile('msd.csv', csv);
                        }}
                        className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium ${ct.button}`}
                        title="Export MSD as CSV"
                      >
                        <Download size={11} /> CSV
                      </button>
                    </div>
                    {(() => {
                      const pts = computeMSD(moleculeData.frames!, moleculeData.box, { timeOriginStride: Math.max(1, Math.floor(moleculeData.frames!.length / 15)) });
                      const data = pts.map(p => ({ x: p.t, y: p.msd }));
                      const last = data[data.length - 1];
                      const slope = last && last.x > 0 ? (last.y / last.x).toFixed(3) : '—';
                      return (
                        <>
                          <div className={`rounded-lg border p-2 ${ct.card}`}>
                            <LineChart data={data} xLabel="lag (frames)" yLabel="MSD (Å²)" theme={theme} height={150} yMin={0} color={theme === 'dark' ? '#d9a05b' : '#b97f3e'} />
                          </div>
                          <p className={`text-[10px] leading-relaxed ${ct.muted}`}>
                            Slope ≈ {slope} Å²/frame — linear = diffusive, plateau = caged/crystal. Averaged over time origins.
                          </p>
                        </>
                      );
                    })()}
                  </section>

                  {/* Density profile */}
                  <section className="space-y-2">
                    <h3 className={`text-xs font-semibold flex items-center gap-1.5 ${ct.header}`}>
                      <BarChart3 size={12} className={ct.accentText} /> Density profile
                    </h3>
                    {(() => {
                      const axis: 'x' | 'y' | 'z' = (moleculeData.box && (moleculeData.box.zhi - moleculeData.box.zlo) < 2) ? 'x' : 'y';
                      const prof = computeDensityProfile(moleculeData.frames!, moleculeData.box, axis, 24);
                      return (
                        <>
                          <div className={`rounded-lg border p-2 ${ct.card}`}>
                            <Histogram bins={prof.bins} xLabel={`${axis} (Å)`} yLabel="count" theme={theme} height={140} />
                          </div>
                          <p className={`text-[10px] ${ct.muted}`}>Histogram of atom counts along <span className="font-mono">{axis}</span> (averaged over all frames) — uniform = homogeneous, peaks = layering.</p>
                        </>
                      );
                    })()}
                  </section>

                  {/* Velocity distribution */}
                  {moleculeData.atoms.some(a => a.vx !== undefined) && (
                    <section className="space-y-2">
                      <h3 className={`text-xs font-semibold flex items-center gap-1.5 ${ct.header}`}>
                        <Sparkles size={12} className={ct.accentText} /> Speed distribution
                      </h3>
                      {(() => {
                        const bins = computeSpeedDistribution(moleculeData.frames ? moleculeData.frames[frameIdx]?.atoms ?? moleculeData.atoms : moleculeData.atoms, 24);
                        if (!bins) return <p className={`text-xs italic ${ct.muted}`}>No velocities in current frame.</p>;
                        return (
                          <>
                            <div className={`rounded-lg border p-2 ${ct.card}`}>
                              <Histogram bins={bins} xLabel="|v| (LJ)" yLabel="count" theme={theme} height={140} color={theme === 'dark' ? '#c9a9d4' : '#7d5a8c'} />
                            </div>
                            <p className={`text-[10px] ${ct.muted}`}>Current frame speed |v| — Maxwell–Boltzmann peak shifts with temperature. Requires dump with vx vy vz.</p>
                          </>
                        );
                      })()}
                    </section>
                  )}

                  {/* Visuals helper */}
                  <section className={`rounded-lg border p-3 space-y-2 ${ct.card}`}>
                    <h3 className={`text-xs font-semibold flex items-center gap-1.5 ${ct.header}`}>
                      <Palette size={12} className={ct.accentText} /> Trajectory visuals
                    </h3>
                    <p className={`text-[10px] leading-relaxed ${ct.muted}`}>
                      Tip: use <span className="font-semibold">View → Atom size</span> and <span className="font-semibold">Scene → Box / Axes</span> while scrubbing.
                      For 2D LJ runs the box is thin (z≈1); enable <span className="font-semibold">Box</span> to see it, and use <span className="font-semibold">Top</span> camera for a clear 2D view. Video `● Rec` captures the live canvas at 60 fps.
                    </p>
                  </section>
                </>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <footer className={`px-3 py-2.5 border-t text-[10px] text-center flex items-center justify-between ${ct.divider}`}>
          <span className={ct.muted}>
            Created by <span className="font-medium">Shuvam Banerji Seal</span>
          </span>
          <button
            onClick={copyShareLink}
            className={`flex items-center gap-1 px-2 py-1 rounded ${linkCopied ? ct.accentText : ct.button}`}
            title="Copy a link that restores this exact view"
          >
            {linkCopied ? <Check size={12} /> : <Link2 size={12} />}
            {linkCopied ? 'Copied' : 'Share'}
          </button>
          <button
            onClick={() => setShowHelp(true)}
            className={`flex items-center gap-1 px-2 py-1 rounded ${ct.button}`}
            title="Keyboard shortcuts (H)"
          >
            <Keyboard size={12} /> Shortcuts
          </button>
        </footer>
      </aside>

      {/* ============================ MAIN CANVAS AREA */}
      <main className={`flex-1 relative min-w-0 ${ct.bg}`}>
        {/* Top-left controls */}
        <div className="absolute top-3 left-3 right-3 z-10 flex items-start justify-between pointer-events-none">
          <div className="pointer-events-auto flex gap-1 sm:gap-2">
            {!isSidebarOpen && (
              <button
                onClick={() => setIsSidebarOpen(true)}
                className={`rounded-lg p-1.5 shadow-lg backdrop-blur sm:p-2.5 ${ct.sidebar}`}
                title="Open sidebar (O)"
              >
                <Menu size={18} />
              </button>
            )}
            {moleculeData && (
              <div className={`rounded-lg px-2 py-1.5 text-[10px] shadow-lg backdrop-blur hidden sm:block sm:px-3 sm:py-2 sm:text-[11px] font-mono ${ct.card}`}>
                {moleculeData.atoms.length.toLocaleString()} atoms
                {moleculeData.bonds.length > 0 && ` · ${moleculeData.bonds.length.toLocaleString()} bonds`}
                {moleculeData.box && ' · cell'}
                {frameCount > 1 && ` · ${frameCount} frames`}
              </div>
            )}
            {isParsing && (
              <div className={`flex items-center gap-1.5 px-3 py-2 rounded-lg shadow-lg text-[11px] backdrop-blur ${ct.card}`} role="status">
                <Loader2 size={13} className={`animate-spin ${ct.accentText}`} />
                Parsing structure…
              </div>
            )}
          </div>
          <div className="pointer-events-auto flex gap-1 sm:gap-2">
            <button
              onClick={() => emitCameraCommand({ type: 'zoom', delta: 1 })}
              className={`rounded-lg p-1.5 shadow-lg backdrop-blur sm:p-2.5 ${ct.sidebar}`}
              title="Zoom in (+)"
            >
              <ZoomIn size={18} />
            </button>
            <button
              onClick={() => emitCameraCommand({ type: 'zoom', delta: -1 })}
              className={`rounded-lg p-1.5 shadow-lg backdrop-blur sm:p-2.5 ${ct.sidebar}`}
              title="Zoom out (−)"
            >
              <ZoomOut size={18} />
            </button>
            <button
              onClick={() => setShowHelp(true)}
              className={`rounded-lg p-1.5 shadow-lg backdrop-blur sm:p-2.5 ${ct.sidebar}`}
              title="Keyboard shortcuts (H)"
            >
              <HelpCircle size={18} />
            </button>
          </div>
        </div>

        {/* Bottom toolbar */}
        <div className={`absolute bottom-2 left-1/2 z-10 flex -translate-x-1/2 items-center gap-0.5 rounded-full border px-1 py-1 shadow-2xl backdrop-blur sm:bottom-4 sm:gap-1 sm:px-2 sm:py-1.5 ${
          theme === 'dark' ? 'bg-[#1e1913]/95 border-[#3f3526]' : 'bg-white/95 border-[#ddd2bd]'
        }`}>
          <button
            onClick={() => setAutoRotate(v => !v)}
            className={`flex items-center gap-1 px-2 py-1.5 rounded-full text-xs font-medium transition-colors sm:gap-1.5 sm:px-3 sm:py-2 ${
              autoRotate ? ct.accentText : `${ct.muted}`
            }`}
            title="Auto-rotate (Space)"
          >
            {autoRotate ? <Pause size={16} /> : <Play size={16} />}
            <span className="hidden sm:inline">Rotate</span>
          </button>
          <div className={`w-px h-5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={() => emitCameraCommand({ type: 'fit' })}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium ${ct.muted}`}
            title="Fit view (R)"
          >
            <Maximize2 size={16} />
            <span className="hidden sm:inline">Fit</span>
          </button>
          <div className={`w-px h-5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={() => updateConfig('showBox', !vizConfig.showBox)}
            disabled={!moleculeData?.box}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium disabled:opacity-30 ${
              vizConfig.showBox ? ct.accentText : ct.muted
            }`}
            title="Simulation box (X)"
          >
            <Box size={16} />
            <span className="hidden sm:inline">Box</span>
          </button>
          <div className={`w-px h-5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={() => updateConfig('showLabels', !vizConfig.showLabels)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium ${
              vizConfig.showLabels ? ct.accentText : ct.muted
            }`}
            title="Element labels (L)"
          >
            <Layers size={16} />
            <span className="hidden sm:inline">Labels</span>
          </button>
          <div className={`w-px h-5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={() => (isRecording ? stopRecording() : startRecording())}
            disabled={savingVideo}
            className={`flex items-center gap-1 px-2 py-1.5 rounded-full text-xs font-medium transition-colors sm:gap-1.5 sm:px-3 sm:py-2 ${
              isRecording ? 'text-red-400' : savingVideo ? 'opacity-50' : ct.muted
            }`}
            title={isRecording ? 'Stop recording & save video' : 'Record high-quality video (MP4 where supported)'}
          >
            {isRecording ? (
              <>
                <span className="w-3 h-3 rounded-sm bg-red-500 animate-pulse" />
                <span className="font-mono tabular-nums hidden sm:inline">
                  {String(Math.floor(recordingMs / 60000)).padStart(2, '0')}:
                  {String(Math.floor((recordingMs % 60000) / 1000)).padStart(2, '0')}
                </span>
                <span className="sm:hidden">Stop</span>
              </>
            ) : (
              <>
                <Circle size={13} className="text-red-400" fill="currentColor" />
                <span className="hidden sm:inline">{savingVideo ? 'Saving…' : 'Rec'}</span>
              </>
            )}
          </button>
          <div className={`w-px h-5 ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-300'}`} />
          <button
            onClick={doScreenshot}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium ${ct.muted}`}
            title="Screenshot (S)"
          >
            <Camera size={16} />
            <span className="hidden sm:inline">Shot</span>
          </button>
        </div>

        {/* Trajectory playback bar (P5) */}
        {frameCount > 1 && (
          <div className={`absolute bottom-14 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1 rounded-full border px-2 py-1 shadow-2xl backdrop-blur sm:bottom-16 sm:gap-2 sm:px-3 sm:py-1.5 ${
            theme === 'dark' ? 'bg-[#1e1913]/95 border-[#3f3526]' : 'bg-white/95 border-[#ddd2bd]'
          }`}>
            <button
              onClick={() => setFrameIdx(i => (i - 1 + frameCount) % frameCount)}
              className={`p-1.5 rounded-full ${ct.button}`}
              title="Previous frame (,)"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              onClick={() => setTrajPlaying(v => !v)}
              className={`p-1.5 rounded-full ${trajPlaying ? ct.accentText : ct.muted}`}
              title="Play / pause trajectory (P)"
            >
              {trajPlaying ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              onClick={() => setFrameIdx(i => (i + 1) % frameCount)}
              className={`p-1.5 rounded-full ${ct.button}`}
              title="Next frame (.)"
            >
              <ChevronRight size={14} />
            </button>
            <input
              type="range"
              min="0"
              max={frameCount - 1}
              value={Math.min(frameIdx, frameCount - 1)}
              onChange={e => { setTrajPlaying(false); setFrameIdx(parseInt(e.target.value, 10)); }}
              className={`w-32 sm:w-48 ${theme === 'dark' ? "accent-[#7fa66b]" : "accent-[#4e7a41]"}`}
              aria-label="Trajectory frame"
            />
            <span className={`text-[10px] font-mono tabular-nums ${ct.muted}`}>
              {Math.min(frameIdx, frameCount - 1) + 1}/{frameCount}
            </span>
            <select
              value={trajFps}
              onChange={e => setTrajFps(parseInt(e.target.value, 10))}
              className={`text-[10px] rounded border bg-transparent ${ct.input} py-0.5`}
              title="Playback speed"
            >
              {[2, 5, 10, 30].map(f => <option key={f} value={f}>{f} fps</option>)}
            </select>
          </div>
        )}

        {/* Measurement panel (P4) */}
        {selectedIds.length > 0 && activeData && (
          <div className={`absolute top-16 left-3 z-10 px-3 py-2.5 rounded-xl border shadow-xl backdrop-blur text-xs space-y-1.5 ${
            theme === 'dark' ? 'bg-[#1e1913]/95 border-[#3f3526]' : 'bg-white/95 border-[#ddd2bd]'
          }`} role="status">
            <div className="flex items-center gap-1.5 font-semibold">
              <Ruler size={13} className={ct.accentText} /> Measurement
            </div>
            {measurement ? (
              <div className="font-mono text-sm">
                <span className={ct.muted}>{measurementGlyph(measurement.kind)} = </span>
                <span className={`${ct.accentText} font-bold`}>{measurement.label}</span>
              </div>
            ) : (
              <div className={ct.muted}>{measurementHint}</div>
            )}
            <button
              onClick={() => setSelectedIds([])}
              className={`w-full py-1 rounded-lg text-[11px] font-medium ${ct.button}`}
              title="Clear selection (C)"
            >
              Clear selection
            </button>
          </div>
        )}

        {/* Hint for first-time users */}
        {moleculeData && frameCount <= 1 && selectedIds.length === 0 && (
          <div className={`absolute bottom-14 left-1/2 z-[5] hidden -translate-x-1/2 rounded-full px-3 py-1 text-[10px] sm:bottom-20 sm:block ${ct.muted}`}>
            Press <kbd className={`px-1 rounded ${ct.chip}`}>H</kbd> for keyboard shortcuts · drag & drop files anywhere
          </div>
        )}

        {activeData ? (
          <MoleculeCanvas
            data={activeData}
            autoRotate={autoRotate}
            config={vizConfig}
            selectedIds={selectedIds}
            onSelectAtom={toggleSelectAtom}
            forceContinuousRender={isRecording || savingVideo}
          />
        ) : (
          <div className={`w-full h-full flex flex-col items-center justify-center ${ct.muted}`}>
            <div className="w-14 h-14 border-4 rounded-full animate-spin mb-4 ${ct.loader}" />
            <p>Waiting for structure data…</p>
          </div>
        )}
      </main>

      {/* ============================ HELP OVERLAY */}
      {showHelp && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => setShowHelp(false)}
          role="dialog"
          aria-modal="true"
          aria-label="Keyboard shortcuts"
        >
          <div
            className={`max-w-lg w-full rounded-2xl border shadow-2xl p-6 space-y-4 max-h-[85vh] overflow-y-auto ${ct.card}`}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold flex items-center gap-2">
                <Keyboard size={18} /> Keyboard shortcuts
              </h2>
              <button onClick={() => setShowHelp(false)} className={`p-1.5 rounded-lg ${ct.button}`}>
                <X size={16} />
              </button>
            </div>
            <ul className="divide-y divide-transparent">
              {SHORTCUT_CATALOG.map(sc => (
                <li key={sc.keys} className="flex items-center justify-between py-1.5 text-xs">
                  <kbd className={`px-2 py-1 rounded-md font-mono text-[11px] ${ct.chip}`}>{sc.keys}</kbd>
                  <span className={`${ct.muted} text-right`}>{sc.action}</span>
                </li>
              ))}
            </ul>
            <p className={`text-[10px] ${ct.muted} flex items-center gap-1.5`}>
              <Settings size={11} /> Shortcuts are ignored while typing in the paste box.
            </p>
            <div className="flex items-center gap-2">
              <button onClick={() => { setShowHelp(false); doScreenshot(); }} className={`flex-1 py-2 rounded-lg text-xs font-semibold ${ct.accent}`}>
                <RotateCw size={12} className="inline mr-1" /> Take screenshot now
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ViewerModule;
