import { useEffect, useRef } from 'react';

export interface ShortcutActionMap {
  onToggleRotate: () => void;
  onFit: () => void;
  onViewPreset: (preset: 'front' | 'back' | 'left' | 'right' | 'top' | 'bottom' | 'iso') => void;
  onZoom: (delta: 1 | -1) => void;
  onOrbit: (dx: number, dy: number) => void;
  onVisualizationMode: (mode: 0 | 1 | 2 | 3) => void;
  onToggleBonds: () => void;
  onToggleBox: () => void;
  onToggleLabels: () => void;
  onCycleLighting: () => void;
  onToggleTheme: () => void;
  onToggleSidebar: () => void;
  onScreenshot: () => void;
  onHelp: () => void;
  onEscape: () => void;
}

const isTypingTarget = (target: EventTarget | null): boolean => {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable;
};

/**
 * Global keyboard shortcut layer.
 * Ignored while typing in inputs/textareas. Single-letter keys only fire
 * without modifier keys so browser shortcuts (Ctrl+R etc.) pass through.
 */
export const useKeyboardShortcuts = (actions: ShortcutActionMap): void => {
  const actionsRef = useRef(actions);
  actionsRef.current = actions;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      const a = actionsRef.current;
      switch (e.key) {
        case ' ':
          e.preventDefault();
          a.onToggleRotate();
          break;
        case 'r':
        case 'R':
          e.preventDefault();
          a.onFit();
          break;
        case '1': a.onVisualizationMode(0); break;
        case '2': a.onVisualizationMode(1); break;
        case '3': a.onVisualizationMode(2); break;
        case '4': a.onVisualizationMode(3); break;
        case 'b':
        case 'B':
          a.onToggleBonds();
          break;
        case 'x':
        case 'X':
          a.onToggleBox();
          break;
        case 'l':
        case 'L':
          a.onToggleLabels();
          break;
        case 'g':
        case 'G':
          a.onCycleLighting();
          break;
        case 't':
        case 'T':
          a.onToggleTheme();
          break;
        case 'o':
        case 'O':
          a.onToggleSidebar();
          break;
        case 's':
        case 'S':
          e.preventDefault();
          a.onScreenshot();
          break;
        case '?':
        case 'h':
        case 'H':
          a.onHelp();
          break;
        case '+':
        case '=':
          a.onZoom(1);
          break;
        case '-':
        case '_':
          a.onZoom(-1);
          break;
        case 'ArrowLeft':
          e.preventDefault();
          a.onOrbit(-0.12, 0);
          break;
        case 'ArrowRight':
          e.preventDefault();
          a.onOrbit(0.12, 0);
          break;
        case 'ArrowUp':
          e.preventDefault();
          a.onOrbit(0, -0.12);
          break;
        case 'ArrowDown':
          e.preventDefault();
          a.onOrbit(0, 0.12);
          break;
        case '7': a.onViewPreset('front'); break;
        case '8': a.onViewPreset('top'); break;
        case '9': a.onViewPreset('right'); break;
        case '5': a.onViewPreset('iso'); break;
        case 'Escape':
          a.onEscape();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);
};

/** Static shortcut catalogue rendered by the help overlay and README/wiki. */
export const SHORTCUT_CATALOG: { keys: string; action: string }[] = [
  { keys: 'Space', action: 'Play / pause auto-rotation' },
  { keys: 'R', action: 'Reset & fit view' },
  { keys: '1 / 2 / 3 / 4', action: 'Ball & Stick / Space Fill / Wireframe / Licorice' },
  { keys: '7 / 8 / 9 / 5', action: 'Camera: front / top / right / isometric' },
  { keys: '← → ↑ ↓', action: 'Orbit the camera' },
  { keys: '+ / −', action: 'Zoom in / out' },
  { keys: 'B', action: 'Toggle bonds' },
  { keys: 'X', action: 'Toggle simulation box' },
  { keys: 'L', action: 'Toggle element labels (≤400 atoms)' },
  { keys: 'G', action: 'Cycle lighting preset' },
  { keys: 'T', action: 'Toggle dark / light theme' },
  { keys: 'O', action: 'Toggle sidebar' },
  { keys: 'S', action: 'Save PNG screenshot' },
  { keys: 'H or ?', action: 'Show this help' },
  { keys: 'Esc', action: 'Close panels / overlays' },
];
