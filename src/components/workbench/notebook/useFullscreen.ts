import { useCallback, useEffect, useRef, useState, type RefObject } from 'react';

/**
 * Full-screen state of one element: 'native' when the Fullscreen API put it on screen, 'overlay' when
 * the page falls back to a CSS overlay (no Fullscreen API, or the request was refused, as on iPhone Safari),
 * null when it is not full screen.
 */
export type FullscreenMode = 'native' | 'overlay' | null;

export interface Fullscreen {
  active: boolean;
  mode: FullscreenMode;
  toggle: () => void;
  exit: () => void;
}

/**
 * Full-screen mode for the element behind `ref`. Uses element.requestFullscreen() and keeps `mode` in sync with
 * the fullscreenchange event; falls back to an overlay (the caller styles it) when the API is missing or refused.
 * Escape leaves the overlay (a native full-screen page handles Escape itself).
 */
export const useFullscreen = (ref: RefObject<HTMLElement | null>): Fullscreen => {
  const [mode, setMode] = useState<FullscreenMode>(null);
  const modeRef = useRef<FullscreenMode>(null);
  modeRef.current = mode;

  const exit = useCallback(() => {
    if (typeof document !== 'undefined' && document.fullscreenElement && typeof document.exitFullscreen === 'function') {
      void document.exitFullscreen().catch(() => {
        // already left (the browser exited on its own); the state follows fullscreenchange
      });
    }
    setMode(null);
  }, []);

  const enter = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const canNative = typeof document !== 'undefined'
      && document.fullscreenEnabled !== false
      && typeof el.requestFullscreen === 'function';
    if (!canNative) {
      setMode('overlay');
      return;
    }
    let request: Promise<void>;
    try {
      request = el.requestFullscreen();
    } catch {
      setMode('overlay');
      return;
    }
    request.then(
      () => setMode('native'),
      () => setMode('overlay'),
    );
  }, [ref]);

  const toggle = useCallback(() => {
    if (modeRef.current !== null) exit();
    else enter();
  }, [enter, exit]);

  // The browser reports entering and leaving native full screen; an overlay is not in the Fullscreen API.
  useEffect(() => {
    const onChange = () => {
      const el = ref.current;
      const native = el !== null && document.fullscreenElement === el;
      if (native) setMode('native');
      else setMode((prev) => (prev === 'native' ? null : prev));
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, [ref]);

  // Escape closes the overlay fallback.
  useEffect(() => {
    if (mode !== 'overlay') return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') exit();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mode, exit]);

  // Leaving the view (unmount) leaves native full screen too.
  useEffect(() => () => {
    const el = ref.current;
    if (el && typeof document !== 'undefined' && document.fullscreenElement === el && typeof document.exitFullscreen === 'function') {
      void document.exitFullscreen().catch(() => {});
    }
  }, [ref]);

  return { active: mode !== null, mode, toggle, exit };
};

export default useFullscreen;
