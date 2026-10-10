import React, { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import * as THREE from 'three';
import { Camera, Film, Images, Undo2, X } from 'lucide-react';
import type { getThemeTokens } from '../../../theme';
import { dataUrlBytes, downloadFile } from '../../../lammps/exporter';
import { thermoUnit } from '../../../lammps/thermoUnits';
import { captureActiveCanvas, getActiveGL } from '../../../services/glRegistry';
import { emitCameraCommand } from '../../../services/cameraBus';
import { recordingFormats, startCanvasRecording, type RecordingFormat, type RecordingHandle } from '../../../services/recorder';
import type { TrajectoryBuffer } from './trajectory';
import {
  QUALITY_LABEL, RESOLUTION_LABEL, bitrateFor, framesZip, orbitRadians, overlayText, repeatIndices, resolveSize,
  selectFrames, type ResolutionPreset, type VideoQuality,
} from './renderFrames';

type Tokens = ReturnType<typeof getThemeTokens>;

const VIDEO_FPS = [10, 15, 25, 30, 60] as const;
const RESOLUTIONS: ResolutionPreset[] = ['current', '720p', '1080p'];
const QUALITIES: VideoQuality[] = ['low', 'medium', 'high'];
const MAX_LOOPS = 20;

const nextFrame = (): Promise<void> =>
  new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve())));

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const stamp = (): string => new Date().toISOString().replace(/[:.]/g, '-');

const toNum = (s: string): number => (s.trim() === '' ? NaN : Number(s));

const saveBytes = (name: string, bytes: Uint8Array<ArrayBuffer>, mime: string): void => {
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

/** The step and time caption drawn in the bottom-left corner of a frame. */
const drawOverlay = (ctx: CanvasRenderingContext2D, width: number, height: number, text: string): void => {
  const px = Math.max(12, Math.round(height * 0.04));
  const pad = Math.round(px * 0.5);
  const boxH = Math.round(px * 1.6);
  ctx.font = `${px}px ui-monospace, Menlo, Consolas, monospace`;
  ctx.textBaseline = 'alphabetic';
  const boxW = Math.ceil(ctx.measureText(text).width) + pad * 2;
  const boxX = pad;
  const boxY = height - pad - boxH;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(boxX, boxY, boxW, boxH);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, boxX + pad, boxY + boxH - Math.round(px * 0.45));
};

interface SequenceOptions {
  /** 0-based buffer indices, in output order. */
  indices: number[];
  /** Output size, or null for the view's current drawing size. */
  size: { width: number; height: number } | null;
  /** The composed output canvas (already sized). */
  out: HTMLCanvasElement;
  overlay: { dt: number | null; units: string | undefined; timeUnit: string | null } | null;
  orbitDeg: number;
  /** Called once per output frame, after it is drawn into `out`. */
  onFrame: (position: number, index: number) => Promise<void>;
}

export interface RenderPanelProps {
  ct: Tokens;
  btn: string;
  buffer: TrajectoryBuffer;
  /** Changes whenever the buffer changes. */
  version: number;
  /** The frame the view shows (null follows the newest); restored after a render. */
  index: number | null;
  /** The last run's timestep and units (time in the overlay); null when no run has a timestep. */
  dt: number | null;
  units: string | undefined;
  /** A run is going: rendering waits for it. */
  live: boolean;
  /** A dump file loaded for rendering (its name), replacing the run's recorded frames in the buffer. */
  source: { name: string } | null;
  /** The run's own frames are set aside while a file is loaded: "Back to the run's frames" puts them back. */
  canRestore: boolean;
  onRestore: () => void;
  /** Error from loading a file for rendering. */
  notice: string | null;
  canvasHost: React.RefObject<HTMLDivElement | null>;
  /** Shows recorded frame i (null follows the newest). */
  onShow: (index: number | null) => void;
  /** True while frames are rendered: the 3D view keeps drawing. */
  onVideoActive: (on: boolean) => void;
  onClose: () => void;
}

/**
 * Render panel of the notebook: snapshot of the view, frames of a recorded range as a zip of PNGs, and a video
 * of the range (format, fps, quality, resolution, loops), with an optional step and time overlay and a slow
 * orbit. Each frame is shown, drawn, and only then captured; the camera and the shown frame are restored after.
 */
const RenderPanel: React.FC<RenderPanelProps> = ({
  ct, btn, buffer, version, index, dt, units, live, source, canRestore, onRestore, notice,
  canvasHost, onShow, onVideoActive, onClose,
}) => {
  const count = buffer.length;
  const formats = useMemo<RecordingFormat[]>(
    () => (typeof MediaRecorder === 'undefined' ? [] : recordingFormats((m) => MediaRecorder.isTypeSupported(m))),
    [],
  );
  const [startS, setStartS] = useState('1');
  const [endS, setEndS] = useState('');
  const [strideS, setStrideS] = useState('1');
  const [resolution, setResolution] = useState<ResolutionPreset>('current');
  const [overlay, setOverlay] = useState(true);
  const [orbitS, setOrbitS] = useState('0');
  const [loopsS, setLoopsS] = useState('1');
  const [formatIdx, setFormatIdx] = useState(0);
  const [fps, setFps] = useState<number>(25);
  const [quality, setQuality] = useState<VideoQuality>('high');
  const [progress, setProgress] = useState<{ kind: 'frames' | 'video'; done: number; total: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const cancelRef = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const busy = progress !== null;

  useEffect(() => { heading.current?.focus(); }, []);

  const endValue = endS.trim() === '' ? count : toNum(endS);
  const selection = selectFrames(count, { start: toNum(startS), end: endValue, stride: toNum(strideS) });
  const orbitDeg = Number(orbitS);
  const loops = Math.min(MAX_LOOPS, Math.max(1, Math.floor(toNum(loopsS) || 1)));
  const format = formats[formatIdx] ?? null;
  const blocked = busy || live || count === 0;

  /** The view's renderer and camera, if the registered 3D view is the one in this notebook. */
  const liveView = () => {
    const gl = getActiveGL();
    const host = canvasHost.current;
    return gl && host && host.contains(gl.gl.domElement) ? gl : null;
  };

  /**
   * Draws the selected frames in order into `out`: each frame is shown, the view is re-rendered explicitly and
   * the result copied into `out` in the same task, so the captured pixels are exactly that frame.
   * Restores the size, the camera pose and the shown frame afterwards, also after an error or a cancel.
   */
  const runSequence = async (opts: SequenceOptions): Promise<'done' | 'cancelled'> => {
    const view = liveView();
    if (!view) throw new Error('No 3D view to render yet.');
    const renderer = view.gl;
    const camera = view.camera;
    const perspective = camera instanceof THREE.PerspectiveCamera ? camera : null;
    const restoreIndex = index;
    const css = renderer.getSize(new THREE.Vector2());
    const pixelRatio = renderer.getPixelRatio();
    const aspect = perspective?.aspect ?? 1;
    const pose = camera.position.clone();
    const ctx = opts.out.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D is not available in this browser.');
    const orbit = orbitRadians(opts.orbitDeg);
    let outcome: 'done' | 'cancelled' = 'done';
    try {
      if (opts.size) {
        renderer.setPixelRatio(1);
        renderer.setSize(opts.size.width, opts.size.height, false);
        if (perspective) {
          perspective.aspect = opts.size.width / opts.size.height;
          perspective.updateProjectionMatrix();
        }
      }
      const W = opts.out.width;
      const H = opts.out.height;
      for (let k = 0; k < opts.indices.length; k++) {
        if (cancelRef.current) { outcome = 'cancelled'; break; }
        const idx = opts.indices[k];
        const rec = buffer.get(idx);
        flushSync(() => onShow(idx));
        await nextFrame();
        if (k > 0 && orbit > 0) emitCameraCommand({ type: 'orbit', dx: orbit, dy: 0 });
        renderer.render(view.scene, camera);
        ctx.drawImage(renderer.domElement, 0, 0, W, H);
        if (opts.overlay && rec) {
          drawOverlay(ctx, W, H, overlayText(rec.step, opts.overlay.dt, opts.overlay.units, opts.overlay.timeUnit));
        }
        await opts.onFrame(idx + 1, idx);
      }
    } finally {
      emitCameraCommand({ type: 'pose', position: [pose.x, pose.y, pose.z] });
      if (opts.size) {
        renderer.setPixelRatio(pixelRatio);
        renderer.setSize(css.x, css.y);
        if (perspective) {
          perspective.aspect = aspect;
          perspective.updateProjectionMatrix();
        }
      }
      flushSync(() => onShow(restoreIndex));
      renderer.render(view.scene, camera);
    }
    return outcome;
  };

  /** Output size and a canvas of that size for a render (the current drawing size for "Current view"). */
  const targetFor = (): { size: { width: number; height: number } | null; out: HTMLCanvasElement } => {
    const view = liveView();
    if (!view) throw new Error('No 3D view to render yet.');
    const size = resolveSize(resolution);
    const width = size ? size.width : view.gl.domElement.width;
    const height = size ? size.height : view.gl.domElement.height;
    const out = document.createElement('canvas');
    out.width = width;
    out.height = height;
    return { size, out };
  };

  const overlayOpts = () => (overlay
    ? { dt, units, timeUnit: thermoUnit(units, 'time') }
    : null);

  const snapshot = () => {
    const url = captureActiveCanvas();
    if (!url) { setMessage('No 3D view to capture yet.'); return; }
    downloadFile(`snapshot-${stamp()}.png`, url);
    setMessage('Snapshot saved as PNG.');
  };

  const exportFrames = async () => {
    if (!selection.ok) { setMessage(selection.error); return; }
    setMessage(null);
    setProgress({ kind: 'frames', done: 0, total: selection.indices.length });
    onVideoActive(true);
    cancelRef.current = false;
    const files: { position: number; step: number; png: Uint8Array }[] = [];
    try {
      const { size, out } = targetFor();
      const result = await runSequence({
        indices: selection.indices, size, out, overlay: overlayOpts(), orbitDeg,
        onFrame: async (position, idx) => {
          const bin = dataUrlBytes(out.toDataURL('image/png'));
          if (!bin) throw new Error('PNG encoding failed.');
          files.push({ position, step: buffer.get(idx)?.step ?? 0, png: bin.bytes });
          setProgress({ kind: 'frames', done: files.length, total: selection.indices.length });
        },
      });
      if (result === 'cancelled') setMessage(`Export cancelled after ${files.length} frames.`);
      else {
        saveBytes(`frames-${stamp()}.zip`, framesZip(files), 'application/zip');
        setMessage(`Saved ${files.length} PNG frames to a zip.`);
      }
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Export failed.');
    } finally {
      setProgress(null);
      onVideoActive(false);
    }
  };

  const recordVideo = async () => {
    if (!selection.ok) { setMessage(selection.error); return; }
    if (!format) { setMessage('This browser cannot record video (MediaRecorder or its codecs are missing). Use Export frames instead.'); return; }
    const indices = repeatIndices(selection.indices, loops);
    setMessage(null);
    setProgress({ kind: 'video', done: 0, total: indices.length });
    onVideoActive(true);
    cancelRef.current = false;
    let handle: RecordingHandle | null = null;
    let cancelled = false;
    try {
      const { size, out } = targetFor();
      handle = startCanvasRecording(out, {
        fps, format, videoBitsPerSecond: bitrateFor(quality), manualFrames: true,
      });
      const tick = 1000 / fps;
      // each captured frame is due one tick after the previous one; a frame that renders slower than a tick
      // is captured as soon as it is ready, and the schedule does not try to catch up with a burst of frames
      let due = performance.now();
      const result = await runSequence({
        indices, size, out, overlay: overlayOpts(), orbitDeg,
        onFrame: async (_position, _idx) => {
          // one captured frame per drawn frame: no frame is repeated or dropped
          handle?.capture?.();
          setProgress((p) => (p ? { ...p, done: p.done + 1 } : p));
          due = Math.max(due + tick, performance.now());
          await sleep(Math.max(0, due - performance.now()));
        },
      });
      cancelled = result === 'cancelled';
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Recording failed.');
    } finally {
      if (handle) {
        try {
          const rec = await handle.stop();
          if (!cancelled && rec.blob.size > 0) {
            const url = URL.createObjectURL(rec.blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `render-${stamp()}.${rec.ext}`;
            a.click();
            window.setTimeout(() => URL.revokeObjectURL(url), 5000);
            setMessage(`Video saved (${rec.label}, ${indices.length} frames at ${fps} fps).`);
          } else if (cancelled) {
            setMessage('Video cancelled.');
          } else {
            setMessage('The recording was empty; nothing was saved.');
          }
        } catch (e) {
          setMessage(e instanceof Error ? e.message : 'Recording failed.');
        }
      }
      setProgress(null);
      onVideoActive(false);
    }
  };

  const cancel = () => { cancelRef.current = true; };

  const fieldClass = `min-h-6 w-16 rounded border px-1 py-0.5 font-mono text-xs ${ct.input}`;
  const selectClass = `min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`;
  const label = `flex items-center gap-1 text-xs ${ct.muted}`;
  const sourceLine = source
    ? `From ${source.name}: ${count} ${count === 1 ? 'frame' : 'frames'}`
    : `Recorded frames: ${count}`;

  return (
    <section aria-label="Render" role="dialog" aria-labelledby="render-panel-title" aria-busy={busy}
      onKeyDown={(e) => { if (e.key === 'Escape' && !busy) onClose(); }}
      className={`fixed inset-x-0 bottom-0 z-40 max-h-[70vh] overflow-y-auto border-t p-3 shadow-2xl sm:inset-x-auto sm:bottom-4 sm:right-4 sm:w-[26rem] sm:max-h-[80vh] sm:rounded-xl sm:border ${ct.panel}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h2 id="render-panel-title" ref={heading} tabIndex={-1} className="text-sm font-semibold outline-none">Render</h2>
          <p className={`text-[11px] ${ct.muted}`} aria-live="polite">{sourceLine}</p>
        </div>
        <button className={`${btn} ${ct.button}`} aria-label="Close render panel" title="Close (Escape)" disabled={busy} onClick={onClose}>
          <X size={13} aria-hidden="true" />
        </button>
      </div>

      {canRestore && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <button className={`${btn} ${ct.button}`} disabled={busy} onClick={onRestore}
            title="Show the frames the run recorded again (the loaded file's frames are dropped)">
            <Undo2 size={13} aria-hidden="true" />Back to the run&apos;s frames
          </button>
        </div>
      )}
      {notice && <p role="alert" className="mt-2 text-xs text-red-400">{notice}</p>}
      {live && <p className={`mt-2 text-xs ${ct.muted}`}>Wait for the run to finish, then render.</p>}
      {!live && count === 0 && (
        <p className={`mt-2 text-xs ${ct.muted}`}>Run a cell to record frames, or render a dump from Run output.</p>
      )}

      <fieldset disabled={blocked} className="mt-3 space-y-2">
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Frames</legend>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>From</span>
            <input type="number" min={1} step={1} inputMode="numeric" aria-label="First frame" value={startS}
              onChange={(e) => setStartS(e.target.value)} className={fieldClass} />
          </label>
          <label className={label}>
            <span>to</span>
            <input type="number" min={1} step={1} inputMode="numeric" aria-label="Last frame" value={endS}
              placeholder={String(count)} onChange={(e) => setEndS(e.target.value)} className={fieldClass} />
          </label>
          <label className={label}>
            <span>every</span>
            <input type="number" min={1} step={1} inputMode="numeric" aria-label="Frame stride" value={strideS}
              onChange={(e) => setStrideS(e.target.value)} className={fieldClass} />
            <span>frame(s)</span>
          </label>
        </div>
        <p role="status" className={`text-[11px] ${selection.ok ? ct.muted : 'text-red-400'}`}>
          {selection.ok ? `${selection.indices.length} frame${selection.indices.length === 1 ? '' : 's'} selected` : selection.error}
        </p>

        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Size</span>
            <select aria-label="Output resolution" value={resolution} onChange={(e) => setResolution(e.target.value as ResolutionPreset)}
              className={selectClass}>
              {RESOLUTIONS.map((r) => <option key={r} value={r}>{RESOLUTION_LABEL[r]}</option>)}
            </select>
          </label>
          <label className={label}>
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} aria-label="Step and time overlay" />
            <span>Step and time overlay</span>
          </label>
          <label className={label} title="The camera orbits this many degrees around the target on each frame after the first (0 keeps it still)">
            <span>Orbit</span>
            <input type="number" min={0} max={30} step={0.5} inputMode="decimal" aria-label="Orbit degrees per frame" value={orbitS}
              onChange={(e) => setOrbitS(e.target.value)} className={fieldClass} />
            <span>°/frame</span>
          </label>
        </div>
      </fieldset>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <button className={`${btn} ${ct.button}`} disabled={busy || count === 0} onClick={snapshot}
          title="Save the current view as a PNG (at its current size, without overlay)">
          <Camera size={13} aria-hidden="true" />Snapshot
        </button>
        <button className={`${btn} ${ct.button}`} disabled={blocked || !selection.ok} onClick={() => void exportFrames()}
          title="Render the selected frames to PNG files and download them as one zip">
          <Images size={13} aria-hidden="true" />Export frames (.zip)
        </button>
      </div>

      <fieldset disabled={blocked} className="mt-3 space-y-2">
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Video</legend>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Format</span>
            <select aria-label="Video format" value={formatIdx} onChange={(e) => setFormatIdx(Number(e.target.value))}
              disabled={formats.length === 0} className={selectClass}>
              {formats.length === 0 && <option value={0}>not available</option>}
              {formats.map((f, i) => <option key={f.mime} value={i}>{f.label}</option>)}
            </select>
          </label>
          <label className={label}>
            <span>FPS</span>
            <select aria-label="Video frames per second" value={fps} onChange={(e) => setFps(Number(e.target.value))} className={selectClass}>
              {VIDEO_FPS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </label>
          <label className={label}>
            <span>Quality</span>
            <select aria-label="Video quality" value={quality} onChange={(e) => setQuality(e.target.value as VideoQuality)} className={selectClass}>
              {QUALITIES.map((q) => <option key={q} value={q}>{QUALITY_LABEL[q]}</option>)}
            </select>
          </label>
          <label className={label}>
            <span>Loops</span>
            <input type="number" min={1} max={MAX_LOOPS} step={1} inputMode="numeric" aria-label="Loop count" value={loopsS}
              onChange={(e) => setLoopsS(e.target.value)} className={fieldClass} />
          </label>
        </div>
      </fieldset>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        {progress === null ? (
          <button className={`${btn} ${ct.accent}`} disabled={blocked || !selection.ok || !format} onClick={() => void recordVideo()}
            title="Play the selected frames as a video of the 3D view, at the chosen size, FPS and quality">
            <Film size={13} aria-hidden="true" />Record video
          </button>
        ) : (
          <>
            <span role="status" aria-live="polite" className={`text-xs ${ct.muted}`}>
              {progress.kind === 'video' ? 'Recording' : 'Rendering'} frame {progress.done} of {progress.total}
              {' '}({Math.round((100 * progress.done) / Math.max(1, progress.total))}%)
            </span>
            <progress max={progress.total} value={progress.done} aria-label="Render progress" className="h-1.5 w-32" />
            <button className={`${btn} ${ct.button}`} onClick={cancel}>Cancel</button>
          </>
        )}
      </div>
      {!format && (
        <p className={`mt-1 text-[11px] ${ct.muted}`}>No video recording in this browser; frame export still works.</p>
      )}
      <p role="status" aria-live="polite" className={`mt-2 text-[11px] ${ct.muted}`}>{message ?? ''}</p>
    </section>
  );
};

export default RenderPanel;
