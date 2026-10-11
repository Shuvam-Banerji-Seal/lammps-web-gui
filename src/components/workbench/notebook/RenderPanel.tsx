import React, { useEffect, useMemo, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import * as THREE from 'three';
import { Camera, Film, Images, Undo2, X, Image as ImageIcon, FileImage } from 'lucide-react';
import type { getThemeTokens } from '../../../theme';
import { dataUrlBytes, downloadFile } from '../../../lammps/exporter';
import { thermoUnit } from '../../../lammps/thermoUnits';
import { captureActiveCanvas, getActiveGL } from '../../../services/glRegistry';
import { emitCameraCommand } from '../../../services/cameraBus';
import { recordingFormats, startCanvasRecording, type RecordingFormat, type RecordingHandle } from '../../../services/recorder';
import type { TrajectoryBuffer, TrajectoryRecord } from './trajectory';
import {
  QUALITY_LABEL, bitrateFor, framesZip, overlayText, selectFrames, type VideoQuality,
} from './renderFrames';
import {
  ASPECT_LABEL, SIZE_LABEL, DEFAULT_LIMITS, outputSize, planRenderSize, type AspectChoice, type GpuLimits, type RenderSizePlan,
  type SizeChoice, type Size, type Supersample,
} from './render/size';
import {
  COLOUR_LABEL, SCALAR_MODES, atomColours, colourBarLabels, heatColour, rangeOf, scalarValues, type ColourMode, type ValueRange,
} from './render/colour';
import { CAMERA_LABEL, cameraFrame, orbitPose, reviveOrbitDegrees, type CameraMode } from './render/camera';
import { MAX_OUTPUT_FRAMES, SPEED_CHOICES, gifDelayCentiseconds, playbackPlan } from './render/timing';
import {
  CORNER_LABEL, FONT_LABEL, colourBarLayout, drawColourBar, drawTextBlock, drawTriad, layoutTextBlock, triadArrows,
  triadLayout, type Corner, type FontSize, type TextLine,
} from './render/overlay';
import {
  BACKGROUND_LABEL, BOX_LABEL, RADIUS_SCALE_CHOICES, backgroundHex, supportsTransparency, unmatte,
  type BackgroundChoice, type BoxChoice,
} from './render/look';
import { GifWriter } from './render/gif';
import { NO_OVERRIDE, setAtomRenderOverride } from './render/atomOverride';

type Tokens = ReturnType<typeof getThemeTokens>;

const VIDEO_FPS = [10, 15, 25, 30, 60] as const;
const QUALITIES: VideoQuality[] = ['low', 'medium', 'high'];
const MAX_LOOPS = 20;
const MAX_HOLD = 300;
const SIZES: SizeChoice[] = ['current', '720p', '1080p', 'custom'];
const ASPECTS: AspectChoice[] = ['native', '16:9', '1:1', '4:3', '9:16'];
const BACKGROUNDS: BackgroundChoice[] = ['theme', 'black', 'white', 'custom', 'transparent'];
const COLOUR_MODES: ColourMode[] = ['type', 'speed', 'ke', 'x', 'y', 'z', 'id', 'single'];
const CAMERAS: CameraMode[] = ['static', 'orbit', 'zoom'];
const CORNERS: Corner[] = ['tl', 'tr', 'bl', 'br'];
const FONTS: FontSize[] = ['s', 'm', 'l'];
const BOXES: BoxChoice[] = ['view', 'off'];

const nextFrame = (): Promise<void> =>
  new Promise((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve())));

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

const stamp = (): string => new Date().toISOString().replace(/[:.]/g, '-');

const toNum = (s: string): number => (s.trim() === '' ? NaN : Number(s));

/** A whole number from input text, clamped to [min, max]; the fallback when it is not a number. */
const wholeIn = (s: string, min: number, max: number, fallback: number): number => {
  const n = Math.floor(toNum(s));
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

const saveBytes = (name: string, bytes: Uint8Array<ArrayBuffer>, mime: string): void => {
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

/** Sampled colour map stops for the legend. */
const LEGEND_STOPS: [number, number, number][] = Array.from({ length: 9 }, (_, i) => heatColour(i / 8));

/** Canvas of w × h (an output or a scratch canvas). */
const makeCanvas = (w: number, h: number): HTMLCanvasElement => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

/** Everything a render needs to draw one frame, fixed for the whole clip. */
interface Look {
  /** A transparent frame (PNG outputs): drawn over black and white and matted. */
  matte: boolean;
  /** The background colour of a pass, or null to keep the view's own. */
  bgHex: string | null;
  colourBy: ColourMode;
  /** The value range of a scalar colouring over the whole clip (null for the other modes). */
  range: ValueRange | null;
  single: string;
  colourBar: boolean;
  radiusScale: number;
  box: BoxChoice;
  axes: boolean;
  camera: CameraMode;
  orbitDeg: number;
  overlay: boolean;
  title: string;
  caption: string;
  corner: Corner;
  fontSize: FontSize;
  dt: number | null;
  units: string | undefined;
  timeUnit: string | null;
}

interface RenderJob {
  /** Buffer indices, one per output frame, in output order. */
  indices: number[];
  /** The output canvas (drawn into for every frame) and the size the 3D view is drawn at. */
  canvas: HTMLCanvasElement;
  render: Size;
  look: Look;
  /** Called after each output frame is drawn into the canvas. */
  onFrame: (k: number, idx: number) => Promise<void>;
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

/** GPU limits for render targets: the smaller side limit of the context, and the default area cap. */
const gpuLimitsOf = (renderer: THREE.WebGLRenderer): GpuLimits => {
  try {
    const gl = renderer.getContext() as WebGL2RenderingContext;
    const dims = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
    const rb = gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number;
    const side = Math.min(dims[0], dims[1], rb);
    return { maxSide: side > 0 ? side : DEFAULT_LIMITS.maxSide, maxArea: DEFAULT_LIMITS.maxArea };
  } catch {
    return DEFAULT_LIMITS;
  }
};

const capNote = (sp: RenderSizePlan): string | null =>
  sp.capped ? 'Capped for this GPU: the size or the supersampling was reduced.' : null;

const sizeText = (sp: RenderSizePlan): string =>
  `${sp.out.width} × ${sp.out.height}${sp.supersample === 2 ? ' (2× supersampled)' : ''}`;

/**
 * Render panel of the notebook: a snapshot of the view, one high-resolution PNG, a zip of PNG frames, an animated
 * GIF and a video (WebM or MP4 where the browser records it) of a frame range. Every output has a size (aspect,
 * custom, supersampling capped to the GPU), a look (background, colouring by type, speed, kinetic energy, height
 * or atom id, a single colour; radius; box; axes triad), a camera path (static, orbit, slow zoom), overlays (title,
 * caption, step and time, a colour bar) and timing (speed factor, holds, loops). Each frame is shown, drawn, and
 * only then captured; the camera, background, box, colours and shown frame are restored afterwards.
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
  // frames and timing
  const [startS, setStartS] = useState('1');
  const [endS, setEndS] = useState('');
  const [strideS, setStrideS] = useState('1');
  const [speed, setSpeed] = useState<number>(1);
  const [holdFirstS, setHoldFirstS] = useState('0');
  const [holdLastS, setHoldLastS] = useState('0');
  const [loopsS, setLoopsS] = useState('1');
  const [fps, setFps] = useState<number>(25);
  // size
  const [sizeChoice, setSizeChoice] = useState<SizeChoice>('current');
  const [aspect, setAspect] = useState<AspectChoice>('native');
  const [customWS, setCustomWS] = useState('1280');
  const [customHS, setCustomHS] = useState('720');
  const [supersample, setSupersample] = useState<Supersample>(1);
  // look
  const [background, setBackground] = useState<BackgroundChoice>('theme');
  const [bgColour, setBgColour] = useState('#101820');
  const [colourBy, setColourBy] = useState<ColourMode>('type');
  const [singleColour, setSingleColour] = useState('#ffcc00');
  const [colourBar, setColourBar] = useState(true);
  const [radiusScale, setRadiusScale] = useState<number>(1);
  const [box, setBox] = useState<BoxChoice>('view');
  const [axes, setAxes] = useState(false);
  // camera
  const [camera, setCamera] = useState<CameraMode>('static');
  const [orbitS, setOrbitS] = useState('360');
  // overlays
  const [overlay, setOverlay] = useState(true);
  const [title, setTitle] = useState('');
  const [caption, setCaption] = useState('');
  const [corner, setCorner] = useState<Corner>('tl');
  const [fontSize, setFontSize] = useState<FontSize>('m');
  // video output
  const [formatIdx, setFormatIdx] = useState(0);
  const [quality, setQuality] = useState<VideoQuality>('high');
  const [progress, setProgress] = useState<{ kind: 'frames' | 'video' | 'gif' | 'png'; done: number; total: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const cancelRef = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const busy = progress !== null;

  useEffect(() => { heading.current?.focus(); }, []);

  const endValue = endS.trim() === '' ? count : toNum(endS);
  const selection = selectFrames(count, { start: toNum(startS), end: endValue, stride: toNum(strideS) });
  const loops = wholeIn(loopsS, 1, MAX_LOOPS, 1);
  const holdFirst = wholeIn(holdFirstS, 0, MAX_HOLD, 0);
  const holdLast = wholeIn(holdLastS, 0, MAX_HOLD, 0);
  const orbitDeg = reviveOrbitDegrees(orbitS);
  const plan = selection.ok ? playbackPlan(selection.indices, { speed, holdFirst, holdLast, loops }) : [];
  const tooLong = plan.length > MAX_OUTPUT_FRAMES;
  const format = formats[formatIdx] ?? null;
  const blocked = busy || live || count === 0;
  const scalarColour = SCALAR_MODES.includes(colourBy);
  // speed and kinetic energy need the timestep (the displacement between recorded frames is divided by it)
  const speedUsable = dt !== null && Number.isFinite(dt) && dt !== 0;

  /** The view's renderer and camera, if the registered 3D view is the one in this notebook. */
  const liveView = () => {
    const gl = getActiveGL();
    const host = canvasHost.current;
    return gl && host && host.contains(gl.gl.domElement) ? gl : null;
  };

  /** The output and render sizes for the current settings, with the GPU cap applied. */
  const sizePlan = (): RenderSizePlan => {
    const view = liveView();
    const desired = outputSize({
      size: sizeChoice,
      aspect,
      view: { width: view ? view.gl.domElement.width : 1280, height: view ? view.gl.domElement.height : 720 },
      custom: { width: wholeIn(customWS, 2, 16384, 1280), height: wholeIn(customHS, 2, 16384, 720) },
    });
    return planRenderSize(desired, supersample, view ? gpuLimitsOf(view.gl) : DEFAULT_LIMITS);
  };
  // the preview is recomputed when the view is rebuilt (version changes with the buffer)
  const preview = useMemo(() => sizePlan(), [sizeChoice, aspect, customWS, customHS, supersample, version]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Draws the planned frames in order into the job's canvas (see the file comment on RenderPanel). */
  const runSequence = async (job: RenderJob): Promise<'done' | 'cancelled'> => {
    const view = liveView();
    if (!view) throw new Error('No 3D view to render yet.');
    const renderer = view.gl;
    const scene = view.scene;
    const cam = view.camera;
    const perspective = cam instanceof THREE.PerspectiveCamera ? cam : null;
    const restoreIndex = index;
    const css = renderer.getSize(new THREE.Vector2());
    const pixelRatio = renderer.getPixelRatio();
    const aspectWas = perspective?.aspect ?? 1;
    const pose0: [number, number, number] = [cam.position.x, cam.position.y, cam.position.z];
    const prevBg = scene.background;
    const boxObj = scene.getObjectByName('simulation-box');
    const boxWasVisible = boxObj ? boxObj.visible : true;
    const out = job.canvas;
    const ctx = out.getContext('2d', { willReadFrequently: job.look.matte });
    if (!ctx) throw new Error('Canvas 2D is not available in this browser.');
    ctx.imageSmoothingQuality = 'high';
    const W = out.width;
    const H = out.height;
    const look = job.look;
    const K = job.indices.length;
    const scratchA = look.matte ? makeCanvas(W, H).getContext('2d', { willReadFrequently: true }) : null;
    const scratchB = look.matte ? makeCanvas(W, H).getContext('2d', { willReadFrequently: true }) : null;
    const setBg = (hex: string | null) => {
      scene.background = hex ? new THREE.Color(hex) : prevBg;
    };
    const colourFor = (idx: number): Float32Array | null => {
      if (look.colourBy === 'type') return null;
      const rec = buffer.get(idx);
      if (!rec) return null;
      const n = rec.topo.id.length;
      if (!SCALAR_MODES.includes(look.colourBy)) return atomColours(null, n, null, look.single);
      const vals = scalarValues(look.colourBy, buffer.get(idx - 1), rec, buffer.get(idx + 1), look.dt);
      return atomColours(vals, n, look.range, look.single);
    };
    let outcome: 'done' | 'cancelled' = 'done';
    try {
      renderer.setPixelRatio(1);
      renderer.setSize(job.render.width, job.render.height, false);
      if (perspective) {
        perspective.aspect = job.render.width / job.render.height;
        perspective.updateProjectionMatrix();
      }
      for (let k = 0; k < K; k++) {
        if (cancelRef.current) { outcome = 'cancelled'; break; }
        const idx = job.indices[k];
        const rec = buffer.get(idx);
        flushSync(() => onShow(idx));
        await nextFrame();
        if (look.camera !== 'static') {
          const f = cameraFrame(look.camera, k, K, look.orbitDeg);
          emitCameraCommand({ type: 'pose', position: orbitPose(pose0, f.angle, f.scale) });
        }
        setAtomRenderOverride({ colours: colourFor(idx), radiusScale: look.radiusScale });
        if (boxObj) boxObj.visible = look.box === 'off' ? false : boxWasVisible;
        if (look.matte && scratchA && scratchB) {
          // two passes over black and white, then the alpha is the difference (see render/look.ts)
          setBg('#000000');
          renderer.render(scene, cam);
          scratchA.drawImage(renderer.domElement, 0, 0, W, H);
          setBg('#ffffff');
          renderer.render(scene, cam);
          scratchB.drawImage(renderer.domElement, 0, 0, W, H);
          const a = scratchA.getImageData(0, 0, W, H).data;
          const b = scratchB.getImageData(0, 0, W, H).data;
          ctx.clearRect(0, 0, W, H);
          ctx.putImageData(new ImageData(unmatte(a, b), W, H), 0, 0);
        } else {
          setBg(look.bgHex);
          renderer.render(scene, cam);
          ctx.drawImage(renderer.domElement, 0, 0, W, H);
        }
        drawFrameOverlays(ctx, W, H, rec, look, cam);
        await job.onFrame(k, idx);
      }
    } finally {
      setAtomRenderOverride(NO_OVERRIDE);
      scene.background = prevBg;
      if (boxObj) boxObj.visible = boxWasVisible;
      emitCameraCommand({ type: 'pose', position: pose0 });
      renderer.setPixelRatio(pixelRatio);
      renderer.setSize(css.x, css.y);
      if (perspective) {
        perspective.aspect = aspectWas;
        perspective.updateProjectionMatrix();
      }
      flushSync(() => onShow(restoreIndex));
      renderer.render(scene, cam);
    }
    return outcome;
  };

  /** The overlays of one frame: the text block, the colour bar and the axes triad. */
  const drawFrameOverlays = (
    ctx: CanvasRenderingContext2D,
    W: number,
    H: number,
    rec: TrajectoryRecord | undefined,
    look: Look,
    cam: THREE.Camera,
  ) => {
    const lines: TextLine[] = [];
    if (look.title) lines.push({ text: look.title, bold: true, scale: 1.4 });
    if (look.caption) lines.push({ text: look.caption });
    if (look.overlay && rec) lines.push({ text: overlayText(rec.step, look.dt, look.units, look.timeUnit) });
    const measure = (text: string, px: number, bold: boolean) => {
      ctx.font = `${bold ? 'bold ' : ''}${px}px ui-sans-serif, system-ui, sans-serif`;
      return ctx.measureText(text).width;
    };
    const block = layoutTextBlock(W, H, look.corner, lines, look.fontSize, measure);
    if (block) drawTextBlock(ctx, block);
    if (look.colourBar && look.range && SCALAR_MODES.includes(look.colourBy)) {
      const labels = colourBarLabels(look.range);
      drawColourBar(ctx, colourBarLayout(W, H), COLOUR_LABEL[look.colourBy], labels.low, labels.high, LEGEND_STOPS);
    }
    if (look.axes) {
      const q = cam.quaternion;
      drawTriad(ctx, triadLayout(W, H), triadArrows([q.x, q.y, q.z, q.w]));
    }
  };

  /** The look for an output format, from the panel's settings (transparency only where the format keeps alpha). */
  const lookFor = (format: 'png' | 'gif' | 'video', range: ValueRange | null): Look => {
    const transparent = background === 'transparent';
    const matte = transparent && supportsTransparency(format);
    return {
      matte,
      bgHex: transparent && !matte ? null : backgroundHex(background, bgColour),
      colourBy,
      range,
      single: singleColour,
      colourBar,
      radiusScale,
      box,
      axes,
      camera,
      orbitDeg,
      overlay,
      title: title.trim(),
      caption: caption.trim(),
      corner,
      fontSize,
      dt,
      units,
      timeUnit: thermoUnit(units, 'time'),
    };
  };

  /** Min and max of a scalar colouring over the distinct frames of a job (null for the other modes). */
  const rangeFor = (indices: readonly number[]): ValueRange | null => {
    if (!scalarColour) return null;
    let range: ValueRange | null = null;
    const seen = new Set<number>();
    for (const idx of indices) {
      if (seen.has(idx)) continue;
      seen.add(idx);
      const cur = buffer.get(idx);
      if (!cur) continue;
      const r = rangeOf([scalarValues(colourBy, buffer.get(idx - 1), cur, buffer.get(idx + 1), dt)]);
      if (r) range = range ? { min: Math.min(range.min, r.min), max: Math.max(range.max, r.max) } : r;
    }
    return range;
  };

  /** A job over the planned indices: its canvas, its look (colour range included) and the per-frame callback. */
  const makeJob = (
    indices: number[],
    fmt: 'png' | 'gif' | 'video',
    sp: RenderSizePlan,
    onFrame: RenderJob['onFrame'],
  ): RenderJob => ({
    indices,
    canvas: makeCanvas(sp.out.width, sp.out.height),
    render: sp.render,
    look: lookFor(fmt, rangeFor(indices)),
    onFrame,
  });

  const tooLongMessage = `At most ${MAX_OUTPUT_FRAMES} output frames per render; reduce the range, the loops or the holds.`;

  const snapshot = () => {
    const url = captureActiveCanvas();
    if (!url) { setMessage('No 3D view to capture yet.'); return; }
    downloadFile(`snapshot-${stamp()}.png`, url);
    setMessage('Snapshot saved as PNG.');
  };

  /** One high-resolution PNG of the shown frame, at the chosen size, with the look and the overlays. */
  const highResPng = async () => {
    const shown = count === 0 ? -1 : Math.min(count - 1, index ?? count - 1);
    if (shown < 0) { setMessage('Record a frame first.'); return; }
    setMessage(null);
    setProgress({ kind: 'png', done: 0, total: 1 });
    onVideoActive(true);
    cancelRef.current = false;
    try {
      const sp = sizePlan();
      const job = makeJob([shown], 'png', sp, async () => {
        const bin = dataUrlBytes(job.canvas.toDataURL('image/png'));
        if (!bin) throw new Error('PNG encoding failed.');
        saveBytes(`render-${stamp()}.png`, bin.bytes, 'image/png');
        setProgress({ kind: 'png', done: 1, total: 1 });
      });
      const result = await runSequence(job);
      setMessage(result === 'cancelled'
        ? 'High-resolution PNG cancelled.'
        : `Saved a ${sizeText(sp)} PNG.${capNote(sp) ? ` ${capNote(sp)}` : ''}`);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Render failed.');
    } finally {
      setProgress(null);
      onVideoActive(false);
    }
  };

  const exportFrames = async () => {
    if (!selection.ok) { setMessage(selection.error); return; }
    if (tooLong) { setMessage(tooLongMessage); return; }
    setMessage(null);
    setProgress({ kind: 'frames', done: 0, total: plan.length });
    onVideoActive(true);
    cancelRef.current = false;
    const files: { position: number; step: number; png: Uint8Array }[] = [];
    try {
      const sp = sizePlan();
      const job = makeJob(plan, 'png', sp, async (k, idx) => {
        const bin = dataUrlBytes(job.canvas.toDataURL('image/png'));
        if (!bin) throw new Error('PNG encoding failed.');
        files.push({ position: k + 1, step: buffer.get(idx)?.step ?? 0, png: bin.bytes });
        setProgress({ kind: 'frames', done: files.length, total: plan.length });
      });
      const result = await runSequence(job);
      if (result === 'cancelled') setMessage(`Export cancelled after ${files.length} frames.`);
      else {
        saveBytes(`frames-${stamp()}.zip`, framesZip(files), 'application/zip');
        setMessage(`Saved ${files.length} PNG frames (${sizeText(sp)}) to a zip.${capNote(sp) ? ` ${capNote(sp)}` : ''}`);
      }
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Export failed.');
    } finally {
      setProgress(null);
      onVideoActive(false);
    }
  };

  const exportGif = async () => {
    if (!selection.ok) { setMessage(selection.error); return; }
    if (tooLong) { setMessage(tooLongMessage); return; }
    setMessage(null);
    setProgress({ kind: 'gif', done: 0, total: plan.length });
    onVideoActive(true);
    cancelRef.current = false;
    try {
      const sp = sizePlan();
      const writer = new GifWriter(sp.out.width, sp.out.height, { loops: 0 });
      const delay = gifDelayCentiseconds(fps);
      const job = makeJob(plan, 'gif', sp, async (k) => {
        const ctx = job.canvas.getContext('2d');
        if (!ctx) throw new Error('Canvas 2D is not available in this browser.');
        writer.addFrame(ctx.getImageData(0, 0, job.canvas.width, job.canvas.height).data, delay);
        setProgress({ kind: 'gif', done: k + 1, total: plan.length });
      });
      const result = await runSequence(job);
      if (result === 'cancelled') setMessage('GIF cancelled.');
      else {
        const bytes = writer.finish();
        saveBytes(`render-${stamp()}.gif`, bytes, 'image/gif');
        setMessage(`Saved a ${writer.frames}-frame GIF at ${fps} fps (${sizeText(sp)}).${capNote(sp) ? ` ${capNote(sp)}` : ''}`);
      }
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'GIF export failed.');
    } finally {
      setProgress(null);
      onVideoActive(false);
    }
  };

  const recordVideo = async () => {
    if (!selection.ok) { setMessage(selection.error); return; }
    if (!format) { setMessage('This browser cannot record video (MediaRecorder or its codecs are missing). Use Export frames instead.'); return; }
    if (tooLong) { setMessage(tooLongMessage); return; }
    setMessage(null);
    setProgress({ kind: 'video', done: 0, total: plan.length });
    onVideoActive(true);
    cancelRef.current = false;
    let handle: RecordingHandle | null = null;
    let cancelled = false;
    const sp = sizePlan();
    // each captured frame is due one tick after the previous one; a frame that renders slower than a tick is
    // captured as soon as it is ready, and the schedule does not try to catch up with a burst of frames
    const tick = 1000 / fps;
    let due = performance.now();
    try {
      const job = makeJob(plan, 'video', sp, async () => {
        // one captured frame per drawn frame: no frame is repeated or dropped
        handle?.capture?.();
        setProgress((p) => (p ? { ...p, done: p.done + 1 } : p));
        due = Math.max(due + tick, performance.now());
        await sleep(Math.max(0, due - performance.now()));
      });
      handle = startCanvasRecording(job.canvas, {
        fps, format, videoBitsPerSecond: bitrateFor(quality), manualFrames: true,
      });
      cancelled = (await runSequence(job)) === 'cancelled';
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
            setMessage(`Video saved (${rec.label}, ${plan.length} frames at ${fps} fps, ${sizeText(sp)}).${capNote(sp) ? ` ${capNote(sp)}` : ''}`);
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
  const textClass = `min-h-6 w-40 rounded border px-1 py-0.5 text-xs ${ct.input}`;
  const selectClass = `min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`;
  const label = `flex items-center gap-1 text-xs ${ct.muted}`;
  const sourceLine = source
    ? `From ${source.name}: ${count} ${count === 1 ? 'frame' : 'frames'}`
    : `Recorded frames: ${count}`;
  const planNote = selection.ok
    ? `${plan.length} output frame${plan.length === 1 ? '' : 's'}${plan.length !== selection.indices.length ? ` (from ${selection.indices.length} recorded)` : ''}`
    : selection.error;

  return (
    <section aria-label="Render" role="dialog" aria-labelledby="render-panel-title" aria-busy={busy}
      onKeyDown={(e) => { if (e.key === 'Escape' && !busy) onClose(); }}
      className={`fixed inset-x-0 bottom-0 z-40 max-h-[70vh] overflow-y-auto border-t p-3 shadow-2xl sm:inset-x-auto sm:bottom-4 sm:right-4 sm:w-[28rem] sm:max-h-[80vh] sm:rounded-xl sm:border ${ct.panel}`}>
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
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Frames and timing</legend>
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
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Speed</span>
            <select aria-label="Playback speed factor" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className={selectClass}>
              {SPEED_CHOICES.map((s) => <option key={s} value={s}>{s === 1 ? 'normal' : `${s}×${s < 1 ? ' (slow motion)' : ''}`}</option>)}
            </select>
          </label>
          <label className={label} title="Extra output frames holding the first frame">
            <span>Hold first</span>
            <input type="number" min={0} max={MAX_HOLD} step={1} inputMode="numeric" aria-label="Hold first frame (output frames)" value={holdFirstS}
              onChange={(e) => setHoldFirstS(e.target.value)} className={fieldClass} />
          </label>
          <label className={label} title="Extra output frames holding the last frame">
            <span>Hold last</span>
            <input type="number" min={0} max={MAX_HOLD} step={1} inputMode="numeric" aria-label="Hold last frame (output frames)" value={holdLastS}
              onChange={(e) => setHoldLastS(e.target.value)} className={fieldClass} />
          </label>
          <label className={label}>
            <span>FPS</span>
            <select aria-label="Video frames per second" value={fps} onChange={(e) => setFps(Number(e.target.value))} className={selectClass}>
              {VIDEO_FPS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </label>
          <label className={label}>
            <span>Loops</span>
            <input type="number" min={1} max={MAX_LOOPS} step={1} inputMode="numeric" aria-label="Loop count" value={loopsS}
              onChange={(e) => setLoopsS(e.target.value)} className={fieldClass} />
          </label>
        </div>
        <p role="status" className={`text-[11px] ${selection.ok && !tooLong ? ct.muted : 'text-red-400'}`}>
          {tooLong ? `${plan.length} output frames: over the limit of ${MAX_OUTPUT_FRAMES}.` : planNote}
        </p>
      </fieldset>

      <fieldset disabled={blocked} className="mt-3 space-y-2">
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Size</legend>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Size</span>
            <select aria-label="Output size" value={sizeChoice} onChange={(e) => setSizeChoice(e.target.value as SizeChoice)} className={selectClass}>
              {SIZES.map((s) => <option key={s} value={s}>{SIZE_LABEL[s]}</option>)}
            </select>
          </label>
          {sizeChoice !== 'custom' && (
            <label className={label}>
              <span>Aspect</span>
              <select aria-label="Output aspect" value={aspect} onChange={(e) => setAspect(e.target.value as AspectChoice)} className={selectClass}>
                {ASPECTS.map((a) => <option key={a} value={a}>{ASPECT_LABEL[a]}</option>)}
              </select>
            </label>
          )}
          {sizeChoice === 'custom' && (
            <label className={label}>
              <span>Width × height</span>
              <input type="number" min={2} max={16384} step={2} inputMode="numeric" aria-label="Custom width" value={customWS}
                onChange={(e) => setCustomWS(e.target.value)} className={fieldClass} />
              <span>×</span>
              <input type="number" min={2} max={16384} step={2} inputMode="numeric" aria-label="Custom height" value={customHS}
                onChange={(e) => setCustomHS(e.target.value)} className={fieldClass} />
            </label>
          )}
          <label className={label}>
            <span>Supersampling</span>
            <select aria-label="Supersampling" value={supersample} onChange={(e) => setSupersample(Number(e.target.value) as Supersample)} className={selectClass}>
              <option value={1}>1× (none)</option>
              <option value={2}>2× (smoother edges)</option>
            </select>
          </label>
        </div>
        <p role="status" className={`text-[11px] ${preview.capped ? 'text-amber-400' : ct.muted}`}>
          Output {sizeText(preview)}{capNote(preview) ? ` · ${capNote(preview)}` : ''}
        </p>
      </fieldset>

      <fieldset disabled={blocked} className="mt-3 space-y-2">
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Look</legend>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Background</span>
            <select aria-label="Background" value={background} onChange={(e) => setBackground(e.target.value as BackgroundChoice)} className={selectClass}>
              {BACKGROUNDS.map((b) => <option key={b} value={b}>{BACKGROUND_LABEL[b]}</option>)}
            </select>
          </label>
          {background === 'custom' && (
            <input type="color" aria-label="Background colour" value={bgColour} onChange={(e) => setBgColour(e.target.value)} className="h-6 w-8" />
          )}
          <label className={label}>
            <span>Colour atoms by</span>
            <select aria-label="Colour atoms by" value={colourBy} onChange={(e) => setColourBy(e.target.value as ColourMode)} className={selectClass}>
              {COLOUR_MODES.map((m) => {
                const needsDt = m === 'speed' || m === 'ke';
                return (
                  <option key={m} value={m} disabled={needsDt && !speedUsable}>
                    {COLOUR_LABEL[m]}{needsDt && !speedUsable ? ' (needs a timestep)' : ''}
                  </option>
                );
              })}
            </select>
          </label>
          {colourBy === 'single' && (
            <input type="color" aria-label="Atom colour" value={singleColour} onChange={(e) => setSingleColour(e.target.value)} className="h-6 w-8" />
          )}
          {scalarColour && (
            <label className={label}>
              <input type="checkbox" checked={colourBar} onChange={(e) => setColourBar(e.target.checked)} aria-label="Colour bar" />
              <span>Colour bar</span>
            </label>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Atom size</span>
            <select aria-label="Atom radius scale in the render" value={radiusScale} onChange={(e) => setRadiusScale(Number(e.target.value))} className={selectClass}>
              {RADIUS_SCALE_CHOICES.map((r) => <option key={r} value={r}>{r}×</option>)}
            </select>
          </label>
          <label className={label}>
            <span>Box</span>
            <select aria-label="Simulation box in the render" value={box} onChange={(e) => setBox(e.target.value as BoxChoice)} className={selectClass}>
              {BOXES.map((b) => <option key={b} value={b}>{BOX_LABEL[b]}</option>)}
            </select>
          </label>
          <label className={label}>
            <input type="checkbox" checked={axes} onChange={(e) => setAxes(e.target.checked)} aria-label="Axes triad" />
            <span>Axes triad</span>
          </label>
        </div>
      </fieldset>

      <fieldset disabled={blocked} className="mt-3 space-y-2">
        <legend className={`text-[11px] font-semibold ${ct.muted}`}>Camera and overlays</legend>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Camera</span>
            <select aria-label="Camera path" value={camera} onChange={(e) => setCamera(e.target.value as CameraMode)} className={selectClass}>
              {CAMERAS.map((c) => <option key={c} value={c}>{CAMERA_LABEL[c]}</option>)}
            </select>
          </label>
          {camera === 'orbit' && (
            <label className={label} title="Degrees the camera turns about the vertical axis over the clip">
              <span>Turn</span>
              <input type="number" min={0} max={3600} step={5} inputMode="decimal" aria-label="Orbit degrees over the clip" value={orbitS}
                onChange={(e) => setOrbitS(e.target.value)} className={fieldClass} />
              <span>°</span>
            </label>
          )}
          <label className={label}>
            <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} aria-label="Step and time overlay" />
            <span>Step and time</span>
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Title</span>
            <input type="text" maxLength={80} value={title} aria-label="Title overlay" onChange={(e) => setTitle(e.target.value)} className={textClass} />
          </label>
          <label className={label}>
            <span>Caption</span>
            <input type="text" maxLength={120} value={caption} aria-label="Caption overlay" onChange={(e) => setCaption(e.target.value)} className={textClass} />
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
          <label className={label}>
            <span>Text corner</span>
            <select aria-label="Text corner" value={corner} onChange={(e) => setCorner(e.target.value as Corner)} className={selectClass}>
              {CORNERS.map((c) => <option key={c} value={c}>{CORNER_LABEL[c]}</option>)}
            </select>
          </label>
          <label className={label}>
            <span>Font size</span>
            <select aria-label="Font size" value={fontSize} onChange={(e) => setFontSize(e.target.value as FontSize)} className={selectClass}>
              {FONTS.map((f) => <option key={f} value={f}>{FONT_LABEL[f]}</option>)}
            </select>
          </label>
        </div>
      </fieldset>

      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <button className={`${btn} ${ct.button}`} disabled={busy || count === 0} onClick={snapshot}
          title="Save the current view as a PNG (at its current size, without overlay)">
          <Camera size={13} aria-hidden="true" />Snapshot
        </button>
        <button className={`${btn} ${ct.button}`} disabled={blocked || !selection.ok || tooLong} onClick={() => void highResPng()}
          title="Render the shown frame as one PNG at the chosen size, with the look and overlays">
          <ImageIcon size={13} aria-hidden="true" />High-res PNG
        </button>
        <button className={`${btn} ${ct.button}`} disabled={blocked || !selection.ok || tooLong} onClick={() => void exportFrames()}
          title="Render the planned frames to PNG files and download them as one zip">
          <Images size={13} aria-hidden="true" />Export frames (.zip)
        </button>
        <button className={`${btn} ${ct.button}`} disabled={blocked || !selection.ok || tooLong} onClick={() => void exportGif()}
          title="Render the planned frames to an animated GIF (256 colours per frame; 720p or smaller keeps the file modest)">
          <FileImage size={13} aria-hidden="true" />Export GIF
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
            <span>Quality</span>
            <select aria-label="Video quality" value={quality} onChange={(e) => setQuality(e.target.value as VideoQuality)} className={selectClass}>
              {QUALITIES.map((q) => <option key={q} value={q}>{QUALITY_LABEL[q]}</option>)}
            </select>
          </label>
        </div>
      </fieldset>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        {progress === null ? (
          <button className={`${btn} ${ct.accent}`} disabled={blocked || !selection.ok || !format || tooLong} onClick={() => void recordVideo()}
            title="Play the planned frames as a video of the 3D view, at the chosen size, FPS and quality">
            <Film size={13} aria-hidden="true" />Record video
          </button>
        ) : (
          <>
            <span role="status" aria-live="polite" className={`text-xs ${ct.muted}`}>
              {progress.kind === 'video' ? 'Recording' : progress.kind === 'gif' ? 'Encoding GIF' : 'Rendering'} frame {progress.done} of {progress.total}
              {' '}({Math.round((100 * progress.done) / Math.max(1, progress.total))}%)
            </span>
            <progress max={progress.total} value={progress.done} aria-label="Render progress" className="h-1.5 w-32" />
            <button className={`${btn} ${ct.button}`} onClick={cancel}>Cancel</button>
          </>
        )}
      </div>
      {!format && (
        <p className={`mt-1 text-[11px] ${ct.muted}`}>No video recording in this browser; frame export and GIF still work.</p>
      )}
      <p role="status" aria-live="polite" className={`mt-2 text-[11px] ${ct.muted}`}>{message ?? ''}</p>
    </section>
  );
};

export default RenderPanel;
