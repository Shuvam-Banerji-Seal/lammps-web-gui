import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, ChevronLeft, ChevronRight, Video, Download, Box, Clapperboard } from 'lucide-react';
import type { getThemeTokens } from '../../../theme';
import { downloadFile } from '../../../lammps/exporter';
import { startCanvasRecording, type RecordingHandle } from '../../../services/recorder';
import { framesToDump, framesToExtXYZ, type TrajectoryBuffer } from './trajectory';

type Tokens = ReturnType<typeof getThemeTokens>;

const SPEEDS = [1, 2, 5, 10, 20, 30] as const;
const VIDEO_FPS = [10, 15, 25, 30] as const;

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));

export interface TrajectoryPlayerProps {
  ct: Tokens;
  btn: string;
  buffer: TrajectoryBuffer;
  /** Changes whenever the buffer changes (the buffer itself is mutable). */
  version: number;
  /** Index of the frame shown; null follows the newest recorded frame. */
  index: number | null;
  /** A run is going: the view follows the live frame and playback stops. */
  live: boolean;
  /** Shows recorded frame i (or null to follow the newest). */
  onShow: (index: number | null) => void;
  /** The element that holds the 3D canvas (video records its first canvas). */
  canvasHost: React.RefObject<HTMLDivElement | null>;
  /** True while a video is being recorded: the 3D view keeps rendering. */
  onVideoActive: (on: boolean) => void;
  /** Opens the trajectory in the Molecule 3D Viewer as a dump file. */
  onOpenInViewer?: (file: { name: string; text: string }) => void;
  /** Opens the Render panel (snapshot, frame export, video options). */
  onRender?: () => void;
}

/**
 * Player for the recorded trajectory under the 3D view: transport, scrub, speed, loop, video recording of the
 * frames (start to end, through recorder.ts) and export (extended XYZ or LAMMPS dump text).
 */
const TrajectoryPlayer: React.FC<TrajectoryPlayerProps> = ({
  ct, btn, buffer, version, index, live, onShow, canvasHost, onVideoActive, onOpenInViewer, onRender,
}) => {
  const count = buffer.length;
  const shown = count === 0 ? -1 : Math.min(count - 1, index ?? count - 1);
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState<number>(10);
  const [loop, setLoop] = useState(false);
  const [videoFps, setVideoFps] = useState<number>(25);
  const [video, setVideo] = useState<{ done: number; total: number } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<'xyz' | 'dump'>('xyz');
  const cancelVideo = useRef(false);
  // the timers read the current index without re-subscribing on every frame
  const shownRef = useRef(shown);
  shownRef.current = shown;

  // a run starting stops playback; the view follows it
  useEffect(() => {
    if (live) setPlaying(false);
  }, [live]);

  // playback: one recorded frame per tick
  useEffect(() => {
    if (!playing || live || count === 0) return;
    const id = window.setInterval(() => {
      const next = shownRef.current + 1;
      if (next >= buffer.length) {
        if (loop && buffer.length > 1) onShow(0);
        else setPlaying(false);
        return;
      }
      onShow(next);
    }, 1000 / fps);
    return () => window.clearInterval(id);
    // count and version are the buffer's identity for this effect
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, fps, loop, live, count, version, onShow]);

  const frameStep = (k: number) => (k >= 0 && k < count ? buffer.get(k)?.step ?? 0 : 0);

  const togglePlay = () => {
    if (playing) { setPlaying(false); return; }
    if (count === 0) return;
    // at the end (and not looping) play starts again from the first frame
    if (shown >= count - 1 && !loop) onShow(0);
    setPlaying(true);
  };

  const stepTo = (k: number) => {
    setPlaying(false);
    onShow(Math.max(0, Math.min(count - 1, k)));
  };

  const download = (name: string, text: string) => downloadFile(name, text);
  const exportTrajectory = () => {
    const frames = buffer.list();
    if (!frames.length) return;
    if (exportFormat === 'xyz') download('trajectory.xyz', framesToExtXYZ(frames));
    else download('trajectory.dump', framesToDump(frames));
  };
  const openInViewer = () => {
    const frames = buffer.list();
    if (!frames.length || !onOpenInViewer) return;
    onOpenInViewer({ name: 'trajectory.dump', text: framesToDump(frames) });
  };

  /** Plays the recorded frames from first to last while the canvas is captured (MP4 where supported, else WebM). */
  const recordVideo = async () => {
    if (count === 0 || video) return;
    const canvas = canvasHost.current?.querySelector('canvas');
    if (!canvas) { setMessage('No 3D view to record yet.'); return; }
    setPlaying(false);
    setMessage(null);
    let handle: RecordingHandle;
    try {
      handle = startCanvasRecording(canvas as HTMLCanvasElement, { fps: videoFps });
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Recording failed to start.');
      return;
    }
    const restoreIndex = index;
    cancelVideo.current = false;
    onVideoActive(true);
    setVideo({ done: 0, total: count });
    const tick = 1000 / videoFps;
    let cancelled = false;
    try {
      for (let k = 0; k < count; k++) {
        if (cancelVideo.current) { cancelled = true; break; }
        onShow(k);
        setVideo({ done: k + 1, total: count });
        await sleep(tick);
      }
    } finally {
      const result = await handle.stop();
      if (!cancelled && !cancelVideo.current) {
        const url = URL.createObjectURL(result.blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `trajectory-${new Date().toISOString().replace(/[:.]/g, '-')}.${result.ext}`;
        a.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 5000);
        setMessage(`Video saved (${result.label}, ${count} frames at ${videoFps} fps).`);
      } else {
        setMessage('Video cancelled.');
      }
      setVideo(null);
      onVideoActive(false);
      onShow(restoreIndex);
    }
  };

  const cancel = () => { cancelVideo.current = true; };
  const busy = video !== null;
  const shownFrame = shown >= 0 ? buffer.get(shown) : undefined;
  const thinned = buffer.thinned > 0;

  return (
    <section aria-label="Trajectory player" className={`space-y-1.5 border-t px-2 py-2 ${ct.divider}`}>
      <div className="flex flex-wrap items-center gap-1.5">
        <button className={`${btn} ${ct.button}`} aria-label="First frame" title="First frame"
          disabled={count === 0 || busy} onClick={() => stepTo(0)}>
          <SkipBack size={13} aria-hidden="true" />
        </button>
        <button className={`${btn} ${ct.button}`} aria-label="Previous frame" title="Previous frame"
          disabled={count === 0 || busy} onClick={() => stepTo(shown - 1)}>
          <ChevronLeft size={13} aria-hidden="true" />
        </button>
        <button className={`${btn} ${ct.accent}`} aria-label={playing ? 'Pause' : 'Play'} title={playing ? 'Pause' : 'Play the recorded frames'}
          disabled={count === 0 || busy || live} onClick={togglePlay}>
          {playing ? <Pause size={13} aria-hidden="true" /> : <Play size={13} aria-hidden="true" />}
          {playing ? 'Pause' : 'Play'}
        </button>
        <button className={`${btn} ${ct.button}`} aria-label="Next frame" title="Next frame"
          disabled={count === 0 || busy} onClick={() => stepTo(shown + 1)}>
          <ChevronRight size={13} aria-hidden="true" />
        </button>
        <button className={`${btn} ${ct.button}`} aria-label="Last frame" title="Last frame"
          disabled={count === 0 || busy} onClick={() => stepTo(count - 1)}>
          <SkipForward size={13} aria-hidden="true" />
        </button>
        <label className={`flex items-center gap-1 text-xs ${ct.muted}`}>
          <span>Speed</span>
          <select aria-label="Playback speed (frames per second)" value={fps} onChange={(e) => setFps(Number(e.target.value))}
            disabled={busy} className={`min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s} fps</option>)}
          </select>
        </label>
        <label className={`flex items-center gap-1 text-xs ${ct.muted}`}>
          <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} aria-label="Loop playback" />
          <span>Loop</span>
        </label>
        <span className={`ml-auto font-mono text-[11px] ${ct.muted}`} aria-live="polite">
          {count === 0
            ? 'no frames yet'
            : live
              ? `live · frame ${count} · step ${shownFrame?.step ?? 0}`
              : `frame ${shown + 1} of ${count} · step ${shownFrame?.step ?? frameStep(shown)}`}
        </span>
      </div>

      <input type="range" min={0} max={Math.max(0, count - 1)} step={1} value={Math.max(0, shown)}
        disabled={count <= 1 || live || busy} aria-label="Scrub frames" aria-valuetext={count ? `frame ${shown + 1} of ${count}, step ${shownFrame?.step ?? 0}` : 'no frames'}
        onChange={(e) => stepTo(Number(e.target.value))} className="w-full" />

      {thinned && (
        <p role="status" className={`text-[11px] ${ct.muted}`}>
          Over the memory cap, so every other frame was dropped: now one frame in {buffer.stride} is kept.
        </p>
      )}

      <div className="flex flex-wrap items-center gap-1.5">
        <label className={`flex items-center gap-1 text-xs ${ct.muted}`}>
          <span>Video fps</span>
          <select aria-label="Video frames per second" value={videoFps} onChange={(e) => setVideoFps(Number(e.target.value))}
            disabled={busy} className={`min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`}>
            {VIDEO_FPS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        {!busy ? (
          <button className={`${btn} ${ct.button}`} disabled={count === 0 || live} onClick={() => void recordVideo()}
            title="Play every recorded frame from start to end and save the 3D view as a video (MP4 where the browser supports it, else WebM)">
            <Video size={13} aria-hidden="true" />Record video
          </button>
        ) : (
          <>
            <span role="status" className={`text-xs ${ct.muted}`} aria-live="polite">
              Recording frame {video.done} of {video.total} ({Math.round((100 * video.done) / video.total)}%)
            </span>
            <progress max={video.total} value={video.done} aria-label="Video progress" className="h-1.5 w-24" />
            <button className={`${btn} ${ct.button}`} onClick={cancel}>Cancel video</button>
          </>
        )}
        {message && <span role="status" className={`text-xs ${ct.muted}`}>{message}</span>}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <label className={`flex items-center gap-1 text-xs ${ct.muted}`}>
          <span>Format</span>
          <select aria-label="Trajectory export format" value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as 'xyz' | 'dump')}
            className={`min-h-6 rounded border px-1 py-0.5 text-xs ${ct.input}`}>
            <option value="xyz">Extended XYZ</option>
            <option value="dump">LAMMPS dump</option>
          </select>
        </label>
        <button className={`${btn} ${ct.button}`} disabled={count === 0 || live || busy} onClick={exportTrajectory}
          title="Download every recorded frame">
          <Download size={13} aria-hidden="true" />Download trajectory
        </button>
        {onRender && (
          <button className={`${btn} ${ct.button}`} disabled={live || busy} onClick={onRender}
            title="Render panel: snapshot, frame range as PNG files, or a video with size, overlay and orbit options">
            <Clapperboard size={13} aria-hidden="true" />Render
          </button>
        )}
        {onOpenInViewer && (
          <button className={`${btn} ${ct.button}`} disabled={count === 0 || live || busy} onClick={openInViewer}
            title="Open the recorded frames in the Molecule 3D Viewer (RDF, MSD and measurements there)">
            <Box size={13} aria-hidden="true" />Open in 3D Viewer
          </button>
        )}
      </div>
    </section>
  );
};

export default TrajectoryPlayer;
