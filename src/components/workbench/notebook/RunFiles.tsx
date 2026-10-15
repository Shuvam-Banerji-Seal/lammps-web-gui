import React, { useMemo, useState } from 'react';
import { Download, Eye, EyeOff, ExternalLink, Archive, ScrollText, Table2, Clapperboard } from 'lucide-react';
import type { getThemeTokens } from '../../../theme';
import { downloadFile, downloadTextFile } from '../../../lammps/exporter';
import { buildZip } from '../../../lammps/zip';
import {
  canOpenInViewer, classifyFile, countFrames, fileBytes, fileBytesSize, runLogText, thermoCsv,
  type FileKind, type RunOutput,
} from './runFiles';

/** Lines in a text preview by default, and the most a "show more" preview shows. */
const PREVIEW_LINES = 40;
const PREVIEW_MAX = 2000;

const formatBytes = (n: number): string =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;

const downloadBlob = (filename: string, bytes: Uint8Array<ArrayBuffer>, mime: string): void => {
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 4000);
};

interface RunFilesProps {
  ct: ReturnType<typeof getThemeTokens>;
  /** Every file the engine wrote this session: name -> text (images as data URLs). */
  files: Record<string, string>;
  /** Cells in notebook order, and each cell's output, for the log and CSV downloads. */
  cells: { id: string }[];
  runs: Record<string, RunOutput | undefined>;
  /** Opens a structure or trajectory file in the 3D Viewer. */
  onOpenInViewer?: (file: { name: string; text: string }) => void;
  /** Loads a LAMMPS dump the run wrote into the notebook's player and opens the Render panel on it. */
  onRenderFile?: (file: { name: string; text: string }) => void;
}

/** A LAMMPS dump text (ITEM: TIMESTEP blocks), the trajectories the Render panel can load. */
const isDumpText = (kind: FileKind, text: string): boolean =>
  kind === 'trajectory' && /^ITEM: TIMESTEP/m.test(text.slice(0, 200));

interface Entry {
  name: string;
  text: string;
  kind: FileKind;
  size: number;
  frames: number | null;
}

const RunFiles: React.FC<RunFilesProps> = ({ ct, files, cells, runs, onOpenInViewer, onRenderFile }) => {
  const [imagePick, setImagePick] = useState<number | null>(null);
  const [preview, setPreview] = useState<{ name: string; full: boolean } | null>(null);
  const [note, setNote] = useState<string | null>(null);

  const entries = useMemo<Entry[]>(() => Object.keys(files).map((name) => {
    const text = files[name];
    const kind = classifyFile(name, text);
    return { name, text, kind, size: fileBytesSize(text), frames: countFrames(name, text) };
  }), [files]);
  const images = useMemo(() => entries.filter((e) => e.kind === 'image'), [entries]);
  const hasLog = useMemo(() => Object.values(runs).some((r) => r && (r.logs.length > 0 || r.tables.length > 0)), [runs]);

  if (entries.length === 0 && !hasLog) return null;

  const btn = `inline-flex min-h-6 min-w-6 items-center justify-center gap-1 rounded px-2 py-1 text-xs font-medium transition-colors disabled:opacity-50 ${ct.hoverSurface}`;
  const i = images.length ? Math.min(imagePick ?? images.length - 1, images.length - 1) : 0;
  const shownImage = images[i];
  const previewEntry = preview ? entries.find((e) => e.name === preview.name) : undefined;

  const zipAll = () => {
    const bytes = buildZip(entries.map((e) => ({ name: e.name, data: fileBytes(e.text) })));
    downloadBlob('lammps-run-files.zip', bytes, 'application/zip');
    setNote(`Saved ${entries.length} file${entries.length === 1 ? '' : 's'} to a zip.`);
  };
  const downloadLog = () => {
    downloadTextFile('run-log.txt', runLogText(cells, runs));
    setNote('Saved the run log.');
  };
  const downloadCsv = () => {
    const csv = thermoCsv(cells, runs);
    if (csv === null) return;
    downloadTextFile('thermo.csv', csv);
    setNote('Saved the thermo CSV.');
  };

  const previewLines = previewEntry ? previewEntry.text.split('\n') : [];
  const previewCap = preview?.full ? PREVIEW_MAX : PREVIEW_LINES;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <p className={`mr-auto text-[11px] font-semibold ${ct.muted}`}>Run output</p>
        {entries.length > 0 && (
          <button className={`${btn} ${ct.button}`} onClick={zipAll} title="Every file this session wrote, in one zip">
            <Archive size={12} aria-hidden="true" />Download all (.zip)
          </button>
        )}
        {hasLog && (
          <button className={`${btn} ${ct.button}`} onClick={downloadLog}
            title="Log lines and thermo tables of every cell, in cell order">
            <ScrollText size={12} aria-hidden="true" />Download log
          </button>
        )}
        <button className={`${btn} ${ct.button}`} onClick={downloadCsv} disabled={!thermoCsv(cells, runs)}
          title="The thermo rows of the latest thermo table, as CSV">
          <Table2 size={12} aria-hidden="true" />Thermo CSV
        </button>
      </div>
      <p role="status" aria-live="polite" className={`text-[11px] ${ct.muted}`}>{note ?? ''}</p>

      {shownImage && (
        <div>
          <p className={`text-[11px] font-semibold ${ct.muted}`}>Images written ({images.length})</p>
          <img src={shownImage.text} alt={`dump image ${shownImage.name}`} className={`mt-1 max-h-80 max-w-full rounded border ${ct.divider}`} />
          {images.length > 1 && (
            <input type="range" min={0} max={images.length - 1} value={i} className="mt-1 w-full"
              aria-label={`Image ${i + 1} of ${images.length}`}
              onChange={(e) => { const k = Number(e.target.value); setImagePick(k === images.length - 1 ? null : k); }} />
          )}
        </div>
      )}

      {entries.length > 0 && (
        <div>
          <p className={`text-[11px] font-semibold ${ct.muted}`}>Files written ({entries.length})</p>
          <ul className="mt-1 space-y-1" aria-label="Files written">
            {entries.map((e) => {
              const isOpen = preview?.name === e.name;
              const textual = e.kind !== 'image';
              return (
                <li key={e.name} className={`rounded border p-1.5 ${ct.divider}`}>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                    <span className="min-w-0 break-all font-mono">{e.name}</span>
                    <span className={`rounded px-1.5 py-0.5 text-[10px] ${ct.chip}`}>{e.kind}</span>
                    <span className={ct.muted}>{formatBytes(e.size)}</span>
                    {e.frames !== null && <span className={ct.muted}>{e.frames} {e.frames === 1 ? 'frame' : 'frames'}</span>}
                    <span className="ml-auto flex flex-wrap items-center gap-1">
                      {textual && (
                        <button className={`${btn} ${ct.button}`} aria-expanded={isOpen} aria-label={`${isOpen ? 'Hide' : 'Preview'} ${e.name}`}
                          onClick={() => setPreview(isOpen ? null : { name: e.name, full: false })}>
                          {isOpen ? <EyeOff size={12} aria-hidden="true" /> : <Eye size={12} aria-hidden="true" />}
                          {isOpen ? 'Hide' : 'Preview'}
                        </button>
                      )}
                      {onOpenInViewer && canOpenInViewer(e.kind) && (
                        <button className={`${btn} ${ct.button}`} aria-label={`Open ${e.name} in 3D Viewer`}
                          title="Open in the 3D Viewer"
                          onClick={() => onOpenInViewer({ name: e.name, text: e.text })}>
                          <ExternalLink size={12} aria-hidden="true" />Open in 3D Viewer
                        </button>
                      )}
                      {onRenderFile && isDumpText(e.kind, e.text) && (
                        <button className={`${btn} ${ct.button}`} aria-label={`Render video from ${e.name}`}
                          title="Load this dump into the player and open the Render panel to make a video of it"
                          onClick={() => onRenderFile({ name: e.name, text: e.text })}>
                          <Clapperboard size={12} aria-hidden="true" />Render video
                        </button>
                      )}
                      <button className={`${btn} ${ct.button}`} aria-label={`Download ${e.name}`}
                        onClick={() => downloadFile(e.name, e.text)}>
                        <Download size={12} aria-hidden="true" />Download
                      </button>
                    </span>
                  </div>
                  {isOpen && previewEntry && (
                    <div className="mt-1.5">
                      <pre className={`max-h-64 overflow-auto rounded p-2 text-[11px] leading-snug ${ct.chip}`}
                        aria-label={`Preview of ${e.name}`}>
                        {previewLines.slice(0, previewCap).join('\n')}
                      </pre>
                      <p className={`mt-0.5 text-[11px] ${ct.muted}`}>
                        {previewLines.length > previewCap
                          ? `Lines 1 to ${previewCap} of ${previewLines.length}. `
                          : `All ${previewLines.length} lines. `}
                        {!preview?.full && previewLines.length > PREVIEW_LINES && (
                          <button className="underline" onClick={() => setPreview({ name: e.name, full: true })}>
                            Show more (up to {PREVIEW_MAX} lines)
                          </button>
                        )}
                      </p>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default RunFiles;
