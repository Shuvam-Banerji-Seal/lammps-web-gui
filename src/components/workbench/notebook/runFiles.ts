import { dataUrlBytes, isImageDataUrl } from '../../../lammps/exporter';
import { formatNumber } from '../../../engine/script';

/**
 * Pure helpers for the run-output panel: what kind of file the engine wrote, how many frames a trajectory has,
 * how big it is, and the text exports (run log, thermo CSV). No DOM here, so they are unit-tested directly.
 */

export type FileKind = 'trajectory' | 'data' | 'restart' | 'image' | 'table' | 'text';

export const FILE_KIND_LABEL: Record<FileKind, string> = {
  trajectory: 'trajectory',
  data: 'data',
  restart: 'restart',
  image: 'image',
  table: 'table',
  text: 'text',
};

const extOf = (name: string): string => {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i + 1).toLowerCase();
};

const TRAJECTORY_EXT = new Set(['lammpstrj', 'dump', 'xyz', 'trj', 'traj']);
const DATA_EXT = new Set(['data', 'lmp', 'dat', 'pdb', 'cif']);
const TABLE_EXT = new Set(['csv', 'tsv', 'avg']);

/** Kind of a file the engine wrote, from its name and content (images are data URLs). */
/**
 * A gzip file the engine stored one character per byte (write_data or a dump with a .gz name): the gzip magic 1f 8b,
 * which no text file starts with.
 */
export const isGzipText = (text: string): boolean => text.charCodeAt(0) === 0x1f && text.charCodeAt(1) === 0x8b;

/** The binary trajectory formats the engine writes (dump dcd and xtc), stored one character per byte. */
const BINARY_TRAJECTORY_EXT = new Set(['dcd', 'xtc']);

/** A DCD or XTC file: binary trajectory frames, which the 3D Viewer does not read. */
export const isBinaryTrajectory = (name: string): boolean => BINARY_TRAJECTORY_EXT.has(extOf(name));

/** A file whose characters are bytes (gzip, DCD, XTC), not text to preview or encode as UTF-8. */
export const isBinaryFile = (name: string, text: string): boolean => isGzipText(text) || isBinaryTrajectory(name);

export const classifyFile = (name: string, text: string): FileKind => {
  if (isImageDataUrl(text)) return 'image';
  if (isBinaryTrajectory(name)) return 'trajectory';
  if (isGzipText(text)) {
    // classified by the name without .gz (the viewer decompresses it); its bytes are not text to preview
    const inner = name.replace(/\.gz$/i, '');
    const ext = extOf(inner);
    if (TRAJECTORY_EXT.has(ext)) return 'trajectory';
    if (DATA_EXT.has(ext) || /(^|\/)data\./i.test(inner)) return 'data';
    return 'text';
  }
  const ext = extOf(name);
  if (TRAJECTORY_EXT.has(ext) || /^ITEM: TIMESTEP/m.test(text.slice(0, 200))) return 'trajectory';
  if (/restart/i.test(name) || ext === 'rst') return 'restart';
  if (/^log/i.test(name)) return 'text'; // log files (log.lammps) are text even though they hold numbers
  if (DATA_EXT.has(ext) || /\b(xlo|ylo|zlo)\s+(xhi|yhi|zhi)\b/.test(text.slice(0, 4000))) return 'data';
  if (TABLE_EXT.has(ext)) return 'table';
  // fix ave/* and compute outputs start with a '#' header naming the averaging or the columns
  const firstLine = text.slice(0, 300).split('\n', 1)[0] ?? '';
  if (firstLine.startsWith('#') && /ave|averag|chunk|row|column/i.test(firstLine)) return 'table';
  return 'text';
};

/** Frames in a dump (ITEM: TIMESTEP blocks) or an XYZ file (count line, comment line, count atom lines). */
export const countFrames = (name: string, text: string): number | null => {
  if (isImageDataUrl(text) || isGzipText(text)) return null;
  if (isBinaryTrajectory(name)) return binaryFrames(extOf(name), text);
  if (/^ITEM: TIMESTEP/m.test(text)) {
    let n = 0;
    const re = /^ITEM: TIMESTEP[ \t]*\r?$/gm;
    while (re.exec(text)) n++;
    return n;
  }
  if (extOf(name) !== 'xyz') return null;
  const lines = text.split('\n');
  let frames = 0;
  let i = 0;
  while (i < lines.length) {
    const t = lines[i].trim();
    if (t === '') { i++; continue; }
    const count = Number(t);
    if (!Number.isInteger(count) || count < 0) break;
    frames++;
    i += 2 + count;
  }
  return frames;
};

/**
 * Frames of a binary trajectory: a DCD file's NFILE header field (the 32-bit integer at byte 8, little-endian), or the
 * frames of an XTC file walked one by one (magic 1995, natoms, step, time, 9 box floats, then natoms again and either
 * 3 natoms floats, for 9 atoms or fewer, or the precision, 6 integer bounds, the small index and the byte count of the
 * compressed coordinates, padded to 4 bytes). null when the bytes are not such a file.
 */
const binaryFrames = (ext: string, text: string): number | null => {
  const u32 = (at: number, little: boolean): number => {
    const b = [0, 1, 2, 3].map((k) => text.charCodeAt(at + k) & 0xff);
    return (little ? (b[3] << 24) | (b[2] << 16) | (b[1] << 8) | b[0] : (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]) >>> 0;
  };
  if (ext === 'dcd') return text.length >= 12 && text.slice(4, 8) === 'CORD' ? u32(8, true) : null;
  let at = 0;
  let frames = 0;
  while (at < text.length) {
    if (at + 56 > text.length || u32(at, false) !== 1995) return null;
    const n = u32(at + 52, false);
    at += 56;
    if (n <= 9) at += 12 * n;
    else {
      if (at + 36 > text.length) return null;
      const bytes = u32(at + 32, false);
      at += 36 + bytes + ((4 - (bytes % 4)) % 4);
    }
    if (at > text.length) return null;
    frames++;
  }
  return frames;
};

/** Size in bytes: decoded size for an image data URL, UTF-8 length for text, one byte per character for binary files. */
export const fileBytesSize = (text: string, name = ''): number => {
  if (isBinaryFile(name, text)) return text.length;
  if (isImageDataUrl(text)) {
    const comma = text.indexOf(',');
    const b64 = text.slice(comma + 1);
    const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
    return Math.max(0, Math.floor((b64.length * 3) / 4) - pad);
  }
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0x80) n += 1;
    else if (c < 0x800) n += 2;
    else if (c >= 0xd800 && c <= 0xdbff) { n += 4; i++; }
    else n += 3;
  }
  return n;
};

/** Bytes to store in a zip: the decoded binary for an image data URL, UTF-8 for text. */
export const fileBytes = (text: string, name = ''): Uint8Array<ArrayBuffer> => {
  const bin = dataUrlBytes(text);
  if (bin) return bin.bytes;
  // a gzip, DCD or XTC file is one character per byte: its bytes, not the UTF-8 encoding of those characters
  if (isBinaryFile(name, text)) {
    const out = new Uint8Array(new ArrayBuffer(text.length));
    for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i) & 0xff;
    return out;
  }
  return new TextEncoder().encode(text);
};

/** Whether the viewer can open a file (structures and trajectories). */
export const canOpenInViewer = (kind: FileKind): boolean => kind === 'trajectory' || kind === 'data';

/** A cell's output as the notebook keeps it. */
export interface RunOutput {
  logs: string[];
  error?: string | null;
  tables: { keywords: string[]; rows: Record<string, number>[] }[];
}

const INTEGER_KEYS = new Set(['step', 'elapsed', 'atoms']);
const fmtLog = (k: string, v: number | undefined): string =>
  v === undefined ? '' : INTEGER_KEYS.has(k) ? String(v) : formatNumber(v, '%.6g');

/**
 * The run as text, cell by cell in notebook order: each cell's log lines, then its thermo tables
 * (header row of keywords, then one row per thermo output).
 */
export const runLogText = (cells: { id: string }[], runs: Record<string, RunOutput | undefined>): string => {
  const out: string[] = [];
  cells.forEach((cell, i) => {
    const run = runs[cell.id];
    if (!run) return;
    out.push(`# cell ${i + 1}`);
    for (const line of run.logs) out.push(line);
    if (run.error) out.push(`ERROR: ${run.error}`);
    for (const t of run.tables) {
      out.push(t.keywords.join(' '));
      for (const r of t.rows) out.push(t.keywords.map((k) => fmtLog(k, r[k])).join(' '));
    }
    out.push('');
  });
  return out.join('\n');
};

/** The thermo rows of the latest thermo table (last cell in notebook order that has one), as CSV; null if none. */
export const thermoCsv = (cells: { id: string }[], runs: Record<string, RunOutput | undefined>): string | null => {
  let last: RunOutput['tables'][number] | null = null;
  for (const cell of cells) {
    const tables = runs[cell.id]?.tables ?? [];
    if (tables.length) last = tables[tables.length - 1];
  }
  if (!last) return null;
  const lines = [last.keywords.join(',')];
  for (const r of last.rows) lines.push(last.keywords.map((k) => (r[k] === undefined ? '' : String(r[k]))).join(','));
  return lines.join('\n') + '\n';
};
