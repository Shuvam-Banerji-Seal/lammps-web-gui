/**
 * Notebook files: the `.lmpnb` format (versioned JSON), the importers (`.lmpnb`, Jupyter `.ipynb`,
 * plain LAMMPS input) and the exporters (`.lmpnb`, plain LAMMPS input, Markdown). Pure: no DOM,
 * no React, so it is unit-tested directly (tests/notebookFile.test.ts).
 *
 * Format version 1 (a JSON object):
 *   { "format": "lammps-web-notebook", "version": 1, "name": "Melt", "created": ISO, "modified": ISO,
 *     "cells": [ { "id": "c1", "kind": "code" | "markdown", "text": "units lj" } ],
 *     "inputFiles": { "data.txt": "..." },            // optional, only when the user chose to include them
 *     "outputs": { "c1": { "logs": [], "error": null, "tables": [] } } }   // optional, off by default
 * A later version bumps `version` and adds a step to MIGRATIONS (from version N to N+1).
 */
import type { Cell } from './cells';

export const NOTEBOOK_FORMAT = 'lammps-web-notebook';
export const NOTEBOOK_VERSION = 1;
export const NOTEBOOK_EXTENSION = '.lmpnb';
/** Cell title every new notebook starts with. */
export const UNTITLED = 'Untitled notebook';

/** Thermo output of one table, as the notebook shows it. */
export interface OutputTable {
  keywords: string[];
  labels?: string[];
  units?: string;
  rows: Record<string, number>[];
}

/** The output of one cell: its log lines, the error (null if none) and its thermo tables. */
export interface CellOutput {
  logs: string[];
  error: string | null;
  tables: OutputTable[];
}

/** A parsed, validated notebook file. */
export interface NotebookDoc {
  name: string;
  created: string;
  modified: string;
  cells: Cell[];
  inputFiles?: Record<string, string>;
  outputs?: Record<string, CellOutput>;
}

/** Thrown with a message a person can act on (wrong format, newer version, malformed cell). */
export class NotebookFileError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'NotebookFileError';
  }
}

type Raw = Record<string, unknown>;
/** Migration step from version N to N+1, keyed by N. Version 1 is the first, so there are none yet. */
export type Migrations = Record<number, (raw: Raw) => Raw>;
export const MIGRATIONS: Migrations = {};

const isObject = (v: unknown): v is Raw => typeof v === 'object' && v !== null && !Array.isArray(v);

function fail(message: string): never {
  throw new NotebookFileError(message);
}

const isIso = (v: unknown): v is string => typeof v === 'string' && !Number.isNaN(Date.parse(v));

/** Checks one output table; throws with the cell id in the message. */
const checkTable = (where: string, t: unknown): OutputTable => {
  if (!isObject(t) || !Array.isArray(t.keywords) || !t.keywords.every((k) => typeof k === 'string')) {
    fail(`${where}: a thermo table needs a "keywords" list of names.`);
  }
  if (!Array.isArray(t.rows) || !t.rows.every((r) => isObject(r) && Object.values(r).every((x) => typeof x === 'number'))) {
    fail(`${where}: thermo rows must be objects of numbers.`);
  }
  const out: OutputTable = { keywords: t.keywords as string[], rows: t.rows as Record<string, number>[] };
  if (t.labels !== undefined) {
    if (!Array.isArray(t.labels) || !t.labels.every((l) => typeof l === 'string')) fail(`${where}: "labels" must be a list of strings.`);
    out.labels = t.labels as string[];
  }
  if (t.units !== undefined) {
    if (typeof t.units !== 'string') fail(`${where}: "units" must be a string.`);
    out.units = t.units as string;
  }
  return out;
};

/** Validates the `outputs` block of a version 1 file. Unknown cell ids are kept as they are. */
const checkOutputs = (raw: unknown): Record<string, CellOutput> => {
  if (!isObject(raw)) fail('"outputs" must be an object keyed by cell id.');
  const out: Record<string, CellOutput> = {};
  for (const [id, o] of Object.entries(raw as Raw)) {
    const where = `outputs of cell "${id}"`;
    if (!isObject(o)) fail(`${where}: expected an object with logs, error and tables.`);
    if (!Array.isArray(o.logs) || !o.logs.every((l) => typeof l === 'string')) fail(`${where}: "logs" must be a list of strings.`);
    if (o.error !== null && typeof o.error !== 'string') fail(`${where}: "error" must be a string or null.`);
    if (!Array.isArray(o.tables)) fail(`${where}: "tables" must be a list.`);
    out[id] = {
      logs: o.logs as string[],
      error: o.error as string | null,
      tables: (o.tables as unknown[]).map((t, k) => checkTable(`${where}, table ${k + 1}`, t)),
    };
  }
  return out;
};

/**
 * Parses and validates a `.lmpnb` file. Throws NotebookFileError with a message for: not JSON, not this
 * format, a version newer than this app (or not a whole number), a malformed cell or block, and duplicate
 * cell ids. Older versions are migrated step by step through `migrations`.
 */
export const parseNotebookFile = (text: string, migrations: Migrations = MIGRATIONS, current: number = NOTEBOOK_VERSION): NotebookDoc => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    fail('This file is not JSON, so it is not a notebook file.');
  }
  if (!isObject(parsed)) fail('This file is not a notebook: its top level is not an object.');
  let raw = parsed as Raw;
  if (raw.format !== NOTEBOOK_FORMAT) {
    fail(`This is not a LAMMPS web notebook (format is ${JSON.stringify(raw.format ?? null)}, expected "${NOTEBOOK_FORMAT}").`);
  }
  const version = raw.version;
  if (typeof version !== 'number' || !Number.isInteger(version) || version < 1) {
    fail(`The notebook's "version" must be a whole number from 1 up, not ${JSON.stringify(version ?? null)}.`);
  }
  if (version > current) {
    fail(`This notebook was saved by a newer version of the notebook format (version ${version}). This app reads version ${current}; update the app to open it.`);
  }
  for (let v = version as number; v < current; v++) {
    const step = migrations[v];
    if (!step) fail(`No migration from notebook format version ${v} to ${v + 1}.`);
    raw = step(raw);
  }
  // Version 1 fields
  if (typeof raw.name !== 'string' || raw.name.trim() === '') fail('The notebook needs a non-empty "name".');
  if (!isIso(raw.created)) fail('"created" must be an ISO date-time string.');
  if (!isIso(raw.modified)) fail('"modified" must be an ISO date-time string.');
  if (!Array.isArray(raw.cells) || raw.cells.length === 0) fail('"cells" must be a non-empty list.');
  const seen = new Set<string>();
  const cells: Cell[] = raw.cells.map((c, i) => {
    const n = i + 1;
    if (!isObject(c)) fail(`Cell ${n} is not an object.`);
    if (typeof c.id !== 'string' || c.id === '') fail(`Cell ${n} needs a non-empty string "id".`);
    if (seen.has(c.id)) fail(`Cell ${n}: the id "${c.id}" is used by another cell.`);
    seen.add(c.id);
    if (c.kind !== 'code' && c.kind !== 'markdown') fail(`Cell ${n} ("${c.id}"): "kind" must be "code" or "markdown", not ${JSON.stringify(c.kind ?? null)}.`);
    if (typeof c.text !== 'string') fail(`Cell ${n} ("${c.id}"): "text" must be a string.`);
    return c.kind === 'markdown' ? { id: c.id, kind: 'markdown', text: c.text } : { id: c.id, kind: 'code', text: c.text };
  });
  const doc: NotebookDoc = { name: raw.name, created: raw.created as string, modified: raw.modified as string, cells };
  if (raw.inputFiles !== undefined) {
    if (!isObject(raw.inputFiles) || !Object.values(raw.inputFiles).every((t) => typeof t === 'string')) {
      fail('"inputFiles" must map file names to their text.');
    }
    doc.inputFiles = { ...(raw.inputFiles as Record<string, string>) };
  }
  if (raw.outputs !== undefined) doc.outputs = checkOutputs(raw.outputs);
  return doc;
};

/** The text of a `.lmpnb` file: pretty JSON with a final newline. Cells always carry their kind. */
export const serializeNotebookFile = (doc: NotebookDoc): string => {
  const out: Raw = {
    format: NOTEBOOK_FORMAT,
    version: NOTEBOOK_VERSION,
    name: doc.name,
    created: doc.created,
    modified: doc.modified,
    cells: doc.cells.map((c) => ({ id: c.id, kind: c.kind === 'markdown' ? 'markdown' : 'code', text: c.text })),
  };
  if (doc.inputFiles && Object.keys(doc.inputFiles).length > 0) out.inputFiles = doc.inputFiles;
  if (doc.outputs && Object.keys(doc.outputs).length > 0) out.outputs = doc.outputs;
  return `${JSON.stringify(out, null, 2)}\n`;
};

/** Output of the cells that have any: log lines, an error or thermo tables. Cells with nothing are left out. */
export const outputsOf = (runs: Record<string, { logs: string[]; error: string | null; tables: OutputTable[] }>): Record<string, CellOutput> => {
  const out: Record<string, CellOutput> = {};
  for (const [id, r] of Object.entries(runs)) {
    if (r.logs.length || r.error !== null || r.tables.length) {
      out[id] = { logs: r.logs.slice(), error: r.error, tables: r.tables.map((t) => ({ ...t, rows: t.rows.slice() })) };
    }
  }
  return out;
};

/** A safe file name from a notebook name: no path separators or control characters. */
export const safeFileBase = (name: string): string => {
  const base = name.replace(/[\\/:*?"<>|\x00-\x1f]+/g, '-').replace(/\s+/g, ' ').trim().replace(/^\.+/, '');
  return base === '' ? 'notebook' : base.slice(0, 120);
};

/** A file name for an export: the notebook name with the format's extension. */
export const exportFileName = (name: string, format: 'lmpnb' | 'lammps' | 'markdown'): string =>
  `${safeFileBase(name)}${format === 'lmpnb' ? NOTEBOOK_EXTENSION : format === 'lammps' ? '.in' : '.md'}`;

/* ------------------------------------------------------------------ importers */

export type ImportKind = 'lmpnb' | 'ipynb' | 'script';

export interface ImportResult {
  kind: ImportKind;
  name: string;
  cells: Cell[];
  inputFiles?: Record<string, string>;
  outputs?: Record<string, CellOutput>;
  /** What the import skipped or changed, for the person to read (empty when nothing). */
  notes: string[];
}

/** Name of a file without its extension (in.melt keeps its name: it has no extension to drop). */
const baseName = (file: string): string => {
  const last = file.split(/[\\/]/).pop() ?? file;
  return last.replace(/\.(lmpnb|json|ipynb|in|lmp|lammps)$/i, '') || last;
};

/**
 * Plain LAMMPS input -> cells. The rule: a line starting with `# ---` ends a cell and starts the next one
 * (the marker line itself is dropped). With no marker the whole file is one code cell. Blank cells are dropped.
 * `newId(k)` gives the id of the k-th cell (from 1).
 */
export const splitScript = (text: string, newId: (k: number) => string): Cell[] => {
  const segments: string[][] = [[]];
  for (const line of text.replace(/\r\n?/g, '\n').split('\n')) {
    if (/^# ---/.test(line)) segments.push([]);
    else segments[segments.length - 1].push(line);
  }
  return segments
    .map((lines) => lines.join('\n').replace(/\s+$/, ''))
    .filter((t) => t.trim() !== '')
    .map((text, i) => ({ id: newId(i + 1), kind: 'code' as const, text }));
};

/** Jupyter nbformat 4 -> cells: markdown cells stay markdown, code cells stay code; outputs are ignored. Raw cells become markdown. */
export const importIpynb = (text: string, newId: (k: number) => string): { cells: Cell[]; notes: string[] } => {
  let nb: unknown;
  try {
    nb = JSON.parse(text);
  } catch {
    fail('This .ipynb file is not JSON.');
  }
  if (!isObject(nb) || !Array.isArray(nb.cells)) {
    if (isObject(nb) && Array.isArray(nb.worksheets)) fail('This is an nbformat 3 notebook. Open and save it in Jupyter as nbformat 4 first.');
    fail('This .ipynb file has no "cells" list.');
  }
  const notes: string[] = [];
  let skipped = 0;
  const cells: Cell[] = [];
  nb.cells.forEach((c: unknown) => {
    if (!isObject(c)) { skipped++; return; }
    const src = Array.isArray(c.source) ? c.source.map(String).join('') : typeof c.source === 'string' ? c.source : null;
    if (src === null) { skipped++; return; }
    const k = cells.length + 1;
    if (c.cell_type === 'code') cells.push({ id: newId(k), kind: 'code', text: src });
    else if (c.cell_type === 'markdown' || c.cell_type === 'raw') cells.push({ id: newId(k), kind: 'markdown', text: src });
    else skipped++;
  });
  if (skipped > 0) notes.push(`${skipped} cell${skipped === 1 ? '' : 's'} of another kind were skipped.`);
  if (cells.length === 0) fail('The notebook has no cells to import.');
  return { cells, notes };
};

/**
 * Any supported file -> cells, by its name (and content when the name is not enough):
 * `.lmpnb` (or JSON with the format key), `.ipynb`, and plain LAMMPS input (`.in`, `.lmp`, `.lammps`,
 * or a file named `in.*`). Anything else is refused with a message.
 */
export const importNotebookFile = (name: string, text: string, newId: (k: number) => string = (k) => `imp${k}`): ImportResult => {
  const lower = name.toLowerCase();
  const base = baseName(name);
  if (lower.endsWith('.lmpnb') || (lower.endsWith('.json') && text.trimStart().startsWith('{') && text.includes(`"${NOTEBOOK_FORMAT}"`))) {
    const doc = parseNotebookFile(text);
    return { kind: 'lmpnb', name: doc.name, cells: doc.cells, inputFiles: doc.inputFiles, outputs: doc.outputs, notes: [] };
  }
  if (lower.endsWith('.ipynb')) {
    const { cells, notes } = importIpynb(text, newId);
    return { kind: 'ipynb', name: base, cells, notes };
  }
  if (/\.(in|lmp|lammps)$/.test(lower) || /^in\./i.test(name.split(/[\\/]/).pop() ?? name)) {
    const cells = splitScript(text, newId);
    if (cells.length === 0) fail(`${name} has no LAMMPS input in it.`);
    return { kind: 'script', name: base, cells, notes: [] };
  }
  fail(`${name}: the notebook cannot open this type of file. Use a ${NOTEBOOK_EXTENSION} notebook, a Jupyter .ipynb, or a LAMMPS input script (in.* or .in / .lmp).`);
};

/**
 * True for a name importNotebookFile opens as a notebook or script by its name alone: `.lmpnb`, `.ipynb`,
 * and LAMMPS input (`.in`, `.lmp`, `.lammps`, or a file named `in.*`). A `.json` notebook is recognised by its
 * content only, so it is not decided here.
 */
export const isNotebookName = (name: string): boolean => {
  const lower = name.toLowerCase();
  return lower.endsWith('.lmpnb') || lower.endsWith('.ipynb') || /\.(in|lmp|lammps)$/.test(lower)
    || /^in\./i.test(name.split(/[\\/]/).pop() ?? name);
};

/**
 * Classifies dropped files by name, in the order given: 'notebook' for a notebook or LAMMPS script
 * (see isNotebookName), 'input' for everything else (data, potentials, structures, gzip files, ...).
 */
export const classifyDropped = (names: readonly string[]): ('notebook' | 'input')[] =>
  names.map((n) => (isNotebookName(n) ? 'notebook' : 'input'));

/* ------------------------------------------------------------------ exporters */

/** The notebook as one LAMMPS input script: code cells in order, each followed by `# ---`; markdown as `#` comment lines. */
export const exportLammps = (cells: readonly Cell[]): string => {
  const blocks = cells.map((c) => {
    if (c.kind === 'markdown') {
      return c.text.replace(/\s+$/, '').split('\n').map((l) => (l.trim() === '' ? '#' : `# ${l}`)).join('\n');
    }
    return c.text.replace(/\s+$/, '');
  }).filter((b) => b.trim() !== '');
  return blocks.length ? `${blocks.join('\n# ---\n')}\n` : '';
};

/** The notebook as Markdown: markdown cells as they are, code cells in fenced `lammps` blocks (a longer fence if the code has backticks). */
export const exportMarkdown = (name: string, cells: readonly Cell[]): string => {
  const parts: string[] = [`# ${name.replace(/\s+/g, ' ').trim()}`];
  for (const c of cells) {
    if (c.kind === 'markdown') {
      if (c.text.trim() !== '') parts.push(c.text.replace(/\s+$/, ''));
    } else if (c.text.trim() !== '') {
      const body = c.text.replace(/\s+$/, '');
      let fence = '```';
      while (body.includes(fence)) fence += '`';
      parts.push(`${fence}lammps\n${body}\n${fence}`);
    }
  }
  return `${parts.join('\n\n')}\n`;
};
