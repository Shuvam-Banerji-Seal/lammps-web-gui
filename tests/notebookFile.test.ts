import { describe, expect, it } from 'vitest';
import {
  MIGRATIONS, NOTEBOOK_FORMAT, NotebookFileError, exportFileName, exportLammps, exportMarkdown, importIpynb,
  importNotebookFile, outputsOf, parseNotebookFile, safeFileBase, serializeNotebookFile, splitScript, type NotebookDoc,
} from '../src/components/workbench/notebook/notebookFile';
import type { Cell } from '../src/components/workbench/notebook/cells';

const cells: Cell[] = [
  { id: 'c1', kind: 'code', text: 'units lj\natom_style atomic' },
  { id: 'c2', kind: 'markdown', text: '## Notes\n- a' },
  { id: 'c3', kind: 'code', text: 'run 10' },
];
const doc: NotebookDoc = { name: 'Melt', created: '2026-10-10T10:00:00.000Z', modified: '2026-10-11T08:00:00.000Z', cells };
const idsOf = (k: number) => `x${k}`;

describe('.lmpnb format', () => {
  it('writes version 1 with the format key, and parses back to the same notebook', () => {
    const text = serializeNotebookFile(doc);
    const obj = JSON.parse(text);
    expect(obj.format).toBe(NOTEBOOK_FORMAT);
    expect(obj.version).toBe(1);
    expect(obj.cells[1]).toEqual({ id: 'c2', kind: 'markdown', text: '## Notes\n- a' });
    expect(obj).not.toHaveProperty('inputFiles');
    expect(obj).not.toHaveProperty('outputs');
    expect(text.endsWith('\n')).toBe(true);
    expect(parseNotebookFile(text)).toEqual(doc);
  });

  it('writes the kind of a code cell too, and includes input files and outputs only when given', () => {
    const withExtras: NotebookDoc = {
      ...doc,
      inputFiles: { 'data.txt': 'hello' },
      outputs: { c3: { logs: ['Loop time of 0.1'], error: null, tables: [{ keywords: ['step', 'temp'], rows: [{ step: 0, temp: 3 }] }] } },
    };
    const back = parseNotebookFile(serializeNotebookFile(withExtras));
    expect(back.inputFiles).toEqual({ 'data.txt': 'hello' });
    expect(back.outputs).toEqual(withExtras.outputs);
    expect(back.cells[0].kind).toBe('code');
  });

  it('refuses a file of another format, with the format it found', () => {
    expect(() => parseNotebookFile(JSON.stringify({ format: 'jupyter', version: 1 }))).toThrow(/format is "jupyter"/);
    expect(() => parseNotebookFile('units lj\nrun 10')).toThrow(NotebookFileError);
    expect(() => parseNotebookFile('[1,2]')).toThrow(/top level/);
  });

  it('refuses a newer version and says what to do', () => {
    const text = JSON.stringify({ ...JSON.parse(serializeNotebookFile(doc)), version: 2 });
    expect(() => parseNotebookFile(text)).toThrow(/newer version.*version 2.*reads version 1/);
  });

  it('refuses a version that is not a whole number from 1', () => {
    const base = JSON.parse(serializeNotebookFile(doc));
    expect(() => parseNotebookFile(JSON.stringify({ ...base, version: 0 }))).toThrow(/whole number/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, version: '1' }))).toThrow(/whole number/);
    const { version: _v, ...noVersion } = base;
    expect(() => parseNotebookFile(JSON.stringify(noVersion))).toThrow(/"version"/);
  });

  it('names the malformed cell: bad kind, missing id, duplicate id, bad text', () => {
    const base = JSON.parse(serializeNotebookFile(doc));
    const with1 = (cell: unknown) => JSON.stringify({ ...base, cells: [cell] });
    expect(() => parseNotebookFile(with1({ id: 'a', kind: 'html', text: '' }))).toThrow(/Cell 1 \("a"\).*"kind"/);
    expect(() => parseNotebookFile(with1({ kind: 'code', text: '' }))).toThrow(/Cell 1 needs a non-empty string "id"/);
    expect(() => parseNotebookFile(with1({ id: 'a', kind: 'code', text: 3 }))).toThrow(/Cell 1 \("a"\): "text"/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, cells: [{ id: 'a', kind: 'code', text: '' }, { id: 'a', kind: 'code', text: '' }] })))
      .toThrow(/Cell 2: the id "a" is used/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, cells: [] }))).toThrow(/non-empty list/);
  });

  it('checks the dates and the name', () => {
    const base = JSON.parse(serializeNotebookFile(doc));
    expect(() => parseNotebookFile(JSON.stringify({ ...base, modified: 'yesterday' }))).toThrow(/"modified"/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, name: '  ' }))).toThrow(/"name"/);
  });

  it('rejects malformed outputs and input files', () => {
    const base = JSON.parse(serializeNotebookFile(doc));
    expect(() => parseNotebookFile(JSON.stringify({ ...base, inputFiles: { a: 1 } }))).toThrow(/inputFiles/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, outputs: { c1: { logs: 'x', error: null, tables: [] } } })))
      .toThrow(/outputs of cell "c1": "logs"/);
    expect(() => parseNotebookFile(JSON.stringify({ ...base, outputs: { c1: { logs: [], error: null, tables: [{ keywords: ['step'], rows: [{ step: 'x' }] }] } } })))
      .toThrow(/thermo rows must be objects of numbers/);
  });

  it('migrates older versions step by step through the migration hooks', () => {
    // A pretend version 1 file, read by a pretend version 3 app: step 1 renames a field, step 2 adds one.
    const old = JSON.stringify({ format: NOTEBOOK_FORMAT, version: 1, title: 'Old', created: doc.created, modified: doc.modified, cells: doc.cells });
    const migrations = {
      1: (raw: Record<string, unknown>) => ({ ...raw, version: 2, name: raw.title }),
      2: (raw: Record<string, unknown>) => ({ ...raw, version: 3, inputFiles: { 'added.txt': 'x' } }),
    };
    const got = parseNotebookFile(old, migrations, 3);
    expect(got.name).toBe('Old');
    expect(got.inputFiles).toEqual({ 'added.txt': 'x' });
    expect(() => parseNotebookFile(old, { 2: migrations[2] }, 3)).toThrow(/No migration from notebook format version 1 to 2/);
    expect(() => parseNotebookFile(old, migrations, 0)).toThrow(/reads version 0/);
    expect(MIGRATIONS).toEqual({});
  });
});

describe('importers', () => {
  it('imports a .lmpnb file by name, and a JSON file that carries the format key', () => {
    const imp = importNotebookFile('melt.lmpnb', serializeNotebookFile(doc));
    expect(imp).toMatchObject({ kind: 'lmpnb', name: 'Melt', notes: [] });
    expect(imp.cells).toEqual(cells);
    const json = importNotebookFile('whatever.json', serializeNotebookFile(doc));
    expect(json.kind).toBe('lmpnb');
  });

  it('splits a LAMMPS input at lines starting with "# ---" and keeps everything else in the cell', () => {
    const script = 'units lj\n# a comment\nrun 10\n# ---\n\n# ---\nthermo 5\nrun 20\n';
    const cs = splitScript(script, idsOf);
    expect(cs.map((c) => c.text)).toEqual(['units lj\n# a comment\nrun 10', 'thermo 5\nrun 20']);
    expect(cs.map((c) => c.id)).toEqual(['x1', 'x2']);
    expect(cs.every((c) => c.kind === 'code')).toBe(true);
  });

  it('makes one code cell from a script without markers', () => {
    const imp = importNotebookFile('in.melt', 'units lj\nrun 5\n', idsOf);
    expect(imp.kind).toBe('script');
    expect(imp.name).toBe('in.melt');
    expect(imp.cells).toEqual([{ id: 'x1', kind: 'code', text: 'units lj\nrun 5' }]);
    expect(importNotebookFile('melt.lmp', 'run 1\n', idsOf).name).toBe('melt');
    expect(importNotebookFile('run.in', 'run 1\n', idsOf).kind).toBe('script');
  });

  it('refuses a script with no input in it, and an unknown file type with the list of what it reads', () => {
    expect(() => importNotebookFile('in.empty', '\n\n# ---\n', idsOf)).toThrow(/no LAMMPS input/);
    expect(() => importNotebookFile('notes.txt', 'hello')).toThrow(/Use a \.lmpnb notebook, a Jupyter \.ipynb, or a LAMMPS input script/);
  });

  it('imports a Jupyter notebook: markdown stays markdown, code stays code, outputs are ignored', () => {
    const nb = {
      nbformat: 4, nbformat_minor: 5, metadata: {},
      cells: [
        { cell_type: 'markdown', metadata: {}, source: ['# Title\n', 'text'] },
        { cell_type: 'code', metadata: {}, source: 'units lj\nrun 1', execution_count: 1, outputs: [{ output_type: 'stream', text: 'x' }] },
        { cell_type: 'raw', metadata: {}, source: 'raw text' },
      ],
    };
    const imp = importNotebookFile('demo.ipynb', JSON.stringify(nb), idsOf);
    expect(imp.kind).toBe('ipynb');
    expect(imp.name).toBe('demo');
    expect(imp.cells).toEqual([
      { id: 'x1', kind: 'markdown', text: '# Title\ntext' },
      { id: 'x2', kind: 'code', text: 'units lj\nrun 1' },
      { id: 'x3', kind: 'markdown', text: 'raw text' },
    ]);
    expect(imp.notes).toEqual([]);
  });

  it('reports skipped cells and refuses nbformat 3 and non-notebooks', () => {
    const r = importIpynb(JSON.stringify({ cells: [{ cell_type: 'heading', source: 'x' }, { cell_type: 'code', source: 'run 1' }] }), idsOf);
    expect(r.cells).toHaveLength(1);
    expect(r.notes[0]).toMatch(/1 cell of another kind/);
    expect(() => importIpynb(JSON.stringify({ worksheets: [] }), idsOf)).toThrow(/nbformat 3/);
    expect(() => importIpynb('{}', idsOf)).toThrow(/no "cells" list/);
    expect(() => importIpynb(JSON.stringify({ cells: [] }), idsOf)).toThrow(/no cells to import/);
  });
});

describe('exporters', () => {
  it('exports LAMMPS input: code in order, separated by # ---, markdown as # comments', () => {
    const text = exportLammps(cells);
    expect(text).toBe('units lj\natom_style atomic\n# ---\n# ## Notes\n# - a\n# ---\nrun 10\n');
    // and it imports back as the same code cells (markdown comes back as a code cell of comments)
    const back = splitScript(text, idsOf);
    expect(back.map((c) => c.text)).toEqual(['units lj\natom_style atomic', '# ## Notes\n# - a', 'run 10']);
    expect(exportLammps([])).toBe('');
  });

  it('exports Markdown with code in lammps fences, longer when the code has backticks', () => {
    const md = exportMarkdown('Melt', [...cells, { id: 'c4', kind: 'code', text: 'print "```x```"' }]);
    expect(md).toBe('# Melt\n\n```lammps\nunits lj\natom_style atomic\n```\n\n## Notes\n- a\n\n```lammps\nrun 10\n```\n\n````lammps\nprint "```x```"\n````\n');
  });

  it('names files safely and with their extension', () => {
    expect(exportFileName('Melt: run/1', 'lmpnb')).toBe('Melt- run-1.lmpnb');
    expect(exportFileName('Melt', 'lammps')).toBe('Melt.in');
    expect(exportFileName('Melt', 'markdown')).toBe('Melt.md');
    expect(safeFileBase('  ..  ')).toBe('notebook');
  });

  it('keeps only the outputs of cells that have something to show', () => {
    const out = outputsOf({
      c1: { logs: [], error: null, tables: [] },
      c2: { logs: ['Loop time'], error: null, tables: [] },
      c3: { logs: [], error: 'unknown command', tables: [] },
    });
    expect(Object.keys(out)).toEqual(['c2', 'c3']);
  });
});
