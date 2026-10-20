import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { filesReadBy } from '../src/engine/script';
import { parseNotebookFile } from '../src/components/workbench/notebook/notebookFile';
import { parseFile } from '../src/services/fileParser';
import { convertStructure } from '../src/lammps/structureToLammps';

/*
 * The example notebooks of the notebook menu (public/notebooks, "Open an example…"): the index lists files that exist and
 * parse, every file a notebook reads is among its input files, the data files converted from CIFs are what the import
 * dialog writes today, and every notebook runs in the engine (its runs shortened to keep the suite fast; the full runs
 * were checked against native LAMMPS b332f28 when the notebooks were written).
 */

const DIR = join(__dirname, '..', 'public', 'notebooks');
const index = JSON.parse(readFileSync(join(DIR, 'index.json'), 'utf8')) as { notebooks: { file: string; name: string; summary: string }[] };

describe('example notebooks', () => {
  it('the index lists notebooks that exist, with a name and a summary', () => {
    expect(index.notebooks.length).toBeGreaterThanOrEqual(6);
    for (const e of index.notebooks) {
      expect(existsSync(join(DIR, e.file)), e.file).toBe(true);
      expect(e.name.length > 0 && e.summary.length > 0).toBe(true);
    }
  });

  for (const e of index.notebooks) {
    const doc = parseNotebookFile(readFileSync(join(DIR, e.file), 'utf8'));
    const code = doc.cells.filter((c) => c.kind !== 'markdown').map((c) => c.text).join('\n');

    it(`${e.file}: names the notebook and carries every file its cells read`, () => {
      expect(doc.name).toBe(e.name);
      for (const f of filesReadBy(code)) expect(Object.keys(doc.inputFiles ?? {}), `${e.file} reads ${f}`).toContain(f);
    });

    it(`${e.file}: runs in the engine without an error`, async () => {
      const events: EngineEvent[] = [];
      const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => undefined });
      for (const [name, text] of Object.entries(doc.inputFiles ?? {})) session.addFile(name, text);
      const short = code.replace(/^(run\s+)\d+/gm, '$120').replace(/^(minimize\s+\S+\s+\S+\s+)\d+\s+\d+/gm, '$120 200');
      await session.execute(short);
      expect(events.some((ev) => ev.kind === 'error')).toBe(false);
      expect(events.some((ev) => ev.kind === 'thermo')).toBe(true);
    }, 120_000);
  }

  it('the data files converted from CIFs are what the import dialog writes from them', () => {
    const cases = [['quartz-cif.lmpnb', 'quartz.cif', 'quartz.data', 'charge', false], ['ice-cif.lmpnb', 'ice_ih.cif', 'ice_ih.data', 'full', true],
      ['urea-cif.lmpnb', 'urea.cif', 'urea.data', 'full', true]] as const;
    for (const [file, cif, data, style, bonds] of cases) {
      const files = parseNotebookFile(readFileSync(join(DIR, file), 'utf8')).inputFiles ?? {};
      const mol = parseFile(files[cif], 'cif');
      const res = convertStructure(mol, { target: 'data', source: 'elements', atomStyle: style, typeBy: 'element', bonds, padding: 10,
        fileName: data, title: `Converted from ${cif} by the LAMMPS workbench`, box: mol.box });
      expect(files[data], `${file}: ${data}`).toBe(res.text);
      // the CIF is the same file the Structure Viewer lists
      expect(files[cif]).toBe(readFileSync(join(__dirname, '..', 'public', 'examples', cif), 'utf8'));
    }
  });
});
