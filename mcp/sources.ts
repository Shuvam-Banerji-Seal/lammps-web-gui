/**
 * Reads the files mcp/knowledge.ts builds from (node only): the doc index, the example notebooks and the guides.
 * Used by mcp/build.mjs and the tests.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { DocIndex, Guide, KnowledgeSources } from './knowledge';

/** The guides: id, title, one-line description, and the file (relative to the repository root). */
export const GUIDES: { id: string; title: string; description: string; file: string }[] = [
  { id: 'writing-inputs', title: 'Writing a LAMMPS input script', description: 'Structure of an input script, the parsing rules, a complete example and a checklist, citing docs.lammps.org.', file: 'mcp/guides/writing-inputs.md' },
  { id: 'browser-engine', title: 'The in-browser LAMMPS engine', description: 'What the LAMMPS Web GUI notebook runs, how it is checked against native LAMMPS, threads and the GPU.', file: 'docs/design/notebook.md' },
  { id: 'script-check', title: 'Script check', description: 'The rules of the script checker (validate_lammps_script) and what they catch.', file: 'wiki/Script-Check.md' },
  { id: 'file-formats', title: 'File formats', description: 'Structure and trajectory formats the site reads and writes (LAMMPS data, dump, XYZ, PDB, CIF, ...).', file: 'wiki/File-Formats.md' },
  { id: 'performance', title: 'Performance', description: 'How fast the browser engine runs and what limits it.', file: 'wiki/Performance.md' },
];

export const SITE = 'https://shuvam-banerji-seal.github.io/lammps-web-gui/';

export const readSources = (root: string): KnowledgeSources => {
  const read = (rel: string) => readFileSync(join(root, rel), 'utf8');
  const notebookIndex = JSON.parse(read('public/notebooks/index.json')) as KnowledgeSources['notebookIndex'];
  const notebooks: Record<string, string> = {};
  for (const e of notebookIndex.notebooks) notebooks[e.file] = read(`public/notebooks/${e.file}`);
  const guides: Guide[] = GUIDES.map(({ file, ...g }) => ({ ...g, text: read(file) }));
  return {
    version: (JSON.parse(read('package.json')) as { version: string }).version,
    site: SITE,
    docIndex: JSON.parse(read('mcp/data/lammps-doc-index.json')) as DocIndex,
    notebookIndex,
    notebooks,
    guides,
  };
};
