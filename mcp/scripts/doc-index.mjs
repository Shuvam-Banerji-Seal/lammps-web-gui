#!/usr/bin/env node
/**
 * Builds mcp/data/lammps-doc-index.json from the LAMMPS documentation sources (doc/src/*.rst of a LAMMPS checkout):
 * every command and style name the docs index (".. index:: fix nvt"), the page that documents it, the page's
 * title and the first sentence of its Description section. The MCP server (mcp/server.ts) searches it and maps a
 * name to its page; the page text itself is fetched from GitHub at the same commit when an agent asks for it.
 *
 *   node mcp/scripts/doc-index.mjs /path/to/lammps [commit]
 *
 * The commit defaults to the checkout's HEAD; it must be the commit the engine is checked against (AGENTS.md, b332f28).
 */
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.argv[2];
if (!root) {
  console.error('usage: node mcp/scripts/doc-index.mjs /path/to/lammps [commit]');
  process.exit(2);
}
const commit = process.argv[3] ?? execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
const docDir = join(root, 'doc', 'src');

/** Accelerator variants share the page of the plain style (pair_style lj/cut/omp is lj/cut on OPENMP). */
const ACCEL = /\/(gpu|intel|kk|kk\/host|kk\/device|omp|opt)$/;

/** rst markup to plain text: roles, emphasis, literals, substitutions. */
const plain = (s) => s
  .replace(/:(doc|ref|math|file|cite):`([^`<]*?)\s*<[^>]*>`/g, '$2')
  .replace(/:(doc|ref|math|file|cite):`([^`]*)`/g, '$2')
  .replace(/``([^`]*)``/g, '$1')
  .replace(/`([^`<]*?)\s*<[^>]*>`_+/g, '$1')
  .replace(/`([^`]*)`_*/g, '$1')
  .replace(/\*\*([^*]+)\*\*/g, '$1')
  .replace(/\*([^*\s][^*]*)\*/g, '$1')
  .replace(/\|[^|\s]+\|/g, '')
  .replace(/\s+/g, ' ')
  .trim();

/** The first sentence of the Description section (at most 240 characters). */
const summaryOf = (lines) => {
  const k = lines.findIndex((l, i) => l.trim() === 'Description' && /^"{3,}|^-{3,}|^\^{3,}|^~{3,}/.test(lines[i + 1] ?? ''));
  if (k < 0) return '';
  const para = [];
  for (let i = k + 2; i < lines.length; i++) {
    const l = lines[i];
    if (!l.trim()) { if (para.length) break; continue; }
    if (/^\.\. /.test(l) || /^\s/.test(l) && !para.length) { if (para.length) break; continue; }
    para.push(l.trim());
  }
  const text = plain(para.join(' '));
  const m = /^(.+?[.!?])(\s|$)/.exec(text);
  const first = m ? m[1] : text;
  return first.length > 240 ? `${first.slice(0, 237).trimEnd()}...` : first;
};

/** The page's titles: the heading lines underlined with '=' before the first other section. */
const titlesOf = (lines) => {
  const out = [];
  for (let i = 0; i + 1 < lines.length; i++) {
    if (/^={3,}\s*$/.test(lines[i + 1]) && lines[i].trim() && !/^={3,}/.test(lines[i])) out.push(plain(lines[i]));
    if (/^"{3,}\s*$/.test(lines[i + 1]) && out.length) break;
  }
  return out;
};

const entries = [];
const pages = {};
for (const file of readdirSync(docDir).filter((f) => f.endsWith('.rst')).sort()) {
  const page = file.slice(0, -4);
  const lines = readFileSync(join(docDir, file), 'utf8').split('\n');
  const names = [];
  for (const l of lines) {
    const m = /^\.\. index:: (.+?)\s*$/.exec(l);
    if (m && !ACCEL.test(m[1])) names.push(m[1]);
  }
  if (!names.length) continue;
  pages[page] = { title: titlesOf(lines).join(' / '), summary: summaryOf(lines) };
  for (const name of names) entries.push({ name, page });
}
// the style tables of the Commands_*.rst pages name a few styles their page does not index (pair_style lj/cut/dipole/sf)
const TABLES = { Commands_fix: 'fix', Commands_compute: 'compute', Commands_pair: 'pair_style', Commands_kspace: 'kspace_style', Commands_dump: 'dump', Commands_bond: 'bond_style' };
const known = new Set(entries.map((e) => e.name));
for (const [file, first] of Object.entries(TABLES)) {
  let kind = first;
  for (const l of readFileSync(join(docDir, `${file}.rst`), 'utf8').split('\n')) {
    const anchor = /^\.\. _(bond|angle|dihedral|improper):/.exec(l);
    if (anchor) kind = `${anchor[1]}_style`;
    for (const m of l.matchAll(/:doc:`([^`<]+?)\s*(?:\([a-z, ]+\))?\s*<([^>]+)>`/g)) {
      const name = `${kind} ${m[1].trim()}`;
      if (m[1].trim() === kind || m[1].trim() === kind.replace('_style', '') || known.has(name) || !pages[m[2]]) continue;
      known.add(name);
      entries.push({ name, page: m[2] });
    }
  }
}
entries.sort((a, b) => a.name.localeCompare(b.name) || a.page.localeCompare(b.page));

const out = {
  source: 'LAMMPS documentation, doc/src/*.rst (https://github.com/lammps/lammps), GPL-2.0',
  commit,
  docs: 'https://docs.lammps.org/',
  raw: `https://raw.githubusercontent.com/lammps/lammps/${commit}/doc/src/`,
  entries,
  pages,
};
const dest = new URL('../data/lammps-doc-index.json', import.meta.url);
writeFileSync(dest, `${JSON.stringify(out, null, 1)}\n`);
console.log(`${entries.length} names on ${Object.keys(pages).length} pages (commit ${commit.slice(0, 7)}) -> ${dest.pathname}`);
