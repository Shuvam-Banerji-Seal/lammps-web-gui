#!/usr/bin/env node
/**
 * Generates src/engine/accelAliases.ts: the accelerator-suffix aliases (lj/cut/omp -> lj/cut, nve/kk -> nve,
 * harmonic/omp for bonds, ...) that the notebook's interpreter maps to base style names.
 *
 * usage (from the repo root):  node scripts/gen-accel-aliases.mjs <LAMMPS doc/src directory> [output file]
 *
 * The aliases are read from the entries of the five style indexes, for example
 *   * :doc:`table (gko) <pair_table>`
 * where each letter names an accelerator package: g = gpu, i = intel, k = kk, o = omp, t = opt. The
 * script fails when a letter is unknown, when one alias name would map to two different base styles,
 * and when a listed style already ends in an accelerator suffix. Rerun it whenever the docs change.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_OUT = join(ROOT, 'src/engine/accelAliases.ts');

/** The index pages that list accelerated styles, with the kind each file lists by default. */
const INDEX_FILES = {
  'Commands_pair.rst': 'pair',
  'Commands_fix.rst': 'fix',
  'Commands_compute.rst': 'compute',
  'Commands_bond.rst': 'bond',
  'Commands_kspace.rst': 'kspace',
};

/** Accelerator letters of the index pages (Commands_pair.rst: "g = GPU, i = INTEL, k = KOKKOS, o = OPENMP, t = OPT."). */
const LETTERS = { g: 'gpu', i: 'intel', k: 'kk', o: 'omp', t: 'opt' };
const SUFFIXES = new Set(Object.values(LETTERS));

/** The sentence every index page states before its list (compared after whitespace is collapsed). */
const INDEX_SENTENCE = 'Some styles have accelerated versions. This is indicated by additional letters in parenthesis: g = GPU, i = INTEL, k = KOKKOS, o = OPENMP, t = OPT.';

/** "All LAMMPS :doc:`bond_style <bond_style>` commands." sets the kind of the entries below it (Commands_bond.rst). */
const SECTION = /All LAMMPS :doc:`(pair|bond|angle|dihedral|improper|kspace)_style </;
/** "   * :doc:`table (gko) <pair_table>`" */
const ENTRY = /^\s*\*\s+:doc:`([^`(<]+?) \(([a-z]+)\) <[^`>]+>`\s*$/;

const usage = 'usage: node scripts/gen-accel-aliases.mjs <LAMMPS doc/src directory> [output file]';

/** @returns {{ alias: string, base: string, kind: string, letter: string, index: string }[]} */
export const readAliases = (docSrc) => {
  const rows = [];
  for (const [file, defaultKind] of Object.entries(INDEX_FILES)) {
    const text = readFileSync(join(docSrc, file), 'utf8');
    if (!text.replace(/\s+/g, ' ').includes(INDEX_SENTENCE)) throw new Error(`${file}: the accelerator sentence changed; read it again before generating`);
    let kind = defaultKind;
    let found = 0;
    for (const line of text.split('\n')) {
      const sec = SECTION.exec(line);
      if (sec) kind = sec[1];
      const m = ENTRY.exec(line);
      if (!m) continue;
      const [, base, letters] = m;
      if (SUFFIXES.has(base.split('/').pop())) throw new Error(`${file}: style ${base} already ends in an accelerator suffix`);
      const seen = new Set();
      for (const letter of letters) {
        if (!(letter in LETTERS)) throw new Error(`${file}: ${base} has the unknown accelerator letter '${letter}'`);
        if (seen.has(letter)) throw new Error(`${file}: ${base} lists the letter '${letter}' twice`);
        seen.add(letter);
        rows.push({ alias: `${base}/${LETTERS[letter]}`, base, kind, letter, index: file });
        found++;
      }
    }
    if (found === 0) throw new Error(`${file}: no accelerated entries found`);
  }
  return rows;
};

/** The map from alias name to base name; fails when one alias names two different bases. */
export const buildMap = (rows) => {
  const map = new Map();
  for (const r of rows) {
    const prev = map.get(r.alias);
    if (prev !== undefined && prev !== r.base) {
      throw new Error(`alias ${r.alias} maps to two base styles: ${prev} and ${r.base} (${r.index})`);
    }
    map.set(r.alias, r.base);
  }
  return new Map([...map.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
};

const HEADER = `/*
 * GENERATED FILE — do not edit. Regenerate with:
 *   node scripts/gen-accel-aliases.mjs <LAMMPS doc/src directory>
 * (scripts/gen-accel-aliases.mjs reads the five style indexes below and writes this file.)
 *
 * The accelerated style names of LAMMPS, as aliases of their base styles. Index pages:
 * docs.lammps.org/Commands_pair.html, docs.lammps.org/Commands_bond.html (bond, angle, dihedral and
 * improper sections), docs.lammps.org/Commands_fix.html, docs.lammps.org/Commands_compute.html and
 * docs.lammps.org/Commands_kspace.html. Each says: "Some styles have accelerated versions.  This is
 * indicated by additional letters in parenthesis: g = GPU, i = INTEL, k = KOKKOS, o = OPENMP, t = OPT."
 *
 * docs.lammps.org/accel_styles.html says: "Styles with a *gpu*, *intel*, *kk*, *omp*, or *opt* suffix
 * are functionally the same as the corresponding style without the suffix." and "The accelerated styles
 * take the same arguments and should produce the same results, except for round-off and precision
 * issues."
 *
 * The aliases are names only. The engine never picks an accelerator from them: the session maps each
 * name to its base style before anything is created (session.ts), and the browser engine chooses CPU
 * threads or the GPU itself.
 */

/** Accelerated style name -> base style name. */
export const ACCEL_ALIASES: ReadonlyMap<string, string> = new Map<string, string>([
`;

const render = (map) => `${[...map.entries()].map(([a, b]) => `  [${JSON.stringify(a)}, ${JSON.stringify(b)}],`).join('\n')}\n]);\n`;

const main = () => {
  const [docSrc, outArg] = process.argv.slice(2);
  if (!docSrc) throw new Error(usage);
  const rows = readAliases(resolve(docSrc));
  const map = buildMap(rows);
  const perKind = {};
  for (const r of rows) perKind[r.kind] = (perKind[r.kind] ?? 0) + 1;
  writeFileSync(outArg ? resolve(outArg) : DEFAULT_OUT, HEADER + render(map));
  console.log(`accelerator aliases: ${map.size} names (${rows.length} kind/style/letter entries)`);
  for (const k of Object.keys(perKind).sort()) console.log(`  ${k}: ${perKind[k]}`);
};

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (e) {
    console.error(e instanceof Error ? e.message : String(e));
    process.exit(1);
  }
}
