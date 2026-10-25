import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

/*
 * Parity matrices: oracle cases (tests/oracle/*.in) with one word of one line changed, then run 0. Native LAMMPS runs
 * every row (scripts/parity/run-parity.mjs writes tests/fixtures/parity/<matrix>.json) and the engine must give the
 * same first ERROR line, or run without one (tests/engineParity.<matrix>.test.ts). A matrix takes one line per style:
 * the line of the smallest case that has it, so each row checks how one style reads one changed word.
 *
 * This module is shared by the test harness (tests/parityHarness.ts) and the native runner, which imports it under
 * Node's type stripping: erasable TypeScript only.
 */

/** One row: the case, the line replaced (0-based; the lines before it are kept) and the line that replaces it. */
export interface ParityRow {
  case: string;
  line: number;
  text: string;
  /** what the row exercises, e.g. "fix langevin" */
  style: string;
}

export interface ParityFixtureRow extends ParityRow {
  /** native result: the first ERROR line ("ERROR: ..." without its source location), OK, or CRASH rc=N */
  native: string;
  /** the native build that ran the row when the default binary lacks the package (e.g. ML-PACE) */
  build?: string;
  /** why the row is not compared (a native crash, or a package no native build here has) */
  skip?: string;
}

export interface ParityFixture {
  matrix: string;
  about: string;
  lammps: string;
  rows: ParityFixtureRow[];
}

interface Case { name: string; text: string }

export interface MatrixDef {
  about: string;
  rows: (cases: Case[]) => ParityRow[];
  /**
   * Rows (by style) whose native result comes from undefined behaviour, with the reason: native reads or writes outside
   * its arrays, so its result is not a rule the engine can follow. The runner marks them skip.
   */
  undefined?: Record<string, string>;
}

export const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

/** Commands after which the case's lines are no longer read in order, so a row is not taken from such a case. */
const FLOW = /^\s*(include|jump|label|next|read_restart)\b/;

/** The oracle cases, smallest file first (a style's row comes from the smallest case that has it). */
export const oracleCases = (dir: string): Case[] =>
  readdirSync(dir)
    .filter((f) => f.endsWith('.in'))
    .map((f) => ({ f, size: statSync(join(dir, f)).size }))
    .sort((a, b) => a.size - b.size || (a.f < b.f ? -1 : a.f > b.f ? 1 : 0))
    .map(({ f }) => ({ name: f.slice(0, -3), text: readFileSync(join(dir, f), 'utf8') }));

/**
 * The first fix or compute line of each style whose words from word 4 on hold a number, with that number replaced
 * by `value`.
 */
const fixComputeFirstNumber = (value: string) => (cases: Case[]): ParityRow[] => {
  const rows = new Map<string, ParityRow>();
  for (const c of cases) {
    const lines = c.text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const code = lines[i].split('#')[0];
      const m = /^\s*(fix|compute)\s+\S+\s+\S+\s+(\S+)/.exec(code);
      if (!m) continue;
      const style = `${m[1]} ${m[2]}`;
      if (rows.has(style) || code.includes('&') || code.includes('$')) continue;
      if (lines.slice(0, i + 1).some((l) => FLOW.test(l))) continue;
      const w = code.trim().split(/\s+/);
      const k = w.findIndex((x, j) => j >= 4 && NUMBER.test(x));
      if (k < 0) continue;
      w[k] = value;
      rows.set(style, { case: c.name, line: i, text: w.join(' '), style });
    }
  }
  return [...rows.values()].sort((a, b) => (a.style < b.style ? -1 : a.style > b.style ? 1 : 0));
};

export const MATRICES: Record<string, MatrixDef> = {
  'fix-compute-word': {
    about: 'the first number of each fix and compute style (from word 4) replaced by the word bogus',
    rows: fixComputeFirstNumber('bogus'),
  },
  'fix-compute-zero': {
    about: 'the first number of each fix and compute style (from word 4) replaced by 0',
    rows: fixComputeFirstNumber('0'),
    undefined: {
      'fix smd/wall_surface': 'native creates the wall atoms with type 0, outside the atom types (the engine refuses the type)',
    },
  },
  'fix-compute-negative': {
    about: 'the first number of each fix and compute style (from word 4) replaced by -1',
    rows: fixComputeFirstNumber('-1'),
    undefined: {
      'fix restrain': 'native looks atom ID -1 up as map_array[-1], outside its atom map (the engine refuses the ID)',
      'fix smd/wall_surface': 'native creates the wall atoms with type -1, outside the atom types (the engine refuses the type)',
    },
  },
};

/** The row's input: the case up to its line (without the # oracle header lines), the replacing line, then run 0. */
export const parityInput = (caseText: string, row: ParityRow): string => {
  const kept = caseText.split('\n').slice(0, row.line).filter((l) => !l.startsWith('# oracle'));
  return `${kept.join('\n')}\n${row.text}\nrun 0\n`;
};

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter((w) => w !== '') : [];
};

/**
 * The files the row's input reads, by the name it uses: the case's oracle-inputs (next to the case), oracle-potentials
 * (third_party/lammps/potentials) and oracle-examples (third_party/lammps/examples, by base name), and any file of the
 * oracle directory that the input names without a directive. `latin1` marks gzipped inputs, read as bytes.
 */
export const parityFiles = (root: string, caseText: string, input: string): Record<string, { path: string; latin1: boolean }> => {
  const oracle = join(root, 'tests', 'oracle');
  const files: Record<string, { path: string; latin1: boolean }> = {};
  for (const f of directive(caseText, 'oracle-inputs')) files[f] = { path: join(oracle, f), latin1: f.endsWith('.gz') };
  for (const f of directive(caseText, 'oracle-potentials')) files[f] = { path: join(root, 'third_party', 'lammps', 'potentials', f), latin1: false };
  for (const f of directive(caseText, 'oracle-examples')) files[f.split('/').pop()!] = { path: join(root, 'third_party', 'lammps', 'examples', f), latin1: f.endsWith('.gz') };
  const own = new Set(readdirSync(oracle).filter((f) => !f.endsWith('.in')));
  for (const w of input.split(/\s+/)) if (own.has(w) && !(w in files)) files[w] = { path: join(oracle, w), latin1: w.endsWith('.gz') };
  return files;
};
