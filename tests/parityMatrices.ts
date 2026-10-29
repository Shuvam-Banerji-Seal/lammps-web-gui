import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { PAIR_EXTRACT } from '../src/engine/fix/adaptPairsTable.ts';
import { BONDED_EXTRACT } from '../src/engine/fix/adaptBondedTable.ts';

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
  /**
   * A row that compares values: the case's lines are kept up to this index (exclusive), the first run, minimize or
   * rerun after the changed line, so the lines that complete the setup stay.
   */
  until?: number;
  /** Lines run before "run 0" (a row that compares values sets a thermo_style that computes them). */
  pre?: string;
  /** Lines run after "run 0" (a row that compares values prints them: print "VAL ..."). */
  tail?: string;
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
  /** the rows, from the oracle cases (smallest first) and the oracle directory */
  rows: (cases: Case[], oracleDir: string) => ParityRow[];
  /**
   * Rows (by style) whose native result comes from undefined behaviour, with the reason: native reads or writes outside
   * its arrays, so its result is not a rule the engine can follow. The runner marks them skip.
   */
  undefined?: Record<string, string>;
  /**
   * Rows (by style) whose printed values are compared within this absolute tolerance instead of 1e-10: a value that is a
   * near-complete cancellation of much larger terms, so the order of the floating-point sums (which the engine does not
   * follow) shows above the relative 1e-8. Each entry says why in a comment.
   */
  abs?: Record<string, number>;
  /**
   * Rows (by style) native runs that this engine refuses with an error naming what it cannot do (the reason): the suite
   * checks the engine's error instead of native's result.
   */
  gaps?: Record<string, string>;
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

/** The words of a line before any comment; a line that continues (&) or substitutes variables ($) gives null. */
const words = (line: string): string[] | null => {
  const code = line.split('#')[0];
  if (code.includes('&') || code.includes('$')) return null;
  const w = code.trim().split(/\s+/).filter((x) => x !== '');
  return w.length ? w : null;
};

/** The key of a line (one row per key), or null when the line is not taken. */
type LineKey = (w: string[], lines: string[], i: number) => string | null;
/** The changed words of a line (with the lines of its case), or null when the change does not apply to it. */
type Change = (w: string[], lines: string[]) => string[] | null;

/**
 * A matrix: for each key, the first line (smallest case first, then in order) that has the key and to which the change
 * applies. A case is read only up to its first include, jump, label, next or read_restart line; a line continued with &,
 * the lines continuing it, and the lines of a triple-quoted block are not taken.
 */
const pick = (keyOf: LineKey, change: Change) => (cases: Case[]): ParityRow[] => {
  const rows = new Map<string, ParityRow>();
  for (const c of cases) {
    const lines = c.text.split('\n');
    let continued = false, quoted = false;
    for (let i = 0; i < lines.length; i++) {
      if (FLOW.test(lines[i])) break;
      const code = lines[i].split('#')[0];
      const inBlock = continued || quoted || code.includes('"""');
      continued = code.trimEnd().endsWith('&');
      if ((code.split('"""').length - 1) % 2 === 1) quoted = !quoted;
      if (inBlock) continue;
      const w = words(lines[i]);
      if (!w) continue;
      const key = keyOf(w, lines, i);
      if (key === null || rows.has(key)) continue;
      const changed = change(w, lines);
      if (changed) rows.set(key, { case: c.name, line: i, text: changed.join(' '), style: key });
    }
  }
  return [...rows.values()].sort((a, b) => (a.style < b.style ? -1 : a.style > b.style ? 1 : 0));
};

/** A fix or compute line: "fix <style>" or "compute <style>". */
const fixComputeKey: LineKey = (w) => ((w[0] === 'fix' || w[0] === 'compute') && w.length >= 4 ? `${w[0]} ${w[3]}` : null);

/** The first number from word `from` on replaced by `value`. */
const firstNumber = (from: number, value: string): Change => (w, _lines) => {
  const k = w.findIndex((x, j) => j >= from && NUMBER.test(x));
  return k < 0 ? null : w.map((x, j) => (j === k ? value : x));
};

/** Commands that are not checked: control flow, variables and output only. */
const SKIP = new Set(['variable', 'print', 'if', 'label', 'jump', 'next', 'include', 'shell', 'python', 'run', 'rerun', 'minimize', 'log', 'echo', 'quit', 'clear']);
const COEFF = new Set(['pair_coeff', 'bond_coeff', 'angle_coeff', 'dihedral_coeff', 'improper_coeff']);
/** The word that names a command's variant (its style or keyword), when it is not word 1 of a *_style or *_modify line. */
const VARIANT: Record<string, number> = {
  region: 2, dump: 3, dump_modify: 2, group: 2, velocity: 2, displace_atoms: 2, create_atoms: 1, delete_atoms: 1,
  lattice: 1, set: 3, fix_modify: 2, compute_modify: 2, change_box: 2, read_data: 2, create_box: 2, write_data: 2,
};

/**
 * A coefficient line: "<command> <style>" with the style of the last *_style line before it (and the sub-style named on
 * the line under a hybrid style, word 3 of pair_coeff or word 2 of the others).
 */
const coeffKey: LineKey = (w, lines, i) => {
  if (!COEFF.has(w[0])) return null;
  const kind = w[0].replace('_coeff', '_style');
  let style: string | null = null;
  for (let j = i - 1; j >= 0 && style === null; j--) {
    const v = words(lines[j]);
    if (v && v[0] === kind && v.length > 1) style = v[1];
  }
  if (style === null) return null;
  const sub = w[0] === 'pair_coeff' ? w[3] : w[2];
  return style.startsWith('hybrid') && sub !== undefined ? `${w[0]} ${style} ${sub}` : `${w[0]} ${style}`;
};

/** Any other command: "<command> <variant>" (its style or keyword, see VARIANT), or the command alone. */
const commandKey: LineKey = (w) => {
  if (!/^[a-z][a-z_0-9]*$/.test(w[0]) || SKIP.has(w[0]) || COEFF.has(w[0]) || w[0] === 'fix' || w[0] === 'compute') return null;
  const at = VARIANT[w[0]] ?? (w[0].endsWith('_style') || w[0].endsWith('_modify') ? 1 : -1);
  return at >= 0 && w[at] !== undefined ? `${w[0]} ${w[at]}` : w[0];
};

/** Any line that is checked: a fix or compute, a coefficient line or another command. */
const lineKey: LineKey = (w, lines, i) => fixComputeKey(w, lines, i) ?? coeffKey(w, lines, i) ?? commandKey(w, lines, i);

const trailing: Change = (w) => [...w, 'extra'];
/** The last word dropped, when at least `keep` words remain. */
const truncated = (keep: number): Change => (w) => (w.length > keep ? w.slice(0, -1) : null);
/** The last number from word `from` on replaced by `value`. */
const lastNumber = (from: number, value: string): Change => (w) => {
  let k = -1;
  for (let j = w.length - 1; j >= from && k < 0; j--) if (NUMBER.test(w[j])) k = j;
  return k < 0 ? null : w.map((x, j) => (j === k ? value : x));
};
/** The first number after the type words of a coefficient line (pair_coeff I J ..., the others N ...). */
const coeffNumber = (value: string): Change => (w, lines) => firstNumber(w[0] === 'pair_coeff' ? 3 : 2, value)(w, lines);
/** The word at index k replaced (when the line has it). */
const wordAt = (k: number, value: string): Change => (w, _lines) => (w[k] !== undefined ? w.map((x, j) => (j === k ? value : x)) : null);

/** The group-ID word of the commands that name one, or -1. */
const groupWord = (w: string[]): number => {
  if (w[0] === 'fix' || w[0] === 'compute' || w[0] === 'dump') return 2;
  if (w[0] === 'velocity' || w[0] === 'displace_atoms') return 1;
  if ((w[0] === 'delete_atoms' || w[0] === 'set') && w[1] === 'group') return 2;
  return -1;
};
const groupKey: LineKey = (w, lines, i) => (groupWord(w) >= 0 ? lineKey(w, lines, i) : null);
const badGroup: Change = (w, lines) => wordAt(groupWord(w), 'nogroup')(w, lines);

/** The first c_ID, f_ID or v_name reference (not in a variable definition) replaced by c_nosuch, f_nosuch or v_nosuch. */
const REF = /^([cfv])_[A-Za-z0-9_]/;
const badReference: Change = (w) => {
  const k = w.findIndex((x, j) => j >= 1 && REF.test(x));
  return k < 0 ? null : w.map((x, j) => (j === k ? `${x[0]}_nosuch` : x));
};

/** The atom-type word of a coefficient line (the first type), of mass, or of create_atoms. */
const typeWord = (w: string[]): number => (COEFF.has(w[0]) || w[0] === 'mass' || w[0] === 'create_atoms' ? 1 : -1);
const typeKey: LineKey = (w, lines, i) => (typeWord(w) >= 0 && NUMBER.test(w[1] ?? '') ? lineKey(w, lines, i) : null);

/** The first yes/no/on/off value after word 1 replaced by maybe. */
const LOGICAL = new Set(['yes', 'no', 'on', 'off']);
const badLogical: Change = (w) => {
  const k = w.findIndex((x, j) => j >= 2 && LOGICAL.has(x));
  return k < 0 ? null : w.map((x, j) => (j === k ? 'maybe' : x));
};

/** The word after a region keyword (not the region command itself) replaced by noregion. */
const badRegion: Change = (w) => {
  if (w[0] === 'region') return null;
  const k = w.findIndex((x, j) => j >= 1 && w[j - 1] === 'region');
  return k < 0 ? null : w.map((x, j) => (j === k ? 'noregion' : x));
};

/**
 * The input file names of a case: its oracle-inputs, oracle-potentials and oracle-examples (base names), and the files of
 * the oracle directory (a case may name one without a directive).
 */
const caseFileNames = (lines: string[], oracleFiles: Set<string>): Set<string> => {
  const names = new Set(oracleFiles);
  for (const l of lines) {
    const m = /^#\s*oracle-(inputs|potentials|examples):(.*)$/.exec(l);
    if (m) for (const f of m[2].trim().split(/\s+/)) if (f) names.add(f.split('/').pop()!);
  }
  return names;
};

/** Lines that write a file (their file names are outputs, not inputs). */
const WRITERS = new Set(['write_data', 'write_dump', 'write_restart', 'write_coeff', 'dump', 'log']);
const fileKey: LineKey = (w, lines, i) => (WRITERS.has(w[0]) ? null : lineKey(w, lines, i));

/** The first word (after word 0) that names an input file of the case replaced by nofile.txt. */
const badFile = (oracleFiles: Set<string>): Change => (w, lines) => {
  const names = caseFileNames(lines, oracleFiles);
  const k = w.findIndex((x, j) => j >= 1 && names.has(x));
  return k < 0 ? null : w.map((x, j) => (j === k ? 'nofile.txt' : x));
};

/**
 * The values a row that compares values prints after run 0, the potential energy and the pressure, with a thermo_style
 * set before the run that computes both (a variable can read only the thermo keywords the thermo output computes).
 */
const THERMO = 'thermo_style custom step pe press\n';
const VALUES = 'print "VAL $(pe:%.15g) $(press:%.15g)"\n';
/** The same with the force norm, which also sees forces that add no energy or virial (fixes, a changed cutoff). */
const THERMO_F = 'thermo_style custom step pe press fnorm\n';
const VALUES_F = 'print "VAL $(pe:%.15g) $(press:%.15g) $(fnorm:%.15g)"\n';
/** Commands that run steps: a row that compares values keeps the case up to the first of them after its line. */
const RUNS = /^\s*(run|minimize|rerun|tad|prd|neb|temper)\b/;

/**
 * Rows that compare values: each picked row keeps the rest of its case up to the first run (or flow command) after the
 * changed line, then runs 0 steps and prints pe and press.
 */
const throughRun = (pickRows: (cases: Case[]) => ParityRow[], pre = THERMO, tail = VALUES) => (cases: Case[]): ParityRow[] => {
  const text = new Map(cases.map((c) => [c.name, c.text]));
  return pickRows(cases).map((r) => {
    const lines = text.get(r.case)!.split('\n');
    let until = lines.length;
    for (let i = r.line + 1; i < lines.length; i++) {
      if (RUNS.test(lines[i]) || FLOW.test(lines[i])) { until = i; break; }
    }
    // a row's own pre lines (adapt-pair's fix adapt) come before the thermo settings
    return { ...r, until, pre: `${r.pre ?? ''}${pre}`, tail };
  });
};

/**
 * fix adapt on each pair style: the smallest case that has the style up to its first run (adaptRows), then
 * "variable adpv equal 1.1" and "fix adpv all adapt 0 pair <style> <name> <I J> v_adpv scale yes" before its run 0 (so
 * the box exists whatever the order of its setup), one row for each name the style's native extract() knows
 * (src/engine/fix/adaptPairsTable.ts).
 */
const pairStyleKey: LineKey = (w) => (w[0] === 'pair_style' && w.length > 1 && !w[1].startsWith('hybrid') ? `pair_style ${w[1]}` : null);
/**
 * extract() names whose native member is an int (src/ at b332f28: the AMOEBA flags and type arrays, the TIP4P types,
 * the charmm implicit and dihedflag, ewald_order and mix_flag, coul/streitz kspacetype, the spica lj_type array, pace's
 * flag_corerep_factor): fix adapt writes a double through the pointer, so native's result is undefined behaviour.
 */
const INT_PARAMS = new Set([
  'amtype', 'atomic_num', 'bond_flag', 'angle_flag', 'dihedral_flag', 'improper_flag', 'urey_flag', 'pitorsion_flag',
  'bitorsion_flag', 'typeO', 'typeH', 'typeA', 'typeB', 'implicit', 'dihedflag', 'ewald_order', 'ewald_mix', 'kspacetype',
  'lj_type', 'corerep_flag',
]);
/** int extract() members of one style only (EXTRA-PAIR/pair_wf_cut.h: int **nu, **mu; DRUDE/pair_coul_tt.h: int **ntt). */
const INT_PARAMS_OF: Record<string, readonly string[]> = { 'wf/cut': ['nu', 'mu'], 'coul/tt': ['ntt'] };
/**
 * extract() names whose native member is a tetramer array (double ****, src/CG-DNA/pair_oxdna_stk.h and
 * pair_oxdna3_xstk.h at b332f28) handed out as dim 2: fix adapt writes a double through a pointer to pointers.
 */
const STK_TETRAMER = ['cut_st_0', 'cut_st_c', 'cut_st_lo', 'cut_st_hi', 'cut_st_lc', 'cut_st_hc', 'shift_st', 'a_st4', 'dtheta_st4_ast', 'b_st4', 'dtheta_st4_c'];
const TETRAMER_OF: Record<string, readonly string[]> = {
  'oxdna/stk': STK_TETRAMER, 'oxdna2/stk': STK_TETRAMER, 'oxdna3/stk': STK_TETRAMER,
  'oxdna3/xstk': ['33', '55'].flatMap((set) => ['0', 'c', 'lo', 'hi', 'lc', 'hc'].map((k) => `cut_xst_${k}_${set}`)),
};
const adaptPairUndefined = (): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const [style, names] of Object.entries(TETRAMER_OF)) {
    for (const name of names) out[`pair ${style} ${name}`] = `native's ${name} is a tetramer array (double ****) that extract() gives as dim 2 (undefined behaviour)`;
  }
  for (const [style, names] of Object.entries(PAIR_EXTRACT)) {
    for (const [name] of names) {
      if (INT_PARAMS.has(name) || INT_PARAMS_OF[style]?.includes(name)) {
        out[`pair ${style} ${name}`] = `native writes a double through the int ${name} (undefined behaviour)`;
      }
    }
  }
  return out;
};
/**
 * The rows of one adapt matrix: for each pair style, the smallest case that has it as its pair_style, else the smallest
 * that has it once in a hybrid pair_style line, where the type range is the I J of that sub-style's first pair_coeff
 * (native refuses a range with pairs of another sub-style: "Fix adapt type pair range is not valid for pair hybrid
 * sub-style"); one row per extract() name that `names` keeps, with the attribute's lines before run 0.
 */
const adaptRows = (cases: Case[], names: (name: string, style: string) => boolean, line: (style: string, name: string, ij: string) => string): ParityRow[] => {
  const rowsOf = (r: ParityRow, style: string, ij: string): ParityRow[] =>
    (PAIR_EXTRACT[style] ?? []).filter(([name]) => names(name, style)).map(([name]) => ({ ...r, pre: line(style, name, ij), style: `pair ${style} ${name}` }));
  const plain = pick(pairStyleKey, (w) => w)(cases);
  const out = plain.flatMap((r) => rowsOf(r, r.style.split(' ')[1], '* *'));
  const covered = new Set(plain.map((r) => r.style.split(' ')[1]));
  const text = new Map(cases.map((c) => [c.name, c.text]));
  for (const style of Object.keys(PAIR_EXTRACT)) {
    if (covered.has(style)) continue;
    const once: LineKey = (w) => (w[0] === 'pair_style' && w[1]?.startsWith('hybrid') && w.filter((x) => x === style).length === 1 ? `pair_style ${style}` : null);
    for (const r of pick(once, (w) => w)(cases)) {
      const lines = text.get(r.case)!.split('\n');
      const coeff = lines.slice(r.line + 1).map((l) => words(l)).find((w) => w?.[0] === 'pair_coeff' && w[3] === style);
      if (coeff) out.push(...rowsOf(r, style, `${coeff[1]} ${coeff[2]}`));
    }
  }
  return out.sort((a, b) => (a.style < b.style ? -1 : a.style > b.style ? 1 : 0));
};

const adaptPairRows = (cases: Case[]): ParityRow[] => adaptRows(cases, () => true, (style, name, ij) =>
  `variable adpv equal 1.1\nfix adpv all adapt 0 pair ${style} ${name} ${ij} v_adpv scale yes\n`);

/**
 * The cutoff and geometry parameters of the adapt-pair rows (the extract() names with cut in them, and qdist) scaled
 * down, with reset no and a second run 0: "variable adpv equal 0.9" and "fix adpv all adapt 0 pair <style>
 * <name> * * v_adpv scale yes reset no". A smaller cutoff shows a change init_one or the force computation makes at
 * once (a larger one stays inside the cutoffs of the run's init), and the second run shows the value the next init
 * reads.
 */
const adaptPairDownRows = (cases: Case[]): ParityRow[] => adaptRows(cases, (name) => /cut|qdist/.test(name), (style, name, ij) =>
  `variable adpv equal 0.9\nfix adpv all adapt 0 pair ${style} ${name} ${ij} v_adpv scale yes reset no\n`);

/**
 * fix adapt on each bond, angle, dihedral and improper style, as adapt-pair: the smallest case that has the style as
 * its <kind>_style, else the smallest that has it in a hybrid <kind>_style line, where the type is the I of that
 * sub-style's first coefficient line (native refuses a type of another sub-style: "Fix adapt type bond range is not
 * valid for bond hybrid sub-style"); "variable adpv equal 1.1" and "fix adpv all adapt 0 <kind> <style> <name> <I>
 * v_adpv scale yes" before its run 0, one row for each name the style's native extract() knows
 * (src/engine/fix/adaptBondedTable.ts).
 */
const BONDED_KINDS = new Set(['bond', 'angle', 'dihedral', 'improper']);
const bondedStyleKey: LineKey = (w) => {
  const kind = w[0].endsWith('_style') ? w[0].slice(0, -6) : '';
  return BONDED_KINDS.has(kind) && w.length > 1 && w[1] !== 'hybrid' ? `${kind} ${w[1]}` : null;
};
const adaptBondedRows = (cases: Case[]): ParityRow[] => {
  const rowsOf = (r: ParityRow, key: string, it: string): ParityRow[] =>
    (BONDED_EXTRACT[key] ?? []).map((name) => ({
      ...r, pre: `variable adpv equal 1.1\nfix adpv all adapt 0 ${key} ${name} ${it} v_adpv scale yes\n`, style: `${key} ${name}`,
    }));
  const plain = pick(bondedStyleKey, (w) => w)(cases);
  const out = plain.flatMap((r) => rowsOf(r, r.style, '*'));
  const covered = new Set(plain.map((r) => r.style));
  const text = new Map(cases.map((c) => [c.name, c.text]));
  for (const key of Object.keys(BONDED_EXTRACT)) {
    if (covered.has(key)) continue;
    const [kind, style] = key.split(' ');
    const sub: LineKey = (w) => (w[0] === `${kind}_style` && w[1] === 'hybrid' && w.slice(2).includes(style) ? key : null);
    for (const r of pick(sub, (w) => w)(cases)) {
      const lines = text.get(r.case)!.split('\n');
      const coeff = lines.slice(r.line + 1).map((l) => words(l)).find((w) => w?.[0] === `${kind}_coeff` && w[2] === style);
      if (coeff) out.push(...rowsOf(r, key, coeff[1]));
    }
  }
  return out.sort((a, b) => (a.style < b.style ? -1 : a.style > b.style ? 1 : 0));
};
/**
 * extract() names whose native member is an int array (src/ at b332f28: EXTRA-MOLECULE/angle_cosine_periodic.h
 * multiplicity and b, MOFFF/angle_cosine_buck6d.h multiplicity, MOLECULE/dihedral_charmm.h (and charmmfsw) n and d,
 * MOLECULE/improper_cvff.h d and n): fix adapt writes a double through the pointer, so native's result is undefined.
 */
const BONDED_INT_PARAMS: Record<string, readonly string[]> = {
  'angle cosine/periodic': ['multiplicity', 'b'], 'angle cosine/buck6d': ['multiplicity'],
  'dihedral charmm': ['n', 'd'], 'dihedral charmmfsw': ['n', 'd'], 'improper cvff': ['d', 'n'],
};
/**
 * extract() names whose native member is a per-type array of arrays (double **, EXTRA-MOLECULE/angle_gaussian.h and
 * bond_gaussian.h at b332f28: one value per Gaussian term) handed out as dim 1: fix adapt writes doubles over the
 * array's pointers.
 */
const BONDED_POINTER_PARAMS: Record<string, readonly string[]> = {
  'angle gaussian': ['alpha', 'width', 'theta0'], 'bond gaussian': ['alpha', 'width', 'r0'],
};
const adaptBondedUndefined = (): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const [key, names] of Object.entries(BONDED_INT_PARAMS)) {
    for (const name of names) out[`${key} ${name}`] = `native writes a double through the int ${name} (undefined behaviour)`;
  }
  for (const [key, names] of Object.entries(BONDED_POINTER_PARAMS)) {
    for (const name of names) out[`${key} ${name}`] = `native's ${name} is a double ** that extract() gives as dim 1 (undefined behaviour)`;
  }
  return out;
};

/**
 * fix adapt/fep and compute fep share fix adapt's per-name access (adapt-pair checks every name); their own rules are
 * per style (src/FEP/fix_adapt_fep.cpp and compute_fep.cpp init at b332f28: Force::pair_match(pstyle, 1), extract(),
 * the dim, the hybrid range, Pair::reinit). So their rows take, for each case of adapt-pair, the style's first
 * extract() name of the dims given whose native result is defined.
 */
const firstNamesOf = (dims: readonly number[]) => {
  const undef = adaptPairUndefined();
  const chosen = new Map<string, Set<string>>();
  return (name: string, style: string): boolean => {
    if (!chosen.has(style)) {
      const set = new Set<string>();
      for (const d of dims) {
        const e = (PAIR_EXTRACT[style] ?? []).find(([n, dim]) => dim === d && !undef[`pair ${style} ${n}`]);
        if (e) set.add(e[0]);
      }
      chosen.set(style, set);
    }
    return chosen.get(style)!.has(name);
  };
};
/** "fix adpv all adapt/fep 0 pair <style> <name> <I J> v_adpv scale yes" (x = 1.1) before run 0. */
const adaptFepPairRows = (cases: Case[]): ParityRow[] => adaptRows(cases, firstNamesOf([2, 0]), (style, name, ij) =>
  `variable adpv equal 1.1\nfix adpv all adapt/fep 0 pair ${style} ${name} ${ij} v_adpv scale yes\n`);
/**
 * "compute adpf all fep 300.0 pair <style> <name> <I J> v_adpd" with adpd = 0.01, its energy difference c_adpf[1]
 * printed after run 0 (native compute fep writes a dim-0 name through its array pointer: undefined, so dim 2 only).
 */
const computeFepPairRows = (cases: Case[]): ParityRow[] => adaptRows(cases, firstNamesOf([2]), (style, name, ij) =>
  `variable adpd equal 0.01\ncompute adpf all fep 300.0 pair ${style} ${name} ${ij} v_adpd\n`);
const THERMO_FEP = 'thermo_style custom step pe press c_adpf[1]\n';
/**
 * Rows native cannot run or whose result is undefined: the styles only the ML-PACE and APIP native builds have
 * (LMP_BUILDS), which are built without the FEP package, and, for fix adapt/fep, lepton/coul's cut_coul, which
 * extract() hands out without a kspace style as dim 2 pointing at the address of its cut array pointer
 * (src/LEPTON/pair_lepton_coul.cpp at b332f28: "return (void *) &cut").
 */
const fepUndefined = (fix: boolean): Record<string, string> => {
  const out: Record<string, string> = {};
  for (const [style, names] of Object.entries(PAIR_EXTRACT)) {
    if (!/^pace|\/apip$/.test(style)) continue;
    for (const [name] of names) out[`pair ${style} ${name}`] = 'the native builds with ML-PACE and APIP lack the FEP package';
  }
  if (fix) out['pair lepton/coul cut_coul'] = "native's extract() gives &cut as dim 2 without a kspace style (undefined behaviour)";
  return out;
};
const VALUES_FEP = 'print "VAL $(pe:%.15g) $(press:%.15g) $(c_adpf[1]:%.15g)"\n';

/** The first number after the type words of a coefficient line multiplied by `factor`. */
const coeffScaled = (factor: number): Change => (w, _lines) => {
  const from = w[0] === 'pair_coeff' ? 3 : 2;
  const k = w.findIndex((x, j) => j >= from && NUMBER.test(x));
  return k < 0 ? null : w.map((x, j) => (j === k ? String(Number(x) * factor) : x));
};

/** A *_style line (pair, bond, angle, dihedral, improper or kspace): "<command> <style>". */
const STYLES = new Set(['pair_style', 'bond_style', 'angle_style', 'dihedral_style', 'improper_style', 'kspace_style']);
const styleKey: LineKey = (w, lines, i) => (STYLES.has(w[0]) ? commandKey(w, lines, i) : null);

/**
 * The first number from word `from` on multiplied by `factor`: an integer stays an integer (rounded), a real number is
 * written with 12 significant digits. changedOnly: a number the change leaves as it is (an integer 1) gives no row.
 */
const scaled = (from: number, factor: number, changedOnly = false): Change => (w, _lines) => {
  const k = w.findIndex((x, j) => j >= from && NUMBER.test(x));
  if (k < 0) return null;
  const v = /^[-+]?\d+$/.test(w[k]) ? String(Math.round(Number(w[k]) * factor)) : String(Number((Number(w[k]) * factor).toPrecision(12)));
  return changedOnly && v === w[k] ? null : w.map((x, j) => (j === k ? v : x));
};

/** A fix line: "fix <style>". */
const fixKey: LineKey = (w) => (w[0] === 'fix' && w.length >= 4 ? `fix ${w[3]}` : null);

export const MATRICES: Record<string, MatrixDef> = {
  'fix-compute-word': {
    about: 'the first number of each fix and compute style (from word 4) replaced by the word bogus',
    rows: pick(fixComputeKey, firstNumber(4, 'bogus')),
  },
  'fix-compute-zero': {
    about: 'the first number of each fix and compute style (from word 4) replaced by 0',
    rows: pick(fixComputeKey, firstNumber(4, '0')),
    undefined: {
      'fix smd/wall_surface': 'native creates the wall atoms with type 0, outside the atom types (the engine refuses the type)',
    },
  },
  'fix-compute-negative': {
    about: 'the first number of each fix and compute style (from word 4) replaced by -1',
    rows: pick(fixComputeKey, firstNumber(4, '-1')),
    undefined: {
      'fix restrain': 'native looks atom ID -1 up as map_array[-1], outside its atom map (the engine refuses the ID)',
      'fix smd/wall_surface': 'native creates the wall atoms with type -1, outside the atom types (the engine refuses the type)',
    },
  },
  'fix-compute-trailing': { about: 'an extra word after each fix and compute style line', rows: pick(fixComputeKey, trailing) },
  'command-trailing': { about: 'an extra word after each other command (one line per command and style)', rows: pick(commandKey, trailing) },
  'coeff-trailing': { about: 'an extra word after each coefficient line (one line per style)', rows: pick(coeffKey, trailing) },
  'fix-compute-truncated': { about: 'the last word of each fix and compute style line dropped', rows: pick(fixComputeKey, truncated(4)) },
  'command-truncated': { about: 'the last word of each other command dropped', rows: pick(commandKey, truncated(1)) },
  'coeff-truncated': { about: 'the last word of each coefficient line dropped', rows: pick(coeffKey, truncated(2)) },
  'command-word': { about: 'the first number of each other command (from word 1) replaced by the word bogus', rows: pick(commandKey, firstNumber(1, 'bogus')) },
  'coeff-word': { about: 'the first number after the type words of each coefficient line replaced by the word bogus', rows: pick(coeffKey, coeffNumber('bogus')) },
  'fix-compute-last-word': { about: 'the last number of each fix and compute style line replaced by the word bogus', rows: pick(fixComputeKey, lastNumber(4, 'bogus')) },
  'group-id': { about: 'the group ID of each fix, compute, dump, velocity, displace_atoms, delete_atoms group and set group line replaced by nogroup', rows: pick(groupKey, badGroup) },
  'reference': { about: 'the first c_ID, f_ID or v_name reference of each line replaced by c_nosuch, f_nosuch or v_nosuch', rows: pick(lineKey, badReference) },
  'type': {
    about: 'the atom type of each coefficient, mass and create_atoms line replaced by 99',
    rows: pick(typeKey, wordAt(1, '99')),
    undefined: {
      'pair_coeff eff/cut': 'native reads ecp_type[99] past its per-type array for the ECP line (the engine refuses the type)',
    },
  },
  'yes-no': { about: 'the first yes, no, on or off value of each line replaced by maybe', rows: pick(lineKey, badLogical) },
  'region-id': { about: 'the region ID after a region keyword replaced by noregion', rows: pick(lineKey, badRegion) },
  'coeff-scaled': {
    about: 'the first number after the type words of each coefficient line multiplied by 1.1, pe and press compared after run 0',
    rows: throughRun(pick(coeffKey, coeffScaled(1.1))),
  },
  'style-scaled': {
    about: 'the first number after the style word of each pair, bond, angle, dihedral, improper and kspace style line multiplied by 1.1, pe, press and fnorm compared after run 0',
    rows: throughRun(pick(styleKey, scaled(2, 1.1)), THERMO_F, VALUES_F),
  },
  'command-scaled': {
    about: 'the first number of each other command (from word 1) multiplied by 1.1, pe, press and fnorm compared after run 0',
    rows: throughRun(pick(commandKey, scaled(1, 1.1, true)), THERMO_F, VALUES_F),
    abs: {
      // a perfect eam lattice: press -0.0279 bar is the sum x.f of virial terms of order 1e3 eV (1e5 times larger); the
      // engine's sum over owned and ghost atoms differs from native's at 1e-14 of the terms, 3.6e-9 bar
      'mass': 1e-8,
    },
  },
  'fix-scaled': {
    about: 'the first number of each fix line (from word 4) multiplied by 1.1, pe, press and fnorm compared after run 0',
    rows: throughRun(pick(fixKey, scaled(4, 1.1, true)), THERMO_F, VALUES_F),
  },
  'adapt-pair': {
    about: 'fix adapt 0 pair <style> <name> * * v_x scale yes (x = 1.1) for each name of each pair style\'s native extract(), pe, press and fnorm compared after run 0',
    rows: throughRun(adaptPairRows, THERMO_F, VALUES_F),
    undefined: adaptPairUndefined(),
  },
  'adapt-pair-down': {
    about: 'fix adapt 0 pair <style> <name> * * v_x scale yes reset no (x = 0.9) for each cutoff and geometry name of each pair style\'s native extract(), pe, press and fnorm compared after run 0 and after a second run 0',
    rows: throughRun(adaptPairDownRows, THERMO_F, `${VALUES_F}run 0\n${VALUES_F}`),
    undefined: adaptPairUndefined(),
  },
  'adapt-bonded': {
    about: 'fix adapt 0 <kind> <style> <name> * v_x scale yes (x = 1.1) for each name of each bond, angle, dihedral and improper style\'s native extract(), pe, press and fnorm compared after run 0',
    rows: throughRun(adaptBondedRows, THERMO_F, VALUES_F),
    undefined: adaptBondedUndefined(),
  },
  'adapt-fep-pair': {
    about: 'fix adapt/fep 0 pair <style> <name> * * v_x scale yes (x = 1.1) for the first per-type-pair and the first global name of each pair style\'s native extract(), pe, press and fnorm compared after run 0',
    rows: throughRun(adaptFepPairRows, THERMO_F, VALUES_F),
    undefined: { ...adaptPairUndefined(), ...fepUndefined(true) },
  },
  'compute-fep-pair': {
    about: 'compute fep 300.0 pair <style> <name> * * v_d (d = 0.01) for the first per-type-pair name of each pair style\'s native extract(), pe, press and the energy difference c_[1] compared after run 0',
    rows: throughRun(computeFepPairRows, THERMO_FEP, VALUES_FEP),
    undefined: { ...adaptPairUndefined(), ...fepUndefined(false) },
  },
  'file-name': { about: 'the first input file name of each line replaced by nofile.txt', rows: (cases, dir) => pick(fileKey, badFile(new Set(readdirSync(dir).filter((f) => !f.endsWith('.in')))))(cases) },
};

/** The row's input: the case up to its line (without the # oracle header lines), the replacing line, then run 0. */
export const parityInput = (caseText: string, row: ParityRow): string => {
  const lines = caseText.split('\n');
  if (row.until === undefined) {
    const kept = lines.slice(0, row.line).filter((l) => !l.startsWith('# oracle'));
    return `${kept.join('\n')}\n${row.text}\n${row.pre ?? ''}run 0\n${row.tail ?? ''}`;
  }
  // a row that compares values: the changed line in place, the case up to its first run, then run 0 and the tail
  const kept = lines.slice(0, row.until).map((l, i) => (i === row.line ? row.text : l)).filter((l) => !l.startsWith('# oracle'));
  return `${kept.join('\n')}\n${row.pre ?? ''}run 0\n${row.tail ?? ''}`;
};

/**
 * The result in the fixture's form compared as native's: the status (OK or the first ERROR line) exactly, and the
 * printed values ("OK | VAL a b ...") within a relative 1e-8 (plus 1e-10 absolute), as the oracle cases compare.
 */
export const sameResult = (engine: string, native: string, abs = 1e-10): boolean => {
  const [es, ev = ''] = engine.split(' | ');
  const [ns, nv = ''] = native.split(' | ');
  if (es !== ns) return false;
  const a = ev.split(/\s+/).filter((x) => x !== ''), b = nv.split(/\s+/).filter((x) => x !== '');
  if (a.length !== b.length) return false;
  return a.every((x, i) => {
    if (x === b[i]) return true;
    const p = Number(x), q = Number(b[i]);
    return Number.isFinite(p) && Number.isFinite(q) && Math.abs(p - q) <= abs + 1e-8 * Math.abs(q);
  });
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
