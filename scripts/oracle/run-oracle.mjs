#!/usr/bin/env node
/**
 * Oracle fixtures: runs every input in tests/oracle/*.in through a native
 * LAMMPS binary and stores what the engine must reproduce in
 * tests/fixtures/oracle/<name>.json:
 *   - thermo rows of every thermo output (full precision: the case sets
 *     thermo_modify format float %.15g), keyed by the thermo_style keywords
 *   - the final per-atom state (id, unwrapped x, v, f, type), from a
 *     write_dump appended by this script (the test appends the same command)
 *   - files the input wrote that the case lists in "# oracle-files: ..."
 *
 * LAMMPS is used as a black box (its output only). Usage:
 *   LMP=/path/to/lmp node scripts/oracle/run-oracle.mjs [case-name ...]
 * A case may list data/potential files it reads in "# oracle-inputs: a b";
 * they are looked up next to the .in file.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, readdirSync, writeFileSync, copyFileSync, existsSync, rmSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';

const ROOT = new URL('../..', import.meta.url).pathname;
const CASES = join(ROOT, 'tests/oracle');
const OUT = join(ROOT, 'tests/fixtures/oracle');
const LMP = process.env.LMP ?? 'lmp';

export const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';

const header = (text, key) => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

/** Keywords of the last thermo_style custom command before each run. */
const thermoKeywords = (text) => {
  let kw = ['step', 'temp', 'epair', 'emol', 'etotal', 'press'];
  for (const line of text.split('\n')) {
    const w = line.replace(/#.*/, '').trim().split(/\s+/);
    if (w[0] === 'thermo_style') kw = w[1] === 'custom' ? w.slice(2) : w[1] === 'multi' ? ['etotal', 'ke', 'temp', 'pe', 'ebond', 'eangle', 'edihed', 'eimp', 'evdwl', 'ecoul', 'elong', 'press'] : ['step', 'temp', 'epair', 'emol', 'etotal', 'press'];
  }
  return kw;
};

const parseLog = (log, keywords) => {
  const rows = [];
  const lines = log.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*Step\s/.test(lines[i]) && !/^\s*(Step|Elapsed|Time)\b/.test(lines[i])) continue;
    const ncol = lines[i].trim().split(/\s+/).length;
    for (let j = i + 1; j < lines.length; j++) {
      // warnings (e.g. "FENE bond too long") are printed between thermo rows
      if (/^WARNING/.test(lines[j])) continue;
      const w = lines[j].trim().split(/\s+/);
      if (w.length !== ncol || !w.every((x) => /^[-+0-9.eE]+$/.test(x) || /^-?(nan|inf)$/i.test(x))) break;
      const row = {};
      keywords.forEach((k, c) => { row[k] = Number(w[c]); });
      rows.push(row);
    }
  }
  return rows;
};

const parseDump = (text) => {
  const lines = text.split('\n');
  const k = lines.indexOf(lines.find((l) => l.startsWith('ITEM: ATOMS')));
  const cols = lines[k].split(/\s+/).slice(2);
  const atoms = [];
  for (const l of lines.slice(k + 1)) {
    if (!l.trim()) continue;
    const w = l.trim().split(/\s+/).map(Number);
    const a = {};
    cols.forEach((c, i) => { a[c] = w[i]; });
    atoms.push(a);
  }
  return atoms;
};

const names = process.argv.slice(2);
const cases = readdirSync(CASES).filter((f) => f.endsWith('.in')).map((f) => basename(f, '.in')).filter((n) => !names.length || names.includes(n));
let failed = 0;
for (const name of cases) {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const dir = mkdtempSync(join(tmpdir(), `oracle-${name}-`));
  for (const f of header(text, 'oracle-inputs')) copyFileSync(join(CASES, f), join(dir, f));
  writeFileSync(join(dir, 'in.case'), `${text}\n${FINAL_DUMP}\n`);
  try {
    execFileSync(LMP, ['-in', 'in.case', '-log', 'log.lammps', '-screen', 'none'], { cwd: dir, stdio: 'pipe' });
  } catch (e) {
    const log = existsSync(join(dir, 'log.lammps')) ? readFileSync(join(dir, 'log.lammps'), 'utf8') : '';
    console.error(`✗ ${name}: LAMMPS failed\n${(log.match(/ERROR.*/) ?? [String(e)])[0]}`);
    failed++;
    continue;
  }
  const log = readFileSync(join(dir, 'log.lammps'), 'utf8');
  const thermo = parseLog(log, thermoKeywords(text));
  const atoms = parseDump(readFileSync(join(dir, 'oracle_final.dump'), 'utf8'));
  const files = {};
  for (const f of header(text, 'oracle-files')) files[f] = readFileSync(join(dir, f), 'utf8');
  const version = (/LAMMPS \((.*?)\)/.exec(log) ?? [])[1] ?? 'unknown';
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ lammps: version, thermo, atoms, files }, null, 0) + '\n');
  console.log(`✓ ${name}: ${thermo.length} thermo rows, ${atoms.length} atoms (LAMMPS ${version})`);
  rmSync(dir, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
