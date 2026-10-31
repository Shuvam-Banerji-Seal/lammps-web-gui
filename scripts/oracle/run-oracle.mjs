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
 * they are looked up next to the .in file. "# oracle-potentials: a b" lists
 * unmodified LAMMPS potential files from third_party/lammps/potentials; "# oracle-examples: dir/a" lists
 * unmodified files of the LAMMPS examples tree in third_party/lammps/examples (copied under their base name; a
 * directory "dir/sub/" is copied with its files under sub/).
 * The header oracle-partitions: N runs the case on N partitions (replicas) with
 * mpirun -np N lmp -partition Nx1. Two kinds:
 *  - a NEB case (its universe log has the header line starting with Step MaxReplicaForce or MaxReplicaTorque): the fixture holds the
 *    NEB table rows of the universe log (each printed step is one row) and the Climbing replica lines; no
 *    write_dump is appended.
 *  - any other partitioned case (e.g. fix pimd): the fixture holds, per partition K, the thermo rows of its log
 *    (log.lammps.K) and its final atom state. A tempering case (temper, temper/npt, temper/grem) keeps the swap table of its universe log
 *    (the "Step T0 T1 ..." rows), and a prd or tad case keeps its event table ("Step CPU Clock ..." or "Step CPU N M ..."
 *    rows, CPU dropped); both are fixture key table, compared exactly by the harness. A world-style variable oracle_p (0..N-1) is defined and the final dump
 *    is written to oracle_final.${oracle_p}.dump, so every partition writes its own file. The fixture's perPartition
 *    array holds {thermo, atoms} for K = 0..N-1.
 */
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, writeFileSync, copyFileSync, existsSync, rmSync } from 'node:fs';
import { join, basename } from 'node:path';
import { tmpdir } from 'node:os';

const ROOT = new URL('../..', import.meta.url).pathname;
const CASES = join(ROOT, 'tests/oracle');
const OUT = join(ROOT, 'tests/fixtures/oracle');
const POTENTIALS = join(ROOT, 'third_party/lammps/potentials');
const EXAMPLES = join(ROOT, 'third_party/lammps/examples');
const LMP = process.env.LMP ?? 'lmp';
const MPIRUN = process.env.MPIRUN ?? 'mpirun';

export const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';
/** Binary output files (dump dcd, xtc and h5md), stored in the fixture as base64. */
const BINARY_FILE = /\.(dcd|xtc|h5)$/;

/** Partitioned (non-NEB) cases: the per-partition final dump, one file per partition (see the header above). */
export const partitionFinalDump = (n) => `variable oracle_p world ${[...Array(n).keys()].join(' ')}\nwrite_dump all custom oracle_final.\${oracle_p}.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id`;

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

/**
 * A wildcard keyword (f_2[*][1]) expands to a run of native header names of the same shape (f_2[1][1], f_2[2][1]):
 * the count comes from the log header, the other keywords keep their position.
 */
const expandWildcardKeywords = (keywords, header) => {
  if (!keywords.some((k) => k.includes('*'))) return keywords;
  const out = [];
  let h = 0;
  for (const k of keywords) {
    if (!k.includes('*')) { out.push(k); h++; continue; }
    const re = new RegExp('^' + k.replace(/[.+?^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '\\d+') + '$');
    while (h < header.length && re.test(header[h])) { out.push(header[h]); h++; }
  }
  return out;
};

const parseLog = (log, keywords) => {
  const rows = [];
  const lines = log.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (!/^\s*Step\s/.test(lines[i]) && !/^\s*(Step|Elapsed|Time)\b/.test(lines[i])) continue;
    const header = lines[i].trim().split(/\s+/);
    const ncol = header.length;
    const kws = expandWildcardKeywords(keywords, header);
    // a table with another column count is an earlier thermo_style (keywords of the last style do not apply)
    if (ncol !== kws.length) continue;
    for (let j = i + 1; j < lines.length; j++) {
      // warnings (e.g. "FENE bond too long") are printed between thermo rows, some followed by a
      // "For more information see https://docs.lammps.org/err...." line
      if (/^WARNING/.test(lines[j]) || /^For more information see/.test(lines[j])) continue;
      // fix SRP/REACT remakes its bond particles during the run and logs it between thermo rows
      if (/^Removed\/inserted \d+\/\d+ bond particles\.$/.test(lines[j].trim())) continue;
      const w = lines[j].trim().split(/\s+/);
      if (w.length !== ncol || !w.every((x) => /^[-+0-9.eE]+$/.test(x) || /^-?(nan|inf)$/i.test(x))) break;
      const row = {};
      kws.forEach((k, c) => { row[k] = Number(w[c]); });
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

/**
 * The NEB table of a universe log (neb.rst; neb/spin prints "Step MaxReplicaTorque ..."): the header line names the
 * columns; each following line with the same number of numeric fields is one printed step. Climbing lines
 * are kept in order with the row index they follow.
 */
const parseNeb = (log) => {
  const rows = [];
  const climbing = [];
  const lines = log.split('\n');
  let keys = null;
  // neb/spin (neb_spin.cpp print_status) writes "..." as a literal in its header and names only the first and last
  // verbose pair, so its columns are rebuilt from the row width: 9 + 2N values (9 + 4N verbose, the GradV<r>dottan
  // and DN<r> pairs after the RD/PE pairs)
  let spinVerbose = null;
  for (const line of lines) {
    if (/^\s*Step\s+MaxReplicaTorque\b/.test(line)) { spinVerbose = /dottan/.test(line); keys = null; continue; }
    if (/^\s*Step\s+MaxReplicaForce\b/.test(line)) { spinVerbose = null; keys = line.trim().split(/\s+/); continue; }
    const c = /^Climbing replica = (\d+)/.exec(line);
    if (c) { climbing.push({ replica: Number(c[1]), row: rows.length }); continue; }
    if (spinVerbose !== null && keys === null) {
      const w0 = line.trim().split(/\s+/);
      if (w0.length < 9 || !w0.every((x) => /^[-+0-9.eE]+$/.test(x) || /^-?(nan|inf)$/i.test(x))) continue;
      const n = (w0.length - 9) / (spinVerbose ? 4 : 2);
      if (!Number.isInteger(n) || n < 2) continue;
      keys = ['Step', 'MaxReplicaTorque', 'MaxAtomTorque', 'GradV0', 'GradV1', 'GradVc', 'EBF', 'EBR', 'RDT'];
      for (let r = 1; r <= n; r++) keys.push(`RD${r}`, `PE${r}`);
      if (spinVerbose) for (let r = 0; r < n; r++) keys.push(`GradV${r}dottan`, `DN${r}`);
    }
    if (!keys) continue;
    const w = line.trim().split(/\s+/);
    if (w.length !== keys.length || !w.every((x) => /^[-+0-9.eE]+$/.test(x) || /^-?(nan|inf)$/i.test(x))) continue;
    const row = {};
    keys.forEach((k, c2) => { row[k] = Number(w[c2]); });
    rows.push(row);
  }
  return { rows, climbing };
};

/**
 * The table of a universe log (the swap table of a tempering run: "Step T0 T1 ..." and integers; the event table of
 * prd and tad: "Step CPU Clock Event Correlated Coincident Replica" (prd.html) or "Step CPU N M Status Barrier Margin
 * t_lo delt_lo" (tad.html)). Each row is {column: value} with the CPU column dropped (wall-clock time, not compared);
 * numeric fields are numbers, the status word of tad is a string. Returns null when the log has no such header.
 */
const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const parseTable = (log) => {
  let keys = null;
  const rows = [];
  for (const line of log.split('\n')) {
    const w = line.trim().split(/\s+/);
    if (w[0] === 'Step' && w.length > 1 && (w[1] === 'CPU' || w.slice(1).every((x) => /^T\d+$/.test(x)))) {
      keys = w;
      continue;
    }
    if (!keys || w.length !== keys.length || !w.every((x) => x !== '')) continue;
    const row = {};
    keys.forEach((k, c) => { if (k !== 'CPU') row[k] = NUMBER.test(w[c]) ? Number(w[c]) : w[c]; });
    rows.push(row);
  }
  return keys ? rows : null;
};

const names = process.argv.slice(2);
const cases = readdirSync(CASES).filter((f) => f.endsWith('.in')).map((f) => basename(f, '.in')).filter((n) => !names.length || names.includes(n));
let failed = 0;
for (const name of cases) {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const dir = mkdtempSync(join(tmpdir(), `oracle-${name}-`));
  for (const f of header(text, 'oracle-inputs')) copyFileSync(join(CASES, f), join(dir, f));
  for (const f of header(text, 'oracle-potentials')) copyFileSync(join(POTENTIALS, f), join(dir, f));
  for (const f of header(text, 'oracle-examples')) {
    // a directory (a name ending in "/") is copied with its files under its own name
    if (f.endsWith('/')) {
      const sub = join(dir, basename(f.slice(0, -1)));
      mkdirSync(sub, { recursive: true });
      for (const g of readdirSync(join(EXAMPLES, f))) copyFileSync(join(EXAMPLES, f, g), join(sub, g));
    } else copyFileSync(join(EXAMPLES, f), join(dir, basename(f)));
  }
  const partitions = Number(header(text, 'oracle-partitions')[0] ?? 1);
  // a NEB case (its input runs the neb command) gets no dump: every replica would write it
  const isNeb = /^\s*neb(\/spin)?\s/m.test(text);
  writeFileSync(join(dir, 'in.case'), partitions > 1 ? (isNeb ? `${text}\n` : `${text}\n${partitionFinalDump(partitions)}\n`) : `${text}\n${FINAL_DUMP}\n`);
  try {
    // the partitions' ranks yield while they wait for each other: busy-polling ranks on a loaded machine made a 5-partition
    // hop1 NEB take 24 s instead of 0.6 s (Open MPI mpi_yield_when_idle; the results are the same)
    if (partitions > 1) execFileSync(MPIRUN, ['-np', String(partitions), LMP, '-partition', `${partitions}x1`, '-in', 'in.case', '-log', 'log.lammps', '-screen', 'none'], { cwd: dir, stdio: 'pipe', env: { ...process.env, OMPI_MCA_mpi_yield_when_idle: '1' } });
    else execFileSync(LMP, ['-in', 'in.case', '-log', 'log.lammps', '-screen', 'none'], { cwd: dir, stdio: 'pipe' });
  } catch (e) {
    const log = existsSync(join(dir, 'log.lammps')) ? readFileSync(join(dir, 'log.lammps'), 'utf8') : '';
    console.error(`✗ ${name}: LAMMPS failed\n${(log.match(/ERROR.*/) ?? [String(e)])[0]}`);
    failed++;
    continue;
  }
  const log = readFileSync(join(dir, 'log.lammps'), 'utf8');
  const files = {};
  // binary files (dump dcd, xtc and h5md) are kept as base64 with a "base64:" prefix
  for (const f of header(text, 'oracle-files')) files[f] = BINARY_FILE.test(f) ? `base64:${readFileSync(join(dir, f)).toString('base64')}` : readFileSync(join(dir, f), 'utf8');
  const version = (/LAMMPS \((.*?)\)/.exec(log) ?? [])[1] ?? 'unknown';
  if (partitions > 1 && /^\s*Step\s+MaxReplica(Force|Torque)\b/m.test(log)) {
    const { rows, climbing } = parseNeb(log);
    writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ lammps: version, partitions, neb: rows, climbing, thermo: [], atoms: [], files }, null, 0) + '\n');
    console.log(`✓ ${name}: ${rows.length} NEB rows, ${climbing.length} climbing line(s) on ${partitions} partitions (LAMMPS ${version})`);
    rmSync(dir, { recursive: true, force: true });
    continue;
  }
  if (partitions > 1) {
    const keywords = thermoKeywords(text);
    const perPartition = [];
    for (let k = 0; k < partitions; k++) {
      const plog = readFileSync(join(dir, `log.lammps.${k}`), 'utf8');
      const atoms = parseDump(readFileSync(join(dir, `oracle_final.${k}.dump`), 'utf8'));
      perPartition.push({ thermo: parseLog(plog, keywords), atoms });
    }
    // a tempering run keeps its swap table (fixture key tempering); a prd or tad run its event table (key table)
    const table = parseTable(log);
    writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ lammps: version, partitions, perPartition, thermo: [], atoms: [], files, ...(table ? (/^\s*Step\s+CPU\b/m.test(log) ? { table } : { tempering: table }) : {}) }, null, 0) + '\n');
    console.log(`✓ ${name}: ${perPartition.map((p) => `${p.thermo.length} thermo rows, ${p.atoms.length} atoms`).join(' | ')} on ${partitions} partitions (LAMMPS ${version})${table ? `, ${table.length} table rows` : ''}`);
    rmSync(dir, { recursive: true, force: true });
    continue;
  }
  const thermo = parseLog(log, thermoKeywords(text));
  const atoms = parseDump(readFileSync(join(dir, 'oracle_final.dump'), 'utf8'));
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ lammps: version, thermo, atoms, files }, null, 0) + '\n');
  console.log(`✓ ${name}: ${thermo.length} thermo rows, ${atoms.length} atoms (LAMMPS ${version})`);
  rmSync(dir, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
