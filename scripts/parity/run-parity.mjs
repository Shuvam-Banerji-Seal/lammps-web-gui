#!/usr/bin/env node
/**
 * Parity matrix fixtures: builds the rows of each matrix of tests/parityMatrices.ts (an oracle case up to one line,
 * that line with one word changed, then run 0), runs every row through a native LAMMPS binary and writes
 * tests/fixtures/parity/<matrix>.json with each row's native result: the first ERROR line (its source location
 * dropped), OK (exit 0 and the "Total wall time" line), or CRASH rc=N; then the lines a row prints that start with
 * VAL (a row that compares values), as "OK | VAL a b ...".
 *
 *   LMP=/path/to/lmp node scripts/parity/run-parity.mjs [matrix ...]
 *
 * A row whose style is in a package the binary lacks ("... is part of the X package which is not enabled", or
 * "Must enable X package") is run again on the build listed for that package in LMP_BUILDS, a comma list of
 * PACKAGE=/path/to/lmp (e.g. ML-PACE=/opt/lammps-pace/lmp,APIP=/opt/lammps-apip/lmp); the fixture names the build.
 * A row is marked skip (with the reason) when native LAMMPS crashes on it, fails a memory allocation (whose size comes
 * from its own memory growth), runs into undefined behaviour (listed per style in the matrix), or no build has its
 * package.
 * JOBS (default 4) rows run at once. Needs Node 24 (it imports the TypeScript matrix module).
 */
import { spawn } from 'node:child_process';
import { copyFileSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { MATRICES, oracleCases, parityFiles, parityInput } from '../../tests/parityMatrices.ts';

const ROOT = new URL('../..', import.meta.url).pathname;
const OUT = join(ROOT, 'tests/fixtures/parity');
const LMP = process.env.LMP ?? 'lmp';
const JOBS = Number(process.env.JOBS ?? 4);
const BUILDS = Object.fromEntries((process.env.LMP_BUILDS ?? '').split(',').filter(Boolean).map((b) => b.split('=')));

/** Runs one input in a fresh directory with its files; resolves to the native result and the printed version. */
const runNative = (lmp, input, files) => new Promise((resolve) => {
  const dir = mkdtempSync(join(tmpdir(), 'parity-'));
  for (const [name, f] of Object.entries(files)) copyFileSync(f.path, join(dir, name));
  writeFileSync(join(dir, 'in.parity'), input);
  const p = spawn(lmp, ['-in', 'in.parity', '-log', 'none', '-echo', 'none'], { cwd: dir, env: { ...process.env, OMP_NUM_THREADS: '1' } });
  let out = '';
  p.stdout.on('data', (d) => { out += d; });
  p.stderr.on('data', (d) => { out += d; });
  const timer = setTimeout(() => p.kill('SIGKILL'), 120000);
  p.on('close', (code, signal) => {
    clearTimeout(timer);
    rmSync(dir, { recursive: true, force: true });
    const version = (/LAMMPS \((.*?)\)/.exec(out) ?? [])[1];
    const err = out.split('\n').find((l) => /^\s*ERROR/.test(l));
    let result;
    if (err) result = err.replace(/ \(src\/[^)]*\)/, '').replace(/^\s*ERROR( on proc \d+)?: /, 'ERROR: ').trimEnd();
    else if (code === 0 && out.includes('Total wall time')) result = 'OK';
    else result = `CRASH rc=${code ?? 128 + ({ SIGSEGV: 11, SIGABRT: 6, SIGKILL: 9, SIGFPE: 8, SIGBUS: 7 }[signal] ?? 0)}`;
    // the values a row prints (print "VAL ..."), after the status: "OK | VAL a b"
    const vals = out.split('\n').filter((l) => l.startsWith('VAL ')).map((l) => l.trimEnd());
    if (vals.length) result += ` | ${vals.join(' ')}`;
    resolve({ result, version });
  });
});

const pool = async (items, n, fn) => {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: n }, async () => {
    while (next < items.length) { const i = next++; out[i] = await fn(items[i], i); }
  }));
  return out;
};

const cases = oracleCases(join(ROOT, 'tests/oracle'));
const text = Object.fromEntries(cases.map((c) => [c.name, c.text]));
mkdirSync(OUT, { recursive: true });
for (const name of process.argv.length > 2 ? process.argv.slice(2) : Object.keys(MATRICES)) {
  const def = MATRICES[name];
  if (!def) throw new Error(`unknown matrix ${name} (${Object.keys(MATRICES).join(', ')})`);
  let lammps = 'unknown';
  const rows = await pool(def.rows(cases, join(ROOT, 'tests/oracle')), JOBS, async (row) => {
    const input = parityInput(text[row.case], row);
    const files = parityFiles(ROOT, text[row.case], input);
    const first = await runNative(LMP, input, files);
    if (first.version) lammps = first.version;
    let out = { ...row, native: first.result };
    const pkg = /is part of the (\S+) package which is not enabled|Must enable (\S+) package/.exec(first.result);
    if (pkg) {
      const p = pkg[1] ?? pkg[2];
      if (BUILDS[p]) out = { ...row, native: (await runNative(BUILDS[p], input, files)).result, build: p };
      else out.skip = `needs the ${p} package, which no native build here has`;
    }
    if (out.native.startsWith('CRASH')) out.skip = `native LAMMPS crashes (${out.native})`;
    if (def.undefined?.[row.style]) out.skip = def.undefined[row.style];
    // a failed allocation names a size that comes from native's own memory growth (atom->nmax), not from the input
    if (out.native.startsWith('ERROR: Failed to allocate')) out.skip = `native LAMMPS fails a memory allocation (${out.native.slice(7)})`;
    return out;
  });
  writeFileSync(join(OUT, `${name}.json`), JSON.stringify({ matrix: name, about: def.about, lammps, rows }, null, 1) + '\n');
  const skipped = rows.filter((r) => r.skip).length;
  console.log(`✓ ${name}: ${rows.length} rows, ${rows.filter((r) => r.native.startsWith('ERROR')).length} native errors, ${skipped} skipped (LAMMPS ${lammps})`);
}
