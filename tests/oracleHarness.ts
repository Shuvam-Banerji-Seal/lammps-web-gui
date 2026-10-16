import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Oracle parity harness (the suites are tests/engineOracle*.test.ts): every input in tests/oracle/*.in was run through native
 * LAMMPS by scripts/oracle/run-oracle.mjs; the fixture holds its thermo rows,
 * final per-atom state and selected output files. The engine runs the same
 * input and must agree within the case's tolerances.
 *
 * Case directives (comment lines in the .in file):
 *   # oracle-inputs: files the input reads (data, potentials), next to the .in file
 *   # oracle-potentials: unmodified LAMMPS potential files from third_party/lammps/potentials
 *   # oracle-files: files it writes that are compared token by token
 *   # oracle-compare: thermo=all|first|first-last atoms=all|none rel=1e-8 abs=1e-10 skip=kw1,kw2 atomskip=c1,c2 printed=N
 *                     atomrel=R filerel=R (optional: the per-atom and written-file tolerances, relative, default
 *                     100*max(rel,1e-9) and 1e-6). printed=N (NEB tables only): the engine's table values are rounded
 *                     to N significant digits before the comparison, as the native table prints them (%12.8g: N=8)
 *   # oracle-partitions: N   the case runs on N replicas (partitions). A NEB case's fixture holds the NEB table
 *                            rows and the "Climbing replica" lines, compared with the same rel/abs. Any other
 *                            case's fixture holds each partition's thermo rows and final atoms (perPartition),
 *                            compared partition by partition with the same tolerances
 */


/** A case input: UTF-8 text when it decodes, else its bytes as latin1 (the app's rule, fileStore.ts readInputText). */
const readInputBytes = (path: string, latin1: boolean): string => {
  const buf = readFileSync(path);
  if (latin1) return buf.toString('latin1');
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buf);
  } catch {
    return buf.toString('latin1');
  }
};

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';
const PARTITION_FINAL_DUMP = 'write_dump all custom oracle_final.${oracle_p}.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';

interface Fixture {
  lammps: string;
  thermo: ThermoRow[];
  atoms: Record<string, number>[];
  files: Record<string, string>;
  /** partitioned (NEB) cases: the table rows and the climbing lines */
  partitions?: number;
  neb?: ThermoRow[];
  climbing?: { replica: number; row: number }[];
  /** partitioned (non-NEB) cases: the thermo rows and final atoms of each partition K = 0..N-1 */
  perPartition?: { thermo: ThermoRow[]; atoms: Record<string, number>[] }[];
  /** tempering cases: the swap table rows {Step, T0, T1, ...} (key tempering); prd and tad cases: the event table rows (CPU dropped, key table) */
  table?: Record<string, number | string>[];
  tempering?: Record<string, number | string>[];
}

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

const options = (text: string) => {
  const o: { thermo: string; atoms: string; rel: number; abs: number; atomrel?: number; filerel?: number; printed?: number; skip: Set<string>; atomskip: Set<string> } = {
    thermo: 'all', atoms: 'all', rel: 1e-8, abs: 1e-10, skip: new Set<string>(), atomskip: new Set<string>(),
  };
  for (const w of directive(text, 'oracle-compare')) {
    const [k, v] = w.split('=');
    if (k === 'thermo') o.thermo = v;
    else if (k === 'atoms') o.atoms = v;
    else if (k === 'rel') o.rel = Number(v);
    else if (k === 'abs') o.abs = Number(v);
    else if (k === 'atomrel') o.atomrel = Number(v);
    else if (k === 'filerel') o.filerel = Number(v);
    else if (k === 'printed') o.printed = Number(v);
    else if (k === 'skip') for (const s of v.split(',')) o.skip.add(s);
    else if (k === 'atomskip') for (const s of v.split(',')) o.atomskip.add(s);
  }
  return o;
};

const close = (a: number, b: number, rel: number, abs: number) =>
  (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

/** Token-wise comparison of two text files: numbers within tolerance, words equal. */
const sameText = (a: string, b: string, rel: number): string | null => {
  const la = a.trim().split('\n'), lb = b.trim().split('\n');
  if (la.length !== lb.length) return `line count ${la.length} vs ${lb.length}`;
  for (let i = 0; i < la.length; i++) {
    // brackets around a number (the basis vectors of fix phonon's log: "[2.5 0 0]") are not part of the number
    const ta = la[i].trim().split(/\s+/).map((w) => w.replace(/[[\]]/g, '')), tb = lb[i].trim().split(/\s+/).map((w) => w.replace(/[[\]]/g, ''));
    if (ta.length !== tb.length) return `line ${i + 1}: '${la[i]}' vs '${lb[i]}'`;
    for (let k = 0; k < ta.length; k++) {
      const x = Number(ta[k]), y = Number(tb[k]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        if (!close(x, y, rel, 1e-12)) return `line ${i + 1} token ${k + 1}: ${ta[k]} vs ${tb[k]}`;
      } else if (ta[k] !== tb[k]) return `line ${i + 1} token ${k + 1}: '${ta[k]}' vs '${tb[k]}'`;
    }
  }
  return null;
};

/**
 * The table of a run: the engine's session log lines of a tempering swap table ("Step T0 T1 ...", temper.cpp
 * print_status) or of prd / tad ("Step CPU Clock Event ..." of prd.cpp log_event, "Step CPU N M Status ..." of tad.cpp
 * log_event), and the rows after each header. The CPU column is dropped (wall-clock time); the same rows as the
 * universe log of native LAMMPS (run-oracle.mjs parseTable).
 */
const NUMBER = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
const tableRows = (events: EngineEvent[]): Record<string, number | string>[] => {
  let keys: string[] | null = null;
  const rows: Record<string, number | string>[] = [];
  for (const e of events) {
    if (e.kind !== 'log') continue;
    for (const line of e.text.split('\n')) {
      const w = line.trim().split(/\s+/);
      if (w[0] === 'Step' && w.length > 1 && (w[1] === 'CPU' || w.slice(1).every((x) => /^T\d+$/.test(x)))) { keys = w; continue; }
      if (!keys || w.length !== keys.length || !w.every((x) => x !== '')) continue;
      const row: Record<string, number | string> = {};
      keys.forEach((k, c) => { if (k !== 'CPU') row[k] = NUMBER.test(w[c]) ? Number(w[c]) : w[c]; });
      rows.push(row);
    }
  }
  return rows;
};
const allCases = existsSync(CASES) ? readdirSync(CASES).filter((f) => f.endsWith('.in')).map((f) => f.slice(0, -3)).sort() : [];

/**
 * The cases are split over ORACLE_SHARDS test files (tests/engineOracle*.test.ts) so that vitest runs them in parallel
 * workers; case i (in name order) belongs to shard i % ORACLE_SHARDS. Run one case with
 * `npx vitest run tests/engineOracle -t <case>` (the path prefix matches every shard file).
 */
export const ORACLE_SHARDS = 4;

export const oracleSuite = (shard: number): void => {
const cases = allCases.filter((_, i) => i % ORACLE_SHARDS === shard);

describe(`oracle parity with native LAMMPS (shard ${shard + 1}/${ORACLE_SHARDS})`, () => {
  for (const name of cases) {
    it(name, async () => {
      const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
      const fx: Fixture = JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8'));
      const o = options(text);
      const events: EngineEvent[] = [];
      const files = new Map<string, string>();
      const partitions = Number(directive(text, 'oracle-partitions')[0] ?? 1);
      const session = new Session({
        emit: (e) => events.push(e),
        writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
      }, undefined, 0, partitions);
      // partitioned (non-NEB) cases: the other replicas' thermo rows and dumps are passed on (Session.recordPartitionOutput)
      session.recordPartitionOutput = fx.perPartition !== undefined;
      // a gzipped input travels as one character per byte, as the notebook uploads it (System.readFile inflates it)
      // as the app reads uploads (fileStore.ts readInputText): UTF-8 text, else the bytes as latin1 (a binary restart file)
      for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readInputBytes(join(CASES, f), f.endsWith('.gz')));
      for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(POTENTIALS, f), 'utf8'));
      try {
        // as scripts/oracle/run-oracle.mjs: a partitioned non-NEB case writes one final dump per partition (oracle_p
        // is a world-style variable, 0..N-1); a NEB case writes none
        const partitionDump = `variable oracle_p world ${Array.from({ length: partitions }, (_, k) => k).join(' ')}\n${PARTITION_FINAL_DUMP}`;
        const tail = partitions > 1 ? (fx.perPartition ? `\n${partitionDump}` : '') : `\n${FINAL_DUMP}`;
        await session.execute(`${text}${tail}\n`);
      } catch (e) {
        const err = events.find((ev) => ev.kind === 'error');
        throw new Error(`engine failed: ${err && 'message' in err ? err.message : String(e)}`);
      }
      if (fx.perPartition) {
        // each partition: its thermo rows (the rows carry the partition they were printed by) and its final atoms
        expect(fx.perPartition.length, 'partitions').toBe(partitions);
        const keysOf = (rows: ThermoRow[]) => (rows.length ? Object.keys(rows[0]) : []);
        for (let k = 0; k < partitions; k++) {
          const want = fx.perPartition[k];
          const keys = keysOf(want.thermo);
          const rows = events
            .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo' && (e.partition ?? 0) === k)
            .map((e) => e.row)
            .filter((row) => keys.every((key) => key in row));
          expect(rows.length, `partition ${k} thermo row count`).toBe(want.thermo.length);
          let thermoWorst = 0;
          for (let r = 0; r < want.thermo.length; r++) {
            for (const [key, v] of Object.entries(want.thermo[r])) {
              if (o.skip.has(key)) continue;
              const g = rows[r][key];
              if (!close(g, v, o.rel, o.abs)) throw new Error(`${name}: partition ${k} thermo row ${r} (step ${want.thermo[r].step}) ${key}: engine ${g} vs LAMMPS ${v}`);
              if (Math.abs(v) > o.abs) thermoWorst = Math.max(thermoWorst, Math.abs(g - v) / Math.abs(v));
            }
          }
          // ORACLE_MEASURE=1 prints the measured errors for the case header ("# Measured with native LAMMPS: ...")
          if (process.env.ORACLE_MEASURE) console.log(`MEASURE ${name} partition ${k} thermo ${thermoWorst.toExponential(1)}`);
          if (o.atoms === 'all') {
            const dump = files.get(`oracle_final.${k}.dump`) ?? '';
            const lines = dump.trim().split('\n');
            const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
            const cols = lines[k0].split(/\s+/).slice(2);
            const mine = lines.slice(k0 + 1).map((l) => {
              const w = l.trim().split(/\s+/).map(Number);
              const a: Record<string, number> = {};
              cols.forEach((c, i) => { a[c] = w[i]; });
              return a;
            });
            expect(mine.length, `partition ${k} atom count`).toBe(want.atoms.length);
            let worst = 0;
            let where = '';
            for (let i = 0; i < mine.length; i++) {
              const a = mine[i], b = want.atoms[i];
              expect(a.id, `partition ${k} atom ${i} id`).toBe(b.id);
              expect(a.type, `partition ${k} atom ${i} type`).toBe(b.type);
              for (const c of ['xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']) {
                if (o.atomskip.has(c)) continue;
                const scale = Math.max(1, Math.abs(b[c]));
                const d = Math.abs(a[c] - b[c]) / scale;
                if (d > worst) { worst = d; where = `atom ${b.id} ${c}: engine ${a[c]} vs LAMMPS ${b[c]}`; }
              }
            }
            const atomTol = o.atomrel ?? Math.max(o.rel, 1e-9) * 100;
            if (worst > atomTol) throw new Error(`${name}: partition ${k} per-atom mismatch ${worst.toExponential(2)} (${where})`);
            if (process.env.ORACLE_MEASURE) console.log(`MEASURE ${name} partition ${k} atoms ${worst.toExponential(1)}`);
          }
        }
        const table = fx.table ?? fx.tempering;
        if (table !== undefined) expect(tableRows(events), 'table rows').toEqual(table);
        return;
      }
      if (partitions > 1) {
        // NEB table rows are the thermo rows that carry the RD and PE columns; the minimize rows before them do not
        const nebRows: ThermoRow[] = [];
        const climbing: { replica: number; row: number }[] = [];
        for (const e of events) {
          if (e.kind === 'thermo' && 'RD1' in e.row) nebRows.push(e.row);
          if (e.kind !== 'log') continue;
          const m = /^Climbing replica = (\d+)$/.exec(e.text);
          if (m) climbing.push({ replica: Number(m[1]), row: nebRows.length });
        }
        expect(nebRows.length, 'NEB table rows').toBe(fx.neb!.length);
        for (let r = 0; r < fx.neb!.length; r++) {
          for (const [k, v] of Object.entries(fx.neb![r])) {
            if (o.skip.has(k)) continue;
            const raw = nebRows[r][k];
            // the native table prints each value with 8 significant digits (print_status, "%12.8g"): compare what it prints
            const g = o.printed !== undefined && Number.isFinite(raw) ? Number(raw.toPrecision(o.printed)) : raw;
            if (!close(g, v, o.rel, o.abs)) throw new Error(`${name}: NEB row ${r} (step ${fx.neb![r].Step}) ${k}: engine ${g} vs LAMMPS ${v}`);
          }
        }
        expect(climbing, 'climbing lines').toEqual(fx.climbing ?? []);
        return;
      }
      // thermo
      // rows of an earlier thermo_style (another column set) are not part of this fixture
      const keys = fx.thermo.length ? Object.keys(fx.thermo[0]) : [];
      const rows = events
        .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
        .map((e) => e.row)
        .filter((row) => keys.every((k) => k in row));
      const pick = (r: ThermoRow[]) => (o.thermo === 'first' ? r.slice(0, 1) : o.thermo === 'first-last' ? [r[0], r[r.length - 1]] : r);
      const want = pick(fx.thermo), got = pick(rows);
      expect(got.length, 'thermo row count').toBe(want.length);
      for (let r = 0; r < want.length; r++) {
        for (const [k, v] of Object.entries(want[r])) {
          if (o.skip.has(k) || (o.thermo === 'first-last' && r === 1 && k === 'step')) continue;
          const g = got[r][k];
          if (!close(g, v, o.rel, o.abs)) {
            throw new Error(`${name}: thermo row ${r} (step ${want[r].step}) ${k}: engine ${g} vs LAMMPS ${v}`);
          }
        }
      }
      // final per-atom state
      if (o.atoms === 'all') {
        const dump = files.get('oracle_final.dump') ?? '';
        const lines = dump.trim().split('\n');
        const k0 = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
        const cols = lines[k0].split(/\s+/).slice(2);
        const mine = lines.slice(k0 + 1).map((l) => {
          const w = l.trim().split(/\s+/).map(Number);
          const a: Record<string, number> = {};
          cols.forEach((c, i) => { a[c] = w[i]; });
          return a;
        });
        expect(mine.length, 'atom count').toBe(fx.atoms.length);
        let worst = 0;
        let where = '';
        for (let i = 0; i < mine.length; i++) {
          const a = mine[i], b = fx.atoms[i];
          expect(a.id).toBe(b.id);
          expect(a.type).toBe(b.type);
          for (const c of ['xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz']) {
            if (o.atomskip.has(c)) continue;
            const scale = Math.max(1, Math.abs(b[c]));
            const d = Math.abs(a[c] - b[c]) / scale;
            if (d > worst) { worst = d; where = `atom ${b.id} ${c}: engine ${a[c]} vs LAMMPS ${b[c]}`; }
          }
        }
        const atomTol = o.atomrel ?? Math.max(o.rel, 1e-9) * 100;
        if (worst > atomTol) throw new Error(`${name}: per-atom mismatch ${worst.toExponential(2)} (${where})`);
      }
      // written files
      for (const [f, ref] of Object.entries(fx.files)) {
        const mine = files.get(f);
        expect(mine, `file ${f}`).toBeDefined();
        // the data-file title line names the writing program; a "DATE: YYYY-MM-DD" header field (fix ttm grid files,
        // potential files written by the engine) is the day of the run, so it is not compared
        const strip = (t: string) => (f.endsWith('.data') ? t.split('\n').slice(1).join('\n') : t)
          .replace(/\bDATE: \d{4}-\d{2}-\d{2}\b/g, 'DATE: <date>');
        const diff = sameText(strip(mine!), strip(ref), o.filerel ?? 1e-6);
        if (diff) throw new Error(`${name}: file ${f} differs at ${diff}`);
      }
    }, 120_000);
  }
});
};
