import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * Oracle parity: every input in tests/oracle/*.in was run through native
 * LAMMPS by scripts/oracle/run-oracle.mjs; the fixture holds its thermo rows,
 * final per-atom state and selected output files. The engine runs the same
 * input and must agree within the case's tolerances.
 *
 * Case directives (comment lines in the .in file):
 *   # oracle-inputs: files the input reads (data, potentials), next to the .in file
 *   # oracle-potentials: unmodified LAMMPS potential files from third_party/lammps/potentials
 *   # oracle-files: files it writes that are compared token by token
 *   # oracle-compare: thermo=all|first|first-last atoms=all|none rel=1e-8 abs=1e-10 skip=kw1,kw2 atomskip=c1,c2
 *   # oracle-partitions: N   the case runs on N replicas (partitions); its fixture holds the NEB table
 *                            rows and the "Climbing replica" lines, compared with the same rel/abs
 */

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');
const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';

interface Fixture {
  lammps: string;
  thermo: ThermoRow[];
  atoms: Record<string, number>[];
  files: Record<string, string>;
  /** partitioned (NEB) cases: the table rows and the climbing lines */
  partitions?: number;
  neb?: ThermoRow[];
  climbing?: { replica: number; row: number }[];
}

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

const options = (text: string) => {
  const o = { thermo: 'all', atoms: 'all', rel: 1e-8, abs: 1e-10, skip: new Set<string>(), atomskip: new Set<string>() };
  for (const w of directive(text, 'oracle-compare')) {
    const [k, v] = w.split('=');
    if (k === 'thermo') o.thermo = v;
    else if (k === 'atoms') o.atoms = v;
    else if (k === 'rel') o.rel = Number(v);
    else if (k === 'abs') o.abs = Number(v);
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

const cases = existsSync(CASES) ? readdirSync(CASES).filter((f) => f.endsWith('.in')).map((f) => f.slice(0, -3)) : [];

describe('oracle parity with native LAMMPS', () => {
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
      // a gzipped input travels as one character per byte, as the notebook uploads it (System.readFile inflates it)
      for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), f.endsWith('.gz') ? 'latin1' : 'utf8'));
      for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(POTENTIALS, f), 'utf8'));
      try {
        await session.execute(partitions > 1 ? `${text}\n` : `${text}\n${FINAL_DUMP}\n`);
      } catch (e) {
        const err = events.find((ev) => ev.kind === 'error');
        throw new Error(`engine failed: ${err && 'message' in err ? err.message : String(e)}`);
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
            const g = nebRows[r][k];
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
        if (worst > Math.max(o.rel, 1e-9) * 100) throw new Error(`${name}: per-atom mismatch ${worst.toExponential(2)} (${where})`);
      }
      // written files
      for (const [f, ref] of Object.entries(fx.files)) {
        const mine = files.get(f);
        expect(mine, `file ${f}`).toBeDefined();
        // the data-file title line names the writing program; a "DATE: YYYY-MM-DD" header field (fix ttm grid files,
        // potential files written by the engine) is the day of the run, so it is not compared
        const strip = (t: string) => (f.endsWith('.data') ? t.split('\n').slice(1).join('\n') : t)
          .replace(/\bDATE: \d{4}-\d{2}-\d{2}\b/g, 'DATE: <date>');
        const diff = sameText(strip(mine!), strip(ref), 1e-6);
        if (diff) throw new Error(`${name}: file ${f} differs at ${diff}`);
      }
    }, 120_000);
  }
});
