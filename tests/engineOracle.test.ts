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
 *   # oracle-compare: thermo=all|first|first-last atoms=all|none rel=1e-8 abs=1e-10 skip=kw1,kw2
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
}

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

const options = (text: string) => {
  const o = { thermo: 'all', atoms: 'all', rel: 1e-8, abs: 1e-10, skip: new Set<string>() };
  for (const w of directive(text, 'oracle-compare')) {
    const [k, v] = w.split('=');
    if (k === 'thermo') o.thermo = v;
    else if (k === 'atoms') o.atoms = v;
    else if (k === 'rel') o.rel = Number(v);
    else if (k === 'abs') o.abs = Number(v);
    else if (k === 'skip') for (const s of v.split(',')) o.skip.add(s);
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
    const ta = la[i].trim().split(/\s+/), tb = lb[i].trim().split(/\s+/);
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
      const session = new Session({
        emit: (e) => events.push(e),
        writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
      });
      for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
      for (const f of directive(text, 'oracle-potentials')) session.addFile(f, readFileSync(join(POTENTIALS, f), 'utf8'));
      try {
        await session.execute(`${text}\n${FINAL_DUMP}\n`);
      } catch (e) {
        const err = events.find((ev) => ev.kind === 'error');
        throw new Error(`engine failed: ${err && 'message' in err ? err.message : String(e)}`);
      }
      // thermo
      const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
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
        // the data-file title line names the writing program
        const strip = (t: string) => (f.endsWith('.data') ? t.split('\n').slice(1).join('\n') : t);
        const diff = sameText(strip(mine!), strip(ref), 1e-6);
        if (diff) throw new Error(`${name}: file ${f} differs at ${diff}`);
      }
    }, 120_000);
  }
});
