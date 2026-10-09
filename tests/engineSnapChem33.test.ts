import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * SNAP chemflag 1 (explicit multi-element bispectrum): tighter-than-generic
 * checks against the native LAMMPS fixtures for tests/oracle/w33snapchem_*.
 * The per-atom bispectrum is compared at 1e-8 (relative) and pe / forces at
 * 1e-7, as required by the chemflag acceptance test.
 */

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
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

const run = async (name: string, append = '') => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  await session.execute(`${text}\n${append}\n`);
  return { files, events };
};

const parseDump = (text: string) => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  return lines.slice(k + 1).map((l) => {
    const w = l.trim().split(/\s+/).map(Number);
    const o: Record<string, number> = {};
    cols.forEach((c, i) => { o[c] = w[i]; });
    return o;
  });
};

const close = (a: number, b: number, rel: number, abs: number) =>
  (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) <= abs + rel * Math.max(Math.abs(a), Math.abs(b));

describe('SNAP chemflag 1 (w33)', () => {
  it('compute sna/atom chem matches the native per-atom bispectrum to 1e-8', async () => {
    const fx: Fixture = JSON.parse(readFileSync(join(FIX, 'w33snapchem_sna.json'), 'utf8'));
    const { files } = await run('w33snapchem_sna');
    const got = parseDump(files.get('w33snapchem_sna.dump')!);
    const want = parseDump(fx.files['w33snapchem_sna.dump']);
    expect(got.length).toBe(want.length);
    let worst = 0, where = '';
    for (let i = 0; i < want.length; i++) {
      expect(got[i].id).toBe(want[i].id);
      for (let c = 1; c <= 240; c++) {
        const key = `c_b[${c}]`;
        // relative comparison for the bispectrum (magnitudes vary over orders)
        const d = Math.abs(got[i][key] - want[i][key]) / Math.max(1e-30, Math.abs(want[i][key]));
        if (d > worst) { worst = d; where = `atom ${want[i].id} col ${c}: engine ${got[i][key]} vs LAMMPS ${want[i][key]}`; }
      }
    }
    if (worst > 1e-8) throw new Error(`w33snapchem_sna: worst relative ${worst.toExponential(2)} (${where})`);
  }, 120_000);

  it('pair_style snap chemflag 1 matches native pe and forces to 1e-7', async () => {
    const fx: Fixture = JSON.parse(readFileSync(join(FIX, 'w33snapchem_pair.json'), 'utf8'));
    const { files, events } = await run('w33snapchem_pair', FINAL_DUMP);
    const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
    const pe = rows[rows.length - 1].pe;
    if (!close(pe, fx.thermo[0].pe, 1e-7, 1e-10)) throw new Error(`w33snapchem_pair: pe engine ${pe} vs LAMMPS ${fx.thermo[0].pe}`);
    const got = parseDump(files.get('oracle_final.dump')!);
    expect(got.length).toBe(fx.atoms.length);
    let worst = 0, where = '';
    for (let i = 0; i < fx.atoms.length; i++) {
      const b = fx.atoms[i];
      expect(got[i].id).toBe(b.id);
      expect(got[i].type).toBe(b.type);
      for (const c of ['fx', 'fy', 'fz']) {
        const d = Math.abs(got[i][c] - b[c]) / Math.max(1, Math.abs(b[c]));
        if (d > worst) { worst = d; where = `atom ${b.id} ${c}: engine ${got[i][c]} vs LAMMPS ${b[c]}`; }
      }
    }
    if (worst > 1e-7) throw new Error(`w33snapchem_pair: worst force rel ${worst.toExponential(2)} (${where})`);
  }, 120_000);
});
