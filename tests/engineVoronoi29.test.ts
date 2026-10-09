import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute voronoi/atom in 2d and in restricted triclinic boxes
 * (docs.lammps.org/compute_voronoi_atom.html):
 *   - in 2d the Voro++ cell is a column: "The cross-sectional area of each
 *     Voronoi cell can be obtained by dividing its volume by the z extent of
 *     the simulation box", and "two exterior faces at the top and bottom of the
 *     simulation box" are part of the face count;
 *   - for a triclinic box the periodic images of the atoms sit on the tilted
 *     edge vectors A, B, C (Howto_triclinic.html).
 * The native LAMMPS per-atom dumps are the oracle fixtures (rel 1e-9).
 */

const CASES = join(__dirname, 'oracle');
const FIX = join(__dirname, 'fixtures', 'oracle');
const FINAL_DUMP = 'write_dump all custom oracle_final.dump id type xu yu zu vx vy vz fx fy fz modify format float %.17g sort id';

const directive = (text: string, key: string): string[] => {
  const m = new RegExp(`^#\\s*${key}:\\s*(.*)$`, 'm').exec(text);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};

interface Dump { cols: string[]; rows: number[][] }

const parseDump = (text: string): Dump => {
  const lines = text.trim().split('\n');
  const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
  const cols = lines[k].split(/\s+/).slice(2);
  const rows = lines.slice(k + 1).filter((l) => l.trim()).map((l) => l.trim().split(/\s+/).map(Number));
  return { cols, rows };
};

const runCase = async (name: string): Promise<{ files: Map<string, string>; events: EngineEvent[] }> => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  await session.execute(`${text}\n${FINAL_DUMP}\n`);
  return { files, events };
};

const cases = existsSync(CASES)
  ? ['w29voro_2d_sq', 'w29voro_2d_kw', 'w29voro_2d_radius', 'w29voro_tri', 'w29voro_tri_kw']
  : [];

describe('compute voronoi/atom — 2d and triclinic parity at 1e-9', () => {
  for (const name of cases) {
    it(`${name}: dump matches native at rel 1e-9`, async () => {
      const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
      const fx = JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')) as { files: Record<string, string> };
      const { files } = await runCase(name);
      for (const f of directive(text, 'oracle-files')) {
        const mine = files.get(f);
        expect(mine, `file ${f}`).toBeDefined();
        const a = parseDump(mine!);
        const b = parseDump(fx.files[f]);
        expect(a.cols).toEqual(b.cols);
        expect(a.rows.length).toBe(b.rows.length);
        let worst = 0;
        let where = '';
        for (let r = 0; r < a.rows.length; r++) {
          for (let c = 0; c < a.cols.length; c++) {
            const x = a.rows[r][c], y = b.rows[r][c];
            if (['id', 'type'].includes(a.cols[c])) {
              expect(x).toBe(y);
              continue;
            }
            const d = Math.abs(x - y) / Math.max(1, Math.abs(y));
            if (d > worst) { worst = d; where = `${f} row ${r + 1} ${a.cols[c]}: engine ${x} vs native ${y}`; }
          }
        }
        expect(worst, where).toBeLessThan(1e-9);
      }
    }, 120_000);
  }

  it('2d column: volume is the cross-sectional area times lz and every cell has six faces', async () => {
    const { files } = await runCase('w29voro_2d_sq');
    const d = parseDump(files.get('w29voro_2d_sq.dump')!);
    const iv = d.cols.indexOf('c_v[1]'), ifc = d.cols.indexOf('c_v[2]');
    expect(d.rows.length).toBe(9);
    for (const r of d.rows) {
      // a 3x3 square lattice of spacing 1 in a box with lz = 2: area 1 * 2
      expect(r[iv]).toBeCloseTo(2, 9);
      // four in-plane faces plus the two periodic z faces
      expect(r[ifc]).toBe(6);
    }
  });

  it('triclinic: the same simple-cubic density gives 8 faces per atom in the sheared box', async () => {
    const { files } = await runCase('w29voro_tri');
    const d = parseDump(files.get('w29voro_tri.dump')!);
    const iv = d.cols.indexOf('c_v[1]'), ifc = d.cols.indexOf('c_v[2]');
    expect(d.rows.length).toBe(27);
    for (const r of d.rows) {
      expect(r[iv]).toBeCloseTo(1, 9);
      expect(r[ifc]).toBe(8);
    }
  });

  it('a general (rotated) triclinic box is rejected by name', async () => {
    const session = new Session({ emit: () => {} });
    await expect(session.execute(`
      lattice custom 1.0 a1 -0.5 0.5 0.5 a2 0.5 -0.5 0.5 a3 0.5 0.5 -0.5 basis 0.0 0.0 0.0 triclinic/general
      create_box 1 NULL 0 1 0 1 0 1
      create_atoms 1 box
      mass * 1.0
      pair_style lj/cut 0.5
      pair_coeff * * 0.0 1.0
      compute v all voronoi/atom
      compute s all reduce sum c_v[1]
      thermo_style custom step c_s
      run 0
    `)).rejects.toThrow(/general triclinic/);
  });
});
