import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * compute voronoi/atom at non-periodic boundaries
 * (docs.lammps.org/compute_voronoi_atom.html): a non-periodic box face clips
 * the tessellation at the face itself, and the resulting exterior face is part
 * of the face count and edge histogram but only joins the surface for the
 * built-in all group. The native LAMMPS per-atom dumps are the oracle fixtures
 * (rel 1e-9). See src/engine/compute/voronoi.ts for the measured wall geometry.
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

const runCase = async (name: string): Promise<Map<string, string>> => {
  const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const f of directive(text, 'oracle-inputs')) session.addFile(f, readFileSync(join(CASES, f), 'utf8'));
  await session.execute(`${text}\n${FINAL_DUMP}\n`);
  return files;
};

const cases = existsSync(CASES)
  ? ['w31voronp_3d_ff', 'w31voronp_3d_wall', 'w31voronp_2d_ffp', 'w31voronp_surface', 'w31voronp_radius', 'w31voronp_single', 'w31voronp_sm']
  : [];

describe('compute voronoi/atom — non-periodic boundaries at 1e-9', () => {
  for (const name of cases) {
    it(`${name}: dump matches native at rel 1e-9`, async () => {
      const text = readFileSync(join(CASES, `${name}.in`), 'utf8');
      const fx = JSON.parse(readFileSync(join(FIX, `${name}.json`), 'utf8')) as { files: Record<string, string> };
      const files = await runCase(name);
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

  it('single atom in a 2x3x4 f f f box: the cell is the box, 6 faces, surface all is the box surface', async () => {
    const files = await runCase('w31voronp_single');
    const d = parseDump(files.get('w31voronp_single.dump')!);
    const iv = d.cols.indexOf('c_v[1]'), ifc = d.cols.indexOf('c_v[2]'), isf = d.cols.indexOf('c_v[3]');
    expect(d.rows.length).toBe(1);
    expect(d.rows[0][iv]).toBeCloseTo(24, 9);
    expect(d.rows[0][ifc]).toBe(6);
    expect(d.rows[0][isf]).toBeCloseTo(52, 9);
  });

  it('cluster in a 10^3 f f f box: the far corner cell reaches the three walls (volume 512)', async () => {
    const files = await runCase('w31voronp_3d_wall');
    const d = parseDump(files.get('w31voronp_3d_wall.dump')!);
    const ix = d.cols.indexOf('x'), iy = d.cols.indexOf('y'), iz = d.cols.indexOf('z'), iv = d.cols.indexOf('c_v[1]');
    const corner = d.rows.find((r) => r[ix] > 2 && r[iy] > 2 && r[iz] > 2)!;
    expect(corner[iv]).toBeCloseTo(512, 9);
  });

  it('a triclinic non-periodic face is rejected by name (native wall placement not reproduced)', async () => {
    const session = new Session({ emit: () => {} });
    await expect(session.execute(`
      boundary f p p
      lattice sc 1.0 origin 0.5 0.5 0.5
      region box block 0 6 0 3 0 3 units box
      create_box 1 box
      create_atoms 1 box
      mass 1 1.0
      change_box all triclinic
      change_box all xy final 0.5 remap units box
      pair_style lj/cut 2.5
      pair_coeff 1 1 0.0 1.0
      compute v all voronoi/atom
      compute s all reduce sum c_v[1]
      thermo_style custom step c_s
      run 0
    `)).rejects.toThrow(/non-periodic boundaries with a triclinic box/);
  });

  it('2d f f p: interior cell keeps its area, a wall-corner cell fills the x/y margin, all faces have four edges', async () => {
    const text = readFileSync(join(CASES, 'w31voronp_2d_ffp.in'), 'utf8');
    const events: EngineEvent[] = [];
    const files = new Map<string, string>();
    const session = new Session({
      emit: (e) => events.push(e),
      writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
    });
    await session.execute(`${text}\n${FINAL_DUMP}\n`);
    const d = parseDump(files.get('w31voronp_2d_ffp.dump')!);
    const ix = d.cols.indexOf('x'), iy = d.cols.indexOf('y'), iv = d.cols.indexOf('c_v[1]'), ifc = d.cols.indexOf('c_v[2]');
    expect(d.rows.length).toBe(9);
    // centre atom (1.5, 1.5): 1x1 in-plane area times lz = 1
    const centre = d.rows.find((r) => Math.abs(r[ix] - 1.5) < 1e-9 && Math.abs(r[iy] - 1.5) < 1e-9)!;
    expect(centre[iv]).toBeCloseTo(1, 9);
    expect(centre[ifc]).toBe(6);
    // far corner atom (2.5, 2.5): x and y run to the boxes at 5 -> 3x3
    const far = d.rows.find((r) => r[ix] > 2 && r[iy] > 2)!;
    expect(far[iv]).toBeCloseTo(9, 9);
    // every face is a quadrilateral: histogram only has entry 4 populated
    const row = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row).pop()!;
    expect(row['c_v[4]']).toBe(54);
    expect(row['c_v[1]'] + row['c_v[2]'] + row['c_v[3]'] + row['c_v[5]'] + row['c_v[6]'] + row['c_v[7]'] + row['c_v[8]'] + row['c_v[9]']).toBe(0);
  });
});
