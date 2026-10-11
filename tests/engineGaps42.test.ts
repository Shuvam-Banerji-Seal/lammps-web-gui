import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { selectPppmAuto } from '../src/engine/force/kspace/pppm';

/*
 * Parity gaps of the 42nd wave (native LAMMPS, black box):
 *  - a temperature compute used before the first run (velocity create) excludes the dof of fix shake and
 *    fix rigid (the oracle cases w42gap_shake_velocity and w42gap_rigid_velocity hold the full comparison);
 *  - fix ave/histo, ave/time, ave/chunk and ave/correlate do not write or sample a step twice when a run
 *    starts on the step the previous run ended on, and their windows carry over the run boundary;
 *  - the automatic PPPM G-ewald of non-cubic boxes uses the rms over the edges of the k-space error.
 */

const TESTS = __dirname;
const LJ_FCC = `
units lj
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.44 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
variable vx atom vx
variable vt equal temp
`;

interface RunResult {
  thermo: ThermoRow[];
  files: Map<string, string>;
  logs: string[];
}

const execute = async (script: string, inputs: Record<string, string> = {}): Promise<RunResult> => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t),
  });
  for (const [name, text] of Object.entries(inputs)) session.addFile(name, text);
  await session.execute(script);
  const errors = events.filter((e): e is Extract<EngineEvent, { kind: 'error' }> => e.kind === 'error');
  if (errors.length) throw new Error(`engine error: ${errors[0].message}`);
  return {
    thermo: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row),
    files,
    logs: events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text),
  };
};

const close = (a: number, b: number, rel: number) => Math.abs(a - b) <= 1e-12 + rel * Math.max(Math.abs(a), Math.abs(b));

/** Data lines of a histogram or time-average file: "step" headers and rows as numbers. */
const blocks = (text: string): string[][] => text.trim().split('\n').filter((l) => !l.startsWith('#')).map((l) => l.trim().split(/\s+/));

describe('velocity create sees the constraint dof of fix shake and fix rigid', () => {
  it('fix shake: temperature 300 after velocity create and run 0 (native: 300, dof 3 N - 3 - 3 molecules)', async () => {
    const data = readFileSync(join(TESTS, 'oracle', 'w42gap_spce64.data'), 'utf8');
    const r = await execute(`
units real
atom_style full
read_data w42gap_spce64.data
pair_style lj/cut/coul/long 6.0
pair_coeff * * 0.0 1.0
pair_coeff 1 1 0.15535 3.166
pair_modify table 0
kspace_style pppm 1.0e-4
bond_style harmonic
angle_style harmonic
bond_coeff 1 1000.00 1.000
angle_coeff 1 100.0 109.47
special_bonds lj/coul 0.0 0.0 0.0
neighbor 2.0 bin
timestep 2.0
fix 1 all shake 0.0001 20 0 b 1 a 1
velocity all create 300.0 6244325
thermo_style custom step temp ke
thermo_modify format float %.15g
run 0
`, { 'w42gap_spce64.data': data });
    expect(r.thermo[0].temp).toBeCloseTo(300, 9);
    // native kinetic energy at step 0: 113.568862905 kcal/mol
    expect(r.thermo[0].ke).toBeCloseTo(113.568862905, 8);
  });

  it('fix rigid molecule: temperature 308.381749285181 after velocity create and run 0 (native)', async () => {
    const data = readFileSync(join(TESTS, 'oracle', 'w42gap_spce64.data'), 'utf8');
    const r = await execute(`
units real
atom_style full
read_data w42gap_spce64.data
pair_style lj/cut/coul/long 6.0
pair_coeff * * 0.0 1.0
pair_coeff 1 1 0.15535 3.166
pair_modify table 0
kspace_style pppm 1.0e-4
bond_style harmonic
angle_style harmonic
bond_coeff 1 1000.00 1.000
angle_coeff 1 100.0 109.47
special_bonds lj/coul 0.0 0.0 0.0
neighbor 2.0 bin
timestep 2.0
fix 1 all rigid molecule
velocity all create 300.0 6244325
thermo_style custom step temp
thermo_modify format float %.15g
run 0
`, { 'w42gap_spce64.data': data });
    expect(close(r.thermo[0].temp, 308.381749285181, 1e-9)).toBe(true);
  });
});

describe('a run that starts where the previous one ended', () => {
  it('fix ave/histo writes the TimeStep 100 block once; the file matches the stored native output', async () => {
    const r = await execute(`${LJ_FCC}
fix h all ave/histo 1 1 100 -3.0 3.0 12 v_vx file h.txt mode vector
thermo 100
run 100
run 100
`);
    const mine = blocks(r.files.get('h.txt') ?? '');
    const native = blocks(readFileSync(join(TESTS, 'fixtures', 'w42gap', 'avehisto_two_runs_native.txt'), 'utf8'));
    expect(mine.filter((l) => l.length === 6 && l[1] === '12').map((l) => l[0])).toEqual(['0', '100', '200']);
    expect(mine.length).toBe(native.length);
    for (let i = 0; i < native.length; i++) {
      expect(mine[i].length, `line ${i}`).toBe(native[i].length);
      for (let k = 0; k < native[i].length; k++) {
        const x = Number(mine[i][k]), y = Number(native[i][k]);
        if (Number.isFinite(y) && native[i][k] !== '0') {
          expect(close(x, y, 1e-8), `line ${i} token ${k}: ${mine[i][k]} vs ${native[i][k]}`).toBe(true);
        } else expect(mine[i][k], `line ${i} token ${k}`).toBe(native[i][k]);
      }
    }
  });

  it('fix ave/time with Nrepeat 2 keeps the window across the boundary at step 15 (native: 0.673768 at 20)', async () => {
    const r = await execute(`${LJ_FCC}
fix t all ave/time 5 2 20 v_vt file t.txt
run 15
run 15
run 10
`);
    const rows = blocks(r.files.get('t.txt') ?? '');
    expect(rows.map((l) => l[0])).toEqual(['20', '40']);
    expect(Math.abs(Number(rows[0][1]) - 0.673768)).toBeLessThan(5e-7);
    expect(Math.abs(Number(rows[1][1]) - 0.698947)).toBeLessThan(5e-7);
  });

  it('fix ave/chunk keeps the window across the boundary at step 15 (native: step-20 block, chunk 1 count 17)', async () => {
    const r = await execute(`${LJ_FCC}
compute cc all chunk/atom bin/1d x lower 0.25
fix c all ave/chunk 5 2 20 cc vx file c.txt
run 15
run 25
`);
    const lines = (r.files.get('c.txt') ?? '').trim().split('\n');
    const headers = lines.filter((l) => /^\d+ 16 256$/.test(l));
    expect(headers.map((l) => l.split(' ')[0])).toEqual(['20', '40']);
    const first = lines.find((l) => /^1 0\.20995 /.test(l))!;
    // the first step-20 chunk row (the second block starts at the line after the 40 header)
    const idx = lines.indexOf('20 16 256');
    const row = lines[idx + 1].trim().split(/\s+/);
    expect(row[0]).toBe('1');
    expect(Number(row[2])).toBe(17);
    expect(Math.abs(Number(row[3]) - 0.447377)).toBeLessThan(5e-7);
    expect(first).toBeDefined();
  });

  it('fix ave/correlate keeps the lag counts across the boundary (native: lag 0 count 5 at step 40)', async () => {
    const r = await execute(`${LJ_FCC}
fix co all ave/correlate 5 4 20 v_vt v_vt file co.txt
run 10
run 40
`);
    const lines = (r.files.get('co.txt') ?? '').trim().split('\n');
    const i = lines.indexOf('40 4');
    expect(i).toBeGreaterThan(0);
    const lag0 = lines[i + 1].trim().split(/\s+/);
    expect(Number(lag0[2])).toBe(5);
    expect(Math.abs(Number(lag0[3]) - 0.475363)).toBeLessThan(5e-7);
  });
});

describe('automatic PPPM G-ewald on orthorhombic boxes (native black box, printed to 8 digits)', () => {
  // lj units, two atoms per unit cell, reduced density rho: cell edge (2 / rho)^(1/3); charges +-1
  const cases: { name: string; cells: [number, number, number]; rc: number; acc: number; order?: number; density?: number; g: number; mesh: [number, number, number] }[] = [
    { name: 'cubic 4', cells: [4, 4, 4], rc: 2.5, acc: 1e-5, g: 1.3833153, mesh: [32, 32, 32] },
    { name: 'cubic 5', cells: [5, 5, 5], rc: 3.0, acc: 1e-4, g: 1.0365648, mesh: [20, 20, 20] },
    { name: 'cubic 8, acc 1e-3', cells: [8, 8, 8], rc: 4.0, acc: 1e-3, g: 0.68872316, mesh: [15, 15, 15] },
    { name: 'cubic 16 at lattice scale 0.5, acc 1e-6', cells: [16, 16, 16], rc: 4.0, acc: 1e-6, density: 0.5, g: 0.91990882, mesh: [144, 144, 144] },
    { name: 'cubic 8, order 4', cells: [8, 8, 8], rc: 3.0, acc: 1e-4, order: 4, g: 1.0294684, mesh: [45, 45, 45] },
    { name: 'box 6x8x10', cells: [6, 8, 10], rc: 2.5, acc: 1e-4, g: 1.2504352, mesh: [32, 40, 45] },
    { name: 'box 12x6x6', cells: [12, 6, 6], rc: 3.5, acc: 1e-5, g: 0.98484711, mesh: [60, 36, 36] },
    { name: 'box 10x12x14', cells: [10, 12, 14], rc: 4.5, acc: 1e-3, g: 0.60009333, mesh: [16, 18, 24] },
  ];
  for (const c of cases) {
    it(`${c.name}: G-ewald and mesh as native`, () => {
      const S = Math.cbrt(2 / (c.density ?? 1));
      const natoms = 2 * c.cells[0] * c.cells[1] * c.cells[2];
      const r = selectPppmAuto({
        lx: c.cells[0] * S, ly: c.cells[1] * S, lz: c.cells[2] * S, natoms, q2: natoms, rc: c.rc,
        accE: c.acc, order: c.order ?? 5, gUser: 0, meshUser: null,
      });
      expect(Math.abs(r.g - c.g)).toBeLessThan(5e-8);
      expect(r.mesh).toEqual(c.mesh);
    });
  }
});
