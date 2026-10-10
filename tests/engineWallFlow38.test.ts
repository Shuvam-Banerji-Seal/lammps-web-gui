import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { fluxNormal } from '../src/engine/fix/wall_flow';
import { RanMars } from '../src/engine/rng';

/*
 * fix wall/flow (docs.lammps.org/fix_wall_flow.html): parsing, the native
 * error texts, the exact draws measured with native LAMMPS (black box) and a
 * statistical check of the generated velocities over many crossings. Oracle
 * parity for whole runs is in tests/oracle/w38wflow_*.in.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  const rows = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, rows, error };
};

const BASE = `
units           lj
atom_style      atomic
boundary        p p p
region          box block 0 10 0 10 0 10
create_box      1 box
create_atoms    1 single 2.99 5 5
mass            1 1.0
velocity        all set 3.0 0.2 -0.1 sum no units box
pair_style      zero 2.5
pair_coeff      * *
fix             1 all nve
`;

describe('fix wall/flow: parsing and refusals', () => {
  it('accepts every documented form (axis, keyword units lattice|box, default lattice)', async () => {
    for (const tail of [
      'x 0.5 0.4 123 2 3.0 7.0',
      'x 0.5 0.4 123 2 3.0 7.0 units lattice',
      'y 0.5 2.0 99 1 4.0 units box',
      'z 1.0 1.0 7 3 1.5 2.5 8.0 units box',
    ]) {
      const { error } = await runScript(`${BASE}\nfix 2 all wall/flow ${tail}\nrun 1\n`);
      expect(error, tail).toBeNull();
    }
  });

  it('refuses bad input with the native error texts', async () => {
    const cases: [string, RegExp][] = [
      ['w 0.5 0.4 123 1 3.0', /axis must by x or y or z, but w specified/],
      ['x -0.5 0.4 123 1 3.0', /negative direction is not supported yet/],
      ['x 0.0 0.4 123 1 3.0', /velocity cannot be 0/],
      ['x 0.5 0.4 0 1 3.0', /random seed must be positive integer/],
      ['x 0.5 0.4 123 0 3.0', /wall count must be positive integer/],
      ['x 0.5 0.4 123 2 3.0', /wall count/],
      ['x 0.5 0.4 123 1 3.0 units foo', /units command/],
      ['x 0.5 0.4 123 2 7.0 3.0', /wall ordering or some walls are outside/],
      ['x 0.5 0.4 123 1 0.0', /wall ordering or some walls are outside/],
      ['x 0.5 0.4 123 1 10.0', /wall ordering or some walls are outside/],
      ['x 0.5 0.4 123 2 3.0 3.0', /wall ordering or some walls are outside/],
    ];
    for (const [tail, re] of cases) {
      const { error } = await runScript(`${BASE}\nfix 2 all wall/flow ${tail}\nrun 1\n`);
      expect(error?.message ?? '', tail).toMatch(re);
    }
  });

  it('refuses a non-periodic flow axis and a zero or negative temperature', async () => {
    const nonPer = `${BASE.replace('boundary        p p p', 'boundary        f p p')}\nfix 2 all wall/flow x 0.5 0.4 123 1 3.0\nrun 1\n`;
    expect((await runScript(nonPer)).error?.message ?? '').toMatch(/non-periodic boundary along the flow axis/);
    const zeroT = `${BASE}\nfix 2 all wall/flow x 0.5 0.0 123 1 3.0\nrun 1\n`;
    expect((await runScript(zeroT)).error?.message ?? '').toMatch(/temperature T must be a positive number/);
  });
});

describe('fix wall/flow: native draws', () => {
  // Measured with native LAMMPS (black box): one atom at x = 2.99 with vx = 3, wall at 3.0, flow 0.5,
  // T = 1, seed 12345 (lj units): the new velocity after the crossing at step 1 is (-2.69664594826187,
  // -0.220607254486375, -0.473224025462307); the second crossing (step 2, from the right) gives
  // (1.23587584795196, 1.03475555095725, -2.13474644916938).
  it('reproduces the first and second reassignment of a one-atom run', async () => {
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (name, text) => files.set(name, (files.get(name) ?? '') + text) });
    await session.execute(`${BASE}\nfix 2 all wall/flow x 0.5 1.0 12345 1 3.0\nrun 1\nwrite_dump all custom a.dump id vx vy vz modify format float %.15g\nrun 1\nwrite_dump all custom b.dump id vx vy vz modify format float %.15g\n`);
    const row = (name: string) => files.get(name)!.trim().split('\n').slice(-1)[0].split(/\s+/).map(Number);
    const [, vx1, vy1, vz1] = row('a.dump');
    expect(vx1).toBeCloseTo(-2.69664594826187, 11);
    expect(vy1).toBeCloseTo(-0.220607254486375, 11);
    expect(vz1).toBeCloseTo(-0.473224025462307, 11);
    const [, vx2, vy2, vz2] = row('b.dump');
    expect(vx2).toBeCloseTo(1.23587584795196, 11);
    expect(vy2).toBeCloseTo(1.03475555095725, 11);
    expect(vz2).toBeCloseTo(-2.13474644916938, 11);
  });

  it('flow component equals the measured native values for the draws of RanMars(12345)', () => {
    // draw 0 is the warm-up; crossing 1 uses draws 1, 2 (flow) and then the polar pair (5, 6);
    // crossing 2 uses draws 7, 8 (flow). Native values are at sigma = 1 (T = 1, m = 1), vflow = 0.5.
    const g = new RanMars(12345);
    g.uniform();
    const a1 = g.uniform(), b1 = g.uniform();
    expect(fluxNormal(0.5, a1, b1)).toBeCloseTo(-2.69664594826187, 11);
    g.polarPair(); // draws 3..6 (3, 4 rejected; 5, 6 accepted)
    const a2 = g.uniform(), b2 = g.uniform();
    expect(fluxNormal(0.5, a2, b2)).toBeCloseTo(1.23587584795196, 11);
  });
});

describe('fix wall/flow: statistics of the generated velocity', () => {
  // The flow component has density proportional to |x| exp(-(x - r)^2 / 2) on the whole line
  // (r = vflow / sigma): P(x < 0) = (phi(r) - r Phi(-r)) / Z and E[x] = int x |x| phi(x - r) / Z.
  const phi = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const Phi = (z: number) => {
    // Abramowitz-Stegun 26.2.17 is accurate to 7.5e-8, enough for a statistical check
    const t = 1 / (1 + 0.2316419 * Math.abs(z));
    const p = phi(z) * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return z >= 0 ? 1 - p : p;
  };

  it('matches the theoretical sign fraction and mean over 40000 draws', () => {
    for (const r of [0.0, 0.5, 2.0]) {
      const g = new RanMars(4242);
      g.uniform();
      const n = 40000;
      let neg = 0;
      let sum = 0;
      for (let k = 0; k < n; k++) {
        const x = fluxNormal(r, g.uniform(), g.uniform());
        if (x < 0) neg++;
        sum += x;
      }
      const Z = 2 * phi(r) + r * (2 * Phi(r) - 1);
      const pNeg = (phi(r) - r * Phi(-r)) / Z;
      const mean = sum / n;
      // E[x] = (int_{-inf}^{inf} x |x| phi(x - r) dx) / Z, from the closed forms of the two half-lines
      const A = (1 + r * r) * Phi(r) + r * phi(r);
      const B = (1 + r * r) * Phi(-r) - r * phi(r);
      const meanTheory = (A - B) / Z;
      expect(Math.abs(neg / n - pNeg), `P(neg) at r=${r}`).toBeLessThan(0.01);
      expect(Math.abs(mean - meanTheory), `mean at r=${r}`).toBeLessThan(0.03);
    }
  });

  it('velocities of 400 crossings in one step: flow mean and sigma^2 transverse variance', async () => {
    // 400 atoms in a slab 0.01 wide just below the wall at x = 5 with vx = 3: all cross in step 1
    const files = new Map<string, string>();
    const session = new Session({ emit: () => {}, writeFile: (name, text) => files.set(name, (files.get(name) ?? '') + text) });
    await session.execute(`
units           lj
atom_style      atomic
boundary        p p p
region          box block 0 10 0 10 0 10
region          slab block 4.99 4.999 0 10 0 10
create_box      1 box
create_atoms    1 random 400 777 slab
mass            1 1.0
velocity        all set 3.0 0.0 0.0 sum no units box
pair_style      zero 2.5
pair_coeff      * *
fix             1 all nve
fix             2 all wall/flow x 0.5 1.0 555 1 5.0
run 1
write_dump      all custom flow.dump id vx vy vz
`);
    const text = files.get('flow.dump') ?? '';
    const lines = text.trim().split('\n');
    const data = lines.slice(lines.indexOf('ITEM: ATOMS id vx vy vz') + 1).map((l) => l.split(/\s+/).map(Number));
    expect(data.length).toBe(400);
    const n = data.length;
    const mean = (k: number) => data.reduce((a, row) => a + row[k], 0) / n;
    const variance = (k: number) => {
      const m = mean(k);
      return data.reduce((a, row) => a + (row[k] - m) ** 2, 0) / n;
    };
    // T = 1, m = 1: sigma = 1; the flow component has mean E[x] = 0.92757 (r = 0.5), see the statistics above
    // columns: 0 id, 1 vx (flow), 2 vy, 3 vz
    expect(Math.abs(mean(1) - 0.92757)).toBeLessThan(0.25);
    expect(Math.abs(mean(2))).toBeLessThan(0.2);
    expect(Math.abs(mean(3))).toBeLessThan(0.2);
    expect(Math.abs(variance(2) - 1)).toBeLessThan(0.25);
    expect(Math.abs(variance(3) - 1)).toBeLessThan(0.25);
  });
});
