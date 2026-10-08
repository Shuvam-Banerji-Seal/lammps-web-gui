import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix gjf (docs.lammps.org/fix_gjf.html): argument errors and the defining
 * statistical property of the GJ methods. Gronbech-Jensen 2020 (arXiv:1909.04380,
 * Eqs. (21) and (47)): for a harmonic oscillator the GJ trajectory samples the
 * configurational Boltzmann distribution for any time step within the stability
 * limit, so kappa <x^2> = kB T. The test checks this at a large time step for
 * the default method (GJ-I) and for method 4 (GJ-IV), on independent oscillators
 * tethered by fix spring/self. The numerical agreement with native LAMMPS is in
 * tests/oracle/w9gjf_*.in.
 */

const runScript = async (text: string, files: Map<string, string> = new Map()) => {
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (name, t, append) => files.set(name, (append ? files.get(name) ?? '' : '') + t),
  });
  let error: unknown = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e;
  }
  return { error, session, files };
};

const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e));

const SETUP = `
units           lj
atom_style      atomic
atom_modify     map array sort 0 0.0
lattice         sc 0.5
region          box block 0 4 0 4 0 4
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      zero 2.5
pair_coeff      * *
velocity        all create 1.0 4928 loop all
`;

describe('fix gjf argument errors', () => {
  const base = `${SETUP}\n`;
  it('needs Tstart Tstop damp seed', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0\nrun 1`);
    expect(errorText(r.error)).toMatch(/usage: fix ID group gjf/);
  });
  it('damp must be positive', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 0.0 9182\nrun 1`);
    expect(errorText(r.error)).toMatch(/damp must be > 0/);
  });
  it('seed must be a positive integer', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 -5\nrun 1`);
    expect(errorText(r.error)).toMatch(/seed must be a positive integer/);
  });
  it('vel must be vfull or vhalf', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 9182 vel half\nrun 1`);
    expect(errorText(r.error)).toMatch(/vel must be vfull or vhalf/);
  });
  it('method must be 1-8', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 9182 method 9\nrun 1`);
    expect(errorText(r.error)).toMatch(/method must be 1-8/);
  });
  it('methods 7 and 8 are named as not implemented', async () => {
    const r7 = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 9182 method 7 0.95\nrun 1`);
    expect(errorText(r7.error)).toMatch(/method 7 \(GJ-VII\) is not implemented/);
    const r8 = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 9182 method 8\nrun 1`);
    expect(errorText(r8.error)).toMatch(/method 8 \(GJ-VIII\) is not implemented/);
  });
  it('unknown keywords throw a StyleError naming them', async () => {
    const r = await runScript(`${base}fix 1 all gjf 1.0 1.0 2.0 9182 tally yes\nrun 1`);
    expect(errorText(r.error)).toMatch(/unknown fix gjf keyword 'tally'/);
  });
});

/**
 * Configurational temperature of independent harmonic oscillators, kappa <dx^2>
 * per component, sampled along a GJ trajectory with a time step well above the
 * Verlet limit for the frequency used (omega dt = 0.8 ... kept below 2).
 */
const configurationalT = async (method: string): Promise<number> => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const session = new Session({
    emit: (ev) => events.push(ev),
    writeFile: (name, t, append) => files.set(name, (append ? files.get(name) ?? '' : '') + t),
  });
  const parse = (text: string) => {
    const lines = text.trim().split('\n');
    const k = lines.findIndex((l) => l.startsWith('ITEM: ATOMS'));
    return lines.slice(k + 1).map((l) => l.trim().split(/\s+/).map(Number));
  };
  await session.execute(`${SETUP}
timestep        0.2
fix             1 all spring/self 1.0
fix             2 all gjf 1.0 1.0 1.0 4711 ${method}
write_dump      all custom ref.dump id xu yu zu modify format float %.17g sort id
`);
  // the spring origin is the lattice site at fix creation, i.e. the step-0 positions
  const ref = parse(files.get('ref.dump')!);
  await session.execute('run 500'); // equilibrate
  const samples: number[] = [];
  for (let s = 0; s < 400; s++) {
    await session.execute('run 5\nwrite_dump all custom snap.dump id xu yu zu modify format float %.17g sort id');
    const snap = parse(files.get('snap.dump')!);
    for (let i = 0; i < snap.length; i++) {
      for (let c = 1; c <= 3; c++) {
        const d = snap[i][c] - ref[i][c];
        samples.push(d * d);
      }
    }
  }
  // kappa = 1: kappa <dx^2> is the configurational temperature
  let sum = 0;
  for (const x of samples) sum += x;
  return sum / samples.length;
};

describe('fix gjf statistics', () => {
  it('GJ-I (default) samples the configurational temperature of a harmonic oscillator at dt = 0.2', async () => {
    const tc = await configurationalT('');
    expect(Math.abs(tc - 1)).toBeLessThan(0.05);
  }, 120_000);

  it('GJ-IV samples the same configurational temperature', async () => {
    const tc = await configurationalT('method 4');
    expect(Math.abs(tc - 1)).toBeLessThan(0.05);
  }, 120_000);
});
