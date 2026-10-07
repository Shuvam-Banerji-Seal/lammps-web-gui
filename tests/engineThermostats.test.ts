import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { CpuForceBackend } from '../src/engine/cpu/forces';
import { newPairTable, resolvePairs, setPairCoeff } from '../src/engine/pairs';
import { temperature } from '../src/engine/observables';
import { run as runSteps } from '../src/engine/md';
import { FixNvt } from '../src/engine/thermostats';
import { EngineError, type EngineEvent, type ThermoRow } from '../src/engine/types';

/** Runs input text in a fresh session and collects the thermo rows / error. */
const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  let error: EngineError | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as EngineError;
  }
  const thermo = events
    .filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo')
    .map((e) => e.row);
  return { session, thermo, error };
};

/** The 500-atom examples/melt liquid (5x5x5 fcc at rho* 0.8442) via the interpreter. */
const LIQUID = (fixes: string, o: { velT?: number; steps?: number; seed?: number } = {}): string => `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create ${o.velT ?? 3.0} ${o.seed ?? 4928}
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
${fixes}
thermo 50
run ${o.steps ?? 4000}
`;

const meltTable = () => {
  const t = newPairTable(1, 2.5);
  setPairCoeff(t, '1', '1', 1.0, 1.0, 2.5);
  expect(resolvePairs(t)).toEqual([]);
  return t;
};

const mean = (a: number[]): number => {
  let s = 0;
  for (const v of a) s += v;
  return a.length > 0 ? s / a.length : 0;
};

describe('thermostats through the interpreter (500-atom lj melt)', () => {
  const holdsTarget = async (fixes: string, label: string) => {
    const { thermo } = await runScript(LIQUID(fixes, { steps: 4000 }));
    const half = thermo.filter((r) => (r.step ?? 0) > 2000).map((r) => r.temp as number);
    const t = mean(half);
    console.log(`[evidence] ${label}: mean T over steps 2000-4000 = ${t.toFixed(4)} (${half.length} rows)`);
    expect(Math.abs(t - 1.5) / 1.5).toBeLessThan(0.03);
  };

  it('nve + langevin damp 0.5 holds T = 1.5 within 3%', async () => {
    await holdsTarget('fix 1 all nve\nfix 2 all langevin 1.5 1.5 0.5 12345', 'langevin');
  }, 240000);

  it('nvt damp 0.5 holds T = 1.5 within 3%', async () => {
    await holdsTarget('fix 1 all nvt temp 1.5 1.5 0.5', 'nvt');
  }, 240000);

  it('nve + temp/berendsen damp 0.5 holds T = 1.5 within 3%', async () => {
    await holdsTarget('fix 1 all nve\nfix 2 all temp/berendsen 1.5 1.5 0.5', 'temp/berendsen');
  }, 240000);

  it('temp/rescale 10: T is exactly 1.5 right after each rescale; window 10 never rescales', async () => {
    const rescaled = await runScript(LIQUID('fix 1 all nve\nfix 2 all temp/rescale 10 1.5 1.5 0.0 1.0', { steps: 2000 }));
    for (const r of rescaled.thermo) {
      if ((r.step ?? 0) > 0) expect(Math.abs((r.temp ?? 0) - 1.5)).toBeLessThan(1e-9);
    }
    const windowed = await runScript(LIQUID('fix 1 all nve\nfix 2 all temp/rescale 10 1.5 1.5 10.0 1.0', { steps: 2000 }));
    const plain = await runScript(LIQUID('fix 1 all nve', { steps: 2000 }));
    expect(windowed.thermo.map((r) => r.temp)).toEqual(plain.thermo.map((r) => r.temp));
  }, 240000);

  it('temp/berendsen ramps T from 1.0 to 2.0 over 3000 steps', async () => {
    const { thermo } = await runScript(LIQUID('fix 1 all nve\nfix 2 all temp/berendsen 1.0 2.0 0.1', { velT: 1.0, steps: 3000 }));
    const first = thermo.filter((r) => (r.step ?? 0) <= 300).map((r) => r.temp as number);
    const last = thermo.filter((r) => (r.step ?? 0) >= 2700).map((r) => r.temp as number);
    const tFirst = mean(first);
    const tLast = mean(last);
    console.log(`[evidence] berendsen ramp: mean T first 300 steps = ${tFirst.toFixed(4)}, last 300 = ${tLast.toFixed(4)}`);
    expect(tFirst).toBeGreaterThan(0.9);
    expect(tFirst).toBeLessThan(1.25);
    expect(tLast).toBeGreaterThan(1.85);
    expect(tLast).toBeLessThan(2.05);
  }, 240000);

  it('langevin fluctuation-dissipation on an ideal gas: T and <vx^2> match kB T/m', async () => {
    const head = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 2.0
velocity all create 1.2 87287
pair_style lj/cut 2.5
pair_coeff 1 1 0.0 1.0 2.5
fix 1 all langevin 1.2 1.2 0.1 9871
fix 2 all nve
thermo 50
run 2000`;
    const { session } = await runScript(head);
    const s = session.system!;
    const vx2: number[] = [];
    const temps: number[] = [];
    for (let c = 0; c < 40; c++) {
      await session.execute('run 100');
      let sum = 0;
      const v = s.v;
      for (let i = 0; i < s.n; i++) sum += v[3 * i] * v[3 * i];
      vx2.push(sum / s.n);
      temps.push(temperature(s));
    }
    // velocity-Verlet Langevin with the friction on the half-step velocity has
    // a known O(dt/damp) temperature bias (here dt/damp = 0.05); the 2%/3%
    // bands below allow for it
    const tMean = mean(temps);
    const vx2Mean = mean(vx2);
    console.log(`[evidence] ideal gas langevin: mean T (steps 2000-6000) = ${tMean.toFixed(4)}, <vx^2> = ${vx2Mean.toFixed(4)} (kB T/m = 0.6)`);
    expect(Math.abs(tMean - 1.2) / 1.2).toBeLessThan(0.02);
    expect(Math.abs(vx2Mean - 0.6) / 0.6).toBeLessThan(0.03);
  }, 240000);
});

describe('fix nvt conserved quantity', () => {
  it('drift |H\'(end) - H\'(start)|/N < 2e-3 over 2000 steps at dt 0.005, Tdamp 0.5', async () => {
    const setup = `
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 5 0 5 0 5
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 1.5 4928
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
pair_modify shift yes
timestep 0.005
fix 1 all nvt temp 1.5 1.5 0.5`;
    const { session } = await runScript(setup);
    const s = session.system!;
    // shifted energy: without it the energy jump of pairs crossing the
    // cutoff dominates any integrator drift (plain NVE drifts too)
    const table = meltTable();
    table.shift = true;
    const backend = new CpuForceBackend();
    const fix = new FixNvt('1', 1.5, 1.5, 0.5);
    const h0 = fix.conservedEnergy(s, backend.compute(s, table).pe);
    await runSteps(s, table, backend, [fix], 2000, { thermoEvery: 0, keywords: ['step', 'temp'] });
    const h1 = fix.conservedEnergy(s, backend.compute(s, table).pe);
    const drift = Math.abs(h1 - h0) / s.n;
    console.log(`[evidence] nvt conserved energy: H'/N start = ${(h0 / s.n).toFixed(6)}, end = ${(h1 / s.n).toFixed(6)}, |drift|/N = ${drift.toExponential(3)}`);
    expect(drift).toBeLessThan(2e-3);
    expect(Math.abs(temperature(s) - 1.5)).toBeLessThan(0.15);
  }, 240000);
});

describe('thermostat errors', () => {
  it('temp/berendsen fails on a zero temperature with an explicit error', async () => {
    // two atoms beyond the cutoff: no forces, so the temperature stays exactly 0
    const { error } = await runScript(`
units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 single 0.5 0.5 0.5
create_atoms 1 single 2.5 2.5 2.5
mass 1 1.0
velocity all set 0 0 0 units box
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
fix 2 all temp/berendsen 1.5 1.5 0.5
run 5`);
    expect(error?.message).toMatch(/zero temperature/);
  });

  it('nve + nvt together fail with the both-integrate error', async () => {
    const { error } = await runScript(`
units lj
atom_style atomic
lattice sc 1.0
region b block 0 3 0 3 0 3
create_box 1 b
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 1.1
pair_coeff 1 1 1.0 1.0
fix 1 all nve
fix 2 all nvt temp 1 1 0.5
run 1`);
    expect(error?.message).toMatch(/both integrate/);
  });
});
