import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { correlationPairs, correlationLabelPairs } from '../src/engine/fix/gk';

/*
 * Green-Kubo tools of wave 11: fix ave/correlate, compute heat/flux and the Muller-Plathe fixes.
 * Expected correlation values are computed by brute force from the definition in
 * docs.lammps.org/fix_ave_correlate.html ("C_ij(dt) = <V_i(t) V_j(t+dt)>") over the sample window
 * that the measured native-LAMMPS behaviour selects (both samples in [T_prev, T] for ave one).
 */

const BASE = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.44 87287 loop all
pair_style      lj/cut 2.5
pair_coeff      * * 1.0 1.0 2.5
timestep        0.001
fix             1 all nve
variable        s equal step
variable        t equal step*step
`;

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const files = new Map<string, string>();
  const session = new Session({
    emit: (e) => events.push(e),
    writeFile: (n, txt, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + txt),
  });
  try {
    await session.execute(BASE + script);
  } catch (e) {
    throw new Error(String(e));
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && 'message' in err) throw new Error(err.message);
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { files, thermo };
};

/** Brute-force C(lag) over samples in [lo, hi] (sample spacing every) for series f and g. */
const corr = (f: (s: number) => number, g: (s: number) => number, lag: number, lo: number, hi: number, every = 1): { value: number; count: number } => {
  let sum = 0, count = 0;
  for (let a = lo; a <= hi; a += every) {
    const b = a + lag;
    if (b > hi) continue;
    sum += f(a) * g(b);
    count++;
  }
  return { value: count ? sum / count : 0, count };
};

describe('fix ave/correlate values', () => {
  const s = (x: number) => x;
  const t = (x: number) => x * x;

  it('ave one: window [T_prev, T] with the sample at T carried into the next window', async () => {
    const { files } = await run('fix JJ all ave/correlate 1 3 5 v_s type auto ave one file gk_one.dat\nrun 10\n');
    const lines = files.get('gk_one.dat')!.trim().split('\n');
    const at = lines.indexOf('10 3');
    expect(at).toBeGreaterThan(0);
    const rows = lines.slice(at + 1, at + 4).map((l) => l.trim().split(/\s+/).map(Number));
    for (let lag = 0; lag < 3; lag++) {
      const want = corr(s, s, lag, 5, 10);
      expect(rows[lag][1]).toBe(lag);
      expect(rows[lag][2]).toBe(want.count);
      expect(rows[lag][3]).toBeCloseTo(want.value, 3); // file values are %g (six digits)
    }
    const first = lines.slice(lines.indexOf('5 3') + 1, lines.indexOf('5 3') + 2)[0].trim().split(/\s+/).map(Number);
    expect(first[2]).toBe(6);
    expect(first[3]).toBeCloseTo(corr(s, s, 0, 0, 5).value, 3);
  });

  it('ave running: every pair since the first sample; prefactor scales the averages', async () => {
    const { files } = await run('fix JJ all ave/correlate 2 2 4 v_t prefactor 0.5 ave running file gk_run.dat\nrun 8\n');
    const lines = files.get('gk_run.dat')!.trim().split('\n');
    const at = lines.indexOf('8 2');
    expect(at).toBeGreaterThan(0);
    const row0 = lines[at + 1].trim().split(/\s+/).map(Number);
    const want = corr(t, t, 0, 0, 8, 2);
    expect(row0[2]).toBe(want.count);
    expect(row0[3]).toBeCloseTo(0.5 * want.value, 9);
  });

  it('start keyword: sampling begins on the start step and prefactor multiplies the output', async () => {
    const { files } = await run('fix JJ all ave/correlate 2 3 6 v_s start 4 prefactor 0.5 ave one file gk_start.dat\nrun 6\n');
    const lines = files.get('gk_start.dat')!.trim().split('\n');
    expect(lines.indexOf('0 3')).toBe(-1); // no output before the start step
    const at = lines.indexOf('6 3');
    expect(at).toBeGreaterThan(0);
    const row0 = lines[at + 1].trim().split(/\s+/).map(Number);
    const row1 = lines[at + 2].trim().split(/\s+/).map(Number); // lag 1 = dt of 2 steps
    expect(row0[2]).toBe(2); // samples at steps 4 and 6
    expect(row0[3]).toBeCloseTo(0.5 * corr(s, s, 0, 4, 6, 2).value, 9);
    expect(row1[1]).toBe(2);
    expect(row1[2]).toBe(1);
    expect(row1[3]).toBeCloseTo(0.5 * 4 * 6, 9);
  });

  it('thermo reads the array columns and trap() integrates a column', async () => {
    const { thermo } = await run('fix JJ all ave/correlate 1 4 5 v_s type auto ave running\nvariable tr equal trap(f_JJ[3])\nthermo_style custom step f_JJ[1][3] v_tr\nthermo 5\nrun 5\n');
    const last = thermo[thermo.length - 1] as unknown as Record<string, number>;
    expect(last['f_JJ[1][3]']).toBeCloseTo(corr(s, s, 0, 0, 5).value, 9);
    expect(Number.isFinite(last['v_tr'])).toBe(true);
  });
});

describe('ave/correlate pair lists', () => {
  it('orders the columns as the doc page lists them', () => {
    expect(correlationPairs('upper', 3)).toEqual([[0, 1], [0, 2], [1, 2]]);
    expect(correlationPairs('lower', 3)).toEqual([[1, 0], [2, 0], [2, 1]]);
    expect(correlationPairs('auto/lower', 2)).toEqual([[0, 0], [1, 0], [1, 1]]);
    expect(correlationPairs('full', 2)).toHaveLength(4);
    expect(correlationPairs('first', 3)).toEqual([[0, 0], [0, 1], [0, 2]]);
  });

  it('header label pairs follow the measured LAMMPS header', () => {
    expect(correlationLabelPairs('lower', 2)).toHaveLength(0);
    expect(correlationLabelPairs('lower', 4)).toHaveLength(3);
    expect(correlationLabelPairs('auto/lower', 3)).toHaveLength(3);
  });
});

describe('argument errors', () => {
  const bad = (script: string, msg: RegExp) => expect(run(script + "run 0\n")).rejects.toThrow(msg);

  it('ave/correlate rejects invalid arguments', async () => {
    await bad('fix JJ all ave/correlate 0 3 5 v_s\n', /positive integer/);
    await bad('fix JJ all ave/correlate 2 3 5 v_s\n', /multiple of Nevery/);
    await bad('fix JJ all ave/correlate 1 7 5 v_s\n', /Nfreq must be >= \(Nrepeat-1\)\*Nevery/);
    await bad('fix JJ all ave/correlate 1 3 5 v_s overwrite\n', /overwrite keyword can only be used with the ave running/);
    await bad('fix JJ all ave/correlate 1 3 5 v_s type sideways\n', /type must be one of/);
    await bad('fix JJ all ave/correlate 1 3 5 v_s bogus 1\n', /invalid input value 'bogus'/);
    await bad('fix JJ all ave/correlate 1 3 5\n', /usage: fix JJ/);
  });

  it('viscosity and thermal/conductivity reject bad layer counts and keywords', async () => {
    await bad('fix V all viscosity 10 x z 5\n', /even number/);
    await bad('fix V all viscosity 10 q z 4\n', /vdim must be x, y or z/);
    await bad('fix V all viscosity 10 x z 4 swap 0\n', /swap Nswap must be a positive integer/);
    await bad('fix V all viscosity 10 x z 4 bogus 2\n', /unknown keyword 'bogus'/);
    await bad('fix E all thermal/conductivity 10 z 3\n', /even number/);
    await bad('fix E all thermal/conductivity 10 z 4 vtarget 1\n', /unknown keyword 'vtarget'/);
  });

  it('compute heat/flux needs exactly three compute IDs', async () => {
    await bad('compute F all heat/flux ke pe\n', /usage: compute ID group-ID heat\/flux/);
  });
});
