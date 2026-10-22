import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { parseStoppingTable } from '../src/engine/fix/electron_stopping';

/*
 * Unit tests for fix dt/reset, fix electron/stopping and fix ave/moments: parsing, refusals and the
 * rules measured with native LAMMPS (black box). Parity over whole runs is in tests/oracle/w47pkg_*.in.
 */

const TABLE = '# test table\n# eV eV/Ang\n10 0.5\n1000 0.5\n';
const TABLE2 = '# eV eV/Ang\n10 0\n250 60\n750 100\n';

const session = (files: Record<string, string> = {}) => {
  const ev: EngineEvent[] = [];
  const s = new Session({ emit: (e) => ev.push(e), writeFile: () => {} });
  for (const [n, t] of Object.entries(files)) s.addFile(n, t);
  return { s, ev };
};

const thermoRows = (ev: EngineEvent[]): ThermoRow[] =>
  ev.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);

/** One atom with a velocity and a constant force (fix addforce), lj units; returns the dt set by run 0. */
const dtFor = async (vx: number, vy: number, fx: number, fy: number, xmax: number, mass: number, extra = ''): Promise<number> => {
  const { s, ev } = session();
  await s.execute(`units lj
atom_style atomic
region box block -10 10 -10 10 -10 10
create_box 1 box
create_atoms 1 single 0 0 0
mass 1 ${mass}
pair_style zero 3.0
pair_coeff * *
velocity all set ${vx} ${vy} 0 units box
fix 1 all nve
fix 2 all addforce ${fx} ${fy} 0
timestep 0.01
fix 5 all dt/reset 1 NULL NULL ${xmax} units box ${extra}
thermo_style custom step dt
run 0
`);
  return thermoRows(ev).pop()!.dt as number;
};

describe('fix dt/reset', () => {
  it('parses N, Tmin, Tmax, Xmax and the emax / units keywords', async () => {
    const { s } = session();
    await expect(s.execute(`units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
fix 5 all dt/reset 10 1.0e-5 0.01 0.1 emax 30 units box
`)).resolves.toBeUndefined();
  });

  it('refuses malformed arguments', async () => {
    const base = 'units lj\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\n';
    await expect(session().s.execute(base + 'fix 5 all dt/reset 0 NULL NULL 0.1\n')).rejects.toThrow(/positive integer/);
    await expect(session().s.execute(base + 'fix 5 all dt/reset 1 0.5 0.1 0.1\n')).rejects.toThrow(/Tmin must not exceed Tmax/);
    await expect(session().s.execute(base + 'fix 5 all dt/reset 1 NULL NULL 0\n')).rejects.toThrow(/Xmax/);
    await expect(session().s.execute(base + 'fix 5 all dt/reset 1 NULL NULL 0.1 emax -1\n')).rejects.toThrow(/emax/);
    await expect(session().s.execute(base + 'fix 5 all dt/reset 1 NULL NULL 0.1 units metric\n')).rejects.toThrow(/units must be lattice or box/);
    await expect(session().s.execute(base + 'fix 5 all dt/reset 1 NULL NULL 0.1 foo 3\n')).rejects.toThrow(/unknown keyword 'foo'/);
  });

  it('chooses the timestep from the displacement rule (values measured with native LAMMPS)', async () => {
    // v = 0.5, no force: dt = Xmax / |v|
    expect(await dtFor(0.5, 0, 0, 0, 0.1, 1)).toBeCloseTo(0.2, 14);
    // v = 0, a = 1: dt = sqrt(2 Xmax / |a|)
    expect(await dtFor(0, 0, 1, 0, 0.1, 1)).toBeCloseTo(Math.sqrt(0.2), 12);
    // v = (0.5, 0), a = (0, 1): dt = Xmax / |(0.5, 0.1)|
    expect(await dtFor(0.5, 0, 0, 1, 0.1, 1)).toBeCloseTo(0.19611613513818404, 12);
    // a opposes v: the force is ignored, dt = Xmax / |v|
    expect(await dtFor(0.5, 0, -1, 0, 0.1, 1)).toBeCloseTo(0.2, 14);
    // mass 2 halves the acceleration (a = 0.1): dt = Xmax / |(0.5 + 0.5 * 0.1 * 0.2 / 0.5 ...)| (value measured)
    expect(await dtFor(0.5, 0, 0.2, 0, 0.1, 2)).toBeCloseTo(0.19607843137254901689, 12);
  });

  it('applies emax as dt <= emax / (|F| |v|) and keeps Tmin / Tmax bounds', async () => {
    expect(await dtFor(0.5, 0, 1, 0, 0.1, 1, 'emax 0.01')).toBeCloseTo(0.02, 14);
    expect(await dtFor(0.5, 0, 2, 0, 0.1, 1, 'emax 0.01')).toBeCloseTo(0.01, 14);
    // a single atom with no motion: Xmax places no limit (native reports 1e20)
    expect(await dtFor(0, 0, 0, 0, 0.1, 1)).toBe(1e20);
  });

  it('keeps thermo time exact when dt changes and reports the last reset step as f_ID', async () => {
    const { s, ev } = session();
    await s.execute(`units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
fix 1 all nve
timestep 0.005
fix 5 all dt/reset 1 NULL NULL 0.1 units box
thermo_style custom step dt time f_5
thermo 1
run 6
`);
    const rows = thermoRows(ev);
    let t = 0;
    for (let k = 1; k < rows.length; k++) {
      // time after step k is the sum of the timesteps used for steps 1..k: row k-1's dt is the step-k dt
      t += rows[k - 1].dt as number;
      expect(rows[k].time as number).toBeCloseTo(t, 12);
    }
    // the scalar is the last step on which dt changed to a new value
    let last = 0;
    for (let k = 1; k < rows.length; k++) if (rows[k].dt !== rows[k - 1].dt) last = rows[k].step as number;
    expect(rows[rows.length - 1].f_5 as number).toBe(last);
  });
});

describe('fix electron/stopping', () => {
  const es = (v: number, ecut: number, file: string, extra = '') => `units metal
atom_style atomic
region box block -50 50 -50 50 -50 50
create_box 1 box
create_atoms 1 single 0 0 0
mass 1 1.0
pair_style zero 3.0
pair_coeff * *
velocity all set ${v} 0 0 units box
fix 1 all nve
timestep 0.001
fix 2 all electron/stopping ${ecut} ${file} minneigh 0 ${extra}
compute vs all reduce sum vx
thermo_style custom step c_vs f_2
thermo_modify format float %.15g
thermo 1
run 2
`;

  it('parses the table: comment lines skipped, N+1 columns, sorted energies', () => {
    const t = parseStoppingTable('# c\n\n10 0.5 1\n250 60 80\n', 2, 'tab');
    expect(Array.from(t.e)).toEqual([10, 250]);
    expect(Array.from(t.se[0])).toEqual([0.5, 60]);
    expect(Array.from(t.se[1])).toEqual([1, 80]);
    expect(() => parseStoppingTable('10 0\n', 2, 'tab')).toThrow(/expected 3/);
    expect(() => parseStoppingTable('250 1\n10 1\n', 1, 'tab')).toThrow(/sorted/);
    expect(() => parseStoppingTable('10 1\n', 1, 'tab')).toThrow(/at least two/);
  });

  it('refuses a still-unsupported style and bad arguments', async () => {
    // electron/stopping/fit is a GPL port now (src/engine-gpl/extrafix); graphics/isosurface is still refused
    await expect(session().s.execute('units metal\natom_style atomic\nlattice fcc 0.8442\nregion box block 0 3 0 3 0 3\ncreate_box 1 box\ncreate_atoms 1 box\nmass 1 1.0\nfix 3 all graphics/isosurface\n'))
      .rejects.toThrow(/isosurface/);
    await expect(session({ 'tab2.txt': TABLE2 }).s.execute(es(1388, 0, 'tab2.txt'))).rejects.toThrow(/Ecut must be a positive energy/);
    await expect(session({ 'tab2.txt': TABLE2 }).s.execute(es(1388, 1, 'tab2.txt', 'region nowhere'))).rejects.toThrow(/nowhere/);
  });

  it('removes energy along the velocity with the interpolated stopping power (values measured with native LAMMPS)', async () => {
    // Se = 0.5 eV/A, mass 1, v = 1388 A/ps: one step removes half a step of 0.5 * dt * ftm2v = 2.41 A/ps (first step)
    const { s, ev } = session({ 'tab1.txt': TABLE });
    await s.execute(es(1388, 99, 'tab1.txt'));
    const rows = thermoRows(ev);
    expect(rows[1].c_vs as number).toBeCloseTo(1385.58786654418, 6);
    expect(rows[1].f_2 as number).toBeCloseTo(0.694, 9);
    expect(rows[2].c_vs as number).toBeCloseTo(1380.76359963255, 6);
    expect(rows[2].f_2 as number).toBeCloseTo(1.38558786654418, 9);
    // Ecut above the kinetic energy (99.8 eV): no friction
    const none = session({ 'tab1.txt': TABLE });
    await none.s.execute(es(1388, 101, 'tab1.txt'));
    expect(thermoRows(none.ev).every((r) => r.c_vs === 1388 && r.f_2 === 0)).toBe(true);
    // interpolation between (10, 0) and (250, 60): Se = 22.45 eV/A at 99.8 eV
    const tab2 = session({ 'tab2.txt': TABLE2 });
    await tab2.s.execute(es(1388, 1, 'tab2.txt'));
    expect(thermoRows(tab2.ev)[1].f_2 as number).toBeCloseTo(31.173127144743, 6);
  });

  it('refuses kinetic energies above the largest table energy', async () => {
    await expect(session({ 'tab2.txt': TABLE2 }).s.execute(es(4000, 1, 'tab2.txt'))).rejects.toThrow(/above the largest table energy/);
  });

  it('applies no friction below the first table energy (measured)', async () => {
    const { s, ev } = session({ 'tab2.txt': TABLE2 });
    await s.execute(es(310, 1, 'tab2.txt'));
    expect(thermoRows(ev).every((r) => r.c_vs === 310 && r.f_2 === 0)).toBe(true);
  });

  it('counts neighbours within the neighbour-list cutoff (pair cutoff plus skin, measured)', async () => {
    const dimer = (d: number, mn: string) => `units metal
atom_style atomic
region box block -50 50 -50 50 -50 50
create_box 1 box
create_atoms 1 single 0 0 0
create_atoms 1 single ${d} 0 0
mass 1 1.0
pair_style zero 3.0
pair_coeff * *
velocity all set 1388 0 0 units box
fix 1 all nve
timestep 0.001
fix 2 all electron/stopping 1 tab1.txt ${mn}
compute vs all reduce sum vx
thermo_style custom step c_vs f_2
thermo_modify format float %.15g
thermo 1
run 1
`;
    const near = session({ 'tab1.txt': TABLE });
    await near.s.execute(dimer(4.0, 'minneigh 1'));
    expect(thermoRows(near.ev)[1].f_2 as number).toBeCloseTo(1.388, 9);
    const far = session({ 'tab1.txt': TABLE });
    await far.s.execute(dimer(6.0, 'minneigh 1'));
    expect(thermoRows(far.ev)[1].f_2 as number).toBe(0);
    const two = session({ 'tab1.txt': TABLE });
    await two.s.execute(dimer(2.5, 'minneigh 2'));
    expect(thermoRows(two.ev)[1].f_2 as number).toBe(0);
  });
});

describe('fix ave/moments', () => {
  const base = `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 single 0 0 0
mass 1 1.0
pair_style zero 3.0
pair_coeff * *
fix 1 all nve
variable s equal step
`;

  it('refuses malformed arguments', async () => {
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 4 4 v_s\n')).rejects.toThrow(/no moments/);
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 4 4 v_s median\n')).rejects.toThrow(/no moments|median/);
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 4 4 mean\n')).rejects.toThrow(/no input values/);
    await expect(session().s.execute(base + 'fix 2 all ave/moments 0 4 4 v_s mean\n')).rejects.toThrow(/Nevery/);
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 4 4 v_s mean history 0\n')).rejects.toThrow(/history/);
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 4 4 v_s mean foo 1\n')).rejects.toThrow(/unknown keyword 'foo'/);
    // Nevery 1 with Nrepeat > Nfreq: native output differs from the documented window (file comment)
    await expect(session().s.execute(base + 'fix 2 all ave/moments 1 5 4 v_s mean\n')).rejects.toThrow(/Nrepeat > Nfreq/);
  });

  it('computes the moments of the window with the doc definitions (values measured with native LAMMPS)', async () => {
    // Nevery 1, Nrepeat 8, Nfreq 8: the window at step 8 is the samples of steps 1..8
    const { s, ev } = session();
    await s.execute(base + 'variable q equal "step*step - 3*step + (step % 4)"\n'
      + 'fix 2 all ave/moments 1 8 8 v_q mean stddev variance skew kurtosis\n'
      + 'thermo_style custom step f_2[1] f_2[2] f_2[3] f_2[4] f_2[5]\nthermo_modify format float %.15g\nthermo 8\nrun 8\n');
    const last = thermoRows(ev).pop()!;
    expect(last['f_2[1]']).toBeCloseTo(13.5, 12);
    expect(last['f_2[2]']).toBeCloseTo(15.3529894715748, 9);
    expect(last['f_2[3]']).toBeCloseTo(235.714285714286, 9);
    expect(last['f_2[4]']).toBeCloseTo(0.879189153563413, 12);
    expect(last['f_2[5]']).toBeCloseTo(-0.663817520661157, 12);
  });

  it('outputs zero for windows with fewer than three samples and NaN kurtosis at n = 3', async () => {
    const { s, ev } = session();
    await s.execute(base + 'fix 2 all ave/moments 1 2 2 v_s mean stddev skew\n'
      + 'fix 3 all ave/moments 1 3 3 v_s mean skew kurtosis\n'
      + 'thermo_style custom step f_2[1] f_2[3] f_3[1] f_3[2] f_3[3]\nthermo_modify format float %.15g\nthermo 1\nrun 3\n');
    const rows = thermoRows(ev);
    // two samples (steps 1, 2) are too few: the output is zero for every moment
    expect(rows[2]['f_2[1]']).toBe(0);
    expect(rows[3]['f_2[3]']).toBe(0);
    // three samples (steps 1, 2, 3): mean 2, skew 0, kurtosis not defined
    expect(rows[3]['f_3[1]']).toBeCloseTo(2, 12);
    expect(rows[3]['f_3[2]']).toBeCloseTo(0, 12);
    expect(Number.isNaN(rows[3]['f_3[3]'] as number)).toBe(true);
  });

  it('starts output after the start step and keeps history columns (most recent first)', async () => {
    const { s, ev } = session();
    await s.execute(base + 'variable q equal "step*step + 1"\n'
      + 'fix 2 all ave/moments 1 4 4 v_q mean start 4 history 3\n'
      + 'thermo_style custom step f_2[1] f_2[1][1] f_2[1][2]\nthermo_modify format float %.15g\nthermo 1\nrun 8\n');
    const rows = thermoRows(ev);
    expect(rows[6]['f_2[1]']).toBeCloseTo(8.5, 12);    // held from step 4, window steps 1..4: 2, 5, 10, 17
    expect(rows[8]['f_2[1]']).toBeCloseTo(44.5, 12);   // window steps 5..8: 26, 37, 50, 65
    expect(rows[8]['f_2[1][1]']).toBe(rows[8]['f_2[1]']);
    expect(rows[8]['f_2[1][2]']).toBe(rows[6]['f_2[1]']);
  });
});
