import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';
import { test as haltTest } from '../src/engine/fix/halt';

/*
 * fix halt (docs.lammps.org/fix_halt.html), fix hyper/global (fix_hyper_global.html), compute event/displace and
 * the hyper command (hyper.html): argument parsing, refusals and the run-stopping behaviour. Numbers of the
 * dynamics are checked against native LAMMPS in tests/oracle/w49hy_*.in (engineOracle.test.ts); here the
 * inputs are small and the expectations are the documented rules and the measured halt steps.
 */

interface Out {
  events: EngineEvent[];
  error: string | null;
  thermo: ThermoRow[];
  logs: string[];
}

const runText = async (text: string): Promise<Out> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = String(e instanceof Error ? e.message : e);
  }
  for (const ev of events) if (ev.kind === 'error' && error === null) error = ev.message;
  return {
    events,
    error,
    thermo: events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row),
    logs: events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text),
  };
};

/** A small LJ crystal with a moving atom: nve, thermo every step. */
const BASE = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 3 0 3 0 3
create_box      1 box
create_atoms    1 box
mass            1 1.0
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
displace_atoms  all move 0.05 0.02 0.01 units box
velocity        all set 0.3 0.1 -0.2
fix             1 all nve
timestep        0.005
variable        t equal step*dt
variable        s equal step
thermo_style    custom step pe v_s
thermo_modify   format float %.15g
thermo          1
`;

/** A 2d square lattice with one adatom (examples/hyper/in.hyper.global, cut down). */
const HYPER_BASE = `
units           lj
dimension       2
atom_style      atomic
atom_modify     map array
boundary        p f p
lattice         sq 0.8
region          box block 0 6 0 3 -2.0 2.0
create_box      1 box
create_atoms    1 box
mass            1 1.0
create_atoms    1 single 1.677 3.1 0 units box
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 1 delay 0 check yes
region          sub block INF INF INF 0.6 -0.5 0.5 units box
group           sub region sub
group           mobile subtract all sub
fix             1 mobile nve
fix             2 mobile enforce2d
fix             3 sub setforce 0.0 0.0 0.0
velocity        mobile set 0.0 0.0 0.0
timestep        0.005
`;

describe('fix halt: arguments', () => {
  it('accepts the documented forms', async () => {
    const out = await runText(`${BASE}
fix h all halt 3 v_t > 0.0105 error soft message yes
fix h2 all halt 10 v_s != 0 error continue message no
fix h3 all halt 100 v_s |^ 0
`);
    expect(out.error).toBeNull();
  });

  it('refuses tlimit and diskfree by name, with the reason', async () => {
    const t = await runText(`${BASE}\nfix h all halt 10 tlimit > 5.0\n`);
    expect(t.error).toMatch(/tlimit is not supported/);
    const d = await runText(`${BASE}\nfix h all halt 10 diskfree < 100.0 path "dump/."\n`);
    expect(d.error).toMatch(/diskfree is not supported/);
  });

  it('refuses universe yes, the path keyword, bad operators, keywords and N', async () => {
    expect((await runText(`${BASE}\nfix h all halt 10 v_s > 1 universe yes\n`)).error).toMatch(/universe yes is not supported/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_s > 1 path "x"\n`)).error).toMatch(/path is only used with attribute diskfree/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_s => 1\n`)).error).toMatch(/operator must be one of/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_s > 1 error maybe\n`)).error).toMatch(/error must be hard, soft or continue/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_s > 1 message maybe\n`)).error).toMatch(/message must be yes or no/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_s > 1 color red\n`)).error).toMatch('Unknown fix halt keyword color');
    expect((await runText(`${BASE}\nfix h all halt 0 v_s > 1\n`)).error).toMatch(/N must be an integer >= 1/);
    expect((await runText(`${BASE}\nfix h all halt 10 v_ > 1\n`)).error).toMatch(/missing variable name|attribute must be/);
  });

  it('evaluates the operators, with XOR true when exactly one value is 0.0', () => {
    expect(haltTest(1, '<', 2)).toBe(true);
    expect(haltTest(2, '<=', 2)).toBe(true);
    expect(haltTest(2, '>', 2)).toBe(false);
    expect(haltTest(2, '>=', 2)).toBe(true);
    expect(haltTest(2, '==', 2)).toBe(true);
    expect(haltTest(2, '!=', 2)).toBe(false);
    expect(haltTest(0, '|^', 3)).toBe(true);
    expect(haltTest(3, '|^', 0)).toBe(true);
    expect(haltTest(3, '|^', 2)).toBe(false);
    expect(haltTest(0, '|^', 0)).toBe(false);
  });
});

describe('fix halt: stopping runs', () => {
  it('checks on absolute multiples of N, stops after the halting step and prints no row for an off-interval step', async () => {
    const out = await runText(`${BASE}
thermo          2
run             1
fix             h all halt 3 v_s > 0 error soft message no
run             10
`);
    expect(out.error).toBeNull();
    // the check happens at step 3 (a multiple of 3), not at step 1 + 3; thermo every 2 steps prints no row for step 3
    expect(out.thermo.map((r) => r.step)).toEqual([0, 1, 1, 2]);
  });

  it('error hard stops the script with the documented message', async () => {
    const out = await runText(`${BASE}
fix             h all halt 3 v_t > 0.0105 error hard message yes
run             10
print           "not reached"
`);
    expect(out.error).toMatch(/Fix halt condition for fix-id h met on step 3 with value 0\.015/);
    expect(out.logs.some((l) => l === 'not reached')).toBe(false);
  });

  it('message yes prints the one line message', async () => {
    const out = await runText(`${BASE}
fix             h all halt 3 v_t > 0.0105 error soft message yes
run             10
`);
    expect(out.logs).toContain('Fix halt condition for fix-id h met on step 3 with value 0.015');
    expect(out.thermo[out.thermo.length - 1].step).toBe(3);
  });

  it('error soft skips later runs until timer timeout off; timer timeout with another value is refused', async () => {
    const out = await runText(`${BASE}
fix             h all halt 3 v_t > 0.0105 error soft message no
run             10
run             5
print           "after soft step=$(step)"
timer           timeout off
run             4
`);
    expect(out.error).toBeNull();
    expect(out.logs).toContain('after soft step=3');
    expect(out.thermo.map((r) => r.step)).toEqual([0, 1, 2, 3, 3, 4, 5, 6]);
    const refused = await runText(`${BASE}\ntimer timeout 100\n`);
    expect(refused.error).toMatch(/timer timeout is not supported/);
  });

  it('error continue lets later runs proceed', async () => {
    const out = await runText(`${BASE}
fix             h all halt 3 v_t > 0.0105 error continue message no
run             10
run             2
print           "continued step=$(step)"
`);
    expect(out.error).toBeNull();
    expect(out.logs).toContain('continued step=5');
  });
});

describe('fix hyper/global and the hyper command', () => {
  it('checks the fix hyper/local arguments and refuses its diagnostic keywords', async () => {
    expect((await runText(`${HYPER_BASE}\nfix HL mobile hyper/local 1.5 0.3 0.5 1.0 0.5\n`)).error).toMatch(/Illegal fix hyper\/local command/);
    expect((await runText(`${HYPER_BASE}\nfix HL mobile hyper/local 1.5 0.3 0.5 1.0 3.0 0.5 0.5\n`)).error).toMatch(/Illegal fix hyper\/local command/);
    expect((await runText(`${HYPER_BASE}\nfix HL mobile hyper/local 1.5 0.3 0.5 1.0 3.0 0.5 1.2 check/ghost\n`)).error).toMatch(/keyword check\/ghost is not supported/);
    expect((await runText(`${HYPER_BASE}\nfix HL mobile hyper/local 1.5 0.3 0.5 1.0 3.0 0.5 1.2 bogus\n`)).error).toMatch(/Illegal fix hyper\/local command/);
  });

  it('accepts fix hyper/local in the hyper command (bond list, bias and the 26 vector entries)', async () => {
    const out = await runText(`${HYPER_BASE}\nfix HL mobile hyper/local 1.5 0.3 0.5 1.0 3.0 0.5 1.2\nthermo_style custom step f_HL f_HL[1] f_HL[26]\ncompute ev all event/displace 0.5\nhyper 40 20 HL ev min 1.0e-5 1.0e-5 100 100\n`);
    expect(out.error).toBeNull();
    expect(out.logs.join('\n')).toMatch(/Final hyper stats/);
  });

  it('checks the fix arguments', async () => {
    expect((await runText(`${HYPER_BASE}\nfix HG mobile hyper/global 1.5 0.3 0.5\n`)).error).toMatch(/Illegal fix hyper\/global command/);
    expect((await runText(`${HYPER_BASE}\nfix HG mobile hyper/global -1.0 0.3 0.5 1.0\n`)).error).toMatch(/Illegal fix hyper\/global command/);
    expect((await runText(`${HYPER_BASE}\nfix HG mobile hyper/global 1.5 0.3 0.5 0.0\n`)).error).toMatch(/Illegal fix hyper\/global command/);
  });

  it('refuses the hyper command arguments it cannot run', async () => {
    const setup = `${HYPER_BASE}\ncompute ev all event/displace 0.5\nfix HG mobile hyper/global 1.5 0.3 0.5 1.0\n`;
    expect((await runText(`${setup}\nhyper 20 6 HG ev\n`)).error).toMatch(/N must be a multiple of Nevent/);
    expect((await runText(`${setup}\nhyper 0 0 HG ev\n`)).error).toMatch(/N must be > 0/);
    expect((await runText(`${setup}\nhyper 20 0 HG ev\n`)).error).toMatch(/Nevent must be >= 1/);
    expect((await runText(`${setup}\nhyper 20 20 HG ev rebond 7\n`)).error).toMatch(/Nrebond must be a multiple of Nevent/);
    expect((await runText(`${setup}\nhyper 20 20 HG ev dump nope\n`)).error).toMatch(/dump nope does not exist/);
    expect((await runText(`${setup}\nhyper 20 20 HG ev color\n`)).error).toContain('Illegal hyper command');
    expect((await runText(`${setup}\nhyper 20 20 HG nope\n`)).error).toMatch(/compute nope does not exist/);
    expect((await runText(`${setup}\ncompute c2 all temp\nhyper 20 20 HG c2\n`)).error).toMatch(/is not an event computer/);
    expect((await runText(`${HYPER_BASE}\ncompute ev all event/displace 0.5\nfix HG mobile nve\nhyper 20 20 HG ev\n`)).error).toMatch(/is not a bond-boost bias fix/);
  });

  it('runs a short hyper run: thermo rows with the bias vector, hyper stats in the log', async () => {
    const out = await runText(`${HYPER_BASE}
compute         ev all event/displace 0.5
fix             HG mobile hyper/global 1.5 0.3 0.5 1.0
thermo_style    custom step pe f_HG f_HG[1] f_HG[11] f_HG[12]
thermo_modify   format float %.15g
thermo          50
min_modify      line backtrack
hyper           40 20 HG ev min 1.0e-5 1.0e-5 100 100
`);
    expect(out.error).toBeNull();
    expect(out.logs.some((l) => l.startsWith('Final hyper stats'))).toBe(true);
    const last = out.thermo[out.thermo.length - 1];
    expect(last['f_HG[1]']).toBeGreaterThan(0);
    expect(Number.isFinite(last['f_HG[11]'])).toBe(true);
  });
});
