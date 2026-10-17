import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix sgcmc (src/engine-gpl/mc/fix_sgcmc.ts, ported from src/MC/fix_sgcmc.cpp at b332f28). The Monte Carlo itself
 * (the random streams, the sampling windows, the EAM, atomic-energy and generic energy changes, the variance
 * constraint, the serial mode and the oversized window) is pinned by the w204sg_* oracle cases; these unit tests
 * cover the argument and setup errors, with native's messages, and that the fix does nothing at setup.
 */

const run = async (script: string): Promise<{ error: string | null; events: EngineEvent[] }> => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  return { error, events };
};

const LJ = (cells: number, ntypes = 2) => `units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 ${cells} 0 ${cells} 0 ${cells}
create_box ${ntypes} box
create_atoms 1 box
set type 1 type/fraction 2 0.3 777
mass * 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
velocity all create 1.0 87287 loop geom
fix integrate all nve
`;

describe('fix sgcmc', () => {
  it('checks its arguments as native does', async () => {
    const cases: [string, RegExp][] = [
      ['fix mc all sgcmc 10 0.2', /Illegal fix sgcmc command: missing argument/],
      ['fix mc all sgcmc 0 0.2 1.0 0.1', /Invalid number of MD timesteps 0/],
      ['fix mc all sgcmc 10 1.5 1.0 0.1', /Invalid fraction 1.5 of swap atoms/],
      ['fix mc all sgcmc 10 0.2 0 0.1', /Temperature 0 invalid/],
      ['fix mc all sgcmc 10 0.2 1.0', /Too few chemical potentials specified/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 randseed 0', /Random number seed 0 must be positive/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 window_moves 0', /Invalid number 0 of sampling window moves/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 window_size 0.3', /Sampling window size 0.3 is out of range/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 variance -1 0.2', /Variance constraint parameter must not be negative/],
      // species 1 is checked first: its target is 1 - 1.2
      ['fix mc all sgcmc 10 0.2 1.0 0.1 variance 10 1.2', /Target concentration -0.19999999999999996 for species 1 is out of range/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 atomic/energy maybe', /Expected boolean parameter instead of 'maybe'/],
      ['fix mc all sgcmc 10 0.2 1.0 0.1 bogus', /Unknown fix sgcmc keyword: bogus/],
    ];
    for (const [cmd, re] of cases) {
      const { error } = await run(`${LJ(6)}${cmd}\n`);
      expect(error, cmd).toMatch(re);
    }
  });

  it('needs two atom types, an orthogonal box, per-type masses and one fix', async () => {
    expect((await run(`${LJ(6, 1).replace('set type 1 type/fraction 2 0.3 777\n', '')}fix mc all sgcmc 10 0.2 1.0\n`)).error)
      .toMatch(/at least two atom types/);
    const tri = LJ(6).replace('region box block 0 6 0 6 0 6', 'region box prism 0 6 0 6 0 6 0.5 0 0');
    expect((await run(`${tri}fix mc all sgcmc 10 0.2 1.0 0.1\n`)).error).toMatch(/non-orthogonal simulation boxes/);
    const sphere = `units lj
atom_style sphere
lattice fcc 0.8442
region box block 0 6 0 6 0 6
create_box 2 box
create_atoms 1 box
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix integrate all nve/sphere
fix mc all sgcmc 10 0.2 1.0 0.1
run 1
`;
    expect((await run(sphere)).error).toMatch(/requires per atom type masses/);
    expect((await run(`${LJ(6)}fix mc all sgcmc 10 0.2 1.0 0.1\nfix mc2 all sgcmc 10 0.2 1.0 0.1\nrun 1\n`)).error)
      .toMatch(/More than one fix sgcmc defined/);
    expect((await run(`${LJ(6)}fix mc all sgcmc 10 0.2 1.0 0.1 atomic/energy yes\nrun 1\n`)).error)
      .toMatch(/Pair style does not support atomic energy method/);
  });

  it('refuses a box smaller than 4 cutoffs at the first event, and does no Monte Carlo at setup', async () => {
    expect((await run(`${LJ(4)}fix mc all sgcmc 5 0.2 1.0 0.1\nrun 5\n`)).error)
      .toMatch(/Per-node simulation cell is too small for fix sgcmc/);
    const { error, events } = await run(`${LJ(6)}fix mc all sgcmc 5 0.2 1.0 0.1
thermo_style custom step f_mc[1] f_mc[2] f_mc[3] f_mc[4]
run 0
`);
    expect(error).toBeNull();
    const rows = events.filter((e) => e.kind === 'thermo');
    expect(rows.length).toBeGreaterThan(0);
    const last = rows[rows.length - 1];
    expect(last.kind === 'thermo' && last.row['f_mc[1]']).toBe(0);
    expect(last.kind === 'thermo' && last.row['f_mc[2]']).toBe(0);
  });
});
