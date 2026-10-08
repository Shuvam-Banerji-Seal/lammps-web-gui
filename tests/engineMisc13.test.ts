import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import type { FixAveGrid } from '../src/engine/fix/ave_grid';
import type { FixController } from '../src/engine/fix/controller';
import type { ComputeEventDisplace } from '../src/engine/compute/event_displace';

/*
 * Wave 13: compute event/displace, fix controller, fix ave/grid.
 * Oracle parity is in tests/oracle/w13misc_*.in (engineOracle.test.ts). Here:
 * argument errors and analytic checks. Values marked "native" were measured
 * with native LAMMPS as a black box (see the comments in the sources).
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev) });
  await session.execute(text);
  return session;
};

const close = (got: number, want: number, rel = 1e-9, abs = 1e-12) =>
  Math.abs(got - want) <= abs + rel * Math.max(Math.abs(got), Math.abs(want));

/** One atom at rest, pair style zero: a fix sees the positions and velocities it is given. */
const ONE_ATOM = `
units lj
atom_style atomic
region box block -5 5 -5 5 -5 5
create_box 1 box
create_atoms 1 single 0 0 0 units box
mass 1 1.0
pair_style zero 2.5
pair_coeff * *
`;

/** Four atoms in a 4 x 4 x 4 box, grid 2 x 2 x 2 (cell size 2): the layout of the native measurements. */
const FOUR_ATOMS = `
units lj
atom_style atomic
region box block 0 4 0 4 0 4
create_box 2 box
create_atoms 1 single 0.5 0.5 0.5 units box
create_atoms 2 single 1.5 0.5 0.5 units box
create_atoms 1 single 2.5 2.5 2.5 units box
create_atoms 2 single 0.5 2.5 0.5 units box
mass 1 1.0
mass 2 2.0
set atom 1 vx 1.0
set atom 2 vx 3.0
set atom 3 vx 5.0
set atom 4 vx -1.0
pair_style zero 2.5
pair_coeff * *
`;

describe('compute event/displace', () => {
  it('needs exactly one threshold argument', async () => {
    await expect(runScript(`${ONE_ATOM}\ncompute e all event/displace`)).rejects.toThrow('expected one argument (threshold)');
    await expect(runScript(`${ONE_ATOM}\ncompute e all event/displace 0.5 0.5`)).rejects.toThrow('expected one argument (threshold)');
  });

  it('rejects a threshold that is not a positive number', async () => {
    await expect(runScript(`${ONE_ATOM}\ncompute e all event/displace 0`)).rejects.toThrow('Distance must be > 0 for compute event/displace');
    await expect(runScript(`${ONE_ATOM}\ncompute e all event/displace -1`)).rejects.toThrow('Distance must be > 0 for compute event/displace');
    await expect(runScript(`${ONE_ATOM}\ncompute e all event/displace abc`)).rejects.toThrow(/expected a number/);
  });

  it('reports the flag as 0 in plain runs (native: 0 in every run tried)', async () => {
    const s = await runScript(`${ONE_ATOM}
velocity all set 200 0 0
timestep 0.005
fix 1 all nve
compute e all event/displace 0.5
thermo_style custom step c_e
run 5`);
    const c = s.sys.compute('e') as ComputeEventDisplace;
    expect(c.threshold).toBe(0.5);
    expect(c.scalarValue()).toBe(0);
  });
});

describe('fix controller', () => {
  it('checks its argument count and Nevery', async () => {
    await expect(runScript(`${ONE_ATOM}\nvariable c internal 0\nfix f all controller 1 1 1 1 1 v_c 0.5`)).rejects.toThrow('expected Nevery');
    await expect(runScript(`${ONE_ATOM}\nvariable c internal 0\nfix f all controller 0 1 1 1 1 v_c 0.5 c`)).rejects.toThrow('Nevery must be a positive integer');
  });

  it('checks the process variable reference', async () => {
    await expect(runScript(`${ONE_ATOM}\nvariable c internal 0\nfix f all controller 1 1 1 1 1 x_c 0.5 c`)).rejects.toThrow('pvar must be c_ID, c_ID[I], f_ID, f_ID[I] or v_name');
  });

  it('requires an internal-style control variable', async () => {
    await expect(runScript(`${ONE_ATOM}\nvariable c equal 0\nvariable p equal 1\nfix f all controller 1 1 1 1 1 v_p 0.5 c\nrun 1`))
      .rejects.toThrow('variable c is not internal-style');
    await expect(runScript(`${ONE_ATOM}\nvariable p equal 1\nfix f all controller 1 1 1 1 1 v_p 0.5 nothere\nrun 1`))
      .rejects.toThrow('variable nothere does not exist');
  });

  it('applies the discrete proportional law c_n = c_(n-1) - alpha Kp tau e_n (tau = Nevery dt)', async () => {
    // pvar = 2 c, setpoint 1, alpha 1, Kp 1, Ki = Kd = 0, dt 0.1, Nevery 1:
    // c1 = 0 - 0.1 (2*0 - 1) = 0.1 ; c2 = 0.1 - 0.1 (2*0.1 - 1) = 0.18 ; P term of step 2 = 0.08
    const s = await runScript(`${ONE_ATOM}
timestep 0.1
variable c internal 0
variable p equal 2*v_c
fix ctl all controller 1 1 1 0 0 v_p 1 c
run 2`);
    expect(s.sys.vars.get('c')?.num).toBeCloseTo(0.18, 12);
    const f = s.sys.fix('ctl') as FixController;
    expect(f.computeVector(0)).toBeCloseTo(0.08, 12);
    expect(f.computeVector(1)).toBeCloseTo(0, 12);
    expect(f.computeVector(2)).toBeCloseTo(0, 12);
  });

  it('applies the integral term over the sum of all errors and the derivative on the second update', async () => {
    // Ki = 1, Kp = Kd = 0, alpha 1, tau = 0.1: e1 = -1, I1 = -1 * 0.01 * (-1) = 0.01, c1 = 0.01.
    // e2 = 2 * 0.01 - 1 = -0.98, sum e = -1.98, I2 = 0.0198, c2 = 0.0298.
    const s = await runScript(`${ONE_ATOM}
timestep 0.1
variable c internal 0
variable p equal 2*v_c
fix ctl all controller 1 1 0 1 0 v_p 1 c
run 2`);
    expect(s.sys.vars.get('c')?.num).toBeCloseTo(0.0298, 12);
    expect((s.sys.fix('ctl') as FixController).computeVector(1)).toBeCloseTo(0.0198, 12);
  });

  it('updates only on multiples of Nevery (tau = 2 dt)', async () => {
    // Kp = 1, alpha 1, dt 0.1, Nevery 2: update at step 2 gives c = 0.2 (e = -1);
    // update at step 4: e = 2*0.2 - 1 = -0.6, c = 0.2 + 0.2 * 0.6 = 0.32.
    const s = await runScript(`${ONE_ATOM}
timestep 0.1
variable c internal 0
variable p equal 2*v_c
fix ctl all controller 2 1 1 0 0 v_p 1 c
run 3`);
    expect(s.sys.vars.get('c')?.num).toBeCloseTo(0.2, 12);
    await s.execute('run 1');
    expect(s.sys.vars.get('c')?.num).toBeCloseTo(0.32, 12);
  });
});

describe('fix ave/grid', () => {
  it('checks its arguments', async () => {
    const base = `${ONE_ATOM}\n`;
    await expect(runScript(`${base}fix a all ave/grid 2 1 3 2 2 2 vx`)).rejects.toThrow('Inconsistent nevery/nrepeat/pergrid_nfreq values');
    await expect(runScript(`${base}fix a all ave/grid 1 3 2 2 2 2 vx`)).rejects.toThrow('Inconsistent nevery/nrepeat/pergrid_nfreq values');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 0 2 2 vx`)).rejects.toThrow('Nx must be an integer >= 1');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2`)).rejects.toThrow('missing argument');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 xyz`)).rejects.toThrow("invalid input value 'xyz'");
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx bias tb`)).rejects.toThrow('keyword bias is not supported');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx norm bogus`)).rejects.toThrow('norm must be all, sample or none');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx ave window 0`)).rejects.toThrow('ave window M must be an integer >= 1');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx discard maybe`)).rejects.toThrow('discard must be yes or no');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx foo 1`)).rejects.toThrow("invalid input value 'foo'");
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 c_nope:grid:data`)).rejects.toThrow('per-grid input');
    await expect(runScript(`${base}fix a all ave/grid 1 1 1 2 2 2 vx`)).resolves.toBeDefined();
  });

  it('rejects Nz other than 1 in a 2d system', async () => {
    await expect(runScript(`units lj\natom_style atomic\ndimension 2\nregion box block 0 4 0 4 -0.5 0.5\ncreate_box 1 box\ncreate_atoms 1 single 1 1 0 units box\nmass 1 1\npair_style zero 2.5\npair_coeff * *\nfix a all ave/grid 1 1 1 2 2 2 vx`))
      .rejects.toThrow('for 2d simulations Nz must be 1');
  });

  it('has no global output, so f_ID[i][j] is refused', async () => {
    await expect(runScript(`${FOUR_ATOMS}\nfix a all ave/grid 1 1 1 2 2 2 vx\nvariable q equal f_a[1][1]\nprint "\${q}"`)).rejects.toThrow();
  });

  it('averages per-atom values in each cell (native layout: cell 0 = atoms 1,2; cell 2 = atom 4; cell 7 = atom 3)', async () => {
    const s = await runScript(`${FOUR_ATOMS}
fix ag all ave/grid 1 1 1 2 2 2 vx density/number density/mass mass temp
run 0`);
    const g = s.sys.fix('ag') as FixAveGrid;
    // columns: vx, density/number, density/mass, mass, temp; cell index x fastest
    expect(g.gridCount(0)).toBe(2);
    expect(close(g.gridValue(0, 0), 2)).toBe(true);          // (1 + 3) / 2
    expect(close(g.gridValue(0, 1), 0.25)).toBe(true);       // 2 atoms / volume 8
    expect(close(g.gridValue(0, 2), 0.375)).toBe(true);      // (1 + 2) / 8
    expect(close(g.gridValue(0, 3), 1.5)).toBe(true);        // (1 + 2) / 2
    expect(close(g.gridValue(0, 4), 19 / 6)).toBe(true);     // 2 KE / (3 * 2 atoms), KE = 0.5 + 9
    expect(close(g.gridValue(2, 0), -1)).toBe(true);
    expect(close(g.gridValue(2, 3), 2)).toBe(true);
    expect(close(g.gridValue(2, 4), 2 / 3)).toBe(true);      // KE = 0.5 * 2 * 1 = 1
    expect(close(g.gridValue(7, 0), 5)).toBe(true);
    expect(close(g.gridValue(7, 4), 25 / 3)).toBe(true);     // KE = 12.5
    // an empty cell is 0 in every column
    for (let j = 0; j < 5; j++) expect(g.gridValue(1, j)).toBe(0);
    expect(g.gridCount(1)).toBe(0);
  });

  it('norm all / sample / none over Nrepeat = 3 samples (native values)', async () => {
    // atom 4 starts at x = 1.2 with vx = 100 and crosses into cell 3 after step 1 (samples at steps 1, 2, 3)
    const moving = FOUR_ATOMS.replace('create_atoms 2 single 0.5 2.5 0.5 units box', 'create_atoms 2 single 1.2 2.5 0.5 units box')
      .replace('set atom 4 vx -1.0', 'set atom 4 vx 100.0');
    const s = await runScript(`${moving}
fix nv all nve
fix aa all ave/grid 1 3 3 2 2 2 vx density/number mass temp norm all
fix as all ave/grid 1 3 3 2 2 2 vx density/number mass temp norm sample
fix an all ave/grid 1 3 3 2 2 2 vx density/number mass temp norm none
timestep 0.005
run 3`);
    // Measured with native LAMMPS (black box), output at step 3, cells 3 (index 2) and 4 (index 3):
    // norm all:    cell 3: vx 100, density 0.0416667, mass 2, temp 6666.67, count 0.333333
    //              cell 4: vx 100, density 0.0833333, mass 2, temp 6666.67, count 0.666667
    // norm sample: cell 3: vx 33.3333, density 0.0416667, mass 0.666667, temp 2222.22
    //              cell 4: vx 66.6667, density 0.0833333, mass 1.33333, temp 4444.44
    // norm none:   cell 3: vx 33.3333, density 0.0416667, mass 0.666667, temp 6666.67
    //              cell 4: vx 66.6667, density 0.0833333, mass 1.33333, temp 6666.67
    const rows: [string, number, number[]][] = [
      ['aa', 2, [100, 0.0416667, 2, 6666.67, 0.333333]],
      ['aa', 3, [100, 0.0833333, 2, 6666.67, 0.666667]],
      ['as', 2, [33.3333, 0.0416667, 0.666667, 2222.22, 0.333333]],
      ['as', 3, [66.6667, 0.0833333, 1.33333, 4444.44, 0.666667]],
      ['an', 2, [33.3333, 0.0416667, 0.666667, 6666.67, 0.333333]],
      ['an', 3, [66.6667, 0.0833333, 1.33333, 6666.67, 0.666667]],
    ];
    for (const [name, cell, want] of rows) {
      const g = s.sys.fix(name) as FixAveGrid;
      want.forEach((w, j) => {
        const got = j < 4 ? g.gridValue(cell, j) : g.gridCount(cell);
        expect(close(got, w, 2e-5, 1e-9), `${name} cell ${cell} column ${j}: ${got} vs ${w}`).toBe(true);
      });
    }
  });

  it('ave one / running / window (native values for Nrepeat = 1, one output per step)', async () => {
    const moving = FOUR_ATOMS.replace('create_atoms 2 single 0.5 2.5 0.5 units box', 'create_atoms 2 single 1.2 2.5 0.5 units box')
      .replace('set atom 4 vx -1.0', 'set atom 4 vx 100.0');
    const s = await runScript(`${moving}
fix nv all nve
fix a1 all ave/grid 1 1 1 2 2 2 vx norm all ave one
fix a2 all ave/grid 1 1 1 2 2 2 vx norm all ave running
fix a3 all ave/grid 1 1 1 2 2 2 vx norm all ave window 2
timestep 0.005
run 0`);
    // Measured with native LAMMPS (black box): vx in cell 3 (index 2) and cell 4 (index 3) per step
    // ave one:     step 0: 100, 0 ; step 1: 100, 0 ; step 2: 0, 100 ; step 3: 0, 100 ; step 4: 0, 100
    // ave running: step 2: 66.6667, 33.3333 ; step 3: 50, 50 ; step 4: 40, 60
    // ave window 2: step 2: 50, 50 ; step 3: 0, 100 ; step 4: 0, 100
    const f = (name: string) => s.sys.fix(name) as FixAveGrid;
    const check = async (step: number, expected: Record<string, number[]>) => {
      if (step > 0) await s.execute('run 1');
      for (const [name, want] of Object.entries(expected)) {
        const g = f(name);
        want.forEach((w, k) => expect(close(g.gridValue(k === 0 ? 2 : 3, 0), w, 2e-5, 1e-9), `${name} step ${step} cell ${k}`).toBe(true));
      }
    };
    await check(0, { a1: [100, 0], a2: [100, 0], a3: [100, 0] });
    await check(1, { a1: [100, 0], a2: [100, 0], a3: [100, 0] });
    await check(2, { a1: [0, 100], a2: [66.6667, 33.3333], a3: [50, 50] });
    await check(3, { a1: [0, 100], a2: [50, 50], a3: [0, 100] });
    await check(4, { a1: [0, 100], a2: [40, 60], a3: [0, 100] });
  });

  it('output at step 0 is zero while the Nrepeat window is incomplete', async () => {
    const s = await runScript(`${FOUR_ATOMS}
fix b all ave/grid 1 3 3 2 2 2 vx norm all
run 0`);
    const g = s.sys.fix('b') as FixAveGrid;
    expect(g.gridValue(0, 0)).toBe(0);
    expect(g.gridCount(0)).toBe(0);
  });
});
