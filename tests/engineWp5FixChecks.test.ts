import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * Argument, compatibility and lost-atom checks of the fix work package (WP5 of the logic audit). Each message is
 * the native one measured with LAMMPS (black box, OMP_NUM_THREADS=1) unless noted; the numbers are the same runs.
 */

const runScript = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (ev) => events.push(ev), writeFile: () => {} });
  let error: Error | null = null;
  try {
    await session.execute(text);
  } catch (e) {
    error = e as Error;
  }
  return { error, events };
};

const SYS = `
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 2 0 2 0 2
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 1.44 87287 loop geom
pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
timestep        0.005
fix             1 all nve
`;

describe('fix argument and compatibility errors (WP5)', () => {
  it('fix ave/time: a malformed value is an error of the fix line, before any run', async () => {
    // native: "ERROR: Illegal fix ave/time command" at the fix line (the wording here names the value)
    const { error } = await runScript(`${SYS}\ncompute t all temp\nfix a all ave/time 1 1 5 c_t bogus\n`);
    expect(error?.message).toContain("invalid input value 'bogus'");
    expect(error?.message).toMatch(/^line \d+:/);
  });

  it('thermo steps must be compatible with the output steps of a fix ave/time', async () => {
    // Measured with native LAMMPS (black box): "ERROR: Thermo and fix a not computed at compatible times" for thermo 2;
    // thermo 4 with Nfreq 4 runs (oracle w89fix_avetime_norm)
    const fix = `compute p all pe\nfix a all ave/time 2 2 4 c_p ave one\nthermo_style custom step pe f_a\n`;
    const bad = await runScript(`${SYS}\n${fix}thermo 2\nrun 8\n`);
    expect(bad.error?.message).toContain('Thermo and fix a not computed at compatible times');
    const good = await runScript(`${SYS}\n${fix}thermo 4\nrun 8\n`);
    expect(good.error).toBeNull();
  });

  it('fix npt refuses ptemp together with a thermostat', async () => {
    // Measured with native LAMMPS (black box): "ERROR: Thermostat in fix npt is incompatible with ptemp command"
    const bad = await runScript(`${SYS}\nfix 2 all npt temp 1.0 1.0 0.5 iso 1 1 1 ptemp 1.0\nrun 1\n`);
    expect(bad.error?.message).toContain('incompatible with ptemp command');
    const nph = await runScript(`${SYS}\nfix 2 all nph iso 1 1 1 ptemp 1.0\nrun 1\n`);
    expect(nph.error).toBeNull();
  });

  it('rigid/small refuses a body whose extent exceeds the ghost cutoff', async () => {
    // Measured with native LAMMPS (black box, lj/cut 0.5, atoms spaced 1 apart): extents 1, 1, 2, 2, 4 for chains of
    // 2, 3, 4, 5 and 8 atoms; "ERROR: Rigid body extent 4 > ghost atom cutoff - use comm_modify cutoff"
    const chain = (n: number) => {
      const atoms = Array.from({ length: n }, (_, i) => `create_atoms 1 single ${i} 0 0 units box`).join('\n');
      return `units lj
atom_style molecular
region box block -10 10 -10 10 -10 10
create_box 1 box
${atoms}
set group all mol 1
mass 1 1.0
velocity all set 0 0 0
pair_style lj/cut 0.5
pair_coeff 1 1 1.0 1.0
fix 1 all rigid/small molecule
run 1
`;
    };
    const eight = await runScript(chain(8));
    expect(eight.error?.message).toContain('Rigid body extent 4 > ghost atom cutoff - use comm_modify cutoff');
    const two = await runScript(chain(2));
    expect(two.error?.message).toContain('Rigid body extent 1 > ghost atom cutoff');
    const plain = await runScript(chain(8).replace('rigid/small', 'rigid'));
    expect(plain.error).toBeNull();
  });

  it('lost atoms through a fixed boundary: native wording and count', async () => {
    // Measured with native LAMMPS (black box, boundary f f f, 256 lj atoms at T=3, run 10): "ERROR: Lost atoms:
    // original 256 current 215" after step 6 (the same step as the engine)
    const { error } = await runScript(`units lj
atom_style atomic
boundary f f f
lattice fcc 0.8442
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
velocity all create 3.0 87287 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0 2.5
timestep 0.005
fix 1 all nve
thermo_style custom step temp pe atoms
thermo 1
run 10
`);
    expect(error?.message).toContain('Lost atoms: original 256 current 215');
  });

  it('fix store/state refuses an attribute that is not a store/state input (temperature)', async () => {
    const { error } = await runScript(`${SYS}\nfix s all store/state 0 temperature\n`);
    expect(error?.message).toContain("invalid input value 'temperature'");
  });

  it('fix store/state stores a custom d_name property (fix property/atom) at the first run', async () => {
    const session = new Session({ emit: () => {}, writeFile: () => {} });
    await session.execute(`${SYS}\nfix p all property/atom d_W\nset atom 1 d_W 2.5\nfix s all store/state 0 d_W\nrun 0\n`);
    expect((session.sys.fix('s') as unknown as { vectorAtom: Float64Array }).vectorAtom[0]).toBe(2.5);
  });

  it('fix store/state refuses mol and q when the atom style has no molecule ID or charge (native Cannot use)', async () => {
    const { error } = await runScript(`${SYS}\nfix s all store/state 0 mol\n`);
    expect(error?.message).toContain('Cannot use fix store/state mol for atom style atomic');
    const q = await runScript(`${SYS}\nfix s all store/state 0 q\n`);
    expect(q.error?.message).toContain('Cannot use fix store/state q for atom style atomic');
  });

  it('fix store/state accepts the history keyword and refuses a second thresh keyword', async () => {
    const h = await runScript(`${SYS}\nfix s all store/state 0 x history 5 10 0\n`);
    expect(h.error).toBeNull();
    const t = await runScript(`${SYS}\nvariable g equal 1\nfix s all store/state 0 x thresh v_g > 0 thresh v_g > 0\n`);
    expect(t.error?.message).toContain('the thresh keyword may be used only once');
  });
});
