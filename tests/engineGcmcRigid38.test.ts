import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * fix gcmc keyword rigid and fix_modify dynamic/dof (src/engine/fix/gcmc.ts, src/engine/fix/rigid.ts).
 * Refusals and the keyword values are checked against docs.lammps.org/fix_gcmc.html (keyword rigid:
 * "specifying as its value the ID of a separate fix rigid/small command"); dynamic/dof against
 * docs.lammps.org/fix_modify.html ("dynamic/dof value = yes or no"). The body bookkeeping (one rigid
 * body per inserted molecule, removed with its molecule) is checked through the fix's body count.
 */

const CO2 = `# CO2 molecule file. TraPPE model.

3 atoms
2 bonds
1 angles

Coords

1        0.0 0.0 0.0
2        -1.16 0.0 0.0
3        1.16 0.0 0.0

Types

1        1
2        2
3        2

Charges

1        0.7
2       -0.35
3       -0.35

Bonds

1   1      1      2
2   1      1      3

Angles

1   1      2      1      3

Special Bond Counts

1 2 0 0
2 1 1 0
3 1 1 0

Special Bonds

1 2 3
2 1 3
3 1 2
`;

const head = `units real
atom_style full
boundary p p p
pair_style lj/cut 8.0
bond_style harmonic
angle_style harmonic
lattice sc 5.0
region box block 0 10 0 10 0 10 units box
create_box 2 box bond/types 1 angle/types 1 extra/bond/per/atom 2 extra/angle/per/atom 1 extra/special/per/atom 2
molecule co2mol CO2.txt
create_atoms 0 single 2.5 2.5 2.5 units box mol co2mol 464563
pair_coeff 1 1 0.053649 2.8
pair_coeff 2 2 0.156973 3.05
bond_coeff 1 0 1.16
angle_coeff 1 0 180
mass 1 12.0107
mass 2 15.9994
group co2 type 1 2
velocity all create 338.0 54654
timestep 1.0
`;

const run = async (script: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  session.addFile('CO2.txt', CO2);
  let error: string | null = null;
  try {
    await session.execute(script);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((e) => e.kind === 'error');
  if (err && err.kind === 'error') error = err.message;
  return { error, session };
};

const rigidFix = `fix myrigid co2 rigid/small molecule mol co2mol\n`;
const gcmc = (extra: string, mu = '5.0', X = 1, N = 1000) =>
  `fix mygcmc co2 gcmc ${N} ${X} 0 0 54341 338.0 ${mu} 0.5 mol co2mol tfac_insert 1.6667 group co2 ${extra}\n`;

describe('fix gcmc keyword rigid: refusals and parsing', () => {
  it('names a missing rigid fix ID when the run starts', async () => {
    const { error } = await run(`${head}${gcmc('rigid nosuch')}run 0\n`);
    expect(error).toMatch(/fix gcmc rigid: fix ID nosuch does not exist/);
  });

  it('refuses rigid together with shake (native: Cannot use fix gcmc rigid and shake)', async () => {
    const { error } = await run(`${head}${rigidFix}${gcmc('rigid myrigid shake nosuch')}run 0\n`);
    expect(error).toMatch(/Cannot use fix gcmc rigid and shake/);
  });

  it('refuses rigid without the mol keyword (native: Cannot use fix gcmc rigid and not molecule)', async () => {
    const { error } = await run(`${head}${rigidFix}fix mygcmc co2 gcmc 1000 1 0 0 54341 338.0 5.0 0.5 group co2 rigid myrigid\nrun 0\n`);
    expect(error).toMatch(/Cannot use fix gcmc rigid and not molecule/);
  });

  it('refuses rigid with MC moves (M must be zero)', async () => {
    const { error } = await run(`${head}${rigidFix}${gcmc('rigid myrigid', '5.0', 1, 1000).replace('gcmc 1000 1 0 0', 'gcmc 1000 1 1 0')}run 0\n`);
    expect(error).toMatch(/Cannot use fix gcmc rigid with MC moves/);
  });

  it('refuses a rigid fix that is not rigid/small', async () => {
    const { error } = await run(`${head}fix other co2 nve\n${gcmc('rigid other')}run 0\n`);
    expect(error).toMatch(/fix other \(nve\) is not a fix rigid\/small/);
  });

  it('refuses a rigid/small fix whose molecule template differs from the mol keyword', async () => {
    const { error } = await run(`${head}fix myrigid co2 rigid/small molecule\n${gcmc('rigid myrigid')}run 0\n`);
    expect(error).toMatch(/Fix gcmc and fix rigid\/small not using same molecule template ID/);
  });

  it('needs a fix ID value after rigid', async () => {
    const { error } = await run(`${head}fix mygcmc co2 gcmc 1000 1 0 0 54341 338.0 5.0 0.5 mol co2mol rigid\nrun 0\n`);
    expect(error).toMatch(/keyword 'rigid' needs 1 value/);
  });
});

describe('fix_modify dynamic/dof', () => {
  it('accepts yes and no on a rigid/small fix', async () => {
    const { error, session } = await run(`${head}${rigidFix}fix_modify myrigid dynamic/dof yes\n${gcmc('rigid myrigid')}run 0\n`);
    expect(error).toBeNull();
    expect((session.sys.fix('myrigid') as unknown as { dynamicDofSet: boolean }).dynamicDofSet).toBe(true);
  });

  it('refuses a value other than yes or no', async () => {
    const { error } = await run(`${head}${rigidFix}fix_modify myrigid dynamic/dof maybe\nrun 0\n`);
    expect(error).toMatch(/fix_modify dynamic\/dof must be yes or no, got 'maybe'/);
  });
});

describe('rigid body bookkeeping with fix gcmc rigid', () => {
  // one body per molecule: the body count (the rigid fix's array rows) equals atoms / 3 after every run
  const bodies = (session: Session) => (session.sys.fix('myrigid') as unknown as { sizeArrayRows: number }).sizeArrayRows;

  it('adds one body for an accepted insertion', async () => {
    const { error, session } = await run(`${head}${rigidFix}fix_modify myrigid dynamic/dof yes\n${gcmc('rigid myrigid')}thermo_style custom step atoms\nrun 1\n`);
    expect(error).toBeNull();
    expect(session.sys.state.n).toBe(6);
    expect(bodies(session)).toBe(2);
  });

  it('removes the body of a deleted molecule and never empties the box', async () => {
    // mu -30: deletions are strongly favoured; the last remaining molecule is never deleted
    const { error, session } = await run(`${head}${rigidFix}${gcmc('rigid myrigid', '-30.0', 2, 1000)}thermo_style custom step atoms\nthermo 1\nrun 6\n`);
    expect(error).toBeNull();
    expect(session.sys.state.n).toBe(3);
    expect(bodies(session)).toBe(1);
  });

  it('keeps the body count equal to the molecule count over mixed exchanges', async () => {
    const { error, session } = await run(`${head}${rigidFix}${gcmc('rigid myrigid', '2.0', 3, 1)}thermo_style custom step atoms\nthermo 1\nrun 12\n`);
    expect(error).toBeNull();
    expect(session.sys.state.n % 3).toBe(0);
    expect(bodies(session)).toBe(session.sys.state.n / 3);
  });
});

describe('gcmc region translation rejection (fix_gcmc.rst: rejected after 1000 tries)', () => {
  it('a translation with no trial inside the region leaves the atom in place and runs to the end', async () => {
    // one atom at the sphere's centre; displace 1000 makes a trial inside the 2.5 sphere very unlikely
    const script = `units lj
atom_style atomic
pair_style lj/cut 3.0
region box block 0 20 0 20 0 20
create_box 1 box
create_atoms 1 single 10.0 10.0 10.0 units box
region sphere sphere 10.0 10.0 10.0 2.5 units box
pair_coeff * * 1.0 1.0
mass * 1.0
group gcmcgroup type 1
fix mygcmc gcmcgroup gcmc 1 0 1 1 29494 2.0 0.0 1000.0 region sphere
thermo_style custom step atoms f_mygcmc[1] f_mygcmc[2]
thermo 1
run 3
`;
    const events: EngineEvent[] = [];
    const session = new Session({ emit: (e) => events.push(e) });
    await session.execute(script);
    const logs = events.filter((e) => e.kind === 'log').map((e) => (e.kind === 'log' ? e.text : ''));
    const warnings = logs.filter((t) => t.includes('could not place a region-restricted trial move'));
    expect(warnings.length).toBeLessThanOrEqual(1);
    const counts = (session.sys.fix('mygcmc') as unknown as { computeVector(i: number): number }).computeVector(0);
    expect(counts).toBe(3);
    expect(session.sys.state.n).toBe(1);
  });
});
