import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * fix balance and balance command weighted-balance options (docs.lammps.org/
 * fix_balance.html, balance.html #weighted_balance). Expected values are from
 * native LAMMPS (black box, 1 process) and the matching oracle cases
 * tests/oracle/w22bal_*.in; see the measured notes in src/engine/fix/balance.ts.
 */

const run = async (text: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  await session.execute(text);
  const thermo = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { events, thermo };
};

/** Values agree within the oracle tolerance (native stores %0.15g). */
const near = (got: number[], want: number[], rel = 1e-9) => {
  expect(got.length).toBe(want.length);
  for (let i = 0; i < want.length; i++) expect(Math.abs(got[i] - want[i])).toBeLessThanOrEqual(1e-12 + rel * Math.abs(want[i]));
};

const base = `
units lj
atom_style atomic
lattice sc 1.0
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass 1 1.0
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0
neighbor 0.3 bin
neigh_modify every 1 delay 0 check no
velocity all create 1.0 87287
fix p all property/atom d_W
compute cw all property/atom d_W
compute s1 all reduce sum c_cw
compute s2 all reduce sumsq c_cw
compute s3 all reduce min c_cw
compute s4 all reduce max c_cw
`;

describe('fix balance weighted balance: var, time and store', () => {
  it('weight var stores x+2*y+1 per atom at setup and every step (native values)', async () => {
    const { thermo } = await run(`${base}variable wt atom x+2*y+1.0
fix 1 all nve
fix b all balance 1 2.0 shift x 5 1.1 weight var wt weight store W
thermo_style custom step atoms f_b c_s1 c_s2 c_s3 c_s4
thermo_modify format float %.15g norm no
thermo 1
run 4
`);
    near(thermo.map((r) => r.c_s1), [352, 440, 440, 440, 440]);
    expect(thermo[0].c_s2).toBeCloseTo(2336, 9);
    expect(thermo[1].c_s2).toBeCloseTo(3534.75903365429, 9);
    expect(thermo[1].c_s3).toBeCloseTo(1.00504726062887, 12);
    expect(thermo[1].c_s4).toBeCloseTo(12.9777657772798, 9);
  });

  it('weight time on one rank is a no-op (measured): stored weights equal the var weights', async () => {
    const { thermo } = await run(`${base}variable wt atom x+2*y+1.0
fix 1 all nve
fix b all balance 1 2.0 shift x 5 1.1 weight var wt weight time 2.0 weight store W
thermo_style custom step atoms f_b c_s1 c_s2 c_s3 c_s4
thermo_modify format float %.15g norm no
thermo 1
run 4
`);
    near(thermo.map((r) => r.c_s1), [352, 440, 440, 440, 440]);
    expect(thermo[1].c_s2).toBeCloseTo(3534.75903365429, 9);
  });

  it('weight neigh + group stores 2x group weights at setup then pairs/atoms x group weights', async () => {
    const { thermo } = await run(`${base}group g1 id 1:10
fix 1 all nve
fix b all balance 1 1.0 shift x 5 1.1 weight neigh 1.0 weight group 1 g1 2.0 weight store W
thermo_style custom step atoms f_b c_s1 c_s2 c_s3 c_s4
thermo_modify format float %.15g norm no
thermo 1
run 4
`);
    expect(thermo[0].c_s1).toBeCloseTo(74, 12);
    expect(thermo[0].c_s2).toBeCloseTo(94, 12);
    for (let r = 1; r <= 4; r++) {
      expect(thermo[r].c_s1).toBeCloseTo(962, 9);
      expect(thermo[r].c_s2).toBeCloseTo(15886, 9);
      expect(thermo[r].c_s3).toBeCloseTo(13, 12);
      expect(thermo[r].c_s4).toBeCloseTo(26, 12);
    }
  });

  it('rejects unknown styles, bad factors and a missing store vector', async () => {
    await expect(run(`${base}fix 1 all balance 1 2.0 shift x 5 1.1 weight time 0\nrun 0\n`)).rejects.toThrow(/positive/);
    await expect(run(`${base}fix 1 all balance 1 2.0 shift x 5 1.1 weight var nosuch\nrun 0\n`)).rejects.toThrow(/does not exist/);
    await expect(run(`${base}variable we equal 2.0\nfix 1 all balance 1 2.0 shift x 5 1.1 weight var we\nrun 0\n`)).rejects.toThrow(/invalid style/);
    await expect(run(`${base}fix 1 all balance 1 2.0 shift x 5 1.1 weight store nosuch\nrun 0\n`)).rejects.toThrow(/does not exist/);
    await expect(run(`${base}fix 1 all balance 1 2.0 shift x 5 1.1 weight other\nrun 0\n`)).rejects.toThrow(/not supported/);
  });
});

describe('balance command weighted balance: store', () => {
  it('weight var + group + store writes (x+2*y+1) x group weight, matching native', async () => {
    const { thermo } = await run(`${base}variable wt atom x+2*y+1.0
group g1 id 1:10
balance 1.0 shift x 5 1.1 weight var wt weight group 1 g1 3.0 weight store W
thermo_style custom step atoms c_s1 c_s2 c_s3 c_s4
thermo_modify format float %.15g norm no
run 0
`);
    expect(thermo.length).toBe(1);
    expect(thermo[0].c_s1).toBe(430);
    expect(thermo[0].c_s2).toBe(3752);
    expect(thermo[0].c_s3).toBe(1);
    expect(thermo[0].c_s4).toBe(18);
  });

  it('store snapshots the product before it (keyword order matters, matching native)', async () => {
    const { thermo } = await run(`${base}variable wt atom x+2*y+1.0
group g1 id 1:10
balance 1.0 shift x 5 1.1 weight group 1 g1 3.0 weight store W weight var wt
thermo_style custom step atoms c_s1 c_s3 c_s4
thermo_modify format float %.15g norm no
run 0
`);
    expect(thermo[0].c_s1).toBe(84);
    expect(thermo[0].c_s3).toBe(1);
    expect(thermo[0].c_s4).toBe(3);
  });
});
