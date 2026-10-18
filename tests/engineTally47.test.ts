import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent, ThermoRow } from '../src/engine/types';

/*
 * compute group/group, pe/tally, force/tally and stress/tally (src/engine/compute/tally.ts): the parsing and
 * refusal paths, the sum rules that the native examples in examples/PACKAGES/tally state (the tally of all
 * atoms with all atoms equals the pair energy, the group/group forces of all atoms cancel, group pairs
 * partition), and the analytic energy, force and virial of one pair. Native parity is in tests/oracle/w47ta_*.
 */

/** Runs an input and returns the first error message and the thermo rows. */
async function run(script: string) {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e), writeFile: () => {} });
  try {
    await session.execute(script);
  } catch {
    // the failure is reported as an error event below
  }
  const err = events.find((e) => e.kind === 'error');
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as ThermoRow);
  return { error: err && 'message' in err ? String(err.message) : undefined, rows };
}

/** Charged fcc crystal, two types, three molecules; thermo norm off so that extensive values are not divided by N. */
const CRYSTAL = `units lj
atom_style full
boundary p p p
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
group odd id 1:108:2
set group odd type 2
mass * 1.0
group m1 id 1:108:3
group m2 id 2:108:3
group m3 id 3:108:3
set group odd charge 0.6
group even subtract all odd
set group even charge -0.6
set group m1 mol 1
set group m2 mol 2
set group m3 mol 3
region lowz block INF INF INF INF INF 2.5 units box
group lower region lowz
group upper subtract all lower
pair_style lj/cut/coul/cut 2.5
pair_coeff * * 1.0 1.0
pair_coeff 1 2 0.9 1.0
`;

const thermoOf = async (setup: string, keys: string) => {
  const r = await run(`${CRYSTAL}${setup}\nthermo_style custom step ${keys}\nthermo_modify norm no format float %.15g\nrun 0\n`);
  expect(r.error).toBeUndefined();
  return r.rows[0] as Record<string, number>;
};

describe('compute group/group, pe/tally, force/tally, stress/tally: parsing and refusals', () => {
  it('refuses kspace yes by name', async () => {
    const r = await run(`${CRYSTAL}compute g all group/group all kspace yes\nthermo_style custom step c_g\nrun 0\n`);
    expect(r.error).toMatch(/group\/group: kspace yes is not supported/);
  });

  it('refuses unknown and malformed keywords', async () => {
    expect((await run(`${CRYSTAL}compute g odd group/group even bogus yes\nrun 0\n`)).error).toMatch(/unknown keyword 'bogus'/);
    expect((await run(`${CRYSTAL}compute g odd group/group even pair maybe\nrun 0\n`)).error).toMatch(/pair must be yes or no/);
    expect((await run(`${CRYSTAL}compute g odd group/group even molecule both\nrun 0\n`)).error).toMatch(/molecule must be off, inter or intra/);
    expect((await run(`${CRYSTAL}compute g odd group/group even pair\nrun 0\n`)).error).toMatch(/keyword pair needs a value/);
  });

  it('needs the group2-ID and an existing group', async () => {
    expect((await run(`${CRYSTAL}compute g odd group/group\nrun 0\n`)).error).toMatch(/needs the group2-ID/);
    expect((await run(`${CRYSTAL}compute g odd group/group nosuch\nrun 0\n`)).error).toBeDefined();
  });

  it('the tally computes take exactly one group2-ID and no keywords', async () => {
    expect((await run(`${CRYSTAL}compute t odd pe/tally\nrun 0\n`)).error).toMatch(/takes one group2-ID/);
    expect((await run(`${CRYSTAL}compute t odd force/tally even molecule inter\nrun 0\n`)).error).toMatch(/takes one group2-ID/);
  });

  it('does not register the tally styles that are not implemented', async () => {
    for (const style of ['heat/flux/tally', 'heat/flux/virial/tally']) {
      const r = await run(`${CRYSTAL}compute t odd ${style} even\nrun 0\n`);
      expect(r.error, style).toBeDefined();
      expect(r.error, style).toContain(style);
    }
  });

  it('refuses a pair style without single(), naming it', async () => {
    const r = await run(`units lj
atom_style atomic
lattice fcc 0.8442
region box block 0 3 0 3 0 3
create_box 2 box
create_atoms 1 box
mass * 1.0
pair_style zero 2.5
pair_coeff * *
compute g all group/group all
thermo_style custom step c_g
run 0
`);
    expect(r.error).toBeDefined();
    expect(r.error).toMatch(/needs a pair_style that supports single\(\).*pair_style zero does not/);
  });

  it('pair no gives zero energy and force', async () => {
    const v = await thermoOf('compute g odd group/group even pair no', 'c_g c_g[1] c_g[2] c_g[3]');
    expect(v['c_g']).toBe(0);
    expect(v['c_g[1]']).toBe(0);
    expect(v['c_g[3]']).toBe(0);
  });
});

describe('compute group/group, pe/tally, force/tally, stress/tally: sum rules', () => {
  it('pe/tally of all with all is the total pair energy', async () => {
    const v = await thermoOf('compute t all pe/tally all\ncompute r all reduce sum c_t[1] c_t[2]\n', 'pe c_t c_r[1] c_r[2]');
    expect(v['c_t']).toBeCloseTo(v['pe'], 10);
    expect(v['c_r[1]'] + v['c_r[2]']).toBeCloseTo(v['pe'], 10);
  });

  it('group/group of all with all is the total pair energy and its force cancels', async () => {
    const v = await thermoOf('compute g all group/group all', 'pe c_g c_g[1] c_g[2] c_g[3]');
    expect(v['c_g']).toBeCloseTo(v['pe'], 10);
    expect(Math.abs(v['c_g[1]']) + Math.abs(v['c_g[2]']) + Math.abs(v['c_g[3]'])).toBeLessThan(1e-9);
  });

  it('group/group is symmetric and Newton-third-law antisymmetric in its two groups', async () => {
    const v = await thermoOf('compute ab odd group/group even\ncompute ba even group/group odd', 'c_ab c_ab[1] c_ab[2] c_ab[3] c_ba c_ba[1] c_ba[2] c_ba[3]');
    expect(v['c_ab']).toBeCloseTo(v['c_ba'], 10);
    for (const c of [1, 2, 3]) expect(v[`c_ab[${c}]`]).toBeCloseTo(-v[`c_ba[${c}]`], 10);
  });

  it('pairs with one atom in A count once: E(A, all) = E(A, A) + E(A, B) with B the rest', async () => {
    const v = await thermoOf('compute aall odd group/group all\ncompute aa odd group/group odd\ncompute ab odd group/group even', 'c_aall c_aa c_ab');
    expect(v['c_aall']).toBeCloseTo(v['c_aa'] + v['c_ab'], 10);
  });

  it('molecule inter and intra add up to the molecule-independent pair term', async () => {
    const v = await thermoOf('compute all2 odd group/group even\ncompute mi odd group/group even molecule inter\ncompute mn odd group/group even molecule intra', 'c_all2 c_mi c_mn');
    expect(v['c_mi'] + v['c_mn']).toBeCloseTo(v['c_all2'], 10);
  });

  it('force/tally of all with all is zero in total and per atom', async () => {
    const v = await thermoOf('compute f all force/tally all\ncompute r all reduce sum c_f[1] c_f[2] c_f[3]', 'c_f c_r[1] c_r[2] c_r[3]');
    expect(Math.abs(v['c_f'])).toBeLessThan(1e-9);
    expect(Math.abs(v['c_r[1]']) + Math.abs(v['c_r[2]']) + Math.abs(v['c_r[3]'])).toBeLessThan(1e-9);
  });

  it('force/tally scalar is the magnitude of the group/group force on group1', async () => {
    const v = await thermoOf('compute f odd force/tally even\ncompute g odd group/group even\ncompute r odd reduce sum c_f[1] c_f[2] c_f[3]', 'c_f c_g[1] c_g[2] c_g[3] c_r[1] c_r[2] c_r[3]');
    const mag = Math.hypot(v['c_g[1]'], v['c_g[2]'], v['c_g[3]']);
    expect(v['c_f']).toBeCloseTo(mag, 10);
    expect(v['c_r[1]']).toBeCloseTo(v['c_g[1]'], 10);
  });

  it('stress/tally per-atom sums over all atoms are minus three times the scalar', async () => {
    const v = await thermoOf('compute s all stress/tally all\ncompute r all reduce sum c_s[1] c_s[2] c_s[3]', 'c_s c_r[1] c_r[2] c_r[3]');
    expect(v['c_s']).not.toBe(0);
    expect(v['c_r[1]'] + v['c_r[2]'] + v['c_r[3]']).toBeCloseTo(-3 * v['c_s'], 8);
  });

  it('the pe/tally and group/group scalars of one group pair agree', async () => {
    const v = await thermoOf('compute t odd pe/tally even\ncompute g odd group/group even', 'c_t c_g');
    expect(v['c_t']).toBeCloseTo(v['c_g'], 12);
  });
});

describe('compute group/group, pe/tally, force/tally, stress/tally: one pair', () => {
  // two atoms at distance 1.5 along x, LJ only (epsilon 1, sigma 1, cutoff 2.5), no bonds, no specials
  const PAIR = `units lj
atom_style full
lattice none 1.0
boundary p p p
region box block -10 10 -10 10 -10 10
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 1.5 0.0 0.0
mass * 1.0
pair_style lj/cut/coul/cut 2.5
pair_coeff * * 1.0 1.0
group g1 id 1
group g2 id 2
`;
  const r = 1.5;
  const lj = 4 * (r ** -12 - r ** -6);
  // force on atom 1 along -x: -dE/dr = 4 (12 r^-13 - 6 r^-7); the force on atom 1 (at x = 0) points towards -x
  const fmag = Math.abs(4 * (-12 * r ** -13 + 6 * r ** -7));
  const dF = 4 * (12 * r ** -12 - 6 * r ** -6);

  it('pe/tally and group/group give the LJ energy of the pair, once', async () => {
    const v = await run(`${PAIR}compute t1 g1 pe/tally g2\ncompute g g1 group/group g2\ncompute tt all pe/tally all\ncompute r1 all reduce sum c_t1[1] c_t1[2]\nthermo_style custom step pe c_t1 c_g c_tt c_r1[1] c_r1[2]\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(v.error).toBeUndefined();
    const row = v.rows[0] as Record<string, number>;
    expect(row['c_t1']).toBeCloseTo(lj, 10);
    expect(row['c_g']).toBeCloseTo(lj, 10);
    expect(row['c_tt']).toBeCloseTo(lj, 10);
    // the per-atom values are half of the pair energy on each atom; both columns are evdwl and ecoul
    expect(row['c_r1[1]']).toBeCloseTo(lj, 10);
    expect(row['c_r1[2]']).toBeCloseTo(0, 12);
  });

  it('force/tally and group/group give the force magnitude on atom 1', async () => {
    const v = await run(`${PAIR}compute f1 g1 force/tally g2\ncompute g g1 group/group g2\nthermo_style custom step c_f1 c_g[1] c_g[2] c_g[3]\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(v.error).toBeUndefined();
    const row = v.rows[0] as Record<string, number>;
    expect(row['c_f1']).toBeCloseTo(fmag, 10);
    expect(Math.abs(row['c_g[1]'])).toBeCloseTo(fmag, 10);
    expect(row['c_g[2]']).toBeCloseTo(0, 12);
  });

  it('stress/tally scalar is the virial trace over 3 (energy units, not converted)', async () => {
    const v = await run(`${PAIR}compute s1 g1 stress/tally g2\nthermo_style custom step c_s1\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(v.error).toBeUndefined();
    expect((v.rows[0] as Record<string, number>)['c_s1']).toBeCloseTo(dF / 3, 10);
  });

  it('molecule inter removes a pair of one molecule, intra keeps it', async () => {
    const mol = `set atom 1 mol 7\nset atom 2 mol 7\n`;
    const same = await run(`${PAIR}${mol}compute gi g1 group/group g2 molecule inter\ncompute gn g1 group/group g2 molecule intra\nthermo_style custom step c_gi c_gn\nthermo_modify norm no format float %.15g\nrun 0\n`);
    expect(same.error).toBeUndefined();
    expect((same.rows[0] as Record<string, number>)['c_gi']).toBeCloseTo(0, 12);
    expect((same.rows[0] as Record<string, number>)['c_gn']).toBeCloseTo(lj, 10);
  });
});
