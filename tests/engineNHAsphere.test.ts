import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * fix nvt/asphere, npt/asphere, nph/asphere (src/engine/fix/nh_asphere.ts;
 * docs.lammps.org/fix_nvt_asphere.html, docs.lammps.org/fix_npt_asphere.html,
 * docs.lammps.org/fix_nph_asphere.html). Native parity (108 ellipsoids with
 * rotation active, nvt, npt iso and nph iso, 200 steps each) is in
 * tests/oracle/w19nhasphere_*.in. The checks here are the argument and rule
 * refusals measured with native LAMMPS (black box), the temperature compute the
 * fix creates (temp/asphere, 6N-3 dof), and the coupling of the rotational
 * degrees of freedom to the thermostat.
 */

const newSession = () => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  return session;
};

/** 32 fcc ellipsoids (half-axes 1, 1.4, 1.8 from diameters), density 0.8, a non-trivial angular momentum on each. */
const fluid = (extra = '') => `units lj
atom_style ellipsoid
lattice fcc 0.7
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
set group all shape 1.0 1.4 1.8
set group all density 0.8
set group all quat 0.3 1.0 0.5 47
set group all angmom 0.3 -0.2 0.5
${extra}pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
velocity all create 1.0 4928
timestep 0.002
`;

const run = async (text: string): Promise<string | null> => {
  const session = newSession();
  try {
    await session.execute(text);
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
};

describe('fix nvt/npt/nph asphere rules', () => {
  it('refuses update dipole, naming it', async () => {
    const err = await run(`${fluid()}fix 1 all nvt/asphere temp 1 1 0.5 update dipole\nrun 1\n`);
    expect(err).toMatch(/update dipole/);
    expect(err).toMatch(/ellipsoid/);
  });

  it('accepts disc in 3d and gives the same trajectory as without it', async () => {
    const a = newSession();
    await a.execute(`${fluid()}fix 1 all nvt/asphere temp 1 1 0.5 disc\nrun 20\n`);
    const b = newSession();
    await b.execute(`${fluid()}fix 1 all nvt/asphere temp 1 1 0.5\nrun 20\n`);
    // measured with native LAMMPS (black box): identical thermo rows with and without disc over 20 steps
    expect(Array.from(a.sys.state.v)).toEqual(Array.from(b.sys.state.v));
    expect(Array.from(a.sys.state.quat!)).toEqual(Array.from(b.sys.state.quat!));
  });

  it('refuses a point particle in the group, naming the extended-particle requirement', async () => {
    const err = await run(`${fluid()}set atom 1 shape 0 0 0\nfix 1 all nvt/asphere temp 1 1 0.5\nrun 1\n`);
    expect(err).toMatch(/requires all extended particles/);
  });

  it('refuses an unknown keyword by name', async () => {
    const err = await run(`${fluid()}fix 1 all nvt/asphere temp 1 1 0.5 foo 2\nrun 1\n`);
    expect(err).toMatch(/foo/);
  });

  it('refuses nvt/asphere without a temperature keyword and nph/asphere with one', async () => {
    const e1 = await run(`${fluid()}fix 1 all nvt/asphere iso 1 1 1\nrun 1\n`);
    expect(e1).toMatch(/nvt/);
    const e2 = await run(`${fluid()}fix 1 all nph/asphere temp 1 1 0.5 iso 1 1 1\nrun 1\n`);
    expect(e2).toMatch(/nph/);
  });

  it('refuses a style that is not ellipsoid', async () => {
    const err = await run(`units lj
atom_style sphere
lattice fcc 0.7
region box block 0 2 0 2 0 2
create_box 1 box
create_atoms 1 box
set group all diameter 1.0
set group all density 1.0
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0
fix 1 all nvt/asphere temp 1 1 0.5
run 1
`);
    expect(err).toMatch(/atom style ellipsoid/);
  });
});

describe('fix nvt/asphere temperature and thermostat', () => {
  it('creates temp/asphere for its group with 6N-3 degrees of freedom', async () => {
    const session = newSession();
    await session.execute(`${fluid()}fix 1 all nvt/asphere temp 1 1 0.5\nrun 1\n`);
    const c = session.sys.computes.find((x) => x.id === '1_temp');
    expect(c).toBeDefined();
    expect(c!.style).toBe('temp/asphere');
    // N = 32 ellipsoids: 6N - 3 (extra dof 3 removed, as in compute temp/asphere dof all)
    expect(c!.dof).toBe(6 * 32 - 3);
  });

  it('scales the angular momenta by the same thermostat factor as the velocities', async () => {
    // velocities created at T = 2 with target 1: the thermostat scales both v and L (torque is zero, so L changes only by scaling,
    // hence every component of every atom is multiplied by the same factor over the run)
    const session = newSession();
    await session.execute(`${fluid()}velocity all create 2.0 4928\nfix 1 all nvt/asphere temp 1 1 0.5\nrun 0\n`);
    const s = session.sys.state;
    const L0 = Array.from(s.angmom!);
    await session.execute('run 200\n');
    const L1 = Array.from(s.angmom!);
    const ratios = L0.map((x, k) => L1[k] / x);
    const r = ratios[0];
    expect(Math.abs(r - 1)).toBeGreaterThan(0.01);
    for (const q of ratios) expect(q).toBeCloseTo(r, 12);
  });

  it('leaves the angular momenta alone under nph/asphere (no thermostat, torque zero)', async () => {
    const session = newSession();
    await session.execute(`${fluid()}fix 1 all nph/asphere iso 1.0 1.0 2.0\nrun 0\n`);
    const L0 = Array.from(session.sys.state.angmom!);
    await session.execute('run 20\n');
    expect(Array.from(session.sys.state.angmom!)).toEqual(L0);
  });
});

describe('fix npt/asphere and nph/asphere run with ellipsoid rotation', () => {
  it('npt/asphere changes the box and nph/asphere runs without a thermostat', async () => {
    const npt = newSession();
    await npt.execute(`${fluid()}fix 1 all npt/asphere temp 1 1 0.5 iso 1.5 0.5 2.0\nrun 100\n`);
    const vol = npt.sys.geom.volume(3);
    expect(Number.isFinite(vol)).toBe(true);
    expect(vol).not.toBeCloseTo(32 / 0.7, 3);
    const nph = newSession();
    await nph.execute(`${fluid()}fix 1 all nph/asphere iso 1.5 0.5 2.0\nrun 100\n`);
    expect(Number.isFinite(nph.sys.geom.volume(3))).toBe(true);
  });
});
