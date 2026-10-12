import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { jacobi3, exyzToQ, qToExyz, quatToMat, noSquishRotate, matvec } from '../src/engine-gpl/rigid/math_extra';

/* fix rigid/nvt (src/engine-gpl/rigid/fix_rigid_nh.ts, a port of src/RIGID/fix_rigid_nh.cpp): the math helpers
 * ported from math_extra and math_eigen, the constructor checks, and the named refusals of what is not ported.
 * Dynamics vs native LAMMPS: oracle cases w56rnh_* (tests/engineOracle.test.ts -t w56rnh). */

const SETUP = `units lj
atom_style molecular
lattice sc 1.5
region box block 0 3 0 3 0 3
create_box 1 box
create_atoms 1 box
mass 1 1.0
set atom 1*14 mol 1
set atom 15*27 mol 2
velocity all create 1.0 4928 loop geom
pair_style lj/cut 2.5
pair_coeff 1 1 1.0 1.0
timestep 0.002
`;

/** Runs SETUP plus the given lines; returns the thermo rows and the first error message (or null). */
const run = async (lines: string) => {
  const events: EngineEvent[] = [];
  const session = new Session({ emit: (e) => events.push(e) });
  let error: string | null = null;
  try {
    await session.execute(`${SETUP}${lines}`);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  }
  const err = events.find((ev) => ev.kind === 'error');
  if (!error && err && 'message' in err) error = String(err.message);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row);
  return { rows, error };
};

const norm = (v: number[]) => Math.hypot(...v);

describe('math_extra and math_eigen ports', () => {
  it('jacobi3 gives orthonormal eigenvectors in ascending order that diagonalise the matrix', () => {
    const A = [[4.0, 1.0, -0.5], [1.0, 2.5, 0.3], [-0.5, 0.3, 1.2]];
    const { eval: ev, evec } = jacobi3(A);
    expect(ev[0] <= ev[1] && ev[1] <= ev[2]).toBe(true);
    for (let c = 0; c < 3; c++) {
      const v = [evec[0][c], evec[1][c], evec[2][c]];
      expect(norm(v)).toBeCloseTo(1, 12);
      for (let r = 0; r < 3; r++) {
        const Av = A[r][0] * v[0] + A[r][1] * v[1] + A[r][2] * v[2];
        expect(Av).toBeCloseTo(ev[c] * v[r], 10);
      }
    }
  });

  it('exyz_to_q, q_to_exyz and quat_to_mat agree on a rotation', () => {
    // rotation by 0.7 rad about (1, 2, 2)/3 composed from a unit quaternion
    const ang = 0.7, ax = [1 / 3, 2 / 3, 2 / 3];
    const q = [Math.cos(ang / 2), ...ax.map((a) => a * Math.sin(ang / 2))];
    const { ex, ey, ez } = qToExyz(q);
    const q2 = exyzToQ(ex, ey, ez);
    const sign = Math.sign(q2[0] * q[0] + q2[1] * q[1] + q2[2] * q[2] + q2[3] * q[3]);
    q2.forEach((x, k) => expect(sign * x).toBeCloseTo(q[k], 12));
    const R = quatToMat(q);
    for (let r = 0; r < 3; r++) {
      expect(R[r][0]).toBeCloseTo(ex[r], 12);
      expect(R[r][1]).toBeCloseTo(ey[r], 12);
      expect(R[r][2]).toBeCloseTo(ez[r], 12);
    }
    // the columns are orthonormal
    expect(norm(ex)).toBeCloseTo(1, 12);
    expect(ex[0] * ey[0] + ex[1] * ey[1] + ex[2] * ey[2]).toBeCloseTo(0, 12);
  });

  it('no_squish_rotate keeps the quaternion normalised and matvec maps columns', () => {
    const q = [0.8, 0.36, -0.24, 0.48];
    const p = [0.3, -0.2, 0.5, 0.1];
    const I = [0.5, 1.2, 0.8];
    const n0 = Math.hypot(...q);
    for (const k of [1, 2, 3]) noSquishRotate(k, p, q, I, 0.01);
    // each sub-step rotates q by an angle, so its norm is unchanged (the permutation kq is orthogonal to q)
    expect(Math.hypot(...q)).toBeCloseTo(n0, 12);
    const out = [0, 0, 0];
    matvec([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], out);
    expect(out).toEqual([1, 2, 3]);
  });
});

describe('fix rigid/nvt', () => {
  it('runs with molecule bodies and reports the scalar', async () => {
    const r = await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.5\nthermo_style custom step temp f_1\nthermo 5\nrun 10');
    expect(r.error).toBeNull();
    expect(r.rows.length).toBeGreaterThan(1);
    for (const row of r.rows) expect(Number.isFinite(row.f_1 as number)).toBe(true);
  });

  it('runs with a single body and with group bodies', async () => {
    const single = await run('group body id 1:14\ngroup fluid subtract all body\nfix 1 body rigid/nvt single temp 1.0 1.0 0.5\nfix 2 fluid nve\nthermo 5\nrun 10');
    expect(single.error).toBeNull();
    const grp = await run('group g1 id 1:14\ngroup g2 id 15:27\nfix 1 all rigid/nvt group 2 g1 g2 temp 1.0 1.0 0.5\nthermo 5\nrun 10');
    expect(grp.error).toBeNull();
  });

  it('refuses the pressure keywords of rigid/nvt and names what else is not ported', async () => {
    expect((await run('fix 1 all rigid/nvt molecule iso 1.0 1.0 1.0 temp 1.0 1.0 0.5\nrun 1')).error).toMatch(/pressure keywords \(iso, aniso, x, y, z\) are not ported; use rigid\/npt/);
    expect((await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.5 langevin 1 1 1 5\nrun 1')).error).toMatch(/keyword 'langevin' is not supported/);
    expect((await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.5 mol foo\nrun 1')).error).toMatch(/keyword 'mol' is not supported/);
    expect((await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.5 infile foo.txt\nrun 1')).error).toMatch(/keyword 'infile' is not supported/);
  });

  it('runs the barostat styles rigid/npt and rigid/nph and their /small variants', async () => {
    const npt = await run('fix 1 all rigid/npt molecule temp 1.0 1.0 0.5 iso 1.0 1.0 2.0\nthermo_style custom step temp press f_1 vol\nthermo 5\nrun 10');
    expect(npt.error).toBeNull();
    expect(npt.rows.length).toBeGreaterThan(1);
    const nph = await run('fix 1 all rigid/nph molecule x 0.5 0.5 1.0 y 0.5 0.5 1.0 z 0.5 0.5 1.0 couple xy\nthermo 5\nrun 10');
    expect(nph.error).toBeNull();
    expect((await run('fix 1 all rigid/npt/small molecule temp 1.0 1.0 0.5 iso 1.0 1.0 2.0\nthermo 5\nrun 10')).error).toBeNull();
    expect((await run('fix 1 all rigid/nph/small molecule iso 0.5 0.5 1.0\nthermo 5\nrun 10')).error).toBeNull();
  });

  it('checks the barostat settings of the source constructors', async () => {
    expect((await run('fix 1 all rigid/nph molecule temp 1.0 1.0 0.5 iso 1.0 1.0 2.0\nrun 1')).error).toMatch(/Cannot set temperature for fix rigid\/nph/);
    expect((await run('fix 1 all rigid/npt molecule temp 1.0 1.0 0.5\nrun 1')).error).toMatch(/Did not set temperature or pressure for fix rigid\/npt/);
    expect((await run('fix 1 all rigid/npt molecule temp 1.0 1.0 0.5 couple xy x 1 1 1 y 2 2 1\nrun 1')).error).toMatch(/Invalid fix rigid\/npt command pressure settings/);
    expect((await run('fix 1 all rigid/nph molecule iso 1.0 1.0 0.0\nrun 1')).error).toMatch(/period must be > 0.0/);
  });

  it('refuses what the /small styles do not take (bodystyle single or group, force and torque flags)', async () => {
    expect((await run('fix 1 all rigid/nvt/small single temp 1.0 1.0 0.5\nrun 1')).error).toMatch(/bodystyle single is only allowed for the rigid styles/);
    expect((await run('fix 1 all rigid/nvt/small group 1 all temp 1.0 1.0 0.5\nrun 1')).error).toMatch(/bodystyle group is only allowed for the rigid styles/);
    expect((await run('fix 1 all rigid/nvt/small molecule temp 1.0 1.0 0.5 force 1 off off off\nrun 1')).error).toMatch(/force and torque keywords/);
  });

  it('checks the temperature settings like the source constructor', async () => {
    expect((await run('fix 1 all rigid/nvt molecule\nrun 1')).error).toMatch(/Did not set temperature for fix rigid\/nvt/);
    expect((await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.5 tparam 3 1 4\nrun 1')).error).toMatch(/temperature order must be 3 or 5/);
    expect((await run('fix 1 all rigid/nvt molecule temp 1.0 1.0 0.0\nrun 1')).error).toMatch(/damping parameters must be > 0.0/);
  });

});
