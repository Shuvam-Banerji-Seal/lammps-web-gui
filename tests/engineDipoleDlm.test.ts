import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';

/*
 * update dipole/dlm of fix nve/sphere (the DLM loop ported in src/engine-gpl/dipole/dlm.ts, reached through the lazy
 * HOOKS word 'dipole/dlm'). The numbers are checked against native LAMMPS by the oracle cases w353-w357; these tests
 * check the properties of a free rotor: with no torque each DLM step is a product of exact rotations, so |mu| and
 * |omega| are kept to round-off, a dipole parallel to omega does not turn, a dipole of zero length is left alone, and
 * the dipole turns about omega at the rate |omega| up to the splitting error (the lab-frame omega of a turning dipole
 * moves by O(dt^2), as the five body-frame rotations do not commute).
 */

const script = (fix: string) => `
units           lj
atom_style      hybrid sphere dipole
region          box block -5 5 -5 5 -5 5
create_box      1 box
create_atoms    1 single 0 0 0
create_atoms    1 single 3 0 0
create_atoms    1 single -3 0 0
set             group all diameter 1.0
set             group all density 1.0
set             atom 1 dipole 1.5 0 0
set             atom 2 dipole 0 0 2
set             atom 3 dipole 0 0 0
set             group all omega 0 0 0.8
pair_style      zero 1.0
pair_coeff      * *
fix             1 all nve/sphere update ${fix}
timestep        0.01
run             100
`;

const final = async (fix: string): Promise<{ mu: number[][]; omega: number[][] }> => {
  const session = new Session({ emit: () => {}, writeFile: () => {} });
  await session.execute(script(fix));
  const s = session.sys.state;
  return {
    mu: [0, 1, 2].map((i) => [...s.mu!.slice(4 * i, 4 * i + 4)]),
    omega: [0, 1, 2].map((i) => [...s.omega!.slice(3 * i, 3 * i + 3)]),
  };
};

describe('fix nve/sphere update dipole/dlm', () => {
  it('turns a free dipole about omega, keeping |mu| and omega, and leaves parallel and zero dipoles alone', async () => {
    const { mu, omega } = await final('dipole/dlm');
    // atom 1: 1.5 x-hat turned about z by 0.8 * 1.0 (100 steps of 0.01)
    expect(Math.hypot(mu[0][0], mu[0][1], mu[0][2])).toBeCloseTo(1.5, 12);
    expect(Math.hypot(omega[0][0], omega[0][1], omega[0][2])).toBeCloseTo(0.8, 12);
    expect(Math.atan2(mu[0][1], mu[0][0])).toBeCloseTo(0.8, 4);
    expect(Math.abs(mu[0][2])).toBeLessThan(1e-5);
    expect(Math.abs(omega[0][0]) + Math.abs(omega[0][1])).toBeLessThan(1e-5);
    // atom 2: along omega, no change; atom 3: zero length, untouched (not NaN)
    expect(mu[1].slice(0, 3)).toEqual([0, 0, 2]);
    expect(mu[2].slice(0, 3)).toEqual([0, 0, 0]);
    expect(omega[1]).toEqual([0, 0, 0.8]);
    expect(omega[2]).toEqual([0, 0, 0.8]);
  });

  it('leaves a zero-length dipole alone under update dipole too, as native does', async () => {
    const { mu } = await final('dipole');
    expect(mu[2].slice(0, 3)).toEqual([0, 0, 0]);
    expect(Math.hypot(mu[0][0], mu[0][1], mu[0][2])).toBeCloseTo(1.5, 12);
  });
});
