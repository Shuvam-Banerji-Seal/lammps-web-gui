import { describe, expect, it } from 'vitest';
import { superpose } from '../src/engine-gpl/react/fix_bond_react';

/*
 * The site fit of fix bond/react created atoms (fix_bond_react.cpp insert_atoms_setup, Superpose3D): the
 * least-squares rotation and translation that map the post-reaction template onto the reaction site.
 */
describe('superposition for created atoms', () => {
  it('recovers a known rotation and translation', () => {
    const ax = [0.3, -0.5, 0.8];
    const ang = 0.9;
    const n = Math.hypot(...ax);
    const [x, y, z] = ax.map((a) => a / n);
    const c = Math.cos(ang), s = Math.sin(ang), C = 1 - c;
    const R0 = [
      [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
      [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
      [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ];
    const T0 = [1.5, -2, 0.25];
    const mobile = [[0, 0, 0], [1.2, 0.1, -0.3], [0.2, 1.4, 0.5], [-0.7, 0.4, 1.1], [0.9, -0.8, 0.6]];
    const frozen = mobile.map((p) => [0, 1, 2].map((a) => R0[a][0] * p[0] + R0[a][1] * p[1] + R0[a][2] * p[2] + T0[a]));
    const fit = superpose(frozen, mobile);
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) expect(fit.R[a][b]).toBeCloseTo(R0[a][b], 9);
    for (let a = 0; a < 3; a++) expect(fit.T[a]).toBeCloseTo(T0[a], 9);
  });
});
