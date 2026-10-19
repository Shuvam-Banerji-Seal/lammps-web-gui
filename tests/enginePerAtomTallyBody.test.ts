import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair body/rounded/polygon and body/rounded/polyhedron (src/engine-gpl/body/): per-atom energy and virial
 * (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both on these cases; before the fix pe/atom and
 * stress/atom summed to 0 (the pair tallied only the global energy and the fdotr virial).
 */

const CASES = [
  'w111body_pair_friction', 'w111body_squares_friction', 'w121body_cube_sphere', 'w121body_cubes_approach',
  'w121body_cubes_cutdown', 'w121body_cubes_edge', 'w121body_cubes_friction', 'w126body_cubes_nvt',
  'w126body_cubes_wall', 'w126body_squares_nph', 'w126body_squares_npt', 'w126body_squares_npt_xy',
  'w126body_wall2d', 'w152body_history_cubes', 'w152body_history_squares', 'w152body_history_wall',
  'w152body_pour_cube', 'w182bd_tilt_contact', 'w50bc_collide', 'w50bc_damped', 'w50bc_static',
];

describe('body pair styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
