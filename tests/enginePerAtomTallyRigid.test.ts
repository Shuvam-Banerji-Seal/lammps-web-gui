import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The fix rigid family (src/engine/fix/rigid.ts for rigid, rigid/small, rigid/nve and rigid/nve/small; src/engine-gpl/rigid/
 * for rigid/nvt, rigid/npt, rigid/nph and their /small variants): the per-atom virial of each body atom is the atom's own
 * constraint-force term, as native LAMMPS does (src/RIGID/fix_rigid.cpp and fix_rigid_small.cpp set_xv/set_v at b332f28:
 * v_tally(1, &i, 1.0, vr)). tests/perAtomTally.ts checks A (pe/atom sums to pe) and B (stress/atom sums to the virial).
 * Native LAMMPS passes both checks on every case listed here; before the fix stress/atom threw for these styles.
 */

const CASES = [
  'rigid_force_torque', 'rigid_molecule', 'rigid_nve_molecule', 'w148rg_deposit_nvt',
  'w148rg_dipole', 'w148rg_spheres', 'w178rd_rigid_custom', 'w178rd_rigid_small_mol',
  'w179rg_nh_deform', 'w179rg_srd_line_deform', 'w230re_rigid_nve_erotate', 'w230re_rigid_nvt_small_erotate',
  'w230re_rigid_small_local', 'w251rg_nve_small', 'w26depmol_example_rigid', 'w36rigid_lgv2d',
  'w37grav_rigid', 'w37grav_rigidsmall', 'w37rigid_temp2d', 'w38gcr_one_full',
  'w38gcr_one_ins', 'w38gcr_one_last', 'w38gcr_one_nodyn', 'w42gap_rigid_velocity',
  'w42rfs_sphere_spin', 'w42rfs_wall_gran', 'w56rnh_nph_couple', 'w56rnh_nph_iso',
  'w56rnh_nph_small', 'w56rnh_npt_aniso', 'w56rnh_npt_iso', 'w56rnh_npt_small',
  'w56rnh_nvt_mol', 'w56rnh_nvt_mol_tparam', 'w56rnh_nvt_single', 'w56rnh_nvt_small',
  'w66units_rigidgrav_metal', 'w66units_rigidgrav_real', 'w66units_rigidgrav_si', 'w6rigid_bodyforces',
  'w6rigid_bodyforces_late', 'w6rigid_custom', 'w6rigid_infile', 'w72drude_swm4_nh',
  'w79rigid_tri', 'w79srd_dimer', 'w79srd_line', 'w89fix_rigid_langevin',
  'w92rig2d_box', 'w92rig2d_pour', 'w92rig2d_sphere_single',
];

describe('fix rigid family: per-atom virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
