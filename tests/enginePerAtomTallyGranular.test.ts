import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The granular pair styles (src/engine/force/pair/gran.ts, granular.ts, src/engine-gpl/superellipsoid/ and
 * src/engine-gpl/gransurf/): the per-atom virial of each contact (tests/perAtomTally.ts check B; granular contacts have
 * no energy). Native LAMMPS passes both checks on these cases; before the fix stress/atom summed to 0. The hybrid
 * granular surf/granular cases (w106, w154, w261) also check the global virial, which counts a sphere-surface contact
 * from the contact point as native does (w261gs_screw_press compares the pressure with native).
 */

const CASES = [
  'w106gs_tri_surf', 'w127gsg_line', 'w133mf_viscous_sphere_plain', 'w133mf_viscous_sphere_type',
  'w133mf_viscous_sphere_var', 'w13gran_restitution', 'w154gs_screw_hertz', 'w15gran_mdr_many', 'w15gran_mdr_pair',
  'w169md_tableting', 'w169md_triangle', 'w169md_triaxial', 'w169md_unequal_pair', 'w172pl_hooke', 'w172pl_triaxial',
  'w174mx_pn_gran_hooke_history', 'w183md_tableting_split', 'w228gs_addtorque', 'w228gs_contact_atom',
  'w228gs_cundall', 'w228gs_cundall_var', 'w228gs_viscous_nonlinear', 'w38heat_pour', 'w38heat_two_radius',
  'w38heat_wall_var', 'w45gm_example_sync_verlet', 'w45gm_sync_hertz_steps', 'w4gran_freeze', 'w4gran_hertz_history',
  'w4gran_hooke', 'w4gran_hooke_history', 'w4gran_limit', 'w64surf_line_rest', 'w64surf_tri_rest', 'w6granular_dmt',
  'w6granular_freeze', 'w6granular_hertz', 'w6granular_hooke', 'w6granular_jkr', 'w6wgran_freeze', 'w6wgran_hertz',
  'w6wgran_history', 'w6wgran_hooke', 'w7nhs_gran', 'w80superell_bowling', 'w80superell_drop', 'w261gs_screw_press',
];

describe('granular pair styles: per-atom virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
