import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * angle_style table, cross, sdk/spica (with its 1-3 repulsion) and pair mm3/switch3/coulgauss/long: per-atom energy and
 * virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both on these cases; before the fix stress/atom
 * left those terms out (angle table and cross had no per-atom virial, the spica 1-3 repulsion and the mm3 pair none).
 */

const CASES = [
  'w216tb_angle_linear', 'w216tb_angle_spline_fp', 'w118yaff_angle_cross', 'w118yaff_mil53al_bonded',
  'w118yaff_mil53al_switch3', 'w118yaff_mil53al_switch3_t0', 'w118yaff_mof5_bonded', 'w118yaff_mof5_switch3',
  'w51cs_sdk_angle', 'w51cs_sdk_peg', 'w51cs_spica_angle', 'w51cs_spica_peg', 'w104cf_sds_hybrid', 'w104cf_sds_hybrid_reassign',
  'w104cf_sds_overlay', 'w104cf_sds_overlay_unset', 'w115p_mm3_mil53', 'w115p_mm3_mof5',
];

describe('angle table, cross, sdk/spica and pair mm3: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
