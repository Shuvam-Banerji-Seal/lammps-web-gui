import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The BPM styles (src/engine-gpl/bpm/ bonds bpm/rotational, bpm/spring, bpm/spring/plastic and bpm/peri, pair bpm/spring
 * in src/engine/force/pair/bpm.ts and pair bpm/peri): per-atom energy and virial (tests/perAtomTally.ts checks A and B).
 * Native LAMMPS passes both on CASES; before the fix stress/atom left out the BPM bonds and pairs. In PERI native fails B
 * itself (pair bpm/peri tallies the per-atom virial with one more vfrac factor than its fdotr global virial,
 * pair_bpm_peri.cpp), so only A is checked there; the engine's pe/atom and stress/atom agree with native per atom to
 * 1e-12 on those cases. The cases with pair gran/hooke/history and bpm/rotational bonds also need the granular contact
 * tallies of tests/enginePerAtomTallyGranular.test.ts.
 */

const CASES = [
  'w105cf_bpm_rot_negative', 'w105cf_bpm_rot_zero', 'w131cf_bpm_bond_extra', 'w131cf_efield_dipole',
  'w144bpm_rot_chain', 'w144bpm_spring_store', 'w144bpm_spring_volume', 'w166px_settorque_wire',
  'w175cf_efield_dipole_40', 'w51bpm_impact_cut', 'w51bpm_pair_anh', 'w51bpm_spring_compress',
  'w51bpm_spring_nobreak', 'w51bpm_spring_normalize', 'w51bpm_spring_overlay', 'w51bpm_spring_run0',
  'w53bpm_rot_chain', 'w53bpm_rot_damp', 'w53bpm_rot_stretch', 'w59bpm_dipole', 'w59bpm_plastic',
  'w59bpm_plastic_norm', 'w59bpm_plasticity_cut', 'w59bpm_rot_avg_norm', 'w59bpm_rot_chain3', 'w59bpm_rot_dem',
  'w59bpm_rot_dem_particle', 'w59bpm_rot_particle', 'w59bpm_rot_press', 'w59bpm_rot_shear', 'w96rheo_balloon',
  'w96rheo_icecubes', 'w96rheo_oxidation', 'w96rheo_thermal_react',
];

const PERI = [
  'w163bp_eps', 'w163bp_lps', 'w163bp_ves', 'w59bpm_peri_pmb',
];

describe('BPM styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);

  it.each(PERI)('%s: pe/atom sums to pe', async (name) => {
    const { relA } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
  }, 120000);
});
