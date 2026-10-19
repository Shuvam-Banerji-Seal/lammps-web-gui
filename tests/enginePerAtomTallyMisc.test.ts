import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair extep, pair ylz, the dielectric pair styles and pair rheo, rheo/solid (src/engine-gpl/extep, ylz, dielectric, csdiel,
 * rheo): per-atom energy and virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both on these cases;
 * before the fix extep, ylz and rheo/solid tallied no per-atom virial, lj/cut/coul/cut/dielectric took its global virial
 * as x.f, and pair rheo with newton off gave ghost partners their half (B off by 0.5 to 1).
 */

const CASES = [
  'w145pr_pair_extep', 'w115p_ylz_flat', 'w115p_ylz_vesicle', 'w243de_efield_cutcoul', 'w243de_efield_cutcoul_kspace_kw',
  'w71diel_nopbc_dof', 'w71diel_nopbc_gmres', 'w243de_efield_debye', 'w265de_cutcoul_press', 'w96rheo_harmonic_means',
  'w96rheo_taylor_green', 'w96rheo_pair_solid', 'w96rheo_balloon', 'w96rheo_dam_break', 'w96rheo_icecubes', 'w96rheo_oxidation',
];

describe('extep, ylz and dielectric pair styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
