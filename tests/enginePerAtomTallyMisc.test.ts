import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair extep, pair ylz and the dielectric pair styles (src/engine-gpl/extep, ylz, dielectric, csdiel): per-atom energy and
 * virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both on these cases; before the fix extep and ylz
 * tallied no per-atom virial, and lj/cut/coul/cut/dielectric took its global virial as x.f (B off by 0.5 to 1).
 */

const CASES = [
  'w145pr_pair_extep', 'w115p_ylz_flat', 'w115p_ylz_vesicle', 'w243de_efield_cutcoul', 'w243de_efield_cutcoul_kspace_kw',
  'w71diel_nopbc_dof', 'w71diel_nopbc_gmres', 'w243de_efield_debye', 'w265de_cutcoul_press',
];

describe('extep, ylz and dielectric pair styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
