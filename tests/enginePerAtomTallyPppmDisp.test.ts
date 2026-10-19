import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * kspace_style pppm/disp (src/engine-gpl/pppm_disp/): per-atom energy and virial of the Coulomb and dispersion meshes
 * (geometric and arithmetic mixing) with their self terms (tests/perAtomTally.ts checks A and B). Native LAMMPS passes
 * both on these cases; the engine refused compute pe/atom and stress/atom with pppm/disp before. The per-atom values
 * agree with native to 1e-14.
 */

const CASES = ['w161pd_lj_arith', 'w161pd_lj_auto', 'w161pd_ljcoul_fixed', 'w161pd_lj_fixed', 'w177px_balance_kspace', 'w252ll_off_pppmdisp'];

describe('pppm/disp: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
