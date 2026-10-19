import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * kspace_style pppm/dipole (src/engine-gpl/dipole/pppm_dipole.ts, dipoles and charges) and pair lj/cut/dipole/long:
 * per-atom energy and virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both on these cases; the
 * engine refused compute pe/atom and stress/atom with pppm/dipole, and the pair tallied no per-atom virial. The
 * per-atom values agree with native to 5e-15.
 */

const CASES = [
  'w135pd_auto_fcc', 'w135pd_fixed_fcc', 'w135pd_md_sphere', 'w135pd_noncubic_order3', 'w168pq_charge_dipole_auto',
  'w168pq_charge_partial', 'w168pq_nonneutral', 'w170dq_charge_dipole_small', 'w170dq_charge_special',
];

describe('pppm/dipole: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 240000);
});
