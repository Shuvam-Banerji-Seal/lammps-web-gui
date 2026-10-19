import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * kspace_style ewald/dipole (src/engine/force/kspace/ewald_dipole.ts) and ewald/dipole/spin
 * (src/engine-gpl/spin/ewald_dipole_spin.ts): the per-atom energy and virial of the k-space sum and the dipole self
 * energy, with the pair styles lj/cut/dipole/long and spin/dipole/long (tests/perAtomTally.ts checks A and B). Native
 * LAMMPS passes both checks on these cases; before, compute pe/atom and stress/atom refused both kspace styles and
 * spin/dipole/long tallied no per-atom energy or virial.
 */

const CASES = [
  'w44dl_static_fixed', 'w44dl_static_auto', 'w44dl_noncubic', 'w44dl_special', 'w44dl_md_sphere', 'w51km_dipole_noeffect',
  'w267ed_dipole_ptensor', 'w267ed_spin_ptensor', 'w103spin_iron_ewald_nve',
];

describe('ewald/dipole per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
