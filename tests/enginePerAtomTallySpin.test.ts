import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The SPIN pair styles (src/engine/force/pair/spin_exchange.ts, spin_dmi.ts, src/engine-gpl/spin/pair_spin_neel.ts and
 * pair_spin_dipole_cut.ts): per-atom energy and virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both
 * on CASES; before the fix the engine tallied no per-atom energy (A off by 1) and, with nve/spin lattice moving, its
 * global virial missed the forces on the ghost atoms (B off by 0.9 to 1.7). In LATTICE native fails A by 2e-7 to 9e-4
 * (the fix precession/spin energy is not in pe/atom), and the engine fails it by the same amount, so only B is checked.
 */

const CASES = [
  'w47spin_frozen', 'w47spin_iron', 'w47spin_moving2', 'w47spin_pair2', 'w48sp2_bfo_hybrid', 'w48sp2_biquad', 'w48sp2_dmi',
  'w48sp2_hybrid', 'w48sp2_magelec', 'w48sp2_setforce', 'w259spin_magelec_lattice',
];

const LATTICE = ['w103spin_dipole_cut_lattice', 'w103spin_neel_lattice'];

describe('spin pair styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);

  it.each(LATTICE)('%s: stress/atom sums to the virial', async (name) => {
    const { relB } = await perAtomTally(name);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
