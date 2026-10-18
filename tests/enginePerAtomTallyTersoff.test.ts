import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair_style tersoff and its variants (src/engine/force/pair/tersoff.ts): the per-atom virial of the radial pair term and
 * of the three-body term (tests/perAtomTally.ts check B). Native LAMMPS passes both checks on these cases; before the
 * fix the engine left out the radial term and scaled the three-body one by the unit bond vectors (B off by 0.7 to 1.3).
 */

const CASES = [
  'w22pot_tersoff_si', 'w14tersoff_mod', 'w14tersoff_modc', 'w14tersoff_zbl', 'w22pot_tersoff_bnc', 'w22pot_tersoff_gan',
  'w2tsw_tersoff', 'w102il_ilp_hbn_tersoff',
];

describe('tersoff per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
