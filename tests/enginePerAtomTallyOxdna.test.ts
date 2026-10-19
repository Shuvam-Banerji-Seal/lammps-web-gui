import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The CG-DNA styles (src/engine-gpl/oxdna/): the per-atom molecular virial (tests/perAtomTally.ts check B) of bond
 * oxdna/fene and its variants, pair oxdna/stk, oxrna2/stk and oxdna2/coaxstk (their per-atom energy too). Native LAMMPS
 * passes both checks on these cases; before the fix stress/atom left out those terms (B off by 0.9 to 1.0).
 */

const CASES = [
  'w57oxdna2_coaxstk_full', 'w57oxdna2_dh', 'w57oxdna2_dh_half', 'w57oxdna2_excv_bond', 'w57oxdna_duplex1_full',
  'w57oxdna_duplex2_full', 'w57oxdna_excv_bond', 'w57oxdna_excv_compressed', 'w57oxdna_excv_stk', 'w57oxdna_hbond',
  'w57oxdna_hbond_duplex', 'w57oxdna_langevin', 'w57oxdna_nvedot', 'w57oxdna_terms', 'w57oxdna_xstk_duplex',
  'w57oxrna2_full', 'w78oxdna3_lj_duplex2', 'w78oxdna3_lj_unique_bp', 'w78oxdna3_real_duplex2', 'w78oxdna3_real_unique_bp',
];

describe('oxDNA styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
