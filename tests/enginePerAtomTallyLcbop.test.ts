import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair_style lcbop (src/engine-gpl/lcbop): the per-atom virial of the short- and long-range pair terms, the N_ij and
 * M_ij coordination forces and the bond-order forces (tests/perAtomTally.ts check B), as the native ev_tally, v_tally2
 * and v_tally3 calls of pair_lcbop.cpp tally it. Native LAMMPS passes both checks on these cases; before, compute
 * stress/atom refused pair lcbop.
 */

const CASES = ['w234lc_diamond', 'w234lc_cl1', 'w234lc_graphene', 'w234lc_hotcl', 'w234lc_cl2', 'w234lc_cl3'];

describe('lcbop per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
