import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * fix propel/self (src/engine-gpl/brownian/fix_propel_self.ts, a port of src/BROWNIAN/fix_propel_self.cpp): per-atom
 * virial of the self-propulsion force, v_tally(i, vi) with the unwrapped position (tests/perAtomTally.ts checks A and B).
 * Native LAMMPS passes both on these cases; before the fix stress/atom refused the fix with a StyleError.
 */

const CASES = [
  'w62brown_abp2d', 'w62brown_aligndipole', 'w62brown_alignquat', 'w62brown_propeldipole',
  'w62brown_propelquat', 'w62brown_propeltri', 'w62brown_propelvel',
];

describe('fix propel/self: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
