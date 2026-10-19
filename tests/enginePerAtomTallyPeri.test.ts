import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The peridynamic pair styles (src/engine/force/pair/peri.ts for peri/pmb and peri/lps, src/engine-gpl/peri/ for
 * peri/eps and peri/ves): per-atom energy and virial (tests/perAtomTally.ts checks A and B). Native LAMMPS passes both
 * on these cases; before the fix the engine tallied neither (A and B off by 1). The w260 collision cases are checked
 * against native in tests/oracle too.
 */

const CASES = [
  'w22peri_pmb_block', 'w22peri_pmb_data', 'w22peri_lps_data', 'w107peri_eps_block', 'w107peri_eps_indent',
  'w107peri_ves_block', 'w107peri_ves_indent', 'w240pl_block', 'w240pl_impact', 'w240pl_stretch',
  'w260peri_collide_pmb', 'w260peri_collide_lps',
];

describe('peri pair styles: per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
