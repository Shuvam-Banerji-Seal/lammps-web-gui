import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair_style comb3 (src/engine-gpl/comb3): the per-atom virial of the van der Waals, Coulomb, repulsive and bond-order
 * pair terms and of the i-j-k and j-i-l three-body terms (tests/perAtomTally.ts check B), as the native ev_tally and
 * v_tally3 calls of pair_comb3.cpp tally it. Native LAMMPS passes both checks on these cases; before, compute
 * stress/atom refused pair comb3.
 */

const CASES = ['w158cb_static', 'w158cb_qeq', 'w158cb_culib', 'w158cb_cu'];

describe('comb3 per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);

  // The torsion and radical forces of pair_comb3.cpp have no per-atom virial tally, so with them (the hydrocarbon case)
  // stress/atom does not sum to the virial. Measured with native LAMMPS: its pe/atom and stress/atom on w158cb_hc
  // equal the engine's within 2e-12 of each column's largest value; check B is off by 5% in both.
  it('w158cb_hc: pe/atom sums to pe; stress/atom leaves out the torsion and radical virial, as native', async () => {
    const { relA, relB } = await perAtomTally('w158cb_hc');
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeGreaterThan(0.04);
    expect(relB).toBeLessThan(0.06);
  }, 120000);
});
