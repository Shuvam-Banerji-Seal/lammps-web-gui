import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * pair_style airebo, airebo/morse, airebo/bc and rebo (src/engine-gpl/airebo): the per-atom virial of the REBO pair and
 * bond-order terms, the LJ term with its connectivity switch and the torsion term (tests/perAtomTally.ts check B), as
 * the native ev_tally, v_tally2, v_tally3, v_tally4 and ev_tally4 calls of pair_airebo.cpp tally it; pair rebomos
 * (pair_rebomos.cpp) and pair drip (v_tally2_newton, the drip cases with rebo) likewise. Native LAMMPS passes both
 * checks on these cases; before, compute stress/atom refused airebo and rebomos and left drip's virial out.
 */

const CASES = [
  'w52airebo_airebo_00', 'w52airebo_airebo_10', 'w52airebo_airebo_11', 'w52airebo_morse_11', 'w147ab_airebo_bc',
  'w52airebo_rebo_benzene', 'w52airebo_rebo_cyclohexane', 'w52airebo_rebo_diamond', 'w52airebo_rebo_ethane',
  'w52airebo_rebo_ethylene', 'w52airebo_rebo_graphene', 'w52airebo_rebo_methane',
  'w102il_drip_c', 'w102il_drip_ch', 'w102il_ilp_gr_rebo', 'w102il_kcf_rebo_bilayer', 'w102il_kcz_rebo_bilayer',
  'w102il_saip_gold_gr', 'w115p_rebomos', 'w115p_rebomos_bulk',
];

describe('airebo per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
