import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * run_style respa (src/engine/run/respa.ts): compute pe/atom and stress/atom see the sum over the rRESPA levels of each
 * level's last evaluation (tests/perAtomTally.ts checks A and B), with bonded, pair, hybrid-split pair and kspace terms
 * on different levels and fix filter/corotate. Native LAMMPS passes both checks on these cases; before, the engine
 * refused per-atom computes under run_style respa.
 */

const CASES = [
  'w37respa_bond', 'w37respa_3lev', 'w37respa_hybrid', 'w37respa_kspace', 'w37respa_nvt', 'w214fc_respa',
  'w214fc_clusters_b', 'w214fc_clusters_m',
];

describe('run_style respa per-atom energy and virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
