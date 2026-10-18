import { describe, expect, it } from 'vitest';
import { perAtomTally } from './perAtomTally';

/*
 * The SPH pair styles (src/engine-gpl/sph/ sph/taitwater, sph/taitwater/morris, sph/idealgas and
 * src/engine-gpl/gransph/ sph/lj): the per-atom virial (tests/perAtomTally.ts check B; the pairs tally no energy).
 * Native LAMMPS passes both checks on these cases; before the fix stress/atom summed to 0.
 */

const CASES = ['w228gs_sph_lj', 'w83sph_shock2d', 'w83sph_shock3d', 'w83sph_water'];

describe('SPH pair styles: per-atom virial', () => {
  it.each(CASES)('%s: pe/atom and stress/atom sum to pe and the virial', async (name) => {
    const { relA, relB } = await perAtomTally(name);
    expect(relA).toBeLessThan(1e-10);
    expect(relB).toBeLessThan(1e-10);
  }, 120000);
});
