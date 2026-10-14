import { describe, expect, it } from 'vitest';
import { PAIRS as SRP_PAIRS, FIXES as SRP_FIXES } from '../src/engine-gpl/registry/srp';
import { PairCoulSlaterLong, slaterCoulPair } from '../src/engine-gpl/slater/pair_coul_slater_long';
import { PairE3B } from '../src/engine-gpl/e3b/pair_e3b';
import { StyleError } from '../src/engine/force/types';

/* Parity with native LAMMPS is in the oracle cases tests/oracle/w130pr_*.in; these check the argument rules and refusals. */
describe('pairs ported in batch 6', () => {
  it('coul/slater/long takes lamda and cutoff only', () => {
    const p = new PairCoulSlaterLong();
    expect(() => p.settings(['0.929'])).toThrow(StyleError);
    p.settings(['0.929', '3.0']);
    expect(p.extract('cut_coul')).toBe(3);
    expect(p.extract('lamda')).toBe(0.929);
  });

  it('the Slater Coulomb term vanishes at long range, leaving the erfc tail (Ewald real-space form)', () => {
    const far = slaterCoulPair(400 * 400, 1, -1, 0.3, 1, 0.929, 1);
    expect(Math.abs(far.e)).toBeLessThan(1e-20);
    expect(Math.abs(far.f)).toBeLessThan(1e-20);
  });

  it('e3b needs the O type and a preset or the full keyword set', () => {
    const p = new PairE3B();
    expect(() => p.settings(['1', '2'])).toThrow(StyleError);
    p.settings(['1']);
    expect(p.extract('pvector')).toBeInstanceOf(Float64Array);
  });

  it('srp and srp/react are refused by name, not run as no-ops', () => {
    expect(() => SRP_PAIRS.srp()).toThrow(/pair_style srp is not supported/);
    expect(() => SRP_PAIRS['srp/react']()).toThrow(/pair_style srp\/react is not supported/);
    expect(() => SRP_FIXES.srp()).toThrow(/fix srp is not supported/);
  });
});
