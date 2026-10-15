import { describe, expect, it } from 'vitest';
import { mixingWeights, MIX_WEIGHTS } from '../src/engine-gpl/dpdrx/rx_mixing';
import { PairTableRX } from '../src/engine-gpl/dpdrx/pair_table_rx';
import { PairMultiLucyRX } from '../src/engine-gpl/dpdrx/pair_multi_lucy_rx';

/* Unit checks of the table/rx and multi/lucy/rx port (src/engine-gpl/dpdrx): the species weights of
 * pair_table_rx.cpp getMixingWeights, and the pair style keywords that stay refused by name. The oracle cases
 * (tests/oracle/w162rx_*) cover the physics. */

describe('getMixingWeights (pair_table_rx.cpp)', () => {
  // two species, one atom: current (2, 1) and old (1, 1)
  const cur = [Float64Array.of(2), Float64Array.of(1)];
  const old = [Float64Array.of(1), Float64Array.of(1)];

  it('weights a named site by its fraction (fractional) or its molecule number (molecular)', () => {
    const w = new Float64Array(MIX_WEIGHTS);
    mixingWeights(0, cur, old, 0, 1, true, w, 0);
    expect(w[0]).toBeCloseTo(1 / 2, 14); // site1 old: 1 of 2
    expect(w[2]).toBeCloseTo(2 / 3, 14); // site1 new: 2 of 3
    mixingWeights(0, cur, old, 0, 1, false, w, 0);
    expect(w[0]).toBe(1);
    expect(w[2]).toBe(2);
  });

  it('gives the 1fluid site the sum of the other species (1 minus the named ones for molecules)', () => {
    const w = new Float64Array(MIX_WEIGHTS);
    // site1 named (species 0), site2 1fluid: the one-fluid part is species 1 alone
    mixingWeights(0, cur, old, 0, -1, true, w, 0);
    expect(w[1]).toBeCloseTo(1 / 2, 14); // site2 old: species 1 of 2 in the old state
    expect(w[3]).toBeCloseTo(1 / 3, 14); // site2 new: species 1 of 3 now
    mixingWeights(0, cur, old, 0, -1, false, w, 0);
    expect(w[3]).toBeCloseTo(1 - (3 - 1), 14); // 1 - (total - other species) with the new total 3
  });

  it('refuses an atom with no molecules', () => {
    const w = new Float64Array(MIX_WEIGHTS);
    expect(() => mixingWeights(0, [Float64Array.of(0)], [Float64Array.of(0)], 0, -1, true, w, 0)).toThrow(/less than 10\*DBL_EPSILON/);
  });
});

describe('pair style keywords (refused by name when unsupported)', () => {
  it('table/rx accepts lookup, linear and spline with fractional or molecular', () => {
    const p = new PairTableRX();
    expect(() => p.settings(['linear', '100', 'fractional'])).not.toThrow();
    expect(() => p.settings(['spline', '100', 'molecular'])).not.toThrow();
  });

  it('table/rx refuses bitmap and the long-range keywords by name', () => {
    const p = new PairTableRX();
    expect(() => p.settings(['bitmap', '10'])).toThrow(/bitmap/);
    expect(() => p.settings(['linear', '100', 'ewald'])).toThrow(/'ewald' is not supported/);
    expect(() => p.settings(['linear', '100', 'tip4p'])).toThrow(/'tip4p' is not supported/);
  });

  it('multi/lucy/rx takes lookup or linear and the weighting keywords only', () => {
    const p = new PairMultiLucyRX();
    expect(() => p.settings(['lookup', '100', 'molecular'])).not.toThrow();
    expect(() => p.settings(['spline', '100'])).toThrow(/Unknown table style/);
    expect(() => p.settings(['linear', '100', 'ewald'])).toThrow(/Illegal pair_style command/);
  });
});
