import { describe, expect, it } from 'vitest';
import { esDrand, esGenNextParallelState, esInit, esNormal, esUrand } from '../src/engine-gpl/dpdreact/random_external_state';

/*
 * Reference values from the 64-bit integer arithmetic of random_external_state.h (xorshift64* with the 2685821657736338717
 * multiplier, es_init with 17 discarded draws). Computed with independent integer arithmetic, not from the TS code.
 */
describe('DPD-REACT external random state', () => {
  it('matches the reference integer sequence after es_init(234324)', () => {
    const s = esInit(234324);
    expect([esUrand(s), esUrand(s), esUrand(s)]).toEqual([2867909901, 3095506787, 829454919]);
  });

  it('draws doubles as urand64 / 2^64 and Gaussians by the polar method', () => {
    const s = esInit(234324);
    expect([esDrand(s), esDrand(s), esDrand(s)]).toEqual([0.6919047201127383, 0.8967700306524303, 0.4385100268939635]);
    const g = esInit(234324);
    expect(esNormal(g)).toBeCloseTo(0.30929300130203996, 14);
    expect(esNormal(g)).toBeCloseTo(-0.06530703981381658, 14);
    expect(esNormal(g)).toBeCloseTo(-0.06268744125546892, 14);
  });

  it('builds parallel states from four 16-bit pieces', () => {
    const serial = esInit(234324);
    const a = esGenNextParallelState(serial);
    const b = esGenNextParallelState(serial);
    expect(BigInt(a.hi) * 4294967296n + BigInt(a.lo)).toBe(10950569718103239302n);
    expect(BigInt(b.hi) * 4294967296n + BigInt(b.lo)).toBe(15627959421469898571n);
  });
});
