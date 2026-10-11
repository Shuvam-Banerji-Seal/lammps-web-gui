import { describe, expect, it } from 'vitest';
import { gOfIbar, gPrimeOfIbar } from '../src/engine/force/pair/meam_alloy';
import { alloyPair, makeAlloyModel, type AlloyElement, type AlloyModel, type AlloyPair } from '../src/engine/force/pair/meam_alloy';

/*
 * Multi-element MEAM, meam42 scope: bcc and hcp element references, nn2 = 1 on a bcc element, and ibar = 3 in an
 * alloy. The energies are native LAMMPS values (black box) for the synthetic w42mb entries, checked by
 * tests/engineOracle.test.ts (w42mb_bcc1_*, w42mb_bcc2nn_*, w42mb_hcp_*, w42mb_ibar3_*). This file checks the
 * analytic pair derivative against finite differences, and the G(Gamma) forms.
 * Docs (pair_meam.rst): 3 => G = 2/(1+exp(-Gamma)); 2 => not implemented (ibar = 2 is refused).
 */

const BCC_A: AlloyElement = { z: 8, lat: 'bcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0], ibar: 3 };
const HCP_A: AlloyElement = { z: 12, lat: 'hcp', re: 2.5, alpha: 4.5, Ec: 3.2, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
const FCC_B: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };

const pairOf = (el: AlloyElement, nn2 = false): AlloyPair => ({ Ec: el.Ec, re: el.re, alpha: el.alpha, lat: 'self', nn2 });

/** Two-element model (element 0 = el0, element 1 = FCC_B) with an I-J b1 pair and the given nn2 flags. */
const model2 = (el0: AlloyElement, nn2: boolean): AlloyModel => {
  const ij: AlloyPair = { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1', nn2 };
  return makeAlloyModel(
    [el0, FCC_B],
    [[pairOf(el0, nn2), ij], [ij, pairOf(FCC_B, nn2)]],
    { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, eroseForm: 0 },
    true,
  );
};

describe('MEAM alloy ibar forms (meam42)', () => {
  it('ibar = 3 has G = 2/(1+exp(-Gamma)) and its derivative matches a finite difference', () => {
    for (const g of [-0.7, 0, 0.3, 1.5]) {
      expect(gOfIbar(3, g)).toBeCloseTo(2 / (1 + Math.exp(-g)), 15);
      const h = 1e-6;
      const fd = (gOfIbar(3, g + h) - gOfIbar(3, g - h)) / (2 * h);
      expect(gPrimeOfIbar(3, g)).toBeCloseTo(fd, 7);
    }
  });

  it('ibar = 2 is not implemented (docs: "2 => not implemented") and throws', () => {
    expect(() => gOfIbar(2, 0.1)).toThrow(/ibar 2/);
  });
});

describe('MEAM alloy bcc and hcp elements (meam42)', () => {
  it('the analytic dphi of the bcc self pair (nn2 = 0 and 1) matches a finite difference', () => {
    for (const nn2 of [false, true]) {
      const m = model2(BCC_A, nn2);
      for (const r of [2.3, 2.5, 2.9, 3.3]) {
        const h = 1e-6;
        const fd = (alloyPair(m, 0, 0, r + h).phi - alloyPair(m, 0, 0, r - h).phi) / (2 * h);
        expect(alloyPair(m, 0, 0, r).dphi).toBeCloseTo(fd, 6);
      }
    }
  });

  it('the analytic dphi of the hcp self pair matches a finite difference', () => {
    const m = model2(HCP_A, false);
    for (const r of [2.3, 2.5, 2.9]) {
      const h = 1e-6;
      const fd = (alloyPair(m, 0, 0, r + h).phi - alloyPair(m, 0, 0, r - h).phi) / (2 * h);
      expect(alloyPair(m, 0, 0, r).dphi).toBeCloseTo(fd, 6);
    }
  });
});
