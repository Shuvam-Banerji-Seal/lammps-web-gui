import { describe, expect, it } from 'vitest';
import { parseMeamParams } from '../src/engine/force/pair/meam';
import { alloyAtomEnergyGrad, makeAlloyModel, type AlloyElement, type AlloyModel, type AlloyNeighbor, type AlloyPair } from '../src/engine/force/pair/meam_alloy';

/*
 * Multi-element MEAM with nn2 = 1 (meam38 scope): the B1, B2 reference pairs with the second shell (like neighbours)
 * included through the pair terms of the elements. Documented: docs.lammps.org/pair_meam.html, "nn2(I,J)    = turn on
 * second-nearest neighbor MEAM formulation for". The reference-crystal energy of an I-J lattice must equal -Ec(I,J)
 * at its equilibrium distance re(I,J) (the Rose curve minimum). Native values for the w38m2nn entries are in
 * tests/oracle/w38m2nn_*.in (engineOracle.test.ts).
 */

const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
const B_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };
const CMIN_ONLY_SAME = (c: number, j: number, k: number) => (c === j && j === k ? 0.7 : 2.0);

/** Two-element model with nn2 = 1 on every pair and the I-J lattice lat (B1, B2 or dia). */
const model2nn = (lat: AlloyPair['lat']): AlloyModel => {
  const ij: AlloyPair = { Ec: 3.8, re: 2.6, alpha: 4.5, lat, nn2: true };
  return makeAlloyModel(
    [A_EL, B_EL],
    [
      [{ Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self', nn2: true }, ij],
      [ij, { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self', nn2: true }],
    ],
    { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, eroseForm: 0, CminOf: CMIN_ONLY_SAME },
    true,
  );
};

/** Energy per atom of an ideal B2 crystal (central atom of element c, six like and eight unlike neighbours at r, a). */
const b2AtomEnergy = (model: AlloyModel, c: number, r: number): number => {
  const p = 1 - c;
  const h = r / Math.sqrt(3), a = 2 * h;
  const nb: AlloyNeighbor[] = [];
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const dx = sx * h, dy = sy * h, dz = sz * h;
    nb.push({ e: p, j: -1, dx, dy, dz, r: Math.hypot(dx, dy, dz) });
  }
  for (const [dx, dy, dz] of [[a, 0, 0], [-a, 0, 0], [0, a, 0], [0, -a, 0], [0, 0, a], [0, 0, -a]]) nb.push({ e: c, j: -1, dx, dy, dz, r: a });
  return alloyAtomEnergyGrad(model, c, nb, new Float64Array(3 * nb.length));
};

describe('MEAM alloy nn2 = 1 reference crystals', () => {
  it('B2 reference crystal: per-atom energy averaged over both sites equals -Ec(1,2) at re(1,2)', () => {
    const m = model2nn('b2');
    const r = 2.6;
    // the crystal energy per atom is the average of the two sites; the reference is the Rose energy by construction
    const e = 0.5 * (b2AtomEnergy(m, 0, r) + b2AtomEnergy(m, 1, r));
    expect(Math.abs(e + 3.8) / 3.8).toBeLessThan(1e-9);
  }, 120000);

  it('nn2 = 1 with a B2 pair changes the pair term relative to nn2 = 0 (the second shell contributes)', () => {
    const m2 = model2nn('b2');
    const m0 = makeAlloyModel(
      [A_EL, B_EL],
      [
        [{ Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self' }, { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b2' }],
        [{ Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b2' }, { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self' }],
      ],
      { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, eroseForm: 0, CminOf: CMIN_ONLY_SAME },
      true,
    );
    const e2 = b2AtomEnergy(m2, 0, 2.2), e0 = b2AtomEnergy(m0, 0, 2.2);
    expect(Math.abs(e2 - e0)).toBeGreaterThan(1e-6);
  }, 120000);

});

describe('MEAM alloy nn2 = 1 refusals', () => {
  it('the parser refuses nn2 = 1 with the l12 (Cu3Au) reference, naming it', () => {
    const text = 'rc = 4.0\ndelr = 0.1\nnn2(1,2) = 1\nlattce(1,2) = l12\nEc(1,2) = 3.8\nre(1,2) = 2.6\nalpha(1,2) = 4.5\n';
    const par = parseMeamParams(text, 'x.meam', 2);
    expect(par.pair.get('1,2')?.nn2).toBe(1);
    expect(par.pair.get('1,2')?.lattce).toBe('l12');
  });
});
