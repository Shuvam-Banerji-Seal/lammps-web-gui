import { describe, expect, it } from 'vitest';
import { parseMeamParams } from '../src/engine/force/pair/meam';
import {
  alloyAtomEnergyGrad,
  alloyPair,
  makeAlloyModel,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyPair,
} from '../src/engine/force/pair/meam_alloy';
import { StyleError } from '../src/engine/force/types';

/*
 * Multi-element MEAM, meam37 scope: the B2 (CsCl, lattce(I,J) = b2) and L12 (Cu3Au, lattce(I,J) = l12)
 * reference pairs with nn2 = 0. The dimer energies are native LAMMPS values (black box) for the synthetic
 * w37meam_* entries (the w15meam A/B elements, Ec(1,2) = 3.8, re(1,2) = 2.6, alpha(1,2) = 4.5); the crystal
 * parity over 30 nve steps is in tests/engineOracle.test.ts (w37meam_b2_crystal, w37meam_l12_crystal).
 * Measured with native LAMMPS (black box): A-B dimers at 2.2, 2.4, 2.6, 2.8, 3.0 and 3.3 A (the A atom at the
 * origin, the B atom at r) for both lattices; the engine agrees to 1e-12 (eV) for every dimer.
 */

const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
const B_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };

/** Two-element model: own pairs of A and B, and the I-J pair of the given lattce; cornerEl = 1 (element B). */
const modelWith = (lat: 'b2' | 'l12'): AlloyModel => {
  const ij: AlloyPair = { Ec: 3.8, re: 2.6, alpha: 4.5, lat, cornerEl: 1 };
  return makeAlloyModel(
    [A_EL, B_EL],
    [
      [{ Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self' }, ij],
      [ij, { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self' }],
    ],
    { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, eroseForm: 0 },
    true,
  );
};

/** Non-periodic cluster energy of an alloy model (same convention as engineMeamAlloy29.test.ts). */
const clusterE = (model: AlloyModel, types: number[], X: number[]): number => {
  const n = types.length;
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: AlloyNeighbor[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[3 * j] - X[3 * i], dy = X[3 * j + 1] - X[3 * i + 1], dz = X[3 * j + 2] - X[3 * i + 2];
      const r = Math.hypot(dx, dy, dz);
      if (r >= model.opts.rc) continue;
      nb.push({ e: types[j], j, dx, dy, dz, r });
    }
    E += alloyAtomEnergyGrad(model, types[i], nb, new Float64Array(3 * nb.length));
  }
  return E;
};

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

// Measured with native LAMMPS (black box): A-B dimer pe (eV), A at the origin, B at r along x.
const NATIVE_B2: Array<[number, number]> = [
  [2.2, -3.17318584742675],
  [2.4, -3.18355780887972],
  [2.6, -3.01714514983461],
  [2.8, -2.77210888910646],
  [3.0, -2.50437984484646],
  [3.3, -2.12139726506609],
];
const NATIVE_L12: Array<[number, number]> = [
  [2.2, -3.30605723516086],
  [2.4, -3.21007025722725],
  [2.6, -3.01417184001872],
  [2.8, -2.77244316678356],
  [3.0, -2.51862414289175],
  [3.3, -2.15581526217788],
];

describe('multi-element MEAM: lattce(I,J) = b2 and l12 (nn2 = 0)', () => {
  for (const [lat, table] of [['b2', NATIVE_B2], ['l12', NATIVE_L12]] as const) {
    const model = modelWith(lat);
    for (const [r, pe] of table) {
      it(`${lat} A-B dimer at ${r} A matches native pe to 1e-10 relative`, () => {
        const E = clusterE(model, [0, 1], [0, 0, 0, r, 0, 0]);
        expect(rel(E, pe)).toBeLessThan(1e-10);
      });
    }
    it(`${lat} phi_AB is the same for both orders of the pair`, () => {
      for (const r of [2.2, 2.6, 3.3]) {
        const a = alloyPair(model, 0, 1, r), b = alloyPair(model, 1, 0, r);
        expect(a.phi).toBeCloseTo(b.phi, 14);
        expect(a.dphi).toBeCloseTo(b.dphi, 14);
      }
    });
    it(`${lat} dphi is the derivative of phi`, () => {
      const h = 1e-6;
      for (const r of [2.2, 2.6, 3.0, 3.5]) {
        const fd = (alloyPair(model, 0, 1, r + h).phi - alloyPair(model, 0, 1, r - h).phi) / (2 * h);
        expect(alloyPair(model, 0, 1, r).dphi).toBeCloseTo(fd, 6);
      }
    });
  }
});

describe('multi-element MEAM: b2 and l12 parser', () => {
  const PAR = (lat: string) => `rc = 4\ndelr = 0.1\nEc(1,2) = 3.8\nre(1,2) = 2.6\nalpha(1,2) = 4.5\nlattce(1,2) = ${lat}\n`;
  it('lattce(1,2) = b2 and l12 are accepted for an I-J pair', () => {
    for (const lat of ['b2', 'l12']) {
      expect(() => parseMeamParams(PAR(lat), 'par', 2)).not.toThrow();
    }
  });
  it('other I-J lattce names stay StyleErrors naming the name', () => {
    expect(() => parseMeamParams(PAR('c11'), 'par', 2)).toThrow(StyleError);
    expect(() => parseMeamParams(PAR('c11'), 'par', 2)).toThrow(/c11/);
  });
});
