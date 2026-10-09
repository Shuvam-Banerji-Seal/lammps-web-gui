import { describe, expect, it } from 'vitest';
import {
  alloyAtomEnergyGrad,
  densityPartials,
  makeAlloyModel,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyPair,
} from '../src/engine/force/pair/meam_alloy';

/*
 * MEAM ialloy = 1 and 2 (docs.lammps.org/pair_meam.html, plans/lammps-docs/pair_meam.rst):
 *   "ialloy = integer flag to use alternative averaging rule for t parameters",
 *   "0 = standard averaging (matches ialloy=0 in DYNAMO)",
 *   "1 = alternative averaging (matches ialloy=1 in DYNAMO)",
 *   "2 = no averaging of t (use single-element values)", "default = 0".
 *
 * Measured with native LAMMPS (black box) on plans/scratch/w31meamialloy (synthetic entries with
 * strongly differing t vectors, non-periodic 8-atom A/B cluster): pe = 9.07541777091405 eV
 * (ialloy 0), 9.43607605933164 eV (ialloy 1), 9.36849116893382 eV (ialloy 2).
 *
 * The oracle cases tests/oracle/w31meamialloy_cluster{1,2} and w31meamialloy_crystal{1,2} exercise
 * the same rules (and their forces) over 40 nve steps against native LAMMPS.
 */

const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 5.0, 3.0, 1.0] };
const B_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, -2.0, 1.0, 4.0] };
const PAIRS: AlloyPair[][] = [
  [
    { Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self' },
    { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1' },
  ],
  [
    { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1' },
    { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self' },
  ],
];

const clusterEnergy = (model: AlloyModel, types: number[], X: number[]): number => {
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
    const g = new Float64Array(3 * nb.length);
    E += alloyAtomEnergyGrad(model, types[i], nb, g);
  }
  return E;
};

const clusterForces = (model: AlloyModel, types: number[], X: number[]): Float64Array => {
  const n = types.length;
  const F = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const nb: AlloyNeighbor[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[3 * j] - X[3 * i], dy = X[3 * j + 1] - X[3 * i + 1], dz = X[3 * j + 2] - X[3 * i + 2];
      const r = Math.hypot(dx, dy, dz);
      if (r >= model.opts.rc) continue;
      nb.push({ e: types[j], j, dx, dy, dz, r });
    }
    const g = new Float64Array(3 * nb.length);
    alloyAtomEnergyGrad(model, types[i], nb, g);
    nb.forEach((p, m) => {
      for (let c = 0; c < 3; c++) {
        F[3 * p.j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    });
  }
  return F;
};

const TYPES = [0, 1, 0, 1, 0, 1, 0, 1];
const X = [
  30, 30, 30, 31.9, 30.4, 30.1, 31.1, 32.0, 30.3, 32.6, 31.4, 31.9,
  30.2, 31.7, 32.4, 31.6, 30.2, 32.1, 32.8, 32.7, 30.6, 30.7, 32.9, 31.2,
];

describe('MEAM ialloy 1 and 2', () => {
  it('matches native for the mixed A/B cluster at ialloy 0, 1 and 2', () => {
    const expected = [9.07541777091405, 9.43607605933164, 9.36849116893382];
    for (const ialloy of [0, 1, 2]) {
      const model = makeAlloyModel([A_EL, B_EL], PAIRS, { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, ialloy }, true);
      expect(Math.abs(clusterEnergy(model, TYPES, X) - expected[ialloy])).toBeLessThan(1e-9);
    }
  }, 30000);

  it('differentiates the ialloy 1 density (finite differences of densityPartials)', () => {
    const terms = [
      { e: 0, W: 0.8, A: [0.9, 0.7, 1.1, 1.3], u: [1, 0, 0], t: [1, 5.6, 3, 1] },
      { e: 1, W: 1.2, A: [1.4, 0.6, 0.9, 1.2], u: [0, 1, 0], t: [1, 0.4, 1, 4] },
      { e: 0, W: 0.5, A: [0.7, 1.0, 0.8, 0.6], u: [0, 0, 1], t: [1, 5.6, 3, 1] },
    ] as unknown as Parameters<typeof densityPartials>[0];
    const part = densityPartials(terms, 1, 0, 1);
    const E = () => densityPartials(terms, 1, 0, 1).rb;
    const h = 1e-6;
    let worst = 0;
    const check = (analytic: number, fd: number) => { worst = Math.max(worst, Math.abs(analytic - fd)); };
    for (let m = 0; m < terms.length; m++) {
      const t = terms[m] as unknown as { W: number; A: number[]; u: number[] };
      t.W += h; const wp = E(); t.W -= 2 * h; const wm = E(); t.W += h;
      check(part.gW[m], (wp - wm) / (2 * h));
      for (let n = 0; n < 4; n++) {
        t.A[n] += h; const ap = E(); t.A[n] -= 2 * h; const am = E(); t.A[n] += h;
        check(part.gA[4 * m + n], (ap - am) / (2 * h));
      }
      for (let c = 0; c < 3; c++) {
        t.u[c] += h; const up = E(); t.u[c] -= 2 * h; const um = E(); t.u[c] += h;
        check(part.gU[3 * m + c], (up - um) / (2 * h));
      }
    }
    expect(worst).toBeLessThan(1e-4);
  });

  it('differentiates the ialloy 1 cluster force (finite differences of the energy)', () => {
    const model = makeAlloyModel([A_EL, B_EL], PAIRS, { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8, ialloy: 1 }, true);
    const F = clusterForces(model, TYPES, X);
    const h = 1e-5;
    let worst = 0;
    for (let i = 0; i < 3 * TYPES.length; i++) {
      const Xp = X.slice(), Xm = X.slice();
      Xp[i] += h; Xm[i] -= h;
      const fd = -(clusterEnergy(model, TYPES, Xp) - clusterEnergy(model, TYPES, Xm)) / (2 * h);
      worst = Math.max(worst, Math.abs(fd - F[i]));
    }
    expect(worst).toBeLessThan(1e-5);
  });
});
