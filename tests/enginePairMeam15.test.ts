import { describe, expect, it } from 'vitest';
import {
  DEFAULT_MEAM_OPTIONS,
  PairMeam,
  meamEnergy,
  meamEnergyForces,
  parseMeamLibrary,
  parseMeamParams,
  type MeamElement,
  type MeamOptions,
} from '../src/engine/force/pair/meam';
import { referenceVectors } from '../src/engine/force/pair/meam_lattice';
import {
  alloyAtomEnergyGrad,
  makeAlloyModel,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyPair,
} from '../src/engine/force/pair/meam_alloy';
import { StyleError } from '../src/engine/force/types';

/*
 * MEAM with non-fcc reference structures (bcc, dia) for one element, meam15 scope.
 * Native reference values were measured with LAMMPS as a black box on synthetic
 * entries (tests/oracle/w15meam_bcc_lib.meam, w15meam_bcc.meam); they are numbers,
 * not LAMMPS text. Measured with native LAMMPS (black box): the perfect bcc crystal
 * (2x2x2 cells, 16 atoms) at lattice constant 3.16 A has pe -142.400000000091 eV; at
 * 2.8868 A and 3.3466 A it has -124.560876810954 eV and -137.289643831942 eV.
 */

const W_BCC: MeamElement = {
  z: 8,
  re: 2.73664,
  alpha: 5.0,
  Ec: 8.9,
  A: 1,
  beta: [2.0, 1.5, 2.5, 4.0],
  t: [1, 2.0, 1.0, 1.0],
  ibar: 0,
  lat: 'bcc',
};
const W_DIA: MeamElement = { ...W_BCC, z: 4, re: 2.165064, Ec: 4.6, lat: 'dia' };
const OPTS: MeamOptions = { ...DEFAULT_MEAM_OPTIONS };

/** Cubic cell of a Bravais+basis structure: n x n x n conventional cells. */
const cells = (basis: number[][], a: number, n: number): Float64Array => {
  const p: number[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++)
        for (const b of basis) p.push((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a);
  return Float64Array.from(p);
};
const BCC_BASIS = [[0, 0, 0], [0.5, 0.5, 0.5]];
const DIA_BASIS = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]];

/** Deterministic LCG in [0,1). */
const rng = (seed: number) => {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
};

/** Central-difference check of F = -grad E over every component; returns the worst relative error. */
const checkForces = (el: MeamElement, x: Float64Array, L: [number, number, number]) => {
  const { E, F } = meamEnergyForces(el, OPTS, x, L);
  expect(E).toBeCloseTo(meamEnergy(el, OPTS, x, L), 9);
  const h = 1e-5;
  let worst = 0;
  for (let q = 0; q < x.length; q++) {
    const xp = Float64Array.from(x), xm = Float64Array.from(x);
    xp[q] += h;
    xm[q] -= h;
    const fd = -(meamEnergy(el, OPTS, xp, L) - meamEnergy(el, OPTS, xm, L)) / (2 * h);
    const err = Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2);
    if (err > worst) worst = err;
  }
  return worst;
};

const perturb = (x: Float64Array, amp: number, seed: number) => {
  const R = rng(seed);
  const y = Float64Array.from(x);
  for (let q = 0; q < y.length; q++) y[q] += (R() * 2 - 1) * amp;
  return y;
};

describe('MEAM bcc reference (single element)', () => {
  it('reference shells: 8 nearest neighbours at r, 6 second neighbours screened to zero', () => {
    const nb = referenceVectors('bcc', 2.5, 4.0);
    const near = nb.filter((v) => Math.abs(v.r - 2.5) < 1e-9);
    expect(near.length).toBe(8);
    expect(nb.filter((v) => v.r > 2.5 + 1e-9).length).toBe(6);
  });

  it('perfect crystal energy per atom equals -Ec at the nearest-neighbour distance re', () => {
    const x = cells(BCC_BASIS, (2 * W_BCC.re) / Math.sqrt(3), 2);
    const a = (2 * W_BCC.re) / Math.sqrt(3);
    const E = meamEnergy(W_BCC, OPTS, x, [2 * a, 2 * a, 2 * a]);
    expect(E / 16).toBeCloseTo(-W_BCC.Ec, 9);
  });

  it('perfect crystals match native pe at three lattice constants (1e-8 relative)', () => {
    const native: Array<[number, number]> = [
      [2.8868, -124.560876810954],
      [3.16, -142.400000000091],
      [3.3466, -137.289643831942],
    ];
    for (const [a, pe] of native) {
      const x = cells(BCC_BASIS, a, 2);
      const E = meamEnergy(W_BCC, OPTS, x, [2 * a, 2 * a, 2 * a]);
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-8);
    }
  });

  it('dimers match native pe (angular terms included), 1e-8 relative', () => {
    const native: Array<[number, number]> = [
      [2.2, -7.92102130999959],
      [2.5, -8.89713603446008],
      [2.7, -8.61185420158656],
      [2.9, -8.00882116592403],
      [3.2, -6.91191704779047],
    ];
    for (const [r, pe] of native) {
      const x = Float64Array.from([0, 0, 0, r, 0, 0]);
      const E = meamEnergy(W_BCC, OPTS, x, [40, 40, 40]);
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-8);
    }
  });

  it('forces equal -grad E to 1e-6 relative for a perturbed bcc cell', () => {
    const x = perturb(cells(BCC_BASIS, 3.16, 2), 0.08, 7);
    expect(checkForces(W_BCC, x, [6.32, 6.32, 6.32])).toBeLessThan(1e-6);
  });

  it('forces for a bcc dimer and trimer', () => {
    expect(checkForces(W_BCC, Float64Array.from([0, 0, 0, 2.6, 0.2, 0]), [40, 40, 40])).toBeLessThan(1e-6);
    const tri = Float64Array.from([0, 0, 0, 2.5, 0, 0, 1.4, 2.2, 0.3]);
    expect(checkForces(W_BCC, tri, [40, 40, 40])).toBeLessThan(1e-6);
  });
});

describe('MEAM diamond reference (single element)', () => {
  it('reference shells: 4 nearest neighbours at r, 12 second neighbours screened to zero', () => {
    const nb = referenceVectors('dia', 2.165064, 4.0);
    expect(nb.filter((v) => Math.abs(v.r - 2.165064) < 1e-9).length).toBe(4);
    expect(nb.filter((v) => v.r > 2.2).length).toBe(12);
  });

  it('perfect diamond crystals match native pe, 1e-8 relative', () => {
    const native: Array<[number, number, number]> = [
      [5.0, 64, -294.400000000636],
      [5.3, 64, -283.526028000241],
    ];
    for (const [a, n, pe] of native) {
      const x = cells(DIA_BASIS, a, 2);
      const E = meamEnergy(W_DIA, OPTS, x, [2 * a, 2 * a, 2 * a]);
      expect(n).toBe(x.length / 3);
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-8);
    }
  });

  it('diamond-reference dimers and non-periodic clusters match native pe to 1e-10 relative', () => {
    // Measured with native LAMMPS (black box, lib W dia with the same synthetic parameters):
    // dimers at 2.2, 2.4, 2.6 A: -5.56337126219287, -5.09095676003276, -4.50106696309083 eV;
    // tetrahedral cluster (centre + 4 neighbours at re): -15.8322440471797 eV; the same with a second
    // neighbour at (2.5, 2.5, 0) A: -19.3138040281089 eV; with that atom displaced by (0.04, 0.02, -0.02) A
    // (position (2.54, 2.52, -0.02)): -19.2618591247003 eV.
    const dimers: Array<[number, number]> = [[2.2, -5.56337126219287], [2.4, -5.09095676003276], [2.6, -4.50106696309083]];
    for (const [r, pe] of dimers) {
      const E = meamEnergy(W_DIA, OPTS, Float64Array.from([0, 0, 0, r, 0, 0]), [40, 40, 40]);
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-10);
    }
    const t = 2.165064 / Math.sqrt(3);
    const tetra = [0, 0, 0, t, t, t, t, -t, -t, -t, t, -t, -t, -t, t];
    const L60: [number, number, number] = [60, 60, 60];
    expect(Math.abs(meamEnergy(W_DIA, OPTS, Float64Array.from(tetra), L60) + 15.8322440471797) / 15.8322440471797).toBeLessThan(1e-10);
    const plus = [...tetra, 2.5, 2.5, 0];
    expect(Math.abs(meamEnergy(W_DIA, OPTS, Float64Array.from(plus), L60) + 19.3138040281089) / 19.3138040281089).toBeLessThan(1e-10);
    const disp = [...tetra, 2.54, 2.52, -0.02];
    expect(Math.abs(meamEnergy(W_DIA, OPTS, Float64Array.from(disp), L60) + 19.2618591247003) / 19.2618591247003).toBeLessThan(1e-10);
  });

  it('forces equal -grad E to 1e-6 relative for a perturbed diamond cell', () => {
    // one 8-atom cell with periodic images (ceil(rc/L) = 1 image shell)
    const x = perturb(cells(DIA_BASIS, 5.0, 1), 0.06, 13);
    expect(checkForces(W_DIA, x, [5.0, 5.0, 5.0])).toBeLessThan(1e-6);
  });
});

describe('MEAM reference lattices: parameter and library handling', () => {
  const LIB_BCC = "'W' 'bcc' 8 74 183.84 5.0 2.0 1.5 2.5 4.0 3.16 8.9 1.0 1.0 2.0 1.0 1.0 1.0 0\n";
  const PAR_BCC = 'rc = 4.0\ndelr = 0.1\nEc(1,1) = 8.9\nre(1,1) = 2.73664\nalpha(1,1) = 5.0\nlattce(1,1) = bcc\nzbl(1,1) = 0\n';

  it('bcc and dia library entries are accepted; lattices other than fcc, bcc, dia, hcp and sc are StyleErrors', () => {
    expect(parseMeamLibrary(LIB_BCC, 'W', 'lib').lat).toBe('bcc');
    expect(() => parseMeamParams(PAR_BCC.replace('bcc', 'c11'), 'par')).toThrow(StyleError);
    expect(() => parseMeamParams(PAR_BCC.replace('bcc', 'b1'), 'par')).toThrow(/b1/);
    expect(() => parseMeamParams(PAR_BCC.replace('bcc', 'dim'), 'par')).toThrow(/dim/);
  });

  it('lattce(1,1) must match the library lattice', () => {
    expect(parseMeamParams(PAR_BCC, 'par').lattce).toBe('bcc');
  });
});

/*
 * Multi-element MEAM (meam_alloy.ts) with the synthetic A-B entries of tests/oracle/w15meam_alloy_b1_*.
 * Native reference values (measured with LAMMPS as a black box, same parameters): A-B dimers at 2.2, 2.6 and
 * 3.0 A have pe -3.09974618373519, -3.15152249979223 and -2.64815172340702 eV; A-A dimers at 2.2 and 3.0 A
 * -2.96854645415178 and -2.08217015898284 eV; B-B dimers at 2.2 and 3.0 A -3.82225743844039 and
 * -2.74459781879424 eV. Non-periodic clusters, types [A,B,A] at (0,0,0), (2.4,0,0), (1.2,2.0,0.3): -6.23612331013911 eV;
 * [B,A,A,B] at (0,0,0), (2.5,0.2,0), (1.3,2.2,0.4), (1.1,0.7,2.5): -9.05192035780966 eV;
 * [A,A,B,B] at (0,0,0), (2.6,0,0), (1.3,2.3,0), (1.0,0.8,2.8): -8.28828893530625 eV.
 * B1 crystal (8 atoms, cubic cell a = 5.2 A, nearest-neighbour distance re(1,2)): pe -30.400000000089 eV.
 */

const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0] };
const B_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0] };
const AB_MODEL: AlloyModel = makeAlloyModel(
  [A_EL, B_EL],
  [
    [
      { Ec: 3.5, re: 2.5, alpha: 5.0, lat: 'self' },
      { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1' },
    ],
    [
      { Ec: 3.8, re: 2.6, alpha: 4.5, lat: 'b1' },
      { Ec: 4.0, re: 2.7, alpha: 4.0, lat: 'self' },
    ],
  ] as AlloyPair[][],
  { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8 },
  true,
);

/** Non-periodic cluster energy and dE/dx (F = -grad) of the alloy model. */
const clusterAlloy = (model: AlloyModel, types: number[], X: number[]) => {
  const n = types.length;
  const F = new Float64Array(3 * n);
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: AlloyNeighbor[] = [];
    const idx: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = X[3 * j] - X[3 * i], dy = X[3 * j + 1] - X[3 * i + 1], dz = X[3 * j + 2] - X[3 * i + 2];
      const r = Math.hypot(dx, dy, dz);
      if (r >= model.opts.rc) continue;
      nb.push({ e: types[j], j, dx, dy, dz, r });
      idx.push(j);
    }
    const g = new Float64Array(3 * nb.length);
    E += alloyAtomEnergyGrad(model, types[i], nb, g);
    nb.forEach((p, m) => {
      for (let c = 0; c < 3; c++) {
        F[3 * p.j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    });
  }
  return { E, F };
};

/** Periodic orthorhombic energy and forces of the alloy model (neighbours from all images within rc). */
const periodicAlloy = (model: AlloyModel, types: number[], x: Float64Array, L: number[]) => {
  const n = types.length;
  const nimg = L.map((l) => Math.ceil(model.opts.rc / l));
  const F = new Float64Array(3 * n);
  let E = 0;
  for (let i = 0; i < n; i++) {
    const nb: AlloyNeighbor[] = [];
    for (let j = 0; j < n; j++)
      for (let ia = -nimg[0]; ia <= nimg[0]; ia++)
        for (let ib = -nimg[1]; ib <= nimg[1]; ib++)
          for (let ic = -nimg[2]; ic <= nimg[2]; ic++) {
            if (i === j && !ia && !ib && !ic) continue;
            const dx = x[3 * j] - x[3 * i] + ia * L[0], dy = x[3 * j + 1] - x[3 * i + 1] + ib * L[1], dz = x[3 * j + 2] - x[3 * i + 2] + ic * L[2];
            const r = Math.hypot(dx, dy, dz);
            if (r >= model.opts.rc) continue;
            nb.push({ e: types[j], j, dx, dy, dz, r });
          }
    const g = new Float64Array(3 * nb.length);
    E += alloyAtomEnergyGrad(model, types[i], nb, g);
    nb.forEach((p, m) => {
      for (let c = 0; c < 3; c++) {
        F[3 * p.j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    });
  }
  return { E, F };
};

/** B1 cell: A (type 0) on fcc sites, B (type 1) on the octahedral sites, 8 atoms in a cubic cell of side a. */
const b1Cell = (a: number) => {
  const fcc = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
  const oct = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]];
  const X: number[] = [];
  const types: number[] = [];
  for (const p of fcc) { X.push(p[0] * a, p[1] * a, p[2] * a); types.push(0); }
  for (const p of oct) { X.push(p[0] * a, p[1] * a, p[2] * a); types.push(1); }
  return { X: Float64Array.from(X), types };
};

describe('multi-element MEAM (B1 alloy, fcc elements)', () => {
  it('dimers (A-B, A-A, B-B) match native pe to 1e-10 relative', () => {
    const cases: Array<[number[], number, number]> = [
      [[0, 1], 2.2, -3.09974618373519],
      [[0, 1], 2.6, -3.15152249979223],
      [[0, 1], 3.0, -2.64815172340702],
      [[0, 0], 2.2, -2.96854645415178],
      [[0, 0], 3.0, -2.08217015898284],
      [[1, 1], 2.2, -3.82225743844039],
      [[1, 1], 3.0, -2.74459781879424],
    ];
    for (const [types, r, pe] of cases) {
      const E = clusterAlloy(AB_MODEL, types, [0, 0, 0, r, 0, 0]).E;
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-10);
    }
  });

  it('A-B dimers inside the smoothing window [3.9, 4.0] A match native pe to 1e-10 relative', () => {
    // Measured with native LAMMPS (black box, w15meam_alloy_* entries): A-B dimers at 3.92, 3.95, 3.97, 3.99 A.
    const cases: Array<[number, number]> = [
      [3.92, -1.56249847692229],
      [3.95, -1.41962888480406],
      [3.97, -1.06207996048462],
      [3.99, -0.321820764013811],
    ];
    for (const [r, pe] of cases) {
      const E = clusterAlloy(AB_MODEL, [0, 1], [0, 0, 0, r, 0, 0]).E;
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-10);
    }
  });

  it('mixed clusters (t-averaging weighted by the partial densities) match native to 1e-9 relative', () => {
    const cases: Array<[number[], number[], number]> = [
      [[0, 1, 0], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3], -6.23612331013911],
      [[1, 0, 0, 1], [0, 0, 0, 2.5, 0.2, 0, 1.3, 2.2, 0.4, 1.1, 0.7, 2.5], -9.05192035780966],
      [[0, 0, 1, 1], [0, 0, 0, 2.6, 0, 0, 1.3, 2.3, 0, 1.0, 0.8, 2.8], -8.28828893530625],
    ];
    for (const [types, X, pe] of cases) {
      const E = clusterAlloy(AB_MODEL, types, X).E;
      expect(Math.abs(E - pe) / Math.abs(pe)).toBeLessThan(1e-9);
    }
  });

  it('B1 crystal (8 atoms, a = 5.2 A = 2 re(1,2)) has pe -30.400000000089 eV (periodic)', () => {
    const { X, types } = b1Cell(5.2);
    const E = periodicAlloy(AB_MODEL, types, X, [5.2, 5.2, 5.2]).E;
    expect(Math.abs(E + 30.4) / 30.4).toBeLessThan(1e-9);
  });

  it('forces equal -grad E to 1e-6 relative (mixed cluster and perturbed B1 cell)', () => {
    const X = [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2];
    const types = [0, 1, 0, 1];
    const { F } = clusterAlloy(AB_MODEL, types, X);
    const h = 1e-5;
    let worst = 0;
    for (let q = 0; q < X.length; q++) {
      const xp = X.slice(), xm = X.slice();
      xp[q] += h;
      xm[q] -= h;
      const fd = -(clusterAlloy(AB_MODEL, types, xp).E - clusterAlloy(AB_MODEL, types, xm).E) / (2 * h);
      worst = Math.max(worst, Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2));
    }
    expect(worst).toBeLessThan(1e-6);

    const cell = b1Cell(5.2);
    const R = rng(4);
    const x = Float64Array.from(cell.X, (v) => v + (R() * 2 - 1) * 0.05);
    const pf = periodicAlloy(AB_MODEL, cell.types, x, [5.2, 5.2, 5.2]);
    let worstP = 0;
    for (let q = 0; q < x.length; q++) {
      const xp = Float64Array.from(x), xm = Float64Array.from(x);
      xp[q] += h;
      xm[q] -= h;
      const fd = -(periodicAlloy(AB_MODEL, cell.types, xp, [5.2, 5.2, 5.2]).E - periodicAlloy(AB_MODEL, cell.types, xm, [5.2, 5.2, 5.2]).E) / (2 * h);
      worstP = Math.max(worstP, Math.abs(pf.F[q] - fd) / Math.max(Math.abs(fd), 1e-2));
    }
    expect(worstP).toBeLessThan(1e-6);
  });

  it('a one-element alloy model reproduces the single-element fcc energy and forces', () => {
    const one = makeAlloyModel(
      [{ z: 12, lat: 'fcc', re: 2.55, alpha: 4.95, Ec: 3.54, A: 1, beta: [2, 1, 2, 4], t: [1, 1, 1, 1] }],
      [[{ Ec: 3.54, re: 2.55, alpha: 4.95, lat: 'self' }]],
      { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8 },
      true,
    );
    const { x, L } = (() => {
      const a = 3.615, n = 2;
      const basis = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
      const pts: number[] = [];
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) for (const b of basis) pts.push((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a);
      const R = rng(7);
      for (let q = 0; q < pts.length; q++) pts[q] += (R() * 2 - 1) * 0.08;
      return { x: Float64Array.from(pts), L: [2 * a, 2 * a, 2 * a] as [number, number, number] };
    })();
    const types = Array.from({ length: x.length / 3 }, () => 0);
    const alloy = periodicAlloy(one, types, x, L);
    const single = meamEnergyForces({ z: 12, re: 2.55, alpha: 4.95, Ec: 3.54, A: 1, beta: [2, 1, 2, 4], t: [1, 1, 1, 1], ibar: 0 }, OPTS, x, L);
    expect(Math.abs(alloy.E - single.E) / Math.abs(single.E)).toBeLessThan(1e-12);
    let worst = 0;
    for (let q = 0; q < x.length; q++) worst = Math.max(worst, Math.abs(alloy.F[q] - single.F[q]));
    expect(worst).toBeLessThan(1e-10);
  });
});

describe('multi-element pair_coeff: supported subset and StyleErrors', () => {
  const LIB2 = "'A' 'fcc' 12 29 63.546 5.0 2.0 1.5 2.5 4.0 3.6 3.5 1.0 1.0 2.0 1.0 1.0 1.0 0\n'B' 'fcc' 12 28 58.69 4.0 3.0 1.0 2.0 3.0 3.5 4.0 1.0 1.0 1.5 0.5 2.0 1.0 0\n";
  const PAR2 = [
    'rc = 4.0', 'delr = 0.1', 'Ec(1,1) = 3.5', 're(1,1) = 2.5', 'alpha(1,1) = 5.0', 'Ec(2,2) = 4.0', 're(2,2) = 2.7',
    'alpha(2,2) = 4.0', 'Ec(1,2) = 3.8', 're(1,2) = 2.6', 'alpha(1,2) = 4.5', 'lattce(1,2) = b1', 'zbl(1,1) = 0',
    'zbl(2,2) = 0', 'zbl(1,2) = 0', 'augt1 = 1', '',
  ].join('\n');

  const ctxWith = (files: Record<string, string>) => ({
    s: null,
    readFile: (n: string) => {
      if (!(n in files)) throw new Error(`no file ${n}`);
      return files[n];
    },
    log: () => {},
  });
  const coeffWith = (par: string, args: string[], ntypes = 2, lib = LIB2) => {
    const files = { 'lib.meam': lib, 'par.meam': par };
    const p = new PairMeam();
    p.settings([], ctxWith(files));
    p.allocate(ntypes);
    p.coeff(args, ctxWith(files));
    return p;
  };
  const ARGS = ['*', '*', 'lib.meam', 'A', 'B', 'par.meam', 'A', 'B'];

  it('a two-element B1 alloy with fcc elements is accepted (types mapped to elements)', () => {
    expect(() => coeffWith(PAR2, ARGS)).not.toThrow();
    expect(() => coeffWith(PAR2, ['*', '*', 'lib.meam', 'A', 'B', 'par.meam', 'A', 'B', 'A'], 3)).not.toThrow();
  });

  it('lattce(1,2) other than b1 is a StyleError naming it (L12 is not verified)', () => {
    expect(() => coeffWith(PAR2.replace('b1', 'l12'), ARGS)).toThrow(/l12/);
    expect(() => coeffWith(PAR2.replace('b1', 'l12'), ARGS)).toThrow(StyleError);
  });

  it('missing pair parameters, unknown elements, delta and nn2 are StyleErrors; per-triplet screening is accepted', () => {
    expect(() => coeffWith(PAR2.replace('Ec(1,2) = 3.8\n', ''), ARGS)).toThrow(/Ec\(1,2\)/);
    expect(() => coeffWith(PAR2, ['*', '*', 'lib.meam', 'A', 'C', 'par.meam', 'A', 'C'])).toThrow(/multi-element/);
    expect(() => coeffWith(PAR2 + 'delta(1,2) = 0.1\n', ARGS)).toThrow(/delta/);
    expect(() => coeffWith(PAR2 + 'Cmax(1,1,2) = 3.0\n', ARGS)).not.toThrow();
    expect(() => coeffWith(PAR2 + 'nn2(1,2) = 1\n', ARGS)).toThrow(/nn2/);
  });

  it('an element with a non-fcc reference lattice is not supported in an alloy', () => {
    const libBcc = LIB2.replace("'A' 'fcc'", "'A' 'bcc'");
    expect(() => coeffWith(PAR2, ARGS, 2, libBcc)).toThrow(StyleError);
    expect(() => coeffWith(PAR2 + 'lattce(1,1) = bcc\n', ARGS)).toThrow(StyleError);
  });

  it('pair parameters must satisfy I<=J, and lattce(I,J) for I=J is fcc, bcc or dia', () => {
    expect(() => coeffWith(PAR2 + 'Ec(2,1) = 3.0\n', ARGS)).toThrow(/I<=J/);
    expect(() => parseMeamParams('rc = 4\ndelr = 0.1\nlattce(1,2) = b1\n', 'par', 1)).toThrow(/indexed/);
  });
});
