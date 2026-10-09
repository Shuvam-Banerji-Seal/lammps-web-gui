import { describe, expect, it } from 'vitest';
import {
  makeAlloyModel,
  alloyAtomEnergyGrad,
  type AlloyElement,
  type AlloyModel,
  type AlloyNeighbor,
  type AlloyPair,
} from '../src/engine/force/pair/meam_alloy';
import { DEFAULT_MEAM_OPTIONS, meamEnergy, parseMeamParams, type MeamElement } from '../src/engine/force/pair/meam';

/*
 * Multi-element MEAM, meam35 scope: the element density scaling rozero (library column 18) and its
 * parameter-file override rho0(I), plus the embedding reference density of the diamond reference with
 * ibar = 1 / 3 (needed by the real SiC potential).
 *
 * Density scaling: docs.lammps.org/pair_meam.html "The *rozero* parameter is an element-dependent
 * density scaling that weights the reference background density" and "rho0(I) = relative density for
 * element I (overwrites value read from meamf file)". Measured with native LAMMPS (black box): the
 * scaling multiplies the atomic electron density contributed by a neighbour of the element, so a single
 * element and the perfect B1 crystal are invariant while a mixed cluster changes.
 *
 * Embedding reference density (the denominator of F): measured with native LAMMPS (black box) on a
 * synthetic diamond element (S dia, z = 4, re = 2, alpha = 5, Ec = 4, beta = [3,2,2,2], t = [1,1,0.5,0.3]).
 * The tetrahedral reference has Gamma_ref != 0 through the third moment, so the raw rho0 and the full
 * background rho0*G(Gamma_ref) differ; the ibar = 1 dimer matches native only with the G_ref factor,
 * the ibar = 0 dimer only without it. An fcc reference has Gamma_ref = 0 and cannot tell them apart.
 *
 * All numbers below are native LAMMPS (black box) output of the probes in plans/scratch/w35meamrho and
 * of the oracle cases (tests/oracle/w35meamrho_*). The periodic SiC parity is in engineOracle.test.ts.
 */

/** Non-periodic cluster energy and F = -grad E of an alloy model (box large enough that rc cuts it). */
const clusterAlloy = (model: AlloyModel, types: number[], X: number[]) => {
  const n = types.length;
  const F = new Float64Array(3 * n);
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
    nb.forEach((p, m) => {
      for (let c = 0; c < 3; c++) {
        F[3 * p.j + c] -= g[3 * m + c];
        F[3 * i + c] += g[3 * m + c];
      }
    });
  }
  return { E, F };
};

const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);

const A_EL: AlloyElement = { z: 12, lat: 'fcc', re: 2.5, alpha: 5.0, Ec: 3.5, A: 1, beta: [2.0, 1.5, 2.5, 4.0], t: [1, 2.0, 1.0, 1.0], ibar: 0 };
const bEl = (rozero: number): AlloyElement => ({ z: 12, lat: 'fcc', re: 2.7, alpha: 4.0, Ec: 4.0, A: 1, beta: [3.0, 1.0, 2.0, 3.0], t: [1, 1.5, 0.5, 2.0], ibar: 0, rozero });
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
const modelWith = (rozeroB: number): AlloyModel =>
  makeAlloyModel([A_EL, bEl(rozeroB)], PAIRS, { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8 }, true);

describe('multi-element MEAM: element density scaling rozero / rho0(I)', () => {
  // Measured with native LAMMPS (black box): A-B dimer at r A with rozero(B) = rz, pe in eV.
  const DIMER: Array<[number, number, number]> = [
    [1.0, 2.0, -2.46023965769572], [1.0, 2.2, -3.09974618373519], [1.0, 2.4, -3.25526687222797],
    [1.0, 3.0, -2.64815172340702], [1.0, 3.5, -1.98901594647507],
    [2.0, 2.0, -2.72321261663502], [2.0, 2.4, -3.36918255365997], [2.0, 3.0, -2.68830698326347], [2.0, 3.5, -2.03376827670451],
    [3.5, 2.0, -3.07665733636415], [3.5, 2.4, -3.5366621943715], [3.5, 3.0, -2.73883384026872], [3.5, 3.5, -2.07643864716689],
    [2.25, 2.0, -2.79204347960128], [2.25, 2.4, -3.40492086964905], [2.25, 3.0, -2.70400952839182], [2.25, 3.5, -2.048145083756],
  ];
  for (const [rz, r, pe] of DIMER) {
    it(`A-B dimer at ${r} A with rozero(B) = ${rz} matches native pe to 1e-9 relative`, () => {
      const E = clusterAlloy(modelWith(rz), [0, 1], [0, 0, 0, r, 0, 0]).E;
      expect(rel(E, pe)).toBeLessThan(1e-9);
    });
  }

  // Measured with native LAMMPS (black box): A-B-A trimer at (0,0,0), (2.4,0,0), (1.2,2.0,0.3), pe in eV.
  const TRIMER: Array<[number, number]> = [
    [1.0, -6.23612331013911],
    [2.0, -6.48046168353576],
    [3.5, -6.76079986877474],
    [2.25, -6.54402789282861],
  ];
  for (const [rz, pe] of TRIMER) {
    it(`A-B-A trimer with rozero(B) = ${rz} matches native pe to 1e-9 relative`, () => {
      const E = clusterAlloy(modelWith(rz), [0, 1, 0], [0, 0, 0, 2.4, 0, 0, 1.2, 2.0, 0.3]).E;
      expect(rel(E, pe)).toBeLessThan(1e-9);
    });
  }

  it('rho0(2) = 2.25 with library rozero 1 matches native (the override)', () => {
    // The [2.25, ...] DIMER rows above come from par_b1_rho225.meam on top of a library rozero of 1.
    expect(rel(clusterAlloy(modelWith(2.25), [0, 1], [0, 0, 0, 2.4, 0, 0]).E, -3.40492086964905)).toBeLessThan(1e-9);
  });

  it('the parser records rho0(I) and strips quotes around a keyword value', () => {
    const par = parseMeamParams("rc = 4\ndelr = 0.1\nrho0(2) = 2.25\nlattce(1,2) = 'dia'\nEc(1,2) = 4.0\nre(1,2) = 2.0\nalpha(1,2) = 4.0\n", 'par', 2);
    expect(par.rho0.get(2)).toBe(2.25);
    expect(par.pair.get('1,2')?.lattce).toBe('dia');
  });

  it('a single element is invariant under rozero (the scaling cancels exactly)', () => {
    const withA = (rz: number) =>
      makeAlloyModel([{ ...A_EL, rozero: rz }, bEl(1.0)], PAIRS, { rc: 4.0, delr: 0.1, Cmin: 2.0, Cmax: 2.8 }, true);
    const a = clusterAlloy(withA(1.0), [0, 0], [0, 0, 0, 2.5, 0, 0]).E;
    const b = clusterAlloy(withA(2.0), [0, 0], [0, 0, 0, 2.5, 0, 0]).E;
    expect(a).toBe(b);
  });

  it('forces equal -grad E to 1e-6 relative for a mixed cluster with rozero(B) = 2.25', () => {
    const model = modelWith(2.25);
    const X = [0, 0, 0, 2.4, 0.1, 0, 1.2, 2.0, 0.3, 1.0, -0.2, 2.2];
    const types = [0, 1, 0, 1];
    const { F } = clusterAlloy(model, types, X);
    const h = 1e-5;
    let worst = 0;
    for (let q = 0; q < X.length; q++) {
      const xp = X.slice(), xm = X.slice();
      xp[q] += h;
      xm[q] -= h;
      const fd = -(clusterAlloy(model, types, xp).E - clusterAlloy(model, types, xm).E) / (2 * h);
      worst = Math.max(worst, Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2));
    }
    expect(worst).toBeLessThan(1e-6);
  });
});

describe('single-element MEAM: diamond reference with ibar = 1 / 3', () => {
  const el = (ibar: number, t3 = 0.3): MeamElement => ({
    z: 4, re: 2.0, alpha: 5.0, Ec: 4.0, A: 1, beta: [3, 2, 2, 2], t: [1, 1.0, 0.5, t3], ibar, lat: 'dia',
  });
  const L: [number, number, number] = [100, 100, 100];
  const dimer = (e: MeamElement, r: number) => meamEnergy(e, { ...DEFAULT_MEAM_OPTIONS }, new Float64Array([0, 0, 0, r, 0, 0]), L);

  // Measured with native LAMMPS (black box): S-S diamond dimer, pe in eV.
  const DIA: Array<[number, number, number]> = [
    [0, 2.0, -4.99465376351151], [0, 2.4, -3.62126140055328], [0, 3.0, -2.01016261772884],
    [1, 2.0, -4.63946910167441], [1, 2.4, -3.69698941716855], [1, 3.0, -2.67332894409027],
  ];
  for (const [ibar, r, pe] of DIA) {
    it(`diamond dimer ibar = ${ibar} at ${r} A matches native pe to 1e-12 relative`, () => {
      expect(rel(dimer(el(ibar), r), pe)).toBeLessThan(1e-12);
    });
  }

  // Measured with native LAMMPS (black box): large third moment (t3 = 3) makes Gamma_ref large.
  const DIA_BIG: Array<[number, number, number]> = [
    [0, 2.0, -5.19660944713074], [0, 2.4, -4.01906138084711], [0, 3.0, -2.41116852436907],
    [3, 2.0, -4.94269943182461], [3, 2.4, -3.47450974876774], [3, 3.0, -1.52596762559122],
  ];
  for (const [ibar, r, pe] of DIA_BIG) {
    it(`diamond dimer with t3 = 3, ibar = ${ibar} at ${r} A matches native pe to 1e-12 relative`, () => {
      expect(rel(dimer(el(ibar, 3.0), r), pe)).toBeLessThan(1e-12);
    });
  }

  it('an fcc reference has Gamma_ref = 0, so ibar 0 and 1 differ only through the density G', () => {
    const fcc: MeamElement = { z: 12, re: 2.0, alpha: 5.0, Ec: 4.0, A: 1, beta: [3, 2, 2, 2], t: [1, 1.0, 0.5, 0.3], ibar: 1, lat: 'fcc' };
    // Measured with native LAMMPS (black box): fcc dimer ibar = 1, pe in eV.
    const NAT: Array<[number, number]> = [[2.0, -3.18342255666494], [2.4, -2.5803355638014], [3.0, -2.38558355907171]];
    for (const [r, pe] of NAT) expect(rel(dimer(fcc, r), pe)).toBeLessThan(1e-12);
  });
});

