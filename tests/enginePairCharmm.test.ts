import { describe, expect, it } from 'vitest';
import { PairLJCharmmCoulCharmm, PairLJCharmmCoulCharmmImplicit, PairLJCharmmCoulLong } from '../src/engine/force/pair/charmm';
import { PAIRS } from '../src/engine/registry/pair_charmm';
import { newAccum, type PairCompute, type StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for the CHARMM pair styles: central finite differences proving
 * force = -dE/dr at three geometries per style (inside the inner cutoff, in
 * the switching region, near the outer cutoff), the stored epsilon14/sigma14,
 * the documented arithmetic mixing default, the native-measured Coulomb force
 * treatment (see charmm.ts: for lj/charmm/coul/charmm the switch multiplies
 * the Coulomb energy and the standard force, so only the LJ force and the
 * unswitched region satisfy F = -dE/dr), and StyleError on missing
 * coefficients / unsupported pair_style args / pair_modify options. The
 * oracle parity itself lives in tests/engineOracle.test.ts (w3charmm_*).
 */

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

const make = (st: PairLJCharmmCoulCharmm, settingsArgs: string[], coeffs: string[][], gEwald = 0): PairLJCharmmCoulCharmm => {
  st.allocate(Math.max(1, ...coeffs.map((c) => Math.max(Number(c[0]), Number(c[1])))));
  st.settings(settingsArgs);
  for (const c of coeffs) st.coeff(c);
  st.gEwald = gEwald;
  st.init(CTX);
  return st;
};

/** Two atoms on the x axis (r = x1 - x0), half neighbor list 0 -> 1; runs compute() once. */
const pairEnergyForce = (st: PairLJCharmmCoulCharmm, r: number, qi: number, qj: number) => {
  const x = new Float64Array([0, 0, 0, r, 0, 0]);
  const f = new Float64Array(6);
  const pc = {
    s: null, nb: null, geom: null,
    x, f,
    type: new Int32Array([1, 1]),
    q: new Float64Array([qi, qj]),
    nlocal: 2, nall: 2,
    half: { inum: 2, numneigh: new Int32Array([1, 0]), firstneigh: new Int32Array([0, 1]), neighbors: new Int32Array([1]) },
    full: null,
    specialLJ: new Float64Array([1, 1, 1, 1]),
    specialCoul: new Float64Array([1, 1, 1, 1]),
    qqrd2e: 332.06371,
    acc: newAccum(),
    eatom: new Float64Array(2),
    vatom: null,
  } as unknown as PairCompute;
  st.compute(pc);
  return { e: pc.acc.evdwl + pc.acc.ecoul, evdwl: pc.acc.evdwl, ecoul: pc.acc.ecoul, fx: f[0] };
};

/**
 * Central finite difference through compute(). Moving atom 0 by dx changes the
 * separation r = x1 - x0 by -dx, so F_0x = -dE/dx_0 = +dE/dr.
 */
const checkFD = (st: PairLJCharmmCoulCharmm, radii: readonly number[], qi = 0, qj = 0): void => {
  for (const r of radii) {
    const h = 1e-6 * r;
    const ePlus = pairEnergyForce(st, r + h, qi, qj).e;
    const eMinus = pairEnergyForce(st, r - h, qi, qj).e;
    const fd = (ePlus - eMinus) / (2 * h); // F_0x = +dE/dr
    const { fx } = pairEnergyForce(st, r, qi, qj);
    expect(Math.abs(fx - fd), `r=${r}: fx=${fx} fd=${fd}`).toBeLessThan(1e-5 * (1 + Math.abs(fd)));
  }
};

describe('pair style lj/charmm/coul/charmm', () => {
  it('LJ force = -dE/dr at three geometries (flat, switching region, near outer)', () => {
    const st = make(new PairLJCharmmCoulCharmm(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0'], ['2', '2', '0.1', '1.8']]);
    checkFD(st, [2.0, 4.0, 4.9]);
  });

  it('LJ force = -dE/dr in the unswitched region with charges (C = qqrd2e q q / r)', () => {
    const st = make(new PairLJCharmmCoulCharmm(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0']]);
    checkFD(st, [1.5, 2.9], 0.3, -0.3);
    checkFD(st, [1.5, 2.9], 0.3, 0.3);
    const { ecoul } = pairEnergyForce(st, 2.0, 0.3, -0.3);
    expect(ecoul).toBeCloseTo(332.06371 * 0.3 * -0.3 / 2.0, 10);
  });

  it('switched Coulomb: energy = C(r) S(r), force = S(r) F_std (native-measured)', () => {
    // eps = 0 -> pure Coulomb, matching the native two-atom measurements
    const st = make(new PairLJCharmmCoulCharmm(), ['8.0', '10.0'], [['1', '1', '0.0', '3.405']]);
    const k = 332.06371 * 0.09, a2 = 64, b2 = 100, d3 = 36 ** 3;
    for (const [r, eNat, fNat] of [
      [8.1, 3.66811839637, -0.452854123009],
      [9.0, 1.79853608361, -0.199837342624],
      [9.5, 0.567264126304, -0.0597120132951],
    ] as const) {
      const { ecoul, fx } = pairEnergyForce(st, r, 0.3, 0.3);
      const r2 = r * r;
      const s = (b2 - r2) ** 2 * (b2 + 2 * r2 - 3 * a2) / d3;
      expect(ecoul, `e r=${r}`).toBeCloseTo(k / r * s, 10);
      expect(ecoul, `e vs native r=${r}`).toBeCloseTo(eNat, 9);
      expect(fx, `f vs native r=${r}`).toBeCloseTo(fNat, 9);
    }
  });

  it('stores optional epsilon14/sigma14 and mixes with arithmetic default', () => {
    const st = make(new PairLJCharmmCoulCharmm(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0', '0.1', '1.9'], ['2', '2', '0.1', '1.8', '0.05', '1.7']]);
    expect(st.p.get('epsilon14', 1, 1)).toBe(0.1);
    expect(st.p.get('sigma14', 2, 2)).toBe(1.7);
    // pair_charmm.html: "The default mix value is *arithmetic*"; epsilon mixes geometrically.
    expect(st.mix).toBe('arithmetic');
    expect(st.p.get('epsilon', 1, 2)).toBeCloseTo(Math.sqrt(0.2 * 0.1), 12);
    expect(st.p.get('sigma', 1, 2)).toBeCloseTo(0.5 * (2.0 + 1.8), 12);
    expect(st.p.get('epsilon14', 1, 2)).toBeCloseTo(Math.sqrt(0.1 * 0.05), 12);
    expect(st.p.get('sigma14', 1, 2)).toBeCloseTo(0.5 * (1.9 + 1.7), 12);
    // without 14 coefficients they default to 0 (native zero-initializes them)
    const st2 = make(new PairLJCharmmCoulCharmm(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0'], ['2', '2', '0.1', '1.8']]);
    expect(st2.p.get('epsilon14', 1, 1)).toBe(0);
    expect(st2.p.get('sigma14', 1, 2)).toBe(0);
  });

  it('four cutoff arguments: separate Coulomb switching region', () => {
    const st = make(new PairLJCharmmCoulCharmm(), ['3.0', '5.0', '2.5', '4.5'], [['1', '1', '0.2', '2.0']]);
    checkFD(st, [1.5, 2.4], 0.3, -0.3);
    expect(st.extract('cut_coul')).toBe(4.5);
    // Coulomb switched between 2.5 and 4.5, LJ between 3.0 and 5.0 (both energy-switched)
    const k = 332.06371 * 0.09, r = 3.5;
    const { ecoul } = pairEnergyForce(st, r, 0.3, 0.3);
    const a2 = 2.5 ** 2, b2 = 4.5 ** 2, r2 = r * r;
    const s = (b2 - r2) ** 2 * (b2 + 2 * r2 - 3 * a2) / (b2 - a2) ** 3;
    expect(ecoul).toBeCloseTo(k / r * s, 10);
  });

  it('missing coefficients throw', () => {
    const st = new PairLJCharmmCoulCharmm();
    st.allocate(2);
    st.settings(['3.0', '5.0']);
    st.coeff(['1', '1', '0.2', '2.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });

  it('bad pair_style args and unsupported options throw', () => {
    const st = new PairLJCharmmCoulCharmm();
    st.allocate(1);
    expect(() => st.settings(['3.0'])).toThrow(/usage/);
    expect(() => st.settings(['3.0', '5.0', '2.5'])).toThrow(/usage/);
    expect(() => st.settings(['5.0', '3.0'])).toThrow(/inner cutoff .*outer/);
    expect(() => st.coeff(['1', '1', '0.2'])).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '0.2', '2.0', '0.1'])).toThrow(/usage/);
    st.settings(['3.0', '5.0']);
    st.coeff(['1', '1', '0.2', '2.0']);
    st.shift = true;
    expect(() => st.init(CTX)).toThrow(/shift/);
    st.shift = false;
    st.tail = true;
    expect(() => st.init(CTX)).toThrow(/tail/);
    st.tail = false;
    st.table = 0;
    expect(() => st.init(CTX)).toThrow(/table/);
  });
});

describe('pair style lj/charmm/coul/charmm/implicit', () => {
  it('force = -dE/dr at three geometries; Coulomb varies as 1/r^2 inside the inner cutoff', () => {
    const st = make(new PairLJCharmmCoulCharmmImplicit(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0']]);
    checkFD(st, [1.5, 4.0, 4.9], 0.3, -0.3);
    // E_coul = qqrd2e q1 q2 / r^2 for r <= inner
    const { e } = pairEnergyForce(st, 2.0, 0.3, -0.3);
    const u = pairEnergyForce(st, 2.0, 0, 0).e;
    expect(e - u).toBeCloseTo(332.06371 * 0.3 * -0.3 / (2.0 * 2.0), 10);
  });

  it('missing coefficients throw', () => {
    const st = new PairLJCharmmCoulCharmmImplicit();
    st.allocate(1);
    st.settings(['3.0', '5.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
  });
});

describe('pair style lj/charmm/coul/long', () => {
  it('force = -dE/dr at three geometries (switched LJ, damped Coulomb)', () => {
    const st = make(new PairLJCharmmCoulLong(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0']], 0.3);
    checkFD(st, [1.5, 4.0, 4.9], 0.3, -0.3);
  });

  it('separate Coulomb cutoff (3 args): plain damped cutoff, no switching', () => {
    const st = make(new PairLJCharmmCoulLong(), ['3.0', '5.0', '4.5'], [['1', '1', '0.2', '2.0']], 0.3);
    checkFD(st, [1.5, 3.5, 4.4], 0.3, -0.3);
    expect(st.extract('cut_coul')).toBe(4.5);
  });

  it('supports pair_modify table 0 (polynomial erfc)', () => {
    const st = make(new PairLJCharmmCoulLong(), ['3.0', '5.0'], [['1', '1', '0.2', '2.0']], 0.3);
    st.table = 0;
    st.init(CTX);
    checkFD(st, [4.0, 4.9], 0.3, -0.3);
  });

  it('missing coefficients and bad args throw', () => {
    const st = new PairLJCharmmCoulLong();
    st.allocate(2);
    st.settings(['3.0', '5.0']);
    st.coeff(['1', '1', '0.2', '2.0']);
    expect(() => st.init(CTX)).toThrow(/all pair coeffs are not set/);
    expect(() => st.settings(['3.0', '5.0', '4.0', '4.5'])).toThrow(/usage/);
    expect(() => st.settings(['5.0', '3.0'])).toThrow(/inner cutoff/);
  });
});

describe('registry', () => {
  it('registers the three CHARMM styles', () => {
    for (const name of ['lj/charmm/coul/charmm', 'lj/charmm/coul/charmm/implicit', 'lj/charmm/coul/long']) {
      const st = PAIRS[name]!();
      expect(st.name).toBe(name);
    }
  });
});
