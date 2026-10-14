import { describe, expect, it } from 'vitest';
import { glNodes, glWeights, readMesocntFile, sCubic, spline1, dspline1, spline2, splineCoeff1 } from '../src/engine-gpl/mesont/tables';
import { BondMesoCNT } from '../src/engine-gpl/mesont/bond_mesocnt';
import type { StyleContext } from '../src/engine/force/types';

/* Unit checks of the MESONT port (src/engine-gpl/mesont): the quadrature, the spline evaluators and the bond preset.
 * The native parity is in tests/oracle/w157mt_*.in (npx vitest run tests/engineOracle -t w157mt). */

describe('mesont quadrature and switches', () => {
  for (const quad of [10, 129]) {
    it(`Gauss-Legendre nodes and weights of order ${quad} integrate x^0 and x^2 exactly`, () => {
      const x = glNodes(quad);
      const w = glWeights(quad, x);
      let s0 = 0, s2 = 0;
      for (let i = 0; i < quad; i++) {
        s0 += w[i];
        s2 += w[i] * x[i] * x[i];
      }
      expect(Math.abs(s0 - 2)).toBeLessThan(1e-12);
      expect(Math.abs(s2 - 2 / 3)).toBeLessThan(1e-10);
      for (let i = 0; i < quad; i++) expect(Math.abs(x[i] + x[quad - 1 - i])).toBeLessThan(1e-12);
    });
  }

  it('the cubic switch is 1 below 0, 0 above 1 and 1/2 at 1/2', () => {
    expect(sCubic(-0.3)).toBe(1);
    expect(sCubic(1.4)).toBe(0);
    expect(sCubic(0.5)).toBeCloseTo(0.5, 15);
  });
});

describe('mesont splines', () => {
  it('a 1D cubic spline reproduces linear data and its slope', () => {
    const n = 6, dx = 0.5, xstart = 1;
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) u[i] = 3 * (xstart + i * dx) - 2;
    const c = splineCoeff1(u, dx, n);
    for (const x of [1.2, 1.75, 2.4]) {
      expect(spline1(x, xstart, dx, c, n)).toBeCloseTo(3 * x - 2, 12);
      expect(dspline1(x, xstart, dx, c, n)).toBeCloseTo(3, 12);
    }
  });

  it('the potential-file reader takes the header and the tables of a small file', () => {
    // title line, sizes (uinf, gamma, phi, usemi), then r sigma delta1 delta2, then the four tables
    const lines = ['title', '3 3 3 3', '6.5 3.4 1 2'];
    for (let i = 0; i < 3; i++) lines.push(`${1 + i * 0.5} ${2 * (1 + i * 0.5)}`);
    for (let i = 0; i < 3; i++) lines.push(`${1 + i * 0.5} ${3 * (1 + i * 0.5)}`);
    const table2 = (): void => {
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const x = 1 + i * 0.5, y = 2 + j * 0.25;
          lines.push(`${x} ${y} ${x + 2 * y}`);
        }
      }
    };
    table2();
    table2();
    const tab = readMesocntFile(lines.join('\n'));
    expect(tab.uinfPoints).toBe(3);
    expect(tab.rAng).toBe(6.5);
    expect(tab.delhUinf).toBeCloseTo(0.5, 15);
    expect(tab.delhPhi).toBeCloseTo(0.5, 15);
    expect(tab.delpsiPhi).toBeCloseTo(0.25, 15);
    // bilinear-in-x,y data: the bicubic spline reproduces x + 2y at an interior point
    const v = spline2(1.6, 2.3, tab.hstartPhi, tab.psistartPhi, tab.delhPhi, tab.delpsiPhi, tab.phiCoeff, tab.phiPoints);
    expect(v).toBeCloseTo(1.6 + 2 * 2.3, 10);
  });
});

describe('mesont bond_style mesocnt preset', () => {
  it('C 10 10 r0 gives K = 0.5 (86.64 + 100.56 r) / r0 in metal units (bond_mesocnt.cpp coeff)', () => {
    const bond = new BondMesoCNT();
    bond.allocate(1);
    const ctx = { s: { units: { style: 'metal' } } } as unknown as StyleContext;
    bond.coeff(['1', 'C', '10', '10', '10'], ctx);
    const rAng = Math.sqrt(3 * (100 + 100 + 100)) * 1.421 / (2 * Math.PI);
    const k = bond.params.p('K')[1], r0 = bond.params.p('r0')[1];
    expect(r0).toBe(10);
    expect(k).toBeCloseTo(0.5 * (86.64 + 100.56 * rAng) / 10, 12);
  });

  it('refuses an unknown preset by name', () => {
    const bond = new BondMesoCNT();
    bond.allocate(1);
    const ctx = { s: { units: { style: 'metal' } } } as unknown as StyleContext;
    expect(() => bond.coeff(['1', 'graphite', '1', '2'], ctx)).toThrow(/Unknown graphite preset/);
  });
});
