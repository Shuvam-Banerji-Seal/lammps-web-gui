import { describe, expect, it } from 'vitest';
import { PairLJRelRes } from '../src/engine/force/pair/lj_relres';
import { PAIRS } from '../src/engine/registry/pair_relres';
import type { StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for pair_style lj/relres (docs.lammps.org/pair_lj_relres.html):
 * central finite differences prove force = -dE/dr in each of the four regions
 * (FG, FG->CG switch, CG, CG->zero cut), energy and force are continuous at
 * every switching boundary (energy vanishes at r_co with pair_modify shift yes),
 * mixing of unset cross terms, the ordinary-site cutoff rule, and StyleError on
 * bad arguments and unsupported options. The oracle parity against native LAMMPS
 * lives in tests/engineOracle.test.ts (w6relres_*).
 */

const CTX: StyleContext = { s: null, readFile: () => { throw new Error('no files in unit test'); }, log: () => {} };

// Hybrid type 1 (FG + CG) and ordinary type 2 (FG only) with the pair_style radii 3.0 3.5 6.0 7.0.
const GLOBAL = ['3.0', '3.5', '6.0', '7.0'];
const COEFFS = [
  ['1', '1', '1.0', '1.0', '0.5', '1.2'],
  ['2', '2', '0.8', '1.05', '0.0', '0.0'],
];

const make = (opts: { settings?: string[]; coeffs?: string[][]; shift?: boolean; tail?: boolean } = {}): PairLJRelRes => {
  const st = new PairLJRelRes();
  st.allocate(2);
  st.settings(opts.settings ?? GLOBAL, CTX);
  for (const c of opts.coeffs ?? COEFFS) st.coeff(c, CTX);
  st.shift = opts.shift ?? true;
  st.tail = opts.tail ?? false;
  st.init(CTX);
  return st;
};

const energy = (st: PairLJRelRes, itype: number, jtype: number, r: number): number =>
  st.single(0, 0, itype, jtype, r * r, 1, 1).eng;

/** Force F(r) = -dE/dr from the style's fforce (fforce = F / r). */
const force = (st: PairLJRelRes, itype: number, jtype: number, r: number): number =>
  st.single(0, 0, itype, jtype, r * r, 1, 1).fforce * r;

/** Central finite difference: F(r) must equal -dE/dr. */
const checkFD = (st: PairLJRelRes, itype: number, jtype: number, radii: readonly number[]): void => {
  for (const r of radii) {
    const h = 1e-6;
    const fd = -(energy(st, itype, jtype, r + h) - energy(st, itype, jtype, r - h)) / (2 * h);
    expect(Math.abs(force(st, itype, jtype, r) - fd), `r=${r}`).toBeLessThan(1e-6 * (1 + Math.abs(fd)));
  }
};

describe('pair style lj/relres', () => {
  it('is registered under its documented name', () => {
    expect(Object.keys(PAIRS)).toContain('lj/relres');
    expect(PAIRS['lj/relres']()).toBeInstanceOf(PairLJRelRes);
  });

  it('force = -dE/dr in the FG region (r < r_si), hybrid type', () => {
    checkFD(make(), 1, 1, [0.95, 1.6, 2.4, 2.9]);
  });

  it('force = -dE/dr inside the FG to CG switching range [r_si, r_so)', () => {
    checkFD(make(), 1, 1, [3.05, 3.2, 3.4]);
  });

  it('force = -dE/dr in the CG region [r_so, r_ci)', () => {
    checkFD(make(), 1, 1, [3.6, 4.5, 5.9]);
  });

  it('force = -dE/dr inside the cut switching range [r_ci, r_co)', () => {
    checkFD(make(), 1, 1, [6.1, 6.5, 6.95]);
  });

  it('force = -dE/dr for an ordinary type (eps^CG = 0) with its own cutoffs', () => {
    checkFD(make(), 2, 2, [0.9, 1.5, 2.8, 3.2]);
  });

  it('energy and force are continuous at r_si (FG to switch)', () => {
    const st = make();
    const eps = 1e-9;
    expect(Math.abs(energy(st, 1, 1, 3 - eps) - energy(st, 1, 1, 3 + eps))).toBeLessThan(1e-7);
    expect(Math.abs(force(st, 1, 1, 3 - eps) - force(st, 1, 1, 3 + eps))).toBeLessThan(1e-6);
  });

  it('energy and force are continuous at r_so (switch to CG)', () => {
    const st = make();
    const eps = 1e-9;
    expect(Math.abs(energy(st, 1, 1, 3.5 - eps) - energy(st, 1, 1, 3.5 + eps))).toBeLessThan(1e-7);
    expect(Math.abs(force(st, 1, 1, 3.5 - eps) - force(st, 1, 1, 3.5 + eps))).toBeLessThan(1e-6);
  });

  it('energy and force are continuous at r_ci (CG to cut)', () => {
    const st = make();
    const eps = 1e-9;
    expect(Math.abs(energy(st, 1, 1, 6 - eps) - energy(st, 1, 1, 6 + eps))).toBeLessThan(1e-7);
    expect(Math.abs(force(st, 1, 1, 6 - eps) - force(st, 1, 1, 6 + eps))).toBeLessThan(1e-6);
  });

  it('shift yes: energy and force vanish at r_co from inside', () => {
    const st = make();
    const eps = 1e-9;
    expect(Math.abs(energy(st, 1, 1, 7 - eps))).toBeLessThan(1e-7);
    expect(Math.abs(force(st, 1, 1, 7 - eps))).toBeLessThan(1e-6);
    expect(energy(st, 1, 1, 7 + eps)).toBe(0);
  });

  it('shift no: the energy at r_co is not forced to zero, force still vanishes', () => {
    const st = make({ shift: false });
    expect(Math.abs(energy(st, 1, 1, 7 - 1e-9))).toBeGreaterThan(1e-9);
    expect(Math.abs(force(st, 1, 1, 7 - 1e-9))).toBeLessThan(1e-6);
  });

  it('energy is continuous across r_si = r_so (no switching range)', () => {
    const st = make({ settings: ['3.0', '3.0', '6.0', '7.0'] });
    const eps = 1e-9;
    expect(Math.abs(energy(st, 1, 1, 3 - eps) - energy(st, 1, 1, 3 + eps))).toBeLessThan(1e-7);
    checkFD(st, 1, 1, [2.2, 4.0, 6.5]);
  });

  it('ordinary type (eps^CG = 0) has zero energy and force beyond r_so', () => {
    const st = make();
    expect(energy(st, 2, 2, 3.6)).toBe(0);
    expect(force(st, 2, 2, 3.6)).toBe(0);
  });

  it('per-pair cutoffs override the pair_style radii (all four must be given)', () => {
    const st = make({ coeffs: [['1', '1', '1.0', '1.0', '0.5', '1.2', '2.0', '2.4', '3.0', '3.5'], ['2', '2', '0.8', '1.05', '0.0', '0.0']] });
    // 1-1 cut is 3.5, so the neighbour cutoff squared is 12.25
    expect(st.cutsq[1 * (st.ntypes + 1) + 1]).toBeCloseTo(3.5 * 3.5, 12);
    checkFD(st, 1, 1, [1.5, 2.2, 2.7, 3.2]);
  });

  it('mixes unset cross terms: geometric for sigma, eps and cutoffs (CG epsilon zero when one side is zero)', () => {
    const st = make({ coeffs: [['1', '1', '1.0', '1.0', '0.5', '1.2'], ['2', '2', '0.8', '1.05', '0.0', '0.0']] });
    const p = st.p;
    expect(p.get('epsilon', 1, 2)).toBeCloseTo(Math.sqrt(1.0 * 0.8), 12);
    expect(p.get('sigma', 1, 2)).toBeCloseTo(Math.sqrt(1.0 * 1.05), 12);
    expect(p.get('epsilon_cg', 1, 2)).toBe(0);
    expect(p.get('rsi', 1, 2)).toBeCloseTo(3.0, 12);
    expect(p.get('rco', 1, 2)).toBeCloseTo(7.0, 12);
  });

  it('mixed CG epsilon and sigma use the CG parameters (geometric by default)', () => {
    const st = make({ coeffs: [['1', '1', '1.0', '1.0', '0.5', '1.2'], ['2', '2', '0.8', '1.05', '0.3', '1.0']] });
    expect(st.p.get('epsilon_cg', 1, 2)).toBeCloseTo(Math.sqrt(0.5 * 0.3), 12);
    expect(st.p.get('sigma_cg', 1, 2)).toBeCloseTo(Math.sqrt(1.2 * 1.0), 12);
  });

  it('pair_modify tail yes is refused (the energy is smoothed to 0.0 at the cutoff)', () => {
    expect(() => make({ tail: true })).toThrow(/tail/);
  });

  it('argument errors name the problem', () => {
    const st = new PairLJRelRes();
    st.allocate(2);
    expect(() => st.settings(['3.0', '3.5', '6.0'], CTX)).toThrow(/usage/);
    expect(() => st.settings(['3.5', '3.0', '6.0', '7.0'], CTX)).toThrow(/Rsi <= Rso <= Rci <= Rco/);
    expect(() => st.settings(['3.0', '3.5', '3.2', '7.0'], CTX)).toThrow(/Rsi <= Rso <= Rci <= Rco/);
    expect(() => st.settings(['3.0', '3.5', '6.0', '5.0'], CTX)).toThrow(/Rsi <= Rso <= Rci <= Rco/);
    expect(() => st.settings(['0.0', '3.5', '6.0', '7.0'], CTX)).toThrow(/Rsi/);
    st.settings(GLOBAL, CTX);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '0.5'], CTX)).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '0.5', '1.2', '2.0', '2.4', '3.0'], CTX)).toThrow(/usage/);
    expect(() => st.coeff(['1', '1', '1.0', '1.0', '0.5', '1.2', '4.0', '3.0', '6.0', '7.0'], CTX)).toThrow(/Rsi <= Rso <= Rci <= Rco/);
  });

  it('a type pair with no coefficients and no mixing partner is an error at init', () => {
    const st = new PairLJRelRes();
    st.allocate(2);
    st.settings(GLOBAL, CTX);
    st.coeff(['1', '1', '1.0', '1.0', '0.5', '1.2'], CTX);
    expect(() => st.init(CTX)).toThrow(/not set/);
  });

  it('dataCoeffsIJ writes the effective radii (ordinary sites take r_ci = r_co = r_so)', () => {
    const st = make();
    const lines = st.dataCoeffsIJ();
    expect(lines[0].split(' ').slice(6)).toEqual(['3', '3.5', '6', '7']);
    expect(lines[2].split(' ').slice(6)).toEqual(['3', '3.5', '3.5', '3.5']);
  });
});
