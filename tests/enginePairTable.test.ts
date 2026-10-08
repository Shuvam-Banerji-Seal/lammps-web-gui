import { describe, expect, it } from 'vitest';
import { PairTable } from '../src/engine/force/pair/table';
import { StyleError, type StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for pair_style table (src/engine/force/pair/table.ts).
 *
 * For each interpolation style (lookup, linear): central finite differences
 * of the tabulated energy at three geometries must reproduce the tabulated
 * force (force = -dE/dr up to the table resolution), and incomplete
 * pair_coeff sets must throw a StyleError at init.
 *
 * The test table is an RSQ-grid table generated from an analytic
 * 12-6 LJ potential (eps = 1.5, sigma = 1.1) with the force column
 * F = -dE/dr, so energy and force are consistent by construction:
 *   E(r)  = 4*eps*((sigma/r)^12 - (sigma/r)^6)
 *   F(r)  = 24*eps*(2*(sigma/r)^12 - (sigma/r)^6)/r
 */

const EPS = 1.5;
const SIG = 1.1;
const RLO = 0.9;
const RHI = 2.6;
const NFILE = 4000;

const ljEnergy = (r: number): number => {
  const s = SIG / r;
  return 4 * EPS * (s ** 12 - s ** 6);
};
const ljForce = (r: number): number => {
  const s = SIG / r;
  return (24 * EPS * (2 * s ** 12 - s ** 6)) / r;
};

const tableText = (): string => {
  const out: string[] = ['# test table: analytic LJ 12-6', '', 'T1', `N ${NFILE} RSQ ${RLO} ${RHI}`, ''];
  const lo = RLO * RLO, hi = RHI * RHI;
  for (let k = 0; k < NFILE; k++) {
    const r = Math.sqrt(lo + ((hi - lo) * k) / (NFILE - 1));
    out.push(`${k + 1} ${r.toFixed(10)} ${ljEnergy(r).toExponential(16)} ${ljForce(r).toExponential(16)}`);
  }
  return out.join('\n') + '\n';
};

const ctx: StyleContext = {
  s: null,
  readFile: (name: string) => {
    if (name === 'lj.table') return tableText();
    throw new StyleError(`no such file: ${name}`);
  },
  log: () => {},
};

/** A style initialized with a full 2-type coeff set (types 1, 2). */
const makeStyle = (mode: 'lookup' | 'linear', ntable = NFILE): PairTable => {
  const p = new PairTable();
  p.settings([mode, String(ntable)]);
  p.allocate(2);
  p.coeff(['1', '1', 'lj.table', 'T1', '2.6'], ctx);
  p.coeff(['2', '2', 'lj.table', 'T1', '2.6'], ctx);
  p.coeff(['1', '2', 'lj.table', 'T1', '2.4'], ctx);
  p.init(ctx);
  return p;
};

describe('pair_style table unit tests', () => {
  for (const mode of ['lookup', 'linear'] as const) {
    describe(`force = -dE/dr (${mode})`, () => {
      const p = makeStyle(mode);
      const ntable = NFILE;
      const rsq1 = RLO * RLO;
      const delta = (RHI * RHI - RLO * RLO) / (ntable - 1);

      // three geometries: the rsq bin midpoints nearest to these distances
      const targets = [1.05, 1.5, 2.3];

      it.each(targets.map((r) => [r] as const))(`central finite differences at r ~ %d`, (target) => {
        const pMid = ((target * target - rsq1) / delta);
        const m = Math.max(1, Math.min(ntable - 2, Math.floor(mode === 'lookup' ? pMid : pMid)));
        const rsqMid = rsq1 + (m + 0.5) * delta;
        const rMid = Math.sqrt(rsqMid);
        let ePlus: number, eMinus: number, rPlus: number, rMinus: number, fforceMid: number;
        if (mode === 'lookup') {
          // the lookup energy is constant per bin; straddle the bin so the
          // difference spans the neighbouring entries (3/4 bin half-width)
          ePlus = p.single(0, 0, 1, 1, rsqMid + 0.75 * delta, 0, 1).eng;
          eMinus = p.single(0, 0, 1, 1, rsqMid - 0.75 * delta, 0, 1).eng;
          // the probes land in bins m+1 and m-1, whose entries sit at their midpoints
          rPlus = Math.sqrt(rsqMid + delta);
          rMinus = Math.sqrt(rsqMid - delta);
          fforceMid = p.single(0, 0, 1, 1, rsqMid, 0, 1).fforce;
        } else {
          const h = 1e-6 * rMid;
          ePlus = p.single(0, 0, 1, 1, rsqMid + 2 * rMid * h, 0, 1).eng;
          eMinus = p.single(0, 0, 1, 1, rsqMid - 2 * rMid * h, 0, 1).eng;
          rPlus = rMid + h;
          rMinus = rMid - h;
          fforceMid = p.single(0, 0, 1, 1, rsqMid, 0, 1).fforce;
        }
        const dEdr = (ePlus - eMinus) / (rPlus - rMinus);
        const force = fforceMid * rMid; // radial force -dE/dr = fforce * r
        expect(-dEdr).toBeCloseTo(force, mode === 'linear' ? 2 : 0); // energy and force are interpolated separately
        // and both track the analytic potential the table was generated from
        const analytic = ljForce(rMid);
        expect(force).toBeCloseTo(analytic, mode === 'linear' ? 2 : 0);
      });
    });

    it(`(${mode}) reproduces exact tabulated values at table points (linear) / spline values (lookup)`, () => {
      const p = makeStyle(mode);
      if (mode === 'linear') {
        // Ntable = Nfile with RSQ spacing: the table is the file itself
        const k = 1500;
        const rsq = RLO * RLO + ((RHI * RHI - RLO * RLO) * k) / (NFILE - 1);
        const r = Math.sqrt(rsq);
        const { eng, fforce } = p.single(0, 0, 1, 1, rsq, 0, 1);
        expect(eng).toBeCloseTo(ljEnergy(r), 12);
        expect(fforce * r).toBeCloseTo(ljForce(r), 12);
      } else {
        // lookup picks the single entry spline-sampled at the bin midpoint
        const delta = (RHI * RHI - RLO * RLO) / (NFILE - 1);
        const rsqMid = RLO * RLO + (1500.5) * delta;
        const { eng } = p.single(0, 0, 1, 1, rsqMid - 0.1 * delta, 0, 1);
        const { eng: eng2 } = p.single(0, 0, 1, 1, rsqMid + 0.1 * delta, 0, 1);
        expect(eng).toBe(eng2); // same bin -> same entry
        expect(eng).toBeCloseTo(ljEnergy(Math.sqrt(rsqMid)), 6);
      }
    });
  }

  it('linear interpolation is exact between table points for a consistent table', () => {
    const p = makeStyle('linear');
    // between two table points the interpolated energy is linear in rsq and
    // the force is the interpolated F/r times r; both track the analytic LJ
    for (const r of [1.234, 1.778, 2.213]) {
      const { eng, fforce } = p.single(0, 0, 1, 1, r * r, 0, 1);
      expect(eng).toBeCloseTo(ljEnergy(r), 4);
      expect(fforce * r).toBeCloseTo(ljForce(r), 4);
    }
  });

  it('missing pair coefficients throw at init', () => {
    const p = new PairTable();
    p.settings(['linear', '500']);
    p.allocate(2);
    p.coeff(['1', '1', 'lj.table', 'T1', '2.6'], ctx);
    // pair (2,2) and (1,2) unset: "This pair style does not support mixing.
    // Thus, coefficients for all I,J pairs must be specified explicitly."
    expect(() => p.init(ctx)).toThrow(StyleError);
    expect(() => p.init(ctx)).toThrow(/all pair coeffs are not set/);
  });

  it('coefficients for one type pair do not leak to the cross pair', () => {
    const p = new PairTable();
    p.settings(['linear', '500']);
    p.allocate(2);
    p.coeff(['1', '1', 'lj.table', 'T1', '2.6'], ctx);
    p.coeff(['2', '2', 'lj.table', 'T1', '2.6'], ctx);
    expect(() => p.init(ctx)).toThrow(StyleError);
  });

  it('spline and bitmap interpolation styles are rejected by name', () => {
    const p = new PairTable();
    expect(() => p.settings(['spline', '100'])).toThrow(StyleError);
    expect(() => p.settings(['spline', '100'])).toThrow(/spline/);
    expect(() => p.settings(['bitmap', '12'])).toThrow(/bitmap/);
  });

  it('unknown pair_style keywords are rejected by name', () => {
    const p = new PairTable();
    expect(() => p.settings(['linear', '100', 'pppm'])).toThrow(/pppm/);
    expect(() => p.settings(['linear', '100', 'ewald'])).toThrow(StyleError);
    expect(() => p.settings(['quadratic', '100'])).toThrow(/quadratic/);
  });

  it('a missing table section is rejected by name', () => {
    const p = new PairTable();
    p.settings(['linear', '500']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1', 'lj.table', 'NOPE', '2.6'], ctx)).toThrow(/NOPE/);
  });

  it('a pair_coeff cutoff beyond the table extent is rejected', () => {
    const p = new PairTable();
    p.settings(['linear', '500']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1', 'lj.table', 'T1', '3.0'], ctx)).toThrow(/Pair table cutoff outside of table/);
  });

  it('the optional pair_coeff cutoff truncates the table extent', () => {
    const p = new PairTable();
    p.settings(['linear', String(NFILE)]);
    p.allocate(2);
    p.coeff(['1', '1', 'lj.table', 'T1', '2.0'], ctx);
    p.coeff(['2', '2', 'lj.table', 'T1', '2.0'], ctx);
    p.coeff(['1', '2', 'lj.table', 'T1', '2.0'], ctx);
    p.init(ctx);
    // the pair cutoff is the pair_coeff value, and inside it the resampled table still tracks LJ
    expect(p.initOne(1, 1)).toBe(2.0);
    const { eng, fforce } = p.single(0, 0, 1, 1, 1.5 * 1.5, 0, 1);
    expect(eng).toBeCloseTo(ljEnergy(1.5), 4);
    expect(fforce * 1.5).toBeCloseTo(ljForce(1.5), 4);
  });
});
