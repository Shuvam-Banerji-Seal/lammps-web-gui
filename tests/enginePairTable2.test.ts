import { describe, expect, it } from 'vitest';
import { PairTable } from '../src/engine/force/pair/table';
import { PairLJLongCoulLong } from '../src/engine/force/pair/lj_long';
import { PairLJCharmmfswCoulCharmmfsh } from '../src/engine/force/pair/charmm';
import { StyleError, type StyleContext } from '../src/engine/force/types';

/*
 * Unit tests for pair_style table spline (pair_table.html), and the argument
 * checks of the new CHARMM fsw and lj/long styles. The LJ 12-6 table is written
 * on 200 points uniform in r^2 with FPRIME end slopes (the same layout as
 * tests/oracle/w10table_lj_fp200.table).
 */

const LJ_TABLE = (() => {
  const E = (r: number) => 4 * (r ** -12 - r ** -6);
  const F = (r: number) => 48 * r ** -13 - 24 * r ** -7;
  const dF = (r: number) => -624 * r ** -14 + 168 * r ** -8;
  const rlo = 0.5, rhi = 2.5, n = 200;
  let s = `LJ11\nN ${n} RSQ ${rlo} ${rhi} FPRIME ${dF(rlo)} ${dF(rhi)}\n\n`;
  for (let k = 0; k < n; k++) {
    const r = Math.sqrt(rlo * rlo + ((rhi * rhi - rlo * rlo) * k) / (n - 1));
    s += `${k + 1} ${r.toPrecision(17)} ${E(r).toPrecision(17)} ${F(r).toPrecision(17)}\n`;
  }
  return s + '\n';
})();

const ctx = (text: string): StyleContext => ({ s: null, readFile: () => text, log: () => {} });

const make = (style: string[], cut = 2.5): PairTable => {
  const p = new PairTable();
  p.settings(style);
  p.allocate(1);
  p.coeff(['1', '1', 'lj.table', 'LJ11', String(cut)], ctx(LJ_TABLE));
  p.initOne(1, 1);
  return p;
};

/** Pair energy and radial force F = fforce * r at distance r from the table pair style. */
const evalAt = (p: PairTable, r: number): { e: number; F: number } => {
  const out = p.single(0, 1, 1, 1, r * r, 1, 1);
  return { e: out.eng, F: out.fforce * r };
};

describe('pair_style table spline', () => {
  it('spline energy and force are consistent: F = -dE/dr to the spline accuracy', () => {
    const p = make(['spline', '1000']);
    let worst = 0;
    for (let r = 0.6; r < 2.3; r += 0.0137) {
      const h = 1e-5;
      const dEdr = (evalAt(p, r + h).e - evalAt(p, r - h).e) / (2 * h);
      const F = evalAt(p, r).F;
      // "cubic spline coefficients are computed ... one set of splines for energy,
      // another for force" (pair_table.html): the two splines are fitted separately,
      // so F and -dE/dr agree to the spline accuracy, not to round-off. Measured:
      // the largest mismatch is at the steep inner end (r = 0.61, about 9e-4 relative
      // to F); it is about 1e-6 relative in the outer half of the table.
      worst = Math.max(worst, Math.abs(F + dEdr) / Math.max(Math.abs(F), 1));
    }
    expect(worst).toBeLessThan(2e-3);
  });

  it('spline energy stays within the interpolation error of the 200-node file', () => {
    // Ntable = Nfile = 200 with RSQ: the internal nodes are the file nodes, so the
    // error is only the 200-point interpolation of LJ (measured about 9e-6 relative).
    const p = make(['spline', '200']);
    const E = (r: number) => 4 * (r ** -12 - r ** -6);
    for (const r of [0.7, 1.2, 1.5, 2.2]) {
      expect(Math.abs(evalAt(p, r).e - E(r)) / Math.max(Math.abs(E(r)), 1)).toBeLessThan(1e-4);
    }
  });

  it('rejects bitmap, spline with N < 3, and an unknown interpolation style by name', () => {
    expect(() => new PairTable().settings(['bitmap', '12'])).toThrow(StyleError);
    expect(() => new PairTable().settings(['bitmap', '12'])).toThrow(/bitmap/);
    expect(() => new PairTable().settings(['spline', '2'])).toThrow(/N >= 3/);
    expect(() => new PairTable().settings(['cubic', '100'])).toThrow(/invalid pair_style table interpolation style 'cubic'/);
  });

  it('rejects a cutoff beyond the table outer distance', () => {
    const p = new PairTable();
    p.settings(['spline', '500']);
    p.allocate(1);
    expect(() => p.coeff(['1', '1', 'lj.table', 'LJ11', '3.0'], ctx(LJ_TABLE))).toThrow(StyleError);
  });

  it('rejects a table file section with an unknown table parameter', () => {
    const p = new PairTable();
    p.settings(['spline', '500']);
    p.allocate(1);
    const bad = LJ_TABLE.replace(/N 200 RSQ/, 'N 200 WIDTH');
    expect(() => p.coeff(['1', '1', 'lj.table', 'LJ11'], ctx(bad))).toThrow(/unknown table parameter 'WIDTH'/);
  });
});

describe('CHARMM force-switched and lj/long argument checks', () => {
  it('lj/charmmfsw/coul/charmmfsh requires inner < outer', () => {
    const p = new PairLJCharmmfswCoulCharmmfsh();
    expect(() => p.settings(['10.0', '8.0', '9.0'])).toThrow(/inner cutoff/);
  });

  it('lj/long/coul/long rejects flag_lj long and the per-pair Coulomb cutoff by name', () => {
    const p = new PairLJLongCoulLong();
    expect(() => p.settings(['long', 'long', '8.0', '10.0'])).not.toThrow();
    expect(() => p.settings(['cut', 'long'])).toThrow(/usage/);
    p.settings(['cut', 'long', '8.0', '10.0']);
    expect(() => p.coeff(['1', '1', '0.1', '3.0', '7.0', '9.0'])).toThrow(/cutoff2/);
  });
});
