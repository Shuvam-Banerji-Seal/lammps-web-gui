import { describe, it, expect } from 'vitest';
import { parseMliapSo3Descriptor, So3Engine } from '../src/engine/mliap/so3';

/*
 * Descriptor so3 of pair_style mliap (wave 37, src/engine/mliap/so3.ts). Native parity cases are in
 * tests/oracle/w37so3_*.in (engineOracle.test.ts). Here: values measured with native LAMMPS (black box) on a
 * dimer, and a finite-difference check of the analytic derivative.
 */

const desc = (nmax: number, lmax: number, rc = 5.0, alpha = 2.0, elems = 'X', w = '1') =>
  `rcutfac ${rc}\nnmax ${nmax}\nlmax ${lmax}\nalpha ${alpha}\nnelems 1\nelems ${elems}\nradelems 0.5\nwelems ${w}\n`;

describe('so3 descriptor file', () => {
  it('reads the keywords and counts K = nmax(nmax+1)/2 (lmax+1)', () => {
    const d = parseMliapSo3Descriptor(desc(3, 2), 'x');
    expect(d.K).toBe(18);
    expect(parseMliapSo3Descriptor(desc(3, 4), 'x').K).toBe(30);
  });
  it('refuses keywords the native reader rejects (rfac0, switchflag, rmin0)', () => {
    for (const kw of ['rfac0 0.99', 'switchflag 0', 'rmin0 0.1']) {
      expect(() => parseMliapSo3Descriptor(desc(2, 1) + kw + '\n', 'x')).toThrow(/is not implemented/);
    }
  });
  it('requires welems and refuses nmax 0', () => {
    expect(() => parseMliapSo3Descriptor(desc(2, 1).replace(/welems.*\n/, ''), 'x')).toThrow(/welems/);
    expect(() => parseMliapSo3Descriptor(desc(0, 1), 'x')).toThrow(/nmax/);
  });
});

describe('so3 descriptor values against native LAMMPS (black box)', () => {
  it('dimer, nmax 2, lmax 2, alpha 2, rcutfac 5, per-atom values at r = 1.1', () => {
    // Measured with native LAMMPS (black box): per-atom values = (compute mliap column sums)/2 of a dimer
    // (both atoms see one neighbour at 1.1 A). Order: (n1,n2) = (1,1),(1,2),(2,2) with l = 0,1,2 inside each.
    const native = [0.16318338, 0.20463907, 0.13674726, 0.35016307, 0.37739179, 0.20832180, 0.75138887, 0.69597933, 0.31735899];
    const d = parseMliapSo3Descriptor(desc(2, 2), 'x');
    const eng = new So3Engine(d);
    const B = new Float64Array(d.K);
    eng.evaluate(Float64Array.from([1.1, 0, 0]), Float64Array.from([1]), B);
    for (let c = 0; c < d.K; c++) expect(B[c]).toBeCloseTo(native[c], 7);
  });
});

describe('so3 descriptor derivative', () => {
  it('analytic gradient of sum_c gam_c B_c matches central differences (2 elements, 4 atoms)', () => {
    const d = parseMliapSo3Descriptor(desc(3, 2, 4.5, 1.5), 'x');
    const eng = new So3Engine(d);
    const K = d.K;
    const gam = Float64Array.from({ length: K }, (_, c) => Math.sin(1.3 * c + 0.4));
    // three neighbours of an atom at the origin, with different weights
    const pts = [[1.2, 0.3, -0.4], [-0.5, 1.6, 0.7], [0.4, -1.1, 1.5], [2.1, 0.2, 0.9]];
    const wts = Float64Array.from([1.0, 0.96, 1.0, 0.5]);
    const energy = (p: number[][]) => {
      const nb = Float64Array.from(p.flat());
      const B = new Float64Array(K);
      eng.evaluate(nb, wts, B);
      let e = 0;
      for (let c = 0; c < K; c++) e += gam[c] * B[c];
      return e;
    };
    const nb = Float64Array.from(pts.flat());
    const grad = new Float64Array(3 * pts.length);
    const B = new Float64Array(K);
    eng.evaluate(nb, wts, B, gam, grad);
    const h = 1e-5;
    for (let a = 0; a < pts.length; a++) {
      for (let t = 0; t < 3; t++) {
        const pp = pts.map((p) => p.slice()), pm = pts.map((p) => p.slice());
        pp[a][t] += h; pm[a][t] -= h;
        const fd = (energy(pp) - energy(pm)) / (2 * h);
        expect(grad[3 * a + t]).toBeCloseTo(fd, 6);
      }
    }
  });
});
