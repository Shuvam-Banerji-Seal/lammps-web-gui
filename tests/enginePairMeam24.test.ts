import { describe, expect, it } from 'vitest';
import { gOfIbar, gPrimeOfIbar } from '../src/engine/force/pair/meam_alloy';
import { DEFAULT_MEAM_OPTIONS, SUPPORTED_IBAR, ZBL_GUARD, gOf, gPrimeOf, parseMeamParams } from '../src/engine/force/pair/meam';

/*
 * pair_style meam, task meam24: the ibar forms that are verified and the ZBL blend guard.
 *
 * Docs (plans/lammps-docs/pair_meam.rst): "0 => G = sqrt(1+Gamma)", "1 => G = exp(Gamma/2)",
 * "3 => G = 2/(1+exp(-Gamma))". The -5 form is refused for now (see SUPPORTED_IBAR in meam.ts).
 *
 * Measured with native LAMMPS (black box): displaced fcc Ni (library Ni4 and the real Ni.meam), displaced fcc
 * (library Ni2, ibar = 3), displaced bcc (library WL, ibar = 3) and displaced fcc (library Ni, ibar = 1) all agree
 * with the engine to 1e-9 (tests/oracle/w24meam_*.in). Dia references with ibar = 1 (library Si) and -5 (library
 * SiS) and bcc with -5 (library FeS) do not agree at step 0, so they stay refused.
 */

describe('MEAM G(Gamma) forms', () => {
  it('supports exactly the verified ibar values', () => {
    expect([...SUPPORTED_IBAR]).toEqual([0, 1, 3]);
  });

  for (const ibar of [0, 1, 3]) {
    it(`dG/dGamma matches a finite difference for ibar = ${ibar}`, () => {
      for (const g of [-0.4, -0.05, 0, 0.03, 0.2]) {
        const h = 1e-6;
        const fd = (gOfIbar(ibar, g + h) - gOfIbar(ibar, g - h)) / (2 * h);
        expect(Math.abs(gPrimeOfIbar(ibar, g) - fd)).toBeLessThan(1e-7);
      }
    });
  }

  it('G is 1 at Gamma = 0 for the verified forms (perfect fcc and bcc references)', () => {
    for (const ibar of [0, 1, 3]) expect(gOfIbar(ibar, 0)).toBeCloseTo(1, 14);
  });

  it('the single-element aliases equal the alloy-module helpers', () => {
    for (const ibar of [0, 1, 3]) {
      expect(gOf(ibar, 0.1)).toBe(gOfIbar(ibar, 0.1));
      expect(gPrimeOf(ibar, 0.1)).toBe(gPrimeOfIbar(ibar, 0.1));
    }
  });
});

describe('MEAM ZBL blend guard', () => {
  it('the default zbl = 1 leaves no pair exempt (zblOff is empty)', () => {
    const par = parseMeamParams('rc = 4.0\ndelr = 0.1\n', 'default.meam');
    expect(par.zblOff.size).toBe(0);
  });

  it('an explicit zbl(1,1) = 0 is recorded as an exempt pair', () => {
    const par = parseMeamParams('zbl(1,1) = 0\n', 'zbl0.meam');
    expect(par.zblOff.has('1,1')).toBe(true);
  });

  it('guard distance is 0.8 re and the default cutoff options are unchanged', () => {
    expect(ZBL_GUARD).toBe(0.8);
    expect(DEFAULT_MEAM_OPTIONS.rc).toBe(4.0);
    expect(DEFAULT_MEAM_OPTIONS.delr).toBe(0.1);
  });
});
