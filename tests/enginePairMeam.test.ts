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
import { StyleError } from '../src/engine/force/types';

/*
 * Single-element MEAM (Cu-like synthetic entry, see the header of meam.ts).
 * Forces are checked against central finite differences of meamEnergy, which is
 * the independently written energy path.
 */

const CU: MeamElement = {
  z: 12,
  re: 2.55,
  alpha: 4.95,
  Ec: 3.54,
  A: 1,
  beta: [2, 1, 2, 4],
  t: [1, 1, 1, 1],
  ibar: 0,
};
const OPTS: MeamOptions = { ...DEFAULT_MEAM_OPTIONS };

/** Deterministic LCG in [0,1). */
const rng = (seed: number) => {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
};

/** 2x2x2 fcc conventional cell (32 atoms), perturbed by up to amp per component. */
const fccCell = (amp: number, seed: number) => {
  const a = 3.615, n = 2;
  const basis = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]];
  const pts: number[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++)
        for (const b of basis) pts.push((i + b[0]) * a, (j + b[1]) * a, (k + b[2]) * a);
  const R = rng(seed);
  for (let q = 0; q < pts.length; q++) pts[q] += (R() * 2 - 1) * amp;
  return { x: Float64Array.from(pts), L: [n * a, n * a, n * a] as [number, number, number] };
};

/** Small periodic box with a few atoms placed so that screening is partial. */
const cluster = (pts: number[][], seed: number, amp: number) => {
  const R = rng(seed);
  const x = new Float64Array(pts.length * 3);
  pts.forEach((p, i) => {
    for (let c = 0; c < 3; c++) x[3 * i + c] = p[c] + (R() * 2 - 1) * amp;
  });
  return { x, L: [30, 30, 30] as [number, number, number] };
};

/** Central-difference gradient check over every component; returns the worst relative error. */
const checkForces = (el: MeamElement, o: MeamOptions, x: Float64Array, L: [number, number, number]) => {
  const { E, F } = meamEnergyForces(el, o, x, L);
  expect(E).toBeCloseTo(meamEnergy(el, o, x, L), 9);
  const h = 1e-5;
  let worst = 0;
  for (let q = 0; q < x.length; q++) {
    const xp = Float64Array.from(x), xm = Float64Array.from(x);
    xp[q] += h;
    xm[q] -= h;
    const fd = -(meamEnergy(el, o, xp, L) - meamEnergy(el, o, xm, L)) / (2 * h);
    const err = Math.abs(F[q] - fd) / Math.max(Math.abs(fd), 1e-2);
    if (err > worst) worst = err;
  }
  return worst;
};

describe('MEAM analytic forces', () => {
  it('perturbed fcc cell: forces equal -grad E to 1e-6 relative', () => {
    const { x, L } = fccCell(0.08, 7);
    expect(checkForces(CU, OPTS, x, L)).toBeLessThan(1e-6);
  });

  it('perturbed fcc cell with a different seed and a larger distortion', () => {
    const { x, L } = fccCell(0.15, 99);
    expect(checkForces(CU, OPTS, x, L)).toBeLessThan(1e-6);
  });

  it('trimer (partial screening by the third atom)', () => {
    const { x, L } = cluster([[0, 0, 0], [2.5, 0, 0], [1.3, 2.2, 0.3]], 3, 0.05);
    expect(checkForces(CU, OPTS, x, L)).toBeLessThan(1e-6);
  });

  it('4-atom cluster with partial screening (three screeners on one bond)', () => {
    const { x, L } = cluster(
      [[0, 0, 0], [2.6, 0.1, 0], [1.2, 2.4, 0.2], [1.5, 0.9, 2.3]],
      11,
      0.1,
    );
    expect(checkForces(CU, OPTS, x, L)).toBeLessThan(1e-6);
  });

  it('4-atom cluster with a 2nd-neighbour bond (r in the screened 2nd shell)', () => {
    const { x, L } = cluster(
      [[0, 0, 0], [3.4, 0.2, 0], [1.6, 2.7, 0.4], [1.8, 1.2, 2.9]],
      5,
      0.05,
    );
    expect(checkForces(CU, OPTS, x, L)).toBeLessThan(1e-6);
  });
});

describe('pair_style meam: files and StyleError paths', () => {
  const LIB = "# comment\n'Cu' 'fcc' 12 29 63.546 4.95 2 1 2 4 3.615 3.54 1 1 1 1 1 1 0\n";
  const LIB_SPLIT = "'Ni' 'fcc' 12 28 58.69 4.6 2 1\n 2 4 3.52 4.45 1 1 1 1 1 1 0\n";
  const PAR = 'rc = 4.0\ndelr = 0.1\nEc(1,1) = 3.54\nre(1,1) = 2.55\nalpha(1,1) = 4.95\nlattce(1,1) = fcc\nCmin(1,1,1) = 2.0\nCmax(1,1,1) = 2.8\nzbl(1,1) = 0\n';

  const ctxWith = (files: Record<string, string>) => ({
    s: null,
    readFile: (n: string) => {
      if (!(n in files)) throw new Error(`no file ${n}`);
      return files[n];
    },
    log: () => {},
  });

  const makePair = (files: Record<string, string>, args: string[], ntypes = 1) => {
    const p = new PairMeam();
    p.settings([], ctxWith(files));
    p.allocate(ntypes);
    p.coeff(args, ctxWith(files));
    return p;
  };

  it('library: first matching entry, fields may span lines, missing element is an error', () => {
    expect(parseMeamLibrary(LIB + LIB_SPLIT, 'Ni', 'lib').asub).toBe(1);
    expect(parseMeamLibrary(LIB_SPLIT, 'Ni', 'lib').alpha).toBe(4.6);
    expect(() => parseMeamLibrary(LIB, 'Au', 'lib')).toThrow(StyleError);
  });

  it('parameters: zbl must be 0, unsupported keywords are named', () => {
    expect(parseMeamParams(PAR, 'par').opts.Cmax).toBe(2.8);
    expect(() => parseMeamParams(PAR.replace('zbl(1,1) = 0', 'zbl(1,1) = 1'), 'par')).toThrow(/zbl/);
    expect(() => parseMeamParams(PAR + 'attrac(1,1) = 0.1\n', 'par')).toThrow(/attrac/);
    expect(() => parseMeamParams(PAR + 'gsmooth_factor = 0.5\n', 'par')).toThrow(/gsmooth_factor/);
    expect(() => parseMeamParams(PAR + 'theta(1,1) = 170\n', 'par')).toThrow(/theta/);
    expect(() => parseMeamParams(PAR + 'Ec(2,2) = 1\n', 'par')).toThrow(/indexed/);
  });

  it('pair_coeff: single element, mapped types, NULL parameter file and multi-element are errors', () => {
    const files = { 'lib.meam': LIB, 'par.meam': PAR };
    const p = makePair(files, ['*', '*', 'lib.meam', 'Cu', 'par.meam', 'Cu'], 1);
    expect(p.initOne(1, 1)).toBe(4.0);
    expect(() => makePair(files, ['*', '*', 'lib.meam', 'Cu', 'NULL', 'Cu'])).toThrow(/parameter file/);
    expect(() => makePair(files, ['*', '*', 'lib.meam', 'Cu', 'Ni', 'par.meam', 'Cu', 'Ni'], 2)).toThrow(/multi-element/);
    expect(() => makePair(files, ['*', '*', 'lib.meam', 'Cu', 'par.meam', 'NULL'])).toThrow(/NULL/);
  });

  it('pair_style meam takes no arguments and rejects pair_modify', () => {
    const p = new PairMeam();
    expect(() => p.settings(['x'], ctxWith({}))).toThrow(StyleError);
    expect(() => p.modify('shift', ['yes'])).toThrow(StyleError);
  });
});
