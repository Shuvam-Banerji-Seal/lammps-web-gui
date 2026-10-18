import { describe, expect, it } from 'vitest';
import { PAIRS } from '../src/engine-gpl/registry/xpair2';
import { PairLJCutSphere } from '../src/engine-gpl/xpair2/lj_cut_sphere';
import { PairLJExpandSphere } from '../src/engine-gpl/xpair2/lj_expand_sphere';
import { PairLJGromacsCoulGromacs } from '../src/engine-gpl/xpair2/lj_gromacs_coul_gromacs';
import { PairNMCut } from '../src/engine-gpl/xpair2/nm_cut';
import { PairMorseSmoothLinear } from '../src/engine-gpl/xpair2/morse_smooth_linear';
import { mixEnergyNative, powint } from '../src/engine-gpl/xpair2/common';
import { StyleError, type StyleContext } from '../src/engine/force/types';

/* Parity with native LAMMPS is in the oracle cases tests/oracle/w220xp_*.in; these check the argument rules, mixing,
 * extract and the data-file lines against the native source (src/EXTRA-PAIR at b332f28). */

const noSystem: StyleContext = { s: null, readFile: () => '', log: () => {} };

describe('xpair2 registry', () => {
  it('registers the ten pair styles under their LAMMPS names', () => {
    expect(Object.keys(PAIRS).sort()).toEqual([
      'lj/cut/sphere', 'lj/expand/coul/long', 'lj/expand/sphere', 'lj/gromacs/coul/gromacs', 'lj/pirani',
      'morse/smooth/linear', 'nm/cut', 'nm/cut/coul/cut', 'nm/cut/coul/long', 'nm/cut/split',
    ].sort());
    for (const name of Object.keys(PAIRS)) expect(PAIRS[name]().name).toBe(name);
  });
});

describe('argument rules (native error texts)', () => {
  it('lj/cut/sphere takes one cutoff ratio and 3 or 4 coeff words', () => {
    const p = new PairLJCutSphere();
    expect(() => p.settings([])).toThrow(new StyleError('Illegal pair_style command').message);
    p.settings(['2.5']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1'])).toThrow(StyleError);
    expect(() => p.coeff(['1', '1', '1.0', '2.0', '3.0'])).toThrow(StyleError);
  });

  it('lj/expand/sphere takes 4 or 5 coeff words', () => {
    const p = new PairLJExpandSphere();
    p.settings(['2.5']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1', '1.0'])).toThrow(StyleError);
  });

  it('morse/smooth/linear needs explicit pairs (no mixing) and rejects a short coeff line', () => {
    const p = new PairMorseSmoothLinear();
    p.settings(['2.5']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1', '1.0', '1.5'])).toThrow(StyleError);
    p.coeff(['1', '1', '1.0', '1.5', '1.1']);
    expect(() => p.initOne(1, 2)).toThrow('All pair coeffs are not set');
  });

  it('gromacs takes exactly 2 or 4 settings values and the inner cutoffs may not exceed the outer', () => {
    const p = new PairLJGromacsCoulGromacs();
    expect(() => p.settings(['1.0'])).toThrow(StyleError);
    expect(() => p.settings(['2.0', '1.5'])).toThrow(StyleError);
    p.settings(['1.5', '2.5']);
    expect(() => p.settings(['1.0', '2.0', '1.5', '2.5', '9'])).toThrow(StyleError);
  });

  it('nm/cut needs 6 or 7 coeff words', () => {
    const p = new PairNMCut();
    p.settings(['3.0']);
    p.allocate(2);
    expect(() => p.coeff(['1', '1', '1.0', '1.1', '8'])).toThrow(StyleError);
  });

  it('the sphere styles require the radius attribute at init (no system here)', () => {
    const p = new PairLJCutSphere();
    p.settings(['2.5']);
    p.allocate(1);
    p.coeff(['1', '1', '1.0']);
    expect(() => p.init(noSystem)).toThrow('Pair style lj/cut/sphere requires atom attribute radius');
    const q = new PairLJExpandSphere();
    q.settings(['2.5']);
    q.allocate(1);
    q.coeff(['1', '1', '1.0', '1.0']);
    expect(() => q.init(noSystem)).toThrow('Pair style lj/expand/sphere requires atom attribute radius');
  });
});

describe('mixing and reset rules', () => {
  it('Pair::mix_energy gives 0 for sixthpower with a zero diameter, sqrt(e1 e2) otherwise', () => {
    expect(mixEnergyNative('sixthpower', 1, 4, 0, 0)).toBe(0);
    expect(mixEnergyNative('geometric', 1, 4, 0, 0)).toBe(2);
    expect(mixEnergyNative('arithmetic', 1, 4, 0, 0)).toBe(2);
  });

  it('powint is repeated squaring (negative exponents invert)', () => {
    expect(powint(2, 6)).toBe(64);
    expect(powint(2, -2)).toBe(0.25);
    expect(powint(0, 3)).toBe(0);
    expect(powint(5, 0)).toBe(1);
  });

  it('settings() after coeff resets the cutoff of every set pair (native reset of explicit cutoffs)', () => {
    const p = new PairLJCutSphere();
    p.settings(['2.5']);
    p.allocate(2);
    p.coeff(['1', '1', '1.0', '2.0']);
    p.settings(['3.5']);
    expect((p as unknown as { p: { get(n: string, i: number, j: number): number } }).p.get('cut', 1, 1)).toBe(3.5);
  });

  it('lj/cut/sphere refuses sixthpower mixing at init (with a radius attribute)', () => {
    const p = new PairLJCutSphere();
    p.settings(['2.5']);
    p.allocate(1);
    p.coeff(['1', '1', '1.0']);
    p.mix = 'sixthpower';
    const s = { n: 1, type: new Int32Array([1]), radius: new Float64Array([0.5]) } as unknown as StyleContext['s'];
    expect(() => p.init({ ...noSystem, s })).toThrow('Pair_modify mix sixthpower is not compatible with pair style lj/cut/sphere');
  });
});

describe('extract and data-file lines', () => {
  it('nm/cut writes "type e0 r0 n m" and "i j e0 r0 n m cut" (native write_data / write_data_all)', () => {
    const p = new PairNMCut();
    p.settings(['3.0']);
    p.allocate(2);
    p.coeff(['1', '1', '1.0', '1.1', '8', '4']);
    p.coeff(['2', '2', '0.5', '1.2', '9', '6', '2.5']);
    p.coeff(['1', '2', '0.7', '1.15', '8', '6']);
    expect(p.dataCoeffs()).toEqual(['1 1 1.1 8 4', '2 0.5 1.2 9 6']);
    expect(p.dataCoeffsIJ()).toEqual(['1 1 1 1.1 8 4 3', '1 2 0.7 1.15 8 6 3', '2 2 0.5 1.2 9 6 2.5']);
    expect(p.extract('nn')).toBeInstanceOf(Float64Array);
    expect(p.extract('nope')).toBeUndefined();
  });

  it('lj/gromacs/coul/gromacs extract gives epsilon and sigma; data lines carry epsilon and sigma', () => {
    const p = new PairLJGromacsCoulGromacs();
    p.settings(['1.5', '2.5']);
    p.allocate(2);
    p.coeff(['1', '1', '1.0', '1.0']);
    p.coeff(['2', '2', '0.5', '1.2']);
    expect(p.dataCoeffs()).toEqual(['1 1 1', '2 0.5 1.2']);
    expect(p.extract('sigma')).toBeInstanceOf(Float64Array);
  });

  it('the nm/cut/coul/long style writes no data-file coefficients (the source sets no writedata flag)', () => {
    const p = PAIRS['nm/cut/coul/long']();
    expect(p.dataCoeffs()).toBeNull();
    expect(p.dataCoeffsIJ()).toBeNull();
  });
});
