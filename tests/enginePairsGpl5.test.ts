import { describe, expect, it } from 'vitest';
import { StyleError, type StyleContext } from '../src/engine/force/types';
import { PairREBOMoS } from '../src/engine-gpl/rebomos/rebomos';
import { PairYLZ } from '../src/engine-gpl/ylz/pair_ylz';
import { PairMM3Switch3CoulGaussLong } from '../src/engine-gpl/mm3/pair_mm3_switch3_coulgauss_long';

/*
 * Argument checks of the three pair styles ported in src/engine-gpl (rebomos, ylz, mm3/switch3/coulgauss/long).
 * The numbers are checked against native LAMMPS by tests/oracle/w115p_* (tests/engineOracle.test.ts).
 */
const ctx = (text = ''): StyleContext => ({ s: null, readFile: () => text, log: () => {} });

describe('pair_style rebomos arguments', () => {
  it('takes no pair_style arguments', () => {
    const p = new PairREBOMoS();
    p.allocate(2);
    expect(() => p.settings(['1.0'], ctx())).toThrow(StyleError);
  });

  it('refuses NULL and unknown elements', () => {
    const p = new PairREBOMoS();
    p.allocate(2);
    expect(() => p.coeff(['*', '*', 'MoS.rebomos', 'Mo', 'NULL'], ctx())).toThrow(/NULL/);
    expect(() => p.coeff(['*', '*', 'MoS.rebomos', 'Mo', 'Se'], ctx())).toThrow(StyleError);
  });

  it('reads the 61 parameters of the potential file', () => {
    const p = new PairREBOMoS();
    p.allocate(2);
    expect(() => p.coeff(['*', '*', 'short.rebomos', 'Mo', 'S'], ctx('1.0 # only one value\n'))).toThrow(/61/);
  });
});

describe('pair_style ylz arguments', () => {
  it('needs exactly one global cutoff and eight pair_coeff words', () => {
    const p = new PairYLZ();
    p.allocate(2);
    expect(() => p.settings([], ctx())).toThrow(StyleError);
    expect(() => p.coeff(['*', '*', '1.0', '1.0', '4', '3', '0.0'], ctx())).toThrow(StyleError);
  });
});

describe('pair_style mm3/switch3/coulgauss/long arguments', () => {
  it('takes an LJ cutoff, an optional Coulomb cutoff and the truncation width', () => {
    const p = new PairMM3Switch3CoulGaussLong();
    expect(() => p.settings(['12.0'], ctx())).toThrow(StyleError);
    expect(() => p.settings(['12.0', '4.0', '1.0', '2.0'], ctx())).toThrow(StyleError);
    expect(() => p.settings(['12.0', '4.0'], ctx())).not.toThrow();
  });

  it('needs five or six pair_coeff words', () => {
    const p = new PairMM3Switch3CoulGaussLong();
    p.allocate(2);
    expect(() => p.coeff(['*', '*', '1.0', '2.0'], ctx())).toThrow(StyleError);
  });
});
