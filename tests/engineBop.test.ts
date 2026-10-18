import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PAIRS } from '../src/engine-gpl/registry/bop';
import { TabularFunction } from '../src/engine-gpl/bop/tabular';
import { parseBopPotential } from '../src/engine-gpl/bop/potential';
import { PairBOP } from '../src/engine-gpl/bop/pair_bop';
import type { StyleContext } from '../src/engine/force/types';

const POTENTIALS = join(__dirname, '..', 'third_party', 'lammps', 'potentials');

/** A StyleContext whose files are the given texts; the box is metal units (so the file's UNITS tag matches). */
const ctxWith = (files: Record<string, string>, units = 'metal'): StyleContext => ({
  s: { units: { style: units }, massByType: new Float64Array(10) } as unknown as StyleContext['s'],
  readFile: (name: string) => {
    if (name in files) return files[name];
    return readFileSync(join(POTENTIALS, name), 'utf8');
  },
  log: () => {},
});

/**
 * A one-element potential file in the format of the "No angular table" section with a parabolic angular function
 * (nr = 5, nBOt = 4, no cutL table): the values are simple so the parsed arrays can be checked by hand.
 */
const ONE_ELEMENT = [
  '### test potential',
  '1',
  '1 1.0 X',
  '5 4',
  '0.1 0.1 0.1 0.1 0.1 0.1 0.1',
  '0.5',
  '4.0',
  '1.0 0.0 1.0 1.0',
  '0.0 0.0',
  '0.5 0.0 0.0',
  '1.0 0.0 0.0',
  '1 2 3 4 5',
  '1 1 1 1 1',
  '0 0 0 0 0',
  '0 0 0 0',
  '0.0',
  '0.0',
  '0.0',
].join('\n');

describe('pair_style bop registration and settings', () => {
  it('registers the style under its LAMMPS name', () => {
    const p = PAIRS.bop();
    expect(p).toBeInstanceOf(PairBOP);
    expect(p.name).toBe('bop');
  });

  it('accepts the save keyword and refuses any other argument', () => {
    const p = new PairBOP();
    expect(() => p.settings(['save'])).not.toThrow();
    expect(() => p.settings([])).not.toThrow();
    expect(() => p.settings(['fast'])).toThrow('Illegal pair_style command');
  });
});

describe('TabularFunction', () => {
  it('reproduces its samples at the grid points', () => {
    const t = new TabularFunction();
    const ys = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0];
    t.setValues(ys.length, 0.0, 5.0, ys);
    for (let k = 0; k < ys.length; k++) expect(t.value(k)).toBe(ys[k]);
  });

  it('is exact for linear data, including the derivative', () => {
    const t = new TabularFunction();
    const ys = [1.0, 3.0, 5.0, 7.0, 9.0];
    t.setValues(ys.length, 0.0, 4.0, ys);
    expect(t.value(1.5)).toBeCloseTo(4.0, 12);
    expect(t.dy).toBeCloseTo(2.0, 12);
  });
});

describe('BOP potential file reader', () => {
  it('parses a one-element file with the parabolic angular function and no cutL table', () => {
    const pot = parseBopPotential(ONE_ELEMENT, 'one.bop', ctxWith({}));
    expect(pot.nb).toBe(1);
    expect(pot.names).toEqual(['X']);
    expect(pot.masses).toEqual([1.0]);
    expect(pot.nr).toBe(5);
    expect(pot.nBOt).toBe(4);
    expect(pot.npower).toBe(2);
    expect(pot.cutmax).toBe(4.0);
    expect(pot.pairs).toHaveLength(1);
    expect(pot.pairs[0].cutB).toBe(4.0);
    expect(pot.pairs[0].cutL).toBe(0);
    expect(pot.pairs[0].cphi).toBeNull();
    expect(pot.triplets).toHaveLength(1);
    expect(Array.from(pot.elem3param)).toEqual([0]);
    expect(Array.from(pot.elem2param)).toEqual([0]);
    expect(pot.sigmaC[0]).toBe(1.0);
    expect(pot.piA[0]).toBe(1.0);
  });

  it('reads the CdTe table (two elements, parabolic angular function) with its cutoffs', () => {
    const text = readFileSync(join(POTENTIALS, 'CdTe.bop.table'), 'utf8');
    const pot = parseBopPotential(text, 'CdTe.bop.table', ctxWith({}));
    expect(pot.names).toEqual(['Cd', 'Te']);
    expect(pot.nb).toBe(2);
    expect(pot.pairs).toHaveLength(3);
    expect(pot.nBOt).toBe(2000);
    expect(pot.nr).toBe(2000);
    expect(pot.cutmax).toBeCloseTo(4.9, 12);
    // the triplet tables are the (i, j <= k) combinations with k = j stored once: 2 * 3 for two elements
    expect(pot.triplets).toHaveLength(6);
  });

  it('refuses a units tag that differs from the box units, as native does for potential files', () => {
    const text = `### UNITS: metal\n${ONE_ELEMENT}`;
    expect(() => parseBopPotential(text, 'one.bop', ctxWith({}, 'real'))).toThrow(
      'BOP potential file one.bop requires metal units but real units are in use',
    );
  });

  it('refuses a header that is neither 2 nor 3 values, and a 3-value header whose angular table has npower <= 2', () => {
    const four = ONE_ELEMENT.replace('5 4', '5 4 4 4');
    expect(() => parseBopPotential(four, 'x.bop', ctxWith({}))).toThrow('Unsupported BOP potential file format');
    const npower2 = ONE_ELEMENT.replace('5 4', '5 2 4');
    expect(() => parseBopPotential(npower2, 'x.bop', ctxWith({}))).toThrow(/npower <= 2/);
  });

  it('refuses a file with no elements', () => {
    expect(() => parseBopPotential('0\n', 'x.bop', ctxWith({}))).toThrow('BOP potential file with 0 elements');
  });
});

describe('pair_coeff and init', () => {
  it('maps the elements of the file and refuses an element the file does not have', () => {
    const p = new PairBOP();
    p.ntypes = 2;
    const ctx = ctxWith({ 'one.bop': ONE_ELEMENT });
    expect(() => p.coeff(['*', '*', 'one.bop', 'X', 'Y'], ctx)).toThrow('Element Y not found in bop potential file one.bop');
  });

  it('refuses init before pair_coeff', () => {
    const p = new PairBOP();
    p.ntypes = 1;
    expect(() => p.initOne(1, 1)).toThrow('All pair coeffs are not set');
  });

  it('takes the neighbour cutoff of a pair from the file', () => {
    const p = new PairBOP();
    p.ntypes = 1;
    p.coeff(['*', '*', 'one.bop', 'X'], ctxWith({ 'one.bop': ONE_ELEMENT }));
    expect(p.initOne(1, 1)).toBe(4.0);
  });

  it('refuses a newton pair off setting', () => {
    const p = new PairBOP();
    expect(() => p.initStyle({ ...ctxWith({}), newtonPair: false })).toThrow('Pair style BOP requires newton pair on');
  });
});
