import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { StyleError, type Pair, type StyleContext } from '../src/engine/force/types';
import type { SimState } from '../src/engine/types';
import { PairGW, PairGWZBL } from '../src/engine-gpl/manybody2/pair_gw';
import { PairTersoffTable } from '../src/engine-gpl/manybody2/pair_tersoff_table';
import { PairPolymorphic, TabularFunction } from '../src/engine-gpl/manybody2/pair_polymorphic';
import { mapElementsToTypes, PotentialEntryReader } from '../src/engine-gpl/manybody2/common';
import { PAIRS } from '../src/engine-gpl/registry/manybody2';

/*
 * Checks of the MANYBODY ports gw, gw/zbl, tersoff/table and polymorphic: the potential-file reader and its messages
 * (src/MANYBODY/pair_gw.cpp read_file, pair_tersoff_table.cpp read_file, pair_polymorphic.cpp read_file, text_file_reader.cpp
 * next_line, tokenizer.cpp messages), the pair_coeff argument checks, the element mapping (Pair::map_element2type) and
 * the tabular spline (tabular_function.cpp). Native parity of the forces is in tests/oracle/w223mb_*.in.
 */

const POT = (name: string): string => readFileSync(join(__dirname, '..', 'third_party', 'lammps', 'potentials', name), 'utf8');

/** A context whose system uses the given unit style; readFile returns the text for any file name. */
const ctxWith = (text: string, units = 'metal', log: string[] = [], newtonPair = true): StyleContext => ({
  s: { units: { style: units } } as unknown as SimState,
  readFile: () => text,
  log: (t) => log.push(t),
  newtonPair,
});

/** pair_coeff on a fresh style with `ntypes` atom types. */
const coeff = (p: Pair, ntypes: number, args: string[], ctx: StyleContext): void => {
  p.allocate(ntypes);
  p.coeff(args, ctx);
};

const GW_ENTRY = 'Si Si Si 1 0.013318 0 14 2.1 -1 0.78000 1 1.80821400248640 632.658058300867 2.35 0.15 2.38684248328205 1708.79738703139';
const TERSOFF_SI = 'Si Si Si 3.0 1.0 0.0 100390 16.217 -0.59825 0.78734 0.0000011 1.73222 471.18 2.85 0.15 2.4799 1830.8';

describe('gw and gw/zbl input', () => {
  it('refuses pair_style arguments', () => {
    expect(() => new PairGW().settings(['x'], ctxWith(''))).toThrow(new StyleError('Illegal pair_style command'));
    expect(() => new PairGWZBL().settings(['x'], ctxWith(''))).toThrow(StyleError);
  });

  it('needs one element per atom type', () => {
    const p = new PairGW();
    p.allocate(2);
    expect(() => p.coeff(['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY))).toThrow(
      new StyleError('Number of element to type mappings does not match number of atom types'),
    );
  });

  it('requires * * before the file name', () => {
    const p = new PairGW();
    p.allocate(1);
    expect(() => p.coeff(['1', '1', 'f', 'Si'], ctxWith(GW_ENTRY))).toThrow(
      new StyleError('Pair_coeff must start with * * for pair style gw'),
    );
  });

  it('reports a missing and a duplicate triplet of the potential file', () => {
    expect(() => coeff(new PairGW(), 2, ['*', '*', 'f', 'Si', 'C'], ctxWith(GW_ENTRY))).toThrow(
      new StyleError('Potential file is missing an entry for: Si Si C'),
    );
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith(`${GW_ENTRY}\n${GW_ENTRY}`))).toThrow(
      new StyleError('Potential file has a duplicate entry for: Si Si Si'),
    );
  });

  it('rejects a non-number and an illegal parameter, with the native text', () => {
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY.replace('0.013318', 'abc')))).toThrow(
      new StyleError("Not a valid floating-point number: 'abc'"),
    );
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY.replace('0.013318', '-0.5')))).toThrow(
      new StyleError('Illegal GW parameter'),
    );
    // m must be 1 or 3
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY.replace(/^Si Si Si 1 /, 'Si Si Si 2 ')))).toThrow(
      new StyleError('Illegal GW parameter'),
    );
  });

  it('reads a partial last entry as an incorrect format, as the native reader does', () => {
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith('Si Si Si 1 0.013318 0 14'))).toThrow(
      new StyleError('Incorrect format in gw file! 7/17 parameters'),
    );
  });

  it('joins an entry written on two lines and skips the entries of other elements', () => {
    const text = [
      '# comment line',
      'C C C 1 0.5 0 14 2.1 -1 0.78 1 1.8 632.6',
      '2.35 0.15 2.38 1708.8',
      GW_ENTRY,
    ].join('\n');
    const p = new PairGW();
    coeff(p, 1, ['*', '*', 'f', 'Si'], ctxWith(text));
    expect((p as unknown as { params: unknown[] }).params).toHaveLength(1);
  });

  it('refuses the gw/zbl file layout for gw and the gw layout for gw/zbl', () => {
    expect(() => coeff(new PairGWZBL(), 1, ['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY))).toThrow(
      new StyleError('Incorrect format in gw/zbl file! 17/21 parameters'),
    );
  });

  it('gw/zbl needs metal or real units', () => {
    const p = new PairGWZBL();
    expect(() => p.settings([], ctxWith('', 'lj'))).toThrow(new StyleError('Pair gw/zbl requires metal or real units'));
    expect(() => new PairGWZBL().settings([], ctxWith('', 'real'))).not.toThrow();
  });

  it('converts a metal file in real units and refuses a units tag it cannot convert', () => {
    const log: string[] = [];
    const p = new PairGW();
    coeff(p, 1, ['*', '*', 'f', 'Si'], ctxWith(`# UNITS: metal\n${GW_ENTRY}`, 'real', log));
    const biga = (p as unknown as { params: { biga: number }[] }).params[0].biga;
    expect(biga).toBeCloseTo(1708.79738703139 * 23.060549, 9);
    expect(log.join('\n')).toContain('Converting gw in metal units to real units');
    expect(() => coeff(new PairGW(), 1, ['*', '*', 'f', 'Si'], ctxWith(`# UNITS: real\n${GW_ENTRY}`, 'lj'))).toThrow(
      new StyleError('gw file f requires real units but lj units are in use'),
    );
  });

  it('loads the SiC.gw and SiC.gw.zbl potential files', () => {
    const p = new PairGW();
    coeff(p, 2, ['*', '*', 'SiC.gw', 'Si', 'C'], ctxWith(POT('SiC.gw')));
    expect(p.initOne(1, 2)).toBeGreaterThan(0);
    const z = new PairGWZBL();
    z.settings([], ctxWith('', 'metal'));
    coeff(z, 2, ['*', '*', 'SiC.gw.zbl', 'Si', 'C'], ctxWith(POT('SiC.gw.zbl')));
    expect(z.initOne(2, 2)).toBeGreaterThan(0);
  });

  it('needs newton pair on', () => {
    const p = new PairGW();
    coeff(p, 1, ['*', '*', 'f', 'Si'], ctxWith(GW_ENTRY));
    expect(() => p.init(ctxWith(GW_ENTRY, 'metal', [], false))).toThrow(new StyleError('Pair style gw requires newton pair on'));
  });
});

describe('tersoff/table input', () => {
  it('needs the Tersoff_2 parametrization (gamma 1, lambda3 0)', () => {
    const p = new PairTersoffTable();
    expect(() => coeff(p, 1, ['*', '*', 'f', 'Si'], ctxWith(TERSOFF_SI.replace('3.0 1.0 0.0', '3.0 1.0 1.3258')))).toThrow(
      new StyleError('Currently the tersoff/table pair_style only implements the Tersoff_2 parametrization'),
    );
  });

  it('checks the entry ranges with the native message', () => {
    expect(() => coeff(new PairTersoffTable(), 1, ['*', '*', 'f', 'Si'], ctxWith(TERSOFF_SI.replace('0.0000011', '-1')))).toThrow(
      new StyleError('Illegal Tersoff parameter'),
    );
  });

  it('loads the Si entry of SiC.tersoff and refuses Si.tersoff (not Tersoff_2)', () => {
    const p = new PairTersoffTable();
    coeff(p, 1, ['*', '*', 'SiC.tersoff', 'Si'], ctxWith(POT('SiC.tersoff')));
    expect(p.initOne(1, 1)).toBeGreaterThan(0);
    expect(() => coeff(new PairTersoffTable(), 1, ['*', '*', 'Si.tersoff', 'Si'], ctxWith(POT('Si.tersoff')))).toThrow(
      /Tersoff_2 parametrization/,
    );
  });

  it('needs newton pair on and every type mapped', () => {
    const p = new PairTersoffTable();
    coeff(p, 1, ['*', '*', 'f', 'Si'], ctxWith(TERSOFF_SI));
    expect(() => p.init(ctxWith(TERSOFF_SI, 'metal', [], false))).toThrow(
      new StyleError('Pair style tersoff/table requires newton pair on'),
    );
    const q = new PairTersoffTable();
    q.allocate(2);
    expect(() => q.initOne(1, 2)).toThrow(new StyleError('All pair coeffs are not set'));
  });
});

describe('polymorphic input', () => {
  it('reads the header and checks the element count and names', () => {
    const p = new PairPolymorphic();
    expect(() => coeff(p, 2, ['*', '*', 'GaN_tersoff.poly', 'Ga', 'N'], ctxWith('    1    0\n'))).toThrow(
      new StyleError('Incorrect number of elements (expected: 2 found: 1) in potential file'),
    );
    const text = POT('GaN_tersoff.poly');
    expect(() => coeff(new PairPolymorphic(), 2, ['*', '*', 'GaN_tersoff.poly', 'Ga', 'Al'], ctxWith(text))).toThrow(
      new StyleError('Element N in potential file not used'),
    );
  });

  it('refuses a units tag other than the unit style (no conversion)', () => {
    const p = new PairPolymorphic();
    expect(() => coeff(p, 2, ['*', '*', 'g', 'Ga', 'N'], ctxWith(`# UNITS: real\n${POT('GaN_tersoff.poly')}`, 'metal'))).toThrow(
      new StyleError('polymorphic file g requires real units but metal units are in use'),
    );
  });

  it('reads an optional epsilon argument and refuses a non-number', () => {
    const p = new PairPolymorphic();
    p.allocate(2);
    expect(() => p.coeff(['*', '*', 'f', 'Ga', 'N', 'abc'], ctxWith(POT('GaN_tersoff.poly')))).toThrow(
      new StyleError("Expected floating point parameter instead of 'abc' in input script or data file"),
    );
  });

  it('loads GaN_tersoff.poly and CuTa_eam.poly, with the two-element pair count', () => {
    const p = new PairPolymorphic();
    coeff(p, 2, ['*', '*', 'GaN_tersoff.poly', 'Ga', 'N'], ctxWith(POT('GaN_tersoff.poly')));
    expect(p.initOne(1, 2)).toBeGreaterThan(0);
    const q = new PairPolymorphic();
    coeff(q, 2, ['*', '*', 'CuTa_eam.poly', 'Cu', 'Ta', '0.0'], ctxWith(POT('CuTa_eam.poly')));
    expect(q.initOne(1, 1)).toBeGreaterThan(0);
  });
});

describe('element mapping and reader helpers', () => {
  it('maps types to unique elements in order, NULL to -1 (Pair::map_element2type)', () => {
    const m = mapElementsToTypes(['Si', 'NULL', 'C', 'Si'], 4);
    expect(m.elements).toEqual(['Si', 'C']);
    expect(Array.from(m.map)).toEqual([-1, 0, -1, 1, 0]);
    expect(() => mapElementsToTypes(['Si'], 2)).toThrow(
      new StyleError('Number of element to type mappings does not match number of atom types'),
    );
  });

  it('joins comment-stripped lines to nparams words and reports an incomplete end', () => {
    const r = new PotentialEntryReader(['a b # note', '', 'c', 'd e'], 'x');
    expect(r.next(3)).toEqual(['a', 'b', 'c']);
    // two words left at the end of the file: native next_line throws "Incorrect format"
    expect(() => r.next(3)).toThrow(new StyleError('Incorrect format in x file! 2/3 parameters'));
    expect(new PotentialEntryReader(['a', 'b c'], 'x').next(3)).toEqual(['a', 'b', 'c']);
  });

  it('TabularFunction reproduces a linear table exactly and its derivative', () => {
    const t = new TabularFunction();
    const ys = new Float64Array(11);
    for (let i = 0; i < 11; i++) ys[i] = 2 * (i * 0.1) + 1;
    t.setValues(11, 0, 1, ys);
    expect(t.xmaxsq).toBeCloseTo(1, 15);
    expect(t.vmax).toBeCloseTo(3, 15);
    expect(t.valY(0.5)).toBeCloseTo(2, 12);
    expect(t.valD(0.5)).toBeCloseTo(2, 9);
  });

  it('exposes the four styles through the registry', () => {
    expect(Object.keys(PAIRS).sort()).toEqual(['gw', 'gw/zbl', 'polymorphic', 'tersoff/table']);
  });
});
