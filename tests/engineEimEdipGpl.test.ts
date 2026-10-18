import { describe, expect, it } from 'vitest';
import { parseEimFile, eimUnitFactor } from '../src/engine-gpl/manybody3/pair_eim';
import { readEdipFile } from '../src/engine-gpl/manybody3/pair_edip';
import type { StyleContext } from '../src/engine/force/types';

/*
 * Parsing and native error texts of the GPL ports pair_style eim (src/MANYBODY/pair_eim.cpp) and pair_style
 * edip, edip/multi (src/MANYBODY/pair_edip.cpp, pair_edip_multi.cpp). Numerical parity is in tests/oracle/w14eim_*,
 * w22pot_eim_nacl, w48pb_edip_*.
 */

const ctxWith = (units: string): StyleContext => ({
  s: { units: { style: units } } as unknown as StyleContext['s'],
  readFile: () => '',
  log: () => undefined,
} as unknown as StyleContext);

const EIM_OK = [
  '# comment line',
  'global: 1.0 0.5 1.5',
  'element: Qa 1 22.99 1.0 1.0 1.0 1.0 1.0',
  'element: Qb 2 35.45 3.0 1.0 1.0 1.0 1.0',
  'pair: Qa Qb 6.0 6.0 1.0 2.5 2.0 3.0 2.0 1.0 1.0 4.0 1.0 1.0 1.0 2.0',
  'pair: Qa Qa 5.0 5.0 1.0 2.5 2.0 3.0 2.0 1.0 1.0 4.0 1.0 1.0 1.0 2.0',
  'pair: Qb Qb 5.0 5.0 1.0 2.5 2.0 3.0 2.0 1.0 1.0 4.0 1.0 1.0 1.0 2.0',
].join('\n');

describe('pair_style eim file parsing (pair_eim.cpp EIMPotentialFileReader)', () => {
  it('reads global, element and pair lines and keys pairs by sorted names', () => {
    const f = parseEimFile(EIM_OK);
    expect(f.rbig).toBe(0.5);
    expect(f.rsmall).toBe(1.5);
    expect(f.elements.get('Qb')?.mass).toBe(35.45);
    expect(f.pairs.get('Qa Qb')?.rcutphiA).toBe(6.0);
  });

  it('multiplies Eb and Ac by the conversion factor', () => {
    const f = parseEimFile(EIM_OK, 2);
    expect(f.pairs.get('Qa Qa')?.Eb).toBe(2.0 * 1.0);
    expect(f.pairs.get('Qa Qa')?.Ac).toBe(2.0 * 1.0);
  });

  it('joins continuation lines ending in &', () => {
    const text = EIM_OK.replace('pair: Qa Qa 5.0 5.0 1.0 2.5 2.0 3.0 2.0 1.0 1.0 4.0', 'pair: Qa Qa 5.0 5.0 1.0 2.5 2.0 3.0 & \n 2.0 1.0 1.0 4.0');
    expect(parseEimFile(text).pairs.get('Qa Qa')?.rs).toBe(1.0);
  });

  it('native error texts', () => {
    expect(() => parseEimFile('element: Qa 1 22.99 1.0 1.0 1.0 1.0 1.0\n')).toThrow('Missing global line in EIM potential file');
    expect(() => parseEimFile('global: 1.0 0.5\n')).toThrow('Invalid global line in EIM potential file');
    expect(() => parseEimFile('global: 1.0 0.5 1.5\nelement: Qa 1 22.99 1.0\n')).toThrow('Invalid element line in EIM potential file');
    expect(() => parseEimFile('global: 1.0 0.5 1.5\nelement: Qa 1 22.99 1.0 1.0 1.0 1.0 1.0\nelement: Qa 1 22.99 1.0 1.0 1.0 1.0 1.0\n'))
      .toThrow('Duplicate pair line in EIM potential file');
    expect(() => parseEimFile('global: 1.0 0.5 1.5\npair: Qa Qa 5 5 1 2.5 2 3 2 1 1 4 1 1 1 2\npair: Qa Qa 5 5 1 2.5 2 3 2 1 1 4 1 1 1 2\n'))
      .toThrow('Duplicate pair line in EIM potential file');
    expect(() => parseEimFile('global: 1.0 0.5 1.5\nelement: Qa 1 22.99 x 1.0 1.0 1.0 1.0\n')).toThrow("Not a valid floating-point number: 'x'");
  });

  it('unit factor: metal and real convert, other mismatches name the file', () => {
    expect(eimUnitFactor('# plain\n', 'f.eim', ctxWith('metal'))).toBe(1);
    expect(eimUnitFactor('# UNITS: metal\n', 'f.eim', ctxWith('real'))).toBeCloseTo(23.060549, 9);
    expect(() => eimUnitFactor('# UNITS: si\n', 'f.eim', ctxWith('metal')))
      .toThrow('Potential file f.eim requires si units but metal units are in use');
  });
});

describe('pair_style edip file reading (pair_edip.cpp read_file)', () => {
  const NUMS = '1.0 2.0 3.0 4.0 0.5 0.6 0.7 0.8 0.9 1.1 1.2 1.3 1.4 0.1 0.2 0.3 0.4';
  const ENTRY = `Si Si Si ${NUMS}`;

  it('keeps only entries whose three names are mapped and reads 17 numbers', () => {
    const ps = readEdipFile(`${ENTRY}\nC C C ${NUMS}\n`, 'Si.edip', ['Si'], ctxWith('metal'));
    expect(ps.length).toBe(1);
    expect(ps[0].A).toBe(1.0);
    expect(ps[0].u4).toBe(0.4);
  });

  it('native error texts', () => {
    expect(() => readEdipFile('Si Si Si 1.0 x 3.0 4.0 0.5 0.6 0.7 0.8 0.9 1.1 1.2 1.3 1.4 0.1 0.2 0.3 0.4\n', 'Si.edip', ['Si'], ctxWith('metal')))
      .toThrow("Error reading EDIP potential file: Not a valid floating-point number: 'x'");
    expect(() => readEdipFile('Si Si Si -1.0 2.0 3.0 4.0 0.5 0.6 0.7 0.8 0.9 1.1 1.2 1.3 1.4 0.1 0.2 0.3 0.4\n', 'Si.edip', ['Si'], ctxWith('metal')))
      .toThrow('Illegal EDIP parameter');
    expect(() => readEdipFile('Si Si Si 1.0 2.0\n', 'Si.edip', ['Si'], ctxWith('metal')))
      .toThrow('Incorrect format in edip potential file! 5/20 parameters');
    expect(() => readEdipFile('# UNITS: real\n', 'Si.edip', ['Si'], ctxWith('metal')))
      .toThrow('edip potential file Si.edip requires real units but metal units are in use');
  });
});
