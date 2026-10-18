import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { PAIRS } from '../src/engine-gpl/registry/manyeam';
import { PairADP } from '../src/engine-gpl/manyeam/pair_adp';
import { PairEAMCD } from '../src/engine-gpl/manyeam/pair_eam_cd';
import { PairEAMHE } from '../src/engine-gpl/manyeam/pair_eam_he';
import { interpolate, readAdp, readFs, readHCoeff, readSetfl } from '../src/engine-gpl/manyeam/potential';
import { StyleError, type StyleContext } from '../src/engine/force/types';

/*
 * File-format and argument errors of the setfl / FS(he) / ADP / eam-cd readers, the element mapping of pair_coeff, and the
 * registry of the lazy group gpl_manyeam. The error texts are the native LAMMPS messages (src/MANYBODY/pair_eam.cpp,
 * pair_eam_cd.cpp, pair_adp.cpp, potential_file_reader.cpp); the numbers are not compared here (tests/oracle/w222me_*).
 */

const potential = (name: string): string => readFileSync(join(process.cwd(), 'third_party/lammps/potentials', name), 'utf8');

/** A context with the metal unit style and the given files (the names are the pair_coeff file arguments). */
const ctxWith = (files: Record<string, string>, units = 'metal', masses?: Float64Array): StyleContext =>
  ({
    s: { units: { style: units }, massByType: masses ?? new Float64Array(8).fill(Number.NaN), rmass: null },
    readFile: (name: string) => {
      if (name in files) return files[name];
      throw new Error(`no such file ${name}`);
    },
    log: () => {},
  }) as unknown as StyleContext;

/** A one-element setfl text (Nrho 2, Nr 2) whose data are all zero. */
const TINY_SETFL_ONE = ['c1', 'c2', 'c3', '1 Fe', '2 0.1 2 0.1 3.0', '26 55.8 2.8 bcc', '0 0', '0 0', '0 0', '0 0'].join('\n') + '\n';

describe('manyeam registry', () => {
  it('registers the four pair styles under their native names', () => {
    expect(Object.keys(PAIRS).sort()).toEqual(['adp', 'eam/cd', 'eam/cd/old', 'eam/he']);
    expect(PAIRS['eam/cd/old']().name).toBe('eam/cd/old');
    expect(PAIRS['eam/cd']().name).toBe('eam/cd');
    expect(PAIRS['eam/he']().name).toBe('eam/he');
    expect(PAIRS.adp().name).toBe('adp');
  });

  it('refuses pair_style arguments (native "Illegal pair_style command")', () => {
    expect(() => new PairADP().settings(['x'])).toThrow(StyleError);
    expect(() => new PairEAMHE().settings(['x'])).toThrow('Illegal pair_style command');
    expect(() => new PairEAMCD('eam/cd').settings(['x'])).toThrow('Illegal pair_style command');
  });
});

describe('potential file readers', () => {
  it('reads the grid, element names and masses of a setfl file', () => {
    const f = readSetfl(potential('FeCr.cdeam'), 'FeCr.cdeam', ctxWith({}));
    expect(f.names).toEqual(['Fe', 'Cr']);
    expect(f.nrho).toBe(5000);
    expect(f.nr).toBe(5000);
    expect(f.cut).toBeCloseTo(5.5, 12);
    expect(f.mass[0]).toBeCloseTo(55.845, 9);
    expect(f.z2r[1][0].length).toBe(f.nr + 1);
  });

  it('reads the he grid line (six values, rhomax last) of an eam/he file', () => {
    const f = readFs(potential('PdHHe.eam.he'), 'PdHHe.eam.he', ctxWith({}), true);
    expect(f.names).toEqual(['Pd', 'H', 'He']);
    expect(f.nrho).toBe(10000);
    expect(f.rhomax).toBeCloseTo(137.8101, 6);
    // one density per (site, neighbour) element pair: Nelements^2 arrays
    expect(f.rhor.length).toBe(9);
  });

  it('reads the ADP file with u2r and w2r blocks', () => {
    const f = readAdp(potential('Ni.adp'), 'Ni.adp', ctxWith({}));
    expect(f.names).toEqual(['Ni']);
    expect(f.u2r?.[0][0].length).toBe(f.nr + 1);
    expect(f.w2r?.[0][0].length).toBe(f.nr + 1);
  });

  it('refuses a potential file that requires other units (no conversion here)', () => {
    const text = TINY_SETFL_ONE.replace('c1', 'UNITS: real');
    expect(() => readSetfl(text, 'real.eam', ctxWith({}, 'metal'))).toThrow(/requires real units but metal units are in use/);
  });

  it('rejects an element line that does not list Nelements names', () => {
    const text = ['c1', 'c2', 'c3', '2 Fe', '2 0.1 2 0.1 3.0'].join('\n') + '\n';
    expect(() => readSetfl(text, 'bad.eam', ctxWith({}))).toThrow('Incorrect element names in EAM potential file');
    expect(() => readAdp(text, 'bad.adp', ctxWith({}))).toThrow('Incorrect element names in ADP potential file');
  });

  it('rejects a grid with a non-positive Nrho or dr', () => {
    const text = ['c1', 'c2', 'c3', '1 Fe', '0 0.1 2 0.1 3.0', '26 55.8 2.8 bcc', '0 0'].join('\n') + '\n';
    expect(() => readSetfl(text, 'bad.eam', ctxWith({}))).toThrow('Invalid EAM potential file');
    const text2 = ['c1', 'c2', 'c3', '1 Fe', '2 0.1 2 0.0 3.0', '26 55.8 2.8 bcc', '0 0'].join('\n') + '\n';
    expect(() => readFs(text2, 'bad.fs', ctxWith({}), false)).toThrow('Invalid EAM potential file');
  });

  it('reports a truncated table (native: incorrect format / end of file)', () => {
    const text = ['c1', 'c2', 'c3', '1 Fe', '2 0.1 2 0.1 3.0', '26 55.8 2.8 bcc', '0 0'].join('\n') + '\n';
    expect(() => readSetfl(text, 'short.eam', ctxWith({}))).toThrow(/Incorrect format|unexpected end/);
  });

  it('reads a non-numeric table entry as an error naming the word', () => {
    const text = ['c1', 'c2', 'c3', '1 Fe', '2 0.1 2 0.1 3.0', '26 55.8 2.8 bcc', 'x y', '0 0', '0 0', '0 0'].join('\n') + '\n';
    expect(() => readSetfl(text, 'nan.eam', ctxWith({}))).toThrow("Expected floating-point number but found 'x'");
  });
});

describe('eam/cd h(x) coefficients', () => {
  it('reads the degree and the coefficients from the last line', () => {
    const text = 'x'.repeat(1100) + '\n# comment\n1 0.5 0.25\n';
    expect(Array.from(readHCoeff(text, 'h.eam'))).toEqual([0.5, 0.25]);
  });

  it('needs a file longer than the 1024-byte search window', () => {
    expect(() => readHCoeff('1 0.5 0.25\n', 'short.eam')).toThrow('Failure to seek to end-of-file for reading h(x) coeffs');
  });

  it('rejects a last line whose word count does not match the degree', () => {
    const text = 'x'.repeat(1100) + '\n2 1 2\n';
    expect(() => readHCoeff(text, 'h.eam')).toThrow('Failed to read h(x) function coefficients in EAM file.');
  });
});

describe('spline tables', () => {
  it('reproduces a linear table exactly (slope 0.5 per grid step, value at the node)', () => {
    const n = 6;
    const f = new Float64Array(n + 1);
    for (let m = 1; m <= n; m++) f[m] = 1 + 0.5 * (m - 1);
    const s = interpolate(n, 0.1, f);
    for (let m = 1; m <= n; m++) {
      expect(s[m * 7 + 6]).toBe(f[m]);
      if (m >= 3 && m <= n - 2) expect(s[m * 7 + 5]).toBeCloseTo(0.5, 15);
      if (m <= n - 1) {
        expect(s[m * 7 + 3]).toBeCloseTo(0, 14);
        expect(s[m * 7 + 4]).toBeCloseTo(0, 14);
      }
    }
  });
});

describe('pair_coeff element mapping and arguments', () => {
  it('needs one element (or NULL) per atom type', () => {
    const pair = new PairEAMCD('eam/cd');
    pair.allocate(2);
    expect(() => pair.coeff(['*', '*', 'FeCr.cdeam', 'Fe'], ctxWith({ 'FeCr.cdeam': potential('FeCr.cdeam') }))).toThrow(
      'Number of element to type mappings does not match number of atom types',
    );
  });

  it('refuses an element that the file does not contain', () => {
    const pair = new PairEAMHE();
    pair.allocate(2);
    expect(() => pair.coeff(['*', '*', 'PdHHe.eam.he', 'Pd', 'Xx'], ctxWith({ 'PdHHe.eam.he': potential('PdHHe.eam.he') }))).toThrow(
      /No matching element in EAM potential file/,
    );
  });

  it('sets the atom masses of the mapped types from the file and skips NULL', () => {
    const masses = new Float64Array(3).fill(Number.NaN);
    const pair = new PairEAMHE();
    pair.allocate(2);
    const ctx = ctxWith({ 'PdHHe.eam.he': potential('PdHHe.eam.he') }, 'metal', masses);
    pair.coeff(['*', '*', 'PdHHe.eam.he', 'NULL', 'H'], ctx);
    expect(masses[2]).toBeCloseTo(1.01, 6);
    expect(Number.isNaN(masses[1])).toBe(true);
    pair.initStyle(ctx);
    // a NULL type that is not under hybrid makes the I,I pair unset (native: "All pair coeffs are not set")
    expect(() => pair.initOne(1, 1)).toThrow('All pair coeffs are not set');
    expect(pair.initOne(1, 2)).toBe(0);
    expect(pair.initOne(2, 2)).toBeCloseTo(7.0, 12);
  });

  it('requires exactly one type for each of the first two elements of eam/cd', () => {
    const files = { 'FeCr.cdeam': potential('FeCr.cdeam') };
    const two = new PairEAMCD('eam/cd');
    two.allocate(2);
    expect(() => two.coeff(['*', '*', 'FeCr.cdeam', 'Fe', 'Fe'], ctxWith(files))).toThrow(
      'The first element from the EAM file may only be mapped to a single atom type.',
    );
    const cr = new PairEAMCD('eam/cd');
    cr.allocate(2);
    expect(() => cr.coeff(['*', '*', 'FeCr.cdeam', 'Cr', 'Cr'], ctxWith(files))).toThrow(
      'The second element from the EAM file may only be mapped to a single atom type.',
    );
    const none = new PairEAMCD('eam/cd/old');
    none.allocate(2);
    expect(() => none.coeff(['*', '*', 'FeCr.cdeam', 'Cr', 'NULL'], ctxWith(files))).toThrow(
      'The first element from the EAM file must be mapped to exactly one atom type.',
    );
  });

  it('requires at least two elements in an eam/cd file', () => {
    const pair = new PairEAMCD('eam/cd');
    pair.allocate(1);
    const files = { one: TINY_SETFL_ONE };
    expect(() => pair.coeff(['*', '*', 'one', 'Fe'], ctxWith(files))).toThrow(
      'The EAM file must contain at least 2 elements to be used with the eam/cd pair style.',
    );
  });

  it('ADP takes "* *" and the listed elements only', () => {
    const pair = new PairADP();
    pair.allocate(1);
    const files = { 'Ni.adp': potential('Ni.adp') };
    expect(() => pair.coeff(['1', '*', 'Ni.adp', 'Ni'], ctxWith(files))).toThrow('Incorrect args for pair coefficients');
    expect(() => pair.coeff(['*', '*', 'Ni.adp', 'Cu'], ctxWith(files))).toThrow('No matching element in ADP potential file');
    expect(() => pair.coeff(['*', '*', 'Ni.adp', 'NULL'], ctxWith(files))).toThrow('Incorrect args for pair coefficients');
  });

  it('ADP maps the element to the type and sets its mass', () => {
    const masses = new Float64Array(2).fill(Number.NaN);
    const pair = new PairADP();
    pair.allocate(1);
    pair.coeff(['*', '*', 'Ni.adp', 'Ni'], ctxWith({ 'Ni.adp': potential('Ni.adp') }, 'metal', masses));
    expect(masses[1]).toBeGreaterThan(50);
    expect(masses[1]).toBeLessThan(60);
  });
});
