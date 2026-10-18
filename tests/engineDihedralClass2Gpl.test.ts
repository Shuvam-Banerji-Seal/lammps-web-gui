import { describe, expect, it } from 'vitest';
import { DihedralClass2 } from '../src/engine-gpl/class2/dihedral_class2';
import { DIHEDRALS } from '../src/engine-gpl/registry/dihedral_class2';
import { StyleError } from '../src/engine/force/types';

/*
 * Coefficient parsing and native error texts of the GPL dihedral_style class2 port (src/CLASS2/dihedral_class2.cpp at
 * b332f28: coeff, init, settings, write_data). Energies and forces are checked against native LAMMPS by the w249cd and
 * w35c2dih* oracle cases in engineOracle.test.ts.
 */

const MAIN = ['1', '100.0', '75.0', '70.0', '100.0', '80.0', '60.0'];
const CROSS: string[][] = [
  ['1', 'mbt', '3.5', '0.17', '-0.55', '1.02'],
  ['1', 'ebt', '0.34', '0.33', '-0.90', '0.14', '0.0', '-0.81', '0.95', '1.03'],
  ['1', 'at', '0.0', '-0.185', '-0.796', '-2.02', '0.0', '-0.399', '110.2', '105.1'],
  ['1', 'aat', '-13.5', '110.2', '105.1'],
  ['1', 'bb13', '0.5', '0.95', '1.03'],
];

const complete = (ntypes = 1): DihedralClass2 => {
  const d = new DihedralClass2();
  d.allocate(ntypes);
  d.coeff(MAIN);
  for (const line of CROSS) d.coeff(line);
  return d;
};

describe('dihedral_style class2 (GPL port) coefficients', () => {
  it('is registered as DIHEDRALS class2', () => {
    const made = DIHEDRALS.class2();
    expect(made).toBeInstanceOf(DihedralClass2);
    expect(made.name).toBe('class2');
  });

  it('reads the main line with phi in degrees and writes it back in degrees (write_data)', () => {
    const d = complete();
    d.init();
    expect(d.dataCoeffs()[0]).toBe('1 100 75 70 100 80 60');
  });

  it('converts the at and aat angles to radians and writes them back in degrees', () => {
    const d = complete();
    const sections = d.dataCrossSections();
    expect(sections.map((s) => s.title)).toEqual([
      'AngleAngleTorsion Coeffs', 'EndBondTorsion Coeffs', 'MiddleBondTorsion Coeffs', 'BondBond13 Coeffs', 'AngleTorsion Coeffs',
    ]);
    const byTitle = Object.fromEntries(sections.map((s) => [s.title, s.lines[0]]));
    expect(byTitle['AngleAngleTorsion Coeffs']).toBe('1 -13.5 110.2 105.1');
    expect(byTitle['AngleTorsion Coeffs']).toBe('1 0 -0.185 -0.796 -2.02 0 -0.399 110.2 105.1');
    expect(byTitle['MiddleBondTorsion Coeffs']).toBe('1 3.5 0.17 -0.55 1.02');
    expect(byTitle['EndBondTorsion Coeffs']).toBe('1 0.34 0.33 -0.9 0.14 0 -0.81 0.95 1.03');
    expect(byTitle['BondBond13 Coeffs']).toBe('1 0.5 0.95 1.03');
  });

  it('rejects a cross-term line with the wrong number of values (native narg check and errorurl 21)', () => {
    const d = new DihedralClass2();
    d.allocate(1);
    expect(() => d.coeff(['1', 'mbt', '1', '2', '3'])).toThrow(StyleError);
    expect(() => d.coeff(['1', 'mbt', '1', '2', '3'])).toThrow(
      'Incorrect args for dihedral coefficients\nFor more information see https://docs.lammps.org/err0021',
    );
    expect(() => d.coeff(['1', 'aat', '1', '2'])).toThrow('Incorrect args for dihedral coefficients');
  });

  it('rejects a main line with the wrong number of values', () => {
    const d = new DihedralClass2();
    d.allocate(1);
    expect(() => d.coeff(['1', '1', '2', '3'])).toThrow(/^Incorrect args for dihedral coefficients/);
  });

  it('names a non-numeric value with the native utils::numeric text', () => {
    const d = new DihedralClass2();
    d.allocate(1);
    expect(() => d.coeff(['1', '100', 'x', '70', '100', '80', '60'])).toThrow(
      "Expected floating point parameter instead of 'x' in input script or data file",
    );
    expect(() => d.coeff(['1', 'bb13', '0.5', '', '1.0'])).toThrow(
      'Expected floating point parameter instead of NULL or empty string in input script or data file',
    );
  });

  it('needs a coefficient line for the main term and each cross term (Dihedral::init status text)', () => {
    const d = new DihedralClass2();
    d.allocate(2);
    d.coeff(MAIN);
    for (const line of CROSS) d.coeff(line);
    d.coeff(['2', ...MAIN.slice(1)]);
    expect(() => d.init()).toThrow(/^All dihedral coeffs are not set\. Status:\n {5}1: is set\n {5}2: is not set$/);
  });

  it('rejects a type range outside 1..ntypes', () => {
    const d = new DihedralClass2();
    d.allocate(1);
    expect(() => d.coeff(['2', ...MAIN.slice(1)])).toThrow(StyleError);
  });

  it('takes no style arguments (Dihedral::settings text)', () => {
    const d = new DihedralClass2();
    expect(() => d.settings(['extra'])).toThrow('Illegal dihedral_style class2 argument: extra');
  });
});
