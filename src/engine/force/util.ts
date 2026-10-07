import { StyleError } from './types';
import { formatNumber } from '../script';

/** A finite number argument, or a StyleError naming what was expected. */
export const parseNum = (w: string | undefined, what: string): number => {
  if (w === undefined) throw new StyleError(`missing ${what}`);
  const v = Number(w);
  if (w.trim() === '' || !Number.isFinite(v)) throw new StyleError(`expected a number for ${what}, got '${w}'`);
  return v;
};

export const parseInt_ = (w: string | undefined, what: string): number => {
  const v = parseNum(w, what);
  if (!Number.isInteger(v)) throw new StyleError(`expected an integer for ${what}, got '${w}'`);
  return v;
};

/**
 * Coefficients in data files. Measured with native LAMMPS write_data (pair
 * lj/cut, coul/*, lj/cut/coul/* and bond, angle, dihedral, improper
 * harmonic): every "* Coeffs" value is printed as C %g, e.g. 1.23456789 ->
 * 1.23457, 1234567.89 -> 1.23457e+06, and mixed PairIJ values likewise
 * (0.894427); only the Masses section keeps full precision.
 */
export const fmtCoeff = (v: number): string => formatNumber(v, '%g');
