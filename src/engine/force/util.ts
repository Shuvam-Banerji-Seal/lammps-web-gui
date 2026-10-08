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

/**
 * Joins potential-file entries that continue over several lines (sw,
 * tersoff, vashishta, ...): measured with native LAMMPS (black box), words are
 * read across lines until an entry has `nwords` of them, and extra words on
 * that last line are ignored. Comments (#...) and blank lines are skipped.
 * Returns the lines with each entry joined onto its first line (exactly
 * `nwords` words) and the continuation lines emptied, so line numbers in
 * error messages stay those of the file.
 */
export const joinPotentialEntries = (lines: readonly string[], nwords: number): string[] => {
  const strip = (r: string): string[] => {
    const h = r.indexOf('#');
    const t = (h >= 0 ? r.slice(0, h) : r).trim();
    return t ? t.split(/\s+/) : [];
  };
  const out = lines.slice();
  for (let ln = 0; ln < lines.length; ln++) {
    let words = strip(lines[ln]);
    if (!words.length) continue;
    let k = ln;
    while (words.length < nwords && k + 1 < lines.length) {
      k++;
      words = words.concat(strip(lines[k]));
      out[k] = '';
    }
    out[ln] = words.slice(0, nwords).join(' ');
    ln = k;
  }
  return out;
};

