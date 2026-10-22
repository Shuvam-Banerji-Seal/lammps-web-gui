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

/**
 * The argument checks of native LAMMPS's input parser, with its messages, as native LAMMPS does (src/utils.cpp at
 * b332f28): utils::logical takes yes, on, true or 1 and no, off, false or 0; utils::numeric a decimal number with an
 * optional exponent; utils::inumeric an optional sign and digits; utils::missing_cmd_args names the command.
 */
export const missingArgs = (cmd: string): StyleError => new StyleError(`Illegal ${cmd} command: missing argument(s)`);

export const logical = (w: string | undefined): boolean => {
  if (!w) throw new StyleError('Expected boolean parameter instead of NULL or empty string in input script or data file');
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${w}' in input script or data file`);
};

const DOUBLE = /^[+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?$/;

export const numeric = (w: string | undefined): number => {
  if (!w) throw new StyleError('Expected floating point parameter instead of NULL or empty string in input script or data file');
  if (!DOUBLE.test(w)) throw new StyleError(`Expected floating point parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

/** utils::bnumeric: as inumeric, for a 64-bit integer (timesteps). */
export const bnumeric = (w: string | undefined): number => {
  if (!w) throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  return Number(w);
};

export const inumeric = (w: string | undefined): number => {
  if (!w) throw new StyleError('Expected integer parameter instead of NULL or empty string in input script or data file');
  if (!/^[+-]?\d+$/.test(w)) throw new StyleError(`Expected integer parameter instead of '${w}' in input script or data file`);
  const v = Number(w);
  if (v > 2147483647 || v < -2147483648) throw new StyleError(`Integer ${w} in input script or data file is out of range`);
  return v;
};

/**
 * utils::bounds: a range "N", "*", "*N", "M*" or "M*N" over nmin..nmax, with native's messages (src/utils.cpp at
 * b332f28; a number is read as std::stol reads it, up to the first character that is not a digit).
 */
export const bounds = (str: string, nmin: number, nmax: number): [number, number] => {
  if (/[^*\-0-9]/.test(str)) throw new StyleError(`Invalid range string: ${str}`);
  const stol = (t: string): number => { const m = /^[+-]?\d+/.exec(t); return m ? Number(m[0]) : NaN; };
  const star = str.indexOf('*');
  let lo: number, hi: number;
  if (star < 0) lo = hi = stol(str);
  else if (str.length === 1) { lo = nmin; hi = nmax; }
  else if (star === 0) { lo = nmin; hi = stol(str.slice(1)); }
  else if (star === str.length - 1) { lo = stol(str); hi = nmax; }
  else { lo = stol(str); hi = stol(str.slice(star + 1)); }
  if (!(lo > 0) || !(hi > 0)) throw new StyleError(`Invalid range string: ${str}`);
  const url = '\nFor more information see https://docs.lammps.org/err0019';
  if (lo < nmin) throw new StyleError(`Numeric index ${lo} is out of bounds (${nmin}-${nmax})${url}`);
  if (hi > nmax) throw new StyleError(`Numeric index ${hi} is out of bounds (${nmin}-${nmax})${url}`);
  if (lo > hi) throw new StyleError(`Numeric index ${lo} is out of bounds (${nmin}-${hi})${url}`);
  return [lo, hi];
};
