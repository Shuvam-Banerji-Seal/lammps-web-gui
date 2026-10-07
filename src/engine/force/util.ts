import { StyleError } from './types';

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

/** Coefficients in data files: full precision, compact. */
export const fmtCoeff = (v: number): string => (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(16))));
