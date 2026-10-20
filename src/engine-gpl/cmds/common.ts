/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/version.h (LAMMPS_VERSION), and
 * the helpers of src/utils.cpp that the write_* commands use (current_date, get_potential_date, get_potential_units,
 * errorurl). LAMMPS development team (Sandia National Laboratories). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import type { System } from '../../engine/system';
import { StyleError } from '../../engine/force/types';

/** src/version.h at b332f28: LAMMPS_VERSION "2 Sep 2026" (written into the headers of the coefficient and table files). */
export const LAMMPS_VERSION = '2 Sep 2026';

/** utils::errorurl(code): "\nFor more information see https://docs.lammps.org/err%04d" for a positive code. */
export const errorUrl = (code: number): string => `\nFor more information see https://docs.lammps.org/err${String(code).padStart(4, '0')}`;

/** utils::current_date(): strftime "%Y-%m-%d" of the local day (the engine uses the UTC day of the run). */
export const currentDate = (): string => new Date().toISOString().slice(0, 10);

/** The word after a "KEY:" tag on the first line of a file (utils::get_potential_date / get_potential_units), or ''. */
export const firstLineTag = (text: string, key: string): string => {
  const line = text.split('\n')[0] ?? '';
  const w = line.trim().split(/\s+/);
  for (let k = 0; k < w.length; k++) if (w[k] === `${key}:` && k + 1 < w.length) return w[k + 1];
  return '';
};

/** Whether a file of the session exists (native platform::file_is_readable for the virtual file store). */
export const fileExists = (sys: System, name: string): boolean => sys.files.has(sys.resolveFile(name));

/** utils::inumeric(FLERR, text, false): an integer argument, or native's "Expected integer parameter" error. */
export const nativeInt = (text: string): number => {
  if (!/^[+-]?\d+$/.test(text)) throw new StyleError(`Expected integer parameter instead of '${text}' in input script or data file`);
  return Number(text);
};

/** utils::numeric(FLERR, text, false): a floating point argument, or native's "Expected floating point parameter" error. */
export const nativeFloat = (text: string): number => {
  if (!/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(text)) {
    throw new StyleError(`Expected floating point parameter instead of '${text}' in input script or data file`);
  }
  return Number(text);
};
