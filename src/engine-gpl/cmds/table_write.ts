/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the shared part of src/angle_write.cpp
 * and src/dihedral_write.cpp (AngleWrite::command, DihedralWrite::command: the table file header, the units and date
 * checks, the dummy system of the evaluation, and the printed rows). LAMMPS development team, Sandia National
 * Laboratories. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Session } from '../../engine/session';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { formatInteger, formatNumber } from '../../engine/cformat';
import { minEnergyForce } from '../../engine/run/minimizers';
import { currentDate, fileExists, firstLineTag } from './common';

/** The table file and the header that native angle_write / dihedral_write print before the table rows. */
export interface TableTarget {
  /** the file name the command was given */
  file: string;
  /** "angle_write" or "dihedral_write" */
  cmd: string;
}

/** Starts the table file: appends to an existing one (units and date checked), or creates it with its DATE/UNITS line. */
export const startTableFile = (sys: System, t: TableTarget): { head: string; append: boolean } => {
  if (fileExists(sys, t.file)) {
    const text = sys.readFile(t.file);
    const units = firstLineTag(text, 'UNITS');
    if (units !== '' && units !== sys.units.style) {
      throw new StyleError(`Trying to append to a table file with UNITS: ${units} while units are ${sys.units.style}`);
    }
    sys.log(`Appending to table file ${t.file} with DATE: ${firstLineTag(text, 'DATE')}`);
    return { head: '', append: true };
  }
  sys.log(`Creating table file ${t.file} with DATE: ${currentDate()}`);
  return { head: `# DATE: ${currentDate()} UNITS: ${sys.units.style} Created by ${t.cmd}\n`, append: false };
};

/** Native "%8d %- 22.15g %- 22.15g %- 22.15g" row of the table. */
export const tableRow = (i: number, theta: number, phi: number, f: number): string =>
  `${formatInteger(i, '%8d')} ${formatNumber(theta, '%- 22.15g')} ${formatNumber(phi, '%- 22.15g')} ${formatNumber(f, '%- 22.15g')}\n`;

export const finite = (v: number): boolean => Number.isFinite(v);

/**
 * The dummy LAMMPS instance of native angle_write / dihedral_write: a new session that runs the setup lines and the
 * style's coefficient lines (the dataCoeffs lines, one per type), then the setup run. The energy of a geometry is
 * the total energy of that session's system after its position is moved (minEnergyForce, pair zero).
 */
export const dummySession = async (lines: string[]): Promise<System> => {
  const w = new Session({ emit: () => { /* the dummy system's log and thermo are dropped, as native's -screen none */ } });
  await w.execute(lines.join('\n'));
  return w.sys;
};

/** Parses an integer argument as native utils::inumeric(..., false) does. */
export const intArg = (text: string, what: string): number => {
  if (!/^[+-]?\d+$/.test(text)) throw new StyleError(`Expected integer parameter instead of '${text}' for ${what}`);
  return Number(text);
};
