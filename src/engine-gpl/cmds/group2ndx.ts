/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMMAND/group2ndx.cpp and
 * src/EXTRA-COMMAND/group2ndx.h (Group2Ndx::command and write_group; LAMMPS development team, Sandia National
 * Laboratories). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * group2ndx file [group ...] — group2ndx.html: "write the atom IDs of groups to a GROMACS index file": each group is a
 * "[ name ]" section of atom IDs in ascending order, with the group "all" written as "[ System ]".
 */
import type { Handler } from '../../engine/commands/args';
import { StyleError } from '../../engine/force/types';
import { bitOfIndex } from '../../engine/group';
import type { System } from '../../engine/system';

/** The index file text of one group (native write_group): "[ name ]", then the IDs, cols per line, right aligned. */
const groupSection = (sys: System, k: number, width: number): string => {
  const s = sys.state;
  const bit = bitOfIndex(k);
  const ids: number[] = [];
  for (let i = 0; i < s.n; i++) if ((s.mask[i] & bit) !== 0) ids.push(s.id[i]);
  ids.sort((x, y) => x - y);
  const name = sys.groups.names[k] ?? '';
  // the "all" group of LAMMPS is called "System" in Gromacs
  let out = k === 0 ? '[ System ]\n' : `[ ${name} ]\n`;
  const cols = Math.floor(80 / width);
  let j = 0;
  for (const id of ids) {
    out += String(id).padStart(width);
    j++;
    if (j === cols) { out += '\n'; j = 0; }
  }
  if (j > 0) out += '\n';
  return out;
};

const group2ndx: Handler = ({ sys }, a) => {
  if (a.length < 1) throw new StyleError('Illegal group2ndx command: missing argument(s)');
  const file = a[0];
  sys.log(`Writing groups to index file ${file}:`);
  // native: "width = (int) log10((double) atom->natoms) + 2" characters per ID (the digit count of natoms, plus one);
  // 80 characters per line
  const natoms = sys.hasBox ? sys.state.n : 0;
  const width = (natoms > 0 ? String(natoms).length - 1 : 0) + 2;
  let out = '';
  const groupsToWrite: number[] = [];
  if (a.length === 1) {
    for (let k = 0; k < sys.groups.names.length; k++) if (sys.groups.names[k] !== null) groupsToWrite.push(k);
  } else {
    for (const name of a.slice(1)) {
      const k = sys.groups.find(name);
      if (k < 0) throw new StyleError('Non-existing group requested');
      groupsToWrite.push(k);
    }
  }
  for (const k of groupsToWrite) {
    const name = sys.groups.names[k] ?? '';
    // group names with white space are skipped (native strmatch "\\s+")
    if (/\s/.test(name)) { sys.log(` skipping group ${name}...done`); continue; }
    sys.log(` writing group ${name}...done`);
    out += groupSection(sys, k, width);
  }
  sys.writeFile(file, out, false);
};

export const COMMANDS = { group2ndx };
