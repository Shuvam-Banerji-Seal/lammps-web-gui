/** Recognising a LAMMPS input script by its name or its first command (the storage panel and the Script Builder drop). */

/** A LAMMPS command at the start of a line (first meaningful line of a script). */
const COMMAND_START =
  /^(units|atom_style|atom_modify|boundary|dimension|lattice|region|create_box|create_atoms|read_data|read_restart|pair_style|pair_coeff|pair_modify|bond_style|angle_style|dihedral_style|improper_style|kspace_style|mass|fix|compute|variable|timestep|thermo_style|thermo_modify|run|minimize|min_style|neighbor|neigh_modify|newton|velocity|dump|comm_modify)(\s|$)/;

/** True for a LAMMPS input script: an in.* / *.in / *.lmp / *.lammps name, or text that starts with a LAMMPS command. */
export const isLammpsScript = (name: string, text: string): boolean => {
  if (/^in\./i.test(name) || /\.(in|lmp|lammps)$/i.test(name)) return true;
  const first = text
    .slice(0, 4096)
    .split(/\r?\n/)
    .map((l) => l.trim())
    .find((l) => l !== '' && !l.startsWith('#'));
  return first !== undefined && COMMAND_START.test(first);
};

/** The tab title of an imported script: `in.melt` gives `melt`, `melt.in` or `melt.lmp` gives `melt`. */
export const scriptTitle = (name: string): string => {
  const base = name.split(/[\\/]/).pop() ?? name;
  if (/^in\./i.test(base) && base.length > 3) return base.slice(3);
  return base.replace(/\.[^.]+$/, '') || base;
};
