/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the commands of src/write_coeff.cpp,
 * src/write_molecule.cpp, src/angle_write.cpp, src/dihedral_write.cpp, src/EXTRA-COMMAND/group2ndx.cpp,
 * src/EXTRA-COMMAND/ndx2group.cpp and src/GRAPHICS/region2vmd.cpp (see each file under src/engine-gpl/cmds/ for its
 * header). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The lazy COMMANDS group gpl_cmds (src/engine/lazyStyles.ts): the session loads it before a command of these words.
 */
import type { Handler } from '../../engine/commands/args';
import { COMMANDS as WRITE_COEFF } from '../cmds/write_coeff';
import { COMMANDS as WRITE_MOLECULE } from '../cmds/write_molecule';
import { COMMANDS as ANGLE_WRITE } from '../cmds/angle_write';
import { COMMANDS as DIHEDRAL_WRITE } from '../cmds/dihedral_write';
import { COMMANDS as PAIR_WRITE } from '../cmds/pair_write';
import { COMMANDS as BOND_WRITE } from '../cmds/bond_write';
import { COMMANDS as GROUP2NDX } from '../cmds/group2ndx';
import { COMMANDS as NDX2GROUP } from '../cmds/ndx2group';
import { region2vmd } from '../cmds/region2vmd';

export const COMMANDS: Record<string, Handler> = {
  ...WRITE_COEFF,
  ...WRITE_MOLECULE,
  ...ANGLE_WRITE,
  ...DIHEDRAL_WRITE,
  ...PAIR_WRITE,
  ...BOND_WRITE,
  ...GROUP2NDX,
  ...NDX2GROUP,
  region2vmd,
};
