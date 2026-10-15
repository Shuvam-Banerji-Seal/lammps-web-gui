/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the binary restart reader of src/read_restart.cpp
 * (see engine-gpl/restart/nativeRestart.ts for the list of files). LAMMPS is Copyright (2003) Sandia Corporation and the
 * LAMMPS development team, distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'native-restart' (lazyStyles.ts): read_restart calls it for a file that starts with the native
 * magic string (output/restart.ts readRestartText), so the core never imports this module.
 */
import { readNativeRestart } from '../restart/nativeRestart';

export const HOOKS: Record<string, unknown> = {
  'native-restart': readNativeRestart,
};
