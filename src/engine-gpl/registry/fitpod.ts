/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the fitpod command
 * (src/ML-POD/fitpod_command.cpp; contributing author Ngoc Cuong Nguyen (MIT)). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The lazy COMMANDS group gpl_fitpod (src/engine/lazyStyles.ts): the session loads it before a fitpod command.
 */
import type { Handler } from '../../engine/commands/args';
import { COMMANDS as FITPOD } from '../pod/fitpod';

export const COMMANDS: Record<string, Handler> = {
  ...FITPOD,
};
