/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the command registrations of src/REPLICA/temper.h (Command temper), src/REPLICA/temper_npt.h (temper/npt) and
 * src/REPLICA/temper_grem.h (temper/grem) at commit b332f28; the commands are in ../replica/temper.ts (see its header).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_replica); the session runs them once, over the replicas
 * (src/engine/session.ts REPLICA_DRIVEN).
 */
import { COMMANDS as TEMPER_COMMANDS } from '../replica/temper';

export const COMMANDS = TEMPER_COMMANDS;
