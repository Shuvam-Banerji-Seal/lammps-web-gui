/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the dump styles dcd and xtc
 * (src/EXTRA-DUMP/dump_dcd.cpp and dump_xtc.cpp). This registry module is the lazily loaded helper of the core dump
 * command: its HOOKS entries 'dcd' and 'xtc' are stored under the command words, and the core Dump reads them with
 * gplHook('dcd') and gplHook('xtc') (lazy group gpl_dump_traj in src/engine/lazyStyles.ts; the names listed there must
 * match these keys).
 */
import { createDumpDcd } from '../dump/dump_dcd';
import { createDumpXtc } from '../dump/dump_xtc';

/** Helpers the core reads by command word (styles.ts loadStyleGroup). */
export const HOOKS: Record<string, unknown> = {
  dcd: createDumpDcd,
  xtc: createDumpXtc,
};
