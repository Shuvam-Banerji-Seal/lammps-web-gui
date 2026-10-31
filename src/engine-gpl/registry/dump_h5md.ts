/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the dump style h5md
 * (src/H5MD/dump_h5md.cpp, with lib/h5md/src/ch5md.c). This registry module is the lazily loaded helper of the core
 * dump command: its HOOKS entry 'h5md' is stored under the command word, and the core Dump reads it with gplHook('h5md')
 * (lazy group gpl_dump_h5md in src/engine/lazyStyles.ts; the names listed there must match these keys).
 */
import { createDumpH5md } from '../h5md/dump_h5md';

/** Helpers the core reads by command word (styles.ts loadStyleGroup). */
export const HOOKS: Record<string, unknown> = {
  h5md: createDumpH5md,
};
