/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the dump style cfg (src/dump_cfg.cpp).
 * This registry module is the lazily loaded helper of the core dump command: its HOOKS entry 'cfg' is stored under the
 * command word, and the core Dump reads it with gplHook('cfg') (lazy group gpl_dump_cfg in src/engine/lazyStyles.ts; the
 * name listed there must match this key).
 */
import { createDumpCfg } from '../dump/dump_cfg';

/** Helpers the core reads by command word (styles.ts loadStyleGroup). */
export const HOOKS: Record<string, unknown> = {
  cfg: createDumpCfg,
};
