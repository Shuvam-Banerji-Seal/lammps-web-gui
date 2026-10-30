/*
 * The Bodies section of write_data, ported from LAMMPS (commit b332f28; src/BODY/body_*.cpp pack_data_body and
 * write_data_body, GPL-2.0, see src/engine-gpl/body/write_data_body.ts). Merged into styles.ts through the lazy
 * registry entry gpl_write_data_body in src/engine/lazyStyles.ts (kind HOOKS, word 'body').
 */
import type { BodyWriteHook } from '../../engine/output/data';
import { BODY_WRITE_HOOK } from '../body/write_data_body';

/** The HOOKS entry 'body' (the contract is BodyWriteHook, read by src/engine/output/data.ts writeData). */
export const HOOKS: Record<string, BodyWriteHook> = { body: BODY_WRITE_HOOK };
