/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the dump style vtk (src/EXTRA-DUMP/
 * dump_vtk.cpp, with the writer of src/vtk_writer.cpp). This registry module is the lazily loaded helper of the core dump
 * command: its HOOKS entry 'vtk' is stored under the command word, and the core Dump reads it with gplHook('vtk') (lazy
 * group gpl_dump_vtk in src/engine/lazyStyles.ts; the name listed there must match this key).
 */
import { createDumpVtk } from '../dump/dump_vtk';

/** Helpers the core reads by command word (styles.ts loadStyleGroup). */
export const HOOKS: Record<string, unknown> = {
  vtk: createDumpVtk,
};
