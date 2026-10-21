/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the dump styles grid and grid/vtk
 * (src/dump_grid.cpp, src/dump_grid_vtk.cpp). This registry module is the lazily loaded helper of the core dump command:
 * its HOOKS entries 'grid' and 'grid/vtk' are stored under the command word, and the core Dump reads them with
 * gplHook('grid') and gplHook('grid/vtk') (lazy group gpl_dump_grid in src/engine/lazyStyles.ts; the names listed there
 * must match these keys).
 */
import { dumpGrid, dumpGridVtk } from '../dump/dump_grid';

/** Helpers the core reads by command word (styles.ts loadStyleGroup). */
export const HOOKS: Record<string, unknown> = {
  grid: dumpGrid,
  'grid/vtk': dumpGridVtk,
};
