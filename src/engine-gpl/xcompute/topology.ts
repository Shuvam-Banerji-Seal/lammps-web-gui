/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the neighbour lists of the EXTRA-COMPUTE
 * styles in src/EXTRA-COMPUTE (compute_ackland_atom.cpp, compute_ave_sphere_atom.cpp, compute_composition_atom.cpp,
 * compute_efield_wolf_atom.cpp) request an occasional full list that, in native LAMMPS, leaves out the special pairs
 * with a zero weight (special_bonds). The engine's occasional list (NeighborList.occasionalList) has no special
 * handling, so these styles refuse a system with bonded topology instead of counting the 1-2, 1-3 and 1-4 neighbours.
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

/** Throws when the system has bonded topology (its special neighbours are not handled by the occasional list). */
export const topologyRefusal = (sys: System, name: string): void => {
  const t = sys.state.topo;
  if (t.bonds.n > 0 || t.angles.n > 0 || t.dihedrals.n > 0 || t.impropers.n > 0) {
    throw new StyleError(`compute ${name} with bonded topology (special neighbours) is not supported by the engine`);
  }
};
