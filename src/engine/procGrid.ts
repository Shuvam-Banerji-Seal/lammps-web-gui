import { StyleError } from './force/types';

/**
 * The processors command's grid (processors.html: "Px,Py,Pz = # of processors in each dimension of 3d grid
 * overlaying the simulation domain", "* = let LAMMPS choose"), kept for the commands that create the box. user holds
 * Px Py Pz with 0 for *, core the core grid of grid twolevel (0 for *).
 */
export interface ProcGrid {
  user: [number, number, number];
  grid: 'onelevel' | 'twolevel' | 'numa' | 'custom';
  ncores: number;
  core: [number, number, number];
}

/**
 * Comm::set_proc_grid on the one process the engine runs. create_box, read_data and read_restart call it once the box
 * is known. As native LAMMPS does (src/comm.cpp Comm::set_proc_grid and src/procmap.cpp ProcMap::onelevel_grid and
 * twolevel_grid at b332f28): the factorizations of the process count into a 3d grid are culled by the user's counts
 * (cull_user drops those that differ from a count given as a number), and none left is an error. One process factors
 * only as 1 x 1 x 1, so any count above 1 fails:
 * - onelevel: "Could not create 3d grid of processors";
 * - twolevel: "Processors twogrid requires proc count be a multiple of core count" when ncores does not divide 1,
 *   then "Could not create twolevel 3d grid of processors" when a node or core count is above 1.
 * The numa and custom grids depend on the machine or a file and are not checked.
 * Measured with native LAMMPS: examples/GRAPHICS/in.mixer-drum (processors * * 2) on one process stops at create_box
 * with the error Could not create 3d grid of processors, after printing the Created orthogonal box line.
 */
export const setProcGrid = (g: ProcGrid | null): void => {
  if (!g) return;
  if (g.grid === 'onelevel' && g.user.some((v) => v > 1)) throw new StyleError('Could not create 3d grid of processors');
  if (g.grid === 'twolevel') {
    if (1 % g.ncores !== 0) throw new StyleError('Processors twogrid requires proc count be a multiple of core count');
    if (g.core.some((v) => v > 1) || g.user.some((v) => v > 1)) throw new StyleError('Could not create twolevel 3d grid of processors');
  }
};
