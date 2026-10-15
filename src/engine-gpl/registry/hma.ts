/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_hma.cpp, src/EXTRA-PAIR/pair_lj_smooth_linear.cpp and src/pair.cpp (hessian_twobody)
 * at commit b332f28 (see extra_compute/hma.ts). Copyright the LAMMPS development team (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entry 'hma' (lazyStyles.ts group gpl_hma): the core compute hma calls the cv phi_sum through it.
 */
export { HOOKS } from '../extra_compute/hma';
