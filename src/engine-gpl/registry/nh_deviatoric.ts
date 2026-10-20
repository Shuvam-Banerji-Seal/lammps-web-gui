/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/fix_nh.cpp (compute_sigma, compute_deviatoric, compute_strain_energy) and src/domain.cpp at commit
 * b332f28; the routines are in ../fix/nh_deviatoric.ts. Copyright Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The lazy HOOKS entries x, y, z, xy, yz, xz (lazyStyles.ts group gpl_nh_deviatoric): the keywords of fix nvt, npt,
 * nph and their variants that set native's deviatoric_flag. The core FixNH reads the helpers through gplHook('x')
 * (src/engine/gplHooks.ts); loading the group on any of the words stores them under all six.
 */
import type { NHDeviatoricHook } from '../../engine/fix/nh';
import { boxHInv, computeDeviatoric, computeSigma, computeStrainEnergy } from '../fix/nh_deviatoric';

const hook: NHDeviatoricHook = { boxHInv, computeSigma, computeDeviatoric, computeStrainEnergy };

export const HOOKS: Record<string, NHDeviatoricHook> = { x: hook, y: hook, z: hook, xy: hook, yz: hook, xz: hook };
