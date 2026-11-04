/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair eff/cut as a GPU kernel
 * (src/EFF/pair_eff_cut.cpp and pair_eff_inline.h at b332f28; ../eff/gpu_eff.ts), as the hook the core's accelerated
 * path calls (lazy group gpl_eff_gpu, HOOKS word eff/cut, loaded by Session.execute with the pair_style command).
 */
import type { System } from '../../engine/system';
import type { PairEffCut } from '../eff/pair_eff_cut';
import { gpuEffKernel } from '../eff/gpu_eff';

export const HOOKS = { 'eff/cut': { gpuKernel: (pair: unknown, sys: System) => gpuEffKernel(pair as PairEffCut, sys) } };
