/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair granular as a GPU kernel
 * (src/GRANULAR/pair_granular.cpp, granular_model.cpp and the gran_sub_mod_*.cpp normal, damping and tangential models at
 * b332f28; ../granular/gpu_granular.ts), as the hook the core's accelerated path calls (lazy group gpl_granular_gpu, HOOKS
 * word granular, loaded by Session.execute with the pair_style command; the word's pair/local helper of gpl_pair_local
 * joins it in one object, gplHooks.ts setGplHook).
 */
import type { System } from '../../engine/system';
import { gpuGranularKernel, type GranularPair } from '../granular/gpu_granular';

export const HOOKS = { granular: { gpuKernel: (pair: unknown, sys: System) => gpuGranularKernel(pair as GranularPair, sys) } };
