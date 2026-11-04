/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair gran/hooke, gran/hooke/history
 * and gran/hertz/history as a GPU kernel (src/GRANULAR/pair_gran_hooke.cpp, pair_gran_hooke_history.cpp and
 * pair_gran_hertz_history.cpp at b332f28; ../granular/gpu_gran.ts), as the hook the core's accelerated path calls (lazy
 * group gpl_gran_gpu, HOOKS words the three style names, loaded by Session.execute with the pair_style command; the
 * words' single() helpers of gpl_gran_single join it in one object, gplHooks.ts setGplHook).
 */
import type { System } from '../../engine/system';
import type { PairGranNative } from '../granular/pair_gran_native';
import { gpuGranKernel } from '../granular/gpu_gran';

const HELPER = { gpuKernel: (pair: unknown, sys: System) => gpuGranKernel(pair as PairGranNative, sys) };

export const HOOKS = { 'gran/hooke': HELPER, 'gran/hooke/history': HELPER, 'gran/hertz/history': HELPER };
