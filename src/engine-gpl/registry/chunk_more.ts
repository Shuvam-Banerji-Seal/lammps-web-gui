/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the style registrations of
 * src/compute_vcm_chunk.cpp, compute_torque_chunk.cpp, compute_vacf_chunk.cpp, compute_inertia.cpp, compute_slice.cpp,
 * compute_global_atom.cpp, src/EXTRA-FIX/fix_momentum_chunk.cpp, src/fix_spring_chunk.cpp and src/fix_store_force.cpp at
 * commit b332f28. Copyright (2003) Sandia Corporation (LAMMPS development team). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). The ported classes live in src/engine-gpl/chunk/.
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_chunk_more).
 */
import type { ComputeFactory, FixFactory } from '../../engine/styles';
import { ComputeVCMChunk } from '../chunk/compute_vcm_chunk';
import { ComputeTorqueChunk } from '../chunk/compute_torque_chunk';
import { ComputeVACFChunk } from '../chunk/compute_vacf_chunk';
import { ComputeInertia } from '../chunk/compute_inertia';
import { ComputeSlice } from '../chunk/compute_slice';
import { ComputeGlobalAtom } from '../chunk/compute_global_atom';
import { FixMomentumChunk } from '../chunk/fix_momentum_chunk';
import { FixSpringChunk } from '../chunk/fix_spring_chunk';
import { FixStoreForce } from '../chunk/fix_store_force';

export const COMPUTES: Record<string, ComputeFactory> = {
  'vcm/chunk': (sys, id, group, args) => new ComputeVCMChunk(sys, id, group, args),
  'torque/chunk': (sys, id, group, args) => new ComputeTorqueChunk(sys, id, group, args),
  'vacf/chunk': (sys, id, group, args) => new ComputeVACFChunk(sys, id, group, args),
  inertia: (sys, id, group, args) => new ComputeInertia(sys, id, group, args),
  slice: (sys, id, group, args) => new ComputeSlice(sys, id, group, args),
  'global/atom': (sys, id, group, args) => new ComputeGlobalAtom(sys, id, group, args),
};

export const FIXES: Record<string, FixFactory> = {
  'momentum/chunk': (sys, id, group, args) => new FixMomentumChunk(sys, id, group, args),
  'spring/chunk': (sys, id, group, args) => new FixSpringChunk(sys, id, group, args),
  'store/force': (sys, id, group, args) => new FixStoreForce(sys, id, group, args),
};
