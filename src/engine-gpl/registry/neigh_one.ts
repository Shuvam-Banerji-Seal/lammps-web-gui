/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/nbin_standard.cpp, src/nbin.cpp, src/nstencil.cpp, src/nstencil_bin.cpp, src/npair_bin.cpp,
 * src/npair_nsq.cpp and src/my_page.h at commit b332f28 (see ../neighbor/nbin.ts). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License (the LAMMPS file headers). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The neigh_modify one check's count (lazy group gpl_neigh_one, HOOKS word neigh_one, loaded by Session.execute).
 */
import type { NeighOneHook } from '../../engine/neighbor';
import { nativeMaxCount } from '../neighbor/nbin';

export const HOOKS: Record<string, NeighOneHook> = { neigh_one: { maxCount: nativeMaxCount } };
