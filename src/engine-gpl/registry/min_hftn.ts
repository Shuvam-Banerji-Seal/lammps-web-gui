/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/min_hftn.cpp and min_hftn.h
 * (min_style hftn, Todd Plantenga (SNL)). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU
 * General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Loaded lazily through
 * src/engine/lazyStyles.ts (group gpl_min_hftn).
 */
import type { MinimizerFn } from '../../engine/run/minimizers';
import { minHftn } from '../cmds/min_hftn';

export const MINIMIZERS: Record<string, MinimizerFn> = { hftn: minHftn };
