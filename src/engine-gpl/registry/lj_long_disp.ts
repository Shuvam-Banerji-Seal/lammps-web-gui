/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/pair.cpp (Pair::init_tables_disp, Pair::init_bitmap), see ../pair_disp/dispTable.ts. This file is GPL-2.0 too.
 *
 * The lazy HOOKS entry 'lj/long/coul/long' (lazyStyles.ts group gpl_lj_long_disp): the core pair lj/long/coul/long
 * (flag_lj long, pair_modify table/disp N > 0) calls it for its dispersion table, through gplHook.
 * The PAIRS entry 'lj/long/tip4p/long' is the TIP4P variant (src/KSPACE/pair_lj_long_tip4p_long.cpp, see
 * ../tip4pdisp/pair_lj_long_tip4p_long.ts); it shares this group because it needs the same dispersion table.
 */
import type { DispTableHook } from '../../engine/force/pair/lj_long';
import type { Pair } from '../../engine/force/types';
import { makeDispTable } from '../pair_disp/dispTable';
import { PairLJLongTIP4PLong } from '../tip4pdisp/pair_lj_long_tip4p_long';

export const HOOKS: Record<string, DispTableHook> = { 'lj/long/coul/long': makeDispTable };

export const PAIRS: Record<string, () => Pair> = {
  'lj/long/tip4p/long': () => new PairLJLongTIP4PLong(),
};
