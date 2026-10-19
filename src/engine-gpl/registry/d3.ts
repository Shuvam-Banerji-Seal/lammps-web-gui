/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the pair style dispersion/d3
 * (src/EXTRA-PAIR/pair_dispersion_d3.cpp, pair_dispersion_d3.h and d3_parameters.h; contributing authors Sonia Salomoni
 * and Arthur France-Lanord). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_d3, whose load() calls loadD3: the parameter
 * tables are split over several data modules, imported before the style can be made; the names listed there must
 * match these keys).
 */
import type { Pair } from '../../engine/force/types';
import { PairDispersionD3 } from '../d3/pair_dispersion_d3';
import { loadD3Tables } from '../d3/d3_tables';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'dispersion/d3': () => new PairDispersionD3(),
};

/** The registry, once the parameter tables are loaded. */
export const loadD3 = async (): Promise<{ PAIRS: Record<string, () => Pair> }> => {
  await loadD3Tables();
  return { PAIRS };
};
