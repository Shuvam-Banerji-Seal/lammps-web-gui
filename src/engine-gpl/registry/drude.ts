/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the DRUDE package (src/DRUDE/, commit
 * b332f28), GPL-2.0 (src/engine-gpl/LICENSE). Authors of the ported files are named in their headers (Alain Dequidt,
 * Julien Devemy, Agilio Padua, Kateryna Goloviznina, Zheng Gong, Mark Stevens, Aidan Thompson, Paul Crozier).
 *
 * DRUDE package styles (polarizable core-Drude oscillators) registered for the core through src/engine/lazyStyles.ts
 * (group gpl_drude).
 */
import type { FixFactory, ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import { FixDrude } from '../drude/fix_drude';
import { FixDrudeTransform } from '../drude/fix_drude_transform';
import { FixLangevinDrude } from '../drude/fix_langevin_drude';
import { ComputeTempDrude } from '../drude/compute_temp_drude';
import { PairThole } from '../drude/pair_thole';
import { PairCoulTT } from '../drude/pair_coul_tt';
import { FixTGNVTDrude } from '../drude/fix_tgnvt_drude';
import { FixTGNPTDrude } from '../drude/fix_tgnpt_drude';
import { PairLJCutTholeLong } from '../drude/pair_lj_cut_thole_long';

export const FIXES: Record<string, FixFactory> = {
  drude: (sys, id, group, args) => new FixDrude(sys, id, group, args),
  'drude/transform/direct': (sys, id, group, args) => new FixDrudeTransform(sys, id, group, args, false),
  'drude/transform/inverse': (sys, id, group, args) => new FixDrudeTransform(sys, id, group, args, true),
  'langevin/drude': (sys, id, group, args) => new FixLangevinDrude(sys, id, group, args),
  'tgnvt/drude': (sys, id, group, args) => new FixTGNVTDrude(sys, id, group, args),
  'tgnpt/drude': (sys, id, group, args) => new FixTGNPTDrude(sys, id, group, args),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  'temp/drude': (sys, id, group, args) => new ComputeTempDrude(sys, id, group, args),
};

export const PAIRS: Record<string, () => Pair> = {
  thole: () => new PairThole(),
  'coul/tt': () => new PairCoulTT(),
  'lj/cut/thole/long': () => new PairLJCutTholeLong(),
};
