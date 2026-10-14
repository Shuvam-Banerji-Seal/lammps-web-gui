/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/fix_adapt_fep.cpp, src/FEP/compute_fep.cpp at commit b332f28 (Agilio Padua, ENS de Lyon & CNRS, and
 * the LAMMPS developers). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * FEP package (free-energy perturbation) styles ported from LAMMPS; registered for the core through
 * src/engine/lazyStyles.ts (group gpl_fep).
 */
import type { FixFactory, ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import { FixAdaptFEP } from '../fep/fix_adapt_fep';
import { ComputeFEP } from '../fep/compute_fep';
import { ComputeFEPTA } from '../fep/compute_fep_ta';
import { PairMorseSoft } from '../fep/pair_morse_soft';
import { PairCoulCutSoftGapsys, PairLJCutSoftGapsys } from '../fep/pair_soft_gapsys';
import { PairLJCutTIP4PSoft, PairTIP4PSoft } from '../fep/pair_tip4p_soft';

/** Style name -> factory (the names listed in lazyStyles.ts must match these keys). */
export const FIXES: Record<string, FixFactory> = {
  'adapt/fep': (sys, id, group, args) => new FixAdaptFEP(sys, id, group, args),
};

export const PAIRS: Record<string, () => Pair> = {
  'morse/soft': () => new PairMorseSoft(),
  'lj/cut/soft/gapsys': () => new PairLJCutSoftGapsys(),
  'coul/cut/soft/gapsys': () => new PairCoulCutSoftGapsys(),
  'lj/cut/tip4p/long/soft': () => new PairLJCutTIP4PSoft(),
  'tip4p/long/soft': () => new PairTIP4PSoft(),
};

export const COMPUTES: Record<string, ComputeFactory> = {
  fep: (sys, id, group, args) => new ComputeFEP(sys, id, group, args),
  'fep/ta': (sys, id, group, args) => new ComputeFEPTA(sys, id, group, args),
};
