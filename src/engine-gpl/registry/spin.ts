/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the SPIN package styles of src/SPIN/
 * that the core does not provide (see src/engine-gpl/spin/). LAMMPS is GPL-2.0; this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Loaded lazily through src/engine/lazyStyles.ts (group gpl_spin).
 */
import { StyleError, type Pair } from '../../engine/force/types';
import type { MinimizerFn } from '../../engine/run/minimizers';
import type { FixFactory } from '../../engine/styles';
import type { KSpace } from '../../engine/force/types';
import { PairSpinDipoleCut } from '../spin/pair_spin_dipole_cut';
import { PairSpinNeel } from '../spin/pair_spin_neel';
import { PairSpinDipoleLong } from '../spin/pair_spin_dipole_long';
import { KSpaceEwaldDipoleSpin } from '../spin/ewald_dipole_spin';
import { minSpin } from '../spin/min_spin';
import { minSpinCG } from '../spin/min_spin_cg';
import { minSpinLBFGS } from '../spin/min_spin_lbfgs';
import { FixLangevinSpinT } from '../spin/fix_langevin_spin';

// Not ported (refused by name, see src/engine/refusedStyles.ts): kspace_style pppm/dipole/spin (pppm_dipole_spin.cpp)
const PPPM_DIPOLE_SPIN = 'kspace_style pppm/dipole/spin is not implemented in the browser engine (the PPPM dipolar solver is not ported; use ewald/dipole/spin)';

export const PAIRS: Record<string, () => Pair> = {
  'spin/dipole/cut': () => new PairSpinDipoleCut(),
  'spin/neel': () => new PairSpinNeel(),
  'spin/dipole/long': () => new PairSpinDipoleLong(),
};

export const KSPACES: Record<string, () => KSpace> = {
  'ewald/dipole/spin': () => new KSpaceEwaldDipoleSpin(),
  'pppm/dipole/spin': () => { throw new StyleError(PPPM_DIPOLE_SPIN); },
};

export const FIXES: Record<string, FixFactory> = {
  'langevin/spin': (s, i, g, a) => new FixLangevinSpinT(s, i, g, a),
};

export const MINIMIZERS: Record<string, MinimizerFn> = {
  spin: minSpin,
  'spin/cg': minSpinCG,
  'spin/lbfgs': minSpinLBFGS,
};
