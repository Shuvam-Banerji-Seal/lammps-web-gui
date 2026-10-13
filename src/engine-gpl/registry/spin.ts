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
import { minSpin } from '../spin/min_spin';
import { minSpinCG } from '../spin/min_spin_cg';
import { FixLangevinSpinT } from '../spin/fix_langevin_spin';

// Not ported (refused by name, see src/engine/refusedStyles.ts): the long-range dipole pair style and its kspace
// solvers (pair spin/dipole/long with ewald/dipole/spin and pppm/dipole/spin).
const DIPOLE_LONG = 'pair_style spin/dipole/long is not implemented in the browser engine (the dipolar kspace styles ewald/dipole/spin and pppm/dipole/spin are not ported)';

export const PAIRS: Record<string, () => Pair> = {
  'spin/dipole/cut': () => new PairSpinDipoleCut(),
  'spin/neel': () => new PairSpinNeel(),
  'spin/dipole/long': () => { throw new StyleError(DIPOLE_LONG); },
};

export const KSPACES: Record<string, () => KSpace> = {
  'ewald/dipole/spin': () => { throw new StyleError('kspace_style ewald/dipole/spin is not implemented in the browser engine (the dipolar kspace styles are not ported; see pair spin/dipole/long)'); },
  'pppm/dipole/spin': () => { throw new StyleError('kspace_style pppm/dipole/spin is not implemented in the browser engine (the dipolar kspace styles are not ported; see pair spin/dipole/long)'); },
};

export const FIXES: Record<string, FixFactory> = {
  'langevin/spin': (s, i, g, a) => new FixLangevinSpinT(s, i, g, a),
};

export const MINIMIZERS: Record<string, MinimizerFn> = {
  spin: minSpin,
  'spin/cg': minSpinCG,
  // min_spin_lbfgs.cpp is not ported (refused by name, see src/engine/refusedStyles.ts)
  'spin/lbfgs': () => { throw new StyleError("min_style 'spin/lbfgs' is not supported by the browser engine (the L-BFGS spin minimizer is not ported); supported: spin, spin/cg"); },
};
