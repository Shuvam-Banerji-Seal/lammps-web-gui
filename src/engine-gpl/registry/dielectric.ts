/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the DIELECTRIC package's style registry (src/DIELECTRIC/pair_lj_cut_coul_cut_dielectric.h, msm_dielectric.h,
 * pair_lj_cut_coul_msm_dielectric.h, fix_polarize_bem_gmres.h,
 * fix_polarize_bem_icc.h and the other headers named in the style sources) at commit b332f28. DIELECTRIC package authors:
 * Trung Nguyen and Monica Olvera de la Cruz (Northwestern University), and Honghao Li (Northwestern).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_dielectric; the names listed there must match these keys).
 */
import type { FixFactory } from '../../engine/styles';
import type { KSpace, Pair } from '../../engine/force/types';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { PairLJCutCoulCutDielectric } from '../dielectric/pair_lj_cut_coul_cut_dielectric';
import { PairLJCutCoulLongDielectric } from '../dielectric/pair_lj_cut_coul_long_dielectric';
import { KSpacePPPMDielectric } from '../dielectric/pppm_dielectric';
import { KSpaceMSMDielectric } from '../dielectric/msm_dielectric';
import { PairLJCutCoulMSMDielectric } from '../dielectric/pair_lj_cut_coul_msm_dielectric';
import { FixPolarizeBEMGMRES } from '../dielectric/fix_polarize_bem_gmres';
import { FixPolarizeBEMICC } from '../dielectric/fix_polarize_bem_icc';
import { FixPolarizeFunctional } from '../dielectric/fix_polarize_functional';
import { ComputeEfieldAtom } from '../dielectric/compute_efield_atom';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'lj/cut/coul/cut/dielectric': () => new PairLJCutCoulCutDielectric(),
  'lj/cut/coul/long/dielectric': () => new PairLJCutCoulLongDielectric(),
  'lj/cut/coul/msm/dielectric': () => new PairLJCutCoulMSMDielectric(),
};

/** kspace_style names. */
export const KSPACES: Record<string, () => KSpace> = {
  'pppm/dielectric': () => new KSpacePPPMDielectric(),
  'msm/dielectric': () => new KSpaceMSMDielectric(),
};

/** compute styles. */
export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  'efield/atom': (sys, id, group, args) => new ComputeEfieldAtom(sys, id, group, args),
};

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'polarize/bem/gmres': (sys, id, group, args) => new FixPolarizeBEMGMRES(sys, id, group, args),
  'polarize/bem/icc': (sys, id, group, args) => new FixPolarizeBEMICC(sys, id, group, args),
  'polarize/functional': (sys, id, group, args) => new FixPolarizeFunctional(sys, id, group, args),
};
