/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the MACHDYN package's style registrations (src/MACHDYN/pair_smd_tlsph.cpp, fix_smd_setvel.cpp,
 * fix_smd_integrate_tlsph.cpp, fix_smd_adjust_dt.cpp, compute_smd_tlsph_*.cpp) at commit b332f28. MACHDYN author:
 * Georg C. Ganzenmueller, Fraunhofer EMI. Copyright Sandia Corporation (GNU General Public License); this file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_machdyn; the names listed there must match these keys).
 */
import type { FixFactory, ComputeFactory } from '../../engine/styles';
import type { Pair } from '../../engine/force/types';
import { PairSMDTlsph } from '../machdyn/pair_smd_tlsph';
import { FixSMDAdjustDt, FixSMDIntegrateTlsph, FixSMDSetVel } from '../machdyn/fix_smd';
import {
  ComputeSMDTLSPHNumNeighs, ComputeSMDTLSPHShape, ComputeSMDTLSPHStrain, ComputeSMDTLSPHStrainRate, ComputeSMDTLSPHStress,
  makeSmdDamage, makeSmdPlasticStrain, makeSmdPlasticStrainRate, makeSmdTlsphDt,
} from '../machdyn/compute_smd';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'smd/tlsph': () => new PairSMDTlsph(),
};

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'smd/setvel': (sys, id, group, args) => new FixSMDSetVel(sys, id, group, args),
  'smd/integrate_tlsph': (sys, id, group, args) => new FixSMDIntegrateTlsph(sys, id, group, args),
  'smd/adjust_dt': (sys, id, group, args) => new FixSMDAdjustDt(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, ComputeFactory> = {
  'smd/tlsph/stress': (sys, id, group, args) => new ComputeSMDTLSPHStress(sys, id, group, args),
  'smd/tlsph/strain': (sys, id, group, args) => new ComputeSMDTLSPHStrain(sys, id, group, args),
  'smd/tlsph/num/neighs': (sys, id, group, args) => new ComputeSMDTLSPHNumNeighs(sys, id, group, args),
  'smd/tlsph/strain/rate': (sys, id, group, args) => new ComputeSMDTLSPHStrainRate(sys, id, group, args),
  'smd/tlsph/shape': (sys, id, group, args) => new ComputeSMDTLSPHShape(sys, id, group, args),
  'smd/tlsph/dt': makeSmdTlsphDt,
  'smd/plastic/strain': makeSmdPlasticStrain,
  'smd/plastic/strain/rate': makeSmdPlasticStrainRate,
  'smd/damage': makeSmdDamage,
};
