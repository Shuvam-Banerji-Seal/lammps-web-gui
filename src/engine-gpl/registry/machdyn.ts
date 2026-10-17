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
import { PairSMDHertz } from '../machdyn/pair_smd_hertz';
import { PairSMDUlsph } from '../machdyn/pair_smd_ulsph';
import { PairSMDTriSurface } from '../machdyn/pair_smd_tri_surface';
import { FixSMDAdjustDt, FixSMDIntegrateTlsph, FixSMDSetVel } from '../machdyn/fix_smd';
import { FixSMDIntegrateUlsph } from '../machdyn/fix_smd_integrate_ulsph';
import { FixSMDWallSurface } from '../machdyn/fix_smd_wall_surface';
import {
  ComputeSMDTLSPHNumNeighs, ComputeSMDTLSPHShape, ComputeSMDTLSPHStrain, ComputeSMDTLSPHStrainRate, ComputeSMDTLSPHStress,
  makeSmdContactRadius, makeSmdDamage, makeSmdInternalEnergy, makeSmdPlasticStrain, makeSmdPlasticStrainRate, makeSmdRho, makeSmdTlsphDt,
} from '../machdyn/compute_smd';
import {
  ComputeSMDULSPHNumNeighs, ComputeSMDULSPHStrain, ComputeSMDULSPHStrainRate, ComputeSMDULSPHStress, ComputeSMDVol, makeSmdUlsphEffm,
} from '../machdyn/compute_smd_ulsph';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'smd/tlsph': () => new PairSMDTlsph(),
  'smd/hertz': () => new PairSMDHertz(),
  'smd/ulsph': () => new PairSMDUlsph(),
  'smd/tri_surface': () => new PairSMDTriSurface(),
};

/** fix styles. */
export const FIXES: Record<string, FixFactory> = {
  'smd/setvel': (sys, id, group, args) => new FixSMDSetVel(sys, id, group, args),
  'smd/integrate_tlsph': (sys, id, group, args) => new FixSMDIntegrateTlsph(sys, id, group, args),
  'smd/integrate_ulsph': (sys, id, group, args) => new FixSMDIntegrateUlsph(sys, id, group, args),
  'smd/wall_surface': (sys, id, group, args) => new FixSMDWallSurface(sys, id, group, args),
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
  'smd/contact/radius': makeSmdContactRadius,
  'smd/internal/energy': makeSmdInternalEnergy,
  'smd/rho': makeSmdRho,
  'smd/ulsph/num/neighs': (sys, id, group, args) => new ComputeSMDULSPHNumNeighs(sys, id, group, args),
  'smd/ulsph/strain': (sys, id, group, args) => new ComputeSMDULSPHStrain(sys, id, group, args),
  'smd/ulsph/strain/rate': (sys, id, group, args) => new ComputeSMDULSPHStrainRate(sys, id, group, args),
  'smd/ulsph/stress': (sys, id, group, args) => new ComputeSMDULSPHStress(sys, id, group, args),
  'smd/ulsph/effm': makeSmdUlsphEffm,
  'smd/vol': (sys, id, group, args) => new ComputeSMDVol(sys, id, group, args),
};
