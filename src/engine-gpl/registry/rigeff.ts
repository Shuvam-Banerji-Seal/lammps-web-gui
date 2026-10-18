/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: the style registry of the rigid-body
 * computes (src/RIGID/compute_erotate_rigid.cpp, compute_ke_rigid.cpp, compute_rigid_local.cpp), the EFF fixes and
 * compute (src/EFF/fix_langevin_eff.cpp, fix_nvt_sllod_eff.cpp, compute_temp_deform_eff.cpp), the DPD-REACT table
 * EOS (src/DPD-REACT/fix_eos_table.cpp) and the SMD computes and surface fix (src/MACHDYN/). Original authors (from
 * the sources): Trung Dac Nguyen (ORNL), Andres Jaramillo-Botero (Caltech), Jim Larentzos (ARL), Georg C. Ganzenmueller
 * and Frank Zhang (Fraunhofer EMI). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (group gpl_rigeff; the names are listed there).
 */
import type { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { ComputeERotateRigid } from '../rigeff/compute_erotate_rigid';
import { ComputeKERigid } from '../rigeff/compute_ke_rigid';
import { ComputeRigidLocal } from '../rigeff/compute_rigid_local';
import { ComputeTempDeformEff } from '../rigeff/compute_temp_deform_eff';
import { FixLangevinEff } from '../rigeff/fix_langevin_eff';
import { FixNVTSllodEff } from '../rigeff/fix_nvt_sllod_eff';
import { FixEOStable } from '../rigeff/fix_eos_table';
import { FixSMDMoveTriSurf } from '../rigeff/fix_smd_move_tri_surf';
import { ComputeSMDTriangleVertices, ComputeSMDTLSPHDefgrad, ComputeSMDHourglassError } from '../rigeff/compute_smd';

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'langevin/eff': (sys, id, group, args) => new FixLangevinEff(sys, id, group, args),
  'nvt/sllod/eff': (sys, id, group, args) => new FixNVTSllodEff(sys, id, group, args),
  'eos/table': (sys, id, group, args) => new FixEOStable(sys, id, group, args),
  'smd/move_tri_surf': (sys, id, group, args) => new FixSMDMoveTriSurf(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  'erotate/rigid': (sys, id, group, args) => new ComputeERotateRigid(sys, id, group, args),
  'ke/rigid': (sys, id, group, args) => new ComputeKERigid(sys, id, group, args),
  'rigid/local': (sys, id, group, args) => new ComputeRigidLocal(sys, id, group, args),
  'temp/deform/eff': (sys, id, group, args) => new ComputeTempDeformEff(sys, id, group, args),
  'smd/triangle/vertices': (sys, id, group, args) => new ComputeSMDTriangleVertices(sys, id, group, args),
  'smd/tlsph/defgrad': (sys, id, group, args) => new ComputeSMDTLSPHDefgrad(sys, id, group, args),
  'smd/hourglass/error': (sys, id, group, args) => new ComputeSMDHourglassError(sys, id, group, args),
};
