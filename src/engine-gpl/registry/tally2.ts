/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: the style registrations of src/TALLY (compute heat/flux/tally, compute heat/flux/virial/tally) at commit
 * b332f28. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Loaded through src/engine/lazyStyles.ts (group gpl_tally2); the core does not register these names.
 */
import type { ComputeFactory } from '../../engine/styles';
import { ComputeHeatFluxTally } from '../tally2/compute_heat_flux_tally';
import { ComputeHeatFluxVirialTally } from '../tally2/compute_heat_flux_virial_tally';

export const COMPUTES: Record<string, ComputeFactory> = {
  'heat/flux/tally': (sys, id, group, args) => new ComputeHeatFluxTally(sys, id, group, args),
  'heat/flux/virial/tally': (sys, id, group, args) => new ComputeHeatFluxVirialTally(sys, id, group, args),
};
