/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the DPD-REACT package's pair dpd/fdt and
 * dpd/fdt/energy (src/DPD-REACT/pair_dpd_fdt.cpp, pair_dpd_fdt_energy.cpp), fix dpd/energy and fix eos/cv
 * (fix_dpd_energy.cpp, fix_eos_cv.cpp) and compute dpd (compute_dpd.cpp). Copyright Sandia Corporation (LAMMPS
 * development team, GNU General Public License). This file is GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Registered for the core through src/engine/lazyStyles.ts (the names of this module are listed there). The reactive
 * styles of the package (fix rx, exp6/rx, eos/table/rx) are in registry/dpdrx.ts; compute dpd/atom is here.
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { PairDPDfdt, PairDPDfdtEnergy } from '../dpdreact/pair_dpd_fdt';
import { FixDPDenergy, FixEOScv, ComputeDpd, ComputeDpdAtom } from '../dpdreact/fixes_compute';
import { FixShardlow } from '../dpdreact/fix_shardlow';

export const PAIRS: Record<string, () => Pair> = {
  'dpd/fdt': () => new PairDPDfdt(),
  'dpd/fdt/energy': () => new PairDPDfdtEnergy(),
};

export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'dpd/energy': (sys, id, group, args) => new FixDPDenergy(sys, id, group, args),
  'eos/cv': (sys, id, group, args) => new FixEOScv(sys, id, group, args),
  shardlow: (sys, id, group, args) => new FixShardlow(sys, id, group, args),
};

export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  dpd: (sys, id, group, args) => new ComputeDpd(sys, id, group, args),
  'dpd/atom': (sys, id, group, args) => new ComputeDpdAtom(sys, id, group, args),
};
