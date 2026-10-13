/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the SPH package's style registry
 * (src/SPH: pair sph/rhosum, sph/idealgas, sph/taitwater, sph/taitwater/morris, sph/heatconduction; fix sph,
 * sph/stationary; compute sph/rho/atom, sph/e/atom). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Package author
 * (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Registered for the core through src/engine/lazyStyles.ts (the names of this module are listed there).
 */
import type { Pair } from '../../engine/force/types';
import type { Fix } from '../../engine/fix/fix';
import type { Compute } from '../../engine/compute/compute';
import type { System } from '../../engine/system';
import { PairSPHRhoSum } from '../sph/pair_sph_rhosum';
import { PairSPHIdealGas } from '../sph/pair_sph_idealgas';
import { PairSPHTaitwater } from '../sph/pair_sph_taitwater';
import { PairSPHTaitwaterMorris } from '../sph/pair_sph_taitwater_morris';
import { PairSPHHeatConduction } from '../sph/pair_sph_heatconduction';
import { FixSPH } from '../sph/fix_sph';
import { FixSPHStationary } from '../sph/fix_sph_stationary';
import { ComputeSPHRhoAtom, ComputeSPHEAtom } from '../sph/compute_sph_atom';

/** pair_style names. */
export const PAIRS: Record<string, () => Pair> = {
  'sph/rhosum': () => new PairSPHRhoSum(),
  'sph/idealgas': () => new PairSPHIdealGas(),
  'sph/taitwater': () => new PairSPHTaitwater(),
  'sph/taitwater/morris': () => new PairSPHTaitwaterMorris(),
  'sph/heatconduction': () => new PairSPHHeatConduction(),
};

/** fix styles. */
export const FIXES: Record<string, (sys: System, id: string, group: string, args: string[]) => Fix> = {
  'sph': (sys, id, group, args) => new FixSPH(sys, id, group, args),
  'sph/stationary': (sys, id, group, args) => new FixSPHStationary(sys, id, group, args),
};

/** compute styles. */
export const COMPUTES: Record<string, (sys: System, id: string, group: string, args: string[]) => Compute> = {
  'sph/rho/atom': (sys, id, group, args) => new ComputeSPHRhoAtom(sys, id, group, args),
  'sph/e/atom': (sys, id, group, args) => new ComputeSPHEAtom(sys, id, group, args),
};
