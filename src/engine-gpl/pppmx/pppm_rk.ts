/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pppm_rk.cpp and
 * src/KSPACE/pppm_rk.h (Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL); author of the R/K partition
 * scheme: Brian Dandurand, Queen's University Belfast). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/rk: PPPM whose MPI communicator is split into R-process and K-process partitions (command-line
 * -partition, run_style verlet/split/rk; kspace_style.html: "the use of *pppm/rk* requires the use of the command line
 * `-partition` option"). The browser engine runs one partition, so the style is refused the way native refuses it
 * without partitions: PPPM_RK::setupRKBlock raises "pppm/rk requires 2 partitions" (universe_all) in PPPM_RK::init.
 * The partition scheme has no single-partition meaning, so no solve is written here.
 *
 * Derives from KSpaceBase, not from KSpacePPPM, for the reason given in pppm_stagger.ts: kspace_modify diff ad would
 * otherwise replace this style by the core PPPM solver before init() can refuse it.
 */
import { KSpaceBase } from '../../engine/force/kspace/ewald';
import { StyleError, type KSpaceCompute, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';

const NEEDS_PARTITIONS = 'pppm/rk requires 2 partitions (kspace_style pppm/rk needs the -partition command-line option and run_style verlet/split/rk; the browser engine runs one partition)';

export class KSpacePPPMRK extends KSpaceBase {
  override readonly name: string = 'pppm/rk';

  init(_s: SimState, _geom: Geometry, _cutCoul: number, _qqrd2e: number, _ctx: StyleContext): void {
    throw new StyleError(NEEDS_PARTITIONS);
  }

  compute(_kc: KSpaceCompute): void {
    throw new StyleError(NEEDS_PARTITIONS);
  }
}
