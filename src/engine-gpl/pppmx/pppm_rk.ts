/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pppm_rk.cpp and
 * src/KSPACE/pppm_rk.h (Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL); author of the R/K partition
 * scheme: Brian Dandurand, Queen's University Belfast). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/rk: PPPM whose work is split between the Rspace partition (charge assignment and force
 * interpolation) and the Kspace partition (the FFTs and the Poisson solve), with run_style verlet/split/rk on two
 * partitions (kspace_style.html: "the use of *pppm/rk* requires the use of the command line `-partition` option").
 * PPPM_RK computes PPPM's charge density, Poisson solve, field interpolation and energy (PPPM::make_rho, poisson,
 * fieldforce, the same self-energy and background terms), so the Rspace partition's forces and energies are those of
 * kspace_style pppm. Measured with native LAMMPS: a 64-ion pppm/rk run with run_style verlet/split/rk on two
 * partitions prints the same thermo as kspace_style pppm with run_style verlet, to all 15 printed digits. The one
 * partition here computes the whole solve: this class is the engine's PPPM with native's rk_flag, which the run
 * checks (src/engine/commands/run.ts): pppm/rk needs run_style verlet/split/rk and two partitions.
 * The constructor's group_group_enable = 0 is groupGroupEnable = false.
 */
import { KSpacePPPM } from '../../engine/force/kspace/pppm';

export class KSpacePPPMRK extends KSpacePPPM {
  override readonly name: string = 'pppm/rk';
  override rkFlag = true;
  override groupGroupEnable = false;
}
