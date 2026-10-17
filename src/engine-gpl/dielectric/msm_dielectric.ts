/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/msm_dielectric.cpp and msm_dielectric.h at commit b332f28 (Trung Nguyen and Monica Olvera de
 * la Cruz, Northwestern University, and the DIELECTRIC package authors), on the MSM solver of ../msm/msm.ts
 * (src/KSPACE/msm.cpp). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * kspace_style msm/dielectric: MSM for atom style dielectric. As natively, the grid takes the charges q (MSM::make_rho
 * and qsum_qsq are not overridden), the force on atom i is qqrd2e q_scaled_i E, the per-atom self energy uses the
 * scaled charges, and the field of each owned atom, efield = scale epsilon_i E, is kept for the polarization fixes.
 */
import { StyleError } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { KSpaceMSM } from '../msm/msm';
import { dielArrays } from './common';
import { isDielectricStyle } from '../../engine/atoms';

export class KSpaceMSMDielectric extends KSpaceMSM {
  override readonly name: string = 'msm/dielectric';
  /** Electric field of the grid at each owned atom, LAMMPS efield: scale * epsilon_i * E (3 per atom). */
  efield = new Float64Array(0);
  private eps: Float64Array = new Float64Array(0);

  constructor() {
    super();
    this.peratomName = 'msm/dielectric';
  }

  protected override checkAtoms(s: SimState): void {
    if (!isDielectricStyle(s.atomStyle)) throw new StyleError('msm/dielectric requires atom style dielectric');
  }

  protected override fieldCharges(s: SimState): Float64Array {
    const d = dielArrays(s, 'kspace_style msm/dielectric');
    this.eps = d.epsilon;
    if (this.efield.length < 3 * s.n) this.efield = new Float64Array(3 * s.n);
    return d.qs;
  }

  protected override fieldAtom(i: number, ex: number, ey: number, ez: number): void {
    // efactor = scale * eps[i], with scale = 1
    const efactor = this.eps[i];
    this.efield[3 * i] = efactor * ex;
    this.efield[3 * i + 1] = efactor * ey;
    this.efield[3 * i + 2] = efactor * ez;
  }
}
