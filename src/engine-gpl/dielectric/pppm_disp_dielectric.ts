/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/DIELECTRIC/pppm_disp_dielectric.cpp and pppm_disp_dielectric.h at commit b332f28 (Trung Nguyen, Northwestern
 * University, and the DIELECTRIC package authors), on top of src/KSPACE/pppm_disp.cpp at the same commit (Rolf Isele-Holder,
 * Paul Crozier, and the LAMMPS developers). Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * kspace_style pppm/disp/dielectric is pppm/disp (src/engine-gpl/pppm_disp/kspace.ts) with the dielectric Coulomb terms of
 * native PPPMDispDielectric::compute:
 *  - the Coulomb density of the force pass is made from the scaled charges q_scaled (make_rho_c with use_qscaled);
 *  - the force on atom i is qqrd2e * q_i * E, with the unscaled q_i, and the field E is that of the scaled density
 *    (fieldforce_c_ik);
 *  - efield[i] = scale * epsilon_i * E (scale is 1 here): the field the polarization fixes read;
 *  - the energy and the Coulomb virial are summed over two poisson passes, the scaled density and then ("switch to unscaled
 *    charges") the unscaled density, both added to energy_1 and virial_1 which native does not reset in between; the energy
 *    is 0.5 V (sum) - self terms, times qqrd2e (no 1 / epsilon_ave: compute_ave_epsilon is computed but unused natively);
 *  - the dispersion part is unchanged from pppm/disp.
 *
 * Native accepts it only with pair_style lj/long/coul/long/dielectric ("KSpace style pppm/disp/dielectric is incompatible
 * with Pair style ..." otherwise; kspace.cpp at b332f28). It needs the dispersion parameters (kspace_modify gewald/disp and
 * mesh/disp, or accuracy with disp/auto) as pppm/disp does.
 *
 * Refused (StyleError): the Coulomb automatic G-ewald and grid (set with kspace_modify gewald and mesh; native estimates them
 * from q2 = qqrd2e sum q^2 / epsilon, qsum_qsq), slab (the native slabcorr uses the unscaled charges in its dipole and the
 * per-atom energy is not ported), triclinic boxes, per-atom energy and virial, and the diff ad scheme (pppm/disp refuses it).
 */
import { StyleError, type KSpaceCompute, type Pair, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { KSpacePPPMDisp } from '../pppm_disp/kspace';
import { PairLJLongCoulLongDielectric } from '../csdiel/lj_long_dielectric';
import { dielArrays } from './common';

export class KSpacePPPMDispDielectric extends KSpacePPPMDisp {
  override readonly name: string = 'pppm/disp/dielectric';
  /** Electric field of the Coulomb mesh at each owned atom, native efield: scale * epsilon_i * E (3 per atom). */
  efield = new Float64Array(0);
  private pairLinked: Pair | null = null;
  private pairDie: PairLJLongCoulLongDielectric | null = null;
  /** The dielectric arrays of the current compute (set by compute, read by the hooks). */
  private epsCur: Float64Array<ArrayBufferLike> = new Float64Array(0);
  private qsCur: Float64Array<ArrayBufferLike> = new Float64Array(0);

  override linkPair(pair: Pair | null): void {
    super.linkPair(pair);
    this.pairLinked = pair;
    this.pairDie = pair instanceof PairLJLongCoulLongDielectric ? pair : null;
  }

  override init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (!this.pairDie) {
      const p = this.pairLinked;
      throw new StyleError(`KSpace style pppm/disp/dielectric is incompatible with Pair style ${p ? p.name : 'none'}`);
    }
    dielArrays(s, 'kspace pppm/disp/dielectric');
    super.init(s, geom, cutCoul, qqrd2e, ctx);
  }

  override compute(kc: KSpaceCompute): void {
    const n = kc.s.n;
    const d = dielArrays(kc.s, 'kspace pppm/disp/dielectric');
    this.epsCur = d.epsilon;
    this.qsCur = d.qs;
    if (this.efield.length < 3 * n) this.efield = new Float64Array(3 * n);
    super.compute(kc);
  }

  /** The scaled charges make the Coulomb density of the force pass (make_rho_c with use_qscaled). */
  protected override coulombDensityCharges(s: SimState): Float64Array {
    void s;
    return this.qsCur;
  }

  /** The force on atom i carries the unscaled charge q_i (fieldforce_c_ik: qfactor = qqrd2e * scale * q[i]). */
  protected override coulombForceCharges(s: SimState): Float64Array {
    return s.q;
  }

  /** The energy and virial add the unscaled density (native switches to atom->q for the energy pass). */
  protected override coulombEnergyCharges(s: SimState): Float64Array | null {
    return s.q;
  }

  /** efield = scale * epsilon_i * E with scale 1 (fieldforce_c_ik: efactor = scale * eps[i]). */
  protected override coulombFieldAt(i: number, ex: number, ey: number, ez: number): void {
    const eps = this.epsCur[i];
    this.efield[3 * i] = eps * ex;
    this.efield[3 * i + 1] = eps * ey;
    this.efield[3 * i + 2] = eps * ez;
  }
}
