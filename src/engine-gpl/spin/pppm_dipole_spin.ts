/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/SPIN/pppm_dipole_spin.cpp and pppm_dipole_spin.h (PPPMDipoleSpin, a subclass of PPPMDipole: the spin
 * moments, the precession field of fieldforce_ik_spin, spsum_spsq and the init and compute overrides) at the same
 * commit. Contributing author of pppm_dipole_spin.cpp: Julien Tranchida (SNL). Copyright (2003) Sandia Corporation,
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The solver (mesh, G-ewald, the influence function, the energy, the virial and the mechanical force) is the shared
 * PPPMDipole of src/engine-gpl/dipole/pppm_dipole.ts. The moment of atom i is mu_i = sp_i |sp_i| and the energies
 * carry mub2mu0 = mub^2 mu_0 / (4 pi). The precession field of atom i, fm_long_i = (mub2mu0/hbar) |sp_i| E_i (native
 * fieldforce_ik_spin), is stored per atom id (fmLongOf) for fix nve/spin, not as a torque.
 *
 * Per-atom energy (fieldforce_peratom_spin and the end of compute): the PPPMDipole per-atom path with these moments
 * and prefactor, spx u_x + spy u_y + spz u_z, halved, less the self term of |sp_i sp3_i|^2, times mub2mu0. As native
 * LAMMPS does (compute), the per-atom virial is refused: "Cannot (yet) compute per-atom virial with kspace style
 * pppm/dipole/spin".
 *
 * Not ported (refused by name): the slab correction, kspace_modify diff ad, triclinic boxes.
 */
import { StyleError, type KSpaceCompute } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import { KSpacePPPMDipole } from '../dipole/pppm_dipole';
import { fmLongOf } from './dipole_long_field';
import { MUB2MU0, MUB2MU0HBINV } from './pair_spin_dipole_cut';

export class KSpacePPPMDipoleSpin extends KSpacePPPMDipole {
  /** The precession buffer of this compute (fm_long per atom id), reset at the start of each pass. */
  private fmLong: Float64Array = new Float64Array(0);

  constructor() {
    super('pppm/dipole/spin', 'PPPMDipoleSpin');
  }

  protected checkMoments(s: SimState): void {
    if (!s.sp) throw new StyleError('Kspace style requires atom attribute sp');
  }

  protected checkUnits(s: SimState): void {
    if (s.units.style !== 'metal') throw new StyleError("'metal' units have to be used with spins");
  }

  /** mu_i = sp_i |sp_i|: the direction times the norm (native spsum_spsq and make_rho_spin). */
  protected moments(s: SimState): Float64Array {
    const sp = s.sp!, out = new Float64Array(3 * s.n);
    for (let i = 0; i < s.n; i++) {
      const spn = sp[4 * i + 3];
      out[3 * i] = sp[4 * i] * spn;
      out[3 * i + 1] = sp[4 * i + 1] * spn;
      out[3 * i + 2] = sp[4 * i + 2] * spn;
    }
    return out;
  }

  protected prefactor(_qqrd2e: number): number {
    return MUB2MU0;
  }

  protected noMomentError(): string {
    return 'Using kspace solver PPPMDipoleSpin on system with no spins';
  }

  override compute(kc: KSpaceCompute): void {
    if (kc.vatom) throw new StyleError('Cannot (yet) compute per-atom virial with kspace style pppm/dipole/spin');
    super.compute(kc);
  }

  protected beginField(kc: KSpaceCompute): void {
    this.fmLong = fmLongOf(kc.s, true);
  }

  /** fieldforce_ik_spin: the precession field fm_long_i = (mub2mu0/hbar) |sp_i| E_i, stored per atom id. */
  protected applyField(kc: KSpaceCompute, i: number, _C: number, E: Float64Array): void {
    const s = kc.s;
    const spn = s.sp![4 * i + 3];
    const id = 3 * s.id[i];
    this.fmLong[id] += spn * MUB2MU0HBINV * E[0];
    this.fmLong[id + 1] += spn * MUB2MU0HBINV * E[1];
    this.fmLong[id + 2] += spn * MUB2MU0HBINV * E[2];
  }
}
