/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/BOCS/compute_pressure_bocs.cpp and compute_pressure_bocs.h at commit b332f28 (class ComputePressureBocs,
 * compute pressure/bocs): get_cg_p_corr (analytic basis), send_cg_info and compute_scalar. BOCS was written by
 * Nicholas J. H. Dunn and Michael R. DeLyser (Pennsylvania State University). Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The pressure is the ordinary compute pressure (src/engine/compute/pressure.ts) plus the coarse-grained pressure
 * correction of the analytic basis, which fix bocs sends in (send_cg_info). Without a basis the correction is zero,
 * as native LAMMPS (p_basis_type -1). Only the scalar carries the correction in native LAMMPS (compute_vector does
 * not); the fix reads the diagonal through its pressureTensor hook (see fix_bocs.ts).
 *
 * Documented behaviour (docs.lammps.org/compute_pressure_bocs.html): "compute ID group-ID pressure/bocs temp-ID
 * keyword ..."; the keywords are those of compute pressure ("ke", "pair", "bond", ...); "Compute pressure/bocs must
 * use group all".
 */
import { ComputePressure } from '../../engine/compute/pressure';

export interface CgBasis {
  /** Coefficients phi_1 ... phi_N of the analytic basis (fix bocs analytic keyword). */
  phi: number[];
  /** Number of molecules in the coarse-grained system (N_mol). */
  nMol: number;
  /** Reference volume of the analytic basis (vavg). */
  vavg: number;
}

export class ComputePressureBocs extends ComputePressure {
  readonly style: string = 'pressure/bocs';
  private basis: CgBasis | null = null;

  /** send_cg_info (analytic basis): the fix hands in its coefficients, the molecule count and vavg. */
  setCgBasis(basis: CgBasis): void {
    this.basis = { phi: basis.phi.slice(), nMol: basis.nMol, vavg: basis.vavg };
  }

  /**
   * get_cg_p_corr (analytic basis), compute_pressure_bocs.cpp:
   * correction = - sum_{i=1..N} phi_i (N_mol i / vavg) ((1/vavg)(V - vavg))^(i-1)
   */
  correction(vol: number): number {
    const b = this.basis;
    if (!b) return 0;
    let corr = 0;
    for (let i = 1; i <= b.phi.length; i++) {
      let x = 1;
      const base = (1 / b.vavg) * (vol - b.vavg);
      for (let k = 0; k < i - 1; k++) x *= base;
      corr -= b.phi[i - 1] * (b.nMol * i / b.vavg) * x;
    }
    return corr;
  }

  /** compute_scalar: the compute pressure scalar plus the basis correction at the current volume. */
  protected computeScalar(): number {
    const vol = this.sys.geom.volume(this.sys.dimension);
    return super.computeScalar() + this.correction(vol);
  }
}
