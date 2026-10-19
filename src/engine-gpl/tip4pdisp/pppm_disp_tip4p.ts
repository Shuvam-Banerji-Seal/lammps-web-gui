/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pppm_disp_tip4p.cpp and
 * pppm_disp_tip4p.h (PPPMDispTIP4P::init, particle_map_c, make_rho_c, fieldforce_c_ik, find_M). Contributing authors of
 * the LAMMPS file: Amalie Frischknecht and Ahmed Ismail (SNL), Rolf Isele-Holder (Aachen University). Copyright (2003)
 * Sandia Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too.
 *
 * kspace_style pppm/disp/tip4p is pppm/disp (src/engine-gpl/pppm_disp/kspace.ts) with the Coulomb charge of each TIP4P O
 * on its M site: the mesh spreads the M-site charges and the field interpolated at M is projected on O (1 - alpha) and on
 * each H (alpha / 2), as the native fieldforce_c_ik does. The dispersion part uses the atoms (native overrides only the
 * _c routines). The M site is find_M of the native code: O + alpha/2 (d1 + d2) with d1, d2 the minimum-image H - O
 * vectors (the same sites as the pair style, tip4p_sites.ts). Requires pair style lj/long/tip4p/long (the M-site model).
 *
 * Refused (StyleError): the native triclinic and slab branches (pppm/disp refuses them already), and the per-atom
 * energy and virial. The ad differentiation scheme is refused by the pppm/disp port.
 */
import { StyleError, type KSpaceCompute, type Pair, type StyleContext } from '../../engine/force/types';
import type { SimState } from '../../engine/types';
import type { Geometry } from '../../engine/domain';
import { buildOwnedSites, type Tip4pModel } from '../../engine/force/pair/tip4p_sites';
import { KSpacePPPMDisp } from '../pppm_disp/kspace';
import { PairLJLongTIP4PLong, checkTip4pHydrogens } from './pair_lj_long_tip4p_long';

export class KSpacePPPMDispTIP4P extends KSpacePPPMDisp {
  override readonly name: string = 'pppm/disp/tip4p';
  /** compute pe/atom and stress/atom: the M-site per-atom energy and virial (pppm_disp_tip4p.cpp fieldforce_c_peratom) are not ported. */
  protected override peratomPorted(): boolean { return false; }
  private tip4p: PairLJLongTIP4PLong | null = null;
  /** The M-site model and the owned H indices of the current compute (set by coulombPositions). */
  private model: Tip4pModel | null = null;
  private h1: Int32Array = new Int32Array(0);
  private h2: Int32Array = new Int32Array(0);
  private types: Int32Array = new Int32Array(0);

  override linkPair(pair: Pair | null): void {
    super.linkPair(pair);
    this.tip4p = pair instanceof PairLJLongTIP4PLong ? pair : null;
  }

  override init(s: SimState, geom: Geometry, cutCoul: number, qqrd2e: number, ctx: StyleContext): void {
    if (!this.tip4p) throw new StyleError('kspace_style pppm/disp/tip4p needs pair style lj/long/tip4p/long');
    super.init(s, geom, cutCoul, qqrd2e, ctx);
  }

  protected override coulombPositions(s: SimState, geom: Geometry): Float64Array {
    const m = this.tip4p?.siteModel() ?? null;
    if (!m) throw new StyleError('kspace_style pppm/disp/tip4p needs pair style lj/long/tip4p/long (its M-site model is set at init)');
    checkTip4pHydrogens(s.n, s.type, s.id, m.otype, m.htype);
    const sites = buildOwnedSites(s.n, s.x, s.type, s.id, geom, m);
    this.model = m;
    this.h1 = sites.h1;
    this.h2 = sites.h2;
    this.types = s.type;
    return sites.M;
  }

  /** find_M: the charge of an O sits on M, so its field force is split on O (1 - alpha) and on its H atoms (alpha / 2). */
  protected override addCoulombForce(kc: KSpaceCompute, i: number, fx: number, fy: number, fz: number): void {
    const m = this.model;
    if (!m || this.types[i] !== m.otype) {
      kc.f[3 * i] += fx;
      kc.f[3 * i + 1] += fy;
      kc.f[3 * i + 2] += fz;
      return;
    }
    const alpha = m.alpha;
    const F = [fx, fy, fz];
    const a = this.h1[i], b = this.h2[i];
    for (let c = 0; c < 3; c++) {
      kc.f[3 * i + c] += F[c] * (1 - alpha);
      kc.f[3 * a + c] += 0.5 * alpha * F[c];
      kc.f[3 * b + c] += 0.5 * alpha * F[c];
    }
  }
}
