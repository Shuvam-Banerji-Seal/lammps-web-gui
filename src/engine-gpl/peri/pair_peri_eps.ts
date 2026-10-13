/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/PERI/pair_peri_eps.cpp and pair_peri_eps.h (PairPeriEPS: pair_style peri/eps) at commit b332f28.
 * Contributing authors: Rezwanur Rahman, J.T. Foster (UTSA); the plastic return follows Mitchell 2011 (SAND2011-3166)
 * as cited in the source. Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * pair_style peri/eps: the elastic-plastic state-based peridynamic model (docs.lammps.org/pair_peri.html). pair_coeff
 * I J K G cut s00 alpha yieldstress. The deviatoric force state of an atom is projected back onto the yield surface
 * when its norm exceeds the yield value; the plastic extension of a bond is kept only for atoms that yield in the step
 * (the temporary array of pair_peri_eps.cpp is zeroed each step, so an elastic atom's plastic extension is reset).
 * The per-atom plastic multiplier lambdaValue of the source is not needed for the forces and is not kept.
 */
import { PERI_GEOM_LEN, G_DR, G_DX, G_DY, G_DZ, G_J, G_OM, G_R, G_R0, G_VS, PeriBondPair, type PeriStep } from './pair_peri_bonds';

const PERI_EPS_NAMES = ['K', 'G', 'cut', 's00', 'alpha', 'yield'] as const;

export class PairPeriEPS extends PeriBondPair {
  readonly name = 'peri/eps';
  protected readonly names = PERI_EPS_NAMES;

  /** compute_DeviatoricForceStateNorm of pair_peri_eps.cpp for owned atom i (the state before this step). */
  private deviatoricNorm(st: PeriStep, i: number, g: Float64Array): number {
    const { type, V } = st;
    const ti = type[i];
    const G = this.c('G', ti, ti);
    let norm = 0;
    for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) {
      if (this.pBr[k]) continue;
      this.bondGeom(st, i, k, g);
      const j = g[G_J], r0 = g[G_R0], dr = g[G_DR], vs = g[G_VS], om = g[G_OM];
      const ed = dr - (this.theta[i] * r0) / 3;
      const edPNP1 = this.devA[k];
      const tdtrial = 15 * G * ((om / this.wvol[i]) + (om / this.wvol[j])) * (ed - edPNP1);
      norm += tdtrial * tdtrial * V[j] * vs;
    }
    return Math.sqrt(norm);
  }

  protected bondForces(st: PeriStep, smin_new: Float64Array, s0_new: Float64Array): void {
    const { nl, f, V, type } = st;
    const g = new Float64Array(PERI_GEOM_LEN);
    // the plastic extension of this step (zero for atoms that stay elastic)
    const plastic = new Float64Array(this.devA.length);
    for (let i = 0; i < nl; i++) {
      const ti = type[i];
      const G = this.c('G', ti, ti);
      const yieldStress = this.c('yield', ti, ti);
      const horizon = this.c('cut', ti, ti);
      const tdnorm = this.deviatoricNorm(st, i, g);
      const pointwiseYield = (25.0 / 8.0 / Math.PI / Math.pow(horizon, 5)) * yieldStress * yieldStress;
      const fsurf = (tdnorm * tdnorm) / 2.0 - pointwiseYield;
      const elastic = !(fsurf > 0.0);
      // the plastic multiplier of this step (pair_peri_eps.cpp: deltalambda); the cumulative lambdaValue is not kept
      const deltalambda = elastic ? 0 : ((tdnorm / Math.sqrt(2.0 * pointwiseYield)) - 1.0) * this.wvol[i] / (15.0 * G);
      let first = true;
      for (let k = this.pStart[i]; k < this.pStart[i + 1]; k++) {
        if (this.pBr[k]) continue;
        this.bondGeom(st, i, k, g);
        const j = g[G_J], dx = g[G_DX], dy = g[G_DY], dz = g[G_DZ];
        const r = g[G_R], dr = g[G_DR], r0 = g[G_R0], om = g[G_OM], vs = g[G_VS];
        const tj = type[j];
        const wi = this.wvol[i], wj = this.wvol[j];
        const thi = this.theta[i];
        // elastic part: the volumetric (dilatation) force state
        const rk = 3.0 * this.c('K', ti, ti) * ((om * thi / wi) + (om * this.theta[j] / wj)) * r0;
        let fbond = r > 0 ? -((rk / r) * V[j] * vs) : 0;
        // plastic part: the deviatoric trial state and its return
        const devext = dr - (thi * r0) / 3.0;
        const edpNp1 = this.devA[k];
        const tdtrial = 15 * G * ((om / wi) + (om / wj)) * (devext - edpNp1);
        let rkNew: number;
        if (elastic) {
          rkNew = tdtrial;
        } else {
          // closest-point return onto the yield surface: rkNew = (sqrt(2 psi0) / ||t_d||) t_d_trial
          rkNew = (Math.sqrt(2.0 * pointwiseYield) * tdtrial) / tdnorm;
          plastic[k] = edpNp1 + rkNew * deltalambda * (0.5 * r0);
        }
        const fbondEP = r > 0 ? -((rkNew / r) * V[j] * vs) : 0;
        fbond += fbondEP;
        f[3 * i] += dx * fbond; f[3 * i + 1] += dy * fbond; f[3 * i + 2] += dz * fbond;
        const evdwl = (0.5 * 15 * G / wi * om * (devext - edpNp1) * (devext - edpNp1)) * V[j] * vs;
        this.tallyBond(st, i, evdwl, fbond, g);
        this.breakAndTrack(i, k, dr / r0, smin_new, s0_new, first, ti, tj, j);
        first = false;
      }
    }
    // the new plastic extensions replace the old ones (memcpy of pair_peri_eps.cpp)
    this.devA = plastic;
  }
}
