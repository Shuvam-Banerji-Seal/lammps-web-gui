/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/pair_sph_heatconduction.cpp and
 * pair_sph_heatconduction.h (PairSPHHeatConduction: pair_style sph/heatconduction). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Heat conduction between SPH particles (docs.lammps.org/pair_sph_heatconduction.html): the rate of internal energy of
 * atom i gets 2 m_i m_j / (m_i + m_j) (rho_i + rho_j) / (rho_i rho_j) alpha (e_i - e_j) dW/dr, the partner receives the
 * opposite rate. No forces. The pair loop covers the half neighbour list; ghost partners are reverse-summed.
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from './pair_sph_base';
import { LUCY_DERIV_2D, LUCY_DERIV_3D, sphArrays } from './common';

export class PairSPHHeatConduction extends PairSPHBase {
  readonly name = 'sph/heatconduction';
  protected readonly attributesText = 'attributes energy and density';
  protected readonly attributeFlags = ['esph', 'rho'] as const;
  override needsHalf = true;
  /** Diffusion coefficient per type pair. */
  private alpha = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.alpha = new Float64Array(this.nt * this.nt);
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal number of arguments for pair_style sph/heatconduction');
  }

  /** pair_coeff I J diffusion_coefficient cutoff. */
  coeff(args: string[]): void {
    const r = this.coeffRange(args, 2, 'sph/heatconduction');
    const alpha = parseNum(r.values[0], 'diffusion coefficient'), cut = parseNum(r.values[1], 'cutoff');
    this.forPairs(r.ilo, r.ihi, r.jlo, r.jhi, (i, j) => { this.alpha[i * this.nt + j] = alpha; this.h[i * this.nt + j] = cut; });
  }

  protected override mirror(i: number, j: number): void {
    this.alpha[j * this.nt + i] = this.alpha[i * this.nt + j];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const { rho, esph, desph } = sphArrays(s, 'Pair sph/heatconduction');
    const { x, type } = pc;
    const owner = pc.nb.owner;
    const mass = s.massByType;
    const dim3 = s.dimension === 3;
    const nt = this.nt, h = this.h, cutsq = this.cutsq, alpha = this.alpha;
    const list = pc.half!;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const it = type[i];
      const imass = mass[it];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jt = type[j];
        const t = it * nt + jt;
        if (rsq >= cutsq[t]) continue;
        const hh = h[t];
        const ih = 1.0 / hh;
        const ihsq = ih * ih;
        // kernel derivative; the missing factor of 1/r is recovered in deltaE (see pair_sph_taitwater.cpp)
        let wfd = hh - Math.sqrt(rsq);
        if (dim3) wfd = LUCY_DERIV_3D * wfd * wfd * ihsq * ihsq * ihsq * ih;
        else wfd = LUCY_DERIV_2D * wfd * wfd * ihsq * ihsq * ihsq;
        const jmass = mass[jt];
        const oj = owner[j];
        const D = alpha[t];
        let deltaE = 2.0 * imass * jmass / (imass + jmass);
        deltaE *= (rho[i] + rho[oj]) / (rho[i] * rho[oj]);
        deltaE *= D * (esph[i] - esph[oj]) * wfd;
        desph[i] += deltaE;
        desph[oj] -= deltaE;
      }
    }
  }
}
