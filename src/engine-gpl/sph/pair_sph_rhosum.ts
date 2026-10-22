/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/pair_sph_rhosum.cpp and
 * pair_sph_rhosum.h (PairSPHRhoSum: pair_style sph/rhosum N). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Density summation (docs.lammps.org/pair_sph_rhosum.html): every N timesteps (N = 0 never) the density of each owned
 * atom is set to its own kernel contribution mass * W(0) plus the sum over its neighbours mass_j * W(r_ij), with the
 * quadric kernel. The pair uses a full neighbour list; the density of a ghost neighbour is its owner's.
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from './pair_sph_base';
import { QUADRIC_2D, QUADRIC_3D, sphArrays } from './common';

export class PairSPHRhoSum extends PairSPHBase {
  readonly name = 'sph/rhosum';
  protected readonly attributesText = 'attribute density';
  protected readonly attributeFlags = ['rho'] as const;
  override needsFull = true;
  override needsHalf = false;
  /** Density is recomputed every nstep timesteps (0: never). */
  nstep = 0;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('Illegal number of arguments for pair_style sph/rhosum');
    this.nstep = Math.trunc(parseNum(args[0], 'nstep'));
  }

  coeff(args: string[]): void {
    const r = this.coeffRange(args, 1, 'sph/rhosum');
    const h = parseNum(r.values[0], 'cutoff');
    this.forPairs(r.ilo, r.ihi, r.jlo, r.jhi, (i, j) => { this.h[i * this.nt + j] = h; });
  }

  /** pair_sph_rhosum.cpp compute(): self contribution, then the neighbour sums (full list). */
  compute(pc: PairCompute): void {
    const s = pc.s;
    if (this.nstep === 0 || s.step % this.nstep !== 0) return;
    const rho = sphArrays(s, 'Pair sph/rhosum').rho;
    const { x, type } = pc;
    const nt = this.nt, h = this.h;
    const mass = s.massByType;
    const dim3 = s.dimension === 3;
    // every owned atom of a type this style is given pair coefficients for: the self contribution, mass * W(0).
    // (Atoms of other types are not in the sub-style's list in the source, so their density is left alone.)
    for (let i = 0; i < pc.nlocal; i++) {
      const it = type[i];
      if (!this.setflag[it * nt + it]) continue;
      const h0 = h[it * nt + it];
      const wf = dim3 ? QUADRIC_3D / (h0 * h0 * h0) : QUADRIC_2D / (h0 * h0);
      rho[i] = mass[it] * wf;
    }
    const list = pc.full!;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti + type[j];
        if (rsq >= this.cutsq[t]) continue;
        const hh = h[t];
        const ih = 1.0 / hh;
        const ihsq = ih * ih;
        let wf = 1.0 - rsq * ihsq;
        wf = wf * wf;
        wf = wf * wf;
        wf = dim3 ? QUADRIC_3D * wf * ihsq * ih : QUADRIC_2D * wf * ihsq;
        rho[i] += mass[type[j]] * wf;
      }
    }
  }
}
