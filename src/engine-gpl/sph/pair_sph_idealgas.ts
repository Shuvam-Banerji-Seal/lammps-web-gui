/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/pair_sph_idealgas.cpp and
 * pair_sph_idealgas.h (PairSPHIdealGas: pair_style sph/idealgas). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Ideal gas equation of state with heat capacity ratio gamma = 1.4 (docs.lammps.org/pair_sph_idealgas.html): the
 * pressure over rho squared is 0.4 esph / m / rho, the sound speed sqrt(0.4 esph / m), and the artificial viscosity
 * of pair_sph_taitwater is applied the same way. Half neighbour list.
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from './pair_sph_base';
import { LUCY_DERIV_2D, LUCY_DERIV_3D, aboveOf, nativeRank, sphArrays } from './common';

export class PairSPHIdealGas extends PairSPHBase {
  readonly name = 'sph/idealgas';
  protected readonly attributesText = 'energy, density, and velocity estimates';
  override needsHalf = true;
  /**
   * Artificial viscosity per type pair. pair_sph_idealgas.cpp init_one() mirrors the cutoff only ("cut[j][i] = cut[i][j]"),
   * not the viscosity, so the entry (j, i) with j > i stays at its initial zero. The pair's viscosity is
   * viscosity[itype][jtype] with itype the atom that owns the pair in LAMMPS' half list: the lower native index for two
   * owned atoms, the owned atom for a ghost above it (compute). Measured with native LAMMPS (black box): two atoms of
   * types 1 and 2 at distance 0.5, approaching, give the velocities of the lower-index atom's viscosity entry.
   */
  private viscosity = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.viscosity = new Float64Array(this.nt * this.nt);
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal number of arguments for pair_style sph/idealgas');
  }

  /** pair_coeff I J viscosity cutoff. */
  coeff(args: string[]): void {
    const r = this.coeffRange(args, 2, 'sph/idealgas');
    const visc = parseNum(r.values[0], 'viscosity'), cut = parseNum(r.values[1], 'cutoff');
    this.forPairs(r.ilo, r.ihi, r.jlo, r.jhi, (i, j) => { this.viscosity[i * this.nt + j] = visc; this.h[i * this.nt + j] = cut; });
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const a = sphArrays(s, 'Pair sph/idealgas');
    const { rho, drho, desph, esph, vest: v } = a;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const mass = s.massByType;
    const dim3 = s.dimension === 3;
    const nt = this.nt, h = this.h, cutsq = this.cutsq, visc = this.viscosity;
    // native rank of each atom (the LAMMPS atom order, used to name the owner of an owned-owned pair)
    const rank = nativeRank(s);
    const list = pc.half!;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vxi = v[3 * i], vyi = v[3 * i + 1], vzi = v[3 * i + 2];
      const it = type[i];
      const imass = mass[it];
      // ideal gas EOS; this expression is fi = pressure / rho^2
      const fi = 0.4 * esph[i] / imass / rho[i];
      // speed of sound with heat capacity ratio gamma = 1.4
      const ci = Math.sqrt(0.4 * esph[i] / imass);
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jt = type[j];
        const t = it * nt + jt;
        if (rsq >= cutsq[t]) continue;
        const jmass = mass[jt];
        const hh = h[t];
        const ih = 1.0 / hh;
        const ihsq = ih * ih;
        let wfd = hh - Math.sqrt(rsq);
        if (dim3) wfd = LUCY_DERIV_3D * wfd * wfd * ihsq * ihsq * ihsq * ih;
        else wfd = LUCY_DERIV_2D * wfd * wfd * ihsq * ihsq * ihsq;
        const oj = owner[j];
        const fj = 0.4 * esph[oj] / jmass / rho[oj];
        const delVdotDelR = delx * (vxi - v[3 * oj]) + dely * (vyi - v[3 * oj + 1]) + delz * (vzi - v[3 * oj + 2]);
        let fvisc = 0;
        if (delVdotDelR < 0.) {
          const cj = Math.sqrt(0.4 * esph[oj] / jmass);
          const mu = hh * delVdotDelR / (rsq + 0.01 * hh * hh);
          // the list owner of the pair decides which viscosity entry applies (see the class comment)
          const iOwns = j < pc.nlocal ? rank[i] < rank[j] : aboveOf(xi, yi, zi, x[3 * j], x[3 * j + 1], x[3 * j + 2]);
          const vt = iOwns ? t : jt * nt + it;
          fvisc = -visc[vt] * (ci + cj) * mu / (rho[i] + rho[oj]);
        }
        const fpair = -imass * jmass * (fi + fj + fvisc) * wfd;
        const deltaE = -0.5 * fpair * delVdotDelR;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        drho[i] += jmass * delVdotDelR * wfd;
        desph[i] += deltaE;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        desph[oj] += deltaE;
        drho[oj] += imass * delVdotDelR * wfd;
      }
    }
  }
}
