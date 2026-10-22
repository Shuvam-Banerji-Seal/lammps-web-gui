/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/pair_sph_taitwater.cpp and
 * pair_sph_taitwater.h (PairSPHTaitwater: pair_style sph/taitwater). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * Tait equation of state with an artificial viscosity (docs.lammps.org/pair_sph_taitwater.html): the pressure of an
 * atom is B ((rho / rho0)^7 - 1) / rho^2 with B = c^2 rho0 / 7; the pair force is -m_i m_j (f_i + f_j + f_visc) dW/dr,
 * the viscous term is switched on for approaching pairs only, and the energy and density rates follow from the same
 * pair. The velocity used is the estimate vest. Half neighbour list; ghost partners are reverse-summed into their owners.
 */
import { StyleError, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from './pair_sph_base';
import { LUCY_DERIV_2D, LUCY_DERIV_3D, sphArrays, sphGhostVest } from './common';
import { tallyAtom } from '../../engine/force/pair/lj_cut';

export class PairSPHTaitwater extends PairSPHBase {
  readonly name: string = 'sph/taitwater';
  protected readonly attributesText = 'attributes energy, density, and velocity estimates';
  protected readonly attributeFlags = ['esph', 'rho', 'vest'] as const;
  override needsHalf = true;
  /** Per type: reference density, sound speed, and the Tait bulk factor B. */
  protected rho0 = new Float64Array(0);
  protected soundspeed = new Float64Array(0);
  protected bulk = new Float64Array(0);
  /** Artificial viscosity coefficient per type pair (pair_coeff visc). */
  protected viscosity = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = this.nt;
    this.rho0 = new Float64Array(nt);
    this.soundspeed = new Float64Array(nt);
    this.bulk = new Float64Array(nt);
    this.viscosity = new Float64Array(nt * nt);
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal number of arguments for pair_style sph/taitwater');
  }

  /** pair_coeff I J rho0 soundspeed viscosity cutoff (pair_sph_taitwater.cpp coeff). */
  coeff(args: string[]): void {
    const r = this.coeffRange(args, 4, 'sph/taitwater');
    const rho0 = parseNum(r.values[0], 'rho0'), c = parseNum(r.values[1], 'soundspeed');
    const visc = parseNum(r.values[2], 'viscosity'), cut = parseNum(r.values[3], 'cutoff');
    const B = c * c * rho0 / 7.0;
    for (let i = r.ilo; i <= r.ihi; i++) { this.rho0[i] = rho0; this.soundspeed[i] = c; this.bulk[i] = B; }
    this.forPairs(r.ilo, r.ihi, r.jlo, r.jhi, (i, j) => { this.viscosity[i * this.nt + j] = visc; this.h[i * this.nt + j] = cut; });
  }

  protected override mirror(i: number, j: number): void {
    this.viscosity[j * this.nt + i] = this.viscosity[i * this.nt + j];
  }

  /** The Lucy-type derivative prefactor for the dimension (pair_sph_taitwater.cpp compute). */
  protected derivative(dim3: boolean, h: number, rsq: number, ih: number, ihsq: number): number {
    let wfd = h - Math.sqrt(rsq);
    if (dim3) wfd = LUCY_DERIV_3D * wfd * wfd * ihsq * ihsq * ihsq * ih;
    else wfd = LUCY_DERIV_2D * wfd * wfd * ihsq * ihsq * ihsq;
    return wfd;
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const a = sphArrays(s, 'Pair sph/taitwater');
    const { rho, drho, desph, vest: v } = a;
    // the ghosts' vest of a setup, when it is not their owners' (sph/common.ts sphGhostVest)
    const gv = sphGhostVest(s);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const mass = s.massByType;
    const dim3 = s.dimension === 3;
    const nt = this.nt, h = this.h, cutsq = this.cutsq, visc = this.viscosity;
    const list = pc.half!;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const vxi = v[3 * i], vyi = v[3 * i + 1], vzi = v[3 * i + 2];
      const it = type[i];
      const imass = mass[it];
      let tmp = rho[i] / this.rho0[it];
      let fi = tmp * tmp * tmp;
      fi = this.bulk[it] * (fi * fi * tmp - 1.0) / (rho[i] * rho[i]);
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
        const wfd = this.derivative(dim3, hh, rsq, ih, ihsq);
        const oj = owner[j];
        const vj = gv && j >= pc.nlocal ? gv : v;
        tmp = rho[oj] / this.rho0[jt];
        let fj = tmp * tmp * tmp;
        fj = this.bulk[jt] * (fj * fj * tmp - 1.0) / (rho[oj] * rho[oj]);
        const delVdotDelR = delx * (vxi - vj[3 * oj]) + dely * (vyi - vj[3 * oj + 1]) + delz * (vzi - vj[3 * oj + 2]);
        let fvisc = 0;
        if (delVdotDelR < 0.) {
          const mu = hh * delVdotDelR / (rsq + 0.01 * hh * hh);
          fvisc = -visc[t] * (this.soundspeed[it] + this.soundspeed[jt]) * mu / (rho[i] + rho[oj]);
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
        // ev_tally(i, j, nlocal, newton_pair, 0, 0, fpair, del) of pair_sph_taitwater.cpp: no energy
        if (pc.vatom) tallyAtom(pc, i, j, 0, fpair, delx, dely, delz);
        desph[oj] += deltaE;
        drho[oj] += imass * delVdotDelR * wfd;
      }
    }
  }
}
