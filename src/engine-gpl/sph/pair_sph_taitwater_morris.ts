/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPH/pair_sph_taitwater_morris.cpp and
 * pair_sph_taitwater_morris.h (PairSPHTaitwaterMorris: pair_style sph/taitwater/morris). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * The Tait pressure of pair_style sph/taitwater with Morris' laminar viscosity (Morris, 1996, as cited in the source):
 * the viscous force is 2 mu / (rho_i rho_j) m_i m_j dW/dr (v_i - v_j) for every pair in the cutoff, the energy rate
 * takes the work of both terms, and the viscous term does not enter the density rate. docs.lammps.org/pair_sph_taitwater_morris.html
 */
import { type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { PairSPHTaitwater } from './pair_sph_taitwater';
import { sphArrays } from './common';

export class PairSPHTaitwaterMorris extends PairSPHTaitwater {
  override readonly name: string = 'sph/taitwater/morris';

  compute(pc: PairCompute): void {
    const s = pc.s;
    const a = sphArrays(s, 'Pair sph/taitwater/morris');
    const { rho, drho, desph, vest: v } = a;
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
        // the derivative of the Lucy weight lacks a factor of r; delV . delX and delX (not its unit vector) recover it
        const wfd = this.derivative(dim3, hh, rsq, ih, ihsq);
        const oj = owner[j];
        tmp = rho[oj] / this.rho0[jt];
        let fj = tmp * tmp * tmp;
        fj = this.bulk[jt] * (fj * fj * tmp - 1.0) / (rho[oj] * rho[oj]);
        const velx = vxi - v[3 * oj], vely = vyi - v[3 * oj + 1], velz = vzi - v[3 * oj + 2];
        const delVdotDelR = delx * velx + dely * vely + delz * velz;
        // Morris viscosity (Morris, 1996)
        let fvisc = 2 * visc[t] / (rho[i] * rho[oj]);
        fvisc *= imass * jmass * wfd;
        const fpair = -imass * jmass * (fi + fj) * wfd;
        const deltaE = -0.5 * (fpair * delVdotDelR + fvisc * (velx * velx + vely * vely + velz * velz));
        f[3 * i] += delx * fpair + velx * fvisc;
        f[3 * i + 1] += dely * fpair + vely * fvisc;
        f[3 * i + 2] += delz * fpair + velz * fvisc;
        drho[i] += jmass * delVdotDelR * wfd;
        desph[i] += deltaE;
        f[3 * j] -= delx * fpair + velx * fvisc;
        f[3 * j + 1] -= dely * fpair + vely * fvisc;
        f[3 * j + 2] -= delz * fpair + velz * fvisc;
        desph[oj] += deltaE;
        drho[oj] += imass * delVdotDelR * wfd;
      }
    }
  }
}
