/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/SPH/pair_sph_lj.cpp and
 * pair_sph_lj.h (PairSPHLJ: pair_style sph/lj, Lennard-Jones equation of state of Ree, 1980). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Package author (from src/SPH/README): Georg Ganzenmuller (Fraunhofer EMI).
 *
 * docs.lammps.org/pair_sph_lj.html: "pair_style sph/lj" takes no arguments; "pair_coeff 1 1 viscosity cutoff". The style
 * needs the atom attributes energy, density, specific heat and velocity estimates (atom_style sph). The pressure and the
 * sound speed come from the Lennard-Jones equation of state (LJEOS2 in the source); the pair force, the viscosity and
 * the rates of change follow pair_sph_taitwater (same pair loop over the half neighbour list, ghost partners read from
 * their owners), with the long-range correction lrc added to the second atom's density term.
 */
import { StyleError, typeBounds, type PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { PairSPHBase } from '../sph/pair_sph_base';
import { sphArrays, sphGhostVest } from '../sph/common';
import { tallyAtom } from '../../engine/force/pair/lj_cut';

/**
 * LJEOS2 of pair_sph_lj.cpp: the pressure p and the sound speed c of the Lennard-Jones fluid at density rho, internal
 * energy e and heat capacity cv (Ree 1980, Journal of Chemical Physics 73, 5401). The operation order follows the source.
 */
export const ljEOS2 = (rho: number, e: number, cv: number): { p: number; c: number } => {
  const T = e / cv;
  const beta = 1.0 / T;
  const beta_sqrt = Math.sqrt(beta);
  const x = rho * Math.sqrt(beta_sqrt);
  const xsq = x * x;
  const xpow3 = xsq * x;
  const xpow4 = xsq * xsq;
  // differential of Helmholtz free energy w.r.t. x
  const diff_A_NkT = 3.629 + 7.264 * x - beta * (3.492 - 18.698 * x + 35.505 * xsq - 31.816 * xpow3 + 11.195 * xpow4)
    - beta_sqrt * (5.369 + 13.16 * x + 18.525 * xsq - 17.076 * xpow3 + 9.32 * xpow4)
    + 10.4925 * xsq + 11.46 * xpow3 + 2.176 * xpow4 * xpow4 * x;
  // differential of Helmholtz free energy w.r.t. x^2
  const d2A_dx2 = 7.264 + 20.985 * x
    + beta * (18.698 - 71.01 * x + 95.448 * xsq - 44.78 * xpow3)
    - beta_sqrt * (13.16 + 37.05 * x - 51.228 * xsq + 37.28 * xpow3)
    + 34.38 * xsq + 19.584 * xpow4 * xpow4;
  const p = rho * T * (1.0 + diff_A_NkT * x);
  const csq = T * (1.0 + 2.0 * diff_A_NkT * x + d2A_dx2 * x * x);
  return { p, c: csq > 0.0 ? Math.sqrt(csq) : 0.0 };
};

export class PairSPHLJ extends PairSPHBase {
  readonly name = 'sph/lj';
  protected readonly attributesText = 'attributes energy, density, specific heat, and velocity estimates';
  protected readonly attributeFlags = ['esph', 'rho', 'cv', 'vest'] as const;
  override needsHalf = true;
  /** Artificial viscosity per type pair (pair_coeff viscosity). */
  private viscosity = new Float64Array(0);

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.viscosity = new Float64Array(this.nt * this.nt);
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal number of arguments for pair_style sph/lj');
  }

  /** pair_coeff I J viscosity cutoff (pair_sph_lj.cpp coeff). */
  coeff(args: string[]): void {
    if (args.length !== 4) throw new StyleError('Incorrect args for pair_style sph/lj coefficients');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const visc = parseNum(args[2], 'viscosity'), cut = parseNum(args[3], 'cutoff');
    this.forPairs(ilo, ihi, jlo, jhi, (i, j) => {
      this.viscosity[i * this.nt + j] = visc;
      this.h[i * this.nt + j] = cut;
    });
  }

  protected override mirror(i: number, j: number): void {
    this.viscosity[j * this.nt + i] = this.viscosity[i * this.nt + j];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const a = sphArrays(s, 'Pair sph/lj');
    const { rho, drho, desph, esph, cv, vest: v } = a;
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
      const eos_i = ljEOS2(rho[i], esph[i], cv[i]);
      let fi = eos_i.p;
      const ci = eos_i.c;
      fi /= (rho[i] * rho[i]);
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
        const ihcub = ihsq * ih;
        let wfd = hh - Math.sqrt(rsq);
        if (dim3) wfd = -25.066903536973515383 * wfd * wfd * ihsq * ihsq * ihsq * ih;
        else wfd = -19.098593171027440292 * wfd * wfd * ihsq * ihsq * ihsq;
        const oj = owner[j];
        const vj = gv && j >= pc.nlocal ? gv : v;
        const eos_j = ljEOS2(rho[oj], esph[oj], cv[oj]);
        let fj = eos_j.p;
        const cj = eos_j.c;
        fj /= (rho[oj] * rho[oj]);
        const lrc = -11.1701 * (ihcub * ihcub * ihcub - 1.5 * ihcub);
        fj += 2.0 * lrc;
        const delVdotDelR = delx * (vxi - vj[3 * oj]) + dely * (vyi - vj[3 * oj + 1]) + delz * (vzi - vj[3 * oj + 2]);
        let fvisc = 0;
        if (delVdotDelR < 0.) {
          const mu = hh * delVdotDelR / (rsq + 0.01 * hh * hh);
          fvisc = -visc[t] * (ci + cj) * mu / (rho[i] + rho[oj]);
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
        // ev_tally(i, j, nlocal, newton_pair, 0, 0, fpair, del) of pair_sph_lj.cpp: no energy
        if (pc.vatom) tallyAtom(pc, i, j, 0, fpair, delx, dely, delz);
        desph[oj] += deltaE;
        drho[oj] += imass * delVdotDelR * wfd;
      }
    }
  }
}
