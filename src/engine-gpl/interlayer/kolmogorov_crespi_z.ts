/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_kolmogorov_crespi_z.cpp and src/INTERLAYER/pair_kolmogorov_crespi_z.h, at commit
 * b332f28. Contributing author in the C++ header: Jaap Kroes (Radboud Universiteit); based on earlier versions by
 * Merel van Wijk and by Marco Raguzzoni. Copyright (2003) Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { InterlayerZBase, tallyZPair } from './zbase';

interface KCZParam {
  ie: number; je: number;
  z0: number; C0: number; C2: number; C4: number; C: number; delta2inv: number; lambda: number; A: number; z06: number;
}

/**
 * pair_style kolmogorov/crespi/z (docs.lammps.org/pair_kolmogorov_crespi_z.html): the simplified Kolmogorov-Crespi
 * potential with all normals along z, used inside pair_style hybrid/overlay. The potential file lines are
 * z0 C0 C2 C4 C delta lambda A S (energies in meV, scaled by S).
 */
export class PairKolmogorovCrespiZ extends InterlayerZBase<KCZParam> {
  readonly name: string = 'kolmogorov/crespi/z';
  protected readonly nwords = 11;

  protected makeParam(nums: number[], ie: number, je: number, scale: number): KCZParam {
    const [z0, C0, C2, C4, C, delta, lambda, A, S] = nums;
    // energies in meV further scaled by S (S = 43.3634 meV = 1 kcal/mol)
    const meV = 1e-3 * S * scale;
    return {
      ie, je, z0, lambda,
      C: C * meV, A: A * meV, C0: C0 * meV, C2: C2 * meV, C4: C4 * meV,
      delta2inv: Math.pow(delta, -2),
      z06: Math.pow(z0, 6),
    };
  }

  protected offsetOf(p: KCZParam, cut: number): number {
    return -p.A * Math.pow(p.z0 / cut, 6);
  }

  override compute(pc: PairCompute): void {
    const list = pc.half;
    if (!list) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nb[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        // rho^2 = r^2 - (n,r) = r^2 - z^2 (all normals along z)
        const rhosq = delx * delx + dely * dely;
        const rsq = rhosq + delz * delz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const p = this.pairP[t]!;
        const r = Math.sqrt(rsq);
        const r6 = rsq * rsq * rsq;
        const r8 = r6 * rsq;
        const rdsq = rhosq * p.delta2inv; // (rho/delta)^2
        const exp1 = Math.exp(-p.lambda * (r - p.z0));
        const exp2 = Math.exp(-rdsq);
        // f(rho_ij) equals f(rho_ji) as the normals are all along z
        const sumC = p.C0 + p.C2 * rdsq + p.C4 * rdsq * rdsq;
        const sumC2 = (2 * p.C2 + 4 * p.C4 * rdsq) * p.delta2inv;
        const frho = exp2 * sumC;
        const sumCff = p.C + 2 * frho;
        // derivatives
        const fpair = -6.0 * p.A * p.z06 / r8 + p.lambda * exp1 / r * sumCff;
        const fpair1 = exp1 * exp2 * (4.0 * p.delta2inv * sumC - 2.0 * sumC2);
        const fsum = fpair + fpair1;
        // fi_z has no d(rho)/dz contribution: rho_ij does not depend on z_i or z_j
        f[3 * i] += delx * fsum;
        f[3 * i + 1] += dely * fsum;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fsum;
        f[3 * j + 1] -= dely * fsum;
        f[3 * j + 2] -= delz * fpair;
        const e = -p.A * p.z06 / r6 + exp1 * sumCff - this.offset[t];
        evdwl += e;
        if (pc.eatom || pc.vatom) {
          const fi = [delx * fpair1, dely * fpair1, 0];
          tallyZPair(pc, i, j, e, fpair, delx, dely, delz, fi);
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
