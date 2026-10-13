/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_lebedeva_z.cpp and src/INTERLAYER/pair_lebedeva_z.h, at commit b332f28. Contributing
 * author in the C++ header: Zbigniew Koziol (National Center for Nuclear Research, Poland); based on the C code of
 * the Kolmogorov-Crespi potential by Jaap Kroes and others. Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { InterlayerZBase, tallyZPair } from './zbase';

interface LebZParam {
  ie: number; je: number;
  z0: number; A: number; B: number; C: number; alpha: number; D1: number; D2: number;
  lambda1: number; lambda2: number; z02: number; z06: number;
}

/**
 * pair_style lebedeva/z (docs.lammps.org/pair_lebedeva_z.html): the Lebedeva potential with the normals along z,
 * used inside pair_style hybrid/overlay. The potential file lines are A B C z0 alpha D1 D2 lambda1 lambda2 S
 * (energies in meV, scaled by S).
 */
export class PairLebedevaZ extends InterlayerZBase<LebZParam> {
  readonly name: string = 'lebedeva/z';
  protected readonly nwords = 12;

  protected makeParam(nums: number[], ie: number, je: number, scale: number): LebZParam {
    const [A, B, C, z0, alpha, D1, D2, lambda1, lambda2, S] = nums;
    // energies in meV further scaled by S (S = 43.3634 meV = 1 kcal/mol)
    const meV = 1e-3 * S * scale;
    return {
      ie, je, z0, alpha, D1, D2, lambda1, lambda2,
      A: A * meV, B: B * meV, C: C * meV,
      z02: Math.pow(z0, 2),
      z06: Math.pow(z0, 6),
    };
  }

  protected offsetOf(p: LebZParam, cut: number): number {
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
        // rho^2 = r^2 - z^2
        const rhosq = delx * delx + dely * dely;
        const rsq = rhosq + delz * delz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const p = this.pairP[t]!;
        const r = Math.sqrt(rsq);
        const r6 = rsq * rsq * rsq;
        const r8 = r6 * rsq;
        const exp1 = Math.exp(-p.alpha * (r - p.z0));
        const exp2 = Math.exp(-p.lambda1 * rhosq);
        const exp3 = Math.exp(-p.lambda2 * (delz * delz - p.z02));
        const sumD = 1 + p.D1 * rhosq + p.D2 * rhosq * rhosq;
        const Ulm = -p.A * p.z06 / r6 + p.B * exp1 + p.C * sumD * exp2 * exp3;
        // derivatives
        const fpair = -6.0 * p.A * p.z06 / r8 + p.B * p.alpha * exp1 / r; // used for x, y, z
        const der = p.D1 + 2 * p.D2 * rhosq - p.lambda1 * sumD; // used for x, y
        const fxy = 2 * p.C * exp2 * exp3 * der;
        const fz = 2 * p.C * p.lambda2 * sumD * exp2 * exp3;
        f[3 * i] += delx * (fpair - fxy);
        f[3 * i + 1] += dely * (fpair - fxy);
        f[3 * i + 2] += delz * (fpair + fz);
        f[3 * j] -= delx * (fpair - fxy);
        f[3 * j + 1] -= dely * (fpair - fxy);
        f[3 * j + 2] -= delz * (fpair + fz);
        const e = Ulm - this.offset[t];
        evdwl += e;
        if (pc.eatom || pc.vatom) {
          const fi = [-delx * fxy, -dely * fxy, delz * fz];
          tallyZPair(pc, i, j, e, fpair, delx, dely, delz, fi);
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
