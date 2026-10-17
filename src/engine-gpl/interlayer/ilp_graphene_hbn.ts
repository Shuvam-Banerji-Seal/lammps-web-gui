/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_ilp_graphene_hbn.cpp and src/INTERLAYER/pair_ilp_graphene_hbn.h, at commit b332f28
 * (the ilp/graphene/hbn variant of the shared ILP code; the normals come from the KC neighbor and normal routines of
 * ./normals.ts, which are identical to the C++ ILP_neigh and calc_normal for this variant), and
 * src/INTERLAYER/interlayer_taper.h. Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import type { PairCompute } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { calcDTap, calcTap } from './taper';
import { KCNormalBase, tallyPairAtoms, tallyXYZAtoms, type NormalParam } from './normals';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';

export interface ILPParam extends NormalParam {
  epsilon: number; C: number; d: number; seff: number; C6: number;
}

/**
 * pair_style ilp/graphene/hbn (docs.lammps.org/pair_ilp_graphene_hbn.html): the registered interlayer potential for
 * graphene, hBN and their mixtures, with normals per molecule. Usage: pair_style ilp/graphene/hbn cutoff [tap_flag]
 * inside pair_style hybrid/overlay (the taper is on by default). The potential file lines are z0 alpha delta epsilon
 * C d sR reff C6 S rcut (energies in meV, scaled by S).
 *
 * The attractive part is summed over half of the pairs (chosen by atom IDs), the repulsive part over every ordered
 * pair; the two energies are the pair style's vector (compute pair: [attractive, repulsive]).
 */
export class PairILPGrapheneHBN extends KCNormalBase<ILPParam> {
  readonly name: string = 'ilp/graphene/hbn';
  protected readonly nwords = 13;

  constructor() {
    super();
    // turn on the taper function by default
    this.tapFlag = 1;
  }

  protected makeParam(nums: number[], ie: number, je: number, scale: number): ILPParam {
    const [z0, alpha, delta, epsilon, C, d, sR, reff, C6, S, rcut] = nums;
    // energies in meV further scaled by S
    const meV = 1e-3 * S * scale;
    return {
      ie, je, z0, rcut, d,
      lambda: alpha / z0,
      delta2inv: libmPow(delta, -2.0),
      seff: sR * reff,
      epsilon: epsilon * meV,
      C: C * meV,
      C6: C6 * meV,
    };
  }

  protected offsetOf(p: ILPParam, cut: number): number {
    return -p.C6 * libmPow(1.0 / cut, 6) / (1.0 + libmExp(-p.d * (cut / p.seff - 1.0)));
  }

  protected calcFvdW(pc: PairCompute, list: NeighList): void {
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const ids = pc.s.id;
    const tapOn = this.tapFlag !== 0;
    let evdwlSum = 0;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      const itag = ids[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const jtag = ids[owner[j]];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          // the same atom (a periodic image): keep one of the two directions
          if (x[3 * j + 2] < zi) continue;
          if (x[3 * j + 2] === zi && x[3 * j + 1] < yi) continue;
          if (x[3 * j + 2] === zi && x[3 * j + 1] === yi && x[3 * j] < xi) continue;
        }
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti * nt + type[j];
        if (!(rsq < cutsq[t] && mol[owner[i]] !== mol[owner[j]])) continue;
        const p = this.pairP[t]!;
        const r = Math.sqrt(rsq);
        const r2inv = 1.0 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        const r8inv = r6inv * r2inv;
        const rc = Math.sqrt(cutsq[t]);
        const tap = tapOn ? calcTap(r, rc) : 1.0;
        const dtap = tapOn ? calcDTap(r, rc) : 0.0;
        const tsvdw = 1.0 + libmExp(-p.d * (r / p.seff - 1.0));
        const tsvdw2inv = libmPow(tsvdw, -2.0);
        const vilp = -p.C6 * r6inv / tsvdw;
        const fpair = -6.0 * p.C6 * r8inv / tsvdw + p.C6 * p.d / p.seff * (tsvdw - 1.0) * tsvdw2inv * r8inv * r;
        const fsum = fpair * tap - vilp * dtap / r;
        f[3 * i] += fsum * dx;
        f[3 * i + 1] += fsum * dy;
        f[3 * i + 2] += fsum * dz;
        f[3 * j] -= fsum * dx;
        f[3 * j + 1] -= fsum * dy;
        f[3 * j + 2] -= fsum * dz;
        const e = vilp * tap;
        evdwlSum += e;
        this.pvector[0] += e;
        tallyPairAtoms(pc, i, j, e, fsum, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwlSum;
  }

  /** Element test of the metal repulsion of saip/metal (a metal atom has no transverse term). Default: no metals. */
  protected metalTest: (element: string) => boolean = () => false;

  protected calcFRep(pc: PairCompute, list: NeighList): void {
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const tapOn = this.tapFlag !== 0;
    const metal = new Array<boolean>(nt).fill(false);
    for (let t = 1; t < nt; t++) metal[t] = this.map[t] >= 0 && this.metalTest(this.elements[this.map[t]]);
    let evdwlSum = 0;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      const ni = [this.normal[3 * i], this.normal[3 * i + 1], this.normal[3 * i + 2]];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti * nt + type[j];
        if (!(rsq < cutsq[t] && mol[owner[i]] !== mol[owner[j]])) continue;
        const p = this.pairP[t]!;
        const r = Math.sqrt(rsq);
        const rc = Math.sqrt(cutsq[t]);
        const tap = tapOn ? calcTap(r, rc) : 1.0;
        const dtap = tapOn ? calcDTap(r, rc) : 0.0;
        if (metal[ti]) {
          // saip/metal: a metal atom's repulsion has no transverse dependence (frho1 = C), so no normal terms
          const exp0m = libmExp(-p.lambda * (r - p.z0));
          const erepm = 0.5 * p.epsilon + p.C;
          const vilpm = exp0m * erepm;
          const fpairm = p.lambda * exp0m / r * erepm;
          const filp = fpairm * tap - vilpm * dtap / r;
          f[3 * i] += delx * filp;
          f[3 * i + 1] += dely * filp;
          f[3 * i + 2] += delz * filp;
          f[3 * j] -= delx * filp;
          f[3 * j + 1] -= dely * filp;
          f[3 * j + 2] -= delz * filp;
          const em = tap * vilpm;
          evdwlSum += em;
          this.pvector[1] += em;
          tallyPairAtoms(pc, i, j, em, filp, delx, dely, delz);
          continue;
        }
        const prodnorm1 = ni[0] * delx + ni[1] * dely + ni[2] * delz;
        const rhosq1 = rsq - prodnorm1 * prodnorm1; // rho_ij
        const rdsq1 = rhosq1 * p.delta2inv; // (rho_ij/delta)^2
        const exp0 = libmExp(-p.lambda * (r - p.z0));
        const exp1 = libmExp(-rdsq1);
        const frho1 = exp1 * p.C;
        const erep = 0.5 * p.epsilon + frho1;
        const vilp = exp0 * erep;
        const fpair = p.lambda * exp0 / r * erep;
        const fpair1 = 2.0 * exp0 * frho1 * p.delta2inv;
        const fsum = fpair + fpair1;
        const dpn = [0, 1, 2].map((b) => this.dnormdri[9 * i + b] * delx + this.dnormdri[9 * i + 3 + b] * dely + this.dnormdri[9 * i + 6 + b] * delz);
        const fp1 = [0, 1, 2].map((a) => prodnorm1 * ni[a] * fpair1);
        const fprod1 = [0, 1, 2].map((a) => prodnorm1 * dpn[a] * fpair1);
        const fkc = [0, 1, 2].map((a) => {
          const del = a === 0 ? delx : a === 1 ? dely : delz;
          return (del * fsum - fp1[a]) * tap - vilp * dtap * del / r;
        });
        f[3 * i] += fkc[0] - fprod1[0] * tap;
        f[3 * i + 1] += fkc[1] - fprod1[1] * tap;
        f[3 * i + 2] += fkc[2] - fprod1[2] * tap;
        f[3 * j] -= fkc[0];
        f[3 * j + 1] -= fkc[1];
        f[3 * j + 2] -= fkc[2];
        // the KC neighbors of i move with the normal
        const del = [delx, dely, delz];
        for (let kk = 0; kk < this.kcN[i]; kk++) {
          const kidx = this.kcJ[this.maxKC * i + kk];
          if (kidx === i) continue;
          const base = 9 * this.maxKC * i + 9 * kk;
          const dp = [0, 1, 2].map((b) => this.dnormal[base + b] * del[0] + this.dnormal[base + 3 + b] * del[1] + this.dnormal[base + 6 + b] * del[2]);
          const fk = [0, 1, 2].map((a) => (-prodnorm1 * dp[a] * fpair1) * tap);
          f[3 * kidx] += fk[0];
          f[3 * kidx + 1] += fk[1];
          f[3 * kidx + 2] += fk[2];
          tallyXYZAtoms(pc, kidx, i, 0, fk[0], fk[1], fk[2], x[3 * kidx] - x[3 * i], x[3 * kidx + 1] - x[3 * i + 1], x[3 * kidx + 2] - x[3 * i + 2]);
        }
        const e = tap * vilp;
        evdwlSum += e;
        this.pvector[1] += e;
        tallyXYZAtoms(pc, i, j, e, fkc[0], fkc[1], fkc[2], delx, dely, delz);
      }
    }
    pc.acc.evdwl += evdwlSum;
  }
}
