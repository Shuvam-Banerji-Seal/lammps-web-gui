/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/INTERLAYER/pair_kolmogorov_crespi_full.cpp and src/INTERLAYER/pair_kolmogorov_crespi_full.h, at commit
 * b332f28. Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 */
import type { PairCompute } from '../../engine/force/types';
import { NEIGHMASK, type NeighList } from '../../engine/neighbor';
import { calcDTap, calcTap } from './taper';
import { KCNormalBase, tallyPairAtoms, tallyXYZAtoms, type NormalParam } from './normals';

export interface KCFullParam extends NormalParam {
  C0: number; C2: number; C4: number; C: number; A: number; z06: number;
}

/**
 * pair_style kolmogorov/crespi/full (docs.lammps.org/pair_kolmogorov_crespi_full.html): the Kolmogorov-Crespi
 * potential with normals per molecule (computed from the three in-plane neighbors of each atom within rcut of the
 * same molecule). Usage: pair_style kolmogorov/crespi/full cutoff [tap_flag], inside pair_style hybrid/overlay.
 * The potential file lines are z0 C0 C2 C4 C delta lambda A S rcut.
 *
 * The attractive part is summed over half of the pairs (chosen by atom IDs), the repulsive part over every ordered
 * pair (its normal belongs to atom i); the two energies are the pair style's vector (compute pair: [attractive,
 * repulsive]).
 */
export class PairKolmogorovCrespiFull extends KCNormalBase<KCFullParam> {
  readonly name: string = 'kolmogorov/crespi/full';

  /** native's initStyle messages misspell the name ("kolmolgorov/crespi/full", pair_kolmogorov_crespi_full.cpp). */
  protected override initName(): string {
    return 'kolmolgorov/crespi/full';
  }
  protected readonly nwords = 12;

  protected makeParam(nums: number[], ie: number, je: number, scale: number): KCFullParam {
    const [z0, C0, C2, C4, C, delta, lambda, A, S, rcut] = nums;
    // energies in meV further scaled by S
    const meV = 1.0e-3 * S * scale;
    return {
      ie, je, z0, lambda, rcut,
      C: C * meV, A: A * meV, C0: C0 * meV, C2: C2 * meV, C4: C4 * meV,
      delta2inv: Math.pow(delta, -2),
      z06: Math.pow(z0, 6),
    };
  }

  protected offsetOf(p: KCFullParam, cut: number): number {
    return -p.A * Math.pow(p.z0 / cut, 6);
  }

  /** Every entry of the file is kept (no ntypes^3 limit in this style). */
  protected override entryLimit(): number {
    return Infinity;
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
        const vkc = -p.A * p.z06 * r6inv;
        const fpair = -6.0 * p.A * p.z06 * r8inv;
        const fsum = fpair * tap - vkc * dtap / r;
        f[3 * i] += fsum * dx;
        f[3 * i + 1] += fsum * dy;
        f[3 * i + 2] += fsum * dz;
        f[3 * j] -= fsum * dx;
        f[3 * j + 1] -= fsum * dy;
        f[3 * j + 2] -= fsum * dz;
        const e = vkc * tap;
        evdwlSum += e;
        this.pvector[0] += e;
        tallyPairAtoms(pc, i, j, e, fsum, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwlSum;
  }

  /** Repulsive part over every ordered pair, with the normal of atom i (calc_FRep). */
  protected calcFRep(pc: PairCompute, list: NeighList): void {
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    const owner = pc.nb.owner;
    const mol = pc.s.molecule;
    const tapOn = this.tapFlag !== 0;
    let evdwlSum = 0;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i];
      const ni = [this.normal[3 * i], this.normal[3 * i + 1], this.normal[3 * i + 2]];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = ti * nt + tj;
        if (!(rsq < cutsq[t] && mol[owner[i]] !== mol[owner[j]])) continue;
        const p = this.pairP[t]!;
        const r = Math.sqrt(rsq);
        const rc = tapOn ? Math.sqrt(cutsq[t]) : 0;
        const tap = tapOn ? calcTap(r, rc) : 1.0;
        const dtap = tapOn ? calcDTap(r, rc) : 0.0;
        const prodnorm1 = ni[0] * delx + ni[1] * dely + ni[2] * delz;
        const rhosq1 = rsq - prodnorm1 * prodnorm1; // rho_ij
        const rho = rhosq1 * p.delta2inv; // (rho_ij/delta)^2
        const exp0 = Math.exp(-p.lambda * (r - p.z0));
        const exp1 = Math.exp(-rho);
        const sumC1 = p.C0 + p.C2 * rho + p.C4 * rho * rho;
        const sumC11 = (p.C2 + 2.0 * p.C4 * rho) * p.delta2inv;
        const frhoIJ = exp1 * sumC1;
        const sumCff = 0.5 * p.C + frhoIJ;
        const vkc = exp0 * sumCff;
        const fpair = p.lambda * exp0 / r * sumCff;
        const fpair1 = 2.0 * exp0 * exp1 * (p.delta2inv * sumC1 - sumC11);
        const fsum = fpair + fpair1;
        // derivative of prodnorm1 with respect to atom i
        const dpn = [0, 1, 2].map((b) => this.dnormdri[9 * i + b] * delx + this.dnormdri[9 * i + 3 + b] * dely + this.dnormdri[9 * i + 6 + b] * delz);
        const fp1 = [0, 1, 2].map((a) => prodnorm1 * ni[a] * fpair1);
        const fprod1 = [0, 1, 2].map((a) => prodnorm1 * dpn[a] * fpair1);
        const fkc = [0, 1, 2].map((a) => {
          const del = a === 0 ? delx : a === 1 ? dely : delz;
          return (del * fsum - fp1[a]) * tap - vkc * dtap * del / r;
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
        const e = tapOn ? tap * vkc : vkc - this.offset[t];
        evdwlSum += e;
        this.pvector[1] += e;
        tallyXYZAtoms(pc, i, j, e, fkc[0], fkc[1], fkc[2], delx, dely, delz);
      }
    }
    pc.acc.evdwl += evdwlSum;
  }}
