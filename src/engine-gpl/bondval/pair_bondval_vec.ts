/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-PAIR/pair_bondval_vec.cpp and pair_bondval_vec.h at commit b332f28 (the bond-valence pair styles by the
 * group of Andrew M. Rappe and their LAMMPS authors). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_bondval.html (verbatim, the page covers pair_style bondval and bondval/vec): "pair_style bondval/vec
 * power cutoff"; the bond-valence vector "\vec{V}_{ij} = V_{ij} \hat{R}_{ij}" with "\hat{R}_{ij}" the unit vector from atom i
 * toward atom j; the bond-valence vector sum "\vec{W}_i = \sum_{j \neq i} \vec{V}_{ij}" and the energy
 * "E_{BVV} = \sum_i D_i ( |\vec{W}_i|^2 - |\vec{W}_{0,i}|^2 )^2". Coefficients: "pair_coeff I J r0 alpha D W0 [cutoff]".
 *
 * The native compute() sums the vector sum W_i over the full neighbour list of each owned atom (with the displacement
 * unit vector scaled as in the native code), copies D_i = S_i power 2 s_i W_i to the ghosts, and runs the force loop
 * over the half list. The virial is the dot product of coordinates and forces (virialFdotr) as native does.
 */
import type { PairCompute } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { PairBondValBase } from './pair_bondval';

/** pair_style bondval/vec: the bond-valence vector energy E_BVV = sum_i D_i (|W_i|^2 - W0_i^2)^2. */
export class PairBondValVec extends PairBondValBase {
  readonly name = 'bondval/vec';
  private wBuf = new Float64Array(0);
  private dBuf = new Float64Array(0);

  compute(pc: PairCompute): void {
    const full = pc.full;
    const half = pc.half;
    if (!full || !half) throw new Error(`pair style ${this.name} needs full and half neighbor lists`);
    const { x, f, type } = pc;
    const nlocal = pc.nlocal, nall = pc.nall;
    const nt = this.nt;
    const cutsq = this.cutsq;
    const { r0, alpha, sp, v0 } = this;
    const power = this.power;
    if (this.wBuf.length !== 3 * nall) {
      this.wBuf = new Float64Array(3 * nall);
      this.dBuf = new Float64Array(3 * nall);
    }
    // bond-valence vector sum W_i over the full neighbours of each owned atom: native adds
    // (r0/r)^alpha (x_i - x_j)/r for the pair seen from i (no alpha gate, the cutoff only)
    const w = this.wBuf;
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let wx = 0, wy = 0, wz = 0;
      const k0: number = full.firstneigh[i], k1: number = k0 + full.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = full.neighbors[k] & NEIGHMASK;
        const t = ti + type[j];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const recip = 1 / Math.sqrt(rsq);
        const g = Math.pow(r0[t] * recip, alpha[t]) * recip;
        wx += g * dx; wy += g * dy; wz += g * dz;
      }
      w[3 * i] = wx; w[3 * i + 1] = wy; w[3 * i + 2] = wz;
    }
    // D_i = S_i power 2 W_i s_i with s_i = |W_i|^2 - V0_i^2; the energy S_i s_i^2 per owned atom
    const d = this.dBuf;
    for (let i = 0; i < nlocal; i++) {
      const ti = type[i] * nt + type[i];
      const wx = w[3 * i], wy = w[3 * i + 1], wz = w[3 * i + 2];
      const s = wx * wx + wy * wy + wz * wz - v0[ti] * v0[ti];
      const c = sp[ti] * power * 2.0 * s;
      d[3 * i] = c * wx; d[3 * i + 1] = c * wy; d[3 * i + 2] = c * wz;
      const phi = sp[ti] * s * s;
      pc.acc.evdwl += phi;
      if (pc.eatom) pc.eatom[i] += phi;
    }
    pc.nb.forwardCopy(d, 3);
    // forces over the half list (native: A = (r0/r)^alpha / r, E = (alpha + 1) / r^2)
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      const k0: number = half.firstneigh[i], k1: number = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const t = ti + type[j];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const recip = 1 / r;
        const recip2 = recip * recip;
        const aij = Math.pow(r0[t] * recip, alpha[t]) * recip;
        const eij = (alpha[t] + 1.0) * recip2;
        // dD = D_i - D_j; F = -A dD + E A del (dD . del): the native fx, fy, fz expressions in vector form
        const ddx = d[3 * i] - d[3 * j], ddy = d[3 * i + 1] - d[3 * j + 1], ddz = d[3 * i + 2] - d[3 * j + 2];
        const dot = ddx * dx + ddy * dy + ddz * dz;
        const c = eij * aij * dot;
        const fx = -aij * ddx + c * dx;
        const fy = -aij * ddy + c * dy;
        const fz = -aij * ddz + c * dz;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        if (pc.vatom) {
          const va = pc.vatom;
          const v = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
          for (const kk of [i, j]) for (let q = 0; q < 6; q++) va[6 * kk + q] += 0.5 * v[q];
        }
      }
    }
  }
}
