/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EXTRA-COMPUTE/compute_basal_atom.cpp and
 * compute_basal_atom.h (ComputeBasalAtom: compute basal/atom, the basal-plane vector of hcp atoms from their ~180 degree
 * bond angles). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License,
 * version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). The neighbour selection is that of the
 * ackland/atom algorithm, as the native source says.
 *
 * docs.lammps.org/compute_basal_atom.html: "compute ID group-ID basal/atom"; "This compute calculates a per-atom array
 * with three columns, which can be accessed by indices 1--3". The three columns are the unit basal-plane vector.
 *
 * Neighbours are the engine's ghost atoms within the pair cutoff (the native full neighbor list, occasional), with the
 * ghost coordinates. The native code's own quirks are kept as they are (the BPV y component of the more-than-three branch
 * uses x5[2], and the z mean of the 6th sign combination accumulates z7), so the result matches native. Native's
 * neighbour order is not reproduced; with ties (a perfect lattice) the order can decide the choice between equal
 * candidates, so tests displace the atoms.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

/** C copysign(1.0, b) (sign taken from the sign bit of b, so -0.0 gives -1). */
const copysign1 = (b: number): number => (b < 0 || Object.is(b, -0) ? -1 : 1);

/** compute_basal_atom.cpp select2: Numerical Recipes partial sort (k smallest, carries the index array). */
const select2 = (k: number, n: number, arr0: Float64Array, iarr0: Int32Array): void => {
  // 1-based views (the native code decrements the pointers)
  const arr = (i: number) => i - 1;
  let i: number, ir: number, j: number, l: number, mid: number, ia: number;
  let a: number;
  l = 1;
  ir = n;
  const A = (x: number) => arr0[arr(x)];
  const sw = (p: number, q: number) => {
    const t = arr0[arr(p)]; arr0[arr(p)] = arr0[arr(q)]; arr0[arr(q)] = t;
    const u = iarr0[arr(p)]; iarr0[arr(p)] = iarr0[arr(q)]; iarr0[arr(q)] = u;
  };
  for (;;) {
    if (ir <= l + 1) {
      if (ir === l + 1 && A(ir) < A(l)) sw(l, ir);
      return;
    }
    mid = (l + ir) >> 1;
    sw(mid, l + 1);
    if (A(l) > A(ir)) sw(l, ir);
    if (A(l + 1) > A(ir)) sw(l + 1, ir);
    if (A(l) > A(l + 1)) sw(l, l + 1);
    i = l + 1;
    j = ir;
    a = A(l + 1);
    ia = iarr0[arr(l + 1)];
    for (;;) {
      do i++; while (A(i) < a);
      do j--; while (A(j) > a);
      if (j < i) break;
      sw(i, j);
    }
    arr0[arr(l + 1)] = arr0[arr(j)];
    arr0[arr(j)] = a;
    iarr0[arr(l + 1)] = iarr0[arr(j)];
    iarr0[arr(j)] = ia;
    if (j >= k) ir = j - 1;
    if (j <= k) l = i;
  }
};

/** Sum of squared deviations from the mean for the three components (the native var5/var6/var7 expression). */
const spread = (x: Float64Array | number[], y: Float64Array | number[], z: Float64Array | number[], mx: number, my: number, mz: number, count: number): number => {
  let v = 0;
  for (let j = 0; j < count; j++) {
    v = v + x[j] * x[j] - 2 * x[j] * mx + mx * mx + y[j] * y[j] - 2 * y[j] * my + my * my + z[j] * z[j] - 2 * z[j] * mz + mz * mz;
  }
  return v;
};

export class ComputeBasalAtom extends Compute {
  readonly style = 'basal/atom';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0) throw new StyleError('Illegal compute basal/atom command');
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
  }

  protected computePeratom(): void {
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError('compute basal/atom requires a pair style be defined');
    this.sys.forces();
    const s = this.sys.state;
    const nb = this.sys.nb;
    const n = s.n;
    const nall = nb.nall;
    const xa = nb.xall;
    let cutforce = 0;
    for (let k = 0; k < pair.cut.length; k++) if (pair.cut[k] > cutforce) cutforce = pair.cut[k];
    const cutsq = cutforce * cutforce;
    const out = new Float64Array(3 * n);
    this.arrayAtom = out;
    const distsq = new Float64Array(nall + 1);
    const nearest = new Int32Array(nall + 1);
    const nearestN0 = new Int32Array(nall + 1);
    const chiMax = Math.max(1, nall * nall);
    const x3 = new Float64Array(chiMax), y3 = new Float64Array(chiMax), z3 = new Float64Array(chiMax);
    const x4 = new Float64Array(3), y4 = new Float64Array(3), z4 = new Float64Array(3);
    const x5 = new Float64Array(3), y5 = new Float64Array(3), z5 = new Float64Array(3);
    const x6 = new Float64Array(3), y6 = new Float64Array(3), z6 = new Float64Array(3);
    const x7 = new Float64Array(3), y7 = new Float64Array(3), z7 = new Float64Array(3);
    const x44 = new Float64Array(3), y44 = new Float64Array(3), z44 = new Float64Array(3);
    const k2 = [0, 0, 1];
    const j1 = [1, 2, 2];
    const j1b = [1, 2];
    const k2b = [0, 0];
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      // neighbours within the pair cutoff (the native full list, self excluded)
      let cnt = 0;
      for (let j = 0; j < nall; j++) {
        if (j === i) continue;
        const dx = xi - xa[3 * j], dy = yi - xa[3 * j + 1], dz = zi - xa[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq < cutsq) {
          distsq[cnt] = rsq;
          nearest[cnt++] = j;
        }
      }
      // native reads six selected neighbours; with fewer the result is undefined there, so it is refused
      if (cnt < 6) throw new StyleError(`compute ${this.id} (basal/atom): atom ${s.id[i]} has ${cnt} neighbors within the cutoff, at least 6 are needed`);
      select2(6, cnt, distsq, nearest);
      // mean squared separation of the six nearest
      let r0sq = 0;
      for (let j = 0; j < 6; j++) r0sq += distsq[j];
      r0sq /= 6.0;
      const n0dist = 1.45 * r0sq, n1dist = 1.55 * r0sq;
      let n0 = 0;
      for (let j = 0; j < cnt; j++) {
        if (distsq[j] < n1dist && distsq[j] < n0dist) nearestN0[n0++] = nearest[j];
      }
      const chi = [0];
      // the angles of all n0 pairs: x3 = (r_ik - r_ij) for each pair (j < k) with cos < -0.945
      for (let j = 0; j < n0; j++) {
        const jj = nearestN0[j];
        const xij = xi - xa[3 * jj], yij = yi - xa[3 * jj + 1], zij = zi - xa[3 * jj + 2];
        const normj = Math.sqrt(xij * xij + yij * yij + zij * zij);
        if (normj <= 0) continue;
        for (let k = j + 1; k < n0; k++) {
          const kk = nearestN0[k];
          const xik = xi - xa[3 * kk], yik = yi - xa[3 * kk + 1], zik = zi - xa[3 * kk + 2];
          const normk = Math.sqrt(xik * xik + yik * yik + zik * zik);
          if (normk <= 0) continue;
          const bondAngle = (xij * xik + yij * yik + zij * zik) / (normj * normk);
          if (-1 <= bondAngle && bondAngle < -0.945) {
            x3[chi[0]] = xik - xij;
            y3[chi[0]] = yik - yij;
            z3[chi[0]] = zik - zij;
            chi[0]++;
          }
        }
      }
      const nchi = chi[0];
      const bpv = [0, 0, 0];
      if (nchi === 2 || nchi === 3) {
        let count = 0;
        const kk2 = nchi === 2 ? k2b : k2;
        const jj1 = nchi === 2 ? j1b : j1;
        let xmean5 = 0, ymean5 = 0, zmean5 = 0, xmean6 = 0, ymean6 = 0, zmean6 = 0, xmean7 = 0, ymean7 = 0, zmean7 = 0;
        for (let j = 0; j < nchi; j++) {
          for (let k = j + 1; k < nchi; k++) {
            x4[count] = y3[jj1[count]] * z3[kk2[count]] - y3[kk2[count]] * z3[jj1[count]];
            y4[count] = z3[jj1[count]] * x3[kk2[count]] - z3[kk2[count]] * x3[jj1[count]];
            z4[count] = x3[jj1[count]] * y3[kk2[count]] - x3[kk2[count]] * y3[jj1[count]];
            x5[count] = x4[count] * copysign1(x4[count]);
            y5[count] = y4[count] * copysign1(x4[count]);
            z5[count] = z4[count] * copysign1(x4[count]);
            x6[count] = x4[count] * copysign1(y4[count]);
            y6[count] = y4[count] * copysign1(y4[count]);
            z6[count] = z4[count] * copysign1(y4[count]);
            x7[count] = x4[count] * copysign1(z4[count]);
            y7[count] = y4[count] * copysign1(z4[count]);
            z7[count] = z4[count] * copysign1(z4[count]);
            xmean5 += x5[count];
            ymean5 += y5[count];
            zmean5 += z5[count];
            xmean6 += x6[count];
            ymean6 += y6[count];
            zmean6 += z6[count];
            xmean7 += x7[count];
            ymean7 += y7[count];
            zmean6 += z7[count]; // native: accumulates into zmean6 (zmean7 stays 0)
            count++;
          }
        }
        if (count > 0) {
          xmean5 /= count; xmean6 /= count; xmean7 /= count;
          ymean5 /= count; ymean6 /= count; ymean7 /= count;
          zmean5 /= count; zmean6 /= count; zmean7 /= count;
        }
        const var5 = spread(x5, y5, z5, xmean5, ymean5, zmean5, count);
        const var6 = spread(x6, y6, z6, xmean6, ymean6, zmean6, count);
        const var7 = spread(x7, y7, z7, xmean7, ymean7, zmean7, count);
        let value: number;
        if (var5 < var6) value = var5 < var7 ? 0 : 2;
        else if (var6 < var7) value = 1;
        else value = 2;
        for (let k = 0; k < count; k++) {
          if (value === 0) { bpv[0] += x5[k]; bpv[1] += y5[k]; bpv[2] += z5[k]; }
          else if (value === 1) { bpv[0] += x6[k]; bpv[1] += y6[k]; bpv[2] += z6[k]; }
          else { bpv[0] += x7[k]; bpv[1] += y7[k]; bpv[2] += z7[k]; }
        }
      } else if (nchi > 3) {
        let S0 = 100000;
        // all combinations of three 180 degree angles
        for (let j = 0; j < nchi; j++) {
          for (let k = j + 1; k < nchi; k++) {
            for (let l = k + 1; l < nchi; l++) {
              x4[0] = x3[j]; x4[1] = x3[k]; x4[2] = x3[l];
              y4[0] = y3[j]; y4[1] = y3[k]; y4[2] = y3[l];
              z4[0] = z3[j]; z4[1] = z3[k]; z4[2] = z3[l];
              let xmean5 = 0, ymean5 = 0, zmean5 = 0, xmean6 = 0, ymean6 = 0, zmean6 = 0, xmean7 = 0, ymean7 = 0, zmean7 = 0;
              for (let m = 0; m < 3; m++) {
                x44[m] = y4[j1[m]] * z4[k2[m]] - y4[k2[m]] * z4[j1[m]];
                y44[m] = z4[j1[m]] * x4[k2[m]] - z4[k2[m]] * x4[j1[m]];
                z44[m] = x4[j1[m]] * y4[k2[m]] - x4[k2[m]] * y4[j1[m]];
                x5[m] = x44[m] * copysign1(x44[m]);
                y5[m] = y44[m] * copysign1(x44[m]);
                z5[m] = z44[m] * copysign1(x44[m]);
                x6[m] = x44[m] * copysign1(y44[m]);
                y6[m] = y44[m] * copysign1(y44[m]);
                z6[m] = z44[m] * copysign1(y44[m]);
                x7[m] = x44[m] * copysign1(z44[m]);
                y7[m] = y44[m] * copysign1(z44[m]);
                z7[m] = z44[m] * copysign1(z44[m]);
                xmean5 = xmean5 + x5[m];
                ymean5 = ymean5 + y5[m];
                zmean5 = zmean5 + z5[m];
                xmean6 = xmean6 + x6[m];
                ymean6 = ymean6 + y6[m];
                zmean6 = zmean6 + z6[m];
                xmean7 = xmean7 + x7[m];
                ymean7 = ymean7 + y7[m];
                zmean6 = zmean6 + z7[m]; // native quirk, as in the two-or-three branch
              }
              xmean5 = xmean5 / 3; xmean6 = xmean6 / 3; xmean7 = xmean7 / 3;
              ymean5 = ymean5 / 3; ymean6 = ymean6 / 3; ymean7 = ymean7 / 3;
              zmean5 = zmean5 / 3; zmean6 = zmean6 / 3; zmean7 = zmean7 / 3;
              const var5 = spread(x5, y5, z5, xmean5, ymean5, zmean5, 3);
              const var6 = spread(x6, y6, z6, xmean6, ymean6, zmean6, 3);
              const var7 = spread(x7, y7, z7, xmean7, ymean7, zmean7, 3);
              // native: the y component of this branch reads x5[2] (x6[2], x7[2]): kept as it is
              if (var5 < S0) {
                S0 = var5;
                bpv[0] = (x5[0] + x5[1] + x5[2]) / 3;
                bpv[1] = (y5[0] + y5[1] + x5[2]) / 3;
                bpv[2] = (z5[0] + z5[1] + z5[2]) / 3;
              }
              if (var6 < S0) {
                S0 = var6;
                bpv[0] = (x6[0] + x6[1] + x6[2]) / 3;
                bpv[1] = (y6[0] + y6[1] + x6[2]) / 3;
                bpv[2] = (z6[0] + z6[1] + z6[2]) / 3;
              }
              if (var7 < S0) {
                S0 = var7;
                bpv[0] = (x7[0] + x7[1] + x7[2]) / 3;
                bpv[1] = (y7[0] + y7[1] + x7[2]) / 3;
                bpv[2] = (z7[0] + z7[1] + z7[2]) / 3;
              }
            }
          }
        }
      }
      const mag = Math.sqrt(bpv[0] * bpv[0] + bpv[1] * bpv[1] + bpv[2] * bpv[2]);
      if (mag > 0) {
        bpv[0] /= mag; bpv[1] /= mag; bpv[2] /= mag;
      }
      out[3 * i] = bpv[0];
      out[3 * i + 1] = bpv[1];
      out[3 * i + 2] = bpv[2];
    }
  }
}
