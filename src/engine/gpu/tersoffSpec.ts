import type { Pair } from '../force/types';

/*
 * pair tersoff on the GPU (gpu/tersoffKernels.ts, webgpuForces.ts computeTersoff). The Tersoff energy of
 * docs.lammps.org/pair_tersoff.html (E = 1/2 sum_i sum_j!=i V_ij, V_ij = f_C [f_R + b_ij f_A] at r_ij + delta, b_ij from
 * the bond-order sum zeta_ij over third atoms k; the doc's formulas are quoted verbatim in force/pair/tersoff.ts) as a
 * gather in three passes, so no atomics are needed:
 *   1. each atom's neighbors within the list cutoff of its type pair;
 *   2. per bond i -> j (a slot of i's list): b_ij and C_ij = 1/2 f_C f_A db/dzeta, the prefactor of the forces from the
 *      derivatives of zeta_ij (b = -1 marks a bond outside the pair cutoff);
 *   3. per atom, the forces of its own bonds (the radial term and the zeta terms it centers), of its neighbors' bonds
 *      toward it (the radial term and the zeta terms where it is the bond's far atom), and of its neighbors' other bonds
 *      in whose zeta it is the third atom. Half of each ordered pair's energy and its virial are counted at the center.
 * The neighbor lists must be symmetric (j in i's list exactly when i is in j's) for step 3 to find every bond that
 * involves an atom, so the list cutoff of a type pair is the larger of its two orders.
 *
 * The angular term g = gamma (1 + c^2/d^2 - c^2/(d^2 + (cos theta - cos theta0)^2)) is evaluated as gamma (1 + c^2 q^2 /
 * (d^2 (d^2 + q^2))), q = cos theta - cos theta0, the same value without the difference of two numbers near c^2/d^2 (the
 * Si and C parameters have c^2/d^2 ~ 4e7, which in fp32 leaves g ~ 1e3 with an error of a few units).
 *
 * The bond order is evaluated from u = ln((beta zeta)^n) = n ln beta + n ln zeta: b = (1 + e^u)^(-1/2n) when u <= 0, and
 * e^(-u/2n) (1 + e^-u)^(-1/2n) when u > 0 (the same value), with db/dzeta = -1/2 b s / zeta, s = e^u / (1 + e^u). The
 * direct beta^n zeta^n overflows fp32 for the Si(B) set (n = 22.956: zeta above about 48, reached in a 4096-atom run at
 * 1600 K by a bond at the cutoff beside a short one, gave zeta^n = inf, b = 0 and db/dzeta = 0 * inf = NaN). The exponent
 * of zeta's exp[lambda3^m (r_ij - r_ik)^m] is capped at 69.0776 as native LAMMPS does (src/MANYBODY/pair_tersoff.cpp
 * zeta() at b332f28: "if (arg > 69.0776) ex_delr = 1.e30"), so that it stays finite in fp32.
 *
 * The parameters come from the engine's style (force/pair/tersoff.ts) through its gpuTersoff() accessor; tersoffGather
 * is the kernels' arithmetic in fp64, which the tests compare with the CPU style.
 */

/** What a style's gpuTersoff() returns: its tables after init (index 0 unused). */
export interface GpuTersoffData {
  ntypes: number;
  /** the type takes part (an element mapped to it; NULL types interact with nothing) */
  active: ArrayLike<boolean>;
  /** pair_style tersoff shift delta */
  shift: number;
  /** per type pair i * nt + j, from the (I,J,J) entry */
  A: ArrayLike<number>;
  B: ArrayLike<number>;
  lam1: ArrayLike<number>;
  lam2: ArrayLike<number>;
  beta: ArrayLike<number>;
  n: ArrayLike<number>;
  R: ArrayLike<number>;
  D: ArrayLike<number>;
  cut2: ArrayLike<number>;
  /** per type triplet (i * nt + j) * nt + k, from the (I,J,K) entry: m, gamma, lambda3^m, c, d, cos theta0, R, D */
  m: ArrayLike<number>;
  gamma: ArrayLike<number>;
  lam3p: ArrayLike<number>;
  c: ArrayLike<number>;
  d: ArrayLike<number>;
  ct0: ArrayLike<number>;
  R3: ArrayLike<number>;
  D3: ArrayLike<number>;
  cut3: ArrayLike<number>;
  valid3: ArrayLike<number>;
}

/** The styles that run this way (by name, and with a gpuTersoff() accessor). */
export const GPU_TERSOFF_STYLES: ReadonlySet<string> = new Set(['tersoff']);

export interface TersoffPair extends Pair {
  /** The tables for the GPU, or null when this configuration cannot run there. */
  gpuTersoff(): GpuTersoffData | null;
}

export const isGpuTersoff = (p: Pair | null): p is TersoffPair =>
  !!p && GPU_TERSOFF_STYLES.has(p.name) && typeof (p as Partial<TersoffPair>).gpuTersoff === 'function';

/** Floats per type pair: [listCutSq cut2 R D] [A lambda1 B lambda2] [n ln beta, n, 0 0] (n ln beta = -1e30 for beta = 0). */
export const TERSOFF_PAIR_STRIDE = 12;
/** Floats per type triplet: [cut3 R D gamma] [c^2 d^2 cos theta0 lambda3^m] [m 0 0 0] (cut3 = 0: no such triplet). */
export const TERSOFF_TRIP_STRIDE = 12;

export interface GpuTersoffSpec {
  /** ntypes + 1 */
  stride: number;
  /** the type pairs' block, then the triplets' (at tripBase) */
  coef: Float64Array;
  tripBase: number;
  shift: number;
  maxCutoff: number;
  key: string;
}

/** The packed tables of a tersoff style, or why the GPU cannot run it. */
export const gpuTersoffSpec = (pair: TersoffPair): GpuTersoffSpec | string => {
  const d = pair.gpuTersoff();
  if (!d) return `pair style ${pair.name} in this configuration`;
  const nt = d.ntypes + 1;
  const tripBase = TERSOFF_PAIR_STRIDE * nt * nt;
  const coef = new Float64Array(tripBase + TERSOFF_TRIP_STRIDE * nt * nt * nt);
  // the list cutoff of the ordered type pair (i, x): the pair cutoff of (i, x) and the third-atom cutoff of every (i, j, x)
  const own = new Float64Array(nt * nt);
  for (let i = 1; i < nt; i++) {
    for (let x = 1; x < nt; x++) {
      if (!d.active[i] || !d.active[x]) continue;
      let c = d.cut2[i * nt + x];
      for (let j = 1; j < nt; j++) {
        const t3 = (i * nt + j) * nt + x;
        if (d.active[j] && d.valid3[t3]) c = Math.max(c, d.cut3[t3]);
      }
      own[i * nt + x] = c;
    }
  }
  let maxCut = 0;
  for (let i = 1; i < nt; i++) {
    for (let j = 1; j < nt; j++) {
      const t = i * nt + j, o = TERSOFF_PAIR_STRIDE * t;
      if (!d.active[i] || !d.active[j]) continue;
      const lc = Math.max(own[t], own[j * nt + i]);
      maxCut = Math.max(maxCut, lc);
      coef[o] = lc * lc;
      coef[o + 1] = d.cut2[t];
      coef[o + 2] = d.R[t];
      coef[o + 3] = d.D[t];
      coef[o + 4] = d.A[t];
      coef[o + 5] = d.lam1[t];
      coef[o + 6] = d.B[t];
      coef[o + 7] = d.lam2[t];
      coef[o + 8] = d.beta[t] > 0 ? d.n[t] * Math.log(d.beta[t]) : -1e30;
      coef[o + 9] = d.n[t];
      for (let k = 1; k < nt; k++) {
        const t3 = t * nt + k, q = tripBase + TERSOFF_TRIP_STRIDE * t3;
        if (!d.active[k] || !d.valid3[t3]) continue;
        if (d.m[t3] !== 1 && d.m[t3] !== 3) return `pair style ${pair.name} with m other than 1 or 3`;
        coef[q] = d.cut3[t3];
        coef[q + 1] = d.R3[t3];
        coef[q + 2] = d.D3[t3];
        coef[q + 3] = d.gamma[t3];
        coef[q + 4] = d.c[t3] * d.c[t3];
        coef[q + 5] = d.d[t3] * d.d[t3];
        coef[q + 6] = d.ct0[t3];
        coef[q + 7] = d.lam3p[t3];
        coef[q + 8] = d.m[t3];
      }
    }
  }
  if (!(maxCut > 0)) return `pair style ${pair.name} without interactions`;
  return { stride: nt, coef, tripBase, shift: d.shift, maxCutoff: maxCut, key: `tersoff|${nt}|${d.shift}|${Array.from(coef).join(',')}` };
};

/** The cap of zeta's exponent (native's 69.0776). */
const EXP_CAP = 69.0776;

/** b and db/dzeta for zeta > 0 from u = n ln beta + n ln zeta (see the header). */
const bondOrder = (nlb: number, nn: number, zeta: number): [number, number] => {
  const u = nlb + nn * Math.log(zeta);
  let b: number, sfac: number;
  if (u <= 0) {
    const t = Math.exp(u);
    b = Math.pow(1 + t, -1 / (2 * nn));
    sfac = t / (1 + t);
  } else {
    const t = Math.exp(-u);
    b = Math.exp(-u / (2 * nn)) * Math.pow(1 + t, -1 / (2 * nn));
    sfac = 1 / (1 + t);
  }
  return [b, (-0.5 * b * sfac) / zeta];
};

/** f_C(r) and its derivative with cutoff R, D (1/2 - 1/2 sin(pi/2 (r - R)/D) between R - D and R + D). */
const cutoff = (r: number, R: number, D: number): [number, number] => {
  if (r < R - D) return [1, 0];
  if (r > R + D) return [0, 0];
  const a = (Math.PI / 2) * (r - R) / D;
  return [0.5 - 0.5 * Math.sin(a), -(Math.PI / (4 * D)) * Math.cos(a)];
};

/**
 * The three kernels' arithmetic in fp64 over every atom (minimum image in an orthogonal periodic box): forces, energy
 * and the scalar virial (the sum of r . F over the pair and three-body terms), in the gather form described above.
 */
export const tersoffGather = (spec: GpuTersoffSpec, n: number, x: ArrayLike<number>, type: ArrayLike<number>, L: number[]): { f: Float64Array; e: number; virial: number } => {
  const nt = spec.stride, c = spec.coef, sft = spec.shift;
  const PS = TERSOFF_PAIR_STRIDE, TS = TERSOFF_TRIP_STRIDE;
  const dvec = (i: number, j: number): number[] => {
    const v = [0, 0, 0];
    for (let k = 0; k < 3; k++) { const u = x[3 * j + k] - x[3 * i + k]; v[k] = u - L[k] * Math.round(u / L[k]); }
    return v;
  };
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  // pass 1: the neighbor lists
  const nbr: number[][] = [];
  for (let i = 0; i < n; i++) {
    const list: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const r2 = dot(dvec(i, j), dvec(i, j));
      if (r2 > 0 && r2 < c[PS * (type[i] * nt + type[j])]) list.push(j);
    }
    nbr.push(list);
  }
  /** the term of atom k in zeta of the bond ctr -> j (unit vector e1, length r1), and its partial derivatives */
  const zterm = (ctr: number, j: number, k: number, e1: number[], r1: number) => {
    const q = spec.tripBase + TS * ((type[ctr] * nt + type[j]) * nt + type[k]);
    const d2 = dvec(ctr, k), r2 = Math.sqrt(dot(d2, d2));
    if (!(r2 < c[q]) || r2 === 0) return null;
    const e2 = d2.map((u) => u / r2);
    const [fc, fcp] = cutoff(r2 + sft, c[q + 1], c[q + 2]);
    const cth = dot(e1, e2), qq = cth - c[q + 6], c2 = c[q + 4], dd = c[q + 5], den = dd + qq * qq, gam = c[q + 3];
    const g = gam * (1 + (c2 * qq * qq) / (dd * den));
    const dgdc = (2 * gam * c2 * qq) / (den * den);
    const dr = r1 - r2, lp = c[q + 7];
    let efil = 1, dphi = 0;
    if (lp !== 0) {
      if (c[q + 8] === 3) { efil = Math.exp(Math.min(lp * dr * dr * dr, EXP_CAP)); dphi = 3 * lp * dr * dr; } else { efil = Math.exp(Math.min(lp * dr, EXP_CAP)); dphi = lp; }
    }
    const H = fc * g * efil;
    return { H, drij: H * dphi, drik: g * efil * fcp - H * dphi, ddc: fc * efil * dgdc, e2, d2, r2 };
  };
  /** F_j and F_k of a zeta term with prefactor C */
  const zforces = (C: number, e1: number[], r1: number, z: NonNullable<ReturnType<typeof zterm>>) => {
    const cth = dot(e1, z.e2);
    const fj = [0, 1, 2].map((m) => -C * (z.drij * e1[m] + (z.ddc * (z.e2[m] - cth * e1[m])) / r1));
    const fk = [0, 1, 2].map((m) => -C * (z.drik * z.e2[m] + (z.ddc * (e1[m] - cth * z.e2[m])) / z.r2));
    return { fj, fk };
  };
  /** V_ij and dV_ij/dr at fixed b */
  const radial = (ti: number, tj: number, r: number, b: number): [number, number] => {
    const o = PS * (ti * nt + tj), rs = r + sft;
    const [fc, fcp] = cutoff(rs, c[o + 2], c[o + 3]);
    const fr = c[o + 4] * Math.exp(-c[o + 5] * rs), fa = -c[o + 6] * Math.exp(-c[o + 7] * rs);
    return [fc * (fr + b * fa), fcp * (fr + b * fa) + fc * (-c[o + 5] * fr - b * c[o + 7] * fa)];
  };
  // pass 2: b and C per bond (b = -1 outside the pair cutoff)
  const bo: number[][] = [], bc: number[][] = [];
  for (let i = 0; i < n; i++) {
    const L1 = nbr[i], bi: number[] = [], ci: number[] = [];
    for (let a = 0; a < L1.length; a++) {
      const j = L1[a], o = PS * (type[i] * nt + type[j]);
      const d1 = dvec(i, j), r1 = Math.sqrt(dot(d1, d1));
      if (!(r1 < c[o + 1])) { bi.push(-1); ci.push(0); continue; }
      const e1 = d1.map((u) => u / r1);
      let zeta = 0;
      for (let b = 0; b < L1.length; b++) {
        if (b === a) continue;
        const z = zterm(i, j, L1[b], e1, r1);
        if (z) zeta += z.H;
      }
      let b = 1, dbdz = 0;
      if (zeta > 0) [b, dbdz] = bondOrder(c[o + 8], c[o + 9], zeta);
      const rs = r1 + sft;
      const fa = -c[o + 6] * Math.exp(-c[o + 7] * rs);
      bi.push(b); ci.push(0.5 * cutoff(rs, c[o + 2], c[o + 3])[0] * fa * dbdz);
    }
    bo.push(bi); bc.push(ci);
  }
  // pass 3: the forces gathered per atom
  const f = new Float64Array(3 * n);
  let e = 0, virial = 0;
  for (let i = 0; i < n; i++) {
    const ti = type[i], L1 = nbr[i];
    const fi = [0, 0, 0];
    for (let a = 0; a < L1.length; a++) {
      if (bo[i][a] < 0) continue;
      const j = L1[a], d1 = dvec(i, j), r1 = Math.sqrt(dot(d1, d1)), e1 = d1.map((u) => u / r1);
      const [v, dRad] = radial(ti, type[j], r1, bo[i][a]);
      for (let m = 0; m < 3; m++) fi[m] += 0.5 * dRad * e1[m];
      e += 0.5 * v;
      virial -= 0.5 * dRad * r1;
      const C = bc[i][a];
      if (C === 0) continue;
      for (let b = 0; b < L1.length; b++) {
        if (b === a) continue;
        const z = zterm(i, j, L1[b], e1, r1);
        if (!z) continue;
        const { fj, fk } = zforces(C, e1, r1, z);
        for (let m = 0; m < 3; m++) fi[m] -= fj[m] + fk[m];
        virial += dot(d1, fj) + dot(z.d2, fk);
      }
    }
    for (const j of L1) {
      const L2 = nbr[j];
      for (let cc = 0; cc < L2.length; cc++) {
        if (bo[j][cc] < 0) continue;
        const l = L2[cc], dl = dvec(j, l), rl = Math.sqrt(dot(dl, dl)), el = dl.map((u) => u / rl);
        const C = bc[j][cc];
        if (l === i) {
          const dRad = radial(type[j], ti, rl, bo[j][cc])[1];
          for (let m = 0; m < 3; m++) fi[m] -= 0.5 * dRad * el[m];
          if (C === 0) continue;
          for (let c2 = 0; c2 < L2.length; c2++) {
            if (c2 === cc) continue;
            const z = zterm(j, i, L2[c2], el, rl);
            if (!z) continue;
            const { fj } = zforces(C, el, rl, z);
            for (let m = 0; m < 3; m++) fi[m] += fj[m];
          }
        } else if (C !== 0) {
          const z = zterm(j, l, i, el, rl);
          if (!z) continue;
          const { fk } = zforces(C, el, rl, z);
          for (let m = 0; m < 3; m++) fi[m] += fk[m];
        }
      }
    }
    f[3 * i] = fi[0]; f[3 * i + 1] = fi[1]; f[3 * i + 2] = fi[2];
  }
  return { f, e, virial };
};
