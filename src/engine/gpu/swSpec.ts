import type { Pair } from '../force/types';

/*
 * pair sw and sw/mod on the GPU (gpu/swKernels.ts, webgpuForces.ts computeSW). The Stillinger-Weber energy
 * (docs.lammps.org/pair_sw.html: "E = sum_i sum_j>i phi2(r_ij) + sum_i sum_j!=i sum_k>j phi3(r_ij, r_ik, theta_jik)",
 * phi2 = A eps (B (sigma/r)^p - (sigma/r)^q) exp(sigma / (r - a sigma)), phi3 = lambda eps (cos theta_jik - cos theta0)^2
 * exp(gamma sigma_ij / (r_ij - a_ij sigma_ij)) exp(gamma sigma_ik / (r_ik - a_ik sigma_ik))) in a gather form: one
 * invocation per atom adds the forces on its own atom, from its pairs, from the triplets it is the center of, and from
 * the triplets centered on a neighbor in which it is a leg, so no atomics are needed. A triplet's energy and virial are
 * counted at its center. sw/mod scales cos theta - cos theta0 by its switching factor f_C (pair_sw.html, sw/mod).
 *
 * The parameters come from the engine's styles (force/pair/sw.ts PairSWBase for sw/mod, the GPL port of sw) through
 * their gpuSW() accessor, as the same arrays: per type pair i * nt + j the two-body entry (I,J,J) and the three-body leg
 * of entry (I,J,J); per type triplet (i * nt + j) * nt + k the angular entry (I,J,K). swGather is the kernels'
 * arithmetic in fp64, which the tests compare with the CPU styles.
 */

/** What a style's gpuSW() returns: its parameters after init, by type (index 0 unused). */
export interface GpuSwData {
  ntypes: number;
  /** the type takes part (an element mapped to it; NULL types interact with nothing) */
  active: ArrayLike<boolean>;
  /** per type pair: the neighbor cutoff squared, the two-body term's cutoff, and its parameters */
  cutsq: ArrayLike<number>;
  cut2: ArrayLike<number>;
  eps: ArrayLike<number>;
  sigma: ArrayLike<number>;
  a: ArrayLike<number>;
  A: ArrayLike<number>;
  B: ArrayLike<number>;
  p: ArrayLike<number>;
  q: ArrayLike<number>;
  /** per type pair, the leg (center i, neighbor j): its cutoff and gamma sigma of the screening factor */
  legCut: ArrayLike<number>;
  gammaSigma: ArrayLike<number>;
  /** per type triplet: lambda eps and cos theta0 */
  lambdaEps: ArrayLike<number>;
  cos0: ArrayLike<number>;
  threebody: boolean;
  /** sw/mod's maxdelcs (f_C switches from delta1 to delta2), or null for sw */
  mod: { delta1: number; delta2: number } | null;
}

/** The styles that run this way (by name, and with a gpuSW() accessor). */
export const GPU_SW_STYLES: ReadonlySet<string> = new Set(['sw', 'sw/mod']);

export interface SwPair extends Pair {
  /** The parameters for the GPU, or null when this configuration cannot run there. */
  gpuSW(): GpuSwData | null;
}

export const isGpuSw = (p: Pair | null): p is SwPair =>
  !!p && GPU_SW_STYLES.has(p.name) && typeof (p as Partial<SwPair>).gpuSW === 'function';

/** Floats per type pair: [cutsq cut2^2 legCut gammaSigma] [A*eps B p q] [sigma a*sigma 0 0]. */
export const SW_PAIR_STRIDE = 12;
/** Floats per type triplet: lambda*eps, cos theta0. */
export const SW_TRIP_STRIDE = 2;

export interface GpuSwSpec {
  /** ntypes + 1 */
  stride: number;
  /** the type pairs' block, then the triplets' (at SW_PAIR_STRIDE * stride^2) */
  coef: Float64Array;
  tripBase: number;
  maxCutoff: number;
  threebody: boolean;
  mod: { delta1: number; delta2: number } | null;
  key: string;
}

/** The packed parameters of an sw or sw/mod style, or why the GPU cannot run it. */
export const gpuSwSpec = (pair: SwPair): GpuSwSpec | string => {
  const d = pair.gpuSW();
  if (!d) return `pair style ${pair.name} in this configuration`;
  const nt = d.ntypes + 1;
  const tripBase = SW_PAIR_STRIDE * nt * nt;
  const coef = new Float64Array(tripBase + SW_TRIP_STRIDE * nt * nt * nt);
  let maxCut = 0;
  for (let i = 1; i < nt; i++) {
    for (let j = 1; j < nt; j++) {
      const t = i * nt + j, o = SW_PAIR_STRIDE * t;
      if (!d.active[i] || !d.active[j]) continue;
      coef[o] = d.cutsq[t];
      coef[o + 1] = d.cut2[t] * d.cut2[t];
      coef[o + 2] = d.threebody ? d.legCut[t] : 0;
      coef[o + 3] = d.gammaSigma[t];
      coef[o + 4] = d.A[t] * d.eps[t];
      coef[o + 5] = d.B[t];
      coef[o + 6] = d.p[t];
      coef[o + 7] = d.q[t];
      coef[o + 8] = d.sigma[t];
      coef[o + 9] = d.a[t] * d.sigma[t];
      if (d.cutsq[t] > 0) maxCut = Math.max(maxCut, Math.sqrt(d.cutsq[t]));
      for (let k = 1; k < nt; k++) {
        if (!d.active[k]) continue;
        const t3 = t * nt + k, t3s = (i * nt + k) * nt + j;
        // the kernels take a triplet's two legs in any order: the entries (I,J,K) and (I,K,J) must agree (native reads
        // (I,J,K) with j the earlier neighbor, so asymmetric entries depend on the neighbor order there)
        if (d.lambdaEps[t3] !== d.lambdaEps[t3s] || d.cos0[t3] !== d.cos0[t3s]) {
          return `pair style ${pair.name} with asymmetric three-body entries (I,J,K) and (I,K,J)`;
        }
        coef[tripBase + SW_TRIP_STRIDE * t3] = d.lambdaEps[t3];
        coef[tripBase + SW_TRIP_STRIDE * t3 + 1] = d.cos0[t3];
      }
    }
  }
  if (!(maxCut > 0)) return `pair style ${pair.name} without interactions`;
  const mod = d.mod;
  return {
    stride: nt, coef, tripBase, maxCutoff: maxCut, threebody: d.threebody, mod,
    key: `sw|${nt}|${d.threebody}|${mod ? `${mod.delta1},${mod.delta2}` : ''}|${Array.from(coef).join(',')}`,
  };
};

/** sw/mod's f_C(delta) and its derivative (pair_sw.html: 1 below delta1, 0 above delta2, a cosine between). */
const switchFc = (mod: { delta1: number; delta2: number } | null, dc: number): [number, number] => {
  if (!mod) return [1, 0];
  const ad = Math.abs(dc);
  if (ad <= mod.delta1) return [1, 0];
  if (ad >= mod.delta2) return [0, 0];
  const wd = mod.delta2 - mod.delta1, arg = (Math.PI * (ad - mod.delta1)) / wd;
  const der = -(Math.PI / (2 * wd)) * Math.sin(arg);
  return [0.5 + 0.5 * Math.cos(arg), dc < 0 ? -der : der];
};

/**
 * The two kernels' arithmetic in fp64 over every atom (minimum image in an orthogonal periodic box): forces, energy
 * and the scalar virial sum_pairs r . F, in the gather form described above.
 */
export const swGather = (spec: GpuSwSpec, n: number, x: ArrayLike<number>, type: ArrayLike<number>, L: number[]): { f: Float64Array; e: number; virial: number } => {
  const nt = spec.stride, c = spec.coef;
  const dvec = (i: number, j: number): [number, number, number] => {
    const v: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 3; k++) { const u = x[3 * j + k] - x[3 * i + k]; v[k] = u - L[k] * Math.round(u / L[k]); }
    return v;
  };
  // pass 1: the neighbor lists
  const nbr: number[][] = [];
  for (let i = 0; i < n; i++) {
    const list: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = dvec(i, j), r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (r2 < c[SW_PAIR_STRIDE * (type[i] * nt + type[j])]) list.push(j);
    }
    nbr.push(list);
  }
  const leg = (i: number, j: number) => {
    const o = SW_PAIR_STRIDE * (type[i] * nt + type[j]);
    const d = dvec(i, j), r = Math.hypot(d[0], d[1], d[2]);
    const lc = c[o + 2];
    if (!(r < lc)) return null;
    const den = r - lc, gs = c[o + 3];
    return { d, r, es: Math.exp(gs / den), w: gs / (den * den) };
  };
  /** forces on legs 1 and 2 of the triplet centered at ctr, and its energy */
  const triplet = (ctr: number, j: number, k: number) => {
    const l1 = leg(ctr, j), l2 = leg(ctr, k);
    if (!l1 || !l2) return null;
    const t3 = (type[ctr] * nt + type[j]) * nt + type[k];
    const P = c[spec.tripBase + SW_TRIP_STRIDE * t3], c0 = c[spec.tripBase + SW_TRIP_STRIDE * t3 + 1];
    if (P === 0) return null;
    const cs = (l1.d[0] * l2.d[0] + l1.d[1] * l2.d[1] + l1.d[2] * l2.d[2]) / (l1.r * l2.r);
    const dc = cs - c0;
    const [fc, fcd] = switchFc(spec.mod, dc);
    const zeta = fc * dc;
    const aE = P * zeta * zeta, aD = P * 2 * zeta * (fc + dc * fcd);
    const esPair = l1.es * l2.es, e3 = aE * esPair, dEdc = aD * esPair;
    const bk = dEdc / (l1.r * l2.r);
    const a1 = (dEdc * cs) / (l1.r * l1.r) + (e3 * l1.w) / l1.r;
    const a2 = (dEdc * cs) / (l2.r * l2.r) + (e3 * l2.w) / l2.r;
    const f1 = [0, 1, 2].map((m) => a1 * l1.d[m] - bk * l2.d[m]);
    const f2 = [0, 1, 2].map((m) => a2 * l2.d[m] - bk * l1.d[m]);
    return { f1, f2, e3, w: f1[0] * l1.d[0] + f1[1] * l1.d[1] + f1[2] * l1.d[2] + f2[0] * l2.d[0] + f2[1] * l2.d[1] + f2[2] * l2.d[2] };
  };
  const f = new Float64Array(3 * n);
  let e = 0, virial = 0;
  for (let i = 0; i < n; i++) {
    const ti = type[i];
    for (const j of nbr[i]) {
      const o = SW_PAIR_STRIDE * (ti * nt + type[j]);
      const d = dvec(i, j), r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      const r = Math.sqrt(r2), den = r - c[o + 9];
      // the kernel's test: inside the cutoff by r^2 and by r (see swKernels.ts)
      if (r2 < c[o + 1] && den < 0) {
        const sig = c[o + 8];
        const u = sig / r, es = Math.exp(sig / den);
        const B = c[o + 5], p = c[o + 6], q = c[o + 7];
        const f2 = B * u ** p - u ** q;
        const dfdu = (p > 0 ? B * p * u ** (p - 1) : 0) - (q > 0 ? q * u ** (q - 1) : 0);
        const dphi = c[o + 4] * es * (-dfdu * sig / r2 - f2 * sig / (den * den));
        const fpair = -dphi / r;
        for (let m = 0; m < 3; m++) f[3 * i + m] -= fpair * d[m];
        e += 0.5 * c[o + 4] * f2 * es;
        virial += 0.5 * fpair * r2;
      }
    }
    if (!spec.threebody) continue;
    // triplets centered at i
    const L1 = nbr[i];
    for (let a = 0; a < L1.length; a++) {
      for (let b = a + 1; b < L1.length; b++) {
        const t = triplet(i, L1[a], L1[b]);
        if (!t) continue;
        for (let m = 0; m < 3; m++) f[3 * i + m] -= t.f1[m] + t.f2[m];
        e += t.e3;
        virial += t.w;
      }
    }
    // triplets centered at a neighbor j with i as a leg
    for (const j of nbr[i]) {
      for (const k of nbr[j]) {
        if (k === i) continue;
        const t = triplet(j, i, k);
        if (!t) continue;
        for (let m = 0; m < 3; m++) f[3 * i + m] += t.f1[m];
      }
    }
  }
  return { f, e, virial };
};
