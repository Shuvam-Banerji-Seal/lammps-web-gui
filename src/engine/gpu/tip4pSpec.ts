import type { Pair } from '../force/types';

/*
 * The TIP4P pair styles on the GPU (gpu/tip4pKernels.ts, webgpuForces.ts computeTip4p): lj/cut/tip4p/long,
 * lj/cut/tip4p/cut, tip4p/long and tip4p/cut. docs.lammps.org/pair_lj_cut_tip4p.rst: "The M site location is used for
 * all Coulomb interactions instead of the oxygen atom location, also with all other atom types, while the location of
 * the oxygen atom is used for the Lennard-Jones interactions." The sites and the projection of an M site's force on its
 * water (force/pair/tip4p_sites.ts: f_O += (1 - alpha) f_M, f_H1 = f_H2 += (alpha/2) f_M) are the CPU style's.
 *
 * The host builds each oxygen's M site in fp64 (buildOwnedSites) and uploads its offset from the oxygen, M - x_O (zero
 * for the other atoms). One kernel pass gathers, per atom, the Lennard-Jones force on the atom and the Coulomb force on
 * its charge site; the site separation of a pair is the atoms' minimum-image separation plus the difference of the two
 * offsets, so the sites need no image of their own. As the CPU style (and native) does, a Coulomb pair is first tested
 * on the atoms' separation against (cut_coul + 2 qdist)^2, then on the sites' against cut_coul^2. The host then adds
 * each atom's site force to the atom, or for an oxygen (1 - alpha) of it to the oxygen and alpha/2 to each hydrogen.
 * Half of each pair's energies and of its virial (r . F with the sites' separation for the Coulomb part, which equals
 * the sum of x . f over the atoms that receive the projected forces) count at each atom.
 *
 * tip4pGather is the kernel's arithmetic in fp64, which the tests compare with the CPU styles.
 */

/** What a style's gpuTip4p() returns: its tables after init (index 0 unused). */
export interface GpuTip4pData {
  ntypes: number;
  /** the Lennard-Jones term (the lj/cut/tip4p styles) */
  hasLJ: boolean;
  /** the Ewald real-space Coulomb term (the long styles), else the bare one */
  coulLong: boolean;
  /** per type pair i * nt + j: the LJ cutoff squared and the lj/cut coefficients */
  cutLJsq: ArrayLike<number>;
  lj1: ArrayLike<number>;
  lj2: ArrayLike<number>;
  lj3: ArrayLike<number>;
  lj4: ArrayLike<number>;
  offset: ArrayLike<number>;
  cutCoul: number;
  qdist: number;
  otype: number;
  htype: number;
  alpha: number;
  gEwald: number;
}

/** The styles that run this way (by name, and with a gpuTip4p() accessor). */
export const GPU_TIP4P_STYLES: ReadonlySet<string> = new Set(['lj/cut/tip4p/long', 'lj/cut/tip4p/cut', 'tip4p/long', 'tip4p/cut']);

export interface Tip4pPair extends Pair {
  /** The tables for the GPU, or null when this configuration cannot run there. */
  gpuTip4p(): GpuTip4pData | null;
}

export const isGpuTip4p = (p: Pair | null): p is Tip4pPair =>
  !!p && GPU_TIP4P_STYLES.has(p.name) && typeof (p as Partial<Tip4pPair>).gpuTip4p === 'function';

/** Floats per type pair: [cutLJsq lj1 lj2 lj3] [lj4 offset 0 0]. */
export const TIP4P_PAIR_STRIDE = 8;

export interface GpuTip4pSpec {
  /** ntypes + 1 */
  stride: number;
  coef: Float64Array;
  hasLJ: boolean;
  coulLong: boolean;
  cutCsq: number;
  /** (cut_coul + 2 qdist)^2: the atoms' separation test of a Coulomb pair */
  cutCsqPlus: number;
  /** G-ewald of the kspace style (0 for the cut styles); the backend reads it at each evaluation */
  gewald: number;
  qqrd2e: number;
  otype: number;
  htype: number;
  alpha: number;
  /** special_bonds weights by order (index 0 = 1), or null without molecular topology */
  special: { lj: number[]; coul: number[] } | null;
  maxCutoff: number;
  key: string;
}

/** The packed tables of a TIP4P style, or why the GPU cannot run it. */
export const gpuTip4pSpec = (pair: Tip4pPair, qqrd2e: number, special: { lj: number[]; coul: number[] } | null): GpuTip4pSpec | string => {
  const d = pair.gpuTip4p();
  if (!d) return `pair style ${pair.name} in this configuration`;
  const nt = d.ntypes + 1;
  const coef = new Float64Array(TIP4P_PAIR_STRIDE * nt * nt);
  let maxCut = d.cutCoul + 2 * d.qdist;
  if (d.hasLJ) {
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const t = i * nt + j, o = TIP4P_PAIR_STRIDE * t;
        coef.set([d.cutLJsq[t], d.lj1[t], d.lj2[t], d.lj3[t], d.lj4[t], d.offset[t]], o);
        if (d.cutLJsq[t] > 0) maxCut = Math.max(maxCut, Math.sqrt(d.cutLJsq[t]));
      }
    }
  }
  if (d.coulLong && !(d.gEwald > 0)) return `pair style ${pair.name} without a kspace style's G-ewald`;
  const cutCsq = d.cutCoul * d.cutCoul, cutCsqPlus = (d.cutCoul + 2 * d.qdist) ** 2;
  return {
    stride: nt, coef, hasLJ: d.hasLJ, coulLong: d.coulLong, cutCsq, cutCsqPlus, gewald: d.coulLong ? d.gEwald : 0, qqrd2e,
    otype: d.otype, htype: d.htype, alpha: d.alpha, special, maxCutoff: maxCut,
    key: `tip4p|${nt}|${d.hasLJ}|${d.coulLong}|${cutCsq}|${cutCsqPlus}|${Array.from(coef).join(',')}`,
  };
};

/** The sites of one evaluation: each atom's site offset (M - x for an oxygen, 0 otherwise) and an oxygen's hydrogens. */
export interface Tip4pSites {
  delta: Float64Array;
  h1: Int32Array;
  h2: Int32Array;
}

/**
 * Adds the gathered forces to f (state order): the atom force fA, and the site force fS on the atom, or for an oxygen
 * (1 - alpha) of it on the oxygen and alpha/2 on each of its hydrogens (tip4p_sites.ts).
 */
export const spreadSiteForces = (n: number, type: ArrayLike<number>, fA: ArrayLike<number>, fS: ArrayLike<number>, sites: Tip4pSites, spec: GpuTip4pSpec, f: Float64Array): void => {
  const wo = 1 - spec.alpha, wh = 0.5 * spec.alpha;
  for (let k = 0; k < 3 * n; k++) f[k] = fA[k];
  for (let i = 0; i < n; i++) {
    if (type[i] === spec.otype && sites.h1[i] >= 0) {
      const a = sites.h1[i], b = sites.h2[i];
      for (let m = 0; m < 3; m++) {
        const s = fS[3 * i + m];
        f[3 * i + m] += wo * s;
        f[3 * a + m] += wh * s;
        f[3 * b + m] += wh * s;
      }
    } else {
      for (let m = 0; m < 3; m++) f[3 * i + m] += fS[3 * i + m];
    }
  }
};

/** The bare and Ewald real-space Coulomb terms of one site pair: [F/r, E] (erfc by native's polynomial fit, erfc.ts). */
export const tip4pCoul = (spec: GpuTip4pSpec, rc: number, qq: number, fc: number): [number, number] => {
  const r = Math.sqrt(rc), pre = qq / r;
  if (spec.gewald > 0) {
    const gr = spec.gewald * r, ex = Math.exp(-gr * gr), t = 1 / (1 + 0.3275911 * gr);
    const erfc = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * ex;
    return [(pre * (erfc + 1.12837917 * gr * ex - (1 - fc))) / rc, pre * (erfc - (1 - fc))];
  }
  return [(fc * pre) / rc, fc * pre];
};

/** vt += s a (x) b in the order xx yy zz xy xz yz (the separation first: native v_tally3's delr[0] * f[1], ...). */
const addOuter = (vt: number[], s: number, a: ArrayLike<number>, b: ArrayLike<number>): void => {
  vt[0] += s * a[0] * b[0]; vt[1] += s * a[1] * b[1]; vt[2] += s * a[2] * b[2];
  vt[3] += s * a[0] * b[1]; vt[4] += s * a[0] * b[2]; vt[5] += s * a[1] * b[2];
};

/**
 * The kernel's arithmetic in fp64 over every atom (minimum image in an orthogonal periodic box) and the host's
 * spreading of the site forces: forces, the van der Waals and Coulomb energies, and the scalar virial. order(i, j) is
 * the special order of the pair (0: none).
 */
export const tip4pGather = (
  spec: GpuTip4pSpec, n: number, x: ArrayLike<number>, type: ArrayLike<number>, q: ArrayLike<number>, sites: Tip4pSites,
  order: (i: number, j: number) => number, L: number[],
): { f: Float64Array; evdwl: number; ecoul: number; virial: number; virialTensor: number[] } => {
  const nt = spec.stride, c = spec.coef;
  const fA = new Float64Array(3 * n), fS = new Float64Array(3 * n);
  let evdwl = 0, ecoul = 0, virial = 0;
  const vt = [0, 0, 0, 0, 0, 0];
  const maxsq = spec.maxCutoff * spec.maxCutoff;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const d = [0, 1, 2].map((k) => { const u = x[3 * i + k] - x[3 * j + k]; return u - L[k] * Math.round(u / L[k]); });
      const r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (r2 >= maxsq) continue;
      const o = order(i, j);
      const flj = o && spec.special ? spec.special.lj[o] : 1, fc = o && spec.special ? spec.special.coul[o] : 1;
      if (flj === 0 && fc === 0 && spec.gewald === 0) continue;
      const k = 8 * (type[i] * nt + type[j]);
      if (spec.hasLJ && r2 < c[k]) {
        const r2inv = 1 / r2, r6inv = r2inv * r2inv * r2inv;
        const fpair = flj * r6inv * (c[k + 1] * r6inv - c[k + 2]) * r2inv;
        for (let m = 0; m < 3; m++) fA[3 * i + m] += fpair * d[m];
        evdwl += 0.5 * flj * (r6inv * (c[k + 3] * r6inv - c[k + 4]) - c[k + 5]);
        virial += 0.5 * fpair * r2;
        addOuter(vt, 0.5 * fpair, d, d);
      }
      if (q[i] === 0 || q[j] === 0 || r2 >= spec.cutCsqPlus) continue;
      const e = [0, 1, 2].map((m) => d[m] + sites.delta[3 * i + m] - sites.delta[3 * j + m]);
      const rc = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
      if (rc >= spec.cutCsq) continue;
      const [cf, ce] = tip4pCoul(spec, rc, spec.qqrd2e * q[i] * q[j], fc);
      for (let m = 0; m < 3; m++) fS[3 * i + m] += cf * e[m];
      ecoul += 0.5 * ce;
      virial += 0.5 * cf * rc;
      addOuter(vt, 0.5 * cf, e, e);
    }
  }
  const f = new Float64Array(3 * n);
  spreadSiteForces(n, type, fA, fS, sites, spec, f);
  return { f, evdwl, ecoul, virial, virialTensor: vt };
};
