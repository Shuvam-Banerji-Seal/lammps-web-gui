import type { Pair } from '../force/types';

/*
 * The pair styles the WebGPU backend runs besides plain lj/cut (gpu/webgpuForces.ts computeSpec): their coefficients,
 * taken from the engine's own style after its init (the arrays its compute() reads), packed per type pair for the WGSL
 * kernel, and the same force law in fp64 (pairSpecEval) for the tests. Each law is the style's documented one, written
 * as the engine's CPU style computes it (force/pair/lj_coul.ts, coul.ts, simple.ts); pairs are unweighted (the GPU path
 * runs systems without molecular topology, so no special-bond factors).
 */

/** Style codes of the kernel (WebGPU uniform P.style). */
export const GPU_PAIR_STYLES: Readonly<Record<string, number>> = {
  'lj/cut/coul/cut': 1,
  'coul/cut': 2,
  buck: 3,
  'buck/coul/cut': 4,
  morse: 5,
  soft: 6,
  yukawa: 7,
  'lj96/cut': 8,
  born: 9,
  // lj/cut of a system with molecular topology (special bonds): the analytic lj/cut kernel has no special weights
  'lj/cut': 10,
};

/** Floats per type pair: [cutsqA cutsqC a b] [c d eshift cscale] [e _ _ _] (three vec4). */
export const SPEC_STRIDE = 12;

export interface GpuPairSpec {
  style: number;
  /** ntypes + 1: type pair (i, j) is entry i * stride + j. */
  stride: number;
  coef: Float64Array;
  maxCutoff: number;
  /** The style has a Coulomb term: the kernel reads the charges. */
  charges: boolean;
  qqrd2e: number;
  /**
   * special_bonds weights of 1-2, 1-3 and 1-4 pairs (index 1..3; index 0 is 1) for a system with molecular topology,
   * else null: the kernel scales such a pair's terms, and skips it when both are 0, as the neighbor list does.
   */
  special: { lj: number[]; coul: number[] } | null;
  /** Changes when the coefficients do (the backend caches the packed buffer by it). */
  key: string;
}

type Arr = Float64Array;
type Internals = Pair & {
  lj1: Arr; lj2: Arr; lj3: Arr; lj4: Arr; offset: Arr; cutsq: Arr; scale: Arr;
  buckA: Arr; buckIR: Arr; buckC: Arr; buckCutSq: Arr; coulCutSq: Arr;
  morseD0: Arr; morseAlpha: Arr; morseR0: Arr;
  pa: Arr; pk: Arr; kappa: number;
  bornA: Arr; bornIR: Arr; bornSig: Arr; bornC: Arr; bornD: Arr;
  p: { get(name: string, i: number, j: number): number };
  coulCut(i: number, j: number): number;
};

/**
 * The spec of an initialized pair style (pair.init() has run), or why the GPU cannot run it. qqrd2e is the Coulomb
 * prefactor of the force evaluation (force/forcefield.ts: units qqr2e times the style's scale over the dielectric).
 */
export const gpuPairSpec = (pair: Pair, ntypes: number, qqrd2e: number, special: { lj: number[]; coul: number[] } | null = null): GpuPairSpec | string => {
  const style = GPU_PAIR_STYLES[pair.name];
  if (style === undefined) return `pair style ${pair.name}`;
  const st = pair as Internals;
  const nt = ntypes + 1;
  const coef = new Float64Array(SPEC_STRIDE * nt * nt);
  let maxCut = 0;
  for (let i = 1; i < nt; i++) {
    for (let j = 1; j < nt; j++) {
      const t = i * nt + j;
      const o = SPEC_STRIDE * t;
      let cutA = 0, cutC = 0, cscale = 1;
      switch (style) {
        case 10: // lj/cut (with special bonds): lj1..lj4 and offset, inside the pair cutoff
          cutA = st.cutsq[t];
          coef.set([st.lj1[t], st.lj2[t], st.lj3[t], st.lj4[t], st.offset[t]], o + 2);
          break;
        case 1: // lj/cut/coul/cut: lj1..lj4 and offset of lj/cut; the Coulomb cutoff as its compute() bounds it
          cutA = st.p.get('cut', i, j) ** 2;
          cutC = Math.min(st.cutsq[t], st.coulCut(i, j) ** 2);
          coef.set([st.lj1[t], st.lj2[t], st.lj3[t], st.lj4[t], st.offset[t]], o + 2);
          break;
        case 2: // coul/cut: the pair cutoff, and fix adapt's scale
          cutC = st.cutsq[t];
          cscale = st.scale[t];
          break;
        case 3: // buck
          cutA = st.cutsq[t];
          coef.set([st.buckA[t], st.buckIR[t], st.buckC[t], 0, st.offset[t]], o + 2);
          break;
        case 4: // buck/coul/cut
          cutA = st.buckCutSq[t];
          cutC = st.coulCutSq[t];
          coef.set([st.buckA[t], st.buckIR[t], st.buckC[t], 0, st.offset[t]], o + 2);
          break;
        case 5: // morse
          cutA = st.cutsq[t];
          coef.set([st.morseD0[t], st.morseAlpha[t], st.morseR0[t], 0, st.offset[t]], o + 2);
          break;
        case 6: // soft: A and pi / rc (no shift: the energy is already 0 at rc)
          cutA = st.cutsq[t];
          coef.set([st.pa[t], st.pk[t]], o + 2);
          break;
        case 7: // yukawa: A and the global kappa
          cutA = st.cutsq[t];
          coef.set([st.pa[t], st.kappa, 0, 0, st.offset[t]], o + 2);
          break;
        case 8: // lj96/cut: lj1..lj4 and offset of its 9-6 law
          cutA = st.cutsq[t];
          coef.set([st.lj1[t], st.lj2[t], st.lj3[t], st.lj4[t], st.offset[t]], o + 2);
          break;
        case 9: // born: A, 1/rho, sigma, C, then D in the third vec4
          cutA = st.cutsq[t];
          coef.set([st.bornA[t], st.bornIR[t], st.bornSig[t], st.bornC[t], st.offset[t]], o + 2);
          coef[o + 8] = st.bornD[t];
          break;
      }
      coef[o] = cutA;
      coef[o + 1] = cutC;
      coef[o + 7] = cscale;
      maxCut = Math.max(maxCut, Math.sqrt(Math.max(cutA, cutC)));
    }
  }
  const charges = style === 1 || style === 2 || style === 4;
  return { style, stride: nt, coef, maxCutoff: maxCut, charges, qqrd2e, special, key: `${style}|${qqrd2e}|${Array.from(coef).join(',')}` };
};

/**
 * One pair's force over r (fpair), van der Waals and Coulomb energies with the spec's law, in fp64: the kernel's
 * arithmetic (webgpuForces.ts KERNEL_STYLES), which the tests compare with the engine's CPU styles.
 */
export const pairSpecEval = (spec: GpuPairSpec, ti: number, tj: number, rsq: number, qi: number, qj: number):
  { fpair: number; evdwl: number; ecoul: number } => {
  const o = SPEC_STRIDE * (ti * spec.stride + tj);
  const c = spec.coef;
  const cutA = c[o], cutC = c[o + 1], a = c[o + 2], b = c[o + 3], cc = c[o + 4], d = c[o + 5], eshift = c[o + 6], cscale = c[o + 7];
  const e5 = c[o + 8];
  let fpair = 0, evdwl = 0, ecoul = 0;
  if (rsq < cutA) {
    if (spec.style === 1 || spec.style === 10) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
      fpair += r6inv * (a * r6inv - b) * r2inv;
      evdwl += r6inv * (cc * r6inv - d) - eshift;
    } else if (spec.style === 3 || spec.style === 4) {
      const r = Math.sqrt(rsq), ex = Math.exp(-r * b), r2inv = 1 / rsq;
      fpair += a * b * ex / r - 6 * cc * r2inv * r2inv * r2inv * r2inv;
      evdwl += a * ex - cc * (r2inv * r2inv * r2inv) - eshift;
    } else if (spec.style === 5) {
      const r = Math.sqrt(rsq), dr = r - cc;
      const e1 = Math.exp(-2 * b * dr), e2 = Math.exp(-b * dr);
      fpair += 2 * b * a * (e1 - e2) / r;
      evdwl += a * (e1 - 2 * e2) - eshift;
    } else if (spec.style === 6) {
      const r = Math.sqrt(rsq), arg = r * b;
      fpair += a * b * Math.sin(arg) / r;
      evdwl += a * (1 + Math.cos(arg));
    } else if (spec.style === 7) {
      const r = Math.sqrt(rsq), ir = 1 / r, ir2 = 1 / rsq, ek = Math.exp(-b * r);
      fpair += a * ek * (b * ir2 + ir2 * ir);
      evdwl += a * ir * ek - eshift;
    } else if (spec.style === 8) {
      const rinv = 1 / Math.sqrt(rsq), r2inv = rinv * rinv, r3inv = r2inv * rinv, r6inv = r3inv * r3inv, r9inv = r6inv * r3inv;
      fpair += r2inv * (a * r9inv - b * r6inv);
      evdwl += r9inv * cc - r6inv * d - eshift;
    } else if (spec.style === 9) {
      const r = Math.sqrt(rsq), rinv = 1 / r, ex = Math.exp((cc - r) * b);
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv, r8inv = r6inv * r2inv;
      fpair += a * b * ex * rinv - 6 * d * r8inv + 8 * e5 * r8inv * r2inv;
      evdwl += a * ex - d * r6inv + e5 * r8inv - eshift;
    }
  }
  if (rsq < cutC && qi !== 0 && qj !== 0) {
    const pre = (spec.qqrd2e * cscale * qi * qj) / Math.sqrt(rsq);
    fpair += pre / rsq;
    ecoul += pre;
  }
  return { fpair, evdwl, ecoul };
};
