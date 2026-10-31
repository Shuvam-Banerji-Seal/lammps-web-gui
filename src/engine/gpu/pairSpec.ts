import type { Pair } from '../force/types';
import { EWALD_F, erfcPoly } from '../force/erfc';

/*
 * The pair styles the WebGPU backend runs besides plain lj/cut (gpu/webgpuForces.ts computeSpec): their coefficients,
 * taken from the engine's own style after its init (the arrays its compute() reads), packed per type pair for the WGSL
 * kernel, and the same force law in fp64 (pairSpecEval) for the tests. Each law is the style's documented one, written
 * as the engine's CPU style computes it (force/pair/lj_coul.ts, coul.ts, simple.ts, coul_long.ts, charmm.ts). Special
 * bonds weight a pair's short-range part by its lj factor and its Coulomb part by its coul factor (special_bonds.html).
 * The long-range Coulomb styles (gewald > 0) give the real-space Ewald term, C q_i q_j erfc(g r)/r, with erfc from the
 * polynomial fit native uses without tables (erfc.ts erfcPoly; pair_modify table N > 0 interpolates the same function,
 * so the GPU's fp32 differs from it by less than its own rounding). For these styles a special pair stays a pair even
 * with both weights 0: special_bonds.html, "the 1-2, 1-3, 1-4 interactions are not excluded from the neighbor list" for
 * kspace, and its coul weight w removes (1 - w) of the bare C q_i q_j / r term (coul_long.ts coulLongPair).
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
  // the real-space part of the long-range Coulomb styles (the kspace style runs on the CPU)
  'coul/long': 11,
  'lj/cut/coul/long': 12,
  'lj/charmm/coul/long': 13,
  // the 9-6 law of lj96/cut (class 2: E = eps (2 (sigma/r)^9 - 3 (sigma/r)^6)), the Buckingham and Born-Mayer-Huggins
  // laws with the long-range Coulomb term
  'lj/class2': 14,
  'lj/class2/coul/cut': 15,
  'lj/class2/coul/long': 16,
  'buck/coul/long': 17,
  'born/coul/long': 18,
};

/**
 * The kernel's short-range law of each style (P.style of the WGSL kernels, styleLaws.ts shortRange): 1 the 12-6 LJ law,
 * 2 none (Coulomb only), 3 Buckingham, 5 Morse, 6 soft, 7 Yukawa, 8 the 9-6 law, 9 Born-Mayer-Huggins, 13 CHARMM's
 * switched 12-6 law.
 */
export const SPEC_LAW: Readonly<Record<number, number>> = {
  1: 1, 2: 2, 3: 3, 4: 3, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 1, 11: 2, 12: 1, 13: 13, 14: 8, 15: 8, 16: 8, 17: 3, 18: 9,
};

/** The styles with the Ewald real-space Coulomb term (spec.gewald, set from the kspace style's G-ewald). */
export const LONG_COUL_STYLES: ReadonlySet<number> = new Set([11, 12, 13, 16, 17, 18]);

/** The styles with a Coulomb term: the kernel reads the charges. */
const CHARGED_STYLES: ReadonlySet<number> = new Set([1, 2, 4, 15, ...LONG_COUL_STYLES]);

/** The pair cutoff of a buck or born type pair with a Coulomb part: its own, or the global one (coul_long2.ts). */
const ownCut = (raw: number, cutGlobal: number): number => (Number.isNaN(raw) ? cutGlobal : raw);

/**
 * Floats per type pair: [cutsqA cutsqC a b] [c d eshift cscale] [e f _ _] (three vec4). lj/charmm/coul/long keeps its
 * switching region in e and f: the inner cutoff squared and 1/(outer^2 - inner^2)^3.
 */
export const SPEC_STRIDE = 12;

export interface GpuPairSpec {
  style: number;
  /** The kernel's short-range law (SPEC_LAW). */
  law: number;
  /** ntypes + 1: type pair (i, j) is entry i * stride + j. */
  stride: number;
  coef: Float64Array;
  maxCutoff: number;
  /** The style has a Coulomb term: the kernel reads the charges. */
  charges: boolean;
  qqrd2e: number;
  /** G-ewald of the kspace style for the long-range Coulomb styles (the run loop refreshes it), else 0. */
  gewald: number;
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
  cutCoul: number; gEwald: number; aL2: number; bL2: number; invDL3: number; cutGlobal: number;
  f9: Arr; f6: Arr; e9: Arr; e6: Arr;
  ta: Arr; tb: Arr; tn: Arr; tm: Arr; toff: Arr; tcutsq: Arr; tcutCsq: Arr;
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
        case 11: // coul/long: the global Coulomb cutoff (its compute() reads no per-pair scale)
          cutC = st.cutCoul ** 2;
          break;
        case 12: // lj/cut/coul/long: lj1..lj4 and offset inside the pair's LJ cutoff, the global Coulomb cutoff
          cutA = st.p.get('cut', i, j) ** 2;
          cutC = st.cutCoul ** 2;
          coef.set([st.lj1[t], st.lj2[t], st.lj3[t], st.lj4[t], st.offset[t]], o + 2);
          break;
        case 13: // lj/charmm/coul/long: lj1..lj4 inside the outer cutoff, switched past the inner one (no offset)
          cutA = st.bL2;
          cutC = st.cutCoul ** 2;
          coef.set([st.lj1[t], st.lj2[t], st.lj3[t], st.lj4[t], 0], o + 2);
          coef[o + 8] = st.aL2;
          coef[o + 9] = st.invDL3;
          break;
        case 14: // lj/class2 (lj_variants2.ts): F/r = (f9 r^-9 - f6 r^-6)/r^2, E = e9 r^-9 - e6 r^-6 - offset
          cutA = st.cutsq[t];
          coef.set([st.f9[t], st.f6[t], st.e9[t], st.e6[t], st.offset[t]], o + 2);
          break;
        case 15: // lj/class2/coul/cut and /coul/long (class2_spica51.ts): E = a r^-9 - b r^-6, F/r = (9 a r^-9 - 6 b r^-6)/r^2
        case 16:
          if (st.tn[t] !== 9 || st.tm[t] !== 6) return `pair style ${pair.name} with exponents ${st.tn[t]}-${st.tm[t]}`;
          cutA = st.tcutsq[t];
          cutC = st.tcutCsq[t];
          coef.set([9 * st.ta[t], 6 * st.tb[t], st.ta[t], st.tb[t], st.toff[t]], o + 2);
          break;
        case 17: // buck/coul/long: the Buckingham pair cutoff (its own or the global one), the global Coulomb cutoff
          cutA = ownCut(st.p.get('cut', i, j), st.cutGlobal) ** 2;
          cutC = st.cutCoul ** 2;
          coef.set([st.buckA[t], st.buckIR[t], st.buckC[t], 0, st.offset[t]], o + 2);
          break;
        case 18: // born/coul/long: as buck/coul/long, with D in the third vec4
          cutA = ownCut(st.p.get('cut', i, j), st.cutGlobal) ** 2;
          cutC = st.cutCoul ** 2;
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
  const charges = CHARGED_STYLES.has(style);
  const law = SPEC_LAW[style];
  const gewald = LONG_COUL_STYLES.has(style) ? st.gEwald : 0;
  if (LONG_COUL_STYLES.has(style) && !(gewald > 0)) return `pair style ${pair.name} without a kspace style's G-ewald`;
  return { style, law, stride: nt, coef, maxCutoff: maxCut, charges, qqrd2e, gewald, special, key: `${style}|${qqrd2e}|${Array.from(coef).join(',')}` };
};

/**
 * One pair's force over r (fpair), van der Waals and Coulomb energies with the spec's law, in fp64: the kernel's
 * arithmetic (webgpuForces.ts KERNEL_STYLES), which the tests compare with the engine's CPU styles. flj and fc are the
 * pair's special_bonds weights (1 for an ordinary pair).
 */
export const pairSpecEval = (spec: GpuPairSpec, ti: number, tj: number, rsq: number, qi: number, qj: number, flj = 1, fc = 1):
  { fpair: number; evdwl: number; ecoul: number } => {
  const o = SPEC_STRIDE * (ti * spec.stride + tj);
  const c = spec.coef;
  const cutA = c[o], cutC = c[o + 1], a = c[o + 2], b = c[o + 3], cc = c[o + 4], d = c[o + 5], eshift = c[o + 6], cscale = c[o + 7];
  const e5 = c[o + 8];
  let fpair = 0, evdwl = 0, ecoul = 0;
  const law = spec.law;
  if (rsq < cutA) {
    if (law === 1) {
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
      fpair += r6inv * (a * r6inv - b) * r2inv;
      evdwl += r6inv * (cc * r6inv - d) - eshift;
    } else if (law === 13) {
      // charmm.ts: E = U S(r) past the inner cutoff, F = F_std S + 12 U (b^2 - r^2)(r^2 - a^2) / (b^2 - a^2)^3
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
      const u = r6inv * (cc * r6inv - d);
      const fstd = r6inv * (a * r6inv - b) * r2inv;
      if (rsq <= e5) {
        fpair += fstd;
        evdwl += u;
      } else {
        const bL2 = cutA, inv = c[o + 9], tt = bL2 - rsq;
        const s = tt * tt * (bL2 + 2 * rsq - 3 * e5) * inv;
        fpair += fstd * s + 12 * u * tt * (rsq - e5) * inv;
        evdwl += u * s;
      }
    } else if (law === 3) {
      const r = Math.sqrt(rsq), ex = Math.exp(-r * b), r2inv = 1 / rsq;
      fpair += a * b * ex / r - 6 * cc * r2inv * r2inv * r2inv * r2inv;
      evdwl += a * ex - cc * (r2inv * r2inv * r2inv) - eshift;
    } else if (law === 5) {
      const r = Math.sqrt(rsq), dr = r - cc;
      const e1 = Math.exp(-2 * b * dr), e2 = Math.exp(-b * dr);
      fpair += 2 * b * a * (e1 - e2) / r;
      evdwl += a * (e1 - 2 * e2) - eshift;
    } else if (law === 6) {
      const r = Math.sqrt(rsq), arg = r * b;
      fpair += a * b * Math.sin(arg) / r;
      evdwl += a * (1 + Math.cos(arg));
    } else if (law === 7) {
      const r = Math.sqrt(rsq), ir = 1 / r, ir2 = 1 / rsq, ek = Math.exp(-b * r);
      fpair += a * ek * (b * ir2 + ir2 * ir);
      evdwl += a * ir * ek - eshift;
    } else if (law === 8) {
      const rinv = 1 / Math.sqrt(rsq), r2inv = rinv * rinv, r3inv = r2inv * rinv, r6inv = r3inv * r3inv, r9inv = r6inv * r3inv;
      fpair += r2inv * (a * r9inv - b * r6inv);
      evdwl += r9inv * cc - r6inv * d - eshift;
    } else if (law === 9) {
      const r = Math.sqrt(rsq), rinv = 1 / r, ex = Math.exp((cc - r) * b);
      const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv, r8inv = r6inv * r2inv;
      fpair += a * b * ex * rinv - 6 * d * r8inv + 8 * e5 * r8inv * r2inv;
      evdwl += a * ex - d * r6inv + e5 * r8inv - eshift;
    }
  }
  fpair *= flj;
  evdwl *= flj;
  if (rsq < cutC && qi !== 0 && qj !== 0) {
    const r = Math.sqrt(rsq);
    const pre = (spec.qqrd2e * cscale * qi * qj) / r;
    if (spec.gewald > 0) {
      // coul_long.ts coulLongPair: the damped term less (1 - w) of the bare one
      const grij = spec.gewald * r, ex = Math.exp(-grij * grij), erfc = erfcPoly(grij, ex);
      fpair += pre * (erfc + EWALD_F * grij * ex - (1 - fc)) / rsq;
      ecoul += pre * (erfc - (1 - fc));
    } else {
      fpair += fc * pre / rsq;
      ecoul += fc * pre;
    }
  }
  return { fpair, evdwl, ecoul };
};
