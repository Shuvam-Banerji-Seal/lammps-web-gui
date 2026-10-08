import { StyleError, type PairCompute } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { PairBorn, PairBuck } from './simple';
import { PairBornCoulLong, PairBuckCoulLong } from './coul_long2';
import { PairCoulLong } from './coul_long';
import { tallyAtom } from './lj_cut';
import { erfcFast, EWALD_F } from '../erfc';

/*
 * Core-shell ("/cs") variants: born/coul/long/cs, buck/coul/long/cs and
 * coul/long/cs. docs.lammps.org/pair_cs.html:
 * "All the styles are identical to the corresponding pair style without
 * the "/cs" in the name:" ... "except that they correctly treat the special
 * case where" "the distance between two charged core and shell atoms in the
 * same core/shell pair approach r = 0.0." (the sentence continues on the
 * next line of the page; the words quoted here are verbatim).
 * "Specifically, the short-range Coulomb interaction between a core and its
 * shell should be turned off using the special_bonds command by setting the
 * 1-2 weight to 0.0, which works because the core and shell atoms are bonded
 * to each other." "This induces a long-range correction approximation which
 * fails at small distances (~< 10e-8). Therefore, the Coulomb term which is
 * used to calculate the correction factor is extended by a minimal distance
 * (r_min = 1.0-6) when the interaction between a core/shell pair is treated,
 * as follows" E = \frac{C q_i q_j}{\epsilon (r + r_{min})} \qquad r \rightarrow 0
 * "where C is an energy-conversion constant," ":math:`q_i` and :math:`q_j`" "are the charges on the core and shell, epsilon is the dielectric"
 * "constant and :math:`r_{min}` is the minimal distance." "For styles that are not used with a long-range solver" (the "/dsf" and "/wolf" styles
 * get only the minimal-distance correction; they are not part of this engine's /cs set).
 * The LJ/Buckingham/Born parts are those of the base styles (pair_born.html,
 * pair_buck.html, pair_coul.html); see coul_long2.ts and coul_long.ts.
 *
 * Coulomb term used here (fc = factor_coul of the pair):
 *   E = C q_i q_j (erfc(g r) - (1 - fc)) / r = C q_i q_j (fc - erf(g r)) / r
 * (the damped term of pair_coul.html less the special-bond part of
 * special_bonds.html; this is what the base styles compute).
 *
 * Measured with native LAMMPS (black box). Two atoms, bonded, special_bonds
 * lj/coul 0.0 0.0 0.0 (fc = 0), qi = +1, qj = -1, g = 0.3, C = 332.06371:
 *   ecoul = 112.408131753734 at r = 1e-7, 112.40813175373 at r = 1e-6,
 *   112.407794530249 at r = 1e-2, 112.374418417316 at r = 0.1, 109.125020951275
 *   at r = 1; each equals C erf(g r)/r to the printed 15 digits. A shift
 *   r -> r + r_min in the energy would change the r = 1e-2 value by 6e-10
 *   relative, which the native values exclude, so the engine evaluates the
 *   formula at r itself. The base polynomial erfc is not accurate enough at
 *   small r (native coul/long at r = 1e-7: pe = 3.32 where the exact value is
 *   0), so for fc < 1 the engine evaluates erf with a power series for g r < 1
 *   (erfAcc) and the force through h(x)/x^3 (hOverX3), with no cancellation.
 * Measured with native LAMMPS, fc = 0.5 and r = 1e-7: ecoul is 820 larger than
 * the exact value (relative 5e-7); no r_min form was found that reproduces it
 * and the engine returns the exact value there (not reproduced).
 * Measured with native LAMMPS, for pairs with fc = 1 the /cs styles do not
 * reproduce the base erfc at the 1e-7 level (about 3e-8 at r = 0.1 to 1,
 * 1e-7 at r = 3 with pair_modify table 0). The engine uses the accurate erfc
 * (erfc.ts erfcFast) for every pair of a /cs style; the oracle cases therefore
 * compare with rel 1e-6.
 */

/*
 * Exact 2/sqrt(pi) for the erf identities below. The force term of the base
 * styles uses EWALD_F (erfc.ts), which is rounded to 8 digits on purpose; the
 * fc < 1 force keeps that convention, the erf series does not (at small x the
 * rounding would show as a relative error of order 1e-5 in h(x)).
 */
const TWO_OVER_SQRTPI = 2 / Math.sqrt(Math.PI);

/** erf(x) for x >= 0 without cancellation at small x (power series below 1). */
export const erfAcc = (x: number): number => {
  if (x < 1) {
    // erf(x) = 2/sqrt(pi) sum_n (-1)^n x^(2n+1) / (n! (2n+1))
    const x2 = x * x;
    let term = x, sum = x;
    for (let n = 1; n < 40; n++) {
      term *= -x2 / n;
      const t = term / (2 * n + 1);
      sum += t;
      if (Math.abs(t) < 1e-18 * Math.abs(sum)) break;
    }
    return TWO_OVER_SQRTPI * sum;
  }
  return 1 - erfcFast(x, Math.exp(-x * x));
};

/**
 * h(x)/x^3 with h(x) = erf(x) - (2/sqrt(pi)) x exp(-x^2). Series for x < 1:
 * h(x)/x^3 = (2/sqrt(pi)) sum_{n>=1} (-1)^(n+1) 2n x^(2n-2) / ((2n+1) n!),
 * which is finite at x = 0 (the first term is (2/sqrt(pi)) 2/3).
 */
export const hOverX3 = (x: number): number => {
  if (x >= 1) {
    const h = erfAcc(x) - TWO_OVER_SQRTPI * x * Math.exp(-x * x);
    return h / (x * x * x);
  }
  const x2 = x * x;
  let sum = 0, pw = 1;
  for (let n = 1; n < 40; n++) {
    // pw = x^(2n-2) (-1)^(n-1) / n!
    const fact = nFact(n);
    const t = 2 * n * pw / ((2 * n + 1) * fact);
    sum += t;
    pw *= -x2;
    if (Math.abs(t) < 1e-18 * Math.abs(sum) && n > 2) break;
  }
  return TWO_OVER_SQRTPI * sum;
};

const FACT: number[] = [1];
const nFact = (n: number): number => {
  while (FACT.length <= n) FACT.push(FACT[FACT.length - 1] * FACT.length);
  return FACT[n];
};

/**
 * Real-space Coulomb pair term for the /cs styles at squared distance rsq.
 * Returns the energy and fpair = F/r (the same convention as coul_long.ts).
 * fc = 1 reproduces the base damped term; fc < 1 is the core-shell form above.
 */
export const csCoulPair = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number,
): { e: number; f: number } => {
  const r = Math.sqrt(rsq);
  const x = g * r;
  const ex = Math.exp(-x * x);
  const K = qqrd2e * qi * qj;
  if (fc >= 1) {
    // Accurate erfc for every pair of a /cs style (see the header): native cs gives
    // -25725.9818206999 for the all-special-1 case where the base poly gives -25725.9779029422.
    const erfc = erfcFast(x, ex);
    const pre = K / r;
    return { e: pre * erfc, f: pre * (erfc + EWALD_F * x * ex) / rsq };
  }
  if (fc === 0) {
    // fpair = -K h(x) / r^3 = -K g^3 h(x)/x^3 (no cancellation)
    return { e: -K * erfAcc(x) / r, f: -K * g * g * g * hOverX3(x) };
  }
  const erf = erfAcc(x);
  return { e: K * (fc - erf) / r, f: K * ((fc - erf) + EWALD_F * x * ex) / (r * rsq) };
};

/** The born and buck /cs styles take the same arguments as their base styles: cutoff (cutoff2). */
const sameArgs = (args: string[], usage: string): void => {
  if (args.length !== 1 && args.length !== 2) throw new StyleError(usage);
};

export class PairBornCoulLongCS extends PairBornCoulLong {
  readonly name: string = 'born/coul/long/cs';

  settings(args: string[], ctx: Parameters<PairBornCoulLong['settings']>[1]): void {
    sameArgs(args, 'usage: pair_style born/coul/long/cs cutoff (cutoff2)');
    super.settings(args, ctx);
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { bornA, bornIR, bornSig, bornC, bornD, offset } = this;
    const bornCutSq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) bornCutSq[i * nt + j] = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < bornCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const rinv = 1 / r;
          const ex = Math.exp((bornSig[t] - r) * bornIR[t]);
          const r2inv = 1 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          const r8inv = r6inv * r2inv;
          fpair += factor * (bornA[t] * bornIR[t] * ex * rinv - 6 * bornC[t] * r8inv + 8 * bornD[t] * r8inv * r2inv);
          const ev = factor * (bornA[t] * ex - bornC[t] * r6inv + bornD[t] * r8inv - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const c = csCoulPair(rsq, qi, q[j], g, pc.qqrd2e, sC[sb]);
          fpair += c.f;
          ecoul += c.e;
          e += c.e;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const b = PairBorn.prototype.single.call(this, _i, _j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
    let eng = b.eng, fforce = b.fforce;
    if (rsq < this.cutCoul * this.cutCoul && qi !== 0 && qj !== 0) {
      const c = csCoulPair(rsq, qi, qj, this.gEwald, this.qqrd2e, factorCoul);
      fforce += c.f;
      eng += c.e;
    }
    return { eng, fforce };
  }
}

export class PairBuckCoulLongCS extends PairBuckCoulLong {
  readonly name: string = 'buck/coul/long/cs';

  settings(args: string[], ctx: Parameters<PairBuckCoulLong['settings']>[1]): void {
    sameArgs(args, 'usage: pair_style buck/coul/long/cs cutoff (cutoff2)');
    super.settings(args, ctx);
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { buckA, buckIR, buckC, offset } = this;
    const buckCutSq = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) buckCutSq[i * nt + j] = nonCoulCut(this.p.get('cut', i, j), this.cutGlobal) ** 2;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const sb = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        let fpair = 0, e = 0;
        if (rsq < buckCutSq[t]) {
          const factor = sLJ[sb];
          const r = Math.sqrt(rsq);
          const ex = Math.exp(-r * buckIR[t]);
          const r2inv = 1 / rsq;
          fpair += factor * (buckA[t] * buckIR[t] * ex / r - 6 * buckC[t] * r2inv * r2inv * r2inv * r2inv);
          const ev = factor * (buckA[t] * ex - buckC[t] * (r2inv * r2inv * r2inv) - offset[t]);
          evdwl += ev;
          e += ev;
        }
        if (rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const c = csCoulPair(rsq, qi, q[j], g, pc.qqrd2e, sC[sb]);
          fpair += c.f;
          ecoul += c.e;
          e += c.e;
        }
        if (fpair === 0 && e === 0) continue;
        fxi += dx * fpair; fyi += dy * fpair; fzi += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi: number, qj: number) {
    const b = PairBuck.prototype.single.call(this, _i, _j, itype, jtype, rsq, factorCoul, factorLJ, qi, qj);
    let eng = b.eng, fforce = b.fforce;
    if (rsq < this.cutCoul * this.cutCoul && qi !== 0 && qj !== 0) {
      const c = csCoulPair(rsq, qi, qj, this.gEwald, this.qqrd2e, factorCoul);
      fforce += c.f;
      eng += c.e;
    }
    return { eng, fforce };
  }
}

export class PairCoulLongCS extends PairCoulLong {
  readonly name: string = 'coul/long/cs';

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style coul/long/cs cutoff');
    super.settings(args);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, q } = pc;
    const cutsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const qi = q[i];
      if (qi === 0) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const qj = q[j];
        if (qj === 0) continue;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq) continue;
        const r = csCoulPair(rsq, qi, qj, g, pc.qqrd2e, sC[jj >>> SBBITS]);
        fxi += dx * r.f; fyi += dy * r.f; fzi += dz * r.f;
        f[3 * j] -= dx * r.f; f[3 * j + 1] -= dy * r.f; f[3 * j + 2] -= dz * r.f;
        ecoul += r.e;
        if (tally) tallyAtom(pc, i, j, r.e, r.f, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

  // write_data: measured with native LAMMPS (black box), coul/long/cs writes no pair coefficient
  // section, like coul/long (PairCoulLong.dataCoeffs returns null).
  dataCoeffsIJ(): string[] | null { return null; }
}

/** Per-pair non-Coulomb cutoff: NaN means use the global pair_style cutoff. */
const nonCoulCut = (raw: number, cutGlobal: number): number => (Number.isNaN(raw) ? cutGlobal : raw);
