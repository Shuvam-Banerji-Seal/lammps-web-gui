import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { coulLongPair } from './coul_long';
import { ErfcTableCache, type ErfcTable } from '../erfc';
import { parseNum } from '../util';

/*
 * pair_style lj/class2/coul/cut, lj/class2/coul/long, lj/spica, lj/spica/coul/long, and the
 * lj/sdk names that are their backward-compatible originals. Written from the documentation
 * (docs.lammps.org/pair_class2.html, docs.lammps.org/pair_spica.html) and black-box runs of the
 * native binary.
 *
 * Every style here is one 2-term power law between the type pair's cutoffs,
 *   E_LJ(r) = A r^-n - B r^-m - offset,
 * with the exponents and prefactors of its family, plus an optional Coulomb term.
 *
 * class 2 (pair_class2.rst): "The *lj/class2* styles compute a 6/9 Lennard-Jones potential given by"
 *   E = \epsilon [ 2 (sigma/r)^9 - 3 (sigma/r)^6 ]        (n = 9, m = 6, A = 2 eps sigma^9, B = 3 eps sigma^6)
 * "The *lj/class2/coul/cut* and *lj/class2/coul/long* styles add a Coulombic term as described for the"
 * Mixing: "Epsilon and sigma are always mixed with the value *sixthpower*\ ."
 *
 * SPICA (pair_spica.rst): "The *lj/spica* styles compute a 9/6, 12/4, 12/5, or 12/6 Lennard-Jones potential,"
 * with prefactors 27/4 (9/6), 3 sqrt(3)/2 (12/4), 12/7 (12/5)^(5/7) (12/5) and 4 (12/6). "Note that sigma is
 * defined in the LJ formula as the zero-crossing distance for the potential, not as the energy minimum."
 * Mixing: "For atom type pairs I,J and I != J, the epsilon and sigma coefficients and cutoff distance for all
 * of the lj/spica pair styles *cannot* be mixed, since different pairs may have different exponents."
 *
 * Measured with native LAMMPS (black box): a lj/spica pair with an unset I,J pair stops with the message
 * No mixing support for lj/spica. Coefficients for all pairs need to be set explicitly.; an unknown cg_type
 * stops with Cannot parse LJ type flag.; a second cutoff on lj/spica/coul/long and on lj/class2/coul/long stops
 * with Incorrect args for pair coefficients; pair_modify tail yes on lj/spica stops with Tail flag not supported
 * by lj/spica pair style (and the same with the style name lj/spica/coul/long).
 *
 * Coulomb parts: lj/class2/coul/cut uses the coul/cut form (pair_lj_cut_coul.html, "Style
 * lj/cut/coul/cut adds a Coulombic pairwise interaction given by" E = C q_i q_j / (eps r)) with the per-pair
 * Coulomb cutoff, and the long styles the Ewald real-space term of coul/long (coul_long.ts coulLongPair,
 * shared with lj/cut/coul/long). pair_modify shift shifts the LJ energy only; tail corrections are
 * class 2 only (as in lj_variants2.ts), the spica styles refuse them.
 */

export type CoulKind = 'cut' | 'long';

/** One type pair's LJ term: E = a r^-n - b r^-m - off, F/r = (n a r^-n - m b r^-m)/r^2. */
abstract class PairPowerLJ extends Pair {
  /** The force field takes the virial as x.f (the loop tallies no virial itself). */
  virialFdotr = true;
  p!: PairParams;
  /** Global cutoffs from pair_style (Coulomb cutoff = cutoff2 or cutoff). */
  cutGlobal = 0;
  cutCoul = 0;
  /** Coulomb prefactor of the last force evaluation (single() reads it). */
  protected qqrd2eLast = 1;
  private readonly erfcTables = new ErfcTableCache();
  /** Per type pair tables, index i*(ntypes+1)+j (both orders). */
  protected tn = new Float64Array(0);
  protected tm = new Float64Array(0);
  protected ta = new Float64Array(0);
  protected tb = new Float64Array(0);
  protected toff = new Float64Array(0);
  protected tcut = new Float64Array(0);
  protected tcutsq = new Float64Array(0);
  protected tcutCsq = new Float64Array(0);
  protected tcutC = new Float64Array(0);
  /** Set for the styles whose pair_modify tail is accepted (class 2). */
  protected tailSupported = true;

  protected abstract readonly paramNames: readonly string[];
  protected abstract readonly coulKind: CoulKind | null;
  /** Explicit values of one pair_coeff (names order), I and J already removed. */
  protected abstract coeffValues(args: string[]): number[];
  /** Fills an unset (i,j) pair from its diagonal pairs, or refuses. */
  protected abstract mixPair(i: number, j: number): void;
  /** Exponents, LJ prefactors and LJ cutoff of a set pair. */
  protected abstract termsOf(i: number, j: number): { n: number; m: number; a: number; b: number; cutLJ: number; cutC: number };

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, this.paramNames);
  }

  settings(args: string[]): void {
    const coul = this.coulKind !== null;
    if (args.length !== 1 && !(coul && args.length === 2)) {
      throw new StyleError(`usage: pair_style ${this.name} cutoff${coul ? ' (cutoff2)' : ''}`);
    }
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.cutCoul = coul ? (args.length === 2 ? parseNum(args[1], 'cutoff2') : this.cutGlobal) : 0;
    if (!(this.cutGlobal > 0) || (coul && !(this.cutCoul > 0))) throw new StyleError('cutoffs must be > 0');
  }

  coeff(args: string[]): void {
    const values = this.coeffValues(args);
    this.p.setRange(args[0], args[1], values);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail && !this.tailSupported) throw new StyleError(`Tail flag not supported by ${this.name} pair style`);
  }

  protected ensureTables(nt: number): void {
    if (this.tn.length === nt * nt) return;
    const n = nt * nt;
    this.tn = new Float64Array(n); this.tm = new Float64Array(n);
    this.ta = new Float64Array(n); this.tb = new Float64Array(n);
    this.toff = new Float64Array(n); this.tcut = new Float64Array(n);
    this.tcutsq = new Float64Array(n); this.tcutCsq = new Float64Array(n); this.tcutC = new Float64Array(n);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) this.mixPair(i, j);
    const nt = this.ntypes + 1;
    this.ensureTables(nt);
    const t = this.termsOf(i, j);
    // pair_modify shift: E(rc) of the LJ term is subtracted (energy zero at the LJ cutoff)
    const off = this.shift && t.cutLJ > 0 ? t.a * Math.pow(t.cutLJ, -t.n) - t.b * Math.pow(t.cutLJ, -t.m) : 0;
    const cutC = this.coulKind === 'long' ? this.cutCoul : this.coulKind === 'cut' ? t.cutC : 0;
    for (const k of [i * nt + j, j * nt + i]) {
      this.tn[k] = t.n; this.tm[k] = t.m; this.ta[k] = t.a; this.tb[k] = t.b; this.toff[k] = off;
      this.tcut[k] = t.cutLJ; this.tcutsq[k] = t.cutLJ * t.cutLJ;
      this.tcutC[k] = cutC; this.tcutCsq[k] = cutC * cutC;
    }
    return Math.max(t.cutLJ, cutC);
  }

  private erfcTable(g: number): ErfcTable | null { return this.erfcTables.get(this.table, g, this.cutCoul * this.cutCoul, this.tabinnerRsq); }

  /** LJ energy (no special factor) and F/r of one pair in table slot t. */
  protected ljPair(t: number, rsq: number): { e: number; f: number } {
    const rinv = 1 / Math.sqrt(rsq);
    const rn = Math.pow(rinv, this.tn[t]), rm = Math.pow(rinv, this.tm[t]);
    return {
      e: this.ta[t] * rn - this.tb[t] * rm - this.toff[t],
      f: (this.tn[t] * this.ta[t] * rn - this.tm[t] * this.tb[t] * rm) * rinv * rinv,
    };
  }

  /** Coulomb energy (with the special weight) and F/r of one pair. */
  protected coulPair(rsq: number, qi: number, qj: number, fc: number, qqrd2e: number, tab: ErfcTable | null): { e: number; f: number } {
    if (this.coulKind === 'long') return coulLongPair(rsq, qi, qj, this.gEwald, qqrd2e, fc, this.table === 0, tab);
    // lj/cut/coul/cut form: E = C q_i q_j / r, F/r = C q_i q_j / r^3
    const r = Math.sqrt(rsq);
    const qq = qqrd2e * qi * qj * fc;
    return { e: qq / r, f: qq / (r * rsq) };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const coul = this.coulKind !== null;
    const g = this.gEwald;
    const tab = this.coulKind === 'long' ? this.erfcTable(g) : null;
    this.qqrd2eLast = pc.qqrd2e;
    let evdwl = 0, ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
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
        if (rsq < this.tcutsq[t]) {
          const fac = sLJ[sb];
          const r = this.ljPair(t, rsq);
          fpair += fac * r.f;
          const ev = fac * r.e;
          e += ev;
          evdwl += ev;
        }
        if (coul && rsq < this.tcutCsq[t] && qi !== 0 && q[j] !== 0) {
          const c = this.coulPair(rsq, qi, q[j], sC[sb], pc.qqrd2e, tab);
          fpair += c.f;
          e += c.e;
          ecoul += c.e;
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

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, factorLJ: number, qi = 0, qj = 0): { eng: number; fforce: number } {
    const t = itype * (this.ntypes + 1) + jtype;
    let eng = 0, fforce = 0;
    if (rsq < this.tcutsq[t]) {
      const r = this.ljPair(t, rsq);
      eng += factorLJ * r.e;
      fforce += factorLJ * r.f;
    }
    if (this.coulKind !== null && rsq < this.tcutCsq[t] && qi !== 0 && qj !== 0) {
      const tab = this.coulKind === 'long' ? this.erfcTable(this.gEwald) : null;
      const c = this.coulPair(rsq, qi, qj, factorCoul, this.qqrd2eLast, tab);
      eng += c.e;
      fforce += c.f;
    }
    return { eng, fforce };
  }

  /**
   * Long-range tail sums (pair_modify tail, Sun equation 5 as in lj_variants2.ts), for the LJ term
   * of E = a r^-n - b r^-m: int_rc^inf u r^2 dr = a rc^(3-n)/(n-3) - b rc^(3-m)/(m-3) and
   * int_rc^inf r^3 u' dr = -n a rc^(3-n)/(n-3) + m b rc^(3-m)/(m-3). The caller divides by V and V^2.
   */
  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const k = i * nt + j;
        const rc = this.tcut[k];
        if (!(rc > 0)) continue;
        const n = this.tn[k], m = this.tm[k], a = this.ta[k], b = this.tb[k];
        const nn = count[i] * count[j];
        const In = Math.pow(rc, 3 - n) / (n - 3), Im = Math.pow(rc, 3 - m) / (m - 3);
        e += nn * (a * In - b * Im);
        pr += nn * (-n * a * In + m * b * Im);
      }
    }
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  extract(name: string): unknown {
    if (name === 'epsilon' || name === 'sigma') return this.p.p(name);
    if (name === 'cut_coul' && this.coulKind !== null) return this.cutCoul;
    return undefined;
  }
}

/** lj/class2/coul/cut and lj/class2/coul/long (pair_class2.rst): sixthpower mixing of epsilon and sigma. */
export class PairLJClass2Coul extends PairPowerLJ {
  readonly name: string;
  protected readonly paramNames = ['epsilon', 'sigma', 'cut', 'cut_coul'] as const;
  protected readonly coulKind: CoulKind;

  constructor(name: 'lj/class2/coul/cut' | 'lj/class2/coul/long') {
    super();
    this.name = name;
    this.coulKind = name === 'lj/class2/coul/long' ? 'long' : 'cut';
  }

  protected coeffValues(args: string[]): number[] {
    // pair_class2.rst: "For *lj/class2/coul/long* only the class 2 cutoff can be specified"
    const coul = this.coulKind === 'cut';
    if (args.length < 4 || args.length > (coul ? 6 : 5)) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff1') : this.cutGlobal;
    // "If only one cutoff is specified, it is used as the cutoff for both class 2 and Coulombic interactions for this type pair."
    const cutC = !coul ? this.cutCoul : args[5] !== undefined ? parseNum(args[5], 'cutoff2') : args[4] !== undefined ? cut : this.cutCoul;
    if (!(cut > 0) || !(cutC > 0)) throw new StyleError('pair_coeff cutoffs must be > 0');
    return [eps, sig, cut, cutC];
  }

  protected mixPair(i: number, j: number): void {
    const p = this.p;
    if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const ei = p.get('epsilon', i, i), ej = p.get('epsilon', j, j), si = p.get('sigma', i, i), sj = p.get('sigma', j, j);
    p.setMixed(i, j, 'epsilon', mixEpsilon('sixthpower', ei, ej, si, sj));
    p.setMixed(i, j, 'sigma', mixDistance('sixthpower', si, sj));
    p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    p.setMixed(i, j, 'cut_coul', mixDistance(this.mix, p.get('cut_coul', i, i), p.get('cut_coul', j, j)));
  }

  protected termsOf(i: number, j: number): { n: number; m: number; a: number; b: number; cutLJ: number; cutC: number } {
    const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j);
    // E = eps (2 (sigma/r)^9 - 3 (sigma/r)^6)
    return {
      n: 9, m: 6, a: 2 * eps * Math.pow(sig, 9), b: 3 * eps * Math.pow(sig, 6),
      // lj/class2/coul/long has one global Coulomb cutoff, read at each init (src/CLASS2/pair_lj_class2_coul_long.cpp
      // init_style at b332f28), which fix adapt may have changed since the pair_coeff command
      cutLJ: this.p.get('cut', i, j), cutC: this.coulKind === 'long' ? this.cutCoul : this.p.get('cut_coul', i, j),
    };
  }
}

/** Cg types of pair_spica.rst: exponents and prefactor of the LJ term. */
const SPICA_TYPES: Readonly<Record<string, { n: number; m: number; pref: number }>> = {
  lj9_6: { n: 9, m: 6, pref: 27 / 4 },
  lj12_4: { n: 12, m: 4, pref: (3 * Math.sqrt(3)) / 2 },
  lj12_5: { n: 12, m: 5, pref: (12 / 7) * Math.pow(12 / 5, 5 / 7) },
  lj12_6: { n: 12, m: 6, pref: 4 },
};

/**
 * lj/spica, lj/spica/coul/long and the lj/sdk originals (pair_spica.rst: "The original implementation of the above
 * styles are style *lj/sdk*, *lj/sdk/coul/long*, and *lj/sdk/coul/msm*, and available for backward compatibility.").
 * Measured with native LAMMPS (black box): lj/sdk and lj/spica give identical energies and forces for lj9_6 with
 * pair_modify shift yes and no.
 */
export class PairLJSpica extends PairPowerLJ {
  readonly name: string;
  protected readonly paramNames = ['epsilon', 'sigma', 'cut', 'cgn', 'cgm', 'cgpref'] as const;
  protected readonly coulKind: CoulKind | null;
  protected tailSupported = false;

  constructor(name: 'lj/spica' | 'lj/sdk' | 'lj/spica/coul/long' | 'lj/sdk/coul/long') {
    super();
    this.name = name;
    this.coulKind = name.endsWith('/coul/long') ? 'long' : null;
  }

  protected coeffValues(args: string[]): number[] {
    // pair_spica.rst: "pair_coeff 1 1 lj9_6 100.0 3.5 12.0"; "For *lj/spica/coul/long* and *lj/spica/coul/msm* only the LJ cutoff can be"
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const cg = SPICA_TYPES[args[2]];
    if (!cg) throw new StyleError('Cannot parse LJ type flag.');
    const eps = parseNum(args[3], 'epsilon');
    const sig = parseNum(args[4], 'sigma');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    if (!(cut > 0)) throw new StyleError('pair_coeff cutoffs must be > 0');
    return [eps, sig, cut, cg.n, cg.m, cg.pref];
  }

  protected mixPair(i: number, j: number): void {
    throw new StyleError(`No mixing support for ${this.name}. Coefficients for all pairs need to be set explicitly.`);
  }

  protected termsOf(i: number, j: number): { n: number; m: number; a: number; b: number; cutLJ: number; cutC: number } {
    const p = this.p;
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), pref = p.get('cgpref', i, j);
    const n = p.get('cgn', i, j), m = p.get('cgm', i, j);
    return { n, m, a: pref * eps * Math.pow(sig, n), b: pref * eps * Math.pow(sig, m), cutLJ: p.get('cut', i, j), cutC: 0 };
  }

  /**
   * The 1-3 repulsion of angle_style spica (angle_spica.rst: "with the *repulsive* part of the non-bonded
   * *lj/spica* pair style between the atoms 1 and 3"). Measured with native LAMMPS (black box): the repulsion
   * is E_LJ(r) + epsilon for r below the LJ minimum r_min = sigma (n/m)^(1/(n-m)) and zero beyond it, with
   * the unshifted LJ term, and it does not depend on the pair cutoff or on pair_modify shift.
   */
  angleRepulsion(itype: number, jtype: number): { a: number; b: number; n: number; m: number; eps: number; rmin: number } | null {
    const nt = this.ntypes + 1;
    if (!this.p.isSet(itype, jtype) || this.tn.length !== nt * nt) return null;
    const k = itype * nt + jtype;
    const n = this.tn[k], m = this.tm[k];
    const sig = this.p.get('sigma', itype, jtype);
    return {
      a: this.ta[k], b: this.tb[k], n, m, eps: this.p.get('epsilon', itype, jtype),
      rmin: sig * Math.pow(n / m, 1 / (n - m)),
    };
  }
}
