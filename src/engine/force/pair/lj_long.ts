import { pairCoeffArgs, StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { gplHook } from '../../gplHooks';
import { PairLJCutCoulLong } from './coul_long';
import { tallyAtom } from './lj_cut';
import { erfcFast, erfcPoly, EWALD_F, ErfcTableCache, type ErfcTable } from '../erfc';
import { libm } from '../../libm';

/*
 * pair_style lj/long/coul/long — docs.lammps.org/pair_lj_long.html
 * (source: plans/lammps-docs/pair_lj_long.rst)
 *
 * Syntax (verbatim):
 *   "*lj/long/coul/long* args = flag_lj flag_coul cutoff (cutoff2)"
 *   "flag_lj = *long* or *cut* or *off*"
 *   "*long* = use Kspace long-range summation for dispersion 1/r\^6 term"
 *   "*cut* = use a cutoff on dispersion 1/r\^6 term"
 *   "*off* = omit disperion 1/r\^6 term entirely"
 *   "flag_coul = *long* or *off*"
 *   "*long* = use Kspace long-range summation for Coulombic 1/r term"
 *   "*off* = omit Coulombic term"
 *
 * Description (verbatim):
 *   "If *flag_lj* is set to *long*, no cutoff is used on the LJ 1/r\^6
 *   dispersion term.  The long-range portion can be calculated by using
 *   the :doc:`kspace_style ewald/disp or pppm/disp <kspace_style>` commands."
 *   "Note that if *flag_lj* is also set to long, then the *ewald/disp* or
 *   *pppm/disp* Kspace style needs to be used to perform the long-range
 *   calculations for both the LJ and Coulombic interactions."
 *   "Note that if you are using *flag_lj* set to *long*, you cannot specify a
 *   LJ cutoff for an atom type pair, since only one global LJ cutoff is allowed."
 *   This engine follows native LAMMPS instead, which accepts the per-pair cutoff
 *   (pair_coeff cutoff1) and uses it for the pairs that set it (measured: case
 *   w131cf_ljlong_pair_cut, src/KSPACE/pair_lj_long_coul_long.cpp at b332f28).
 *   "For atom type pairs I,J and I != J, the epsilon and sigma coefficients and
 *   cutoff distance for all of the lj/long pair styles can be mixed."
 *   "These pair styles support the :doc:`pair_modify <pair_modify>` shift"
 *   (then "option for the energy of the Lennard-Jones portion of the pair interaction,
 *   assuming *flag_lj* is *cut*\ .")
 *   "Thes pair styles do not support the :doc:`pair_modify <pair_modify>`"
 *   (the rest of that sentence is "tail option for adding a long-range tail correction").
 *
 * flag_lj = long (this file, class PairLJLongCoulLong with ljLong = true):
 *  - real space, per pair within the global cutoff: E = A/r^12 - C6 g6(g r)/r^6
 *    with A = 4 eps sigma^12, C6 = 4 eps sigma^6 (lj3, lj4 of PairLJCut) and
 *    g6(x) = exp(-x^2) (1 + x^2 + x^4/2), the Ewald-damped dispersion kernel
 *    (in 't Veld, Ismail, Grest 2007). The k-space part of 1/r^6 is kspace_style ewald/disp
 *    (src/engine-gpl/ewald_disp, the GPL port) or pppm/disp.
 *  - Measured with native LAMMPS (black box, two atoms, cutoff 2.5, gewald/disp 0.4,
 *    pair_modify table/disp 0): evdwl equals A/r^12 - C6 g6/r^6 (no shift) at r = 1.6
 *    (-0.22219283725231) and r = 2.4 (-0.0194305071976297) to all printed digits.
 *  - The Coulomb real-space term is that of lj/cut/coul/long (duplicated here
 *    because coul_long.ts does not export its kernel).
 *  - Measured with native LAMMPS (black box, two atoms, cutoff 2.5, gewald/disp 0.4):
 *    the default dispersion table (table/disp 12) changes the real-space energy
 *    by up to about 2e-7 relative at r = 1.6, and pair_modify table/disp 0 gives
 *    the exact kernel above to all printed digits. So table/disp must be 0 here.
 *  - pair_modify table/disp N > 0 with flag_lj long: the native dispersion lookup table (Pair::init_tables_disp,
 *    src/pair.cpp at b332f28) is built by the lazy helper of src/engine-gpl/pair_disp (HOOKS word lj/long/coul/long) and
 *    used for rsq above its inner cutoff, as the native "table real space" branch.
 *  - Not implemented (StyleError): per-type
 *    LJ cutoffs (the doc forbids them), shift and tail with flag_lj long (the doc
 *    applies shift to flag_lj cut only and has no tail), and special_bonds
 *    weights other than 1 for the LJ part with flag_lj long (the k-space sum
 *    includes excluded pairs, so the correction is not implemented).
 * flag_lj = cut and flag_coul = long/off: unchanged (the class below as before).
 *
 * flag_lj = off (pair_lj_long.rst: "*off* = omit disperion 1/r\^6 term entirely"): measured with native LAMMPS, the
 * LJ terms are computed as with flag_lj cut (the same step-0 evdwl); off differs from cut only in the dispersion flag
 * that the kspace check reads (Pair.dispersionFlag): pppm/disp is accepted, and ewald, pppm, msm and ewald/disp are
 * refused with "KSpace style X is incompatible with Pair style lj/long/coul/long".
 */
const g6kernel = (x2: number): number => libm.exp(-x2) * (1 + x2 + x2 * x2 / 2);

/**
 * The dispersion lookup table of pair_modify table/disp N (native Pair::init_tables_disp): force(rsq) and energy(rsq)
 * are the kernels without lj4 (the table branch of the native compute multiplies them by lj4 = C6), and inner is the
 * rsq below which the series is used.
 */
export interface DispTable {
  readonly inner: number;
  force(rsq: number): number;
  energy(rsq: number): number;
}

/** The helper that builds a DispTable: ntablebits, the dispersion G-ewald, the LJ cutoff and tabinner/disp. */
export type DispTableHook = (ntablebits: number, g: number, cutLJ: number, tabinnerDisp: number) => DispTable;

/** Coulomb real-space kernel, as coul_long.ts coulLongPair. */
const coulShort = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, poly: boolean, table: ErfcTable | null,
): { e: number; f: number } => {
  if (table && rsq > table.inner) {
    // As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp compute at b332f28): the table is interpolated at
    // the double rsq (a doubleRsq table), and the special-bond correction (1 - f) (ctable + fraction dctable) is
    // stored in a float before it is subtracted from qiqj (ftable + fraction dftable)
    const qiqj = qi * qj;
    let F = qqrd2e * table.force(rsq);
    let E = qqrd2e * table.energy(rsq);
    if (fc < 1) {
      const corr = Math.fround((1 - fc) * (qqrd2e * table.coul(rsq)));
      F -= corr;
      E -= corr;
    }
    return { e: qiqj * E, f: qiqj * F / rsq };
  }
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const ex = libm.exp(-grij * grij);
  const erfc = poly || table ? erfcPoly(grij, ex) : erfcFast(grij, ex);
  const pre = qqrd2e * qi * qj / r;
  let forcecoul = pre * (erfc + EWALD_F * grij * ex);
  let e = pre * erfc;
  if (fc < 1) {
    forcecoul -= (1 - fc) * pre;
    e -= (1 - fc) * pre;
  }
  return { e, f: forcecoul / rsq };
};

export class PairLJLongCoulLong extends PairLJCutCoulLong {
  readonly name: string = 'lj/long/coul/long';
  /**
   * single() would give the plain lj/cut term for the LJ part, which is not the dispersion real-space term
   * of this style, so the pairwise energy used by pair/local and hbond/local is refused (an EngineError
   * names it) rather than returned wrong.
   */
  single(): { eng: number; fforce: number } {
    throw new StyleError(`pairwise energies (compute pair/local, hbond/local ehb) are not supported for pair style lj/long/coul/long: its dispersion real-space term is not in single()`);
  }
  /** flag_lj long: the dispersion real-space term; the k-space part is kspace ewald/disp. */
  ljLong = false;
  /** Dispersion G-ewald set by kspace ewald/disp at init (gewald/disp). */
  gEwaldDisp = 0;
  /** flag_coul long (pair_lj_long.html: "flag_coul = long or off"); the k-space solver reads it as its Coulomb term. */
  coulLongFlag = false;
  /**
   * The Coulomb cutoff as given (cutoff2, else the LJ cutoff), kept also with flag_coul off. pair_lj_long.html:
   * "cutoff2 = global cutoff for Coulombic (optional)"; kspace_style ewald/disp estimates its G-ewald from this cutoff
   * when the pair has no Coulomb term (src/KSPACE/pair_lj_long_coul_long.cpp settings at b332f28 keeps cut_coul).
   */
  cutCoulNative = 0;
  /**
   * pair_modify table/disp N (native default 12, pair_modify.rst "The default value of 12
   * (table of length 4096)"). 0 is the series only; N > 0 uses the dispersion table (computeLongLJ).
   */
  tableDisp = 12;
  /** pair_modify tabinner/disp (native tabinner_disp, default sqrt(2)). */
  tabinnerDisp = Math.SQRT2;
  private readonly ljErfcTables = new ErfcTableCache();
  private dispTable: DispTable | null = null;
  private dispKey: number[] = [];
  /**
   * The cutoffs as native LAMMPS keeps them (src/KSPACE/pair_lj_long_coul_long.cpp at b332f28): init_style sets
   * cut_coulsq (coulSqInit), the Coulomb table from cut_coul and the dispersion table from cut_lj_global (dispCut);
   * init_one sets cutsq = max(cut_lj, cut_coul + 2 qdist)^2 (outerSq), also when fix adapt re-initializes the style
   * after changing cut_coul or cut_lj_global, and the force loop leaves out a pair outside it.
   */
  protected coulSqInit = 0;
  protected dispCut = 0;
  protected outerSq = new Float64Array(0);

  /** 2 qdist of the TIP4P variant (lj/long/tip4p/long), in init_one's cutsq; 0 here. */
  protected siteReach(): number { return 0; }

  initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    this.coulSqInit = this.cutCoul * this.cutCoul;
    this.dispCut = this.cutGlobal;
  }

  /**
   * fix adapt cut_coul (ewald_cut the same) and cut_vdwl (cut_LJ the same), native cut_coul and cut_lj_global: the
   * Coulomb cutoff as given (cutCoulNative, the Coulomb term's when flag_coul is long) and the global LJ cutoff.
   */
  override adaptParam(name: string): AdaptAccess | null {
    if (name === 'cut_coul' || name === 'ewald_cut') {
      return {
        dim: 0,
        get: () => this.cutCoulNative,
        set: (v) => { this.cutCoulNative = v; if (this.coulLongFlag) this.cutCoul = v; },
      };
    }
    if (name === 'cut_vdwl' || name === 'cut_LJ') return { dim: 0, get: () => this.cutGlobal, set: (v) => { this.cutGlobal = v; } };
    return super.adaptParam(name);
  }

  settings(args: string[]): void {
    // As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp settings and options at b332f28): the word count,
    // each flag long, cut or off, no Coulomb cut, then the cutoffs (one only when both flags are long)
    if (args.length !== 3 && args.length !== 4) throw new StyleError('Illegal pair_style command');
    const [flagLJ, flagCoul] = args;
    for (const f of [flagLJ, flagCoul]) if (f !== 'long' && f !== 'cut' && f !== 'off') throw new StyleError('Illegal pair_style lj/long/coul/long command');
    if (flagCoul === 'cut') throw new StyleError('Coulomb cut not supported in pair_style lj/long/coul/long');
    this.cutGlobal = parseNum(args[2], 'cutoff');
    if (args.length === 4 && flagLJ === 'long' && flagCoul === 'long') throw new StyleError('Only one cutoff allowed when requesting all long');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style lj/long/coul/long: cutoff must be > 0');
    this.cutCoul = args.length === 4 ? parseNum(args[3], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutCoul > 0)) throw new StyleError('pair_style lj/long/coul/long: Coulomb cutoff must be > 0');
    this.ljLong = flagLJ === 'long';
    // flag_lj off: measured with native LAMMPS, the LJ terms are those of flag_lj cut (step-0 evdwl 5.00207487626398
    // for both on the same charged lattice); only the dispersion flag differs, so pppm/disp is accepted and ewald,
    // pppm and ewald/disp are refused (src/KSPACE/pair_lj_long_coul_long.cpp settings at b332f28 clears
    // dispersionflag only for cut)
    this.dispersionFlag = flagLJ !== 'cut';
    this.coulLongFlag = flagCoul === 'long';
    this.cutCoulNative = this.cutCoul;
    if (flagCoul === 'long') {
      this.coulLong = true;
    } else {
      // "If *flag_coul* is set to *off*, Coulombic interactions are not computed."
      this.coulLong = this.ljLong;   // a long-range LJ part needs a kspace style, as any coulLong pair does
      this.cutCoul = 0;
    }
  }

  coeff(args: string[]): void {
    // As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp coeff at b332f28): 4 or 5 words, so no per-pair
    // Coulomb cutoff although pair_lj_long.rst lists "cutoff2 (distance units)"
    if (args.length < 4 || args.length > 5) throw pairCoeffArgs();
    super.coeff(args);
  }

  /**
   * As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp init_one at b332f28, and its tip4p and dielectric
   * variants): no etail_ij or ptail_ij is computed, so pair_modify tail yes adds nothing.
   */
  tailSums(_count: Float64Array): { etail: number; ptail: number } {
    return { etail: 0, ptail: 0 };
  }

  initOne(i: number, j: number): number {
    let cut = super.initOne(i, j);
    // As native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp init_one at b332f28): with dispersion
    // (EWALD_DISP) a pair without its own pair_coeff cutoff takes the global LJ cutoff, not a mixed one.
    if (this.ljLong && !this.p.isSet(i, j)) {
      this.p.setMixed(i, j, 'cut', this.cutGlobal);
      cut = Math.max(this.cutGlobal, this.cutCoul);
    }
    const nt = this.ntypes + 1;
    if (this.outerSq.length !== nt * nt) this.outerSq = new Float64Array(nt * nt);
    this.outerSq[i * nt + j] = this.outerSq[j * nt + i] = Math.max(this.p.get('cut', i, j), this.cutCoulNative + this.siteReach()) ** 2;
    return cut;
  }

  /** flag_lj long, cut or off: the lj/long/coul/long loop (its Coulomb table and special-bond rounding differ from lj/cut/coul/long's). */
  compute(pc: PairCompute): void {
    this.computeLongLJ(pc);
  }

  /** flag_lj long: LJ repulsion, damped dispersion (real space) and the Coulomb real-space term. */
  private computeLongLJ(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj1, lj2, lj3, lj4, offset } = this;
    const ljLong = this.ljLong;
    // Per type pair LJ cutoff (pair_coeff cutoff1; the global one where not set), squared.
    const cutljsqT = new Float64Array(nt * nt);
    for (let a = 1; a <= this.ntypes; a++) {
      for (let b = 1; b <= this.ntypes; b++) {
        const c = this.p.get('cut', a, b);
        cutljsqT[a * nt + b] = c * c;
      }
    }
    const coul = this.cutCoul > 0;
    const cutcsq = this.coulSqInit;
    const outerSq = this.outerSq;
    const g = this.gEwald;
    const gd = this.gEwaldDisp;
    const gd6 = gd ** 6;
    const tab = coul ? this.ljErfcTables.get(this.table, g, cutcsq, this.tabinnerRsq, true) : null;
    const dtab = this.ljDispTable(gd);
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
        if (rsq >= outerSq[t]) continue;
        let fpair = 0, e = 0;
        if (coul && rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const r = coulShort(rsq, qi, q[j], g, pc.qqrd2e, sC[sb], this.table === 0, tab);
          fpair += r.f;
          ecoul += r.e;
          e += r.e;
        }
        if (rsq < cutljsqT[t] && !ljLong) {
          // flag_lj cut or off: the 12-6 term with its special-bond weight and the pair_modify shift offset
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          const fs = sb === 0 ? 1 : sLJ[sb];
          fpair += fs * r6inv * (r6inv * lj1[t] - lj2[t]) * r2inv;
          const ev = fs * (r6inv * (r6inv * lj3[t] - lj4[t]) - offset[t]);
          evdwl += ev;
          e += ev;
        } else if (rsq < cutljsqT[t]) {
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          const A = lj3[t], C6 = lj4[t];
          // A special pair of weight w scales the repulsion by w and keeps the whole damped dispersion, and the part
          // (1 - w) C6 / r^6 of the dispersion that the k-space sum includes for every pair is taken back here. As
          // native LAMMPS does (src/KSPACE/pair_lj_long_coul_long.cpp compute at b332f28, its special case:
          // force_lj gains t lj2 and evdwl t lj4, with t = (1 - factor) / r^6).
          const w = sb === 0 ? 1 : sLJ[sb];
          const back = (1 - w) * r6inv;
          let ev: number;
          if (dtab !== null && rsq > dtab.inner) {
            // the native table branch (src/KSPACE/pair_lj_long_coul_long.cpp compute): force_lj = 12 A r^-12 - C6 ftable,
            // fpair = force_lj / r^2, and the energy is A r^-12 - C6 etable
            fpair += (12 * w * A * r6inv * r6inv - C6 * dtab.force(rsq) + 6 * C6 * back) * r2inv;
            ev = w * A * r6inv * r6inv - C6 * dtab.energy(rsq) + C6 * back;
          } else {
            // E = A/r^12 - C6 g6/r^6 ;  fpair = -(dE/dr)/r with d/dr [g6/r^6] giving g^6 exp(-x^2)/r^2 extra
            const x2 = gd * gd * rsq;
            const g6 = g6kernel(x2);
            fpair += 12 * w * A * r6inv * r6inv * r2inv - C6 * (6 * g6 * r6inv * r2inv + gd6 * libm.exp(-x2) * r2inv)
              + 6 * C6 * back * r2inv;
            ev = w * A * r6inv * r6inv - C6 * g6 * r6inv + C6 * back;
          }
          evdwl += ev;
          e += ev;
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

  /**
   * The dispersion table for pair_modify table/disp N > 0 with flag_lj long, rebuilt when N, the dispersion G-ewald, the
   * global LJ cutoff or tabinner/disp change. The table comes from the lazy helper (gplHook, src/engine-gpl/pair_disp).
   */
  private ljDispTable(gd: number): DispTable | null {
    if (!this.ljLong || this.tableDisp <= 0) return null;
    const key = [this.tableDisp, gd, this.dispCut, this.tabinnerDisp];
    if (this.dispTable === null || key.some((v, i) => v !== this.dispKey[i])) {
      this.dispTable = gplHook<DispTableHook>('lj/long/coul/long')(this.tableDisp, gd, this.dispCut, this.tabinnerDisp);
      this.dispKey = key;
    }
    return this.dispTable;
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}
