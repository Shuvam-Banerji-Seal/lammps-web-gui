/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/KSPACE/pair_buck_long_coul_long.cpp and
 * .h (contributing author Pieter J. in 't Veld, SNL). Copyright Sandia Corporation (GNU General Public License). This file
 * is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style buck/long/coul/long flag_buck flag_coul cutoff (cutoff2) (docs.lammps.org/pair_buck_long.html):
 * "flag_buck = *long* or *cut* or *off*", "flag_coul = *long* or *off*" (the docs and the settings code at b332f28: "Coulomb
 * cut not supported", "LJ6 off not supported"). The Buckingham repulsion A exp(-r/rho) is the short-range term; the -C/r^6
 * dispersion is damped by the Ewald dispersion sum of kspace_style ewald/disp when flag_buck is long; the Coulomb real-space
 * term is the Ewald term of coul/long. The pair coefficients are not mixed (native init_one errors on an unset pair). The
 * per-pair arithmetic follows the native compute(): the Horner form of the erfc polynomial, the damped kernels a2 exp(-g^2 r^2),
 * and the special-bond corrections; the table branch subtracts the special Coulomb correction after a float32 rounding, as
 * native does (rsq_lookup.f is a float).
 */
import { Pair, PairParams, StyleError, type AdaptAccess, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { ErfcTableCache, type ErfcTable } from '../../engine/force/erfc';
import { makeDispTable } from '../pair_disp/dispTable';
import type { DispTable } from '../../engine/force/pair/lj_long';
import type { EwaldDispPair } from '../ewald_disp/ewald_disp';
import { exp as glibcExp } from '../libm/exp';

/** Ewald order bits (native EWALD_COUL = 1 << 1, EWALD_DISP = 1 << 6) and the polynomial erfc constants of the native compute. */
const EWALD_COUL = 1 << 1;
const EWALD_DISP = 1 << 6;
const EWALD_P = 0.3275911;
const A1 = 0.254829592;
const A2 = -0.284496736;
const A3 = 1.421413741;
const A4 = -1.453152027;
const A5 = 1.061405429;
/** EWALD_F: the force constant of the Ewald sum (2/sqrt(pi) rounded as native). */
const EWALD_F = 1.12837917;

/** float32 rounding (the C union rsq_lookup.f of native compute). */
const f32 = (x: number): number => Math.fround(x);

export class PairBuckLongCoulLong extends Pair {
  readonly name: string = 'buck/long/coul/long';
  virialFdotr = true;
  /** flag_coul long (the k-space sum takes the Coulomb term) and flag_buck long (dispersion sum). */
  coulLongFlag = false;
  dispFlag = false;
  cutBuckGlobal = 0;
  cutCoul = 0;
  /** Dispersion G-ewald (set by kspace ewald/disp through the view). */
  gEwaldDisp = 0;
  tableDisp = 12;
  tabinnerDisp = Math.SQRT2;
  private p!: PairParams;
  private readonly coulTables = new ErfcTableCache();
  private dispTab: DispTable | null = null;
  private dispKey: number[] = [];
  private nt = 0;
  private cutBucksq = new Float64Array(0);
  private cutSq = new Float64Array(0);
  private buckA = new Float64Array(0);
  private buckC = new Float64Array(0);
  private buck1 = new Float64Array(0);
  private buck2 = new Float64Array(0);
  private rhoinv = new Float64Array(0);
  private offsetT = new Float64Array(0);
  private cutBuckT = new Float64Array(0);
  /**
   * init_style's cut_coulsq and the cut_buck_global of its init_tables_disp (as native LAMMPS does,
   * src/KSPACE/pair_buck_long_coul_long.cpp at b332f28); init_one sets cutsq itself (cutSq), also on a fix adapt re-init.
   */
  private coulSqInit = 0;
  private dispCut = 0;

  initStyle(_ctx: StyleContext): void {
    this.coulSqInit = this.cutCoul * this.cutCoul;
    this.dispCut = this.cutBuckGlobal;
  }

  /** fix adapt cut_coul (ewald_cut the same) and cut_vdwl (cut_LJ the same): native cut_coul and cut_buck_global. */
  override adaptParam(name: string): AdaptAccess | null {
    if (name === 'cut_coul' || name === 'ewald_cut') return { dim: 0, get: () => this.cutCoul, set: (v) => { this.cutCoul = v; } };
    if (name === 'cut_vdwl' || name === 'cut_LJ') return { dim: 0, get: () => this.cutBuckGlobal, set: (v) => { this.cutBuckGlobal = v; } };
    return null;
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'rho', 'c', 'cut']);
  }

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3 && args.length !== 4) throw new StyleError('Illegal pair_style command');
    const [flagBuck, flagCoul] = args;
    if (flagBuck === 'off') throw new StyleError('LJ6 off not supported in pair_style buck/long/coul/long');
    if (flagBuck !== 'long' && flagBuck !== 'cut') throw new StyleError('Illegal pair_style buck/long/coul/long command');
    if (flagCoul === 'cut') throw new StyleError('Coulomb cut not supported in pair_style buck/long/coul/coul');
    if (flagCoul !== 'long' && flagCoul !== 'off') throw new StyleError('Illegal pair_style buck/long/coul/long command');
    this.dispFlag = flagBuck === 'long';
    // pair_buck_long_coul_long.cpp settings: dispersionflag is cleared for a cut dispersion term (KSpace::pair_check)
    this.dispersionFlag = this.dispFlag;
    this.coulLongFlag = flagCoul === 'long';
    this.coulLong = this.coulLongFlag;
    if (this.dispFlag && this.coulLongFlag && args.length === 4) throw new StyleError('Only one cutoff allowed when requesting all long');
    this.cutBuckGlobal = parseNum(args[2], 'cutoff');
    this.cutCoul = args.length === 4 ? parseNum(args[3], 'cutoff2') : this.cutBuckGlobal;
    if (!(this.cutBuckGlobal > 0)) throw new StyleError('cutoffs must be > 0');
  }

  /** pair_coeff I J A rho C [cutoff]: no mixing (native coeff; a pair left unset is an error in init_one). */
  coeff(args: string[]): void {
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const a = parseNum(args[2], 'A'), rho = parseNum(args[3], 'rho'), c = parseNum(args[4], 'C');
    const cut = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutBuckGlobal;
    this.p.setRange(args[0], args[1], [a, rho, c, cut]);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.nt !== nt) {
      this.nt = nt;
      this.cutBucksq = new Float64Array(nt * nt); this.cutSq = new Float64Array(nt * nt);
      this.buckA = new Float64Array(nt * nt); this.buckC = new Float64Array(nt * nt);
      this.buck1 = new Float64Array(nt * nt); this.buck2 = new Float64Array(nt * nt);
      this.rhoinv = new Float64Array(nt * nt); this.offsetT = new Float64Array(nt * nt);
      this.cutBuckT = new Float64Array(nt * nt);
    }
    const a = this.p.get('a', i, j), rho = this.p.get('rho', i, j), c = this.p.get('c', i, j);
    const cutRead = this.p.get('cut', i, j);
    // native init_one: the global cutoff for the dispersion sum, the read cutoff otherwise
    const cutBuck = this.dispFlag ? this.cutBuckGlobal : cutRead;
    const cut = Math.max(cutBuck, this.cutCoul);
    const off = this.shift && cutBuck > 0.0 ? a * glibcExp(-cutBuck / rho) - c / cutBuck ** 6.0 : 0.0;
    for (const k of [i * nt + j, j * nt + i]) {
      this.cutBuckT[k] = cutBuck;
      this.cutBucksq[k] = cutBuck * cutBuck;
      this.cutSq[k] = cut * cut;
      this.buckA[k] = a;
      this.buckC[k] = c;
      this.buck1[k] = a / rho;
      this.buck2[k] = 6.0 * c;
      this.rhoinv[k] = 1.0 / rho;
      this.offsetT[k] = off;
    }
    return cut;
  }

  /** The dispersion table of pair_modify table/disp N (native init_tables_disp(cut_buck_global)), rebuilt when its inputs change. */
  private dispTable(gd: number): DispTable | null {
    if (!this.dispFlag || this.tableDisp <= 0) return null;
    const key = [this.tableDisp, gd, this.dispCut, this.tabinnerDisp];
    if (this.dispTab === null || key.some((v, i) => v !== this.dispKey[i])) {
      this.dispTab = makeDispTable(this.tableDisp, gd, this.dispCut, this.tabinnerDisp);
      this.dispKey = key;
    }
    return this.dispTab;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.nt;
    const order1 = this.coulLongFlag, order6 = this.dispFlag;
    const qqrd2e = pc.qqrd2e;
    const g = this.gEwald, gd = this.gEwaldDisp;
    const g2 = gd * gd, g6 = g2 * g2 * g2, g8 = g6 * g2;
    const cutCoulsq = this.coulSqInit;
    const coulTab: ErfcTable | null = order1 ? this.coulTables.get(this.table, g, cutCoulsq, this.tabinnerRsq, true) : null;
    const dispTab = order6 ? this.dispTable(gd) : null;
    const sLJ = pc.specialLJ, sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    const nb = list.neighbors;
    let evdwl = 0, ecoul = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const qi = q[i];
      const qri = qi * qqrd2e;
      const ti = type[i] * nt;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      let fix = 0, fiy = 0, fiz = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const ni = jj >>> SBBITS;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= this.cutSq[t]) continue;
        const r2inv = 1.0 / rsq;
        const r = Math.sqrt(rsq);
        let forceCoul = 0, ec = 0;
        if (order1 && rsq < cutCoulsq) {
          if (!coulTab || rsq <= coulTab.inner) {
            // series real space: the native erfc polynomial in Horner form; t = 1/(1 + p g r), prefactor qqrd2e qi qj
            const grij = g * r;
            let tt = 1.0 / (1.0 + EWALD_P * grij);
            const erfcPoly = ((((tt * A5 + A4) * tt + A3) * tt + A2) * tt + A1);
            let pre = qri * q[j];
            if (ni === 0) {
              pre *= g * glibcExp(-grij * grij);
              tt *= erfcPoly * pre / grij;
              forceCoul = tt + EWALD_F * pre;
              ec = tt;
            } else {
              // special case: the excluded part of the Coulomb term (1 - w) qqrd2e qi qj / r is subtracted
              const adjust = pre * (1.0 - sC[ni]) / r;
              pre *= g * glibcExp(-grij * grij);
              tt *= erfcPoly * pre / grij;
              forceCoul = tt + EWALD_F * pre - adjust;
              ec = tt - adjust;
            }
          } else {
            // table real space: the native tables hold qqrd2e; the special correction is rounded to float32 as native
            const qiqj = q[i] * q[j];
            const kf = qqrd2e * coulTab.force(rsq), ke = qqrd2e * coulTab.energy(rsq);
            if (ni === 0) {
              forceCoul = qiqj * kf;
              ec = qiqj * ke;
            } else {
              const corr = f32((1.0 - sC[ni]) * (qqrd2e * coulTab.coul(rsq)));
              forceCoul = qiqj * (kf - corr);
              ec = qiqj * (ke - corr);
            }
          }
        }
        let forceBuck = 0, eb = 0;
        if (rsq < this.cutBucksq[t]) {
          const r6inv = r2inv * r2inv * r2inv;
          const expr = glibcExp(-r * this.rhoinv[t]);
          const A = this.buckA[t], C = this.buckC[t], buck1 = this.buck1[t], buck2 = this.buck2[t];
          if (order6) {
            if (!dispTab || rsq <= dispTab.inner) {
              // series real space (native gr2 = g^2 r^2, a2 = 1 / gr2, expterm = a2 exp(-gr2) C)
              const gr2 = g2 * rsq, a2 = 1.0 / gr2;
              const expterm = a2 * glibcExp(-gr2) * C;
              const g6term = g6 * ((a2 + 1.0) * a2 + 0.5) * expterm;
              const g8term = g8 * (((6.0 * a2 + 6.0) * a2 + 3.0) * a2 + 1.0) * expterm * rsq;
              if (ni === 0) {
                forceBuck = r * expr * buck1 - g8term;
                eb = expr * A - g6term;
              } else {
                const factor = sLJ[ni], tt = r6inv * (1.0 - factor);
                forceBuck = factor * r * expr * buck1 - g8term + tt * buck2;
                eb = factor * expr * A - g6term + tt * C;
              }
            } else {
              // table real space (native fdisptable and edisptable interpolated at rsq)
              const fdisp = dispTab.force(rsq), edisp = dispTab.energy(rsq);
              if (ni === 0) {
                forceBuck = r * expr * buck1 - fdisp * C;
                eb = expr * A - edisp * C;
              } else {
                const factor = sLJ[ni], tt = r6inv * (1.0 - factor);
                forceBuck = factor * r * expr * buck1 - fdisp * C + tt * buck2;
                eb = factor * expr * A - edisp * C + tt * C;
              }
            }
          } else if (ni === 0) {
            forceBuck = r * expr * buck1 - r6inv * buck2;
            eb = expr * A - r6inv * C - this.offsetT[t];
          } else {
            const factor = sLJ[ni];
            forceBuck = factor * (r * expr * buck1 - r6inv * buck2);
            eb = factor * (expr * A - r6inv * C - this.offsetT[t]);
          }
        }
        const fpair = (forceCoul + forceBuck) * r2inv;
        const fdx = dx * fpair, fdy = dy * fpair, fdz = dz * fpair;
        fix += fdx; fiy += fdy; fiz += fdz;
        f[3 * j] -= fdx; f[3 * j + 1] -= fdy; f[3 * j + 2] -= fdz;
        evdwl += eb;
        ecoul += ec;
        if (tally) tallyAtom(pc, i, j, eb + ec, fpair, dx, dy, dz);
      }
      f[3 * i] += fix; f[3 * i + 1] += fiy; f[3 * i + 2] += fiz;
    }
    pc.acc.evdwl += evdwl;
    pc.acc.ecoul += ecoul;
  }

  /** Native has a single() for this style that this engine does not have yet (the parent's would be wrong): a named gap. */
  protected override singleMissing = true;
  single(): never {
    throw new StyleError('pair style buck/long/coul/long: single() (pair/local, pair_write) is not supported by the browser engine');
  }

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return undefined;
  }

  /** The -C/r^6 coefficients of the set pairs, (ntypes+1)^2 row-major: native extract("B") returns buck_c (pppm_disp init_coeffs). */
  dispCMatrix(): Float64Array {
    const nt = this.ntypes + 1;
    const C = new Float64Array(nt * nt);
    for (let i = 1; i < nt; i++) for (let j = 1; j < nt; j++) C[i * nt + j] = this.p.isSet(i, j) ? this.p.get('c', i, j) : 0;
    return C;
  }

  /**
   * The ewald/disp view (native extract "ewald_order", "ewald_mix", "B"): B = sqrt(C_ii) for the geometric 1/r^6 sum; the
   * mixing rule is the style's pair_modify mix (extract "ewald_mix" is mix_flag), and there are no epsilon and sigma.
   */
  ewaldDisp(): EwaldDispPair {
    return {
      ewaldOrder: () => (this.coulLongFlag ? EWALD_COUL : 0) | (this.dispFlag ? EWALD_DISP : 0),
      mix: this.mix,
      epsilonSigma: false,
      ewaldCut: () => this.cutCoul,
      // ewald/disp reads B = sqrt(4 eps sigma^6) of these values: eps = C/4 and sigma = 1 give B = sqrt(C)
      epsilonOf: (i) => this.p.get('c', i, i) / 4,
      sigmaOf: () => 1,
      setEwald: (_g, g6) => { this.gEwaldDisp = g6; },
    };
  }
}
