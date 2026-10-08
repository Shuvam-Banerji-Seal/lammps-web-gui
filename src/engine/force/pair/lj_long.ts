import { StyleError, type PairCompute } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { parseNum } from '../util';
import { PairLJCutCoulLong } from './coul_long';
import { tallyAtom } from './lj_cut';
import { erfcFast, erfcPoly, EWALD_F, ErfcTableCache, TABLE_INNER_RSQ, type ErfcTable } from '../erfc';

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
 *    (in 't Veld, Ismail, Grest 2007). The k-space part of 1/r^6 is in
 *    kspace_disp / ewald_disp.ts.
 *  - Measured with native LAMMPS (black box, two atoms, cutoff 2.5, gewald/disp 0.4,
 *    pair_modify table/disp 0): evdwl equals A/r^12 - C6 g6/r^6 (no shift) at r = 1.6
 *    (-0.22219283725231) and r = 2.4 (-0.0194305071976297) to all printed digits.
 *  - The Coulomb real-space term is that of lj/cut/coul/long (duplicated here
 *    because coul_long.ts does not export its kernel).
 *  - Measured with native LAMMPS (black box, two atoms, cutoff 2.5, gewald/disp 0.4):
 *    the default dispersion table (table/disp 12) changes the real-space energy
 *    by up to about 2e-7 relative at r = 1.6, and pair_modify table/disp 0 gives
 *    the exact kernel above to all printed digits. So table/disp must be 0 here.
 *  - Not implemented (StyleError): pair_modify table/disp N > 0 with flag_lj long
 *    (the dispersion table is not reproduced), per-type
 *    LJ cutoffs (the doc forbids them), shift and tail with flag_lj long (the doc
 *    applies shift to flag_lj cut only and has no tail), and special_bonds
 *    weights other than 1 for the LJ part with flag_lj long (the k-space sum
 *    includes excluded pairs, so the correction is not implemented).
 * flag_lj = cut and flag_coul = long/off: unchanged (the class below as before).
 *
 * flag_lj = off is not supported (native LAMMPS rejects kspace ewald and pppm
 * with it, measured in an earlier wave).
 */
const g6kernel = (x2: number): number => Math.exp(-x2) * (1 + x2 + x2 * x2 / 2);

/** Coulomb real-space kernel, as coul_long.ts coulLongPair. */
const coulShort = (
  rsq: number, qi: number, qj: number, g: number, qqrd2e: number, fc: number, poly: boolean, table: ErfcTable | null,
): { e: number; f: number } => {
  if (table && rsq >= TABLE_INNER_RSQ) {
    const qq = qqrd2e * qi * qj;
    let forcecoul = qq * table.force(rsq);
    let e = qq * table.energy(rsq);
    if (fc < 1) {
      const bare = qq * table.coul(rsq);
      forcecoul -= (1 - fc) * bare;
      e -= (1 - fc) * bare;
    }
    return { e, f: forcecoul / rsq };
  }
  const r = Math.sqrt(rsq);
  const grij = g * r;
  const ex = Math.exp(-grij * grij);
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
  /** flag_lj long: the dispersion real-space term; the k-space part is kspace ewald/disp. */
  ljLong = false;
  /** Dispersion G-ewald set by kspace ewald/disp at init (gewald/disp). */
  gEwaldDisp = 0;
  /**
   * pair_modify table/disp N (native default 12, pair_modify.rst "The default value of 12
   * (table of length 4096)"). The dispersion table is not implemented: N must be 0.
   * forcefield.ts currently drops the table/disp value (a hook, see the kspace report).
   */
  tableDisp = 12;
  private readonly ljErfcTables = new ErfcTableCache();

  settings(args: string[]): void {
    if (args.length !== 3 && args.length !== 4) {
      throw new StyleError('usage: pair_style lj/long/coul/long flag_lj flag_coul cutoff (cutoff2)');
    }
    const [flagLJ, flagCoul] = args;
    if (flagLJ === 'off') {
      throw new StyleError("pair_style lj/long/coul/long flag_lj 'off' is not supported (native LAMMPS rejects kspace ewald and pppm with it)");
    }
    if (flagLJ !== 'cut' && flagLJ !== 'long') throw new StyleError(`pair_style lj/long/coul/long: invalid flag_lj '${flagLJ}' (long, cut or off)`);
    if (flagCoul !== 'long' && flagCoul !== 'off') throw new StyleError(`pair_style lj/long/coul/long: invalid flag_coul '${flagCoul}' (long or off)`);
    this.cutGlobal = parseNum(args[2], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair_style lj/long/coul/long: cutoff must be > 0');
    this.cutCoul = args.length === 4 ? parseNum(args[3], 'Coulomb cutoff') : this.cutGlobal;
    if (!(this.cutCoul > 0)) throw new StyleError('pair_style lj/long/coul/long: Coulomb cutoff must be > 0');
    this.ljLong = flagLJ === 'long';
    if (flagCoul === 'long') {
      this.coulLong = true;
    } else {
      // "If *flag_coul* is set to *off*, Coulombic interactions are not computed."
      this.coulLong = this.ljLong;   // a long-range LJ part needs a kspace style, as any coulLong pair does
      this.cutCoul = 0;
    }
  }

  coeff(args: string[]): void {
    if (args.length === 6) {
      throw new StyleError('pair_coeff cutoff2 (a per-pair Coulomb cutoff) is not implemented for pair style lj/long/coul/long');
    }
    if (this.ljLong && args.length === 5) {
      throw new StyleError('pair_coeff cutoff1 (a per-pair LJ cutoff) is not allowed with flag_lj long (one global LJ cutoff)');
    }
    super.coeff(args);
  }

  initOne(i: number, j: number): number {
    if (this.ljLong) {
      if (this.shift) throw new StyleError('pair_modify shift yes is not supported with flag_lj long (shift applies to flag_lj cut)');
      if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style lj/long/coul/long');
      if (this.tableDisp !== 0) throw new StyleError('pair_style lj/long/coul/long flag_lj long needs pair_modify table/disp 0 (the dispersion table is not implemented)');
    }
    return super.initOne(i, j);
  }

  compute(pc: PairCompute): void {
    if (!this.ljLong) { super.compute(pc); return; }
    this.computeLongLJ(pc);
  }

  /** flag_lj long: LJ repulsion, damped dispersion (real space) and the Coulomb real-space term. */
  private computeLongLJ(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { lj3, lj4 } = this;
    const cutljsq = this.cutGlobal * this.cutGlobal;
    const coul = this.cutCoul > 0;
    const cutcsq = this.cutCoul * this.cutCoul;
    const g = this.gEwald;
    const gd = this.gEwaldDisp;
    const gd6 = gd ** 6;
    const tab = coul ? this.ljErfcTables.get(this.table, g, cutcsq) : null;
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
        if (coul && rsq < cutcsq && qi !== 0 && q[j] !== 0) {
          const r = coulShort(rsq, qi, q[j], g, pc.qqrd2e, sC[sb], this.table === 0, tab);
          fpair += r.f;
          ecoul += r.e;
          e += r.e;
        }
        if (rsq < cutljsq) {
          if (sb !== 0 && sLJ[sb] !== 1) {
            throw new StyleError('special_bonds lj weight other than 1 is not implemented with pair style lj/long/coul/long flag_lj long');
          }
          const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
          const x2 = gd * gd * rsq;
          const g6 = g6kernel(x2);
          const A = lj3[t], C6 = lj4[t];
          // E = A/r^12 - C6 g6/r^6 ;  fpair = -(dE/dr)/r with d/dr [g6/r^6] giving g^6 exp(-x^2)/r^2 extra
          fpair += 12 * A * r6inv * r6inv * r2inv - C6 * (6 * g6 * r6inv * r2inv + gd6 * Math.exp(-x2) * r2inv);
          const ev = A * r6inv * r6inv - C6 * g6 * r6inv;
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

  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutCoul;
    return super.extract(name);
  }
}
