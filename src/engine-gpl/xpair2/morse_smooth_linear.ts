/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_morse_smooth_linear.cpp
 * and pair_morse_smooth_linear.h (PairMorseSmoothLinear: pair_style morse/smooth/linear). Contributing author (from the
 * source): Axel Kohlmeyer (Temple University). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The Morse potential shifted and tilted so that energy and force vanish at the cutoff (pair_morse.html, morse/smooth/linear):
 * the Morse term minus its value at r_c, plus the linear term (r - r_c) times the force at the cutoff. The energy is
 * built from the native formulas (offset and der_at_cutoff in init_one). Like the native style, pair_modify shift and tail
 * have no effect (pair_morse.rst: "does not support" the shift and tail options).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import { exp } from '../libm/exp';
import { forEachTypePair } from './common';

export class PairMorseSmoothLinear extends Pair {
  readonly name: string = 'morse/smooth/linear';
  virialFdotr = true;
  cutGlobal = 0;
  private p!: PairParams;
  private morse1 = new Float64Array(0);
  private derAtCutoff = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[]): void {
    // pair_morse_smooth_linear.cpp settings(): if (narg != 1) error->all(FLERR,"Illegal pair_style command");
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings(): every explicitly set pair takes the new global cutoff ("reset cutoffs that have been explicitly set")
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['d0', 'alpha', 'r0', 'cut']);
    const nt = ntypes + 1;
    this.morse1 = new Float64Array(nt * nt);
    this.derAtCutoff = new Float64Array(nt * nt);
    this.offset = new Float64Array(nt * nt);
  }

  coeff(args: string[]): void {
    // pair_morse_smooth_linear.cpp coeff(): narg 5 or 6 (I J d0 alpha r0 [cutoff])
    if (args.length < 5 || args.length > 6) throw new StyleError('Incorrect args for pair coefficients');
    const d0 = parseNum(args[2], 'd0');
    const alpha = parseNum(args[3], 'alpha');
    const r0 = parseNum(args[4], 'r0');
    const cutOne = args.length === 6 ? parseNum(args[5], 'cutoff') : this.cutGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [d0, alpha, r0, cutOne]));
  }

  initStyle(ctx: StyleContext): void {
    // pair_morse.html: "does not support" shift and tail; native ignores them (see noTermLog)
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    // init_one(): no mixing; the pair must be set explicitly
    if (!p.isSet(i, j)) throw new StyleError('All pair coeffs are not set');
    const d0 = p.get('d0', i, j), alpha = p.get('alpha', i, j), r0 = p.get('r0', i, j), cut = p.get('cut', i, j);
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.morse1[k1] = 2.0 * d0 * alpha;
    const alphaDr = -alpha * (cut - r0);
    this.offset[k1] = d0 * (exp(2.0 * alphaDr) - 2.0 * exp(alphaDr));
    this.derAtCutoff[k1] = -2.0 * alpha * d0 * (exp(2.0 * alphaDr) - exp(alphaDr));
    this.morse1[k2] = this.morse1[k1];
    this.derAtCutoff[k2] = this.derAtCutoff[k1];
    this.offset[k2] = this.offset[k1];
    p.setMixed(j, i, 'd0', d0);
    p.setMixed(j, i, 'alpha', alpha);
    p.setMixed(j, i, 'r0', r0);
    p.setMixed(j, i, 'cut', cut);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const d0 = this.p.p('d0'), alpha = this.p.p('alpha'), r0 = this.p.p('r0'), cut = this.p.p('cut');
    const { cutsq, morse1, derAtCutoff, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const t = itype * nt + type[j];
        if (rsq < cutsq[t]) {
          const r = Math.sqrt(rsq);
          const dr = r - r0[t];
          const dexp = exp(-alpha[t] * dr);
          const fpartial = morse1[t] * (dexp * dexp - dexp) / r;
          const fpair = factorLJ * (fpartial + derAtCutoff[t] / r);
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          let e = d0[t] * (dexp * dexp - 2.0 * dexp) - offset[t];
          e -= (r - cut[t]) * derAtCutoff[t];
          e *= factorLJ;
          evdwl += e;
          if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const t = itype * (this.ntypes + 1) + jtype;
    const d0 = this.p.get('d0', itype, jtype), alpha = this.p.get('alpha', itype, jtype);
    const r0 = this.p.get('r0', itype, jtype), cut = this.p.get('cut', itype, jtype);
    const r = Math.sqrt(rsq);
    let dr = r - r0;
    const dexp = exp(-alpha * dr);
    const fforce = factorLJ * (this.morse1[t] * (dexp * dexp - dexp) + this.derAtCutoff[t]) / r;
    let phi = d0 * (dexp * dexp - 2.0 * dexp) - this.offset[t];
    dr = cut - r;
    phi += dr * this.derAtCutoff[t];
    return { eng: factorLJ * phi, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('d0', i, i))} ${fmtCoeff(this.p.get('alpha', i, i))} ${fmtCoeff(this.p.get('r0', i, i))}`);
    }
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('d0', i, j))} ${fmtCoeff(this.p.get('alpha', i, j))} ${fmtCoeff(this.p.get('r0', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'd0' || name === 'r0' || name === 'alpha') return this.p.p(name);
    return undefined;
  }
}
