import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { libm } from '../../libm';

/*
 * pair_style pedone, docs.lammps.org/pair_pedone.html (text: plans/lammps-docs-live/pair_pedone.rst).
 *
 * Syntax of the doc: pair_style style args, with the args of pedone being the cutoff, documented as
 * "cutoff = global cutoff for Pedone interactions (distance units)".
 *
 * Energy: the doc formula, non-Coulomb part only (the Coulomb term is left to a Coulomb pair style
 * through hybrid/overlay, as the doc's example does):
 *
 *   E = D0 [ exp(-2 alpha (r - r0)) - 2 exp(-alpha (r - r0)) ] + B0 / r^12,   r < rc
 *
 * The coefficient list of the doc is D0, alpha, r0, C0, cutoff (distance units for the cutoff). The
 * fourth coefficient is the prefactor of the r^-12 term (the formula writes it B0). The doc says that
 * "The last coefficient is optional.  If not specified, the global *pedone* cutoff is used."
 *
 * Measured with native LAMMPS (black box): pedone energies of a dimer agree with the formula to about
 * 1e-15 relative for D0 0.030211, alpha 2.241334, r0 2.923245, prefactor 5.0 at r = 1.5, 2.0, 2.5, 3.0,
 * 3.5, 4.9 and 5.1 (the last with a 5.0 pair cutoff gives 0). An unset I,J pair gives the native error
 * All pair coeffs are not set.
 *
 * Mixing: the doc says "This pair style does not support mixing." Shift: the doc says that this pair
 * style supports the pair_modify shift option for the energy of the pair interaction; the offset is
 * measured (see shiftOffset). Tail is refused. Special bonds: the Morse style this file is modelled on
 * weights the energy and force with the LJ special factor; that choice is not measured here.
 */

const pairTerms = (d0: number, alpha: number, r0: number, b0: number, r: number) => {
  const dr = r - r0;
  const e1 = libm.exp(-2 * alpha * dr), e2 = libm.exp(-alpha * dr);
  const r2 = 1 / (r * r);
  const r12 = r2 ** 6;
  return {
    // -dE/dr = 2 alpha D0 (e1 - e2) + 12 B0 / r^13 ; the force is -dE/dr along rhat
    dEdr: -(2 * alpha * d0 * (e1 - e2)) - 12 * b0 * r12 / r,
    e: d0 * (e1 - 2 * e2) + b0 * r12,
  };
};

/** Energy shift of pair_modify shift yes (see the measured note in initOne). */
const shiftOffset = (d0: number, alpha: number, r0: number, b0: number, rc: number): number => {
  const dr = rc - r0;
  return d0 * (libm.exp(-2 * alpha * dr) - 2 * libm.exp(-alpha * dr)) - b0 / rc ** 12;
};

export class PairPedone extends Pair {
  readonly name: string = 'pedone';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  private d0 = new Float64Array(0);
  private alpha = new Float64Array(0);
  private r0 = new Float64Array(0);
  private b0 = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style pedone cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['D0', 'alpha', 'r0', 'B0', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 6 || args.length > 7) throw pairCoeffArgs('usage: pair_coeff I J D0 alpha r0 C0 [cutoff]');
    const vals = args.slice(2, 6).map((w, k) => parseNum(w, ['D0', 'alpha', 'r0', 'C0'][k]));
    const cut = args[6] !== undefined ? parseNum(args[6], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [...vals, cut]);
  }

  initStyle(_ctx: StyleContext): void {
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    const nt = this.ntypes + 1;
    if (this.d0.length !== nt * nt) {
      this.d0 = new Float64Array(nt * nt);
      this.alpha = new Float64Array(nt * nt);
      this.r0 = new Float64Array(nt * nt);
      this.b0 = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const cut = p.get('cut', i, j);
    const rc = Number.isNaN(cut) ? this.cutGlobal : cut;
    const k1 = i * nt + j, k2 = j * nt + i;
    const d0 = p.get('D0', i, j), alpha = p.get('alpha', i, j), r0 = p.get('r0', i, j), b0 = p.get('B0', i, j);
    this.d0[k1] = this.d0[k2] = d0;
    this.alpha[k1] = this.alpha[k2] = alpha;
    this.r0[k1] = this.r0[k2] = r0;
    this.b0[k1] = this.b0[k2] = b0;
    // Measured with native LAMMPS (black box): with pair_modify shift yes the offset at rc is
    // D0 (e^{-2 alpha (rc-r0)} - 2 e^{-alpha (rc-r0)}) - B0 / rc^12, i.e. the r^-12 term enters with the
    // opposite sign to the energy (dimers of 1-2 and 2-2 pairs agree to 1e-15 relative).
    this.offset[k1] = this.offset[k2] = this.shift && rc > 0 ? shiftOffset(d0, alpha, r0, b0, rc) : 0;
    return rc;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, d0, alpha, r0, b0, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const { dEdr, e } = pairTerms(d0[t], alpha[t], r0[t], b0[t], r);
        const fpair = (-factor * dEdr) / r;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const eij = factor * (e - offset[t]);
        evdwl += eij;
        if (tally) tallyAtom(pc, i, j, eij, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number): { eng: number; fforce: number } {
    const t = itype * (this.ntypes + 1) + jtype;
    const r = Math.sqrt(rsq);
    const { dEdr, e } = pairTerms(this.d0[t], this.alpha[t], this.r0[t], this.b0[t], r);
    return { fforce: (-factorLJ * dEdr) / r, eng: factorLJ * (e - this.offset[t]) };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('D0', i, i))} ${fmtCoeff(this.p.get('alpha', i, i))} ${fmtCoeff(this.p.get('r0', i, i))} ${fmtCoeff(this.p.get('B0', i, i))}`);
    }
    return out;
  }
}
