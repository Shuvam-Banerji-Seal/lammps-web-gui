/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/FEP/pair_morse_soft.cpp and src/FEP/pair_morse_soft.h (class PairMorseSoft, pair morse/soft) at commit
 * b332f28, by Stefan Paquay (TU/e) and the LAMMPS developers. Copyright Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style morse/soft nlambda shift_range cutoff; pair_coeff I J d0 alpha r0 lambda [cutoff]
 * (docs.lammps.org/pair_morse_soft.html). The soft-core Morse energy for dr = r - r0, with
 * x0 = r0, V0 = D e^{-a dr} (e^{-a dr} - 2) and B = -2 D e^{-2 a x0} (e^{a x0} - 1) / 3:
 *   lambda >= shift_range:  E = V0 + B e^{-3 a dr} s1,  s1 = (lambda - 1) / (shift_range - 1)
 *   lambda <  shift_range:  E = (V0 + B e^{-3 a dr}) (lambda / shift_range)^nlambda
 * The force is -dE/dr / r with the same branch. Source notes kept from the file: the energy and force are
 * multiplied by the special_bonds factor of the pair; the mixing rule is not used (every type pair needs its
 * pair_coeff, "All pair coeffs are not set"); pair_modify tail is refused here because the source has no tail
 * term for this style. pair_modify shift is accepted and has no effect on the energy (the source's compute does not
 * subtract its offset; measured with native LAMMPS, black box: the energy with shift yes equals the energy without).
 *
 * Measured with native LAMMPS (black box) in tests/oracle/w69fep_morse.in.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { exp as libmExp } from '../libm/exp';

/** MathSpecial::powint: x^n for integer n by repeated squaring (the source's helper). */
const powint = (x: number, n: number): number => {
  if (x === 0) return 0;
  let nn = n > 0 ? n : -n;
  let ww = x;
  let yy = 1.0;
  for (; nn !== 0; nn >>= 1, ww *= ww) if (nn & 1) yy *= ww;
  return n > 0 ? yy : 1.0 / yy;
};

const VALUES = ['d0', 'alpha', 'r0', 'lambda', 'cut'];

export class PairMorseSoft extends Pair {
  readonly name = 'morse/soft';
  virialFdotr = true;
  private nlambda = 0;
  private shiftRange = 1;
  private cutGlobal = 0;
  p!: PairParams;
  private d0T = new Float64Array(0);
  private alphaT = new Float64Array(0);
  private r0T = new Float64Array(0);
  private lamT = new Float64Array(0);
  private cutT = new Float64Array(0);
  private cutsqT = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 3) throw new StyleError('usage: pair_style morse/soft nlambda shift_range cutoff');
    this.nlambda = parseNum(args[0], 'nlambda');
    this.shiftRange = parseNum(args[1], 'shift_range');
    this.cutGlobal = parseNum(args[2], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('pair morse/soft: the cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, VALUES);
  }

  coeff(args: string[]): void {
    if (args.length < 6 || args.length > 7) throw new StyleError('usage: pair_coeff I J d0 alpha r0 lambda [cutoff] for pair style morse/soft');
    const cut = args.length === 7 ? parseNum(args[6], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [parseNum(args[2], 'd0'), parseNum(args[3], 'alpha'), parseNum(args[4], 'r0'), parseNum(args[5], 'lambda'), cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail yes is not supported for pair style morse/soft (the source has no tail term for this style)');
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}, pair style morse/soft)`);
    const nt = this.ntypes + 1;
    if (this.d0T.length !== nt * nt) {
      const z = () => new Float64Array(nt * nt);
      this.d0T = z(); this.alphaT = z(); this.r0T = z(); this.lamT = z(); this.cutT = z(); this.cutsqT = z();
    }
    const p = this.p;
    const cut = p.get('cut', i, j);
    for (const [a, b] of [[i, j], [j, i]]) {
      const k = a * nt + b;
      this.d0T[k] = p.get('d0', i, j);
      this.alphaT[k] = p.get('alpha', i, j);
      this.r0T[k] = p.get('r0', i, j);
      this.lamT[k] = p.get('lambda', i, j);
      this.cutT[k] = cut;
      this.cutsqT[k] = cut * cut;
    }
    return cut;
  }

  /** Energy and F/r (force over r, before the special factor) of one pair at squared distance rsq, type index k. */
  private pairTerm(rsq: number, k: number): [number, number] {
    const r = Math.sqrt(rsq);
    const D = this.d0T[k], a = this.alphaT[k], x0 = this.r0T[k], l = this.lamT[k];
    const dr = r - x0;
    const dexp = libmExp(-a * dr);
    const dexp2 = dexp * dexp;
    const dexp3 = dexp2 * dexp;
    const ea = libmExp(a * x0);
    const iea2 = libmExp(-2 * a * x0);
    const V0 = D * dexp * (dexp - 2);
    const B = (-2 * D * iea2 * (ea - 1)) / 3;
    if (l >= this.shiftRange) {
      const s1 = (l - 1) / (this.shiftRange - 1);
      const phi = V0 + B * dexp3 * s1;
      const fpair = (3 * a * B * dexp3 * s1 + 2 * a * D * (dexp2 - dexp)) / r;
      return [phi, fpair];
    }
    const llf = powint(l / this.shiftRange, this.nlambda);
    const phi = (V0 + B * dexp3) * llf;
    if (r === 0) return [phi, 0];
    const fpair = ((3 * a * B * dexp3 + 2 * a * D * (dexp2 - dexp)) * llf) / r;
    return [phi, fpair];
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const fl = sLJ[jj >>> SBBITS];
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (!(rsq < this.cutsqT[t])) continue;
        const [phi, fr] = this.pairTerm(rsq, t);
        const fpair = fr * fl;
        const e = phi * fl;
        evdwl += e;
        f[3 * i] += dx * fpair; f[3 * i + 1] += dy * fpair; f[3 * i + 2] += dz * fpair;
        f[3 * j] -= dx * fpair; f[3 * j + 1] -= dy * fpair; f[3 * j + 2] -= dz * fpair;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
    }
    pc.acc.evdwl += evdwl;
  }
}
