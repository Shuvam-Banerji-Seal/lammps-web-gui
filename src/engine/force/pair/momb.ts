import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { libm } from '../../libm';

/*
 * pair_style momb (EXTRA-PAIR), docs.lammps.org/pair_momb.html (text: plans/lammps-docs-live/pair_momb.rst).
 *
 * The doc's syntax: "pair_style momb cutoff s6 d" with "cutoff = global cutoff (distance units)", "s6 = global
 * scaling factor of the exchange-correlation functional used (unitless)" and "d = damping scaling factor of
 * Grimme's method (unitless)". The coefficients are D0, alpha, r0, C6 and Rr, with no per-pair cutoff. The energy
 * the doc gives is E = D0 [exp(-2 alpha (r - r0)) - 2 exp(-alpha (r - r0))] - s6 C6/r^6 f_damp(r, Rr), with
 * f_damp = 1/(1 + exp(-d (r/Rr - 1))). Measured with native LAMMPS (black box): dimers at r = 1.5, 2.0, 2.5, 3.0,
 * 4.0 and 6.0 (D0 0.3, alpha 1.2, r0 2.5, C6 10.2847, Rr 2.361, s6 0.75, d 20) agree with this formula to 1e-15
 * absolute.
 *
 * Mixing: the doc gives no mixing rule; native refuses a pair that is not set ("All pair coeffs are not set").
 * Special bonds: measured with native LAMMPS (black box) on a bonded dimer: the energy scales with the special LJ
 * factor (1-2 pair energy 0.8338 at factor 1.0, 0.4169 at 0.5), so the factor multiplies the whole pair term.
 * pair_modify shift yes: measured, the offset is the Morse term at the cutoff alone, E(r) - D0 [exp(-2 alpha
 * (rc - r0)) - 2 exp(-alpha (rc - r0))] (dimer at r = 4.0, cutoff 5.0: the difference is -0.0291286, equal to the
 * Morse value at 5.0); the dispersion term is not shifted. Tail is refused (not documented).
 */

export class PairMomb extends Pair {
  readonly name: string = 'momb';
  virialFdotr = true;
  cutGlobal = 0;
  s6 = 0;
  dDamp = 0;
  p!: PairParams;
  private d0 = new Float64Array(0);
  private alpha = new Float64Array(0);
  private r0 = new Float64Array(0);
  private c6 = new Float64Array(0);
  private rr = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('usage: pair_style momb cutoff s6 d');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    this.s6 = parseNum(args[1], 's6');
    this.dDamp = parseNum(args[2], 'd');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['D0', 'alpha', 'r0', 'C6', 'Rr']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length !== 7) throw pairCoeffArgs('usage: pair_coeff I J D0 alpha r0 C6 Rr');
    const vals = args.slice(2, 7).map((w, k) => parseNum(w, ['D0', 'alpha', 'r0', 'C6', 'Rr'][k]));
    this.p.setRange(args[0], args[1], vals);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair style momb');
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j}; momb has no mixing rule)`);
    const nt = this.ntypes + 1;
    if (this.d0.length !== nt * nt) {
      this.d0 = new Float64Array(nt * nt); this.alpha = new Float64Array(nt * nt); this.r0 = new Float64Array(nt * nt);
      this.c6 = new Float64Array(nt * nt); this.rr = new Float64Array(nt * nt); this.offset = new Float64Array(nt * nt);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    const d0 = this.p.get('D0', i, j), alpha = this.p.get('alpha', i, j), r0 = this.p.get('r0', i, j);
    this.d0[k1] = this.d0[k2] = d0;
    this.alpha[k1] = this.alpha[k2] = alpha;
    this.r0[k1] = this.r0[k2] = r0;
    this.c6[k1] = this.c6[k2] = this.p.get('C6', i, j);
    this.rr[k1] = this.rr[k2] = this.p.get('Rr', i, j);
    // Measured with native LAMMPS (black box): the Morse term at the cutoff is the shift; the dispersion term is not.
    this.offset[k1] = this.offset[k2] = this.shift ? morse(d0, alpha, r0, this.cutGlobal).e : 0;
    return this.cutGlobal;
  }

  /** Energy and fpair (= -dE/dr / r) of the pair of type t at distance r (before the special factor). */
  private kernel(t: number, r: number): { e: number; fpair: number } {
    const m = morse(this.d0[t], this.alpha[t], this.r0[t], r);
    const disp = dispersion(this.s6, this.c6[t], this.rr[t], this.dDamp, r);
    return { e: m.e - disp.e - this.offset[t], fpair: -(m.de - disp.de) / r };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sLJ[jj >>> SBBITS];
        const r = Math.sqrt(rsq);
        const k2 = this.kernel(t, r);
        const fpair = factor * k2.fpair;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const eij = factor * k2.e;
        evdwl += eij;
        if (tally) tallyAtom(pc, i, j, eij, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number): { eng: number; fforce: number } {
    const t = itype * (this.ntypes + 1) + jtype;
    const k2 = this.kernel(t, Math.sqrt(rsq));
    return { eng: factorLJ * k2.e, fforce: factorLJ * k2.fpair };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('D0', i, i))} ${fmtCoeff(this.p.get('alpha', i, i))} ${fmtCoeff(this.p.get('r0', i, i))} ${fmtCoeff(this.p.get('C6', i, i))} ${fmtCoeff(this.p.get('Rr', i, i))}`);
    }
    return out;
  }
}

/** Morse-type term D0 [exp(-2 alpha x) - 2 exp(-alpha x)], x = r - r0: energy and dE/dr. */
const morse = (d0: number, alpha: number, r0: number, r: number): { e: number; de: number } => {
  const x = r - r0;
  const e1 = libm.exp(-alpha * x), e2 = e1 * e1;
  return { e: d0 * (e2 - 2 * e1), de: d0 * (-2 * alpha * e2 + 2 * alpha * e1) };
};

/** Dispersion term s6 C6/r^6 f_damp(r, Rr), f_damp = 1/(1 + exp(-d (r/Rr - 1))): energy and dE/dr. */
const dispersion = (s6: number, c6: number, rr: number, dDamp: number, r: number): { e: number; de: number } => {
  const u = libm.exp(-dDamp * (r / rr - 1));
  const f = 1 / (1 + u);
  const df = f * f * u * (dDamp / rr);
  const inv6 = 1 / Math.pow(r, 6);
  const g = c6 * inv6 * f;
  // d/dr [C6 r^-6 f] = -6 C6 r^-7 f + C6 r^-6 df/dr
  const dg = -6 * c6 * inv6 / r * f + c6 * inv6 * df;
  return { e: s6 * g, de: s6 * dg };
};
