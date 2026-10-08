import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { fmtCoeff, parseNum } from '../util';

/*
 * pair_style lj/cut cutoff — docs.lammps.org/pair_lj.html:
 *   E = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]   r < rc
 *   coefficients epsilon (energy units), sigma (distance units), LJ cutoff
 *   (distance units). "The last coefficient is optional.  If not specified,
 *   the global LJ cutoff specified in the pair_style command is used."
 *   "For atom type pairs I,J and I != J, the epsilon and sigma coefficients
 *   and cutoff distance for all of the lj/cut pair styles can be mixed."
 *   Supports pair_modify shift and tail.
 * Tail correction: docs.lammps.org/pair_modify.html "The formulas used for
 * the long-range corrections come from equation 5 of (Sun)":
 *   etail = 2 pi / V sum_ij N_i N_j  int_rc^inf u(r) r^2 dr
 *   ptail = -2 pi / (3 V^2) sum_ij N_i N_j  int_rc^inf r^3 u'(r) dr
 * evaluated in closed form for the 12-6 potential.
 */

export class PairLJCut extends Pair {
  readonly name: string = 'lj/cut';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  lj1 = new Float64Array(0);
  lj2 = new Float64Array(0);
  lj3 = new Float64Array(0);
  lj4 = new Float64Array(0);
  offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError(`usage: pair_style ${this.name} cutoff`);
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('usage: pair_coeff I J epsilon sigma [cutoff]');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [eps, sig, cut]);
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const e = mixEpsilon(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), p.get('sigma', i, i), p.get('sigma', j, j));
      const s = mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j));
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'epsilon', e);
      p.setMixed(i, j, 'sigma', s);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.lj1.length !== nt * nt) {
      this.lj1 = new Float64Array(nt * nt); this.lj2 = new Float64Array(nt * nt);
      this.lj3 = new Float64Array(nt * nt); this.lj4 = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    const s6 = sig ** 6, s12 = s6 * s6;
    this.lj1[k1] = this.lj1[k2] = 48 * eps * s12;
    this.lj2[k1] = this.lj2[k2] = 24 * eps * s6;
    this.lj3[k1] = this.lj3[k2] = 4 * eps * s12;
    this.lj4[k1] = this.lj4[k2] = 4 * eps * s6;
    if (this.shift && cut > 0) {
      const r = sig / cut;
      this.offset[k1] = this.offset[k2] = 4 * eps * (r ** 12 - r ** 6);
    } else this.offset[k1] = this.offset[k2] = 0;
    return cut;
  }

  tailSums(count: Float64Array): { etail: number; ptail: number } {
    if (!this.tail) return { etail: 0, ptail: 0 };
    let e = 0, pr = 0;
    const nt = this.ntypes + 1;
    for (let i = 1; i < nt; i++) {
      for (let j = 1; j < nt; j++) {
        const eps = this.p.get('epsilon', i, j), sig = this.p.get('sigma', i, j), rc = this.cut[i * nt + j];
        if (!(rc > 0)) continue;
        const s6 = sig ** 6, s12 = s6 * s6, rc3 = rc ** 3, rc9 = rc3 ** 3;
        const nn = count[i] * count[j];
        e += nn * 4 * eps * (s12 / (9 * rc9) - s6 / (3 * rc3));
        pr += nn * 4 * eps * (-4 * s12 / (3 * rc9) + 2 * s6 / rc3);
      }
    }
    // caller divides by V (energy) and V^2... returns the V-free sums
    return { etail: 2 * Math.PI * e, ptail: -2 * Math.PI / 3 * pr };
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, lj1, lj2, lj3, lj4, offset } = this;
    const sLJ = pc.specialLJ;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = 0; i < list.inum; i++) {
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
        const r2inv = 1 / rsq;
        const r6inv = r2inv * r2inv * r2inv;
        const fpair = factor * r6inv * (lj1[t] * r6inv - lj2[t]) * r2inv;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (r6inv * (lj3[t] * r6inv - lj4[t]) - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number) {
    const t = itype * (this.ntypes + 1) + jtype;
    const r2inv = 1 / rsq, r6inv = r2inv * r2inv * r2inv;
    return {
      fforce: factorLJ * r6inv * (this.lj1[t] * r6inv - this.lj2[t]) * r2inv,
      eng: factorLJ * (r6inv * (this.lj3[t] * r6inv - this.lj4[t]) - this.offset[t]),
    };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'epsilon') return this.p.p('epsilon');
    if (name === 'sigma') return this.p.p('sigma');
    return undefined;
  }
}

/** Per-atom energy/virial split half and half (only when requested). */
export const tallyAtom = (pc: PairCompute, i: number, j: number, e: number, fpair: number, dx: number, dy: number, dz: number): void => {
  if (pc.eatom) { pc.eatom[i] += 0.5 * e; pc.eatom[j] += 0.5 * e; }
  if (pc.vatom) {
    const va = pc.vatom;
    const w = [dx * dx, dy * dy, dz * dz, dx * dy, dx * dz, dy * dz];
    for (let c = 0; c < 6; c++) {
      const h = 0.5 * w[c] * fpair;
      va[6 * i + c] += h; va[6 * j + c] += h;
    }
  }
};
