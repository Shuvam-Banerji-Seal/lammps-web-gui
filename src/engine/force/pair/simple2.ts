import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { fmtCoeff, parseNum } from '../util';
import { tallyAtom } from './lj_cut';
import { libm } from '../../libm';

/*
 * pair_style soft cutoff — docs.lammps.org/pair_soft.html:
 *   E = A \left[ 1 + \cos\left(\frac{\pi r}{r_c}\right) \right]
 *   \qquad r < r_c
 * coefficients:
 *   A (energy units)
 *   cutoff (distance units)
 * "The last coefficient is optional.  If not specified, the global soft
 * cutoff is used."
 * "For atom type pairs I,J and I != J, the A coefficient and cutoff
 * distance for this pair style can be mixed.  A is always mixed via a
 * geometric rule.  The cutoff is mixed according to the pair_modify
 * mix value.  The default mix value is geometric."
 * "This pair style does not support the pair_modify shift option, since
 * the pair interaction goes to 0.0 at the cutoff."
 *
 * Force from the documented energy: dE/dr = -A (pi/rc) sin(pi r / rc), so
 * f(r) = -(1/r) dE/dr = A (pi/rc) sin(pi r / rc) / r (repulsive, finite at r=0).
 */
export class PairSoft extends Pair {
  readonly name: string = 'soft';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  private pa = new Float64Array(0);
  private pk = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style soft cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 4) throw pairCoeffArgs('usage: pair_coeff I J A [cutoff]');
    const a = parseNum(args[2], 'A');
    const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a, cut]);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      // "A is always mixed via a geometric rule" (independent of pair_modify mix)
      const a = mixEpsilon('geometric', p.get('a', i, i), p.get('a', j, j), 0, 0);
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'a', a);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.pa.length !== nt * nt) {
      this.pa = new Float64Array(nt * nt);
      this.pk = new Float64Array(nt * nt);
    }
    const a = p.get('a', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.pa[k1] = this.pa[k2] = a;
    this.pk[k1] = this.pk[k2] = Math.PI / cut;
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, pa, pk } = this;
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
        const r = Math.sqrt(rsq);
        const arg = r * pk[t];
        const fpair = factor * pa[t] * pk[t] * libm.sin(arg) / r;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * pa[t] * (1 + libm.cos(arg));
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('a', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('a', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style yukawa kappa cutoff — docs.lammps.org/pair_yukawa.html:
 *   E = A \frac{e^{- \kappa r}}{r} \qquad r < r_c
 * coefficients:
 *   A (energy\*distance units)
 *   cutoff (distance units)
 * "The last coefficient is optional.  If not specified, the global yukawa
 * cutoff is used."
 * "For atom type pairs I,J and I != J, the A coefficient and cutoff
 * distance for this pair style can be mixed.  A is an energy value mixed
 * like a LJ epsilon.  The default mix value is geometric."
 * "This pair style supports the pair_modify shift option for the energy
 * of the pair interaction."
 * "This pair style does not support the pair_modify tail option for
 * adding long-range tail corrections to energy and pressure."
 *
 * Force from the documented energy: dE/dr = -A e^(-kappa r) (kappa/r + 1/r^2),
 * so f(r) = -(1/r) dE/dr = A e^(-kappa r) (kappa/r^2 + 1/r^3).
 */
export class PairYukawa extends Pair {
  readonly name: string = 'yukawa';
  virialFdotr = true;
  kappa = 0;
  cutGlobal = 0;
  p!: PairParams;
  private pa = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 2) throw new StyleError('usage: pair_style yukawa kappa cutoff');
    this.kappa = parseNum(args[0], 'kappa');
    this.cutGlobal = parseNum(args[1], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 3 || args.length > 4) throw pairCoeffArgs('usage: pair_coeff I J A [cutoff]');
    const a = parseNum(args[2], 'A');
    const cut = args[3] !== undefined ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a, cut]);
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'tail');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      // "A is an energy value mixed like a LJ epsilon"; the cutoff mixes like sigma
      const a = mixEpsilon(this.mix, p.get('a', i, i), p.get('a', j, j), 1, 1);
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'a', a);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.pa.length !== nt * nt) {
      this.pa = new Float64Array(nt * nt);
      this.offset = new Float64Array(nt * nt);
    }
    const a = p.get('a', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.pa[k1] = this.pa[k2] = a;
    // pair_modify shift: subtract the pair energy at the cutoff
    this.offset[k1] = this.offset[k2] = this.shift && cut > 0 ? a * libm.exp(-this.kappa * cut) / cut : 0;
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, pa, offset, kappa } = this;
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
        const r = Math.sqrt(rsq);
        const ir = 1 / r, ir2 = 1 / rsq;
        const ek = libm.exp(-kappa * r);
        const fpair = factor * pa[t] * ek * (kappa * ir2 + ir2 * ir);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * (pa[t] * ir * ek - offset[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('a', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('a', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style gauss cutoff — docs.lammps.org/pair_gauss.html:
 *   E = - A \exp(-B r^2) \qquad r < r_c
 * coefficients:
 *   A (energy units)
 *   B (1/distance^2 units)
 *   cutoff (distance units)
 * "The last coefficient is optional. If not specified, the global cutoff
 * is used."
 * "For atom type pairs I,J and I != J, the A, B, H, sigma_h, r_mh
 * parameters, and the cutoff distance for these pair styles can be mixed:"
 * "The default mix value is geometric. Only arithmetic and
 * geometric mix values are supported."
 * "The A and H parameters are mixed using the same rules normally used to
 * mix the "epsilon" parameter in a Lennard Jones interaction. The sigma_h,
 * r_mh, and the cutoff distance are mixed using the same rules used to mix
 * the "sigma" parameter in a Lennard Jones interaction."
 * "The B parameter is converted to a distance (sigma), before mixing
 * (using sigma=B^-0.5), and converted back to a coefficient
 * afterwards (using B=sigma^2)." — B carries 1/distance^2 units, so the
 * documented round trip is sigma = B^-0.5 mixed, then B = sigma^-2.
 * "Negative A values are converted to positive A values (using abs(A))
 * before mixing, and converted back after mixing
 * (by multiplying by min(sign(Ai),sign(Aj)))."
 * pair_modify shift: as native LAMMPS does (src/EXTRA-PAIR/pair_gauss.cpp init_one and compute at b332f28), offset =
 * A exp(-B rc^2) and E = -(A exp(-B r^2) - offset); tail adds nothing (no etail_ij).
 *
 * Force from the documented energy: dE/dr = 2 A B r e^(-B r^2), so
 * f(r) = -(1/r) dE/dr = -2 A B e^(-B r^2) (attractive for A > 0).
 */
export class PairGauss extends Pair {
  readonly name: string = 'gauss';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  private pa = new Float64Array(0);
  private pb = new Float64Array(0);
  private po = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style gauss cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['a', 'b', 'cut']);
  }

  coeff(args: string[]): void {
    if (args.length < 4 || args.length > 5) throw pairCoeffArgs('usage: pair_coeff I J A B [cutoff]');
    const a = parseNum(args[2], 'A');
    const b = parseNum(args[3], 'B');
    const cut = args[4] !== undefined ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [a, b, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.mix === 'sixthpower') throw new StyleError('pair_modify mix sixthpower is not supported for pair style gauss (only arithmetic and geometric)');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const ai = p.get('a', i, i), aj = p.get('a', j, j);
      const sign = Math.min(Math.sign(ai), Math.sign(aj));
      const a = mixEpsilon(this.mix, Math.abs(ai), Math.abs(aj), 1, 1) * sign;
      const s = mixDistance(this.mix, p.get('b', i, i) ** -0.5, p.get('b', j, j) ** -0.5);
      const b = 1 / (s * s);
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'a', a);
      p.setMixed(i, j, 'b', b);
      p.setMixed(i, j, 'cut', c);
    }
    const nt = this.ntypes + 1;
    if (this.pa.length !== nt * nt) {
      this.pa = new Float64Array(nt * nt);
      this.pb = new Float64Array(nt * nt);
      this.po = new Float64Array(nt * nt);
    }
    const a = p.get('a', i, j), b = p.get('b', i, j), cut = p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.pa[k1] = this.pa[k2] = a;
    this.pb[k1] = this.pb[k2] = b;
    this.po[k1] = this.po[k2] = this.shift ? a * libm.exp(-b * cut * cut) : 0;
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, pa, pb, po } = this;
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
        const ek = libm.exp(-pb[t] * rsq);
        const fpair = factor * (-2 * pa[t] * pb[t]) * ek;
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const e = factor * -(pa[t] * ek - po[t]);
        evdwl += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.evdwl += evdwl;
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('a', i, i))} ${fmtCoeff(this.p.get('b', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('a', i, j))} ${fmtCoeff(this.p.get('b', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }
}

/*
 * pair_style zero cutoff [nocoeff] [full] — docs.lammps.org/pair_zero.html:
 *   pair_style zero cutoff [nocoeff] [full]
 *   * nocoeff = ignore all pair_coeff parameters (optional)
 *   * full = build full neighbor list (optional)
 * "Define a global or per-type cutoff length for the purpose of
 * building a neighbor list and acquiring ghost atoms, but do
 * not compute any pairwise forces or energies."
 * coefficients:
 *   cutoff (distance units)
 * "This coefficient is optional.  If not specified, the global cutoff
 * specified in the pair_style command is used. If the pair_style has
 * been specified with the optional nocoeff flag, then a cutoff
 * pair coefficient is ignored."
 * "The optional nocoeff flag allows to read data files with a PairCoeff
 * section for any pair style. Similarly, any pair_coeff commands
 * will only be checked for the atom type numbers and the rest ignored.
 * In this case, only the global cutoff will be used."
 * "The cutoff distance for this pair style can be mixed.  The default mix
 * value is geometric."
 * "This pair style does not support the pair_modify shift, table, and tail
 * options."
 */
export class PairZero extends Pair {
  readonly name: string = 'zero';
  virialFdotr = true;
  cutGlobal = 0;
  nocoeff = false;
  p!: PairParams;

  settings(args: string[]): void {
    if (args.length < 1) throw new StyleError('usage: pair_style zero cutoff [nocoeff] [full]');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
    this.nocoeff = false;
    this.needsHalf = true;
    this.needsFull = false;
    for (let k = 1; k < args.length; k++) {
      if (args[k] === 'nocoeff') this.nocoeff = true;
      else if (args[k] === 'full') { this.needsFull = true; this.needsHalf = false; }
      else throw new StyleError(`Unknown pair style zero option ${args[k]}`); // As native LAMMPS does (src/pair_zero.cpp PairZero::settings at b332f28)
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['cut']);
  }

  coeff(args: string[]): void {
    // As native LAMMPS does (src/pair_zero.cpp PairZero::coeff at b332f28): with nocoeff "any pair_coeff commands will
    // only be checked for the atom type numbers and the rest ignored"; the pairs are set with the global cutoff
    if (args.length < 2 || (!this.nocoeff && args.length > 3)) throw pairCoeffArgs('usage: pair_coeff I J [cutoff]');
    const cut = !this.nocoeff && args[2] !== undefined ? parseNum(args[2], 'cutoff') : this.cutGlobal;
    this.p.setRange(args[0], args[1], [cut]);
  }

  initStyle(_ctx: StyleContext): void {
  }

  initOne(i: number, j: number): number {
    if (this.nocoeff) return this.cutGlobal;
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
      const c = mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j));
      p.setMixed(i, j, 'cut', c);
    }
    return p.get('cut', i, j);
  }

  // write_data, measured with native LAMMPS: "Pair Coeffs" lists bare type numbers and "PairIJ
  // Coeffs" lists "I J cutoff".
  dataCoeffs(): string[] {
    return Array.from({ length: this.ntypes }, (_, k) => `${k + 1}`);
  }

  dataCoeffsIJ(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) out.push(`${i} ${j} ${fmtCoeff(this.nocoeff ? this.cutGlobal : this.p.get('cut', i, j))}`);
    }
    return out;
  }

  /** As native LAMMPS does (src/pair_zero.cpp at b332f28: single_enable = 1, single() returns 0 with fforce 0). */
  single(): { eng: number; fforce: number } {
    return { eng: 0, fforce: 0 };
  }

  compute(_pc: PairCompute): void {
    // "but do not compute any pairwise forces or energies"
  }
}

/*
 * pair_style zero/coul cutoff [nocoeff] [full] — docs.lammps.org/pair_zero.html: "Pair style *zero/coul*
 * behaves exactly like pair style *zero*, but additionally presents itself as a Coulombic pair style: it
 * declares compatibility with kspace styles and provides its cutoff as the real-space Coulomb cutoff, while
 * still computing no pairwise interactions." Its kspace compatibility is also limited: "pair style *zero/coul*
 * cannot be used with kspace styles that require a pair style providing dispersion or TIP4P support".
 * Measured with native LAMMPS (black box): pair_style zero/coul 6.0 with kspace_style ewald (gewald 0.4)
 * gives E_coul 0 and E_long -1288.45251918596, the same E_long as lj/cut/coul/long with the same kspace;
 * with kspace_style pppm (slab 3.0) E_long was -1416.87452190853, again the pppm value of the lj/cut/coul/long
 * case. pair_modify shift yes and tail yes were accepted with no effect (no energy change). Measured: with
 * kspace_style pppm/disp or pppm/tip4p the error is KSpace style pppm/disp is incompatible with Pair style zero/coul.
 * pair_style zero with kspace_style zero is refused, KSpace style is incompatible with Pair style, because
 * pair style zero is not Coulombic (measured).
 */
export class PairZeroCoul extends PairZero {
  readonly name: string = 'zero/coul';

  /** The global cutoff is the real-space Coulomb cutoff that kspace styles read (cut_coul). */
  extract(name: string): unknown {
    if (name === 'cut_coul') return this.cutGlobal;
    return super.extract(name);
  }

  /** shift and tail are accepted with no effect (measured); zero interactions have nothing to shift. */
  initStyle(_ctx: StyleContext): void {
    // no-op: see the header comment
  }
}
