import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, SBBITS } from '../../neighbor';
import { tallyAtom } from './lj_cut';
import { fmtCoeff, parseNum } from '../util';
import { libm } from '../../libm';

/*
 * pair_style gauss/cut and coul/diel (EXTRA-PAIR), docs.lammps.org/pair_gauss.html and
 * docs.lammps.org/pair_coul_diel.html (texts: plans/lammps-docs-live/pair_gauss.rst, pair_coul_diel.rst).
 *
 * gauss/cut. The doc says: "Style gauss/cut computes a generalized Gaussian interaction potential between pairs of particles:"
 * and gives E = H/(sigma_h sqrt(2 pi)) exp[-(r-r_mh)^2/(2 sigma_h^2)], with coefficients H (energy * distance
 * units), r_mh (distance units), sigma_h (distance units) and an optional cutoff. Measured with native LAMMPS
 * (black box): the dimer energy at r = 2.0, 3.2, 4.0, 5.5, 5.9 agrees with this formula to 1e-15 absolute
 * (H 1.5, r_mh 3.2, sigma_h 0.7, cutoff 6.0), and it is zero from the pair cutoff on.
 *   Mixing: the doc says that "The sigma_h, r_mh, and the cutoff distance are mixed using the same rules" as sigma
 *   and that the A and H parameters are mixed like epsilon. Measured with native LAMMPS (black box): H_ij = sqrt(H_i H_j)
 *   for both geometric and arithmetic; r_mh, sigma_h and the cutoff follow the geometric mean (sqrt) or the
 *   arithmetic mean, as pair_modify mix selects. The doc supports only arithmetic and geometric mixing; sixthpower
 *   is refused.
 *   Shift: the doc says "The gauss/cut style supports the pair_modify shift option for the energy of the Gauss-potential portion"
 *   Measured: with shift yes the energy is E(r) - E(rc) with rc the pair cutoff (0.000502671 at 5.9 becomes 0.000215892
 *   with cutoff 6.0 at the same r); mixed I,J pairs are shifted the same way (oracle).
 *   Special bonds: the Restrictions section of the doc describes the change to special_bonds factors in 28Mar2023
 *   for the gauss styles. Measured with native LAMMPS (black box) on a bonded dimer: the energy is 0.0627219772053445
 *   at a special LJ factor of 1.0 and 0.0313609886026722 at 0.5, so the factor multiplies the energy.
 *
 * coul/diel. The doc gives E = C q_i q_j / (eps r) (eps / eps_D(r) - 1) for r < r_c, with eps_D(r) a tanh of
 * (r - r_me)/sigma_e, and the coefficients epsilon, r_me and sigma_e. The doc says: "This pair style does not support parameter mixing."
 * Measured with native LAMMPS (black box): the energy is C q_i q_j (eps / eps_D(r) - 1) / r, i.e. the 1/eps of the
 * doc's prefactor is not present; the global dielectric (the dielectric command) divides C, as in the other Coulomb
 * styles. Dimers with eps = 78 and 20, r = 0.9 to 3.4, agree to 1e-15 relative. Epsilon: a J,J coefficient, when set,
 * supplies the epsilon of every I,J pair with I <= J (see initOne). Shift: pair_modify shift yes subtracts E(rc)
 * for like-type pairs only (offset 0.0022039 at rc = 2.0 with charges +1 and +1; no change for 1-2 dimers).
 * Mixing: none; an unset I,J pair is refused with native's text "All pair coeffs are not set". Tail is refused (the doc
 * says the tail option is not supported). Special bonds: the Coulomb factors 0 and 1 are used; other values are
 * refused (not measured).
 */

export class PairGaussCut extends Pair {
  readonly name: string = 'gauss/cut';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  private hh = new Float64Array(0);
  private rm = new Float64Array(0);
  private sg = new Float64Array(0);
  private offset = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style gauss/cut cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['H', 'rmh', 'sigma', 'cut']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw pairCoeffArgs('usage: pair_coeff I J H r_mh sigma_h [cutoff]');
    const vals = args.slice(2, 5).map((w, k) => parseNum(w, ['H', 'r_mh', 'sigma_h'][k]));
    if (!(vals[2] > 0)) throw new StyleError('pair gauss/cut: sigma_h must be > 0');
    const cut = args[5] !== undefined ? parseNum(args[5], 'cutoff') : Number.NaN;
    this.p.setRange(args[0], args[1], [...vals, cut]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.mix !== 'geometric' && this.mix !== 'arithmetic') throw new StyleError('pair gauss/cut supports only mix geometric or arithmetic');
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair style gauss/cut');
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      const hi = p.get('H', i, i), hj = p.get('H', j, j);
      if (hi < 0 || hj < 0) throw new StyleError('pair gauss/cut: mixing with a negative H is not supported');
      p.setMixed(i, j, 'H', mixEpsilon(this.mix, hi, hj, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'rmh', mixDistance(this.mix, p.get('rmh', i, i), p.get('rmh', j, j)));
      p.setMixed(i, j, 'sigma', mixDistance(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistance(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const nt = this.ntypes + 1;
    if (this.hh.length !== nt * nt) {
      this.hh = new Float64Array(nt * nt); this.rm = new Float64Array(nt * nt);
      this.sg = new Float64Array(nt * nt); this.offset = new Float64Array(nt * nt);
    }
    const cut = Number.isNaN(p.get('cut', i, j)) ? this.cutGlobal : p.get('cut', i, j);
    const k1 = i * nt + j, k2 = j * nt + i;
    this.hh[k1] = this.hh[k2] = p.get('H', i, j);
    this.rm[k1] = this.rm[k2] = p.get('rmh', i, j);
    this.sg[k1] = this.sg[k2] = p.get('sigma', i, j);
    this.offset[k1] = this.offset[k2] = 0;
    if (this.shift) this.offset[k1] = this.offset[k2] = gaussE(this.hh[k1], this.rm[k1], this.sg[k1], cut);
    return cut;
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const nt = this.ntypes + 1;
    const { cutsq, hh, rm, sg, offset } = this;
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
        const e0 = gaussE(hh[t], rm[t], sg[t], r);
        // -dE/dr / r = E (r - r_mh) / (sigma^2 r)
        const fpair = (factor * e0 * (r - rm[t])) / (sg[t] * sg[t] * r);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const eij = factor * (e0 - offset[t]);
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
    const e0 = gaussE(this.hh[t], this.rm[t], this.sg[t], r);
    return { fforce: (factorLJ * e0 * (r - this.rm[t])) / (this.sg[t] * this.sg[t] * r), eng: factorLJ * (e0 - this.offset[t]) };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('H', i, i))} ${fmtCoeff(this.p.get('rmh', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    }
    return out;
  }
}

/** Gauss/cut energy H/(sigma sqrt(2 pi)) exp(-(r - r_mh)^2 / (2 sigma^2)) for r < cut (0 beyond). */
const gaussE = (H: number, rmh: number, sigma: number, r: number): number => {
  const d = r - rmh;
  return (H / (sigma * Math.sqrt(2 * Math.PI))) * libm.exp(-(d * d) / (2 * sigma * sigma));
};

/** g(r) = (eps/eps_D(r) - 1)/r of the coul/diel energy E = C qi qj g(r) (the doc's prefactor 1/eps is absent, measured). */
const coulDielG = (eps: number, rme: number, se: number, r: number): number => {
  const epsD = (5.2 + eps) / 2 + ((eps - 5.2) / 2) * Math.tanh((r - rme) / se);
  return (eps / epsD - 1) / r;
};

/** dg/dr for coulDielG (analytic: eps_D' = (eps - 5.2)/(2 sigma_e) sech^2((r - r_me)/sigma_e)). */
const coulDielDg = (eps: number, rme: number, se: number, r: number): number => {
  const u = (r - rme) / se;
  const epsD = (5.2 + eps) / 2 + ((eps - 5.2) / 2) * Math.tanh(u);
  const depsD = ((eps - 5.2) / (2 * se)) / (Math.cosh(u) * Math.cosh(u));
  return (-(eps * depsD) / (epsD * epsD)) / r - (eps / epsD - 1) / (r * r);
};

export class PairCoulDiel extends Pair {
  readonly name: string = 'coul/diel';
  virialFdotr = true;
  cutGlobal = 0;
  p!: PairParams;
  private eps = new Float64Array(0);
  private rme = new Float64Array(0);
  private se = new Float64Array(0);
  /** 1 where pair_modify shift yes subtracts the energy at the cutoff (I,J with I = J only, see initOne). */
  private shiftOn = new Uint8Array(0);
  /** Coulomb prefactor of the current run (pc.qqrd2e); single() uses the last-seen value. */
  private qqrd2e = 1;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style coul/diel cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['eps', 'rme', 'se']);
  }

  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 5 || args.length > 6) throw pairCoeffArgs('usage: pair_coeff I J epsilon r_me sigma_e');
    const vals = args.slice(2, 5).map((w, k) => parseNum(w, ['epsilon', 'r_me', 'sigma_e'][k]));
    // As native LAMMPS does (src/EXTRA-PAIR/pair_coul_diel.cpp PairCoulDiel::coeff at b332f28): an optional 6th word is the per-pair cutoff, read as a number.
    // The engine has no per-pair cutoff for coul/diel, so a valid one is refused rather than ignored.
    if (args.length === 6) {
      parseNum(args[5], 'cutoff');
      throw new StyleError('pair coul/diel: a per-pair cutoff is not supported (use the global cutoff of pair_style coul/diel)');
    }
    if (!(vals[2] > 0)) throw new StyleError('pair coul/diel: sigma_e must be > 0');
    this.p.setRange(args[0], args[1], vals);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.tail) throw new StyleError('pair_modify tail is not supported by pair style coul/diel');
  }

  initOne(i: number, j: number): number {
    // Not mixable (doc: "This pair style does not support parameter mixing"); native's error text is used.
    if (!this.p.isSet(i, j)) throw new StyleError('All pair coeffs are not set. (coul/diel does not mix)');
    const nt = this.ntypes + 1;
    if (this.eps.length !== nt * nt) {
      this.eps = new Float64Array(nt * nt); this.rme = new Float64Array(nt * nt); this.se = new Float64Array(nt * nt);
      this.shiftOn = new Uint8Array(nt * nt);
    }
    const k1 = i * nt + j, k2 = j * nt + i;
    // Measured with native LAMMPS (black box): pair_modify shift yes changes the energy of like-type dimers
    // (E(r) - E(rc), e.g. 0.0022039 at rc = 2.0 with charges +1 and +1) but not of I,J dimers with I != J (no change
    // for 1-2 pairs whose own energy at rc is nonzero, with either set of diagonal parameters).
    this.shiftOn[k1] = this.shiftOn[k2] = this.shift && i === j ? 1 : 0;
    // Measured with native LAMMPS (black box): the epsilon of an I,J pair (I <= J) is the epsilon of the J,J
    // coefficient when that is set, whatever the I,J coefficient says (with 1-1, 1-2 and 2-2 all set, a 1-2
    // coefficient of eps = 20 gives the energy of eps = 78 set on 2-2; changing eps_11 leaves 1-2 unchanged; 1-3 and
    // 2-3 use eps_33; the atom order does not matter). Without a J,J coefficient (a hybrid/overlay sub-style that
    // only assigns I,J, as the gauss_diel example does with 1-4) the I,J epsilon is used (measured: eps_12 = 20
    // gives -39.0043087936508 and eps_12 = 78 gives -58.11114925 for the same hybrid dimer). The rme and sigma_e of
    // the I,J coefficient are used.
    this.eps[k1] = this.eps[k2] = this.p.isSet(j, j) ? this.p.get('eps', j, j) : this.p.get('eps', i, j);
    this.rme[k1] = this.rme[k2] = this.p.get('rme', i, j);
    this.se[k1] = this.se[k2] = this.p.get('se', i, j);
    return this.cutGlobal;
  }

  /** Energy and fpair (= -dE/dr / r) of qq = C qi qj at distance r; shift subtracts the value at the cutoff. */
  private kernel(t: number, r: number, qq: number): { e: number; fpair: number } {
    const eps = this.eps[t], rme = this.rme[t], se = this.se[t];
    const e0 = qq * coulDielG(eps, rme, se, r);
    const eShift = this.shiftOn[t] ? qq * coulDielG(eps, rme, se, this.cutGlobal) : 0;
    return { e: e0 - eShift, fpair: (-qq * coulDielDg(eps, rme, se, r)) / r };
  }

  compute(pc: PairCompute): void {
    this.qqrd2e = pc.qqrd2e;
    const list = pc.half!;
    const { x, f, type, q } = pc;
    const nt = this.ntypes + 1;
    const { cutsq } = this;
    const sC = pc.specialCoul;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let ecoul = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const qi = q[i];
      const ti = type[i] * nt;
      let fxi = 0, fyi = 0, fzi = 0;
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const j = jj & NEIGHMASK;
        const dx = xi - x[3 * j], dy = yi - x[3 * j + 1], dz = zi - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const t = ti + type[j];
        if (rsq >= cutsq[t]) continue;
        const factor = sC[jj >>> SBBITS];
        if (factor !== 0 && factor !== 1) throw new StyleError(`pair coul/diel: special_bonds coul factor ${factor} is not supported (only 0.0 and 1.0)`);
        if (factor === 0) continue;
        const r = Math.sqrt(rsq);
        const { e, fpair } = this.kernel(t, r, this.qqrd2e * qi * q[j]);
        const fx = dx * fpair, fy = dy * fpair, fz = dz * fpair;
        fxi += fx; fyi += fy; fzi += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        ecoul += e;
        if (tally) tallyAtom(pc, i, j, e, fpair, dx, dy, dz);
      }
      f[3 * i] += fxi; f[3 * i + 1] += fyi; f[3 * i + 2] += fzi;
    }
    pc.acc.ecoul += ecoul;
  }

  single(_i: number, _j: number, itype: number, jtype: number, rsq: number, factorCoul: number, _factorLJ: number, qi: number, qj: number): { eng: number; fforce: number } {
    if (factorCoul !== 0 && factorCoul !== 1) throw new StyleError(`pair coul/diel: special_bonds coul factor ${factorCoul} is not supported (only 0.0 and 1.0)`);
    if (factorCoul === 0) return { eng: 0, fforce: 0 };
    const t = itype * (this.ntypes + 1) + jtype;
    const { e, fpair } = this.kernel(t, Math.sqrt(rsq), this.qqrd2e * qi * qj);
    return { eng: e, fforce: fpair };
  }

  dataCoeffs(): string[] {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      out.push(`${i} ${fmtCoeff(this.p.get('eps', i, i))} ${fmtCoeff(this.p.get('rme', i, i))} ${fmtCoeff(this.p.get('se', i, i))}`);
    }
    return out;
  }
}
