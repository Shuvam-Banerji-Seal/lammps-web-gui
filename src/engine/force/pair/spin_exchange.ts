import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, pairCoeffArgs } from '../types';
import { NEIGHMASK, type Neighbor } from '../../neighbor';
import { numeric, parseNum, logical } from '../util';
import { PairHybrid } from './hybrid';
import { spinContribute, spinPairSettings, tallyPairEnergy, tallyPairVirial, type SpinPairStyle } from '../spin_book';
import { libm } from '../../libm';

/*
 * pair_style spin/exchange cutoff and pair_style spin/exchange/biquadratic —
 * docs.lammps.org/pair_spin_exchange.html.
 *
 *   pair_style spin/exchange cutoff
 *   pair_coeff * * exchange 4.0 0.0446928 0.003496 1.4885
 *   pair_coeff 1 2 exchange 6.0 -0.01575 0.0 1.965 offset yes
 *   pair_style spin/exchange/biquadratic cutoff
 *   pair_coeff * * biquadratic 4.0 0.05 0.03 1.48 0.05 0.03 1.48 offset no
 *
 * "Style *spin/exchange* computes the exchange interaction between pairs of
 * magnetic spins" with "H_{ex} = -\sum_{i,j}^N J_{ij} (r_{ij}) \,\vec{s}_i
 * \cdot \vec{s}_j". "The summation is over pairs of nearest neighbors." The
 * distance dependence is
 *   f(r) = 4 a (r/d)^2 (1 - b (r/d)^2) e^{-(r/d)^2} Theta(R_c - r),
 * "where :math:`a`, :math:`b` and :math:`d` are the three constant
 * coefficients defined in the associated \"pair_coeff\" command, and :math:`R_c`
 * is the radius cutoff".
 *
 * Biquadratic: "H_{bi} = -\sum_{i, j}^{N} {J}_{ij} \left(r_{ij} \right)\,". The page then adds a K term with
 * "\left(\vec{s}_{i}\cdot \vec{s}_{j}\right)^2" and K(r) given by the same function f(r) with its own (a, b, d)
 * triple. "When a *spin/exchange/biquadratic* pair style is defined, six coefficients" are needed.
 * Measured with native LAMMPS (black box, two atoms, offset no and yes, two geometries): energy
 * -J (s1.s2) - K (s1.s2)^2 (offset yes subtracts 1 from each bracket), precession
 * vector of atom i (J + 2 K s1.s2) s_j / hbar, mechanical force
 * (dJ/dr (s1.s2 - off) + dK/dr ((s1.s2)^2 - off)) e_ij with e_ij = (r_i - r_j)/r.
 *
 * The torque and force of the plain style: "\vec{\omega}_{i} = \frac{1}{\hbar} \sum_{j}^{Neighb} {J}
 * \left(r_{ij} \right)\,\vec{s}_{j}" and "\vec{F}_{i} = \sum_{j}^{Neighb}
 * \frac{\partial {J} \left(r_{ij} \right)}{ \partial r_{ij}} \left(
 * \vec{s}_{i}\cdot \vec{s}_{j} \right) \vec{e}_{ij}" with "\hbar the Planck
 * constant (in metal units)". Measured with native LAMMPS (black box): the
 * precession vector of atom i is (J/hbar) sum_j s_j (fitted to the fmx fmy fmz
 * of compute property/atom for two atoms; the magnitude equals J(r) / hbar for
 * unit spins, independent of the spin magnitude g), with hbar = SPIN_HBAR below.
 *
 * Offset: "offset yes" gives "H_{ex} = -\sum_{i,j}^N J_{ij} (r_{ij}) \,[
 * \vec{s}_i \cdot \vec{s}_j-1 ]" and "this offset only affects the calculation
 * of the energy and mechanical forces. It does not modify the calculation of
 * the precession vectors". Measured with native LAMMPS (black box): energy
 * -J (s1.s2) for offset no and -J (s1.s2 - 1) for offset yes, one count per
 * pair.
 *
 * Spins are read as unit vectors from atom_style spin (sp[0..2]); ghost spins
 * are the spins of their owners (nb.owner). The pair needs a full neighbor
 * list; each owned atom accumulates its own precession vector and force from
 * its full list, and the energy is halved per directed pair. The precession
 * vectors are contributed through spin_book.ts, so several spin pair styles
 * can overlay.
 */

/**
 * hbar in metal units (eV ps). As native LAMMPS does (src/SPIN/pair_spin.cpp and fix_precession_spin.cpp at b332f28:
 * hbar = force->hplanck/MY_2PI, with the metal hplanck 4.135667403e-3 eV ps of src/update.cpp): 6.582119101714715e-4.
 * The rounded 6.5821191e-4 used before was 2.6e-10 too small; min_style spin/lbfgs in a GNEB amplified that to 1e-4.
 */
export const SPIN_HBAR = 4.135667403e-3 / (2 * Math.PI);

/** J(r) of spin/exchange and its derivative with respect to r (pair_spin_exchange.html). */
export const exchangeJ = (a: number, b: number, d: number, r: number): { j: number; dj: number } => {
  const x = (r / d) * (r / d);
  const e = libm.exp(-x);
  // J = 4a x (1 - b x) e^{-x} with x = (r/d)^2; dJ/dr = 4a (2 r / d^2) (1 - 2 b x - x + b x^2) e^{-x}
  const j = 4 * a * x * (1 - b * x) * e;
  const dj = 4 * a * (2 * r / (d * d)) * (1 - 2 * b * x - x + b * x * x) * e;
  return { j, dj };
};

const PARAM_NAMES = ['rc', 'a', 'b', 'd', 'ak', 'bk', 'dk', 'offset'];

export class PairSpinExchange extends Pair implements SpinPairStyle {
  readonly name: string = 'spin/exchange';
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  readonly spinForce = true;
  /** Global cutoff of pair_style spin/exchange (or spin/exchange/biquadratic). */
  cutGlobal = 0;
  /** True for pair_style spin/exchange/biquadratic (adds the K(r) term). */
  readonly biquadratic: boolean = false;
  /** Per type pair: a, b, d (J), ak, bk, dk (K, biquadratic only), offset (1 = yes). Cutoff is the pair_coeff Rc. */
  p!: PairParams;
  private fmBuf = new Float64Array(0);

  constructor(biquadratic = false) {
    super();
    this.biquadratic = biquadratic;
    if (biquadratic) this.name = 'spin/exchange/biquadratic';
  }

  settings(args: string[], ctx?: StyleContext): void {
    const style = this.biquadratic ? 'spin/exchange/biquadratic' : 'spin/exchange';
    spinPairSettings(ctx);
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = numeric(args[0]);
    if (!(this.cutGlobal > 0)) throw new StyleError(`${style}: cutoff must be > 0`);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, PARAM_NAMES);
  }

  coeff(args: string[]): void {
    // docs.lammps.org/pair_spin_exchange.html: "R_c (distance units)", "a (energy units)", "b (adim parameter)",
    // "d (distance units)" for spin/exchange; biquadratic adds a second triple (a_k, b_k, d_k) for K(r_ij).
    // As native LAMMPS does (src/SPIN/pair_spin_exchange.cpp and pair_spin_exchange_biquadratic.cpp coeff at b332f28): the
    // third word must be the keyword exchange or biquadratic.
    const style = this.biquadratic ? 'biquadratic' : 'exchange';
    const k = 3;
    const nv = this.biquadratic ? 7 : 4;
    const usage = this.biquadratic
      ? 'usage: pair_coeff I J biquadratic R_c a_j b_j d_j a_k b_k d_k [offset yes|no]'
      : 'usage: pair_coeff I J exchange R_c a b d [offset yes|no]';
    if (args[2] !== style || args.length < k + nv) throw pairCoeffArgs(usage);
    // As native LAMMPS does (src/SPIN/pair_spin_exchange.cpp and pair_spin_exchange_biquadratic.cpp coeff at b332f28): the total argument count is checked before the keywords are read.
    if (this.biquadratic ? args.length !== 10 && args.length !== 12 : args.length !== 7 && args.length !== 9) throw new StyleError('Incorrect args for pair coefficients');
    const names = this.biquadratic ? ['Rc', 'a_j', 'b_j', 'd_j', 'a_k', 'b_k', 'd_k'] : ['Rc', 'a', 'b', 'd'];
    const v = args.slice(k, k + nv).map((w, n) => parseNum(w, names[n]));
    if (!(v[0] > 0)) throw new StyleError(`${this.name}: R_c must be > 0`);
    if (!(v[3] > 0)) throw new StyleError(`${this.name}: d must be > 0`);
    if (this.biquadratic && !(v[6] > 0)) throw new StyleError(`${this.name}: d_k must be > 0`);
    let offset = 0;
    for (let m = k + nv; m < args.length; m += 2) {
      if (args[m] !== 'offset') throw new StyleError('Incorrect args for pair coefficients');
      // pair_spin_exchange.cpp: utils::logical
      offset = logical(args[m + 1]) ? 1 : 0;
    }
    const ak = this.biquadratic ? v[4] : 0, bk = this.biquadratic ? v[5] : 0, dk = this.biquadratic ? v[6] : 1;
    this.p.setRange(args[0], args[1], [v[0], v[1], v[2], v[3], ak, bk, dk, offset]);
  }

  initStyle(_ctx: StyleContext): void {
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.p.get('rc', i, j);
  }

  /**
   * Precession vector of owned atom i (out[0..2] = (J/hbar) sum_j s_j over its full-list neighbors; biquadratic:
   * ((J + 2 K s_i.s_j)/hbar) s_j) from the current spins of the state.
   */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const nt = this.ntypes + 1;
    const owner = nb.owner;
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    const ti = type[i] * nt;
    const si0 = sp[4 * i], si1 = sp[4 * i + 1], si2 = sp[4 * i + 2];
    for (let k = k0; k < k1; k++) {
      const j = list.neighbors[k] & NEIGHMASK;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const t = ti + type[j];
      if (rsq >= this.cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      const o = owner[j];
      const { j: J } = this.pairJK(type[i], type[j], r);
      const kk = this.biquadratic ? this.pairK(type[i], type[j], r) : 0;
      const sj0 = sp[4 * o], sj1 = sp[4 * o + 1], sj2 = sp[4 * o + 2];
      const ss = si0 * sj0 + si1 * sj1 + si2 * sj2;
      const c = (J + 2 * kk * ss) / SPIN_HBAR;
      out[0] += c * sj0; out[1] += c * sj1; out[2] += c * sj2;
    }
  }

  private pairJK(ti: number, tj: number, r: number): { j: number; dj: number } {
    return exchangeJ(this.p.get('a', ti, tj), this.p.get('b', ti, tj), this.p.get('d', ti, tj), r);
  }

  private pairK(ti: number, tj: number, r: number): number {
    return exchangeJ(this.p.get('ak', ti, tj), this.p.get('bk', ti, tj), this.p.get('dk', ti, tj), r).j;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError(`pair style ${this.name} requires atom_style spin`);
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const sp = s.sp;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    let evdwl = 0;
    if (this.fmBuf.length < 3 * pc.nlocal) this.fmBuf = new Float64Array(3 * pc.nlocal);
    const fmv = this.fmBuf;
    fmv.fill(0, 0, 3 * pc.nlocal);
    // Measured with native LAMMPS (black box): with pair spin/exchange alone (no fix nve/spin) the atom forces stay 0.
    const lattice = s.spinLattice === true;
    for (let i = 0; i < pc.nlocal; i++) {
      const ti = type[i] * nt;
      const si0 = sp[4 * i], si1 = sp[4 * i + 1], si2 = sp[4 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const tj = type[j];
        const t = ti + tj;
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        if (rsq >= cutsq[t]) continue;
        const r = Math.sqrt(rsq);
        const { j: J, dj } = this.pairJK(type[i], tj, r);
        const kk = this.biquadratic ? exchangeJ(this.p.get('ak', type[i], tj), this.p.get('bk', type[i], tj), this.p.get('dk', type[i], tj), r) : null;
        const o = owner[j];
        const sj0 = sp[4 * o], sj1 = sp[4 * o + 1], sj2 = sp[4 * o + 2];
        const sij = si0 * sj0 + si1 * sj1 + si2 * sj2;
        const off = this.p.get('offset', type[i], tj) === 1 ? 1 : 0;
        // Energy -J (s_i.s_j - offset) [- K ((s_i.s_j)^2 - offset)] per unordered pair: half per directed pair.
        let pe = -J * (sij - off);
        let c = J;
        let fpair = dj * (sij - off);
        if (kk) {
          pe += -kk.j * (sij * sij - off);
          c += 2 * kk.j * sij;
          fpair += kk.dj * (sij * sij - off);
        }
        evdwl += 0.5 * pe;
        if (pc.eatom) tallyPairEnergy(pc.eatom, i, j, 0.5 * pe);
        // precession vector of i: (J / hbar) s_j (biquadratic: plus 2 K (s_i.s_j) s_j / hbar)
        const cw = c / SPIN_HBAR;
        fmv[3 * i] += cw * sj0; fmv[3 * i + 1] += cw * sj1; fmv[3 * i + 2] += cw * sj2;
        // mechanical force of the directed pair: half of dH-term/dr e_ij, e_ij = (r_i - r_j)/r, on i and its opposite on
        // j (the (j, i) call adds the other half). As native LAMMPS does (src/SPIN/pair_spin_exchange.cpp compute at
        // b332f28): f[i] += fi and f[j] -= fi, so the f dot r virial over owned and ghost atoms is the pair virial.
        if (lattice) {
          const fp = 0.5 * fpair / r;
          const fx = fp * dx, fy = fp * dy, fz = fp * dz;
          f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
          f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
          if (pc.vatom) tallyPairVirial(pc.vatom, i, j, dx, dy, dz, fx, fy, fz);
        }
      }
    }
    pc.acc.evdwl += evdwl;
    // the magnetic energy alone (compute spin reports it; the pair energy may include other styles)
    spinContribute(s, this, fmv, evdwl);
  }
}

/** The spin pair styles of the force field: the style itself or the sub-styles of pair hybrid (empty if none). */
export const findSpinPairs = (p: Pair | null): SpinPairStyle[] => {
  if (!p) return [];
  if (isSpinStyle(p)) return [p];
  if (p instanceof PairHybrid) {
    const out: SpinPairStyle[] = [];
    for (const sub of p.subs) if (isSpinStyle(sub.style)) out.push(sub.style);
    return out;
  }
  return [];
};

/** A pair style that contributes precession vectors (spin/exchange, spin/dmi, spin/magelec). */
export const isSpinStyle = (p: unknown): p is SpinPairStyle => typeof (p as SpinPairStyle).omegaOf === 'function' && 'spinForce' in (p as object);

/** The first spin/exchange pair style of the force field (null if none). */
export const findSpinPair = (p: Pair | null): PairSpinExchange | null => {
  for (const sp of findSpinPairs(p)) if (sp instanceof PairSpinExchange) return sp;
  return null;
};
