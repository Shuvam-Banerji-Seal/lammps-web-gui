import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../types';
import { NEIGHMASK, type Neighbor } from '../../neighbor';
import { parseNum } from '../util';
import { PairHybrid } from './hybrid';

/*
 * pair_style spin/exchange cutoff and pair_style spin/exchange/biquadratic —
 * docs.lammps.org/pair_spin_exchange.html. This file implements plain
 * spin/exchange.
 *
 *   pair_style spin/exchange cutoff
 *   pair_coeff * * exchange 4.0 0.0446928 0.003496 1.4885
 *   pair_coeff 1 2 exchange 6.0 -0.01575 0.0 1.965 offset yes
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
 * The torque and force: "\vec{\omega}_{i} = \frac{1}{\hbar} \sum_{j}^{Neighb}
 * {J} \left(r_{ij} \right)\,\vec{s}_{j}" and "\vec{F}_{i} = \sum_{j}^{Neighb}
 * \frac{\partial {J} \left(r_{ij} \right)}{ \partial r_{ij}} \left(
 * \vec{s}_{i}\cdot \vec{s}_{j} \right) \vec{e}_{ij}" with "\hbar the Planck
 * constant (in metal units)". Measured with native LAMMPS (black box): the
 * precession vector of atom i is (J/hbar) sum_j s_j with hbar = 6.5821191e-4 eV ps
 * (fitted to the fmx fmy fmz of compute property/atom for two atoms; the
 * magnitude equals J(r) / hbar for unit spins, independent of the spin
 * magnitude g).
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
 * its full list, and the energy is halved per directed pair.
 */

/** Planck constant in metal units (eV ps), measured as described above. */
export const SPIN_HBAR = 6.5821191e-4;

/** J(r) of spin/exchange and its derivative with respect to r (pair_spin_exchange.html). */
export const exchangeJ = (a: number, b: number, d: number, r: number): { j: number; dj: number } => {
  const x = (r / d) * (r / d);
  const e = Math.exp(-x);
  // J = 4a x (1 - b x) e^{-x} with x = (r/d)^2; dJ/dr = 4a (2 r / d^2) (1 - 2 b x - x + b x^2) e^{-x}
  const j = 4 * a * x * (1 - b * x) * e;
  const dj = 4 * a * (2 * r / (d * d)) * (1 - 2 * b * x - x + b * x * x) * e;
  return { j, dj };
};

export class PairSpinExchange extends Pair {
  readonly name: string = 'spin/exchange';
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  /** Global cutoff of pair_style spin/exchange. */
  cutGlobal = 0;
  /** Per type pair: a, b, d, offset (1 = yes). Cutoff is the pair_coeff Rc. */
  p!: PairParams;

  settings(args: string[]): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style spin/exchange cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('spin/exchange: cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['rc', 'a', 'b', 'd', 'offset']);
  }

  coeff(args: string[]): void {
    // docs.lammps.org/pair_spin_exchange.html: the coefficients are R_c (distance units), a (energy units),
    // b (adim parameter) and d (distance units), in that order; the examples put the keyword exchange
    // first (a hybrid sub-style name is removed by pair hybrid before the call, so the keyword may be present).
    const k = args[2] === 'exchange' ? 3 : 2;
    if (args.length < k + 4) throw new StyleError('usage: pair_coeff I J [exchange] R_c a b d [offset yes|no]');
    const v = args.slice(k, k + 4).map((w, n) => parseNum(w, ['Rc', 'a', 'b', 'd'][n]));
    if (!(v[0] > 0)) throw new StyleError('spin/exchange: R_c must be > 0');
    if (!(v[3] > 0)) throw new StyleError('spin/exchange: d must be > 0');
    let offset = 0;
    for (let m = k + 4; m < args.length; m += 2) {
      if (args[m] !== 'offset') throw new StyleError(`pair_coeff spin/exchange: unknown keyword ${args[m]}`);
      if (args[m + 1] === 'yes') offset = 1;
      else if (args[m + 1] === 'no') offset = 0;
      else throw new StyleError(`pair_coeff spin/exchange: offset must be yes or no`);
    }
    this.p.setRange(args[0], args[1], [v[0], v[1], v[2], v[3], offset]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError('pair_modify shift/tail is not supported for pair style spin/exchange');
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.p.get('rc', i, j);
  }

  /**
   * Precession vector of owned atom i (out[0..2] = (J/hbar) sum_j s_j over its full-list neighbors) from the
   * current spins of the state; the same sum the compute uses. Also returns the energy and force sum when
   * `forces` is given (not used by the integrator).
   */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const nt = this.ntypes + 1;
    const owner = nb.owner;
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    const ti = type[i] * nt;
    for (let k = k0; k < k1; k++) {
      const j = list.neighbors[k] & NEIGHMASK;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const t = ti + type[j];
      if (rsq >= this.cutsq[t]) continue;
      const r = Math.sqrt(rsq);
      const o = owner[j];
      const { j: J } = exchangeJ(this.p.get('a', type[i], type[j]), this.p.get('b', type[i], type[j]), this.p.get('d', type[i], type[j]), r);
      const c = J / SPIN_HBAR;
      out[0] += c * sp[4 * o]; out[1] += c * sp[4 * o + 1]; out[2] += c * sp[4 * o + 2];
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style spin/exchange needs a full neighbor list');
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError('pair style spin/exchange requires atom_style spin');
    const { x, f, type } = pc;
    const nb = pc.nb;
    const owner = nb.owner;
    const sp = s.sp, fm = s.fm;
    const nt = this.ntypes + 1;
    const cutsq = this.cutsq;
    let evdwl = 0;
    // Measured with native LAMMPS (black box): with pair spin/exchange alone (no fix nve/spin) the atom forces stay 0.
    const lattice = s.spinLattice === true;
    for (let i = 0; i < pc.nlocal; i++) {
      fm[3 * i] = 0; fm[3 * i + 1] = 0; fm[3 * i + 2] = 0;
    }
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
        const ti2 = type[i], a = this.p.get('a', ti2, tj), b = this.p.get('b', ti2, tj), d = this.p.get('d', ti2, tj);
        const { j: J, dj } = exchangeJ(a, b, d, r);
        const o = owner[j];
        const sj0 = sp[4 * o], sj1 = sp[4 * o + 1], sj2 = sp[4 * o + 2];
        const sij = si0 * sj0 + si1 * sj1 + si2 * sj2;
        const off = this.p.get('offset', ti2, tj) === 1 ? 1 : 0;
        // Energy -J (s_i.s_j - offset) per unordered pair: each pair is in both full lists, so half per directed pair.
        evdwl += 0.5 * (-J * (sij - off));
        // precession vector of i: (J / hbar) s_j
        const c = J / SPIN_HBAR;
        fm[3 * i] += c * sj0; fm[3 * i + 1] += c * sj1; fm[3 * i + 2] += c * sj2;
        // mechanical force on i: F_i = dJ/dr (s_i.s_j - offset) e_ij, e_ij = (r_i - r_j)/r
        if (lattice) {
          const fpair = dj * (sij - off) / r;
          f[3 * i] += fpair * dx; f[3 * i + 1] += fpair * dy; f[3 * i + 2] += fpair * dz;
        }
      }
    }
    pc.acc.evdwl += evdwl;
    // the magnetic energy alone (compute spin reports it; the pair energy may include other styles)
    s.spinEnergy = evdwl;
  }
}

/** The spin/exchange pair style of the force field: the style itself or a hybrid sub-style (null if none). */
export const findSpinPair = (p: Pair | null): PairSpinExchange | null => {
  if (!p) return null;
  if (p instanceof PairSpinExchange) return p;
  if (p instanceof PairHybrid) {
    for (const sub of p.subs) if (sub.style instanceof PairSpinExchange) return sub.style;
  }
  return null;
};
