/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/pair_spin_neel.cpp and
 * pair_spin_neel.h (PairSpinNeel: pair_style spin/neel), with the shared spin pair base of src/SPIN/pair_spin.cpp.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). The source's contributing authors are Julien
 * Tranchida (SNL) and Stan Moore (SNL).
 *
 * pair_style spin/neel cutoff; pair_coeff I J neel rc k1 k2 k3 l1 l2 l3: the Neel (bi-quadratic and dipolar-like)
 * spin-spin interaction with the distance functions g(r) and q(r) of the source, each a Bethe-Slater form
 *   g(r) = 4 g1 x (1 - g2 x) e^{-x},  x = (r/g3)^2,   q(r) = 4 q1 y (1 - q2 y) e^{-y},  y = (r/q3)^2,
 * combined as the dipolar coefficient g1(r) = g(r) + 12 q(r)/35 and the quadrupolar ones q1(r) = 9 q(r)/5,
 * q2(r) = -2 q(r)/5 (src/SPIN/pair_spin_neel.cpp, compute_neel). The source stores g1 = k1/hbar and q1 = l1/hbar
 * for the precession and the unscaled k1, l1 for the mechanical force and the energy. Energy of a directed pair
 * -(1/2)(epd + epq1 + epq2), tallied in full (newton pair on); the precession vector and the mechanical force follow
 * the source (fix nve/spin lattice moving). The source's full list visits each pair twice.
 *
 * Spins are the unit vectors of atom_style spin; the magnitudes are not used by this style.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type Neighbor } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { spinContribute, type SpinPairStyle } from '../../engine/force/spin_book';
import { HBAR } from './pair_spin_dipole_cut';
import { exp as libmExp } from '../libm/exp';

type Vec = [number, number, number];

/** Distance functions of the pair (g1 ... q3 as in the source; the scaled g1 / q1 are per hbar). */
interface NeelCoef { rc: number; g1: number; g1m: number; g2: number; g3: number; q1: number; q1m: number; q2: number; q3: number }

/** compute_neel: precession term of spin j on i, added to fmi. */
const neelPrecession = (c: NeelCoef, rsq: number, e: Vec, fmi: Vec, si: Float64Array, i: number, sj: Float64Array, j: number): void => {
  let ra = rsq / c.g3 / c.g3;
  let gr = 4.0 * c.g1 * ra;
  gr *= (1.0 - c.g2 * ra);
  gr *= libmExp(-ra);
  ra = rsq / c.q3 / c.q3;
  let qr = 4.0 * c.q1 * ra;
  qr *= (1.0 - c.q2 * ra);
  qr *= libmExp(-ra);
  const g1r = gr + 12.0 * qr / 35.0;
  const q1r = 9.0 * qr / 5.0;
  const q2r = -2.0 * qr / 5.0;
  const sp = [si[i], si[i + 1], si[i + 2]], sq = [sj[j], sj[j + 1], sj[j + 2]];
  const eiSi = e[0] * sp[0] + e[1] * sp[1] + e[2] * sp[2];
  const eiSj = e[0] * sq[0] + e[1] * sq[1] + e[2] * sq[2];
  const siSj = sp[0] * sq[0] + sp[1] * sq[1] + sp[2] * sq[2];
  const pd = [0, 0, 0], pq1 = [0, 0, 0], pq2 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    pd[k] = g1r * (eiSj * e[k] - sq[k] / 3.0);
  }
  const eiSi2 = eiSi * eiSi;
  for (let k = 0; k < 3; k++) pq1[k] = -(eiSi2 - siSj / 3.0) * sq[k] / 3.0;
  const coeff1 = eiSj * eiSj - siSj / 3.0;
  for (let k = 0; k < 3; k++) pq1[k] += coeff1 * (2.0 * eiSi * e[k] - sq[k] / 3.0);
  for (let k = 0; k < 3; k++) pq1[k] *= q1r;
  const eiSj3 = eiSj * eiSj * eiSj;
  for (let k = 0; k < 3; k++) pq2[k] = (3.0 * eiSi2 * eiSj * e[k] + eiSj3 * e[k]) * q2r;
  for (let k = 0; k < 3; k++) fmi[k] += pd[k] + pq1[k] + pq2[k];
};

/** compute_neel_mech: mechanical force on i from the pair (i, j), before the 1/2 of the source (returned as fi). */
const neelForce = (c: NeelCoef, rsq: number, e: Vec, fi: Vec, si: Float64Array, i: number, sj: Float64Array, j: number): void => {
  const spi = [si[i], si[i + 1], si[i + 2]], spj = [sj[j], sj[j + 1], sj[j + 2]];
  const drij = Math.sqrt(rsq);
  const sisj = spi[0] * spj[0] + spi[1] * spj[1] + spi[2] * spj[2];
  const eiSi = e[0] * spi[0] + e[1] * spi[1] + e[2] * spi[2];
  const eiSj = e[0] * spj[0] + e[1] * spj[1] + e[2] * spj[2];
  // dipolar part
  const ig3 = 1.0 / (c.g3 * c.g3);
  let ra = rsq * ig3;
  let rr = drij * ig3;
  let gij = 4.0 * c.g1m * ra;
  gij *= (1.0 - c.g2 * ra);
  gij *= libmExp(-ra);
  let dgij = 1.0 - ra - c.g2 * ra * (2.0 - ra);
  dgij *= 8.0 * c.g1m * rr * libmExp(-ra);
  const pdt1 = (dgij - 2.0 * gij / drij) * eiSi * eiSj - sisj * dgij / 3.0;
  const pdt2 = eiSj * gij / drij;
  const pdt3 = eiSi * gij / drij;
  const pd = [0, 0, 0];
  for (let k = 0; k < 3; k++) pd[k] = -(pdt1 * e[k] + pdt2 * spi[k] + pdt3 * spj[k]);
  // quadrupolar part
  const iq3 = 1.0 / (c.q3 * c.q3);
  ra = rsq * iq3;
  rr = drij * iq3;
  const q1ij = 4.0 * c.q1m * ra * (1.0 - c.q2 * ra) * libmExp(-ra);
  const q2ij = -2.0 * q1ij / 9.0;
  let dq1ij = 1.0 - ra - c.q2 * ra * (2.0 - ra);
  dq1ij *= 8.0 * c.q1m * rr * libmExp(-ra);
  const dq2ij = -2.0 * dq1ij / 9.0;
  const eiSi2 = eiSi * eiSi, eiSj2 = eiSj * eiSj;
  const pqt1 = eiSi2 - sisj / 3.0;
  const pqt2 = eiSj2 - sisj / 3.0;
  const pq1 = [0, 0, 0], pq2 = [0, 0, 0];
  for (let k = 0; k < 3; k++) pq1[k] = dq1ij * pqt1 * pqt2 * e[k];
  const eiSi3 = eiSi * eiSi2, eiSj3 = eiSj * eiSj2;
  const sisj2 = sisj * sisj;
  const pqt3 = 2.0 * eiSi * (eiSj2 - sisj / 3.0) / drij;
  const pqt4 = 2.0 * eiSj * (eiSi2 - sisj / 3.0) / drij;
  for (let k = 0; k < 3; k++) pq1[k] += q1ij * (pqt3 * spi[k] + pqt4 * spj[k]);
  const pqt7 = 4.0 * eiSi2 * eiSj2 / drij;
  const pqt8 = 2.0 * sisj2 * eiSj / (3.0 * drij);
  const pqt9 = 2.0 * sisj2 * eiSi / (3.0 * drij);
  for (let k = 0; k < 3; k++) pq1[k] -= q1ij * (pqt7 + pqt8 + pqt9) * e[k];
  const pqt10 = eiSj * eiSi3;
  const pqt11 = eiSi * eiSj3;
  for (let k = 0; k < 3; k++) pq2[k] = dq2ij * (pqt10 + pqt11) * e[k];
  const pqt12 = eiSi3 / drij;
  const pqt13 = eiSj3 / drij;
  const pqt14 = 3.0 * eiSj * eiSi2 / drij;
  const pqt15 = 3.0 * eiSi * eiSj2 / drij;
  for (let k = 0; k < 3; k++) pq2[k] += q2ij * ((pqt12 + pqt15) * spj[k] + (pqt13 + pqt14) * spi[k]);
  const pqt16 = 4.0 * eiSj * eiSi3 / drij;
  const pqt17 = 4.0 * eiSi * eiSj3 / drij;
  for (let k = 0; k < 3; k++) pq2[k] -= q2ij * (pqt16 + pqt17) * e[k];
  for (let k = 0; k < 3; k++) fi[k] = 0.5 * (pd[k] + pq1[k] + pq2[k]);
};

/** compute_neel_energy: energy of the directed pair with the unscaled coefficients (returned with the 1/2 of the source). */
const neelEnergy = (c: NeelCoef, rsq: number, e: Vec, si: Float64Array, i: number, sj: Float64Array, j: number): number => {
  let ra = rsq / c.g3 / c.g3;
  let gr = 4.0 * c.g1m * ra;
  gr *= (1.0 - c.g2 * ra);
  gr *= libmExp(-ra);
  ra = rsq / c.q3 / c.q3;
  let qr = 4.0 * c.q1m * ra;
  qr *= (1.0 - c.q2 * ra);
  qr *= libmExp(-ra);
  const g1r = gr + 12.0 * qr / 35.0;
  const q1r = 9.0 * qr / 5.0;
  const q2r = -2.0 * qr / 5.0;
  const spi = [si[i], si[i + 1], si[i + 2]], spj = [sj[j], sj[j + 1], sj[j + 2]];
  const eiSi = e[0] * spi[0] + e[1] * spi[1] + e[2] * spi[2];
  const eiSj = e[0] * spj[0] + e[1] * spj[1] + e[2] * spj[2];
  const siSj = spi[0] * spj[0] + spi[1] * spj[1] + spi[2] * spj[2];
  const epd = g1r * (eiSi * eiSj - siSj / 3.0);
  const eiSi2 = eiSi * eiSi, eiSj2 = eiSj * eiSj;
  const epq1 = q1r * (eiSi2 - siSj / 3.0) * (eiSj2 - siSj / 3.0);
  const eiSi3 = eiSi * eiSi2, eiSj3 = eiSj * eiSj2;
  const epq2 = q2r * (eiSi * eiSj3 + eiSj * eiSi3);
  return 0.5 * (epd + epq1 + epq2);
};

export class PairSpinNeel extends Pair implements SpinPairStyle {
  readonly name: string = 'spin/neel';
  readonly spinForce = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  cutGlobal = 0;
  /** rc, g1 (k1/hbar), g1m (k1), g2, g3, q1 (l1/hbar), q1m (l1), q2, q3 per type pair. */
  p!: PairParams;
  private fmBuf = new Float64Array(0);

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style spin/neel cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('spin/neel: cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['rc', 'g1', 'g1m', 'g2', 'g3', 'q1', 'q1m', 'q2', 'q3']);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    // docs.lammps.org/pair_spin_neel.html: pair_coeff * * neel rc k1 k2 k3 l1 l2 l3 (ten arguments in all)
    if (args.length !== 10 || args[2] !== 'neel') throw new StyleError('usage: pair_coeff I J neel rc k1 k2 k3 l1 l2 l3 (pair spin/neel)');
    const v = args.slice(3).map((w, n) => parseNum(w, ['rc', 'k1', 'k2', 'k3', 'l1', 'l2', 'l3'][n]));
    const [rc, k1, k2, k3, l1, l2, l3] = v;
    if (!(rc > 0)) throw new StyleError('spin/neel: rc must be > 0');
    if (!(k3 > 0) || !(l3 > 0)) throw new StyleError('spin/neel: the distance parameters k3 and l3 must be > 0');
    this.p.setRange(args[0], args[1], [rc, k1 / HBAR, k1, k2, k3, l1 / HBAR, l1, l2, l3]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair_modify shift/tail is not supported for pair style ${this.name}`);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.cutGlobal;
  }

  private coef(ti: number, tj: number): NeelCoef {
    const p = this.p;
    return {
      rc: p.get('rc', ti, tj), g1: p.get('g1', ti, tj), g1m: p.get('g1m', ti, tj), g2: p.get('g2', ti, tj),
      g3: p.get('g3', ti, tj), q1: p.get('q1', ti, tj), q1m: p.get('q1m', ti, tj), q2: p.get('q2', ti, tj), q3: p.get('q3', ti, tj),
    };
  }

  /** Precession vector of owned atom i from the current spins (compute_single_pair of the source). */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const owner = nb.owner;
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = list.neighbors[k] & NEIGHMASK;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const c = this.coef(type[i], type[j]);
      if (rsq > c.rc * c.rc) continue;
      const inorm = 1 / Math.sqrt(rsq);
      const e: Vec = [dx * inorm, dy * inorm, dz * inorm];
      const fmi: Vec = [0, 0, 0];
      neelPrecession(c, rsq, e, fmi, sp, 4 * i, sp, 4 * owner[j]);
      out[0] += fmi[0]; out[1] += fmi[1]; out[2] += fmi[2];
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError(`pair style ${this.name} requires atom_style spin`);
    if (s.units.style !== 'metal') throw new StyleError('Spin pair styles require metal units');
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const sp = s.sp;
    const lattice = s.spinLattice === true;
    if (this.fmBuf.length < 3 * pc.nlocal) this.fmBuf = new Float64Array(3 * pc.nlocal);
    const fmv = this.fmBuf;
    fmv.fill(0, 0, 3 * pc.nlocal);
    let evdwl = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const c = this.coef(type[i], type[j]);
        if (rsq > c.rc * c.rc) continue;
        const inorm = 1 / Math.sqrt(rsq);
        const e: Vec = [dx * inorm, dy * inorm, dz * inorm];
        const sj = 4 * owner[j];
        const fmi: Vec = [0, 0, 0];
        neelPrecession(c, rsq, e, fmi, sp, 4 * i, sp, sj);
        const fi: Vec = [0, 0, 0];
        if (lattice) neelForce(c, rsq, e, fi, sp, 4 * i, sp, sj);
        f[3 * i] += fi[0]; f[3 * i + 1] += fi[1]; f[3 * i + 2] += fi[2];
        f[3 * j] -= fi[0]; f[3 * j + 1] -= fi[1]; f[3 * j + 2] -= fi[2];
        fmv[3 * i] += fmi[0]; fmv[3 * i + 1] += fmi[1]; fmv[3 * i + 2] += fmi[2];
        evdwl -= neelEnergy(c, rsq, e, sp, 4 * i, sp, sj);
      }
    }
    pc.acc.evdwl += evdwl;
    spinContribute(s, this, fmv, evdwl);
  }
}
