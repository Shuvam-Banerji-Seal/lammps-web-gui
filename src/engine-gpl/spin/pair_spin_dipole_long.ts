/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/pair_spin_dipole_long.cpp and
 * pair_spin_dipole_long.h (PairSpinDipoleLong: pair_style spin/dipole/long, the real-space part of the dipolar
 * interaction between spins, screened by the Ewald splitting of kspace_style ewald/dipole/spin or pppm/dipole/spin),
 * with the erfc approximation of the source (ewald_const.h: EWALD_P and A1..A5). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing authors (from pair_spin_dipole_long.cpp): Julien Tranchida (SNL) and
 * Stan Moore (SNL).
 *
 * The per-directed-pair terms follow the source: with g = g_ewald r, B_0 = erfc(g)/r and B_n from the recursion
 * B_1 = (B_0 + pre1 e^{-g^2}) / r^2 and so on, the precession contribution of spin j on i is
 * (mub^2 mu0 g_i g_j /(hbar 4 pi)) (B_2 r^2 (s_j.e_ij) e_ij - B_1 s_j); the magnetic energy is -(hbar/2) s_i . f_i,j
 * per directed pair. The mechanical force (compute_long_mech) is applied only with fix nve/spin lattice moving. The
 * The kspace solver supplies g_ewald (Pair.gEwald, set by the force field after the kspace init).
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type Neighbor } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { spinContribute, type SpinPairStyle } from '../../engine/force/spin_book';
import { HBAR, MUB2MU0, MUB2MU0HBINV } from './pair_spin_dipole_cut';
import { fmLongOf } from './dipole_long_field';
import { nativeFullLast } from '../neighbor/nativeOrder';
import type { SimState } from '../../engine/types';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
const MY_PIS = 1.77245385090551602729; // math_const.h (sqrt(pi) rounded once; Math.sqrt(Math.PI) is one ulp below)

/** ewald_const.h: EWALD_P and the coefficients of the erfc approximation (LAMMPS src/ewald_const.h, b332f28). */
const EWALD_P = 0.3275911;
const A1 = 0.254829592;
const A2 = -0.284496736;
const A3 = 1.421413741;
const A4 = -1.453152027;
const A5 = 1.061405429;

type Vec = [number, number, number];
type List = { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array };

export class PairSpinDipoleLong extends Pair implements SpinPairStyle {
  readonly name: string = 'spin/dipole/long';
  readonly spinForce = true;
  /** The long-range part needs a kspace style (the force field refuses the pair without one). */
  coulLong = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  /** Global cutoff of pair_style spin/dipole/long (the neighbor-list cutoff and cut_coul). */
  cutGlobal = 0;
  /** Per type pair: the cutoff of the real-space term (pair_coeff's cutoff). */
  p!: PairParams;
  private fmBuf = new Float64Array(0);
  private bbuf = new Float64Array(4);
  /** The state of the last compute: the kspace field fm_long is kept per atom id in it (dipole_long_field.ts). */
  private state: SimState | null = null;
  /** Per owned atom: the engine index of the partner that native's full list stores last (nativeFullLast), or -1. */
  private lastPartner: Int32Array = new Int32Array(0);
  private lastBuild = -1;
  private lastOrder: Int32Array | null = null;
  private lastList: object | null = null;

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style spin/dipole/long cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('spin/dipole/long: cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['rc']);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('usage: pair_coeff I J cutoff (pair spin/dipole/long)');
    const rc = parseNum(args[2], 'cutoff');
    if (!(rc > 0)) throw new StyleError('spin/dipole/long: cutoff must be > 0');
    this.p.setRange(args[0], args[1], [rc]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair_modify shift/tail is not supported for pair style ${this.name}`);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
    return this.cutGlobal;
  }

  extract(name: string): unknown {
    return name === 'cut_coul' || name === 'cut' ? this.cutGlobal : undefined;
  }

  /** Real-space screened terms B_0..B_3 of the source for a pair at distance r (rsq > 0). */
  private bterms(rsq: number, out: Float64Array): void {
    const g = this.gEwald;
    const pre1 = 2.0 * g / MY_PIS;
    const pre2 = 4.0 * libmPow(g, 3.0) / MY_PIS;
    const pre3 = 8.0 * libmPow(g, 5.0) / MY_PIS;
    const r2inv = 1 / rsq;
    const r = Math.sqrt(rsq);
    const rinv = 1 / r;
    const grij = g * r;
    const expm2 = libmExp(-grij * grij);
    const t = 1 / (1 + EWALD_P * grij);
    const erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
    out[0] = erfc * rinv;
    out[1] = (out[0] + pre1 * expm2) * r2inv;
    out[2] = (3 * out[1] + pre2 * expm2) * r2inv;
    out[3] = (5 * out[2] + pre3 * expm2) * r2inv;
  }

  /** compute_long of the source: the screened dipolar term of spin j on i, added to fmi (sp arrays at offsets si, sj). */
  private longTerm(eij: Vec, b: Float64Array, fmi: Vec, spi: Float64Array, si: number, spj: Float64Array, sj: number, rsq: number): void {
    const gigj = spi[si + 3] * spj[sj + 3];
    const pre = gigj * MUB2MU0HBINV;
    const sjeij = spj[sj] * eij[0] + spj[sj + 1] * eij[1] + spj[sj + 2] * eij[2];
    const b1 = b[1], b2 = b[2];
    for (let k = 0; k < 3; k++) fmi[k] += pre * (b2 * rsq * sjeij * eij[k] - b1 * spj[sj + k]);
  }

  /** compute_long_mech of the source: the mechanical force on i from the pair (subtracted from fi). */
  private longMech(eij: Vec, b: Float64Array, fi: Vec, spi: Float64Array, si: number, spj: Float64Array, sj: number, rsq: number): void {
    const gigj = spi[si + 3] * spj[sj + 3];
    const pre = 0.5 * gigj * MUB2MU0;
    const sisj = spi[si] * spj[sj] + spi[si + 1] * spj[sj + 1] + spi[si + 2] * spj[sj + 2];
    const sieij = spi[si] * eij[0] + spi[si + 1] * eij[1] + spi[si + 2] * eij[2];
    const sjeij = spj[sj] * eij[0] + spj[sj + 1] * eij[1] + spj[sj + 2] * eij[2];
    const b2 = b[2], b3 = b[3];
    const r = Math.sqrt(rsq);
    const g1 = sisj;
    const g2 = -rsq * sieij * sjeij;
    const g1b2g2b3 = g1 * b2 + g2 * b3;
    for (let k = 0; k < 3; k++) fi[k] -= pre * r * (eij[k] * g1b2g2b3 + b2 * (sjeij * spi[si + k] + sieij * spj[sj + k]));
  }

  /**
   * Precession vector of owned atom i for fix nve/spin (compute_single_pair of the source) and fm_long of atom i (the
   * kspace field of the last force evaluation, native: fmi += fm_long[ii]). The source resets its accumulator inside
   * the neighbour loop ("fmi[0] = fmi[1] = fmi[2] = 0.0;" before each j in compute_single_pair, src/SPIN/
   * pair_spin_dipole_long.cpp at b332f28), so the sweep keeps only the term of the last neighbour of the list, and that
   * term only when the last neighbour lies inside the cutoff. This is reproduced as measured against native (the
   * min_style path, compute(), accumulates over all neighbours and is unaffected).
   */
  /**
   * compute_single_pair of the source, for fix nve/spin: fmi is reset inside the neighbour loop, so the term that survives
   * is the one of the last neighbour of the list (when it lies inside the cutoff), and fm_long of atom i is added to it
   * (native: fmi[0] += fm_long[ii][0]). The reset overwrites whatever the spin pair styles before this one added to fmi,
   * which is why the order of the hybrid sub-styles changes the native dynamics (measured with native LAMMPS, frozen
   * lattice, bcc iron 432 spins: spin/exchange before spin/dipole/long gives no spin motion, after it the exchange motion).
   */
  singlePair(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, _list: List, fmi: number[]): void {
    let lx = 0, ly = 0, lz = 0;
    const j = i < this.lastPartner.length ? this.lastPartner[i] : -1;
    if (j >= 0) {
      const si = 4 * i;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const rc = this.p.get('rc', type[i], type[j]);
      if (rsq < rc * rc) {
        const rinv = 1 / Math.sqrt(rsq);
        const eij: Vec = [dx * rinv, dy * rinv, dz * rinv];
        const b = this.bbuf;
        this.bterms(rsq, b);
        const term: Vec = [0, 0, 0];
        this.longTerm(eij, b, term, sp, si, sp, 4 * nb.owner[j], rsq);
        lx = term[0]; ly = term[1]; lz = term[2];
      }
    }
    fmi[0] = lx; fmi[1] = ly; fmi[2] = lz;
    const s = this.state;
    if (s && s.fm) {
      const fl = fmLongOf(s, false);
      const id = 3 * s.id[i];
      fmi[0] += fl[id]; fmi[1] += fl[id + 1]; fmi[2] += fl[id + 2];
    }
  }

  /** The sweep vector of atom i alone (singlePair from zero). */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: List, out: Float64Array): void {
    const fmi = [0, 0, 0];
    this.singlePair(i, nb, sp, type, x, list, fmi);
    out[0] = fmi[0]; out[1] = fmi[1]; out[2] = fmi[2];
  }

  /**
   * The precession vector stored by the force evaluation (native compute(): fm[i] += fmi summed over all neighbours of i
   * inside the cutoff, from the current spins; the kspace field fm_long is not in it). Used for the output fm after a step.
   */
  outputOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: List, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const owner = nb.owner;
    const b = this.bbuf;
    const si = 4 * i;
    const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
    for (let k = k0; k < k1; k++) {
      const j = list.neighbors[k] & NEIGHMASK;
      const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
      const rsq = dx * dx + dy * dy + dz * dz;
      const rc = this.p.get('rc', type[i], type[j]);
      if (!(rsq < rc * rc)) continue;
      const rinv = 1 / Math.sqrt(rsq);
      const eij: Vec = [dx * rinv, dy * rinv, dz * rinv];
      this.bterms(rsq, b);
      const fmi: Vec = [0, 0, 0];
      this.longTerm(eij, b, fmi, sp, si, sp, 4 * owner[j], rsq);
      out[0] += fmi[0]; out[1] += fmi[1]; out[2] += fmi[2];
    }
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError(`pair style ${this.name} requires atom_style spin`);
    if (s.units.style !== 'metal') throw new StyleError('Spin pair styles require metal units');
    this.state = s;
    if (this.lastBuild !== pc.nb.nbuild || this.lastOrder !== s.order || this.lastList !== list) {
      this.lastPartner = nativeFullLast(s, pc.nb, pc.geom, list);
      this.lastBuild = pc.nb.nbuild;
      this.lastOrder = s.order;
      this.lastList = list;
    }
    if (!(this.gEwald > 0)) throw new StyleError(`pair style ${this.name} requires a kspace style (kspace_style ewald/dipole/spin or pppm/dipole/spin)`);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const sp = s.sp;
    const lattice = s.spinLattice === true;
    if (this.fmBuf.length < 3 * pc.nlocal) this.fmBuf = new Float64Array(3 * pc.nlocal);
    const fmv = this.fmBuf;
    fmv.fill(0, 0, 3 * pc.nlocal);
    const b = this.bbuf;
    let evdwl = 0;
    for (let i = 0; i < pc.nlocal; i++) {
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      const si = 4 * i;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const rc = this.p.get('rc', type[i], type[j]);
        if (!(rsq < rc * rc)) continue;
        const rinv = 1 / Math.sqrt(rsq);
        const eij: Vec = [dx * rinv, dy * rinv, dz * rinv];
        this.bterms(rsq, b);
        const sj = 4 * owner[j];
        const fmi: Vec = [0, 0, 0];
        this.longTerm(eij, b, fmi, sp, si, sp, sj, rsq);
        const fi: Vec = [0, 0, 0];
        if (lattice) this.longMech(eij, b, fi, sp, si, sp, sj, rsq);
        // energy of the directed pair: evdwl = -(s_i . fmi) (hbar / 2), tallied in full (newton pair on)
        evdwl += -(sp[si] * fmi[0] + sp[si + 1] * fmi[1] + sp[si + 2] * fmi[2]) * 0.5 * HBAR;
        f[3 * i] += fi[0]; f[3 * i + 1] += fi[1]; f[3 * i + 2] += fi[2];
        f[3 * j] -= fi[0]; f[3 * j + 1] -= fi[1]; f[3 * j + 2] -= fi[2];
        fmv[3 * i] += fmi[0]; fmv[3 * i + 1] += fmi[1]; fmv[3 * i + 2] += fmi[2];
      }
    }
    pc.acc.evdwl += evdwl;
    spinContribute(s, this, fmv, evdwl);
  }
}
