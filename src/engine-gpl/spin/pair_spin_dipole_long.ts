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
    const pre1 = 2 * g / Math.sqrt(Math.PI);
    const pre2 = 4 * g ** 3 / Math.sqrt(Math.PI);
    const pre3 = 8 * g ** 5 / Math.sqrt(Math.PI);
    const r2inv = 1 / rsq;
    const r = Math.sqrt(rsq);
    const rinv = 1 / r;
    const grij = g * r;
    const expm2 = Math.exp(-grij * grij);
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
   * Precession vector of owned atom i (compute_single_pair of the source) for fix nve/spin. Not matched to native yet:
   * a 432-spin random configuration with nve/spin (frozen lattice, one step) gives a magnetization change that differs
   * from native by about 10% (the energies and the precession vectors of compute match native to 1e-8, and so does
   * min_style spin), so the spin dynamics are refused by name rather than reproduced approximately.
   */
  omegaOf(_i: number, _nb: Neighbor, _sp: Float64Array, _type: Int32Array, _x: Float64Array, _list: List, _out: Float64Array): void {
    throw new StyleError("pair style spin/dipole/long with fix nve/spin is not supported by the browser engine yet (the spin dynamics do not match native LAMMPS; min_style spin with this pair does)");
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error(`pair style ${this.name} needs a full neighbor list`);
    const s = pc.s;
    if (!s.sp || !s.fm) throw new StyleError(`pair style ${this.name} requires atom_style spin`);
    if (s.units.style !== 'metal') throw new StyleError('Spin pair styles require metal units');
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
