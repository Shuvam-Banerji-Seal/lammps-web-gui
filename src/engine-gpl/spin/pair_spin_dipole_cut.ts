/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/SPIN/pair_spin_dipole_cut.cpp and
 * pair_spin_dipole_cut.h (PairSpinDipoleCut: pair_style spin/dipole/cut), with the shared spin pair base of
 * src/SPIN/pair_spin.cpp (metal units, full neighbor list, newton pair on). LAMMPS is Copyright (2003) Sandia
 * Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE). Contributing authors (from pair_spin_dipole_cut.cpp): Julien Tranchida (SNL) and
 * Stan Moore (SNL).
 *
 * pair_style spin/dipole/cut cutoff; pair_coeff I J cut_long: the magnetic dipole-dipole interaction between spins
 * within cut_long. The per-pair dipolar field of the source is kept: with e_ij = (r_j - r_i)/r, the precession
 * contribution of spin j on i is (mub^2 mu0 /(4 pi hbar)) (g_i g_j / r^3) (3 (s_j.e_ij) e_ij - s_j), and the magnetic
 * energy of a directed pair is -(hbar/2) s_i . f_i,j. The mechanical force is applied only with fix nve/spin lattice
 * moving. Several spin pair styles overlay through the spin book (src/engine/force/spin_book.ts).
 *
 * The constants are those of the source: mub = 9.274e-4, mu_0 = 784.15 eV/Ang/A^2, hbar = hplanck / 2 pi with the
 * metal-unit Planck constant of the engine (units.ts). The source's full list visits each pair twice, and the
 * mechanical force is added to both atoms of every directed pair, as the source does.
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, type Neighbor } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { spinContribute, tallyPairEnergy, tallyPairVirial, type SpinPairStyle } from '../../engine/force/spin_book';

/** Bohr magneton (eV/T) and vacuum permeability in the source's units (eV/Ang/A^2). */
const MUB = 9.274e-4;
const MU0 = 784.15;
/** Metal-unit Planck constant (eV ps; units.ts) and the prefactors of the source. */
export const HPLANCK_METAL = 4.135667403e-3;
export const HBAR = HPLANCK_METAL / (2 * Math.PI);
export const MUB2MU0 = MUB * MUB * MU0 / (4 * Math.PI);
export const MUB2MU0HBINV = MUB2MU0 / HBAR;

type Vec = [number, number, number];

export class PairSpinDipoleCut extends Pair implements SpinPairStyle {
  readonly name: string = 'spin/dipole/cut';
  readonly spinForce = true;
  virialFdotr = true;
  needsFull = true;
  needsHalf = false;
  /** Global cutoff of pair_style spin/dipole/cut (the neighbor-list cutoff). */
  cutGlobal = 0;
  /** Per type pair: rc, the cutoff of the dipolar interaction (pair_coeff's cut_long). */
  p!: PairParams;
  private fmBuf = new Float64Array(0);

  settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 1) throw new StyleError('usage: pair_style spin/dipole/cut cutoff');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    if (!(this.cutGlobal > 0)) throw new StyleError('spin/dipole/cut: cutoff must be > 0');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['rc']);
  }

  coeff(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 3) throw new StyleError('usage: pair_coeff I J cut_long (pair spin/dipole/cut)');
    const rc = parseNum(args[2], 'cut_long');
    if (!(rc > 0)) throw new StyleError('spin/dipole/cut: cut_long must be > 0');
    this.p.setRange(args[0], args[1], [rc]);
  }

  initStyle(_ctx: StyleContext): void {
    if (this.shift || this.tail) throw new StyleError(`pair_modify shift/tail is not supported for pair style ${this.name}`);
  }

  initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) throw new StyleError(`all pair coeffs are not set (pair ${i} ${j})`);
    return this.cutGlobal;
  }

  /** compute_dipolar of the source: adds the dipolar precession term of spin j on i to fmi (already zeroed by the caller). */
  private dipolar(eij: Vec, fmi: Vec, spi: Float64Array, si: number, spj: Float64Array, sj: number, r3inv: number, mub2mu0hbinv: number): void {
    const sjdotr = spj[sj] * eij[0] + spj[sj + 1] * eij[1] + spj[sj + 2] * eij[2];
    const gigjiri3 = (spi[si + 3] * spj[sj + 3]) * r3inv;
    const pre = mub2mu0hbinv * gigjiri3;
    fmi[0] += pre * (3 * sjdotr * eij[0] - spj[sj]);
    fmi[1] += pre * (3 * sjdotr * eij[1] - spj[sj + 1]);
    fmi[2] += pre * (3 * sjdotr * eij[2] - spj[sj + 2]);
  }

  /** compute_dipolar_mech of the source: the mechanical force on i from the pair (i, j), subtracted from fi. */
  private dipolarMech(eij: Vec, fi: Vec, spi: Float64Array, si: number, spj: Float64Array, sj: number, r2inv: number, mub2mu0: number): void {
    const gigjri4 = (spi[si + 3] * spj[sj + 3]) * r2inv * r2inv;
    const sisj = spi[si] * spj[sj] + spi[si + 1] * spj[sj + 1] + spi[si + 2] * spj[sj + 2];
    const sieij = spi[si] * eij[0] + spi[si + 1] * eij[1] + spi[si + 2] * eij[2];
    const sjeij = spj[sj] * eij[0] + spj[sj + 1] * eij[1] + spj[sj + 2] * eij[2];
    const bij = sisj - 5 * sieij * sjeij;
    const pre = 0.5 * 3 * mub2mu0 * gigjri4;
    for (let k = 0; k < 3; k++) fi[k] -= pre * (eij[k] * bij + (sjeij * spi[si + k] + sieij * spj[sj + k]));
  }

  /**
   * Precession vector of owned atom i (compute_single_pair of the source): out = sum over its full-list neighbours
   * inside the pair cutoff of the dipolar term, from the current spins.
   */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void {
    out[0] = 0; out[1] = 0; out[2] = 0;
    const owner = nb.owner;
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
      const fmi: Vec = [0, 0, 0];
      this.dipolar(eij, fmi, sp, si, sp, 4 * owner[j], rinv * rinv * rinv, MUB2MU0HBINV);
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
      const si = 4 * i;
      for (let k = k0; k < k1; k++) {
        const j = list.neighbors[k] & NEIGHMASK;
        const dx = x[3 * j] - x[3 * i], dy = x[3 * j + 1] - x[3 * i + 1], dz = x[3 * j + 2] - x[3 * i + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const rc = this.p.get('rc', type[i], type[j]);
        if (!(rsq < rc * rc)) continue;
        const rinv = 1 / Math.sqrt(rsq);
        const eij: Vec = [dx * rinv, dy * rinv, dz * rinv];
        const r2inv = 1 / rsq;
        const sj = 4 * owner[j];
        const fmi: Vec = [0, 0, 0];
        this.dipolar(eij, fmi, sp, si, sp, sj, r2inv * rinv, MUB2MU0HBINV);
        const fi: Vec = [0, 0, 0];
        if (lattice) this.dipolarMech(eij, fi, sp, si, sp, sj, r2inv, MUB2MU0);
        // energy of the directed pair: evdwl = -(s_i . fmi) (hbar / 2), tallied in full (newton pair on)
        const ev = -(sp[si] * fmi[0] + sp[si + 1] * fmi[1] + sp[si + 2] * fmi[2]) * 0.5 * HBAR;
        evdwl += ev;
        if (pc.eatom) tallyPairEnergy(pc.eatom, i, j, ev);
        // the source's ev_tally_xyz gets -rij = x_i - x_j, the negative of dx here (src/SPIN/pair_spin_dipole_cut.cpp at b332f28)
        if (lattice && pc.vatom) tallyPairVirial(pc.vatom, i, j, -dx, -dy, -dz, fi[0], fi[1], fi[2]);
        f[3 * i] += fi[0]; f[3 * i + 1] += fi[1]; f[3 * i + 2] += fi[2];
        f[3 * j] -= fi[0]; f[3 * j + 1] -= fi[1]; f[3 * j + 2] -= fi[2];
        fmv[3 * i] += fmi[0]; fmv[3 * i + 1] += fmi[1]; fmv[3 * i + 2] += fmi[2];
      }
    }
    pc.acc.evdwl += evdwl;
    spinContribute(s, this, fmv, evdwl);
  }
}
