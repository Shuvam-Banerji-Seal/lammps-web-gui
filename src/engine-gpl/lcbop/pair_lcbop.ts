/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_lcbop.cpp and src/MANYBODY/pair_lcbop.h at commit b332f28 (PairLCBOP:
 * settings, coeff, init_style, init_one, compute, SR_neigh, FSR, FLR, FNij, FMij, bondorder, b); the per-atom virial
 * through the Pair::v_tally2 / v_tally3 ports of ../airebo/tally.ts.
 * Neighbor lists: built here from the owned and ghost positions with a binned search (the SR list of
 * every owned and ghost atom, and the full list of owned atoms within r_2_LR), in place of the LAMMPS
 * neighbor request.
 * Authors credited in the C++ file header: Dominik Wojt (Wroclaw University of Technology), based on
 * pair_airebo by Ase Henry (MIT). Copyright (2003) Sandia Corporation (GNU General Public License).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Known dependence on native ordering: PairLCBOP::FMij reads N[k] with k the neighbor-list slot of atomk (the C++
 * line `double Nki = N[k]-f_c_ik;`), not N[atomk]. The value therefore depends on the order of the SR neighbor list
 * and on the atom indices. The SR lists here are kept in ascending index order (native uses its bin order and its
 * sorted atom order, which this port does not reproduce), so the forces agree with native where those orders agree
 * (oracle w234lc_cl*, w234lc_diamond, w234lc_graphene) and differ for disordered periodic systems (not an oracle case).
 */
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { rowNeighbors } from '../airebo/neigh';
import { readLcbopFile, splineInit, type LcbopParams, type LcbopTables } from './params';
import { cmax, cmin, f_c, f_c_LR, F_conj, gSpline, hSpline, type FConjD, type Ref } from './funcs';
import { exp } from '../libm/exp';
import { pow } from '../libm/pow';
import { vTally2, vTally3 } from '../airebo/tally';

/** TOL in pair_lcbop.cpp */
const TOL = 1.0e-9;

const newRef = (): Ref => ({ v: 0 });

/** PairLCBOP (pair_style lcbop, carbon only). */
export class PairLCBOP extends Pair {
  readonly name = 'lcbop';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected p: LcbopParams | null = null;
  protected T: LcbopTables | null = null;
  protected cut3rebo = 0;
  protected r2sq = 0;
  protected cutLRsq = 0;

  // per-compute state (set by compute)
  protected x: Float64Array = new Float64Array(0);
  protected f: Float64Array = new Float64Array(0);
  protected N: Float64Array = new Float64Array(0);
  /** Per-atom virial of the current compute (6 per owned and ghost atom), or null when not requested. */
  protected vatom: Float64Array | null = null;
  protected M: Float64Array = new Float64Array(0);
  /** SR neighbors of every owned and ghost atom: row a is srList[srStart[a] .. srStart[a+1]) */
  protected srStart: Int32Array = new Int32Array(1);
  protected srList: Int32Array = new Int32Array(0);
  protected dummy = newRef();
  protected dS = newRef();

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    const elements = args.slice(3);
    if (elements.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    // only element "C" is allowed (NULL types take no element and are not mapped)
    const names = [...new Set(elements.filter((e) => e !== 'NULL'))];
    if (names.length !== 1 || names[0] !== 'C') {
      throw new StyleError('Incorrect args for pair coefficients\nFor more information see https://docs.lammps.org/err0021');
    }
    const fileName = args[2];
    const { params, gX, gC, F_data } = readLcbopFile(ctx.readFile(fileName, potentialFile('LCBOP')), fileName, ctx);
    this.p = params;
    this.T = { gX, gC, F_conj_field: splineInit(F_data) };
    // PairLCBOP::init_one: cut3rebo = 3 SR distances; cutLRsq; r_2_sq
    this.cut3rebo = 3.0 * params.r_2;
    this.cutLRsq = params.r_2_LR * params.r_2_LR;
    this.r2sq = params.r_2 * params.r_2;
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.p || !this.T) throw new StyleError('pair_style lcbop needs a pair_coeff command with a potential file');
    if (this.table !== 12) throw new StyleError('pair_style lcbop does not support the pair_modify table option');
  }

  /**
   * PairLCBOP::init_one: the cutoff the ghost shell must cover, cutmax = MAX(cut3rebo, r_2_LR)
   * (the SR term reaches 3 SR distances, the LR term r_2_LR).
   */
  override initOne(_i: number, _j: number): number {
    if (!this.p) throw new StyleError('All pair coeffs are not set');
    return cmax(this.cut3rebo, this.p.r_2_LR);
  }

  override compute(pc: PairCompute): void {
    if (!this.p || !this.T) throw new StyleError('pair_style lcbop has no potential file');
    const nall = pc.nall, nlocal = pc.nlocal;
    this.x = pc.x;
    this.f = pc.f;
    this.vatom = pc.vatom;
    const x = pc.x;
    // every atom is a carbon atom of the element map (NULL types are not mapped to another element)
    const mt = new Int32Array(nall);
    const p = this.p;
    const cutsqSR = new Float64Array([this.r2sq, this.r2sq, this.r2sq, this.r2sq]);
    const sr = rowNeighbors(x, nall, nall, mt, cutsqSR, p.r_2);
    this.srStart = sr.start;
    this.srList = sr.list;
    // The sums N_i and M_i run over the SR neighbors in index order: a fixed order keeps the rounding of N_i - f_C(ij)
    // (which picks the F_conj cell when it lands on an integer) the same for every neighbor list build.
    for (let a = 0; a < nall; a++) this.srList.subarray(this.srStart[a], this.srStart[a + 1]).sort();
    this.N = new Float64Array(nall);
    this.M = new Float64Array(nall);
    this.SR_neigh(nall);
    this.FSR(pc, nlocal);
    this.FLR(pc, nlocal, nall);
  }

  /** PairLCBOP::SR_neigh: N_i (sum of f_C over the SR neighbors) and M_i for owned and ghost atoms. */
  protected SR_neigh(nall: number): void {
    const x = this.x, p = this.p!, N = this.N, M = this.M;
    const srStart = this.srStart, srList = this.srList;
    for (let i = 0; i < nall; i++) {
      N[i] = 0.0;
      for (let kk = srStart[i]; kk < srStart[i + 1]; kk++) {
        const j = srList[kk];
        const delx = x[3 * i] - x[3 * j];
        const dely = x[3 * i + 1] - x[3 * j + 1];
        const delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < this.r2sq) N[i] += f_c(p, Math.sqrt(rsq), p.r_1, p.r_2, this.dS);
      }
    }
    for (let i = 0; i < nall; i++) {
      M[i] = 0.0;
      for (let kk = srStart[i]; kk < srStart[i + 1]; kk++) {
        const j = srList[kk];
        const delx = x[3 * i] - x[3 * j];
        const dely = x[3 * i + 1] - x[3 * j + 1];
        const delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < this.r2sq) {
          const f_c_ij = f_c(p, Math.sqrt(rsq), p.r_1, p.r_2, this.dS);
          const Nji = N[j] - f_c_ij;
          // F(xij) = 1-f_c_LR(Nji, 2,3,&dummy)
          M[i] += f_c_ij * (1 - f_c_LR(Nji, 2, 3, this.dS));
        }
      }
    }
  }

  /** PairLCBOP::FSR: the short-range two-body term VR - B_ij VA over the SR pairs, each physical pair once. */
  protected FSR(pc: PairCompute, nlocal: number): void {
    const p = this.p!;
    const x = this.x, f = this.f, tag = this.tags(pc);
    const acc = pc.acc;
    const eatom = pc.eatom;
    const srStart = this.srStart, srList = this.srList;
    for (let i = 0; i < nlocal; i++) {
      const itag = tag[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let kk = srStart[i]; kk < srStart[i + 1]; kk++) {
        const j = srList[kk];
        const jtag = tag[j];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < ztmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const delx = x[3 * i] - x[3 * j];
        const dely = x[3 * i + 1] - x[3 * j + 1];
        const delz = x[3 * i + 2] - x[3 * j + 2];
        const r_sq = delx * delx + dely * dely + delz * delz;
        const rijmag = Math.sqrt(r_sq);
        const df_c_ij = newRef();
        const f_c_ij = f_c(p, rijmag, p.r_1, p.r_2, df_c_ij);
        if (f_c_ij <= TOL) continue;
        let VR = p.A * exp(-p.alpha * rijmag);
        let dVRdi = -p.alpha * VR;
        dVRdi = dVRdi * f_c_ij + df_c_ij.v * VR; // VR -> VR * f_c_ij
        VR *= f_c_ij;
        let VA = 0.0, dVA = 0.0;
        {
          let term = p.B_1 * exp(-p.beta_1 * rijmag);
          VA += term;
          dVA += -p.beta_1 * term;
          term = p.B_2 * exp(-p.beta_2 * rijmag);
          VA += term;
          dVA += -p.beta_2 * term;
        }
        dVA = dVA * f_c_ij + df_c_ij.v * VA; // VA -> VA * f_c_ij
        VA *= f_c_ij;
        const del = [delx, dely, delz];
        const Bij = this.bondorder(i, j, del, rijmag, VA);
        const dVAdi = Bij * dVA;
        // F = (dVRdi+dVAdi)*(-grad rijmag)
        const fpair = -(dVRdi - dVAdi) / rijmag;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        const evdwl = VR - Bij * VA;
        this.tally(acc, eatom, i, j, evdwl);
        if (this.vatom) vTally2(this.vatom, i, j, fpair, delx, dely, delz);
      }
    }
  }

  /** PairLCBOP::FLR: the long-range Morse-like term over the full list of owned atoms within r_2_LR. */
  protected FLR(pc: PairCompute, nlocal: number, nall: number): void {
    const p = this.p!;
    const x = this.x, f = this.f, tag = this.tags(pc);
    const acc = pc.acc;
    const eatom = pc.eatom;
    const mt = new Int32Array(nall);
    const cutsqLR = new Float64Array([this.cutLRsq, this.cutLRsq, this.cutLRsq, this.cutLRsq]);
    const full = rowNeighbors(x, nlocal, nall, mt, cutsqLR, p.r_2_LR);
    const dfc = newRef();
    for (let i = 0; i < nlocal; i++) {
      const itag = tag[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let kk = full.start[i]; kk < full.start[i + 1]; kk++) {
        const j = full.list[kk];
        const jtag = tag[j];
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < ztmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const delx = x[3 * i] - x[3 * j];
        const dely = x[3 * i + 1] - x[3 * j + 1];
        const delz = x[3 * i + 2] - x[3 * j + 2];
        const r_sq = delx * delx + dely * dely + delz * delz;
        const rijmag = Math.sqrt(r_sq);
        // the derivative of 1-f_C is the negated f_C one; f_c_LR keeps it below r_1_LR (see funcs.ts)
        let f_c_ij = 1 - f_c(p, rijmag, p.r_1, p.r_2, dfc);
        dfc.v = -dfc.v;
        // derivative may be inherited from previous call, see f_c_LR definition
        f_c_ij *= f_c_LR(rijmag, p.r_1_LR, p.r_2_LR, dfc);
        if (f_c_ij <= TOL) continue;
        let V = 0, dVdi = 0;
        if (rijmag < p.r_0) {
          const exp_part = exp(-p.lambda_1 * (rijmag - p.r_0));
          V = p.eps_1 * (exp_part * exp_part - 2 * exp_part) + p.v_1;
          dVdi = 2 * p.eps_1 * p.lambda_1 * exp_part * (1 - exp_part);
        } else {
          const exp_part = exp(-p.lambda_2 * (rijmag - p.r_0));
          V = p.eps_2 * (exp_part * exp_part - 2 * exp_part) + p.v_2;
          dVdi = 2 * p.eps_2 * p.lambda_2 * exp_part * (1 - exp_part);
        }
        dVdi = dVdi * f_c_ij + dfc.v * V; // V -> V * f_c_ij
        V *= f_c_ij;
        // F = (dVdi)*(-grad rijmag)
        const fpair = -dVdi / rijmag;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        this.tally(acc, eatom, i, j, V);
        if (this.vatom) vTally2(this.vatom, i, j, fpair, delx, dely, delz);
      }
    }
  }

  /** PairLCBOP::FNij: forces of the N_ij (coordination) derivative over the SR neighbors k of i, k != j. */
  protected FNij(i: number, j: number, factor: number): void {
    const p = this.p!;
    const x = this.x, f = this.f;
    const atomi = i, atomj = j;
    for (let kk = this.srStart[i]; kk < this.srStart[i + 1]; kk++) {
      const atomk = this.srList[kk];
      if (atomk !== atomj) {
        const rik0 = x[3 * atomi] - x[3 * atomk];
        const rik1 = x[3 * atomi + 1] - x[3 * atomk + 1];
        const rik2 = x[3 * atomi + 2] - x[3 * atomk + 2];
        const riksq = rik0 * rik0 + rik1 * rik1 + rik2 * rik2;
        if (riksq > p.r_1 * p.r_1) {
          const rikmag = Math.sqrt(riksq);
          const df_c_ik = newRef();
          f_c(p, rikmag, p.r_1, p.r_2, df_c_ik);
          // F = factor*df_c_ik*(-grad rikmag)
          const fpair = -factor * df_c_ik.v / rikmag;
          f[3 * atomi] += rik0 * fpair;
          f[3 * atomi + 1] += rik1 * fpair;
          f[3 * atomi + 2] += rik2 * fpair;
          f[3 * atomk] -= rik0 * fpair;
          f[3 * atomk + 1] -= rik1 * fpair;
          f[3 * atomk + 2] -= rik2 * fpair;
          if (this.vatom) vTally2(this.vatom, atomi, atomk, fpair, rik0, rik1, rik2);
        }
      }
    }
  }

  /** PairLCBOP::FMij: forces of the M_ij (coordination correction) derivative. */
  protected FMij(i: number, j: number, factor: number): void {
    const p = this.p!;
    const x = this.x, f = this.f, N = this.N;
    const atomi = i, atomj = j;
    for (let kk = this.srStart[i]; kk < this.srStart[i + 1]; kk++) {
      const atomk = this.srList[kk];
      if (atomk !== atomj) {
        const rik0 = x[3 * atomi] - x[3 * atomk];
        const rik1 = x[3 * atomi + 1] - x[3 * atomk + 1];
        const rik2 = x[3 * atomi + 2] - x[3 * atomk + 2];
        const rikmag = Math.sqrt(rik0 * rik0 + rik1 * rik1 + rik2 * rik2);
        const df_c_ik = newRef();
        const f_c_ik = f_c(p, rikmag, p.r_1, p.r_2, df_c_ik);
        // native reads N[k] with k the neighbor-list slot of atomk (pair_lcbop.cpp FMij), not N[atomk]
        const Nki = N[kk - this.srStart[i]] - f_c_ik;
        const dF = newRef();
        const Fx = 1 - f_c_LR(Nki, 2, 3, dF);
        dF.v = -dF.v;
        if (df_c_ik.v > TOL) {
          const factor2 = factor * df_c_ik.v * Fx;
          // F = factor2*(-grad rikmag)
          const fpair = -factor2 / rikmag;
          f[3 * atomi] += rik0 * fpair;
          f[3 * atomi + 1] += rik1 * fpair;
          f[3 * atomi + 2] += rik2 * fpair;
          f[3 * atomk] -= rik0 * fpair;
          f[3 * atomk + 1] -= rik1 * fpair;
          f[3 * atomk + 2] -= rik2 * fpair;
          if (this.vatom) vTally2(this.vatom, atomi, atomk, fpair, rik0, rik1, rik2);
        }
        if (dF.v > TOL) {
          const factor2 = factor * f_c_ik * dF.v;
          this.FNij(atomk, atomi, factor2);
        }
      }
    }
  }

  /** PairLCBOP::bondorder: B_ij = (b_ij + b_ji + F_conj) / 2 and its force contributions. rij = x_i - x_j. */
  protected bondorder(i: number, j: number, rij: number[], rijmag: number, VA: number): number {
    const p = this.p!, T = this.T!;
    const N = this.N, M = this.M;
    // bij & bji
    const rji = [-rij[0], -rij[1], -rij[2]];
    const bij = this.b(i, j, rij, rijmag, VA);
    const bji = this.b(j, i, rji, rijmag, VA);

    // F_conj
    const df_c_ij = newRef();
    const f_c_ij = f_c(p, rijmag, p.r_1, p.r_2, df_c_ij);
    const Nij = cmin(3, N[i] - f_c_ij);
    const Nji = cmin(3, N[j] - f_c_ij);
    let Mij = M[i] - f_c_ij * (1 - f_c(p, Nji, 2, 3, this.dummy));
    let Mji = M[j] - f_c_ij * (1 - f_c(p, Nij, 2, 3, this.dummy));
    Mij = cmin(Mij, 3);
    Mji = cmin(Mji, 3);
    let Nij_el: number, dNij_el_dNij: number, dNij_el_dMij: number;
    let Nji_el: number, dNji_el_dNji: number, dNji_el_dMji: number;
    {
      const num_Nij_el = 4 - Mij;
      const num_Nji_el = 4 - Mji;
      const den_Nij_el = Nij + 1 - Mij;
      const den_Nji_el = Nji + 1 - Mji;
      Nij_el = num_Nij_el / den_Nij_el;
      Nji_el = num_Nji_el / den_Nji_el;
      dNij_el_dNij = -Nij_el / den_Nij_el;
      dNji_el_dNji = -Nji_el / den_Nji_el;
      dNij_el_dMij = (-1 + Nij_el) / den_Nij_el;
      dNji_el_dMji = (-1 + Nji_el) / den_Nji_el;
    }
    let Nconj: number, dNconj_dNij = 0, dNconj_dNji = 0, dNconj_dNel = 0;
    {
      const num_Nconj = (Nij + 1) * (Nji + 1) * (Nij_el + Nji_el) - 4 * (Nij + Nji + 2);
      const den_Nconj = Nij * (3 - Nij) * (Nji + 1) + Nji * (3 - Nji) * (Nij + 1) + p.eps;
      Nconj = num_Nconj / den_Nconj;
      if (Nconj <= 0) {
        Nconj = 0;
        dNconj_dNij = 0;
        dNconj_dNji = 0;
        dNconj_dNel = 0;
      } else if (Nconj >= 1) {
        Nconj = 1;
        dNconj_dNij = 0;
        dNconj_dNji = 0;
        dNconj_dNel = 0;
      } else {
        dNconj_dNij = (
          ((Nji + 1) * (Nij_el + Nji_el) - 4)
          - Nconj * ((Nji + 1) * (3 - 2 * Nij) + Nji * (3 - Nji))
        ) / den_Nconj;
        dNconj_dNji = (
          ((Nij + 1) * (Nji_el + Nij_el) - 4)
          - Nconj * ((Nij + 1) * (3 - 2 * Nji) + Nij * (3 - Nij))
        ) / den_Nconj;
        dNconj_dNel = (Nij + 1) * (Nji + 1) / den_Nconj;
      }
    }
    const d: FConjD = { dij: 0, dji: 0, dconj: 0 };
    const Fij_conj = F_conj(T, Nij, Nji, Nconj, d);
    const dF_dNij = d.dij, dF_dNji = d.dji, dF_dNconj = d.dconj;

    // forces for Nij
    if (3 - Nij > TOL) {
      const factor = -VA * 0.5 * (dF_dNij + dF_dNconj * (dNconj_dNij + dNconj_dNel * dNij_el_dNij));
      this.FNij(i, j, factor);
    }
    // forces for Nji
    if (3 - Nji > TOL) {
      const factor = -VA * 0.5 * (dF_dNji + dF_dNconj * (dNconj_dNji + dNconj_dNel * dNji_el_dNji));
      this.FNij(j, i, factor);
    }
    // forces for Mij
    if (3 - Mij > TOL) {
      const factor = -VA * 0.5 * (dF_dNconj * dNconj_dNel * dNij_el_dMij);
      this.FMij(i, j, factor);
    }
    if (3 - Mji > TOL) {
      const factor = -VA * 0.5 * (dF_dNconj * dNconj_dNel * dNji_el_dMji);
      this.FMij(j, i, factor);
    }
    return 0.5 * (bij + bji + Fij_conj);
  }

  /** PairLCBOP::b: the bond-order term b_ij (its value is returned; forces go to f). rij = x_i - x_j. */
  protected b(i: number, j: number, rij: number[], rijmag: number, VA: number): number {
    const p = this.p!, T = this.T!;
    const x = this.x, f = this.f;
    const atomi = i, atomj = j;
    const srStart = this.srStart, srList = this.srList;
    // calculate bij magnitude
    let bij = 1.0;
    for (let kk = srStart[i]; kk < srStart[i + 1]; kk++) {
      const atomk = srList[kk];
      if (atomk !== atomj) {
        const rik0 = x[3 * atomi] - x[3 * atomk];
        const rik1 = x[3 * atomi + 1] - x[3 * atomk + 1];
        const rik2 = x[3 * atomi + 2] - x[3 * atomk + 2];
        const rikmag = Math.sqrt(rik0 * rik0 + rik1 * rik1 + rik2 * rik2);
        const delta_ijk = rijmag - rikmag;
        const dummy = newRef();
        const f_c_ik = f_c(p, rikmag, p.r_1, p.r_2, dummy);
        let cos_ijk = (rij[0] * rik0 + rij[1] * rik1 + rij[2] * rik2) / (rijmag * rikmag);
        cos_ijk = cmin(cos_ijk, 1.0);
        cos_ijk = cmax(cos_ijk, -1.0);
        const G = gSpline(T, cos_ijk, dummy);
        const H = hSpline(p, delta_ijk, dummy);
        bij += f_c_ik * G * H;
      }
    }
    bij = pow(bij, -p.delta);

    // bij forces
    for (let kk = srStart[i]; kk < srStart[i + 1]; kk++) {
      const atomk = srList[kk];
      if (atomk !== atomj) {
        const rik0 = x[3 * atomi] - x[3 * atomk];
        const rik1 = x[3 * atomi + 1] - x[3 * atomk + 1];
        const rik2 = x[3 * atomi + 2] - x[3 * atomk + 2];
        const rikmag = Math.sqrt(rik0 * rik0 + rik1 * rik1 + rik2 * rik2);
        const delta_ijk = rijmag - rikmag;
        const df_c_ik = newRef();
        const f_c_ik = f_c(p, rikmag, p.r_1, p.r_2, df_c_ik);
        let cos_ijk = (rij[0] * rik0 + rij[1] * rik1 + rij[2] * rik2) / (rijmag * rikmag);
        cos_ijk = cmin(cos_ijk, 1.0);
        cos_ijk = cmax(cos_ijk, -1.0);
        const dcos_ijk_drj0 = -rik0 / (rijmag * rikmag) + cos_ijk * rij[0] / (rijmag * rijmag);
        const dcos_ijk_drj1 = -rik1 / (rijmag * rikmag) + cos_ijk * rij[1] / (rijmag * rijmag);
        const dcos_ijk_drj2 = -rik2 / (rijmag * rikmag) + cos_ijk * rij[2] / (rijmag * rijmag);
        const dcos_ijk_drk0 = -rij[0] / (rijmag * rikmag) + cos_ijk * rik0 / (rikmag * rikmag);
        const dcos_ijk_drk1 = -rij[1] / (rijmag * rikmag) + cos_ijk * rik1 / (rikmag * rikmag);
        const dcos_ijk_drk2 = -rij[2] / (rijmag * rikmag) + cos_ijk * rik2 / (rikmag * rikmag);
        const dcos_ijk_dri0 = -dcos_ijk_drk0 - dcos_ijk_drj0;
        const dcos_ijk_dri1 = -dcos_ijk_drk1 - dcos_ijk_drj1;
        const dcos_ijk_dri2 = -dcos_ijk_drk2 - dcos_ijk_drj2;
        const dG = newRef(), dH = newRef();
        const G = gSpline(T, cos_ijk, dG);
        const H = hSpline(p, delta_ijk, dH);
        const tmp = -VA * 0.5 * (-0.5 * bij * bij * bij);
        let fi0, fi1, fi2, fj0, fj1, fj2, fk0, fk1, fk2;
        let tmp2 = -tmp * df_c_ik.v * G * H / rikmag;
        // F = tmp*df_c_ik*G*H*(-grad rikmag)
        fi0 = tmp2 * rik0;
        fi1 = tmp2 * rik1;
        fi2 = tmp2 * rik2;
        fk0 = -tmp2 * rik0;
        fk1 = -tmp2 * rik1;
        fk2 = -tmp2 * rik2;
        tmp2 = -tmp * f_c_ik * dG.v * H;
        // F = tmp*f_c_ik*dG*H*(-grad cos_ijk)
        fi0 += tmp2 * dcos_ijk_dri0;
        fi1 += tmp2 * dcos_ijk_dri1;
        fi2 += tmp2 * dcos_ijk_dri2;
        fj0 = tmp2 * dcos_ijk_drj0;
        fj1 = tmp2 * dcos_ijk_drj1;
        fj2 = tmp2 * dcos_ijk_drj2;
        fk0 += tmp2 * dcos_ijk_drk0;
        fk1 += tmp2 * dcos_ijk_drk1;
        fk2 += tmp2 * dcos_ijk_drk2;
        tmp2 = -tmp * f_c_ik * G * dH.v;
        // F = tmp*f_c_ik*G*dH*(-grad delta_ijk)
        fi0 += tmp2 * (rij[0] / rijmag - rik0 / rikmag);
        fi1 += tmp2 * (rij[1] / rijmag - rik1 / rikmag);
        fi2 += tmp2 * (rij[2] / rijmag - rik2 / rikmag);
        fj0 += tmp2 * (-rij[0] / rijmag);
        fj1 += tmp2 * (-rij[1] / rijmag);
        fj2 += tmp2 * (-rij[2] / rijmag);
        fk0 += tmp2 * (rik0 / rikmag);
        fk1 += tmp2 * (rik1 / rikmag);
        fk2 += tmp2 * (rik2 / rikmag);
        f[3 * atomi] += fi0; f[3 * atomi + 1] += fi1; f[3 * atomi + 2] += fi2;
        f[3 * atomj] += fj0; f[3 * atomj + 1] += fj1; f[3 * atomj + 2] += fj2;
        f[3 * atomk] += fk0; f[3 * atomk + 1] += fk1; f[3 * atomk + 2] += fk2;
        if (this.vatom) {
          vTally3(this.vatom, atomi, atomj, atomk, [fj0, fj1, fj2], [fk0, fk1, fk2], [-rij[0], -rij[1], -rij[2]],
            [-rik0, -rik1, -rik2]);
        }
      }
    }
    return bij;
  }

  /** Atom ids of the owned and ghost atoms (the image owner's id). */
  protected tags(pc: PairCompute): Int32Array {
    const owner = pc.nb.owner;
    const tag = new Int32Array(pc.nall);
    for (let a = 0; a < pc.nall; a++) tag[a] = pc.s.id[owner[a]];
    return tag;
  }

  /** Energy tally of one pair (ev_tally with newton on: the full energy, half per atom in eatom). */
  protected tally(acc: PairCompute['acc'], eatom: Float64Array | null, i: number, j: number, evdwl: number): void {
    acc.evdwl += evdwl;
    if (eatom) {
      eatom[i] += 0.5 * evdwl;
      eatom[j] += 0.5 * evdwl;
    }
  }
}

