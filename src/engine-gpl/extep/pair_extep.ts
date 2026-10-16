/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_extep.cpp and src/MANYBODY/pair_extep.h at commit b332f28 (contributing author Jan Los).
 * Copyright Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * pair_style extep — the ExTeP (extended Tersoff potential with a bicubic spline correction F_IJ of the
 * coordination numbers, Los et al. for BN). Docs: docs.lammps.org/pair_extep.html ("pair_style extep" takes no
 * arguments; "pair_coeff * * BN.extep B N" with the potential file and one element per atom type).
 *
 * The energy is the Tersoff form with the bond order B_ij multiplied by the envelope and the spline correction:
 *   E = sum_{i<j} [ f_R(r_ij) + b_ij f_A(r_ij) ] (with the Tersoff terms of the ExTeP parameter set), where the
 *   attractive part carries F_corr(N_ij^d, N_ji^d) * envelope(N_ij^t, N_ji^t) (coordination sums of the cutoff
 *   functions, Nt over all neighbours and Nd over the other-element neighbours).
 * The code follows native compute() line by line: the repulsive pair term uses the atom-ID half rule of the full
 * list; the bond-order terms run over every ordered pair (each with 0.5 factors), as native does.
 *
 * Neighbour data: the engine's full list of the owned atoms (with ghost atoms, as in comb.ts). Native SR_neigh counts
 * the coordination of the ghost atoms from their own lists; a periodic image has the same coordination as its owner
 * (the owner's list is complete within the cutoff), so Nt/Nd of a ghost is taken from its owner.
 * virialFdotr: the global virial is the dot product of positions and forces (ghosts included), as in the force field.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

// LAMMPS: static constexpr int MAXTYPES 8, NSPLINE 5 (pair_extep.h)
const NSPLINE = 5;
const PI2 = Math.PI / 2;
const PI4 = Math.PI / 4;

interface Param {
  lam1: number; lam2: number; lam3: number;
  c: number; d: number; h: number;
  gamma: number; powerm: number; powern: number; beta: number;
  biga: number; bigb: number; bigd: number; bigr: number;
  cut: number; cutsq: number;
  c1: number; c2: number; c3: number; c4: number;
  ielement: number; jelement: number; kelement: number;
  powermint: number;
}

/** F_IJ spline cell (pair_extep.h struct TF_corr_param). */
interface SplineCell {
  f00: number; f01: number; f10: number; f11: number;
  fx00: number; fx01: number; fx10: number; fx11: number;
  fy00: number; fy01: number; fy10: number; fy11: number;
}

/** words of a 3-body entry (native next_line(17)) */
const DELTA_NEXT = 17;

/** PotentialFileReader::next_line(n): the next non-comment line with n words (following lines are joined onto it). */
class EntryReader {
  private pos = 0;
  constructor(private readonly lines: readonly string[]) {}
  next(n: number): string[] | null {
    let words: string[] = [];
    while (this.pos < this.lines.length) {
      const raw = this.lines[this.pos++];
      const h = raw.indexOf('#');
      const t = (h >= 0 ? raw.slice(0, h) : raw).trim();
      if (t) words = words.concat(t.split(/\s+/));
      if (words.length >= n) return words;
    }
    return words.length >= n ? words : null;
  }
}

export class PairExTeP extends Pair {
  readonly name = 'extep';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected fileRead = false;
  protected elements: string[] = [];
  /** element index per atom type (1-based); -1 for NULL. */
  protected map = new Int32Array(0);
  protected params: Param[] = [];
  /** elem3param[(i*nel + j)*nel + k] = parameter index. */
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  /** F_corr_data[((iel*MAXT + jel)*NSPLINE + Ni)*NSPLINE + Nj][0..2] as native (value, dx, dy). */
  protected fcorrData = new Float64Array(0);
  /** F_corr_param cells, indexed like the data without the component index. */
  protected fcorrParam: SplineCell[] = [];
  /** Coordination sums of the owned atoms (SR_neigh): Nt over all neighbours, Nd over other-element ones. */
  protected Nt = new Float64Array(0);
  protected Nd = new Float64Array(0);

  settings(args: string[]): void {
    if (args.length) throw new StyleError(`Illegal pair_style command: pair_style extep takes no arguments (got '${args[0]}')`);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length < 3) throw new StyleError('usage: pair_coeff * * filename elem1 ... elemN (style extep)');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style extep must be * *');
    const filename = args[2];
    const names = args.slice(3);
    if (names.length !== this.ntypes) {
      throw new StyleError(`pair_coeff for style extep needs one element name per atom type (${this.ntypes}), got ${names.length}`);
    }
    const elements: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const name = names[t - 1];
      if (name === 'NULL') throw new StyleError('pair_style extep does not support NULL in pair_coeff (every atom type needs an element)');
      let e = elements.indexOf(name);
      if (e < 0) { e = elements.length; elements.push(name); }
      map[t] = e;
    }
    const text = ctx.readFile(filename);
    const lines = text.split(/\r?\n/);
    const nel = elements.length;
    const params: Param[] = [];
    const reader = new EntryReader(lines);
    // first block: 3-body entries, read until one per element triple (native: nparams >= nelements^3 breaks)
    for (;;) {
      if (params.length >= nel ** 3) break;
      const w = reader.next(DELTA_NEXT);
      if (!w) break;
      const ie = elements.indexOf(w[0]), je = elements.indexOf(w[1]), ke = elements.indexOf(w[2]);
      if (ie < 0 || je < 0 || ke < 0) continue;
      const v = w.slice(3, 17).map((s, k) => {
        const x = Number(s);
        if (!Number.isFinite(x)) throw new StyleError(`ExTeP potential: bad number '${s}' (field ${k})`);
        return x;
      });
      const [powerm, gamma, lam3, c, d, h, powern, beta, lam2, bigb, bigr, bigd, lam1, biga] = v;
      const p: Param = {
        ielement: ie, jelement: je, kelement: ke,
        powerm, gamma, lam3, c, d, h, powern, beta, lam2, bigb, bigr, bigd, lam1, biga,
        cut: 0, cutsq: 0, c1: 0, c2: 0, c3: 0, c4: 0, powermint: Math.trunc(powerm),
      };
      if (p.c < 0 || p.d < 0 || p.powern < 0 || p.beta < 0 || p.lam2 < 0 || p.bigb < 0 || p.bigr < 0 || p.bigd < 0
        || p.bigd > p.bigr || p.lam1 < 0 || p.biga < 0 || p.powerm - p.powermint !== 0
        || (p.powermint !== 3 && p.powermint !== 1) || p.gamma < 0) {
        throw new StyleError('Illegal ExTeP parameter');
      }
      params.push(p);
    }
    // second block: F_IJ spline entries "el el Ni Nj value dvalue/dx dvalue/dy" (native next_line(8))
    const fcorr = new Float64Array(nel * nel * NSPLINE * NSPLINE * 3);
    const at = (a: number, b: number, n: number, m: number) => ((a * nel + b) * NSPLINE + n) * NSPLINE + m;
    for (;;) {
      const w = reader.next(8);
      if (!w) break;
      const ie = elements.indexOf(w[0]), je = elements.indexOf(w[1]);
      if (ie < 0 || je < 0) continue;
      if (!/^[+-]?\d+$/.test(w[2])) continue;
      const ni = Number(w[2]), nj = Number(w[3]);
      const val = Number(w[4]), dx = Number(w[5]), dy = Number(w[6]);
      if (!params.some((p) => p.ielement === ie && p.jelement === je)) continue;
      if (ni < 0 || ni >= NSPLINE || nj < 0 || nj >= NSPLINE) continue;
      fcorr[at(ie, je, ni, nj) * 3] = val;
      fcorr[at(ie, je, ni, nj) * 3 + 1] = dx;
      fcorr[at(ie, je, ni, nj) * 3 + 2] = dy;
      // the symmetric entry swaps the two derivatives (native F_corr_data[jel][iel][Nj][Ni])
      fcorr[at(je, ie, nj, ni) * 3] = val;
      fcorr[at(je, ie, nj, ni) * 3 + 1] = dy;
      fcorr[at(je, ie, nj, ni) * 3 + 2] = dx;
    }
    // setup(): element-triple table, cutoffs (native PairExTeP::setup)
    const elem3 = new Int32Array(nel * nel * nel).fill(-1);
    for (let i = 0; i < nel; i++) for (let j = 0; j < nel; j++) for (let k = 0; k < nel; k++) {
      let n = -1;
      for (let m = 0; m < params.length; m++) {
        const p = params[m];
        if (i === p.ielement && j === p.jelement && k === p.kelement) {
          if (n >= 0) throw new StyleError(`Potential file has a duplicate entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
          n = m;
        }
      }
      if (n < 0) throw new StyleError(`Potential file is missing an entry for: ${elements[i]} ${elements[j]} ${elements[k]}`);
      elem3[(i * nel + j) * nel + k] = n;
    }
    let cutmax = 0;
    for (const p of params) {
      p.cut = p.bigr + p.bigd;
      p.cutsq = p.cut * p.cut;
      p.c1 = (2 * p.powern * 1.0e-16) ** (-1 / p.powern);
      p.c2 = (2 * p.powern * 1.0e-8) ** (-1 / p.powern);
      p.c3 = 1 / p.c2;
      p.c4 = 1 / p.c1;
      if (p.cut > cutmax) cutmax = p.cut;
    }
    // spline_init (native): the 12 coefficients of each bicubic cell
    const cells: SplineCell[] = [];
    for (let iel = 0; iel < nel; iel++) for (let jel = 0; jel < nel; jel++) for (let nij = 0; nij < 4; nij++) for (let nji = 0; nji < 4; nji++) {
      const V = (a: number, b: number, c: number) => fcorr[at(iel, jel, a, b) * 3 + c];
      const f00 = V(nij, nji, 0), f01 = V(nij, nji + 1, 0), f10 = V(nij + 1, nji, 0), f11 = V(nij + 1, nji + 1, 0);
      cells.push({
        f00, f01, f10, f11,
        fx00: V(nij, nji, 1) - f10 + f00,
        fx01: V(nij, nji + 1, 1) - f11 + f01,
        fx10: -(V(nij + 1, nji, 1) - f10 + f00),
        fx11: -(V(nij + 1, nji + 1, 1) - f11 + f01),
        fy00: V(nij, nji, 2) - f01 + f00,
        fy01: -(V(nij, nji + 1, 2) - f01 + f00),
        fy10: V(nij + 1, nji, 2) - f11 + f10,
        fy11: -(V(nij + 1, nji + 1, 2) - f11 + f10),
      });
    }
    this.fileRead = true;
    this.elements = elements;
    this.map = map;
    this.params = params;
    this.elem3param = elem3;
    this.cutmax = cutmax;
    this.fcorrData = fcorr;
    this.fcorrParam = cells;
  }

  initStyle(): void {
    if (!this.fileRead) throw new StyleError('pair_style extep needs a pair_coeff command with an ExTeP potential file');
    if (this.shift || this.tail) throw new StyleError('pair_style extep does not support the pair_modify shift and tail options');
  }

  initOne(): number {
    return this.cutmax;
  }

  private p3(i: number, j: number, k: number): Param {
    const nel = this.elements.length;
    return this.params[this.elem3param[(i * nel + j) * nel + k]];
  }

  // ---- functions of pair_extep.cpp ----
  private fc(r: number, p: Param): number {
    const R = p.bigr, D = p.bigd;
    if (r < R - D) return 1.0;
    if (r > R + D) return 0.0;
    return 0.5 * (1.0 - libmSin(PI2 * (r - R) / D));
  }

  private fcD(r: number, p: Param): number {
    const R = p.bigr, D = p.bigd;
    if (r < R - D) return 0.0;
    if (r > R + D) return 0.0;
    return -(PI4 / D) * libmCos(PI2 * (r - R) / D);
  }

  private fa(r: number, p: Param): number {
    if (r > p.bigr + p.bigd) return 0.0;
    return -p.bigb * Math.exp(-p.lam2 * r) * this.fc(r, p);
  }

  private faD(r: number, p: Param): number {
    if (r > p.bigr + p.bigd) return 0.0;
    return p.bigb * Math.exp(-p.lam2 * r) * (p.lam2 * this.fc(r, p) - this.fcD(r, p));
  }

  private bij(zeta: number, p: Param): number {
    const tmp = p.beta * zeta;
    if (tmp > p.c1) return 1.0 / Math.sqrt(tmp);
    if (tmp > p.c2) return (1.0 - tmp ** (-p.powern) / (2.0 * p.powern)) / Math.sqrt(tmp);
    if (tmp < p.c4) return 1.0;
    if (tmp < p.c3) return 1.0 - tmp ** p.powern / (2.0 * p.powern);
    return (1.0 + tmp ** p.powern) ** (-1.0 / (2.0 * p.powern));
  }

  private bijD(zeta: number, p: Param): number {
    const tmp = p.beta * zeta;
    if (tmp > p.c1) return p.beta * -0.5 * tmp ** -1.5;
    if (tmp > p.c2) return p.beta * (-0.5 * tmp ** -1.5 * (1.0 - 0.5 * (1.0 + 1.0 / (2.0 * p.powern)) * tmp ** (-p.powern)));
    if (tmp < p.c4) return 0.0;
    if (tmp < p.c3) return -0.5 * p.beta * tmp ** (p.powern - 1.0);
    const tmpN = tmp ** p.powern;
    return -0.5 * (1.0 + tmpN) ** (-1.0 - 1.0 / (2.0 * p.powern)) * tmpN / zeta;
  }

  private gijk(cos: number, p: Param): number {
    const c2 = p.c * p.c, d2 = p.d * p.d;
    const hcth = p.h - cos;
    return p.gamma * (1.0 + c2 / d2 - c2 / (d2 + hcth * hcth));
  }

  private gijkD(cos: number, p: Param): number {
    const c2 = p.c * p.c, d2 = p.d * p.d;
    const hcth = p.h - cos;
    const numerator = -2.0 * c2 * hcth;
    const denominator = 1.0 / (d2 + hcth * hcth);
    return p.gamma * numerator * denominator * denominator;
  }

  /** exp(arg) of the lam3 term with native's overflow guards (zeta and zetaterm_d use the same). */
  private exDelr(rij: number, rik: number, p: Param): number {
    const arg = p.powermint === 3 ? (p.lam3 * (rij - rik)) ** 3.0 : p.lam3 * (rij - rik);
    if (arg > 69.0776) return 1.e30;
    if (arg < -69.0776) return 0.0;
    return Math.exp(arg);
  }

  /** zeta term of one k neighbour (native PairExTeP::zeta). */
  private zeta(p: Param, rij: number, rik: number, dij: number[], dik: number[]): number {
    const cos = (dij[0] * dik[0] + dij[1] * dik[1] + dij[2] * dik[2]) / (rij * rik);
    return this.fc(rik, p) * this.gijk(cos, p) * this.exDelr(rij, rik, p);
  }

  /** Bond-order term of one pair (native force_zeta): fpair, prefactor and energy. */
  private forceZeta(p: Param, r: number, zetaIJ: number): { fpair: number; prefactor: number; eng: number } {
    const fa = this.fa(r, p), faD = this.faD(r, p);
    const b = this.bij(zetaIJ, p);
    return {
      fpair: (0.5 * b * faD) / r,
      prefactor: -0.5 * fa * this.bijD(zetaIJ, p),
      eng: 0.5 * b * fa,
    };
  }

  /** native PairExTeP::ters_zetaterm_d: the three-body forces of one (i; j, k) triple, written to fi, fj, fk. */
  private zetaTermD(prefactor: number, rijHat: number[], rij: number, rikHat: number[], rik: number, p: Param,
    dri: number[], drj: number[], drk: number[]): void {
    const fc = this.fc(rik, p), dfc = this.fcD(rik, p);
    const ex = this.exDelr(rij, rik, p);
    const exD = p.powermint === 3
      ? 3.0 * p.lam3 ** 3.0 * (rij - rik) ** 2.0 * ex
      : p.lam3 * ex;
    const cos = rijHat[0] * rikHat[0] + rijHat[1] * rikHat[1] + rijHat[2] * rikHat[2];
    const g = this.gijk(cos, p), gD = this.gijkD(cos, p);
    // costheta_d
    const dcosdrj = [0, 0, 0], dcosdrk = [0, 0, 0], dcosdri = [0, 0, 0];
    for (let q = 0; q < 3; q++) {
      dcosdrj[q] = (-cos * rijHat[q] + rikHat[q]) * (1.0 / rij);
      dcosdrk[q] = (-cos * rikHat[q] + rijHat[q]) * (1.0 / rik);
      dcosdri[q] = -(dcosdrj[q] + dcosdrk[q]);
    }
    for (let q = 0; q < 3; q++) {
      let v = -dfc * g * ex * rikHat[q];
      v = fc * gD * ex * dcosdri[q] + v;
      v = fc * g * exD * rikHat[q] + v;
      v = -fc * g * exD * rijHat[q] + v;
      dri[q] = prefactor * v;
      let w = fc * gD * ex * dcosdrj[q];
      w = fc * g * exD * rijHat[q] + w;
      drj[q] = prefactor * w;
      let u = dfc * g * ex * rikHat[q];
      u = fc * gD * ex * dcosdrk[q] + u;
      u = -fc * g * exD * rikHat[q] + u;
      drk[q] = prefactor * u;
    }
  }

  /** native PairExTeP::envelop_function: value and d/dx of the envelope of the two coordinations. */
  private envelope(x: number, y: number): { val: number; der: number } {
    let fx: number, fy: number, dfx: number;
    let fxy = 1.0, dfxydx = 0.0;
    if (x <= 3.0) {
      fx = 1.0;
      dfx = 0.0;
      if (x < 1.0 && y < 1.0) {
        const gx = 1.0 - x, gy = 1.0 - y;
        const gxsq = gx * gx, gysq = gy * gy;
        fxy = 1.0 - gxsq * gysq;
        dfxydx = 2.0 * gx * gysq;
      }
    } else if (x < 4.0) {
      const del = 4.0 - x;
      const delsq = del * del;
      fx = (3.0 - 2.0 * del) * delsq;
      dfx = -6.0 * del * (1.0 - del);
    } else {
      fx = 0.0;
      dfx = 0.0;
    }
    if (y <= 3.0) {
      fy = 1.0;
    } else if (y < 4.0) {
      const del = 4.0 - y;
      const delsq = del * del;
      fy = (3.0 - 2.0 * del) * delsq;
    } else {
      fy = 0.0;
    }
    return { val: fxy * fx * fy, der: dfxydx * fx * fy + fxy * dfx * fy };
  }

  /** native PairExTeP::F_corr: bicubic spline of the coordinations of elements iel, jel; derivatives in dx, dy. */
  private fCorr(iel: number, jel: number, Ndij: number, Ndji: number): { F: number; dx: number; dy: number } {
    const ni = Math.floor(Ndij), nj = Math.floor(Ndji);
    const x = Ndij - ni, y = Ndji - nj;
    let F = 0, dFx = 0, dFy = 0;
    if (ni < 4 && nj < 4) {
      const nel = this.elements.length;
      const f = this.fcorrParam[((iel * nel + jel) * 4 + ni) * 4 + nj];
      let l = (1 - y) * (1 - x);
      let r = f.f00 + x * x * f.fx10 + y * y * f.fy01;
      F += l * r;
      dFx += -(1 - y) * r + l * 2 * x * f.fx10;
      dFy += -(1 - x) * r + l * 2 * y * f.fy01;
      l = (1 - y) * x;
      r = f.f10 + (1 - x) * (1 - x) * f.fx00 + y * y * f.fy11;
      F += l * r;
      dFx += (1 - y) * r - l * 2 * (1 - x) * f.fx00;
      dFy += -x * r + l * 2 * y * f.fy11;
      l = y * (1 - x);
      r = f.f01 + x * x * f.fx11 + (1 - y) * (1 - y) * f.fy00;
      F += l * r;
      dFx += -y * r + l * 2 * x * f.fx11;
      dFy += (1 - x) * r - l * 2 * (1 - y) * f.fy00;
      l = y * x;
      r = f.f11 + (1 - x) * (1 - x) * f.fx01 + (1 - y) * (1 - y) * f.fy10;
      F += l * r;
      dFx += y * r - l * 2 * (1 - x) * f.fx01;
      dFy += x * r - l * 2 * (1 - y) * f.fy10;
    }
    return { F, dx: dFx, dy: dFy };
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style extep needs a full neighbor list');
    if (!this.fileRead) throw new StyleError('pair_style extep needs a pair_coeff command first');
    const nmask = NEIGHMASK;
    const x = pc.x, f = pc.f, type = pc.type;
    const nlocal = pc.nlocal;
    const owner = pc.nb.owner;
    const ids = pc.s.id;
    const acc = pc.acc;
    const eatom = pc.eatom;
    const map = this.map;
    const tagOf = (j: number): number => ids[owner[j]];
    const ownerOf = (j: number): number => (j < nlocal ? j : owner[j]);

    // SR_neigh: coordination sums of the owned atoms (native loops over owned and ghost atoms; a ghost's sums are its owner's)
    if (this.Nt.length < nlocal) { this.Nt = new Float64Array(nlocal); this.Nd = new Float64Array(nlocal); }
    for (let i = 0; i < nlocal; i++) {
      const itype = map[type[i]];
      let nt = 0, nd = 0;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const jtype = map[type[j]];
        const dx = x[3 * i] - x[3 * j], dy = x[3 * i + 1] - x[3 * j + 1], dz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = dx * dx + dy * dy + dz * dz;
        const p = this.p3(itype, jtype, jtype);
        if (rsq < p.cutsq) {
          const fcv = this.fc(Math.sqrt(rsq), p);
          nt += fcv;
          if (itype !== jtype) nd += fcv;
        }
      }
      this.Nt[i] = nt;
      this.Nd[i] = nd;
    }
    const Nt = this.Nt, Nd = this.Nd;

    for (let i = 0; i < nlocal; i++) {
      const itag = ids[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = map[type[i]];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      const jnum = list.numneigh[i];

      // repulsive pair term (half of the full list by atom IDs)
      for (let kk = k0; kk < k1; kk++) {
        const j = list.neighbors[kk] & nmask;
        const jtag = tagOf(j);
        if (itag > jtag) {
          if ((itag + jtag) % 2 === 0) continue;
        } else if (itag < jtag) {
          if ((itag + jtag) % 2 === 1) continue;
        } else {
          if (x[3 * j + 2] < ztmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
          if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
        }
        const jtype = map[type[j]];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const p = this.p3(itype, jtype, jtype);
        if (rsq > p.cutsq) continue;
        // repulsive (native PairExTeP::repulsive)
        const r = Math.sqrt(rsq);
        const fcv = this.fc(r, p), fcD = this.fcD(r, p);
        const ex = Math.exp(-p.lam1 * r);
        const fpair = (-p.biga * ex * (fcD - fcv * p.lam1)) / r;
        const eng = fcv * p.biga * ex;
        f[3 * i] += delx * fpair; f[3 * i + 1] += dely * fpair; f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair; f[3 * j + 1] -= dely * fpair; f[3 * j + 2] -= delz * fpair;
        acc.evdwl += eng;
        if (eatom) { eatom[i] += 0.5 * eng; eatom[j] += 0.5 * eng; }
      }

      // bond-order terms: every ordered neighbour pair (native second loop over jlist)
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & nmask;
        const jtype = map[type[j]];
        const p = this.p3(itype, jtype, jtype);
        const delr1 = [x[3 * j] - xtmp, x[3 * j + 1] - ytmp, x[3 * j + 2] - ztmp];
        const rsq1 = delr1[0] * delr1[0] + delr1[1] * delr1[1] + delr1[2] * delr1[2];
        if (rsq1 > p.cutsq) continue;
        let zetaIJ = 0.0;
        // F_IJ (1): spline correction and envelope
        let Ntij = Nt[i], Ndij = Nd[i];
        const jo = ownerOf(j);
        let Ntji = Nt[jo], Ndji = Nd[jo];
        const r = Math.sqrt(rsq1);
        const fcIJ = this.fc(r, p);
        Ntij -= fcIJ;
        Ntji -= fcIJ;
        if (jtype !== itype) {
          Ndij -= fcIJ;
          Ndji -= fcIJ;
        }
        if (Ntij < 0) Ntij = 0.0;
        if (Ndij < 0) Ndij = 0.0;
        if (Ntji < 0) Ntji = 0.0;
        if (Ndji < 0) Ndji = 0.0;
        const sp = this.fCorr(itype, jtype, Ndij, Ndji);
        const env = this.envelope(Ntij, Ntji);
        const Fc = env.val * sp.F;
        const dFcdNtij = env.der * sp.F;
        const dFcdNdij = env.val * sp.dx;
        const fa = this.fa(r, p), faD = this.faD(r, p);
        const deng = 0.5 * fa * Fc;
        const fpair = (0.5 * faD * Fc) / r;
        f[3 * i] += delr1[0] * fpair; f[3 * i + 1] += delr1[1] * fpair; f[3 * i + 2] += delr1[2] * fpair;
        f[3 * j] -= delr1[0] * fpair; f[3 * j + 1] -= delr1[1] * fpair; f[3 * j + 2] -= delr1[2] * fpair;
        acc.evdwl += deng;
        if (eatom) { eatom[i] += 0.5 * deng; eatom[j] += 0.5 * deng; }

        for (let kk = k0; kk < k1; kk++) {
          if (kk - k0 === jj) continue;
          const k = list.neighbors[kk] & nmask;
          const ktype = map[type[k]];
          const pk = this.p3(itype, jtype, ktype);
          const delr2 = [x[3 * k] - xtmp, x[3 * k + 1] - ytmp, x[3 * k + 2] - ztmp];
          const rsq2 = delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];
          if (rsq2 > pk.cutsq) continue;
          const r2 = Math.sqrt(rsq2);
          zetaIJ += this.zeta(pk, r, r2, delr1, delr2);
          // F_IJ (2): forces of the coordination sums on i and k
          const pik = this.p3(itype, ktype, 0);
          const fcIKd = this.fcD(r2, pik);
          const pref0 = (1.0 * fcIKd * fa) / r2;
          let pref = dFcdNtij * pref0;
          f[3 * i] += pref * delr2[0]; f[3 * i + 1] += pref * delr2[1]; f[3 * i + 2] += pref * delr2[2];
          f[3 * k] -= pref * delr2[0]; f[3 * k + 1] -= pref * delr2[1]; f[3 * k + 2] -= pref * delr2[2];
          if (itype !== ktype) {
            pref = dFcdNdij * pref0;
            f[3 * i] += pref * delr2[0]; f[3 * i + 1] += pref * delr2[1]; f[3 * i + 2] += pref * delr2[2];
            f[3 * k] -= pref * delr2[0]; f[3 * k + 1] -= pref * delr2[1]; f[3 * k + 2] -= pref * delr2[2];
          }
        }

        // bond-order energy and its force along the bond (native force_zeta)
        const fz = this.forceZeta(p, r, zetaIJ);
        f[3 * i] += delr1[0] * fz.fpair; f[3 * i + 1] += delr1[1] * fz.fpair; f[3 * i + 2] += delr1[2] * fz.fpair;
        f[3 * j] -= delr1[0] * fz.fpair; f[3 * j + 1] -= delr1[1] * fz.fpair; f[3 * j + 2] -= delr1[2] * fz.fpair;
        acc.evdwl += fz.eng;
        if (eatom) { eatom[i] += 0.5 * fz.eng; eatom[j] += 0.5 * fz.eng; }

        // three-body forces (native attractive) on i, j, k
        for (let kk = k0; kk < k1; kk++) {
          if (kk - k0 === jj) continue;
          const k = list.neighbors[kk] & nmask;
          const ktype = map[type[k]];
          const pk = this.p3(itype, jtype, ktype);
          const delr2 = [x[3 * k] - xtmp, x[3 * k + 1] - ytmp, x[3 * k + 2] - ztmp];
          const rsq2 = delr2[0] * delr2[0] + delr2[1] * delr2[1] + delr2[2] * delr2[2];
          if (rsq2 > pk.cutsq) continue;
          const rik = Math.sqrt(rsq2);
          const rijInv = 1.0 / r, rikInv = 1.0 / rik;
          const rijHat = [delr1[0] * rijInv, delr1[1] * rijInv, delr1[2] * rijInv];
          const rikHat = [delr2[0] * rikInv, delr2[1] * rikInv, delr2[2] * rikInv];
          const fi = [0, 0, 0], fj = [0, 0, 0], fk = [0, 0, 0];
          this.zetaTermD(fz.prefactor, rijHat, r, rikHat, rik, pk, fi, fj, fk);
          for (let q = 0; q < 3; q++) {
            f[3 * i + q] += fi[q];
            f[3 * j + q] += fj[q];
            f[3 * k + q] += fk[q];
          }
        }
      }
    }
  }
}
