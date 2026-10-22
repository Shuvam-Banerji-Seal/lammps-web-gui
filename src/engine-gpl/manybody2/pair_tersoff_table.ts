/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair_style tersoff/table
 * (src/MANYBODY/pair_tersoff_table.cpp, pair_tersoff_table.h). Contributing author: Luca Ferraro (CASPUR); the
 * tabulation follows the Tersoff_2 functional form (Tersoff, Phys. Rev. B 39, 5566 (1988)). Copyright Sandia
 * Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_tersoff.html (tersoff/table). The potential file is a Tersoff file; only the Tersoff_2
 * parametrization is accepted (gamma = 1 and lambda3 = 0). The cutoff function, the angular function g, the exponentials
 * and the zeta power beta^n zeta^n are read from pre-computed grids with linear interpolation, as native LAMMPS does.
 * Forces, energy and the three-body loops run in the order of the native compute() (a j-loop over the neighbors, with the
 * k loops below and above j), so the floating-point sums agree.
 */
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import {
  checkOneCoeffArgs, mapElementsToTypes, nextDouble, PotentialEntryReader, potentialUnitFactor, tallyPair,
  tallyTriplet,
} from './common';

// LAMMPS MathConst::MY_PI (src/math_const.h)
const MY_PI = 3.14159265358979323846;

// pair_tersoff_table.cpp: GRIDSTART, the grid densities and leadingDimensionInteractionList (64 neighbors per atom)
const GRIDSTART = 0.1;
const GRIDDENSITY_FCUTOFF = 5000;
const GRIDDENSITY_EXP = 12000;
const GRIDDENSITY_GTETA = 12000;
const GRIDDENSITY_BIJ = 7500;
const LEADING = 64;

const NPARAMS_PER_LINE = 17;

/** One element triplet of the Tersoff file (PairTersoffTable::Param). */
export interface TTParam {
  powerm: number; gamma: number; lam3: number;
  c: number; d: number; h: number;
  powern: number; beta: number; lam2: number;
  bigb: number; cutoffR: number; cutoffS: number;
  lam1: number; biga: number;
  ielement: number; jelement: number; kelement: number;
  cut: number; cutsq: number;
}

export class PairTersoffTable extends Pair {
  override readonly name: string = 'tersoff/table';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected elements: string[] = [];
  protected map: Int32Array = new Int32Array(0);
  protected params: TTParam[] = [];
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  protected fileRead = false;

  // grids (PairTersoffTable::allocateGrids)
  protected exponential = new Float64Array(0);
  protected minArgumentExponential = 0;
  protected numGteta = 0;
  protected gteta = new Float64Array(0);
  protected gtetaD = new Float64Array(0);
  protected ngridMax = 0;
  protected cutoffFunction = new Float64Array(0);
  protected cutoffFunctionD = new Float64Array(0);
  protected zetaMax = 0;
  protected betaZetaPower = new Float64Array(0);
  protected betaZetaPowerD = new Float64Array(0);

  // per-neighbor pre-computed values of one atom (PairTersoffTable::preCutoffFunction and friends)
  protected readonly preCut = new Float64Array(LEADING);
  protected readonly preCutD = new Float64Array(LEADING);
  protected readonly preG = new Float64Array(LEADING * LEADING);
  protected readonly preGD = new Float64Array(LEADING * LEADING);

  override settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    checkOneCoeffArgs(args, this.ntypes, this.name);
    const { elements, map } = mapElementsToTypes(args.slice(3), this.ntypes);
    this.elements = elements;
    this.map = map;
    this.readFile(args[2], ctx);
    this.setupParams();
    this.allocateGrids();
    this.fileRead = true;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style tersoff/table requires newton pair on');
  }

  override initOne(i: number, j: number): number {
    if (!this.fileRead || this.map[i] < 0 || this.map[j] < 0) throw new StyleError('All pair coeffs are not set');
    return this.cutmax;
  }

  /** PairTersoffTable::read_file (Tersoff file layout; the Tersoff_2 checks at the end of each entry). */
  protected readFile(fileName: string, ctx: StyleContext): void {
    const text = ctx.readFile(fileName, potentialFile('tersoff/table'));
    const factor = potentialUnitFactor(text, 'tersoff/table', fileName, ctx);
    const reader = new PotentialEntryReader(text.split(/\r?\n/), 'tersoff/table');
    const params: TTParam[] = [];
    for (;;) {
      const w = reader.next(NPARAMS_PER_LINE);
      if (!w) break;
      const ielement = this.elements.indexOf(w[0]);
      if (ielement < 0) continue;
      const jelement = this.elements.indexOf(w[1]);
      if (jelement < 0) continue;
      const kelement = this.elements.indexOf(w[2]);
      if (kelement < 0) continue;
      const v = (k: number) => nextDouble(w[3 + k]);
      const p: TTParam = {
        ielement, jelement, kelement,
        powerm: v(0), gamma: v(1), lam3: v(2),
        c: v(3), d: v(4), h: v(5), powern: v(6), beta: v(7), lam2: v(8), bigb: v(9),
        cutoffR: 0, cutoffS: 0, lam1: 0, biga: 0, cut: 0, cutsq: 0,
      };
      const bigr = v(10);
      const bigd = v(11);
      p.cutoffR = bigr - bigd;
      p.cutoffS = bigr + bigd;
      p.lam1 = v(12);
      p.biga = v(13);
      if (factor !== 1) {
        p.biga *= factor;
        p.bigb *= factor;
      }
      if (p.c < 0.0 || p.d < 0.0 || p.powern < 0.0 || p.beta < 0.0 || p.lam2 < 0.0 || p.bigb < 0.0 ||
        p.cutoffR < 0.0 || p.cutoffS < 0.0 || p.cutoffR > p.cutoffS || p.lam1 < 0.0 || p.biga < 0.0) {
        throw new StyleError('Illegal Tersoff parameter');
      }
      if (p.gamma !== 1.0 || p.lam3 !== 0.0) {
        throw new StyleError('Currently the tersoff/table pair_style only implements the Tersoff_2 parametrization');
      }
      params.push(p);
    }
    this.params = params;
  }

  /** PairTersoffTable::setup_params: the element triplet table and the cutoffs (cut = cutoffS). */
  protected setupParams(): void {
    const nel = this.elements.length;
    this.elem3param = new Int32Array(nel * nel * nel);
    for (let i = 0; i < nel; i++) {
      for (let j = 0; j < nel; j++) {
        for (let k = 0; k < nel; k++) {
          let n = -1;
          for (let m = 0; m < this.params.length; m++) {
            const p = this.params[m];
            if (i === p.ielement && j === p.jelement && k === p.kelement) {
              if (n >= 0) {
                throw new StyleError(`Potential file has a duplicate entry for: ${this.elements[i]} ${this.elements[j]} ${this.elements[k]}`);
              }
              n = m;
            }
          }
          if (n < 0) {
            throw new StyleError(`Potential file is missing an entry for: ${this.elements[i]} ${this.elements[j]} ${this.elements[k]}`);
          }
          this.elem3param[(i * nel + j) * nel + k] = n;
        }
      }
    }
    for (const p of this.params) {
      p.cut = p.cutoffS;
      p.cutsq = p.cut * p.cut;
    }
    this.cutmax = 0.0;
    for (const p of this.params) if (p.cut > this.cutmax) this.cutmax = p.cut;
  }

  /** PairTersoffTable::allocateGrids: the exponential, angular, cutoff and zeta-power tables. */
  protected allocateGrids(): void {
    const nel = this.elements.length;
    const P = this.params;
    let minMu = P[0].lam2;
    let maxLambda = P[0].lam1;
    for (let i = 1; i < P.length; i++) {
      if (P[i].lam2 < minMu) minMu = P[i].lam2;
      if (P[i].lam1 > maxLambda) maxLambda = P[i].lam1;
    }
    const maxCutoff = this.cutmax;
    this.minArgumentExponential = minMu * GRIDSTART;
    const numExp = Math.trunc((maxLambda * maxCutoff - this.minArgumentExponential) * GRIDDENSITY_EXP) + 2;
    this.exponential = new Float64Array(numExp);
    let r = this.minArgumentExponential;
    const deltaExp = 1.0 / GRIDDENSITY_EXP;
    for (let i = 0; i < numExp; i++) {
      this.exponential[i] = libmExp(-r);
      r += deltaExp;
    }

    // angular function g(theta) of each element, from the (i,i,i) parameters
    this.numGteta = Math.trunc(2.0 * GRIDDENSITY_GTETA) + 2;
    this.gteta = new Float64Array(nel * this.numGteta);
    this.gtetaD = new Float64Array(nel * this.numGteta);
    for (let i = 0; i < nel; i++) {
      r = -1.0;
      const deltaG = 1.0 / GRIDDENSITY_GTETA;
      const p = P[this.elem3param[(i * nel + i) * nel + i]];
      const c = p.c, d = p.d, h = p.h;
      for (let j = 0; j < this.numGteta; j++) {
        this.gteta[i * this.numGteta + j] = 1.0 + (c * c) / (d * d) - (c * c) / (d * d + (h - r) * (h - r));
        this.gtetaD[i * this.numGteta + j] = -2.0 * c * c * (h - r) /
          ((d * d + (h - r) * (h - r)) * (d * d + (h - r) * (h - r)));
        r += deltaG;
      }
    }

    // cutoff function grids of the type pairs (the (i,j,j) cutoffs), with the common width ngridMax
    let ngridMax = -1;
    for (let i = 0; i < nel; i++) {
      for (let j = 0; j < nel; j++) {
        for (let k = 0; k < nel; k++) {
          const p = P[this.elem3param[(i * nel + j) * nel + k]];
          const nOne = Math.trunc((p.cutoffR - GRIDSTART) * GRIDDENSITY_FCUTOFF) + 1;
          const nNot = Math.trunc((p.cutoffS - p.cutoffR) * GRIDDENSITY_FCUTOFF) + 2;
          ngridMax = Math.max(ngridMax, nOne + nNot);
        }
      }
    }
    this.ngridMax = ngridMax;
    this.cutoffFunction = new Float64Array(nel * nel * ngridMax);
    this.cutoffFunctionD = new Float64Array(nel * nel * ngridMax);
    for (let i = 0; i < nel; i++) {
      for (let j = 0; j < nel; j++) {
        const p = P[this.elem3param[(i * nel + j) * nel + j]];
        const cutoffR = p.cutoffR, cutoffS = p.cutoffS;
        const nOne = Math.trunc((cutoffR - GRIDSTART) * GRIDDENSITY_FCUTOFF) + 1;
        const nNot = Math.trunc((cutoffS - cutoffR) * GRIDDENSITY_FCUTOFF) + 2;
        const nTot = nOne + nNot;
        const base = (i * nel + j) * ngridMax;
        let rr = GRIDSTART;
        const deltaF = 1.0 / GRIDDENSITY_FCUTOFF;
        for (let l = 0; l < nOne; l++) {
          this.cutoffFunction[base + l] = 1.0;
          this.cutoffFunctionD[base + l] = 0.0;
          rr += deltaF;
        }
        for (let l = nOne; l < nTot; l++) {
          this.cutoffFunction[base + l] = 0.5 + 0.5 * libmCos(MY_PI * (rr - cutoffR) / (cutoffS - cutoffR));
          this.cutoffFunctionD[base + l] = -0.5 * MY_PI * libmSin(MY_PI * (rr - cutoffR) / (cutoffS - cutoffR)) /
            (cutoffS - cutoffR);
          rr += deltaF;
        }
      }
    }

    // zeta power (beta zeta)^-n grid: beta^n zeta^n of each element, derivatives alongside
    let zetaMax = -1;
    const nZeta = new Int32Array(nel);
    for (let i = 0; i < nel; i++) {
      const p = P[this.elem3param[(i * nel + i) * nel + i]];
      const c = p.c, d = p.d, beta = p.beta;
      nZeta[i] = Math.trunc((1.0 + (c * c) / (d * d) - (c * c) / (d * d + 4)) * beta * LEADING * GRIDDENSITY_BIJ) + 2;
      zetaMax = Math.max(zetaMax, nZeta[i]);
    }
    this.zetaMax = zetaMax;
    this.betaZetaPower = new Float64Array(nel * zetaMax);
    this.betaZetaPowerD = new Float64Array(nel * zetaMax);
    for (let i = 0; i < nel; i++) {
      const p = P[this.elem3param[(i * nel + i) * nel + i]];
      const powern = p.powern;
      const deltaZ = 1.0 / GRIDDENSITY_BIJ;
      const base = i * zetaMax;
      let rz = 0.0;
      this.betaZetaPower[base] = 1.0;
      rz += deltaZ;
      for (let j = 1; j < nZeta[i]; j++) {
        this.betaZetaPower[base + j] = libmPow(1 + libmPow(rz, powern), -1 / (2 * powern));
        this.betaZetaPowerD[base + j] = -0.5 * libmPow(rz, powern - 1.0) *
          libmPow(1 + libmPow(rz, powern), -1 / (2 * powern) - 1);
        rz += deltaZ;
      }
      this.betaZetaPowerD[base] = (this.betaZetaPower[base + 1] - 1.0) * GRIDDENSITY_BIJ;
    }
  }

  /** PairTersoffTable::compute: the pre-computed neighbor values, then the bond terms and their three-body forces. */
  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const x = pc.x, f = pc.f, type = pc.type;
    const nel = this.elements.length;
    const map = this.map, e3 = this.elem3param, P = this.params;
    const ng = this.numGteta, ngr = this.ngridMax, zm = this.zetaMax;
    const ex = this.exponential, minArg = this.minArgumentExponential;
    const gt = this.gteta, gtD = this.gtetaD;
    const cf = this.cutoffFunction, cfD = this.cutoffFunctionD;
    const bzp = this.betaZetaPower, bzpD = this.betaZetaPowerD;
    const preCut = this.preCut, preCutD = this.preCutD, preG = this.preG, preGD = this.preGD;
    const dcij = new Float64Array(3), dcik = new Float64Array(3);
    const drij = new Float64Array(3), drik = new Float64Array(3);
    const fij = new Float64Array(3), fik = new Float64Array(3);

    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const itype = map[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      let fxtmp = 0, fytmp = 0, fztmp = 0;
      const k0 = list.firstneigh[i];
      const jnum = list.numneigh[i];
      if (jnum > LEADING) {
        throw new StyleError(`Too many neighbors for interaction list: ${jnum} vs ${LEADING}.\n` +
          "Check your system or increase 'leadingDimensionInteractionList'");
      }

      // pre-computed cutoff functions of the neighbors and the angular functions of the pairs
      for (let nj = 0; nj < jnum; nj++) {
        const j = list.neighbors[k0 + nj] & NEIGHMASK;
        const dxij = xtmp - x[3 * j], dyij = ytmp - x[3 * j + 1], dzij = ztmp - x[3 * j + 2];
        let rij2 = dxij * dxij + dyij * dyij + dzij * dzij;
        const jtype = map[type[j]];
        const ijparam = e3[(itype * nel + jtype) * nel + jtype];
        if (rij2 > P[ijparam].cutsq) continue;
        rij2 = Math.sqrt(rij2);
        const invRij = 1.0 / rij2;
        dcij[0] = invRij * dxij;
        dcij[1] = invRij * dyij;
        dcij[2] = invRij * dzij;
        const tmpF = (rij2 - GRIDSTART) * GRIDDENSITY_FCUTOFF;
        const idxF = Math.trunc(tmpF);
        const cb = (itype * nel + jtype) * ngr + idxF;
        const y1 = cf[cb], y2 = cf[cb + 1];
        preCut[nj] = y1 + (y2 - y1) * (tmpF - idxF);
        const yd1 = cfD[cb], yd2 = cfD[cb + 1];
        preCutD[nj] = yd1 + (yd2 - yd1) * (tmpF - idxF);

        for (let nk = nj + 1; nk < jnum; nk++) {
          const k = list.neighbors[k0 + nk] & NEIGHMASK;
          const ktype = map[type[k]];
          const ikparam = e3[(itype * nel + ktype) * nel + ktype];
          const ijkparam = e3[(itype * nel + jtype) * nel + ktype];
          const dxik = xtmp - x[3 * k], dyik = ytmp - x[3 * k + 1], dzik = ztmp - x[3 * k + 2];
          let rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
          if (rik2 > P[ikparam].cutsq) continue;
          rik2 = Math.sqrt(rik2);
          const invRik = 1.0 / rik2;
          dcik[0] = invRik * dxik;
          dcik[1] = invRik * dyik;
          dcik[2] = invRik * dzik;
          const cosTeta = dcij[0] * dcik[0] + dcij[1] * dcik[1] + dcij[2] * dcik[2];
          const tmpG = (cosTeta + 1.0) * GRIDDENSITY_GTETA;
          const idxG = Math.trunc(tmpG);
          const gb = itype * ng + idxG;
          const g1 = gt[gb], g2 = gt[gb + 1];
          const gTemp = g1 + (g2 - g1) * (tmpG - idxG);
          const gd1 = gtD[gb], gd2 = gtD[gb + 1];
          const gdTemp = gd1 + (gd2 - gd1) * (tmpG - idxG);
          const gam = P[ijkparam].gamma;
          preG[nj * LEADING + nk] = gam * gTemp;
          preGD[nj * LEADING + nk] = gam * gdTemp;
          preG[nk * LEADING + nj] = gam * gTemp;
          preGD[nk * LEADING + nj] = gam * gdTemp;
        }
      }

      // bond terms
      for (let nj = 0; nj < jnum; nj++) {
        const j = list.neighbors[k0 + nj] & NEIGHMASK;
        const dxij = xtmp - x[3 * j], dyij = ytmp - x[3 * j + 1], dzij = ztmp - x[3 * j + 2];
        let rij2 = dxij * dxij + dyij * dyij + dzij * dzij;
        const jtype = map[type[j]];
        const ijparam = e3[(itype * nel + jtype) * nel + jtype];
        const pij = P[ijparam];
        if (rij2 > pij.cutsq) continue;
        rij2 = Math.sqrt(rij2);
        const invRij = 1.0 / rij2;
        dcij[0] = invRij * dxij;
        dcij[1] = invRij * dyij;
        dcij[2] = invRij * dzij;
        drij[0] = dxij; drij[1] = dyij; drij[2] = dzij;

        const expRep = pij.lam1 * rij2;
        const expAtt = pij.lam2 * rij2;
        const tR = (expRep - minArg) * GRIDDENSITY_EXP;
        const iR = Math.trunc(tR);
        const repExpo = ex[iR] + (ex[iR + 1] - ex[iR]) * (tR - iR);
        const tA = (expAtt - minArg) * GRIDDENSITY_EXP;
        const iA = Math.trunc(tA);
        const attExpo = ex[iA] + (ex[iA + 1] - ex[iA]) * (tA - iA);
        const repPot = pij.biga * repExpo;
        const attPot = -pij.bigb * attExpo;
        const cfIJ = preCut[nj];
        const cfDIJ = preCutD[nj];

        // zeta_ij over the neighbors k below j, then above j (native order)
        let zeta = 0.0;
        for (let nk = 0; nk < nj; nk++) {
          const k = list.neighbors[k0 + nk] & NEIGHMASK;
          const ktype = map[type[k]];
          const ikparam = e3[(itype * nel + ktype) * nel + ktype];
          const dxik = xtmp - x[3 * k], dyik = ytmp - x[3 * k + 1], dzik = ztmp - x[3 * k + 2];
          const rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
          if (rik2 > P[ikparam].cutsq) continue;
          zeta += preCut[nk] * preG[nj * LEADING + nk];
        }
        for (let nk = nj + 1; nk < jnum; nk++) {
          const k = list.neighbors[k0 + nk] & NEIGHMASK;
          const ktype = map[type[k]];
          const ikparam = e3[(itype * nel + ktype) * nel + ktype];
          const dxik = xtmp - x[3 * k], dyik = ytmp - x[3 * k + 1], dzik = ztmp - x[3 * k + 2];
          const rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
          if (rik2 > P[ikparam].cutsq) continue;
          zeta += preCut[nk] * preG[nj * LEADING + nk];
        }

        const tZ = (pij.beta * zeta) * GRIDDENSITY_BIJ;
        const iZ = Math.trunc(tZ);
        const zb = itype * zm + iZ;
        const z1 = bzp[zb], z2 = bzp[zb + 1];
        const betaZetaPowerIJK = z1 + (z2 - z1) * (tZ - iZ);
        const zd1 = bzpD[zb], zd2 = bzpD[zb + 1];
        const betaZetaPowerDerivedIJK = pij.beta * (zd1 + (zd2 - zd1) * (tZ - iZ));

        const factorForceIJ = 0.5 * cfDIJ * (repPot + attPot * betaZetaPowerIJK) +
          0.5 * cfIJ * (-repPot * pij.lam1 - betaZetaPowerIJK * attPot * pij.lam2);
        fij[0] = factorForceIJ * dcij[0];
        fij[1] = factorForceIJ * dcij[1];
        fij[2] = factorForceIJ * dcij[2];
        f[3 * j] += fij[0];
        f[3 * j + 1] += fij[1];
        f[3 * j + 2] += fij[2];
        fxtmp -= fij[0];
        fytmp -= fij[1];
        fztmp -= fij[2];

        const evdwl = cfIJ * repPot + cfIJ * attPot * betaZetaPowerIJK;
        tallyPair(pc, i, j, 0.5 * evdwl, -factorForceIJ * invRij, drij[0], drij[1], drij[2]);

        const factorForceTot = 0.5 * cfIJ * attPot * betaZetaPowerDerivedIJK;
        for (let pass = 0; pass < 2; pass++) {
          // pass 0: neighbors k < j; pass 1: neighbors k > j
          const kFrom = pass === 0 ? 0 : nj + 1;
          const kTo = pass === 0 ? nj : jnum;
          for (let nk = kFrom; nk < kTo; nk++) {
            const k = list.neighbors[k0 + nk] & NEIGHMASK;
            const ktype = map[type[k]];
            const ikparam = e3[(itype * nel + ktype) * nel + ktype];
            const dxik = xtmp - x[3 * k], dyik = ytmp - x[3 * k + 1], dzik = ztmp - x[3 * k + 2];
            let rik2 = dxik * dxik + dyik * dyik + dzik * dzik;
            if (rik2 > P[ikparam].cutsq) continue;
            rik2 = Math.sqrt(rik2);
            const invRik = 1.0 / rik2;
            dcik[0] = invRik * dxik;
            dcik[1] = invRik * dyik;
            dcik[2] = invRik * dzik;
            drik[0] = dxik; drik[1] = dyik; drik[2] = dzik;
            const cosTeta = dcij[0] * dcik[0] + dcij[1] * dcik[1] + dcij[2] * dcik[2];
            const gIJK = preG[nj * LEADING + nk];
            const gDIJK = preGD[nj * LEADING + nk];
            const cfIK = preCut[nk];
            const cfDIK = preCutD[nk];
            const factorForce3IJ = cfIK * gDIJK * invRij * factorForceTot;
            for (let c = 0; c < 3; c++) fij[c] = factorForce3IJ * (dcij[c] * cosTeta - dcik[c]);
            const factor1Force3IK = (cfIK * gDIJK * invRik) * factorForceTot;
            const factor2Force3IK = -(cfDIK * gIJK) * factorForceTot;
            for (let c = 0; c < 3; c++) {
              fik[c] = factor1Force3IK * (dcik[c] * cosTeta - dcij[c]) + factor2Force3IK * dcik[c];
            }
            f[3 * j] -= fij[0];
            f[3 * j + 1] -= fij[1];
            f[3 * j + 2] -= fij[2];
            f[3 * k] -= fik[0];
            f[3 * k + 1] -= fik[1];
            f[3 * k + 2] -= fik[2];
            fxtmp += fij[0] + fik[0];
            fytmp += fij[1] + fik[1];
            fztmp += fij[2] + fik[2];
            tallyTriplet(pc, i, j, k, fij, fik, drij, drik);
          }
        }
      }
      f[3 * i] += fxtmp;
      f[3 * i + 1] += fytmp;
      f[3 * i + 2] += fztmp;
    }
  }
}
