/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair_style gw and gw/zbl
 * (src/MANYBODY/pair_gw.cpp, pair_gw.h, pair_gw_zbl.cpp, pair_gw_zbl.h). Contributing author: German Samolyuk (ORNL),
 * based on PairTersoff by Aidan Thompson (SNL). Copyright Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_gw.html. "pair_style gw" takes no arguments; "pair_coeff * * file elem1 ... elemN".
 * gw/zbl takes no arguments either and needs metal or real units (its ZBL constants depend on them).
 *
 * The loops are those of PairGW::compute over the full neighbor list (atom IDs decide which of two atoms owns a
 * two-body term), with the same operation order; forces on ghost atoms are folded into their owners by the force field.
 */
import { Pair, StyleError, type PairCompute, type PairConstructorWhere, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { sinFma as libmSin, cosFma as libmCos } from '../libm/sincos_fma';
import {
  checkOneCoeffArgs, mapElementsToTypes, nextDouble, PotentialEntryReader, potentialUnitFactor, tagOf,
  tallyPair, tallyTriplet,
} from './common';

// LAMMPS MathConst (src/math_const.h): MY_PI, MY_PI2 = pi/2, MY_PI4 = pi/4 (exact halvings of the double pi).
const MY_PI = 3.14159265358979323846;
const MY_PI2 = 1.57079632679489661923;
const MY_PI4 = 0.78539816339744830962;

/** One element triplet of the potential file (PairGW::Param). */
export interface GWParam {
  lam1: number; lam2: number; lam3: number;
  c: number; d: number; h: number;
  gamma: number; powerm: number;
  powern: number; beta: number;
  biga: number; bigb: number; bigd: number; bigr: number;
  cut: number; cutsq: number;
  c1: number; c2: number; c3: number; c4: number;
  ielement: number; jelement: number; kelement: number;
  powermint: number;
  /** gw/zbl only (PairGWZBL fields). */
  Z_i: number; Z_j: number; ZBLcut: number; ZBLexpscale: number;
}

const NPARAMS_PER_LINE = 17;
const NPARAMS_PER_LINE_ZBL = 21;
/** |arg| bound of the exponential in zeta (pair_gw.cpp: "if (arg > 69.0776) ex_delr = 1.e30"). */
const EXP_LIMIT = 69.0776;

const newParam = (): GWParam => ({
  lam1: 0, lam2: 0, lam3: 0, c: 0, d: 0, h: 0, gamma: 0, powerm: 0, powern: 0, beta: 0,
  biga: 0, bigb: 0, bigd: 0, bigr: 0, cut: 0, cutsq: 0, c1: 0, c2: 0, c3: 0, c4: 0,
  ielement: 0, jelement: 0, kelement: 0, powermint: 0, Z_i: 0, Z_j: 0, ZBLcut: 0, ZBLexpscale: 0,
});

export class PairGW extends Pair {
  override readonly name: string = 'gw';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  protected fileRead = false;
  protected elements: string[] = [];
  protected map: Int32Array = new Int32Array(0);
  protected params: GWParam[] = [];
  /** elem3param[(i*nel + j)*nel + k] = parameter index (PairGW::setup_params). */
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  /** Energy unit factor of the potential file (1 for the same units). */
  protected unitFactor = 1;

  // scratch of the current evaluation (the native locals of the helpers)
  protected ff = 0;
  protected pref = 0;
  protected ee = 0;
  protected readonly fi = new Float64Array(3);
  protected readonly fj = new Float64Array(3);
  protected readonly fk = new Float64Array(3);
  protected readonly d1 = new Float64Array(3);
  protected readonly d2 = new Float64Array(3);

  /** The potential file's word count per entry (17 for gw, 21 for gw/zbl). */
  protected get nparamsPerLine(): number { return NPARAMS_PER_LINE; }
  protected get fileType(): string { return 'gw'; }

  override settings(args: string[], _ctx?: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    checkOneCoeffArgs(args, this.ntypes, this.name);
    const { elements, map } = mapElementsToTypes(args.slice(3), this.ntypes);
    this.elements = elements;
    this.map = map;
    const fileName = args[2];
    this.readFile(fileName, ctx);
    this.setupParams();
    this.fileRead = true;
  }

  /** PairGW::read_file: entries whose three elements are all mapped, parameters checked (native text). */
  protected readFile(fileName: string, ctx: StyleContext): void {
    const text = ctx.readFile(fileName, potentialFile(this.fileType));
    this.unitFactor = potentialUnitFactor(text, this.fileType, fileName, ctx);
    const reader = new PotentialEntryReader(text.split(/\r?\n/), this.fileType);
    const params: GWParam[] = [];
    for (;;) {
      const w = reader.next(this.nparamsPerLine);
      if (!w) break;
      const ielement = this.elements.indexOf(w[0]);
      if (ielement < 0) continue;
      const jelement = this.elements.indexOf(w[1]);
      if (jelement < 0) continue;
      const kelement = this.elements.indexOf(w[2]);
      if (kelement < 0) continue;
      const p = newParam();
      p.ielement = ielement; p.jelement = jelement; p.kelement = kelement;
      this.readValues(p, w);
      if (this.unitFactor !== 1) {
        p.biga *= this.unitFactor;
        p.bigb *= this.unitFactor;
      }
      if (this.illegal(p)) throw new StyleError('Illegal GW parameter');
      params.push(p);
    }
    this.params = params;
  }

  /** The numeric fields after the three element names (ValueTokenizer order of PairGW::read_file). */
  protected readValues(p: GWParam, w: string[]): void {
    const v = (k: number) => nextDouble(w[3 + k]);
    p.powerm = v(0); p.gamma = v(1); p.lam3 = v(2); p.c = v(3); p.d = v(4); p.h = v(5); p.powern = v(6);
    p.beta = v(7); p.lam2 = v(8); p.bigb = v(9); p.bigr = v(10); p.bigd = v(11); p.lam1 = v(12); p.biga = v(13);
    p.powermint = Math.trunc(p.powerm);
  }

  /** The range checks of PairGW::read_file ("currently only allow m exponent of 1 or 3"). */
  protected illegal(p: GWParam): boolean {
    return p.c < 0.0 || p.d < 0.0 || p.powern < 0.0 || p.beta < 0.0 || p.lam2 < 0.0 || p.bigb < 0.0 ||
      p.bigr < 0.0 || p.bigd < 0.0 || p.bigd > p.bigr || p.lam1 < 0.0 || p.biga < 0.0 ||
      p.powerm - p.powermint !== 0.0 || (p.powermint !== 3 && p.powermint !== 1) || p.gamma < 0.0;
  }

  /** PairGW::setup_params: the element triplet table, the derived cutoffs and cutmax. */
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
      p.cut = p.bigr + p.bigd;
      p.cutsq = p.cut * p.cut;
      p.c1 = libmPow(2.0 * p.powern * 1.0e-16, -1.0 / p.powern);
      p.c2 = libmPow(2.0 * p.powern * 1.0e-8, -1.0 / p.powern);
      p.c3 = 1.0 / p.c2;
      p.c4 = 1.0 / p.c1;
    }
    this.cutmax = 0.0;
    for (const p of this.params) if (p.cut > this.cutmax) this.cutmax = p.cut;
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError(`Pair style ${this.name} requires newton pair on`);
  }

  override initOne(i: number, j: number): number {
    if (this.map.length === 0 || this.map[i] < 0 || this.map[j] < 0 || !this.fileRead) {
      throw new StyleError('All pair coeffs are not set');
    }
    return this.cutmax;
  }

  /** PairGW::repulsive: sets this.ff (force over r) and this.ee (energy). */
  protected repulsive(p: GWParam, rsq: number): void {
    const r = Math.sqrt(rsq);
    const fc = this.gwFc(r, p);
    const fcd = this.gwFcD(r, p);
    const expo = libmExp(-p.lam1 * r);
    this.ff = -p.biga * expo * (fcd - fc * p.lam1) / r;
    this.ee = fc * p.biga * expo;
  }

  /** PairGW::zeta: f_C(r_ik) g(theta) exp[lam3^m (r_ij - r_ik)^m]. */
  protected zeta(p: GWParam, rsqij: number, rsqik: number, delrij: Float64Array, delrik: Float64Array): number {
    const rij = Math.sqrt(rsqij);
    const rik = Math.sqrt(rsqik);
    const costheta = (delrij[0] * delrik[0] + delrij[1] * delrik[1] + delrij[2] * delrik[2]) / (rij * rik);
    let arg: number;
    if (p.powermint === 3) arg = libmPow(p.lam3 * (rij - rik), 3.0);
    else arg = p.lam3 * (rij - rik);
    let exDelr: number;
    if (arg > EXP_LIMIT) exDelr = 1.e30;
    else if (arg < -EXP_LIMIT) exDelr = 0.0;
    else exDelr = libmExp(arg);
    return this.gwFc(rik, p) * this.gwGijk(costheta, p) * exDelr;
  }

  /** PairGW::force_zeta: sets this.ff, this.pref (prefactor of the attractive term) and this.ee. */
  protected forceZeta(p: GWParam, rsq: number, zetaIJ: number): void {
    const r = Math.sqrt(rsq);
    const fa = this.gwFa(r, p);
    const faD = this.gwFaD(r, p);
    const bij = this.gwBij(zetaIJ, p);
    this.ff = 0.5 * bij * faD / r;
    this.pref = -0.5 * fa * this.gwBijD(zetaIJ, p);
    this.ee = 0.5 * bij * fa;
  }

  /** PairGW::attractive: fills this.fi, this.fj, this.fk from the separations d1 (i->j) and d2 (i->k). */
  protected attractive(p: GWParam, prefactor: number, rsqij: number, rsqik: number): void {
    const rij = Math.sqrt(rsqij);
    const rijinv = 1.0 / rij;
    const rik = Math.sqrt(rsqik);
    const rikinv = 1.0 / rik;
    const rijHat = [rijinv * this.d1[0], rijinv * this.d1[1], rijinv * this.d1[2]];
    const rikHat = [rikinv * this.d2[0], rikinv * this.d2[1], rikinv * this.d2[2]];
    this.zetatermD(prefactor, rijHat, rij, rikHat, rik, p);
  }

  /** PairGW::gw_zetaterm_d: forces on i, j, k of the attractive term (into this.fi, this.fj, this.fk). */
  protected zetatermD(prefactor: number, rijHat: number[], rij: number, rikHat: number[], rik: number, p: GWParam): void {
    const fc = this.gwFc(rik, p);
    const dfc = this.gwFcD(rik, p);
    let tmp: number;
    if (p.powermint === 3) tmp = libmPow(p.lam3 * (rij - rik), 3.0);
    else tmp = p.lam3 * (rij - rik);
    let exDelr: number;
    if (tmp > EXP_LIMIT) exDelr = 1.e30;
    else if (tmp < -EXP_LIMIT) exDelr = 0.0;
    else exDelr = libmExp(tmp);
    let exDelrD: number;
    // pair_gw.cpp: pow(rij-rik,2.0) is (rij-rik)*(rij-rik) in the native binary (gcc folds a literal 2.0); pow(lam3,3.0) is a call
    if (p.powermint === 3) exDelrD = 3.0 * libmPow(p.lam3, 3.0) * ((rij - rik) * (rij - rik)) * exDelr;
    else exDelrD = p.lam3 * exDelr;
    const cosTheta = rijHat[0] * rikHat[0] + rijHat[1] * rikHat[1] + rijHat[2] * rikHat[2];
    const gijk = this.gwGijk(cosTheta, p);
    const gijkD = this.gwGijkD(cosTheta, p);
    const { dri: dcosdri, drj: dcosdrj, drk: dcosdrk } = costhetaD(rijHat, rij, rikHat, rik, cosTheta);

    // derivative with respect to Ri
    const dri = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      let v = (-dfc * gijk * exDelr) * rikHat[c];
      v = (fc * gijkD * exDelr) * dcosdri[c] + v;
      v = (fc * gijk * exDelrD) * rikHat[c] + v;
      v = (-fc * gijk * exDelrD) * rijHat[c] + v;
      dri[c] = prefactor * v;
    }
    // derivative with respect to Rj
    const drj = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      let v = (fc * gijkD * exDelr) * dcosdrj[c];
      v = (fc * gijk * exDelrD) * rijHat[c] + v;
      drj[c] = prefactor * v;
    }
    // derivative with respect to Rk
    const drk = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      let v = (dfc * gijk * exDelr) * rikHat[c];
      v = (fc * gijkD * exDelr) * dcosdrk[c] + v;
      v = (-fc * gijk * exDelrD) * rikHat[c] + v;
      drk[c] = prefactor * v;
    }
    for (let c = 0; c < 3; c++) {
      this.fi[c] = dri[c];
      this.fj[c] = drj[c];
      this.fk[c] = drk[c];
    }
  }

  /** PairGW::gw_fc: the cutoff function, 1 below R-D, 0 above R+D, a sine ramp between. */
  protected gwFc(r: number, p: GWParam): number {
    const R = p.bigr, D = p.bigd;
    if (r < R - D) return 1.0;
    if (r > R + D) return 0.0;
    return 0.5 * (1.0 - libmSin(MY_PI2 * (r - R) / D));
  }

  protected gwFcD(r: number, p: GWParam): number {
    const R = p.bigr, D = p.bigd;
    if (r < R - D) return 0.0;
    if (r > R + D) return 0.0;
    return -(MY_PI4 / D) * libmCos(MY_PI2 * (r - R) / D);
  }

  protected gwFa(r: number, p: GWParam): number {
    if (r > p.bigr + p.bigd) return 0.0;
    return -p.bigb * libmExp(-p.lam2 * r) * this.gwFc(r, p);
  }

  protected gwFaD(r: number, p: GWParam): number {
    if (r > p.bigr + p.bigd) return 0.0;
    return p.bigb * libmExp(-p.lam2 * r) * (p.lam2 * this.gwFc(r, p) - this.gwFcD(r, p));
  }

  protected gwGijk(costheta: number, p: GWParam): number {
    const c = p.c * p.c;
    const d = p.d * p.d;
    const hcth = p.h - costheta;
    return p.gamma * (1.0 + c / d - c / (d + hcth * hcth));
  }

  protected gwGijkD(costheta: number, p: GWParam): number {
    const c = p.c * p.c;
    const d = p.d * p.d;
    const hcth = p.h - costheta;
    const numerator = -2.0 * c * hcth;
    const denominator = 1.0 / (d + hcth * hcth);
    return p.gamma * numerator * denominator * denominator;
  }

  protected gwBij(zetaIJ: number, p: GWParam): number {
    const tmp = p.beta * zetaIJ;
    return libmPow(tmp, -p.powern);
  }

  protected gwBijD(zetaIJ: number, p: GWParam): number {
    const tmp = p.beta * zetaIJ;
    return -p.powern * libmPow(tmp, -p.powern - 1) * tmp / zetaIJ;
  }

  /** PairGW::compute: two-body terms on half the pairs (by atom IDs), then the three-body terms over the full list. */
  override compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new StyleError(`pair style ${this.name} needs a full neighbor list`);
    const x = pc.x, f = pc.f, type = pc.type;
    const nel = this.elements.length;
    const map = this.map, e3 = this.elem3param, P = this.params;
    const d1 = this.d1, d2 = this.d2;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const itype = map[type[i]];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itag = pc.s.id[i];
      const k0 = list.firstneigh[i];
      const jnum = list.numneigh[i];

      // two-body interactions, skipping half of them
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const jtag = tagOf(pc, j);
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
        const pij = P[e3[(itype * nel + jtype) * nel + jtype]];
        if (rsq > pij.cutsq) continue;
        this.repulsive(pij, rsq);
        const fpair = this.ff;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        tallyPair(pc, i, j, this.ee, fpair, delx, dely, delz);
      }

      // three-body interactions
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const jtype = map[type[j]];
        const pij = P[e3[(itype * nel + jtype) * nel + jtype]];
        d1[0] = x[3 * j] - xtmp;
        d1[1] = x[3 * j + 1] - ytmp;
        d1[2] = x[3 * j + 2] - ztmp;
        const rsq1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
        if (rsq1 > pij.cutsq) continue;

        // bond order zeta_ij from every other neighbor k within the cutoff
        let zetaIJ = 1.0;
        for (let kk = 0; kk < jnum; kk++) {
          if (jj === kk) continue;
          const k = list.neighbors[k0 + kk] & NEIGHMASK;
          const ktype = map[type[k]];
          const pijk = P[e3[(itype * nel + jtype) * nel + ktype]];
          d2[0] = x[3 * k] - xtmp;
          d2[1] = x[3 * k + 1] - ytmp;
          d2[2] = x[3 * k + 2] - ztmp;
          const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
          if (rsq2 > pijk.cutsq) continue;
          zetaIJ += this.zeta(pijk, rsq1, rsq2, d1, d2);
        }

        // pairwise force due to zeta
        this.forceZeta(pij, rsq1, zetaIJ);
        const fpair = this.ff;
        const prefactor = this.pref;
        f[3 * i] += d1[0] * fpair;
        f[3 * i + 1] += d1[1] * fpair;
        f[3 * i + 2] += d1[2] * fpair;
        f[3 * j] -= d1[0] * fpair;
        f[3 * j + 1] -= d1[1] * fpair;
        f[3 * j + 2] -= d1[2] * fpair;
        tallyPair(pc, i, j, this.ee, -fpair, -d1[0], -d1[1], -d1[2]);

        // attractive term via the loop over k
        for (let kk = 0; kk < jnum; kk++) {
          if (jj === kk) continue;
          const k = list.neighbors[k0 + kk] & NEIGHMASK;
          const ktype = map[type[k]];
          const pijk = P[e3[(itype * nel + jtype) * nel + ktype]];
          d2[0] = x[3 * k] - xtmp;
          d2[1] = x[3 * k + 1] - ytmp;
          d2[2] = x[3 * k + 2] - ztmp;
          const rsq2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
          if (rsq2 > pijk.cutsq) continue;
          this.attractive(pijk, prefactor, rsq1, rsq2);
          const fi = this.fi, fj = this.fj, fk = this.fk;
          for (let c = 0; c < 3; c++) {
            f[3 * i + c] += fi[c];
            f[3 * j + c] += fj[c];
            f[3 * k + c] += fk[c];
          }
          tallyTriplet(pc, i, j, k, fj, fk, d1, d2);
        }
      }
    }
  }
}

/**
 * PairGW::costheta_d: derivatives of cos(theta) with respect to Ri (dri), Rj (drj) and Rk (drk) from the unit vectors
 * rij_hat and rik_hat and the distances rij and rik.
 */
const costhetaD = (rijHat: number[], rij: number, rikHat: number[], rik: number, cosTheta: number) => {
  const drj = [0, 0, 0], drk = [0, 0, 0], dri = [0, 0, 0];
  for (let c = 0; c < 3; c++) drj[c] = -cosTheta * rijHat[c] + rikHat[c];
  for (let c = 0; c < 3; c++) drj[c] = (1.0 / rij) * drj[c];
  for (let c = 0; c < 3; c++) drk[c] = -cosTheta * rikHat[c] + rijHat[c];
  for (let c = 0; c < 3; c++) drk[c] = (1.0 / rik) * drk[c];
  for (let c = 0; c < 3; c++) dri[c] = -(drj[c] + drk[c]);
  return { dri, drj, drk };
};

/**
 * pair_style gw/zbl (src/MANYBODY/pair_gw_zbl.cpp): the Gao-Weber terms smoothed into a ZBL core by a Fermi function.
 * The constants of the ZBL screening depend on the unit style (PairGWZBL constructor).
 */
export class PairGWZBL extends PairGW {
  override readonly name: string = 'gw/zbl';
  private globalA0 = 0.529;
  private globalEpsilon0 = 0.00552635;
  private globalE = 1.0;

  protected override get nparamsPerLine(): number { return NPARAMS_PER_LINE_ZBL; }
  protected override get fileType(): string { return 'gw/zbl'; }

  /** The constructor's constants per unit style (metal or real only). */
  override constructorCheck(where: PairConstructorWhere): void {
    const style = where.units;
    if (style === 'metal') {
      this.globalA0 = 0.529;
      this.globalEpsilon0 = 0.00552635;
      this.globalE = 1.0;
    } else if (style === 'real') {
      this.globalA0 = 0.529;
      this.globalEpsilon0 = 0.00552635 * 0.043365121;
      this.globalE = 1.0;
    } else {
      throw new StyleError('Pair gw/zbl requires metal or real units');
    }
  }

  protected override readValues(p: GWParam, w: string[]): void {
    super.readValues(p, w);
    p.Z_i = nextDouble(w[17]);
    p.Z_j = nextDouble(w[18]);
    p.ZBLcut = nextDouble(w[19]);
    p.ZBLexpscale = nextDouble(w[20]);
  }

  protected override illegal(p: GWParam): boolean {
    return p.lam3 < 0.0 || p.c < 0.0 || p.d < 0.0 || p.powern < 0.0 || p.beta < 0.0 || p.lam2 < 0.0 ||
      p.bigb < 0.0 || p.bigr < 0.0 || p.bigd < 0.0 || p.bigd > p.bigr || p.lam3 < 0.0 || p.biga < 0.0 ||
      p.powerm - p.powermint !== 0.0 || (p.powermint !== 3 && p.powermint !== 1) || p.gamma < 0.0 ||
      p.Z_i < 1.0 || p.Z_j < 1.0 || p.ZBLcut < 0.0 || p.ZBLexpscale < 0.0;
  }

  /** Fermi-like smoothing function. */
  protected fermi(r: number, p: GWParam): number {
    return 1.0 / (1.0 + libmExp(-p.ZBLexpscale * (r - p.ZBLcut)));
  }

  /** Derivative of the Fermi-like smoothing function with respect to r. */
  protected fermiD(r: number, p: GWParam): number {
    return p.ZBLexpscale * libmExp(-p.ZBLexpscale * (r - p.ZBLcut)) /
      libmPow(1.0 + libmExp(-p.ZBLexpscale * (r - p.ZBLcut)), 2.0);
  }

  /** PairGWZBL::repulsive: the Gao-Weber repulsion blended with the ZBL screened Coulomb term. */
  protected override repulsive(p: GWParam, rsq: number): void {
    const r = Math.sqrt(rsq);
    const fc = this.gwFc(r, p);
    const fcd = this.gwFcD(r, p);
    const expo = libmExp(-p.lam1 * r);
    const fforceGw = p.biga * expo * (fcd - fc * p.lam1);
    const engGw = fc * p.biga * expo;
    // pair_gw_zbl.cpp: pow(global_e,2.0) is global_e*global_e in the native binary (gcc folds a literal 2.0)
    const esq = this.globalE * this.globalE;
    const aIJ = (0.8854 * this.globalA0) / (libmPow(p.Z_i, 0.23) + libmPow(p.Z_j, 0.23));
    const premult = (p.Z_i * p.Z_j * esq) / (4.0 * MY_PI * this.globalEpsilon0);
    const rOvA = r / aIJ;
    const phi = 0.1818 * libmExp(-3.2 * rOvA) + 0.5099 * libmExp(-0.9423 * rOvA) +
      0.2802 * libmExp(-0.4029 * rOvA) + 0.02817 * libmExp(-0.2016 * rOvA);
    const dphi = (1.0 / aIJ) * (-3.2 * 0.1818 * libmExp(-3.2 * rOvA) -
      0.9423 * 0.5099 * libmExp(-0.9423 * rOvA) -
      0.4029 * 0.2802 * libmExp(-0.4029 * rOvA) -
      0.2016 * 0.02817 * libmExp(-0.2016 * rOvA));
    const fforceZBL = premult * -phi / rsq + premult * dphi / r;
    const engZBL = premult * (1.0 / r) * phi;
    const F = this.fermi(r, p), Fd = this.fermiD(r, p);
    this.ff = -(-Fd * engZBL + (1.0 - F) * fforceZBL + Fd * engGw + F * fforceGw) / r;
    this.ee = (1.0 - F) * engZBL + F * engGw;
  }

  protected override gwFa(r: number, p: GWParam): number {
    if (r > p.bigr + p.bigd) return 0.0;
    return -p.bigb * libmExp(-p.lam2 * r) * this.gwFc(r, p) * this.fermi(r, p);
  }

  protected override gwFaD(r: number, p: GWParam): number {
    if (r > p.bigr + p.bigd) return 0.0;
    const F = this.fermi(r, p), Fd = this.fermiD(r, p);
    return p.bigb * libmExp(-p.lam2 * r) *
      (p.lam2 * this.gwFc(r, p) * F - this.gwFcD(r, p) * F - this.gwFc(r, p) * Fd);
  }
}
