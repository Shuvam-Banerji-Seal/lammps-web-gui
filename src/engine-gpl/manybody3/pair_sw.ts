/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: pair_style sw
 * (src/MANYBODY/pair_sw.cpp and pair_sw.h; contributing author Aidan Thompson (SNL), optimizations for two-body only by
 * Jackson Elowitt (Univ. of Utah); Copyright Sandia Corporation, GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Docs: docs.lammps.org/pair_sw.html. "pair_style sw" takes the optional keyword "threebody on|off" (default on);
 * "pair_coeff * * file elem1 ... elemN" with threebody on, and one or more pair_coeff commands with threebody off.
 * The loops are those of PairSW::compute, with the same operation order. Two-body terms of a pair are taken from the
 * entry (i, j, j) of the atom whose neighbor list visits it, decided by atom IDs on the full list (threebody on) or
 * by the half list (threebody off), as native LAMMPS does.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { NativeFullOrder } from '../neighbor/nativeOrder';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { log as libmLog } from '../libm/log';
import { nextDouble, PotentialEntryReader, potentialUnitFactor, tallyPair, tallyTriplet } from '../manybody2/common';

/** PairSW::NPARAMS_PER_LINE (pair_sw.h): the element names and the 11 numbers of an entry. */
const NPARAMS_PER_LINE = 14;
/** LAMMPS MathConst THIRD = 1.0/3.0 (src/math_const.h). */
const THIRD = 1.0 / 3.0;

/** One element triplet of the potential file (PairSW::Param). */
export interface SWParam {
  ielement: number; jelement: number; kelement: number;
  epsilon: number; sigma: number; littlea: number; lambda: number; gamma: number; costheta: number;
  biga: number; bigb: number; powerp: number; powerq: number; tol: number;
  cut: number; cutsq: number; sigma_gamma: number; lambda_epsilon: number; lambda_epsilon2: number;
  c1: number; c2: number; c3: number; c4: number; c5: number; c6: number;
}

const newParam = (): SWParam => ({
  ielement: 0, jelement: 0, kelement: 0, epsilon: 0, sigma: 0, littlea: 0, lambda: 0, gamma: 0, costheta: 0,
  biga: 0, bigb: 0, powerp: 0, powerq: 0, tol: 0, cut: 0, cutsq: 0, sigma_gamma: 0, lambda_epsilon: 0,
  lambda_epsilon2: 0, c1: 0, c2: 0, c3: 0, c4: 0, c5: 0, c6: 0,
});

export class PairSW extends Pair {
  override readonly name: string = 'sw';
  manybody = true;
  virialFdotr = true;
  /** threebody on: the full list (every pair seen from both ends); off: the half list. */
  needsFull = true;
  needsHalf = false;

  /** The threebody keyword (native skip_threebody_flag is its negation). */
  protected threebody = true;
  /** PairSW::params_mapped: the potential file is read and the element maps are set. */
  protected paramsMapped = false;
  /** Native setflag[i][j] (threebody off: only the pairs of pair_coeff commands; j >= i). */
  protected setflag = new Uint8Array(0);
  protected elements: string[] = [];
  protected map: Int32Array = new Int32Array(0);
  protected params: SWParam[] = [];
  /** elem3param[(i*nel + j)*nel + k] = parameter index (PairSW::setup_params). */
  protected elem3param = new Int32Array(0);
  protected cutmax = 0;
  /** params[elem3param[i][j][j]].cutsq per element pair (i, j), i * nel + j */
  private pairCutsq = new Float64Array(0);
  protected unitFactor = 1;
  protected fileName = '';

  /**
   * Native full-list neighbour order (threebody on): the triplets take the parameters of (i, j, k) with j before k
   * in native's full list (npair_bin.cpp build, HALF = 0), so the order decides which element triplet an unlike
   * triplet uses. Kept between neighbor builds; triclinic boxes keep the engine's list order.
   */
  private readonly natOrder = new NativeFullOrder();
  // per-atom scratch: the short neighbor list of PairSW::compute (neighshort)
  private neighshort = new Int32Array(10);
  /** per short neighbour of the current atom: the leg values of PairSW::threebody (legValues), 8 per neighbour */
  private leg = new Float64Array(80);
  private readonly fj = new Float64Array(3);
  private readonly fk = new Float64Array(3);
  private readonly d1 = new Float64Array(3);
  private readonly d2 = new Float64Array(3);
  /** Two-body result of twobody() and the three-body energy of threebody(). */
  private fpairOut = 0;
  private engOut = 0;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    const np1 = ntypes + 1;
    this.setflag = new Uint8Array(np1 * np1);
    this.map = new Int32Array(np1).fill(-1);
    this.elements = [];
    this.params = [];
    this.paramsMapped = false;
    this.fileName = '';
  }

  /** PairSW::settings: "threebody on|off" only. */
  override settings(args: string[], _ctx: StyleContext): void {
    let iarg = 0;
    while (iarg < args.length) {
      if (args[iarg] === 'threebody') {
        if (iarg + 2 > args.length) throw new StyleError('Illegal pair_style sw command: missing argument for threebody');
        this.threebody = logical(args[iarg + 1]);
        this.needsFull = this.threebody;
        this.needsHalf = !this.threebody;
        iarg += 2;
      } else {
        throw new StyleError(`Illegal pair_style sw keyword: ${args[iarg]}`);
      }
    }
  }

  /**
   * PairSW::coeff. With threebody on, "* * file elem1 ... elemN" reads the file each time (one_coeff). With threebody
   * off the file and the element maps are set by the first command, and each later command sets its type pairs.
   */
  override coeff(args: string[], ctx: StyleContext): void {
    const nt = this.ntypes;
    if (nt === 0) throw new StyleError('Pair_coeff command before simulation box is defined');
    if (args.length < 3) throw new StyleError('Incorrect args for pair coefficients');
    if (this.threebody && (args[0] !== '*' || args[1] !== '*')) {
      throw new StyleError('Incorrect args for pair coefficients');
    }
    if (this.threebody || !this.paramsMapped) {
      if (!this.threebody) this.setflag.fill(0);
      this.mapElements(args.slice(3));
      this.readFile(args[2], ctx);
      this.setupParams();
      this.paramsMapped = true;
    }
    if (!this.threebody) {
      const [ilo, ihi] = typeBounds(args[0], nt);
      const [jlo, jhi] = typeBounds(args[1], nt);
      const np1 = nt + 1;
      let count = 0;
      for (let i = ilo; i <= ihi; i++) {
        if ((this.map[i] >= 0 && args[i + 2] !== this.elements[this.map[i]]) || (this.map[i] < 0 && args[i + 2] !== 'NULL')) {
          throw new StyleError('Must use consistent type to element mappings with threebody off');
        }
        if (this.map[i] < 0) throw new StyleError('Must not set pair_coeff mapped to NULL element');
        for (let j = Math.max(jlo, i); j <= jhi; j++) {
          if ((this.map[j] >= 0 && args[j + 2] !== this.elements[this.map[j]]) || (this.map[j] < 0 && args[j + 2] !== 'NULL')) {
            throw new StyleError('Must use consistent type to element mappings with threebody off');
          }
          if (this.map[j] < 0) throw new StyleError('Must not set pair_coeff mapped to NULL element');
          this.setflag[i * np1 + j] = 1;
          count++;
        }
      }
      if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
    }
  }

  /** Pair::map_element2type (update_setflag is not needed here: setflag is set by coeff). */
  private mapElements(names: string[]): void {
    if (names.length !== this.ntypes) {
      throw new StyleError('Number of element to type mappings does not match number of atom types');
    }
    const elements: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const entry = names[t - 1];
      if (entry === 'NULL') continue;
      let j = elements.indexOf(entry);
      if (j < 0) { j = elements.length; elements.push(entry); }
      map[t] = j;
    }
    this.elements = elements;
    this.map = map;
  }

  /**
   * PairSW::read_file: entries whose three elements are all mapped, in file order. Energies scale with the UNITS
   * conversion; threebody off zeroes lambda; negative parameters are an error.
   */
  private readFile(fileName: string, ctx: StyleContext): void {
    const text = ctx.readFile(fileName, potentialFile('sw'));
    this.fileName = fileName;
    if (!this.threebody) ctx.log('  disabling sw potential three-body terms');
    this.unitFactor = potentialUnitFactor(text, 'sw', fileName, ctx);
    const reader = new PotentialEntryReader(text.split(/\r?\n/), 'sw');
    const params: SWParam[] = [];
    for (;;) {
      const w = reader.next(NPARAMS_PER_LINE);
      if (!w) break;
      const ielement = this.elements.indexOf(w[0]);
      if (ielement < 0) continue;
      const jelement = this.elements.indexOf(w[1]);
      if (jelement < 0) continue;
      const kelement = this.elements.indexOf(w[2]);
      if (kelement < 0) continue;
      const p = newParam();
      p.ielement = ielement; p.jelement = jelement; p.kelement = kelement;
      const v = (k: number) => nextDouble(w[3 + k]);
      p.epsilon = v(0); p.sigma = v(1); p.littlea = v(2); p.lambda = v(3); p.gamma = v(4); p.costheta = v(5);
      p.biga = v(6); p.bigb = v(7); p.powerp = v(8); p.powerq = v(9); p.tol = v(10);
      if (this.unitFactor !== 1) p.epsilon *= this.unitFactor;
      if (!this.threebody) p.lambda = 0;
      if (p.epsilon < 0.0 || p.sigma < 0.0 || p.littlea < 0.0 || p.lambda < 0.0 || p.gamma < 0.0 ||
        p.biga < 0.0 || p.bigb < 0.0 || p.powerp < 0.0 || p.powerq < 0.0 || p.tol < 0.0) {
        throw new StyleError('Illegal Stillinger-Weber parameter');
      }
      params.push(p);
    }
    this.params = params;
  }

  /** PairSW::setup_params: the exact-match triplet table, the cutoffs and the derived constants of each entry. */
  private setupParams(): void {
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
      p.cut = p.sigma * p.littlea;
      let rtmp = p.cut;
      if (p.tol > 0.0) {
        if (p.tol > 0.01) p.tol = 0.01;
        if (p.gamma < 1.0) rtmp = rtmp + p.gamma * p.sigma / libmLog(p.tol);
        else rtmp = rtmp + p.sigma / libmLog(p.tol);
      }
      p.cutsq = rtmp * rtmp;
      p.sigma_gamma = p.sigma * p.gamma;
      p.lambda_epsilon = p.lambda * p.epsilon;
      p.lambda_epsilon2 = 2.0 * p.lambda * p.epsilon;
      p.c1 = p.biga * p.epsilon * p.powerp * p.bigb * libmPow(p.sigma, p.powerp);
      p.c2 = p.biga * p.epsilon * p.powerq * libmPow(p.sigma, p.powerq);
      p.c3 = p.biga * p.epsilon * p.bigb * libmPow(p.sigma, p.powerp + 1.0);
      p.c4 = p.biga * p.epsilon * libmPow(p.sigma, p.powerq + 1.0);
      p.c5 = p.biga * p.epsilon * p.bigb * libmPow(p.sigma, p.powerp);
      p.c6 = p.biga * p.epsilon * libmPow(p.sigma, p.powerq);
    }
    this.cutmax = 0.0;
    for (const p of this.params) {
      const rtmp = Math.sqrt(p.cutsq);
      if (rtmp > this.cutmax) this.cutmax = rtmp;
    }
    // the two-body cutoff of each element pair, params[elem3param[i][j][j]].cutsq, as a flat table for compute
    this.pairCutsq = new Float64Array(nel * nel);
    for (let a = 0; a < nel; a++) {
      for (let b = 0; b < nel; b++) this.pairCutsq[a * nel + b] = this.params[this.elem3param[(a * nel + b) * nel + b]].cutsq;
    }
  }

  override initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) throw new StyleError('pair_style sw does not support the pair_modify table option');
    if (ctx.newtonPair === false) throw new StyleError('Pair style Stillinger-Weber requires newton pair on');
  }

  override initOne(i: number, j: number): number {
    if (!this.paramsMapped) throw new StyleError('All pair coeffs are not set');
    if (!this.threebody && this.setflag[i * (this.ntypes + 1) + j] === 0) {
      throw new StyleError('All pair coeffs are not set');
    }
    return this.cutmax;
  }

  override setsPair(i: number, j: number): boolean {
    return this.threebody || this.setflag[i * (this.ntypes + 1) + j] === 1;
  }

  /** PairSW::twobody: sets this.fpairOut (force over r) and this.engOut (energy). */
  private twobody(p: SWParam, rsq: number): void {
    const r = Math.sqrt(rsq);
    const rinvsq = 1.0 / rsq;
    const rp = libmPow(r, -p.powerp);
    const rq = libmPow(r, -p.powerq);
    const rainv = 1.0 / (r - p.cut);
    const rainvsq = rainv * rainv * r;
    const expsrainv = libmExp(p.sigma * rainv);
    this.fpairOut = (p.c1 * rp - p.c2 * rq + (p.c3 * rp - p.c4 * rq) * rainvsq) * expsrainv * rinvsq;
    this.engOut = (p.c5 * rp - p.c6 * rq) * expsrainv;
  }

  /**
   * The leg values of PairSW::threebody for short neighbour m of the current atom (separation i -> m, its entry of the
   * leg pm): r, 1/r^2, gsrainv / ((r - cut) r) and exp(gsrainv), the expressions of the native threebody. They depend on
   * the leg only, so each is computed once per short neighbour and used by every triplet of that leg, with the same
   * values as native's per-triplet evaluation.
   */
  private legValues(m: number, pm: SWParam, dx: number, dy: number, dz: number): void {
    const L = this.leg;
    L[8 * m] = dx; L[8 * m + 1] = dy; L[8 * m + 2] = dz;
    const rsq = dx * dx + dy * dy + dz * dz;
    const r = Math.sqrt(rsq);
    const rainv = 1.0 / (r - pm.cut);
    const gsrainv = pm.sigma_gamma * rainv;
    L[8 * m + 3] = r;
    L[8 * m + 4] = 1.0 / rsq;
    L[8 * m + 5] = gsrainv * rainv / r;
    L[8 * m + 6] = libmExp(gsrainv);
  }

  /**
   * PairSW::threebody for the legs a (i -> j) and b (i -> k) of the short list (legValues) and the triplet entry pijk:
   * sets this.engOut (the energy facrad) and fills this.fj, this.fk, in native's operation order.
   */
  private threebodyTerm(pijk: SWParam, a: number, b: number): void {
    const fj = this.fj, fk = this.fk, L = this.leg;
    const a8 = 8 * a, b8 = 8 * b;
    const d1x = L[a8], d1y = L[a8 + 1], d1z = L[a8 + 2];
    const d2x = L[b8], d2y = L[b8 + 1], d2z = L[b8 + 2];
    const r1 = L[a8 + 3], rinvsq1 = L[a8 + 4], gsrainvsq1 = L[a8 + 5], expgsrainv1 = L[a8 + 6];
    const r2 = L[b8 + 3], rinvsq2 = L[b8 + 4], gsrainvsq2 = L[b8 + 5], expgsrainv2 = L[b8 + 6];
    const rinv12 = 1.0 / (r1 * r2);
    const cs = (d1x * d2x + d1y * d2y + d1z * d2z) * rinv12;
    const delcs = cs - pijk.costheta;
    const delcssq = delcs * delcs;
    const facexp = expgsrainv1 * expgsrainv2;
    const facrad = pijk.lambda_epsilon * facexp * delcssq;
    const frad1 = facrad * gsrainvsq1;
    const frad2 = facrad * gsrainvsq2;
    const facang = pijk.lambda_epsilon2 * facexp * delcs;
    const facang12 = rinv12 * facang;
    const csfacang = cs * facang;
    const csfac1 = rinvsq1 * csfacang;
    fj[0] = d1x * (frad1 + csfac1) - d2x * facang12;
    fj[1] = d1y * (frad1 + csfac1) - d2y * facang12;
    fj[2] = d1z * (frad1 + csfac1) - d2z * facang12;
    const csfac2 = rinvsq2 * csfacang;
    fk[0] = d2x * (frad2 + csfac2) - d1x * facang12;
    fk[1] = d2y * (frad2 + csfac2) - d1y * facang12;
    fk[2] = d2z * (frad2 + csfac2) - d1z * facang12;
    this.engOut = facrad;
  }

  /** PairSW::compute: the two-body terms of each pair, then the three-body terms over the short neighbor list of i. */
  override compute(pc: PairCompute): void {
    const list = this.threebody ? pc.full : pc.half;
    if (!list) throw new StyleError(`pair style ${this.name} needs a ${this.threebody ? 'full' : 'half'} neighbor list`);
    const x = pc.x, f = pc.f, type = pc.type;
    const nel = this.elements.length;
    const map = this.map, e3 = this.elem3param, P = this.params;
    const three = this.threebody;
    const d1 = this.d1, d2 = this.d2;
    const fj = this.fj, fk = this.fk;
    // the neighbours in native full-list order (the list order for a half list, where only pairs matter)
    const nbrs = three ? this.natOrder.neighbors(pc.s, pc.nb, pc.geom, list) : list.neighbors;
    const cut2 = this.pairCutsq, ids = pc.s.id, owner = pc.nb.owner;

    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const itype = map[type[i]];
      if (itype < 0) continue;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itag = ids[i];
      let fxi = 0, fyi = 0, fzi = 0;
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];

      // two-body interactions (skipping half of the pairs of the full list), gathering the short list
      if (this.neighshort.length < jnum) this.neighshort = new Int32Array(jnum);
      const neighshort = this.neighshort;
      let numshort = 0;
      for (let jj = 0; jj < jnum; jj++) {
        const j = nbrs[k0 + jj] & NEIGHMASK;
        const jtype = map[type[j]];
        if (jtype < 0) continue;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq >= cut2[itype * nel + jtype]) continue;
        const ijparam = e3[(itype * nel + jtype) * nel + jtype];
        neighshort[numshort++] = j;
        if (three) {
          const jtag = ids[owner[j]];
          if (itag > jtag) {
            if ((itag + jtag) % 2 === 0) continue;
          } else if (itag < jtag) {
            if ((itag + jtag) % 2 === 1) continue;
          } else {
            if (x[3 * j + 2] < ztmp) continue;
            if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) continue;
            if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) continue;
          }
        }
        this.twobody(P[ijparam], rsq);
        const fpair = this.fpairOut;
        fxi += delx * fpair;
        fyi += dely * fpair;
        fzi += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        tallyPair(pc, i, j, this.engOut, fpair, delx, dely, delz);
      }

      // three-body interactions over the pairs of the short list (none with threebody off)
      const jnumm1 = three ? numshort - 1 : 0;
      if (jnumm1 > 0) {
        if (this.leg.length < 8 * numshort) this.leg = new Float64Array(8 * numshort);
        for (let m = 0; m < numshort; m++) {
          const j = neighshort[m];
          const jtype = map[type[j]];
          this.legValues(m, P[e3[(itype * nel + jtype) * nel + jtype]], x[3 * j] - xtmp, x[3 * j + 1] - ytmp, x[3 * j + 2] - ztmp);
        }
      }
      const L = this.leg;
      for (let jj = 0; jj < jnumm1; jj++) {
        const j = neighshort[jj];
        const jtype = map[type[j]];
        d1[0] = L[8 * jj]; d1[1] = L[8 * jj + 1]; d1[2] = L[8 * jj + 2];
        let fjx = 0, fjy = 0, fjz = 0;
        for (let kk = jj + 1; kk < numshort; kk++) {
          const k = neighshort[kk];
          const ktype = map[type[k]];
          const ijkparam = e3[(itype * nel + jtype) * nel + ktype];
          this.threebodyTerm(P[ijkparam], jj, kk);
          d2[0] = L[8 * kk]; d2[1] = L[8 * kk + 1]; d2[2] = L[8 * kk + 2];
          const eng = this.engOut;
          fxi -= fj[0] + fk[0];
          fyi -= fj[1] + fk[1];
          fzi -= fj[2] + fk[2];
          fjx += fj[0];
          fjy += fj[1];
          fjz += fj[2];
          f[3 * k] += fk[0];
          f[3 * k + 1] += fk[1];
          f[3 * k + 2] += fk[2];
          pc.acc.evdwl += eng;
          if (pc.eatom) {
            const e3rd = THIRD * eng;
            pc.eatom[i] += e3rd;
            pc.eatom[j] += e3rd;
            pc.eatom[k] += e3rd;
          }
          tallyTriplet(pc, i, j, k, fj, fk, d1, d2);
        }
        f[3 * j] += fjx;
        f[3 * j + 1] += fjy;
        f[3 * j + 2] += fjz;
      }
      f[3 * i] += fxi;
      f[3 * i + 1] += fyi;
      f[3 * i + 2] += fzi;
    }
  }

  /**
   * PairSW::single: the two-body term alone, with threebody off only (native single_enable is set then).
   */
  override single(_i: number, _j: number, itype: number, jtype: number, rsq: number): { eng: number; fforce: number } {
    if (this.threebody) throw new StyleError('pair style sw: single() is only supported with threebody off');
    const nel = this.elements.length;
    const mi = this.map[itype], mj = this.map[jtype];
    if (mi < 0 || mj < 0) return { eng: 0, fforce: 0 };
    const p = this.params[this.elem3param[(mi * nel + mj) * nel + mj]];
    if (rsq >= p.cutsq) return { eng: 0, fforce: 0 };
    this.twobody(p, rsq);
    return { eng: this.engOut, fforce: this.fpairOut };
  }
}

/** utils::logical (src/utils.cpp at b332f28) for the threebody value: the words yes/no, on/off, true/false, 1/0. */
const logical = (word: string): boolean => {
  const w = word.toLowerCase();
  if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
  if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
  throw new StyleError(`Expected boolean parameter instead of '${word}' in input script or data file`);
};
