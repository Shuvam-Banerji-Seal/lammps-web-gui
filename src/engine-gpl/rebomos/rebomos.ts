/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_rebomos.cpp and src/MANYBODY/pair_rebomos.h at commit b332f28.
 * Copyright (2003) Sandia Corporation (GNU General Public License). Reference in the C++ header:
 * Stewart J A and Spearot D E (2013) Atomistic simulations of nanoindentation on the basal plane of
 * crystalline molybdenum disulfide, Modelling Simul. Mater. Sci. Eng. 21.
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { rowNeighbors } from '../airebo/neigh';
import { powint } from '../airebo/splines';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';
import { exp as libmExp } from '../libm/exp';

const TOL = 1.0e-9;
const MY_PI = 3.14159265358979323846;
const MY_2PI = 6.28318530717958647692;

/** Derivative out-parameter (C++ double& dX). */
interface Ref { v: number }

/** Cutoff function Sp(t) of pair_rebomos.h: value returned, derivative in out.v. */
const sp = (xij: number, xmin: number, xmax: number, out: Ref): number => {
  let cutoff: number;
  const t = (xij - xmin) / (xmax - xmin);
  if (t <= 0.0) {
    cutoff = 1.0;
    out.v = 0.0;
  } else if (t >= 1.0) {
    cutoff = 0.0;
    out.v = 0.0;
  } else {
    cutoff = 0.5 * (1.0 + libmCos(t * MY_PI));
    out.v = (-0.5 * MY_PI * libmSin(t * MY_PI)) / (xmax - xmin);
  }
  return cutoff;
};

/** Seven-coefficient polynomial in costh (Horner, the order of gSpline in pair_rebomos.h). */
const poly7 = (b: Float64Array, o: number, c: number, dg: Ref): number => {
  let g = b[o + 6] * c;
  let d = 6.0 * b[o + 6] * c;
  for (let k = 5; k >= 1; k--) {
    g += b[o + k];
    d += k * b[o + k];
    g *= c;
    if (k > 1) d *= c;
  }
  g += b[o];
  dg.v = d;
  return g;
};

/** G(cos theta) of pair_rebomos.h (gSpline); bg holds the gamma polynomial of the 0.5..1 branch. */
const gSpline = (costh: number, typei: number, b: Float64Array, bg: Float64Array, dgdc: Ref): number => {
  const o = typei * 7;
  const d = { v: 0 };
  let g = 0.0;
  if (costh >= -1.0 && costh < 0.5) {
    g = poly7(b, o, costh, d);
    dgdc.v = d.v;
  } else if (costh >= 0.5 && costh <= 1.0) {
    const dgcos = { v: 0 };
    const gcos = poly7(b, o, costh, dgcos);
    const dgamma = { v: 0 };
    const gamma = poly7(bg, o, costh, dgamma);
    const tmp = MY_2PI * (costh - 0.5);
    const psi = 0.5 * (1 - libmCos(tmp));
    const dpsi = MY_PI * libmSin(tmp);
    g = gcos + psi * (gamma - gcos);
    dgdc.v = dgcos.v + dpsi * (gamma - gcos) + psi * (dgamma.v - dgcos.v);
  } else {
    dgdc.v = 0.0;
  }
  return g;
};

/**
 * Pair style rebomos (MoS2 REBO, pair_rebomos.cpp). Two elements, Mo (index 0) and S (index 1).
 * The energy and forces follow the C++ code term by term, including its bond-order force terms
 * (the dgdN contribution is left out there too).
 */
export class PairREBOMoS extends Pair {
  readonly name = 'rebomos';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  /** element index (0 = Mo, 1 = S) per atom type, -1 for NULL. */
  private map = new Int32Array(0);
  private fileRead = false;
  private cut3rebo = 0;
  // 2x2 element-pair arrays, index e0 * 2 + e1
  private rcmin = new Float64Array(4);
  private rcmax = new Float64Array(4);
  private rcmaxsq = new Float64Array(4);
  private Q = new Float64Array(4);
  private alpha = new Float64Array(4);
  private A = new Float64Array(4);
  private BIJc = new Float64Array(4);
  private Beta = new Float64Array(4);
  private rcLJmin = new Float64Array(4);
  private rcLJmax = new Float64Array(4);
  private epsilon = new Float64Array(4);
  private sigma = new Float64Array(4);
  private lj1 = new Float64Array(4);
  private lj2 = new Float64Array(4);
  private lj3 = new Float64Array(4);
  private lj4 = new Float64Array(4);
  // per element: 7 coefficients of b, bg (gSpline) and 4 of a (PijSpline)
  private b = new Float64Array(14);
  private bg = new Float64Array(14);
  private a = new Float64Array(8);

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style rebomos command');
  }

  override coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes) {
      throw new StyleError('Incorrect args for pair coefficients (pair_style rebomos needs * * file and one element per type)');
    }
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients: the first 2 arguments must be * *');
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let i = 3; i < args.length; i++) {
      const w = args[i];
      if (w === 'NULL') {
        throw new StyleError('pair_style rebomos does not support the NULL element mapping (every atom type needs Mo or S)');
      } else if (w === 'Mo' || w === 'M') {
        map[i - 2] = 0;
      } else if (w === 'S') {
        map[i - 2] = 1;
      } else {
        throw new StyleError(`Incorrect args for pair coefficients: element ${w} is not Mo or S (pair_style rebomos)`);
      }
    }
    this.readFile(ctx.readFile(args[2]));
    this.map = map;
    this.fileRead = true;
  }

  /** The 61 numbers of the potential file, in the order of pair_rebomos.cpp read_file (first token of each line). */
  private readFile(text: string): void {
    const vals: number[] = [];
    for (const raw of text.split(/\r?\n/)) {
      const line = raw.split('#')[0].trim();
      if (line === '') continue;
      vals.push(Number(line.split(/\s+/)[0]));
    }
    if (vals.length < 61 || vals.slice(0, 61).some((v) => !Number.isFinite(v))) {
      throw new StyleError('reading rebomos potential file: expected 61 numeric parameters');
    }
    let p = 0;
    const nx = () => vals[p++];
    const rcminMM = nx(), rcminMS = nx(), rcminSS = nx();
    const rcmaxMM = nx(), rcmaxMS = nx(), rcmaxSS = nx();
    const QMM = nx(), QMS = nx(), QSS = nx();
    const aMM = nx(), aMS = nx(), aSS = nx();
    const AMM = nx(), AMS = nx(), ASS = nx();
    const BMM = nx(), BMS = nx(), BSS = nx();
    const BetaMM = nx(), BetaMS = nx(), BetaSS = nx();
    const Mb = [0, 0, 0, 0, 0, 0, 0].map(() => nx());
    const Mbg = [0, 0, 0, 0, 0, 0, 0].map(() => nx());
    const Sb = [0, 0, 0, 0, 0, 0, 0].map(() => nx());
    const Sbg = [0, 0, 0, 0, 0, 0, 0].map(() => nx());
    const Ma = [0, 0, 0, 0].map(() => nx());
    const Sa = [0, 0, 0, 0].map(() => nx());
    const epsMM = nx(), epsSS = nx(), sigMM = nx(), sigSS = nx();

    this.rcmin.set([rcminMM, rcminMS, rcminMS, rcminSS]);
    this.rcmax.set([rcmaxMM, rcmaxMS, rcmaxMS, rcmaxSS]);
    for (let q = 0; q < 4; q++) this.rcmaxsq[q] = this.rcmax[q] * this.rcmax[q];
    this.Q.set([QMM, QMS, QMS, QSS]);
    this.alpha.set([aMM, aMS, aMS, aSS]);
    this.A.set([AMM, AMS, AMS, ASS]);
    this.BIJc.set([BMM, BMS, BMS, BSS]);
    this.Beta.set([BetaMM, BetaMS, BetaMS, BetaSS]);
    this.b.set([...Mb, ...Sb]);
    this.bg.set([...Mbg, ...Sbg]);
    this.a.set([...Ma, ...Sa]);
    const sigma = [sigMM, (sigMM + sigSS) / 2, (sigMM + sigSS) / 2, sigSS];
    this.sigma.set(sigma);
    this.epsilon.set([epsMM, Math.sqrt(epsMM * epsSS), Math.sqrt(epsMM * epsSS), epsSS]);
    this.rcLJmin.set([rcminMM, rcminMS, rcminMS, rcminSS]);
    for (let q = 0; q < 4; q++) this.rcLJmax[q] = 2.5 * this.sigma[q];
  }

  override initStyle(_ctx: StyleContext): void {
    if (!this.fileRead) throw new StyleError('pair_style rebomos needs a pair_coeff command with a potential file');
    if (this.shift || this.tail) throw new StyleError('pair_style rebomos does not support the pair_modify shift and tail options');
    for (let q = 0; q < 4; q++) {
      const s = this.sigma[q], e = this.epsilon[q];
      this.lj1[q] = 48.0 * e * powint(s, 12);
      this.lj2[q] = 24.0 * e * powint(s, 6);
      this.lj3[q] = 4.0 * e * powint(s, 12);
      this.lj4[q] = 4.0 * e * powint(s, 6);
    }
    this.cut3rebo = 3.0 * this.rcmax[0];
  }

  /** PairREBOMoS::init_one: every type pair uses the 3-REBO cutoff for the neighbor and ghost shell. */
  override initOne(i: number, j: number): number {
    if (this.map[i] < 0 || this.map[j] < 0) throw new StyleError('All pair coeffs are not set (pair_style rebomos)');
    return this.cut3rebo;
  }

  override compute(pc: PairCompute): void {
    if (!this.fileRead) throw new StyleError('pair_style rebomos has no potential file');
    if (pc.vatom) throw new StyleError('compute stress/atom is not supported by pair style rebomos');
    const nall = pc.nall, nlocal = pc.nlocal;
    const mt = new Int32Array(nall);
    for (let a = 0; a < nall; a++) {
      const m = this.map[pc.type[a]];
      if (m < 0) throw new StyleError(`pair_style rebomos: atom type ${pc.type[a]} is mapped to NULL`);
      mt[a] = m;
    }
    const tag = new Int32Array(nall);
    const owner = pc.nb.owner;
    for (let a = 0; a < nall; a++) tag[a] = pc.s.id[owner[a]];
    const x = pc.x, f = pc.f;

    // REBO neighbors of owned and ghost atoms (rsq < rcmaxsq), and the coordination sums nM, nS.
    let rcmaxMax = 0;
    for (let q = 0; q < 4; q++) if (this.rcmax[q] > rcmaxMax) rcmaxMax = this.rcmax[q];
    const rebo = rowNeighbors(x, nall, nall, mt, this.rcmaxsq, rcmaxMax);
    const nM = new Float64Array(nall), nS = new Float64Array(nall);
    const scratch: Ref = { v: 0 };
    for (let a = 0; a < nall; a++) {
      for (let kk = rebo.start[a]; kk < rebo.start[a + 1]; kk++) {
        const b = rebo.list[kk];
        const q = mt[a] * 2 + mt[b];
        const dx = x[3 * a] - x[3 * b], dy = x[3 * a + 1] - x[3 * b + 1], dz = x[3 * a + 2] - x[3 * b + 2];
        const w = sp(Math.sqrt(dx * dx + dy * dy + dz * dz), this.rcmin[q], this.rcmax[q], scratch);
        if (mt[b] === 0) nM[a] += w;
        else nS[a] += w;
      }
    }
    const ctx: Ctx = { x, f, mt, tag, rstart: rebo.start, rlist: rebo.list, nM, nS, pc, scratch: { v: 0 } };
    this.frebo(ctx, nlocal);
    this.flj(ctx, nlocal, nall, mt);
  }

  /** PairREBOMoS::FREBO: REBO two-body term with the bond order, each physical pair once. */
  private frebo(c: Ctx, nlocal: number): void {
    const { x, f, mt, tag, rstart, rlist, pc } = c;
    const dwij = { v: 0 }, dwik = { v: 0 };
    for (let i = 0; i < nlocal; i++) {
      const itag = tag[i];
      const itype = mt[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
        const j = rlist[kk];
        const jtag = tag[j];
        if (!keepPair(itag, jtag, j, x, xtmp, ytmp, ztmp)) continue;
        const jtype = mt[j];
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const rij = Math.sqrt(rsq);
        const q = itype * 2 + jtype;
        const wij = sp(rij, this.rcmin[q], this.rcmax[q], dwij);
        if (wij <= TOL) continue;
        const Qij = this.Q[q], Aij = this.A[q], alphaij = this.alpha[q];
        const VR = wij * (1.0 + (Qij / rij)) * Aij * libmExp(-alphaij * rij);
        const pre = wij * Aij * libmExp(-alphaij * rij);
        let dVRdi = pre * ((-alphaij) - (Qij / rsq) - (Qij * alphaij / rij));
        dVRdi += VR / wij * dwij.v;
        const VA = -wij * this.BIJc[q] * libmExp(-this.Beta[q] * rij);
        let dVA = -this.Beta[q] * VA;
        dVA += VA / wij * dwij.v;
        const del = [delx, dely, delz];
        const bij = this.bondorder(c, i, j, del, rij, VA, dwik);
        const dVAdi = bij * dVA;
        const fpair = -(dVRdi + dVAdi) / rij;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        const evdwl = VR + bij * VA;
        pc.acc.evdwl += evdwl;
        if (pc.eatom) {
          pc.eatom[i] += 0.5 * evdwl;
          pc.eatom[j] += 0.5 * evdwl;
        }
      }
    }
  }

  /** PairREBOMoS::bondorder: returns bij = (pij + pji) / 2 and adds the bond-order forces (C++ order kept). */
  private bondorder(c: Ctx, i: number, j: number, rij: number[], rijmag: number, VA: number, dwik: Ref): number {
    const { x, f, mt, rstart, rlist, nM, nS } = c;
    const itype = mt[i], jtype = mt[j];
    const dwij = { v: 0 };
    sp(rijmag, this.rcmin[itype * 2 + jtype], this.rcmax[itype * 2 + jtype], dwij);
    const NijM = nM[i], NijS = nS[i], NjiM = nM[j], NjiS = nS[j];
    const dgdc = { v: 0 };
    const dp = { v: 0 };
    const dS = { v: 0 };
    let tmp = 0.0, tmp2 = 0.0;
    let Etmp = 0.0;
    const rik = [0, 0, 0], dcosdri = [0, 0, 0], dcosdrk = [0, 0, 0], dcosdrj = [0, 0, 0];
    const fi = [0, 0, 0], fj = [0, 0, 0], fk = [0, 0, 0];
    const rijv = rij;

    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const k = rlist[kk];
      if (k === j) continue;
      const ktype = mt[k];
      for (let d = 0; d < 3; d++) rik[d] = x[3 * i + d] - x[3 * k + d];
      const rikmag = Math.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2]);
      const wik = sp(rikmag, this.rcmin[itype * 2 + ktype], this.rcmax[itype * 2 + ktype], dS);
      let cosjik = (rijv[0] * rik[0] + rijv[1] * rik[1] + rijv[2] * rik[2]) / (rijmag * rikmag);
      cosjik = Math.min(cosjik, 1.0);
      cosjik = Math.max(cosjik, -1.0);
      const g = gSpline(cosjik, itype, this.b, this.bg, dgdc);
      Etmp = Etmp + (wik * g);
    }
    dp.v = 0.0;
    const PijS = this.pijSpline(NijM, NijS, itype, dp);
    const pij = 1.0 / Math.sqrt(1.0 + Etmp + PijS);
    tmp = -0.5 * pij * pij * pij;

    // derivatives of the i side
    for (let kk = rstart[i]; kk < rstart[i + 1]; kk++) {
      const k = rlist[kk];
      if (k === j) continue;
      const ktype = mt[k];
      for (let d = 0; d < 3; d++) rik[d] = x[3 * i + d] - x[3 * k + d];
      const rikmag = Math.sqrt(rik[0] * rik[0] + rik[1] * rik[1] + rik[2] * rik[2]);
      const wik = sp(rikmag, this.rcmin[itype * 2 + ktype], this.rcmax[itype * 2 + ktype], dwik);
      let cosjik = (rijv[0] * rik[0] + rijv[1] * rik[1] + rijv[2] * rik[2]) / (rijmag * rikmag);
      cosjik = Math.min(cosjik, 1.0);
      cosjik = Math.max(cosjik, -1.0);
      for (let d = 0; d < 3; d++) {
        dcosdri[d] = ((rijv[d] + rik[d]) / (rijmag * rikmag)) -
          (cosjik * ((rijv[d] / (rijmag * rijmag)) + (rik[d] / (rikmag * rikmag))));
        dcosdrk[d] = (-rijv[d] / (rijmag * rikmag)) + (cosjik * (rik[d] / (rikmag * rikmag)));
        dcosdrj[d] = (-rik[d] / (rijmag * rikmag)) + (cosjik * (rijv[d] / (rijmag * rijmag)));
      }
      const g = gSpline(cosjik, itype, this.b, this.bg, dgdc);
      tmp2 = VA * 0.5 * (tmp * wik * dgdc.v);
      for (let d = 0; d < 3; d++) {
        fj[d] = -tmp2 * dcosdrj[d];
        fi[d] = -tmp2 * dcosdri[d];
        fk[d] = -tmp2 * dcosdrk[d];
      }
      tmp2 = VA * 0.5 * (tmp * dwik.v * g) / rikmag;
      for (let d = 0; d < 3; d++) {
        fi[d] -= tmp2 * rik[d];
        fk[d] += tmp2 * rik[d];
      }
      tmp2 = VA * 0.5 * (tmp * dp.v * dwik.v) / rikmag;
      for (let d = 0; d < 3; d++) {
        fi[d] -= tmp2 * rik[d];
        fk[d] += tmp2 * rik[d];
      }
      for (let d = 0; d < 3; d++) {
        f[3 * i + d] += fi[d];
        f[3 * j + d] += fj[d];
        f[3 * k + d] += fk[d];
      }
    }
    // PIJ force contribution additional term (i side)
    tmp2 = -VA * 0.5 * (tmp * dp.v * dwij.v) / rijmag;
    for (let d = 0; d < 3; d++) {
      f[3 * i + d] += rijv[d] * tmp2;
      f[3 * j + d] -= rijv[d] * tmp2;
    }

    // j side
    tmp = 0.0;
    tmp2 = 0.0;
    Etmp = 0.0;
    const rjl = [0, 0, 0];
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const l = rlist[ll];
      if (l === i) continue;
      const ltype = mt[l];
      for (let d = 0; d < 3; d++) rjl[d] = x[3 * j + d] - x[3 * l + d];
      const rjlmag = Math.sqrt(rjl[0] * rjl[0] + rjl[1] * rjl[1] + rjl[2] * rjl[2]);
      const wjl = sp(rjlmag, this.rcmin[jtype * 2 + ltype], this.rcmax[jtype * 2 + ltype], dS);
      let cosijl = -1.0 * (rijv[0] * rjl[0] + rijv[1] * rjl[1] + rijv[2] * rjl[2]) / (rijmag * rjlmag);
      cosijl = Math.min(cosijl, 1.0);
      cosijl = Math.max(cosijl, -1.0);
      const g = gSpline(cosijl, jtype, this.b, this.bg, dgdc);
      Etmp = Etmp + (wjl * g);
    }
    dp.v = 0.0;
    const PjiS = this.pijSpline(NjiM, NjiS, jtype, dp);
    const pji = 1.0 / Math.sqrt(1.0 + Etmp + PjiS);
    tmp = -0.5 * pji * pji * pji;

    const dcosdri2 = [0, 0, 0], dcosdrj2 = [0, 0, 0], dcosdrl = [0, 0, 0];
    for (let ll = rstart[j]; ll < rstart[j + 1]; ll++) {
      const l = rlist[ll];
      if (l === i) continue;
      const ltype = mt[l];
      for (let d = 0; d < 3; d++) rjl[d] = x[3 * j + d] - x[3 * l + d];
      const rjlmag = Math.sqrt(rjl[0] * rjl[0] + rjl[1] * rjl[1] + rjl[2] * rjl[2]);
      const dwjl = { v: 0 };
      const wjl = sp(rjlmag, this.rcmin[jtype * 2 + ltype], this.rcmax[jtype * 2 + ltype], dwjl);
      let cosijl = (-1.0 * (rijv[0] * rjl[0] + rijv[1] * rjl[1] + rijv[2] * rjl[2])) / (rijmag * rjlmag);
      cosijl = Math.min(cosijl, 1.0);
      cosijl = Math.max(cosijl, -1.0);
      for (let d = 0; d < 3; d++) {
        dcosdri2[d] = (-rjl[d] / (rijmag * rjlmag)) - (cosijl * rijv[d] / (rijmag * rijmag));
        dcosdrj2[d] = ((-rijv[d] + rjl[d]) / (rijmag * rjlmag)) +
          (cosijl * ((rijv[d] / (rijmag * rijmag)) - (rjl[d] / (rjlmag * rjlmag))));
        dcosdrl[d] = (rijv[d] / (rijmag * rjlmag)) + (cosijl * rjl[d] / (rjlmag * rjlmag));
      }
      const g = gSpline(cosijl, jtype, this.b, this.bg, dgdc);
      tmp2 = VA * 0.5 * (tmp * wjl * dgdc.v);
      for (let d = 0; d < 3; d++) {
        fi[d] = -tmp2 * dcosdri2[d];
        fj[d] = -tmp2 * dcosdrj2[d];
        fk[d] = -tmp2 * dcosdrl[d];
      }
      tmp2 = VA * 0.5 * (tmp * dwjl.v * g) / rjlmag;
      for (let d = 0; d < 3; d++) {
        fj[d] -= tmp2 * rjl[d];
        fk[d] += tmp2 * rjl[d];
      }
      tmp2 = VA * 0.5 * (tmp * dp.v * dwjl.v) / rjlmag;
      for (let d = 0; d < 3; d++) {
        fj[d] -= tmp2 * rjl[d];
        fk[d] += tmp2 * rjl[d];
      }
      for (let d = 0; d < 3; d++) {
        f[3 * i + d] += fi[d];
        f[3 * j + d] += fj[d];
        f[3 * l + d] += fk[d];
      }
    }
    // PIJ force contribution additional term (j side, with dp of j)
    tmp2 = -VA * 0.5 * (tmp * dp.v * dwij.v) / rijmag;
    for (let d = 0; d < 3; d++) {
      f[3 * i + d] += rijv[d] * tmp2;
      f[3 * j + d] -= rijv[d] * tmp2;
    }
    return 0.5 * (pij + pji);
  }

  /** PijSpline of pair_rebomos.h; dp receives the derivative. */
  private pijSpline(NM: number, NS: number, typei: number, dp: Ref): number {
    const N = NM + NS;
    const o = typei * 4;
    const a0 = this.a[o], a1 = this.a[o + 1], a2 = this.a[o + 2], a3 = this.a[o + 3];
    dp.v = -a0 + a1 * a2 * libmExp(-a2 * N);
    return -a0 * (N - 1) - a1 * libmExp(-a2 * N) + a3;
  }

  /** PairREBOMoS::FLJ: the Lennard-Jones term of owned atoms with each physical pair once. */
  private flj(c: Ctx, nlocal: number, nall: number, mt: Int32Array): void {
    const { x, f, tag, pc } = c;
    let cutmax = 0;
    const cutsq = new Float64Array(4);
    for (let q = 0; q < 4; q++) {
      cutsq[q] = this.rcLJmax[q] * this.rcLJmax[q];
      if (this.rcLJmax[q] > cutmax) cutmax = this.rcLJmax[q];
    }
    const list = rowNeighbors(x, nlocal, nall, mt, cutsq, cutmax);
    const delij = [0, 0, 0];
    for (let i = 0; i < nlocal; i++) {
      const itag = tag[i];
      const itype = mt[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      for (let kk = list.start[i]; kk < list.start[i + 1]; kk++) {
        const j = list.list[kk];
        const jtag = tag[j];
        if (!keepPair(itag, jtag, j, x, xtmp, ytmp, ztmp)) continue;
        const jtype = mt[j];
        const q = itype * 2 + jtype;
        delij[0] = xtmp - x[3 * j];
        delij[1] = ytmp - x[3 * j + 1];
        delij[2] = ztmp - x[3 * j + 2];
        const rijsq = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
        const rij = Math.sqrt(rijsq);
        const sig = this.sigma[q], eps = this.epsilon[q];
        let VLJ: number, dVLJ: number;
        if (rij > this.rcLJmax[q] || rij < this.rcLJmin[q]) {
          VLJ = 0;
          dVLJ = 0;
        } else if (rij >= 0.95 * sig) {
          const r2inv = 1.0 / rijsq;
          const r6inv = r2inv * r2inv * r2inv;
          VLJ = r6inv * (this.lj3[q] * r6inv - this.lj4[q]);
          dVLJ = -r6inv * (this.lj1[q] * r6inv - this.lj2[q]) / rij;
        } else {
          // below 0.95 sigma: the C2 continuation to the minimum (pair_rebomos.cpp FLJ)
          const dr = 0.95 * sig - this.rcLJmin[q];
          const r6 = powint(sig / (0.95 * sig), 6);
          const vdw = 4 * eps * r6 * (r6 - 1.0);
          const dvdw = (-4 * eps / (0.95 * sig)) * r6 * (12.0 * r6 - 6.0);
          const c2 = ((3.0 / dr) * vdw - dvdw) / dr;
          const c3 = (vdw / (dr * dr) - c2) / dr;
          const drp = rij - this.rcLJmin[q];
          VLJ = drp * drp * (drp * c3 + c2);
          dVLJ = drp * (3.0 * drp * c3 + 2.0 * c2);
        }
        const fpair = -dVLJ / rij;
        f[3 * i] += delij[0] * fpair;
        f[3 * i + 1] += delij[1] * fpair;
        f[3 * i + 2] += delij[2] * fpair;
        f[3 * j] -= delij[0] * fpair;
        f[3 * j + 1] -= delij[1] * fpair;
        f[3 * j + 2] -= delij[2] * fpair;
        pc.acc.evdwl += VLJ;
        if (pc.eatom) {
          pc.eatom[i] += 0.5 * VLJ;
          pc.eatom[j] += 0.5 * VLJ;
        }
      }
    }
  }
}

/** Working arrays of one compute() call. */
interface Ctx {
  x: Float64Array;
  f: Float64Array;
  mt: Int32Array;
  tag: Int32Array;
  rstart: Int32Array;
  rlist: Int32Array;
  nM: Float64Array;
  nS: Float64Array;
  pc: PairCompute;
  scratch: Ref;
}

/**
 * Each physical pair once (pair_rebomos.cpp FREBO and FLJ): by the parity of the ids, and for two images of one atom
 * (equal tags) by the position of the neighbor (larger z, then y, then x).
 */
const keepPair = (itag: number, jtag: number, j: number, x: Float64Array,
  xtmp: number, ytmp: number, ztmp: number): boolean => {
  if (itag > jtag) {
    if ((itag + jtag) % 2 === 0) return false;
  } else if (itag < jtag) {
    if ((itag + jtag) % 2 === 1) return false;
  } else {
    if (x[3 * j + 2] < ztmp) return false;
    if (x[3 * j + 2] === ztmp && x[3 * j + 1] < ytmp) return false;
    if (x[3 * j + 2] === ztmp && x[3 * j + 1] === ytmp && x[3 * j] < xtmp) return false;
  }
  return true;
};
