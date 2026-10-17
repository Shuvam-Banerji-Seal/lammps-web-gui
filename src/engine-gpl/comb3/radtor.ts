/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MANYBODY/pair_comb3.cpp and src/MANYBODY/pair_comb3.h (the radical terms rad_init, rad_calc,
 * rad_int, rad_forceik, rad_force; the torsion terms bbtor1, tor_calc, tor_int, tor_force; the radical and torsion
 * parts of force_zeta and compute), at commit b332f28. Authors credited in the C++ file headers: Tzu-Ray Shan
 * (Sandia) and the COMB3 authors (Ting-Wei Liang and others). Copyright (2003) Sandia Corporation (GNU General
 * Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The C++ keeps these results in class members (brad, btor, ptorr, radtmp, fi_rad ...). They are kept here in one
 * object with the same persistence: ptorr and brad keep their values between calls, as in the C++.
 */
import { StyleError } from '../../engine/force/types';
import type { Comb3Lib, Comb3Param } from './params';
import type { Comb3Kernel } from './terms';
import { pow as libmPow } from '../libm/pow';

const v3 = (): Float64Array => new Float64Array(3);

/** Spline index helpers of pair_comb3.cpp (rad_spl/tor_spl [1][4][4][10][64] and rad_grid/tor_grid [.][5][5][11]). */
const splIndex = (l: number, i: number, j: number, k: number): number => (((l * 4 + i) * 4 + j) * 10 + k) * 64;
const gridIndex = (l: number, i: number, j: number, k: number): number => ((l * 5 + i) * 5 + j) * 11 + k;

export class Comb3RadTor {
  /** brad[0..3]: radical value, pradx, prady, pradz (rad_calc); brad[3] is scaled in force_zeta. */
  readonly brad = new Float64Array(4);
  /** btor[0..3]: torsion value and derivatives (tor_calc). */
  readonly btor = new Float64Array(4);
  /** ptorr: the last pair's attractive energy times btor[0] (force_zeta), used by tor_force. */
  ptorr = 0;
  /** radtmp: the radical derivative of the last rad_forceik. */
  radtmp = 0;
  readonly fiRad = v3();
  readonly fjRad = v3();
  readonly fkRad = v3();
  readonly fiTor = v3();
  readonly fjTor = v3();
  readonly fkTor = v3();
  readonly flTor = v3();
  private readonly out = new Float64Array(4);

  constructor(private readonly kern: Comb3Kernel) {}

  private lib(): Comb3Lib {
    const lib = this.kern.lib;
    if (!lib) throw new StyleError('pair_style comb3: the radical and torsion splines need the library file lib.comb3 (C in the element list)');
    return lib;
  }

  /** rad_init: radical initialisation of one neighbor k of atom i; returns the contribution to kconjug (radtot is not needed). */
  radInit(rsq2: number, p: Comb3Param, cnconj: number): number {
    const r = Math.sqrt(rsq2);
    const fc1k = this.kern.fc(r, p);
    const radtot = -fc1k * p.pcross + cnconj;
    const radcut = this.kern.fcch(radtot);
    return fc1k * p.pcross * radcut;
  }

  /** rad_calc: radical value and derivatives for the pair (i, j) into brad. */
  radCalc(r: number, pi: Comb3Param, pj: Comb3Param, kconjug: number, lconjug: number, xcn: number, ycn: number): void {
    const lib = this.lib();
    let xrad = -this.kern.fc(r, pi) * pi.pcross + xcn;
    let yrad = -this.kern.fc(r, pj) * pj.pcross + ycn;
    let zcon = 1.0 + kconjug * kconjug + lconjug * lconjug;
    if (xrad < 0.0) xrad = 0.0;
    if (yrad < 0.0) yrad = 0.0;
    if (zcon < 1.0) zcon = 1.0;
    if (xrad > lib.maxxc) xrad = lib.maxxc;
    if (yrad > lib.maxyc) yrad = lib.maxyc;
    if (zcon > lib.maxconj) zcon = lib.maxconj;
    const ixmin = Math.trunc(xrad + 1.0e-12);
    const iymin = Math.trunc(yrad + 1.0e-12);
    const izmin = Math.trunc(zcon + 1.0e-12);
    const radindx = pi.rad_flag - 1;
    const out = this.out;
    if (Math.abs(ixmin - xrad) > 1.0e-8 || Math.abs(iymin - yrad) > 1.0e-8 || Math.abs(izmin - zcon) > 1.0e-8) {
      this.radInt(radindx, xrad, yrad, zcon, ixmin, iymin, izmin);
    } else {
      const at = gridIndex(radindx, ixmin, iymin, izmin - 1);
      out[0] = lib.radGrid[at];
      out[1] = lib.radGridx[at];
      out[2] = lib.radGridy[at];
      out[3] = lib.radGridz[at];
    }
    this.brad[0] = out[0];
    this.brad[1] = out[1];
    this.brad[2] = out[2];
    this.brad[3] = out[3];
  }

  /** rad_int: tricubic radical spline (value and derivatives) into this.out. */
  private radInt(radindx: number, xrad: number, yrad: number, zcon: number, l0: number, m0: number, n0: number): void {
    const lib = this.lib();
    const out = this.out;
    out.fill(0);
    let l = l0, m = m0, n = n0;
    if (l >= lib.maxxc - 1) l = lib.maxxc - 1;
    if (m >= lib.maxyc - 1) m = lib.maxyc - 1;
    if (n >= lib.maxconj - 1) n = lib.maxconj - 1;
    const base = splIndex(radindx, l, m, n - 1);
    this.spline(lib.radSpl, base, xrad, yrad, zcon, out);
  }

  /** The 64-term tricubic sum of the spline tables (rad_int / tor_int). Adds into out[0..2] (value, dx, dy) and out[3] (dz). */
  private spline(spl: Float64Array, base: number, xv: number, yv: number, zv: number, out: Float64Array): void {
    for (let j = 0; j < 64; j++) {
      const e0 = Math.trunc(j / 16), e1 = Math.trunc(j / 4) % 4, e2 = j % 4;
      const x = spl[base + j] * libmPow(xv, e0) * libmPow(yv, e1) * libmPow(zv, e2);
      out[0] += x;
      if (xv > 1.0e-8) out[1] += x * e0 / xv;
      if (yv > 1.0e-8) out[2] += x * e1 / yv;
      if (zv > 1.0e-8) out[3] += x * e2 / zv;
    }
  }

  /** rad_forceik: radical force on atom k (fi_rad, fk_rad) and radtmp for the k-m terms. */
  radForceik(pk: Comb3Param, rsq2: number, delrk: Float64Array, conjug: number, radtot: number): void {
    const kern = this.kern;
    const radcut = kern.fcch(radtot);
    const dradcut = kern.fcchD(radtot);
    this.fiRad.fill(0);
    this.fkRad.fill(0);
    this.radtmp = 0.0;
    const rik = Math.sqrt(rsq2);
    const fc1k = kern.fc(rik, pk);
    const fcp1k = kern.fcD(rik, pk);
    const pradk = this.brad[3] * fcp1k * radcut * pk.pcross * 2.0 * conjug;
    this.radtmp = this.brad[3] * fc1k * dradcut * pk.pcross * 2.0 * conjug;
    const ffkk2 = -pradk / rik;
    for (let nm = 0; nm < 3; nm++) {
      const fktmp = -ffkk2 * delrk[nm];
      this.fiRad[nm] = fktmp;
      this.fkRad[nm] = -fktmp;
    }
  }

  /** rad_force: radical force on atom m of the k-m term (fjRad, fkRad). */
  radForce(pm: Comb3Param, rsq3: number, delrm: Float64Array, dpradk: number): void {
    this.fjRad.fill(0);
    this.fkRad.fill(0);
    const rkm = Math.sqrt(rsq3);
    const fcp1m = this.kern.fcD(rkm, pm);
    const comkm = dpradk * fcp1m * pm.pcross;
    const ffmm2 = -comkm / rkm;
    for (let nm = 0; nm < 3; nm++) {
      const fkm = -ffmm2 * delrm[nm];
      this.fjRad[nm] = fkm;
      this.fkRad[nm] = -fkm;
    }
  }

  /** bbtor1: dihedral term of the bond order for the i-j-k-l torsion (returns its value). */
  bbtor1(torindx: number, pk: Comb3Param, pl: Comb3Param, rsq1: number, rsq2: number, rsq3: number,
    delrj: Float64Array, delrk: Float64Array, delrl: Float64Array, srmu: number): number {
    const kern = this.kern;
    const rij = Math.sqrt(rsq1), rik = Math.sqrt(rsq2), rjl = Math.sqrt(rsq3);
    const rmul0 = -(delrj[0] * delrl[0] + delrj[1] * delrl[1] + delrj[2] * delrl[2]) / (rij * rjl);
    const rmul = Math.sqrt(1.0 - rmul0 * rmul0);
    if (rmul > 0.1) {
      const fc1k = kern.fc(rik, pk);
      const fc1l = kern.fc(rjl, pl);
      const TT1 = rik * rjl * rij * rij * srmu * rmul;
      const tork = cross(delrk, delrj);
      const torl = cross(delrj, delrl);
      const TT2 = tork[0] * torl[0] + tork[1] * torl[1] + tork[2] * torl[2];
      if (torindx >= 1) {
        const rmut = (TT2 / TT1) * (TT2 / TT1);
        const btt = 1.0 - rmut;
        return btt * fc1k * fc1l;
      }
      let btt = pk.ptork1 - TT2 / TT1;
      btt = pk.ptork2 * btt * btt;
      return btt * fc1k * fc1l;
    }
    return 0.0;
  }

  /** tor_calc: torsion value and derivatives for the pair (i, j) into btor. */
  torCalc(r: number, pi: Comb3Param, pj: Comb3Param, kconjug: number, lconjug: number, xcn: number, ycn: number): void {
    const lib = this.lib();
    const btor = this.btor;
    let torindx = pi.tor_flag;
    if (torindx < 0) {
      btor.fill(0);
      btor[0] = 1.0;
      return;
    }
    let xtor = -this.kern.fc(r, pi) * pi.pcross + xcn;
    let ytor = -this.kern.fc(r, pj) * pj.pcross + ycn;
    let zcon = 1.0 + kconjug * kconjug + lconjug * lconjug;
    if (xtor < 0.0) xtor = 0.0;
    if (ytor < 0.0) ytor = 0.0;
    if (zcon < 1.0) zcon = 1.0;
    if (xtor > lib.maxxc) xtor = lib.maxxc;
    if (ytor > lib.maxyc) ytor = lib.maxyc;
    if (zcon > lib.maxconj) zcon = lib.maxconj;
    const ixmin = Math.trunc(xtor + 1.0e-12);
    const iymin = Math.trunc(ytor + 1.0e-12);
    const izmin = Math.trunc(zcon + 1.0e-12);
    torindx = torindx - 1;
    const out = this.out;
    if (Math.abs(ixmin - xtor) > 1.0e-8 || Math.abs(iymin - ytor) > 1.0e-8 || Math.abs(izmin - zcon) > 1.0e-8) {
      this.torInt(torindx, xtor, ytor, zcon, ixmin, iymin, izmin);
    } else {
      const at = gridIndex(torindx, ixmin, iymin, izmin - 1);
      out[0] = lib.torGrid[at];
      out[1] = lib.torGridx[at];
      out[2] = lib.torGridy[at];
      out[3] = lib.torGridz[at];
    }
    btor[0] = out[0];
    btor[1] = out[1];
    btor[2] = out[2];
    btor[3] = out[3];
  }

  /** tor_int: tricubic torsion spline into this.out. */
  private torInt(torindx: number, xtor: number, ytor: number, zcon: number, l0: number, m0: number, n0: number): void {
    const lib = this.lib();
    const out = this.out;
    out.fill(0);
    let l = l0, m = m0, n = n0;
    if (l >= lib.maxxc - 1) l = lib.maxxc - 1;
    if (m >= lib.maxyc - 1) m = lib.maxyc - 1;
    if (n >= lib.maxconj - 1) n = lib.maxconj - 1;
    this.spline(lib.torSpl, splIndex(torindx, l, m, n - 1), xtor, ytor, zcon, out);
  }

  /**
   * tor_force: forces of the torsion term on atoms i, j, k, l (fiTor ... flTor) for the i-j-k-l dihedral; uses
   * ptorr from force_zeta of the same pair.
   */
  torForce(torindx: number, pk: Comb3Param, pl: Comb3Param, srmu: number, rsq1: number, rsq2: number, rsq3: number,
    delrj: Float64Array, delrk: Float64Array, delrl: Float64Array): void {
    const kern = this.kern;
    this.fiTor.fill(0);
    this.fjTor.fill(0);
    this.fkTor.fill(0);
    this.flTor.fill(0);
    const rij = Math.sqrt(rsq1), rik = Math.sqrt(rsq2), rjl = Math.sqrt(rsq3);
    const rmu = (delrj[0] * delrk[0] + delrj[1] * delrk[1] + delrj[2] * delrk[2]) / (rij * rik);
    const rmul = -(delrj[0] * delrl[0] + delrj[1] * delrl[1] + delrj[2] * delrl[2]) / (rij * rjl);
    let srmul = Math.sqrt(1.0 - rmul * rmul);
    if (Math.acos(rmul) > Math.PI) srmul = -srmul;
    if (!(srmul > 0.1)) return;
    const fc1k = kern.fc(rik, pk);
    const fcp1k = kern.fcD(rik, pk);
    const fc1l = kern.fc(rjl, pl);
    const fcp1l = kern.fcD(rjl, pl);
    const srmul2 = srmul * srmul;
    const TT1 = rik * rjl * rij * rij * srmu * srmul;
    const dt1dik = -rmu / (srmu * srmu);
    const dt1djl = -rmul / srmul2;
    const tork = cross(delrk, delrj);
    const torl = cross(delrj, delrl);
    const TT2 = tork[0] * torl[0] + tork[1] * torl[1] + tork[2] * torl[2];
    const dt2dik = [
      -delrj[1] * torl[2] + delrj[2] * torl[1],
      -delrj[2] * torl[0] + delrj[0] * torl[2],
      -delrj[0] * torl[1] + delrj[1] * torl[0],
    ];
    const dt2djl = [
      delrj[1] * tork[2] - delrj[2] * tork[1],
      delrj[2] * tork[0] - delrj[0] * tork[2],
      delrj[0] * tork[1] - delrj[1] * tork[0],
    ];
    const dt2dij = [
      -delrk[2] * torl[1] + delrl[2] * tork[1] + delrk[1] * torl[2] - delrl[1] * tork[2],
      -delrk[0] * torl[2] + delrl[0] * tork[2] + delrk[2] * torl[0] - delrl[2] * tork[0],
      -delrk[1] * torl[0] + delrl[1] * tork[0] + delrk[0] * torl[1] - delrl[0] * tork[1],
    ];
    const ptorr = this.ptorr;
    const rmut = TT2 / TT1;
    let btt: number, AA: number;
    if (torindx >= 1) {
      btt = 1.0 - rmut * rmut;
      AA = -2.0 * ptorr * rmut * fc1k * fc1l / TT1;
    } else {
      btt = pk.ptork1 - rmut;
      btt = pk.ptork2 * btt * btt;
      AA = -2.0 * ptorr * pk.ptork2 * (pk.ptork1 - rmut) * fc1k * fc1l / TT1;
    }
    const AA2 = AA * TT2;
    const tfij0 = -(dt1dik * AA2) / rij / rik;
    const tfij1 = AA2 / rij / rij - dt1dik * AA2 * rmu / rij / rij;
    const tfij2 = -dt1djl * AA2 / rij / rjl;
    const tfij3 = AA2 / rij / rij - dt1djl * AA2 * rmul / rij / rij;
    const tfik0 = tfij0;
    const tfik1 = (AA2 / rik - btt * ptorr * fc1l * fcp1k) / rik - dt1dik * AA2 * rmu / rik / rik;
    const tfjl0 = tfij2;
    const tfjl1 = (AA2 / rjl - btt * ptorr * fc1k * fcp1l) / rjl - dt1djl * AA2 * rmul / rjl / rjl;
    for (let c = 0; c < 3; c++) {
      const tj0 = tfij0 * delrk[c] - tfij1 * delrj[c];
      const tj1 = -tfij2 * delrl[c] - tfij3 * delrj[c];
      const tj2 = -dt2dij[c] * AA;
      const tk0 = tfik0 * delrj[c] - tfik1 * delrk[c];
      const tk1 = -dt2dik[c] * AA;
      const tl0 = -tfjl0 * delrj[c] - tfjl1 * delrl[c];
      const tl1 = -dt2djl[c] * AA;
      this.fiTor[c] = tj0 + tj1 + tj2 + tk0 + tk1;
      this.fjTor[c] = -tj0 - tj1 - tj2 + tl0 + tl1;
      this.fkTor[c] = -tk0 - tk1;
      this.flTor[c] = -tl0 - tl1;
    }
  }
}

/** The cross product a x b in the order of pair_comb3.cpp (tork = delrk x delrj, torl = delrj x delrl). */
const cross = (a: Float64Array, b: Float64Array): number[] => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
