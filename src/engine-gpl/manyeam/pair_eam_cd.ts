/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_eam_cd.cpp and
 * pair_eam_cd.h (PairEAMCD, PairEAMCD_OneSite = eam/cd, PairEAMCD_TwoSite = eam/cd/old, derived from PairEAMAlloy).
 * Contributing author: Tim Lau (MIT) and the cd-EAM authors named in that source. Copyright (2003) Sandia Corporation
 * (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * pair_style eam/cd (cdeamVersion 1) and eam/cd/old (cdeamVersion 2): concentration-dependent EAM of Stukowski et al.,
 * with the h(x) polynomial read from the last line of the setfl file.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { readHCoeff, readSetfl, type EamFile } from './potential';
import { EamFamilyBase, forwardCopy, reverseAdd, tallyPair } from './pair_base';

export class PairEAMCD extends EamFamilyBase {
  readonly name: string;
  protected readonly fsLayout = false;
  private readonly cdeamVersion: 1 | 2;
  private speciesA = -1;
  private speciesB = -1;
  private hcoeff: Float64Array = new Float64Array(0);
  private rhoB = new Float64Array(0);
  private dv = new Float64Array(0);
  /** Table index of the last radiusIndex/rhoIndex call (native EAMTableIndex m and p). */
  private im = 0;
  private ip = 0;
  /** phid of the last PhiOfR call with a derivative (native phid by reference). */
  private phid = 0;

  constructor(name: 'eam/cd' | 'eam/cd/old') {
    super();
    this.name = name;
    this.cdeamVersion = name === 'eam/cd' ? 1 : 2;
  }

  protected readFile(text: string, filename: string, ctx: StyleContext): EamFile {
    return readSetfl(text, filename, ctx);
  }

  coeff(args: string[], ctx: StyleContext): void {
    const file = this.mapCoeff(args, ctx, this.name);
    if (file.nelements < 2) {
      throw new StyleError('The EAM file must contain at least 2 elements to be used with the eam/cd pair style.');
    }
    this.hcoeff = readHCoeff(ctx.readFile(args[2]), args[2]);
    this.speciesA = -1;
    this.speciesB = -1;
    for (let i = 1; i <= this.ntypes; i++) {
      if (this.map[i] === 0) {
        if (this.speciesA >= 0) throw new StyleError('The first element from the EAM file may only be mapped to a single atom type.');
        this.speciesA = i;
      }
      if (this.map[i] === 1) {
        if (this.speciesB >= 0) throw new StyleError('The second element from the EAM file may only be mapped to a single atom type.');
        this.speciesB = i;
      }
    }
    if (this.speciesA < 0) throw new StyleError('The first element from the EAM file must be mapped to exactly one atom type.');
    if (this.speciesB < 0) throw new StyleError('The second element from the EAM file must be mapped to exactly one atom type.');
  }

  /** evalH(x) of pair_eam_cd.h: Horner evaluation of the h(x) polynomial. */
  private evalH(x: number): number {
    const h = this.hcoeff;
    let v = 0.0;
    for (let i = h.length - 1; i >= 1; i--) v = (v + h[i]) * x;
    return v + h[0];
  }

  /** evalHprime(x) of pair_eam_cd.h. */
  private evalHprime(x: number): number {
    const h = this.hcoeff;
    let v = 0.0;
    for (let i = h.length - 1; i >= 2; i--) v = (v + i * h[i]) * x;
    return v + h[1];
  }

  /** radiusToTableIndex: sets im and ip from r (note the clamp to [1, nr-1], as pair_eam_cd.h). */
  private radiusIndex(r: number): void {
    const T = this.tables;
    let p = r * T.rdr + 1.0;
    let m = Math.trunc(p);
    m = m <= T.nr - 1 ? m : T.nr - 1;
    m = m > 1 ? m : 1;
    p -= m;
    p = p <= 1.0 ? p : 1.0;
    this.im = m;
    this.ip = p;
  }

  /** rhoToTableIndex: sets im and ip from rho. */
  private rhoIndex(rho: number): void {
    const T = this.tables;
    let p = rho * T.rdrho + 1.0;
    let m = Math.trunc(p);
    m = m <= T.nrho - 1 ? m : T.nrho - 1;
    m = m > 1 ? m : 1;
    p -= m;
    p = p <= 1.0 ? p : 1.0;
    this.im = m;
    this.ip = p;
  }

  /** RhoOfR(index, itype, jtype). */
  private rhoOfR(itype: number, jtype: number): number {
    const T = this.tables;
    const c = T.rhor[T.type2rhor[itype * (this.ntypes + 1) + jtype]];
    const o = this.im * 7;
    const p = this.ip;
    return ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
  }

  /** RhoPrimeOfR(index, itype, jtype). */
  private rhoPrimeOfR(itype: number, jtype: number): number {
    const T = this.tables;
    const c = T.rhor[T.type2rhor[itype * (this.ntypes + 1) + jtype]];
    const o = this.im * 7;
    const p = this.ip;
    return (c[o] * p + c[o + 1]) * p + c[o + 2];
  }

  /** FPrimeOfRho(index, itype). */
  private fPrimeOfRho(itype: number): number {
    const T = this.tables;
    const c = T.frho[T.type2frho[itype]];
    const o = this.im * 7;
    const p = this.ip;
    return (c[o] * p + c[o + 1]) * p + c[o + 2];
  }

  /** FofRho(index, itype). */
  private fOfRho(itype: number): number {
    const T = this.tables;
    const c = T.frho[T.type2frho[itype]];
    const o = this.im * 7;
    const p = this.ip;
    return ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
  }

  /** PhiOfR(index, itype, jtype, oneOverR): phi = z2 / r. */
  private phiOfR(itype: number, jtype: number, oneOverR: number): number {
    const T = this.tables;
    const c = T.z2r[T.type2z2r[itype * (this.ntypes + 1) + jtype]];
    const o = this.im * 7;
    const p = this.ip;
    const z2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
    return z2 * oneOverR;
  }

  /** PhiOfR(index, itype, jtype, oneOverR, phid): returns phi and sets this.phid = z2' / r - phi / r. */
  private phiOfRWithDeriv(itype: number, jtype: number, oneOverR: number): number {
    const T = this.tables;
    const c = T.z2r[T.type2z2r[itype * (this.ntypes + 1) + jtype]];
    const o = this.im * 7;
    const p = this.ip;
    const z2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
    const z2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
    const phi = z2 * oneOverR;
    this.phid = z2p * oneOverR - phi * oneOverR;
    return phi;
  }

  compute(pc: PairCompute): void {
    const half = pc.half;
    if (!half) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const { x, f, type, nlocal, nall } = pc;
    const cutsq = this.cutforcesq;
    const owner = pc.nb.owner;
    const v1 = this.cdeamVersion === 1;
    this.ensureAtomArrays(nall);
    if (this.rhoB.length !== nall) {
      this.rhoB = new Float64Array(nall);
      this.dv = new Float64Array(nall);
    }
    const rho = this.rho, fp = this.fp, rhoB = this.rhoB, dv = this.dv;
    const A = this.speciesA, B = this.speciesB;
    rho.fill(0);
    rhoB.fill(0);
    dv.fill(0);

    // first pass over the half list: rho, rhoB (and D for eam/cd, the Phi_AB sums) on owned and ghost atoms
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
      const k0: number = half.firstneigh[i];
      const k1: number = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          const jtype = type[j];
          const r = Math.sqrt(rsq);
          this.radiusIndex(r);
          let localrho = this.rhoOfR(jtype, itype);
          rho[i] += localrho;
          if (jtype === B) rhoB[i] += localrho;
          localrho = this.rhoOfR(itype, jtype);
          rho[j] += localrho;
          if (itype === B) rhoB[j] += localrho;
          if (v1 && itype !== jtype) {
            if ((itype === A && jtype === B) || (jtype === A && itype === B)) {
              const phiAB = this.phiOfR(itype, jtype, 1.0 / r);
              dv[i] += phiAB;
              dv[j] += phiAB;
            }
          }
        }
      }
    }
    reverseAdd(owner, nlocal, nall, rho, 1);
    reverseAdd(owner, nlocal, nall, rhoB, 1);
    if (v1) reverseAdd(owner, nlocal, nall, dv, 1);

    // embedding: fp = F'(rho) and the energy F(rho) of each owned atom
    const eatom = pc.eatom;
    for (let i = 0; i < nlocal; i++) {
      this.rhoIndex(rho[i]);
      fp[i] = this.fPrimeOfRho(type[i]);
      const phi = this.fOfRho(type[i]);
      pc.acc.evdwl += phi;
      if (eatom) eatom[i] += phi;
    }
    forwardCopy(owner, nlocal, nall, fp, 1);
    forwardCopy(owner, nlocal, nall, rho, 1);
    forwardCopy(owner, nlocal, nall, rhoB, 1);
    if (v1) forwardCopy(owner, nlocal, nall, dv, 1);

    if (!v1) {
      // eam/cd/old: the D sums need the forward-communicated rho and rhoB of the neighbours
      for (let i = 0; i < nlocal; i++) {
        const itype = type[i];
        if (itype !== A && itype !== B) continue;
        const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
        const xI = rhoB[i] / rho[i];
        const k0: number = half.firstneigh[i];
      const k1: number = k0 + half.numneigh[i];
        for (let k = k0; k < k1; k++) {
          const j = half.neighbors[k] & NEIGHMASK;
          const jtype = type[j];
          if (itype === jtype) continue;
          if (jtype !== A && jtype !== B) continue;
          const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
          const rsq = delx * delx + dely * dely + delz * delz;
          if (rsq < cutsq) {
            const r = Math.sqrt(rsq);
            this.radiusIndex(r);
            const phiAB = this.phiOfR(itype, jtype, 1.0 / r);
            const xij = 0.5 * (xI + rhoB[j] / rho[j]);
            const hPrime = this.evalHprime(xij);
            dv[i] += hPrime * phiAB / (2.0 * rho[i] * rho[i]);
            dv[j] += hPrime * phiAB / (2.0 * rho[j] * rho[j]);
          }
        }
      }
      reverseAdd(owner, nlocal, nall, dv, 1);
      forwardCopy(owner, nlocal, nall, dv, 1);
    }

    // forces over the half list
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
      let xI = -1.0;
      let DI = 0.0;
      if ((itype === A || itype === B) && rho[i] !== 0.0) {
        xI = rhoB[i] / rho[i];
        if (v1) {
          const hPrimeI = this.evalHprime(xI);
          DI = dv[i] * hPrimeI / (2.0 * rho[i] * rho[i]);
        } else {
          DI = dv[i];
        }
      }
      const k0: number = half.firstneigh[i];
      const k1: number = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          const jtype = type[j];
          const r = Math.sqrt(rsq);
          this.radiusIndex(r);
          const rhoip = this.rhoPrimeOfR(itype, jtype);
          const rhojp = this.rhoPrimeOfR(jtype, itype);
          let fpair = fp[i] * rhojp + fp[j] * rhoip;
          const recip = 1.0 / r;
          let xJ = -1;
          if ((jtype === A || jtype === B) && rho[j] !== 0.0) {
            xJ = rhoB[j] / rho[j];
            let DJ = 0.0;
            if (v1) {
              const hPrimeJ = this.evalHprime(xJ);
              DJ = dv[j] * hPrimeJ / (2.0 * rho[j] * rho[j]);
            } else {
              DJ = dv[j];
            }
            let t2 = -rhoB[j];
            if (itype === B) t2 += rho[j];
            fpair += DJ * rhoip * t2;
          }
          if (xI !== -1.0) {
            let t1 = -rhoB[i];
            if (jtype === B) t1 += rho[i];
            fpair += DI * rhojp * t1;
          }
          let phi = this.phiOfRWithDeriv(itype, jtype, recip);
          const phip = this.phid;
          if (itype === jtype || xI === -1.0 || xJ === -1.0) {
            fpair += phip;
          } else {
            let h = 0.0;
            if (v1) {
              const hI = this.evalH(xI);
              const hJ = this.evalH(xJ);
              h = 0.5 * (hI + hJ);
            } else {
              const xij = 0.5 * (xI + xJ);
              h = this.evalH(xij);
            }
            fpair += h * phip;
            phi *= h;
          }
          fpair /= -r;
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          tallyPair(pc, i, j, phi, fpair, delx, dely, delz);
        }
      }
    }
  }
}
