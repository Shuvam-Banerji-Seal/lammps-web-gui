/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_eam_he.cpp and
 * pair_eam_he.h (PairEAMHE, derived from PairEAMFS), with the he branch of src/MANYBODY/pair_eam.cpp (compute,
 * compute_embedding<1>, embedding_index<1>, single) and pair_eam.h. Contributing author of pair_eam_he: Xiaowang Zhou
 * (Sandia). Copyright (2003) Sandia Corporation (GNU General Public License). This file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * pair_style eam/he: Finnis-Sinclair tables whose embedding table starts at rhomin = rhomax - (Nrho-1) drho, with the
 * linear extension beyond both ends of the table.
 */
import { StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { readFs, type EamFile } from './potential';
import { EamFamilyBase, forwardCopy, reverseAdd, tallyPair } from './pair_base';

export class PairEAMHE extends EamFamilyBase {
  readonly name: string = 'eam/he';
  protected readonly fsLayout = true;
  private rhomin = 0;
  private rhomax = 0;

  protected readFile(text: string, filename: string, ctx: StyleContext): EamFile {
    return readFs(text, filename, ctx, true);
  }

  coeff(args: string[], ctx: StyleContext): void {
    // native PairEAM::coeff_mapped does not inspect the first two words
    this.mapCoeff(args, ctx, 'eam/he');
  }

  initStyle(ctx: StyleContext): void {
    super.initStyle(ctx);
    const f = this.file as EamFile;
    // native file2array_fs: "if (he_flag) rhomin = rhomax - (nrho - 1) * drho;" with rhomax the file's grid value
    this.rhomax = f.rhomax;
    this.rhomin = this.rhomax - (f.nrho - 1) * f.drho;
  }

  /** embedding_index<1>(rho, m, p): the table index and fraction for the he embedding table. */
  private embeddingIndex(rhoi: number): { m: number; p: number } {
    const T = this.tables;
    let p = (rhoi - this.rhomin) * T.rdrho + 1.0;
    let m = Math.trunc(p);
    if (m < 2) m = 2;
    else if (m > T.nrho - 1) m = T.nrho - 1;
    p -= m;
    if (p < -1.0) p = -1.0;
    else if (p > 1.0) p = 1.0;
    return { m, p };
  }

  compute(pc: PairCompute): void {
    const half = pc.half;
    if (!half) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const { x, f, type, nlocal, nall } = pc;
    const T = this.tables;
    const nt = this.ntypes + 1;
    const cutsq = this.cutforcesq;
    this.ensureAtomArrays(nall);
    if (this.numforce.length !== nlocal) this.numforce = new Int32Array(nlocal);
    const rho = this.rho, fp = this.fp;
    rho.fill(0);

    // rho = density at each atom: pass over the half list, the density of j at i and of i at j (newton on)
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
      let rhotmp = rho[i];
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          const jtype = type[j];
          let p = Math.sqrt(rsq) * T.rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, T.nr - 1);
          p -= m;
          p = Math.min(p, 1.0);
          let c = T.rhor[T.type2rhor[jtype * nt + itype]];
          let o = m * 7;
          rhotmp += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          c = T.rhor[T.type2rhor[itype * nt + jtype]];
          rho[j] += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
        }
      }
      rho[i] = rhotmp;
    }
    reverseAdd(pc.nb.owner, nlocal, nall, rho, 1);

    // compute_embedding<1>: fp = F'(rho), the energy F(rho) with the linear extension past rhomin and rhomax
    const eatom = pc.eatom;
    for (let i = 0; i < nlocal; i++) {
      const { m, p } = this.embeddingIndex(rho[i]);
      const c = T.frho[T.type2frho[type[i]]];
      const o = m * 7;
      fp[i] = (c[o] * p + c[o + 1]) * p + c[o + 2];
      let phi = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      if (rho[i] < this.rhomin) phi += fp[i] * (rho[i] - this.rhomin);
      else if (rho[i] > this.rhomax) phi += fp[i] * (rho[i] - this.rhomax);
      pc.acc.evdwl += phi;
      if (eatom) eatom[i] += phi;
    }
    forwardCopy(pc.nb.owner, nlocal, nall, fp, 1);

    // forces over the half list
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
      const fpi = fp[i];
      let fxtmp = f[3 * i], fytmp = f[3 * i + 1], fztmp = f[3 * i + 2];
      let nforce = 0;
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          ++nforce;
          const jtype = type[j];
          const r = Math.sqrt(rsq);
          let p = r * T.rdr + 1.0;
          let m = Math.trunc(p);
          m = Math.min(m, T.nr - 1);
          p -= m;
          p = Math.min(p, 1.0);
          const o = m * 7;
          let c = T.rhor[T.type2rhor[itype * nt + jtype]];
          const rhoip = (c[o] * p + c[o + 1]) * p + c[o + 2];
          c = T.rhor[T.type2rhor[jtype * nt + itype]];
          const rhojp = (c[o] * p + c[o + 1]) * p + c[o + 2];
          c = T.z2r[T.type2z2r[itype * nt + jtype]];
          const z2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
          const z2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          const recip = 1.0 / r;
          const phi = z2 * recip;
          const phip = z2p * recip - phi * recip;
          const psip = fpi * rhojp + fp[j] * rhoip + phip;
          const fpair = -psip * recip;
          fxtmp += delx * fpair;
          fytmp += dely * fpair;
          fztmp += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
          tallyPair(pc, i, j, phi, fpair, delx, dely, delz);
        }
      }
      this.numforce[i] = nforce;
      f[3 * i] = fxtmp;
      f[3 * i + 1] = fytmp;
      f[3 * i + 2] = fztmp;
    }
    this.computed = true;
  }

  /** Pair term only plus the embedding term at i and j (native single(), pair_eam.cpp). */
  single(i: number, j: number, itype: number, jtype: number, rsq: number): { eng: number; fforce: number } {
    if (!this.computed) throw new StyleError('EAM embedding data required for this calculation is missing');
    const T = this.tables;
    const nt = this.ntypes + 1;
    let phi: number;
    if (this.numforce[i] > 0) {
      const { m, p } = this.embeddingIndex(this.rho[i]);
      const c = T.frho[T.type2frho[itype]];
      const o = m * 7;
      phi = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      if (this.rho[i] < this.rhomin) phi += this.fp[i] * (this.rho[i] - this.rhomin);
      else if (this.rho[i] > this.rhomax) phi += this.fp[i] * (this.rho[i] - this.rhomax);
      phi *= 1.0 / this.numforce[i];
    } else {
      phi = 0.0;
    }
    const r = Math.sqrt(rsq);
    let p = r * T.rdr + 1.0;
    let m = Math.trunc(p);
    m = Math.min(m, T.nr - 1);
    p -= m;
    p = Math.min(p, 1.0);
    const o = m * 7;
    let c = T.rhor[T.type2rhor[itype * nt + jtype]];
    const rhoip = (c[o] * p + c[o + 1]) * p + c[o + 2];
    c = T.rhor[T.type2rhor[jtype * nt + itype]];
    const rhojp = (c[o] * p + c[o + 1]) * p + c[o + 2];
    c = T.z2r[T.type2z2r[itype * nt + jtype]];
    const z2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
    const z2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
    const recip = 1.0 / r;
    phi += z2 * recip;
    const phip = z2p * recip - phi * recip;
    const psip = this.fp[i] * rhojp + this.fp[j] * rhoip + phip;
    return { eng: phi, fforce: -psip * recip };
  }
}
