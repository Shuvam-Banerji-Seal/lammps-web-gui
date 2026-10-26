import { Pair, StyleError, type PairCompute } from '../types';
import { NEIGHMASK } from '../../neighbor';

/*
 * The two loops of pair eam, eam/alloy and eam/fs on a shared table grid (eam.ts), kept apart from the potential files
 * and splines so the pair worker (cpu/threadedPairs.ts) holds only these loops. As native LAMMPS does
 * (src/MANYBODY/pair_eam.cpp compute at b332f28): one table index per pair, p = r rdr + 1, m = (int) p clamped to
 * nr - 1, p -= m clamped to 1, and the seven coefficients of the interval (eam.ts Spline.coef: the derivative's
 * 3 c3/dr, 2 c2/dr and s/dr, then c3, c2, s and y).
 */

/** A table as the loops read it (eam.ts Spline). */
export interface EAMKernelSpline { readonly n: number; readonly rdx: number; readonly coef: Float64Array }

/** The density and r*phi tables by type pair (index i * (ntypes + 1) + j) and the squared cutoffs. */
export interface EAMKernelTables {
  readonly ntypes: number;
  readonly cutsq: Float64Array;
  readonly rhoTab: readonly (EAMKernelSpline | null)[];
  readonly gTab: readonly (EAMKernelSpline | null)[];
}

const NONE = new Float64Array(0);

/**
 * The grid of the density and r*phi tables (eam.ts puts every table of a style on one grid, as native does), or null when
 * the style has no tables (types without a table pair, e.g. NULL elements under pair hybrid, have no list entries).
 */
export const eamGrid = (t: EAMKernelTables): { nr: number; rdr: number } | null => {
  for (const sp of t.rhoTab) if (sp) return { nr: sp.n, rdr: sp.rdx };
  return null;
};

const coefs = (tabs: readonly (EAMKernelSpline | null)[]): Float64Array[] => tabs.map((sp) => sp?.coef ?? NONE);

/**
 * The densities the half-list rows [ilo, inum) add (native's first loop): each pair within the cutoff adds to both of
 * its atoms, owned or ghost, in rho (the caller zeroes it and sums the ghosts to their owners).
 */
export const eamDensities = (t: EAMKernelTables, name: string, pc: PairCompute, rho: Float64Array): void => {
  const half = pc.half;
  if (!half) throw new Error(`pair style ${name} needs a half neighbor list`);
  const grid = eamGrid(t);
  if (!grid) return;
  const { nr, rdr } = grid;
  const { x, type } = pc;
  const nt = t.ntypes + 1;
  const cutsq = t.cutsq;
  const rhoC = coefs(t.rhoTab);
  const first = half.firstneigh, num = half.numneigh, nbr = half.neighbors;
  for (let i = half.ilo ?? 0; i < half.inum; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const itype = type[i];
    const ti = itype * nt;
    let rhotmp = rho[i];
    const k1 = first[i] + num[i];
    for (let k = first[i]; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const jtype = type[j];
      if (rsq >= cutsq[ti + jtype]) continue;
      let p = Math.sqrt(rsq) * rdr + 1.0;
      let m = Math.trunc(p);
      if (m > nr - 1) m = nr - 1;
      p -= m;
      if (p > 1.0) p = 1.0;
      const o = 7 * (m - 1);
      let c = rhoC[ti + jtype];
      rhotmp += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      c = rhoC[jtype * nt + itype];
      rho[j] += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
    }
    rho[i] = rhotmp;
  }
};

/** The forces and pair energies of the half-list rows [ilo, inum) (native's second loop), with F' of every atom in fp. */
export const eamForces = (t: EAMKernelTables, name: string, pc: PairCompute, fp: Float64Array): void => {
  const half = pc.half;
  if (!half) throw new Error(`pair style ${name} needs a half neighbor list`);
  const grid = eamGrid(t);
  if (!grid) return;
  const { nr, rdr } = grid;
  const { x, f, type } = pc;
  const nt = t.ntypes + 1;
  const cutsq = t.cutsq;
  const rhoC = coefs(t.rhoTab), z2C = coefs(t.gTab);
  const first = half.firstneigh, num = half.numneigh, nbr = half.neighbors;
  const eatom = pc.eatom, vatom = pc.vatom;
  let evdwl = 0;
  for (let i = half.ilo ?? 0; i < half.inum; i++) {
    const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
    const itype = type[i];
    const ti = itype * nt;
    let fxtmp = f[3 * i], fytmp = f[3 * i + 1], fztmp = f[3 * i + 2];
    const fptmp = fp[i];
    const k1 = first[i] + num[i];
    for (let k = first[i]; k < k1; k++) {
      const j = nbr[k] & NEIGHMASK;
      const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
      const rsq = delx * delx + dely * dely + delz * delz;
      const jtype = type[j];
      if (rsq >= cutsq[ti + jtype]) continue;
      const r = Math.sqrt(rsq);
      let p = r * rdr + 1.0;
      let m = Math.trunc(p);
      if (m > nr - 1) m = nr - 1;
      p -= m;
      if (p > 1.0) p = 1.0;
      const o = 7 * (m - 1);
      // rhoip: the derivative of the density at j due to i; rhojp: at i due to j
      let c = rhoC[jtype * nt + itype];
      const rhoip = (c[o] * p + c[o + 1]) * p + c[o + 2];
      c = rhoC[ti + jtype];
      const rhojp = (c[o] * p + c[o + 1]) * p + c[o + 2];
      c = z2C[ti + jtype];
      const z2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
      const z2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      const recip = 1.0 / r;
      const phi = z2 * recip;
      const phip = z2p * recip - phi * recip;
      const psip = fptmp * rhojp + fp[j] * rhoip + phip;
      const fpair = -psip * recip;
      fxtmp += delx * fpair;
      fytmp += dely * fpair;
      fztmp += delz * fpair;
      f[3 * j] -= delx * fpair;
      f[3 * j + 1] -= dely * fpair;
      f[3 * j + 2] -= delz * fpair;
      evdwl += phi;
      if (eatom) { eatom[i] += 0.5 * phi; eatom[j] += 0.5 * phi; }
      if (vatom) {
        const v0 = delx * delx * fpair, v1 = dely * dely * fpair, v2 = delz * delz * fpair;
        const v3 = delx * dely * fpair, v4 = delx * delz * fpair, v5 = dely * delz * fpair;
        vatom[6 * i] += 0.5 * v0; vatom[6 * i + 1] += 0.5 * v1; vatom[6 * i + 2] += 0.5 * v2;
        vatom[6 * i + 3] += 0.5 * v3; vatom[6 * i + 4] += 0.5 * v4; vatom[6 * i + 5] += 0.5 * v5;
        vatom[6 * j] += 0.5 * v0; vatom[6 * j + 1] += 0.5 * v1; vatom[6 * j + 2] += 0.5 * v2;
        vatom[6 * j + 3] += 0.5 * v3; vatom[6 * j + 4] += 0.5 * v4; vatom[6 * j + 5] += 0.5 * v5;
      }
    }
    f[3 * i] = fxtmp; f[3 * i + 1] = fytmp; f[3 * i + 2] = fztmp;
  }
  pc.acc.evdwl += evdwl;
};

/**
 * A force thread's copy of an EAM style (cpu/threadedPairs.ts): the engine's tables cloned into it, and only the two
 * loops the threads run (TwoPassPair densities and forces). The embedding step and every other use stay with the
 * engine's style (eam.ts).
 */
export class PairEAMKernel extends Pair implements EAMKernelTables {
  manybody = true;
  virialFdotr = true;
  rhoTab: (EAMKernelSpline | null)[] = [];
  gTab: (EAMKernelSpline | null)[] = [];

  constructor(readonly name: string) { super(); }

  twoPassReady(): boolean { return true; }
  densities(pc: PairCompute, rho: Float64Array): void { eamDensities(this, this.name, pc, rho); }
  forces(pc: PairCompute, fp: Float64Array): void { eamForces(this, this.name, pc, fp); }

  embed(): void { throw new StyleError(`pair style ${this.name}: the embedding step runs on the engine thread`); }
  compute(): void { throw new StyleError(`pair style ${this.name}: a force thread runs only the two loops`); }
  settings(): void { throw new StyleError(`pair style ${this.name}: a force thread's copy takes no settings`); }
  coeff(): void { throw new StyleError(`pair style ${this.name}: a force thread's copy takes no coefficients`); }
  initOne(): number { throw new StyleError(`pair style ${this.name}: a force thread's copy is not initialized`); }
}
