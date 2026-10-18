/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MANYBODY/pair_adp.cpp and
 * pair_adp.h (PairADP, pair_style adp: angular-dependent potential of Mishin et al.). Contributing author: Ray Shan
 * (Sandia) and the authors named in that source. Copyright (2003) Sandia Corporation (GNU General Public License). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The density rho, the dipole vector mu and the quadrupole tensor lambda (six components xx, yy, zz, yz, xz, xy) are summed
 * over the half list, reverse-communicated to the owners, and the embedding derivative fp and the mu and lambda of the
 * owners are forward-communicated to the ghosts before the force loop.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { buildTables, readAdp, type EamFile, type EamTables } from './potential';
import { forwardCopy, reverseAdd } from './pair_base';

export class PairADP extends Pair {
  readonly name = 'adp';
  manybody = true;
  virialFdotr = true;
  private map = new Int32Array(0);
  private file: EamFile | null = null;
  private tabs: EamTables | null = null;
  private cutmax = 0;
  private cutforcesq = 0;
  private readonly cache = new Map<string, EamFile>();
  private rho = new Float64Array(0);
  private fp = new Float64Array(0);
  private mu = new Float64Array(0);
  private lam = new Float64Array(0);

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.map = new Int32Array(ntypes + 1).fill(-1);
    this.file = null;
    this.tabs = null;
  }

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  /** native PairADP::coeff: "pair_coeff * * file El1 ... ElN", one element or NULL per type. */
  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes || args[0] !== '*' || args[1] !== '*') {
      throw new StyleError('Incorrect args for pair coefficients');
    }
    const filename = args[2];
    let file = this.cache.get(filename);
    if (!file) {
      file = readAdp(ctx.readFile(filename), filename, ctx);
      this.cache.set(filename, file);
    }
    this.file = file;
    this.map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let k = 3; k < args.length; k++) {
      if (args[k] === 'NULL') continue;
      const e = file.names.indexOf(args[k]);
      if (e < 0) throw new StyleError('No matching element in ADP potential file');
      this.map[k - 2] = e;
    }
    let count = 0;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        if (this.map[i] >= 0 && this.map[j] >= 0) {
          count++;
          if (i === j) this.setMass(ctx, i, file.mass[this.map[i]]);
        }
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  /** atom->set_mass (atom.cpp): a value <= 0 is ignored with a warning. */
  private setMass(ctx: StyleContext, type: number, value: number): void {
    const s = ctx.s;
    if (!s) return;
    if (s.rmass) throw new StyleError('Cannot set per-type atom mass for this atom style');
    if (!(value > 0)) {
      ctx.log(`WARNING: Ignoring invalid mass value ${value} for atom type ${type}`);
      return;
    }
    s.massByType[type] = value;
  }

  initStyle(ctx: StyleContext): void {
    this.noTermLog(ctx, 'shift');
    this.noTermLog(ctx, 'tail');
    if (this.table !== 12) throw new StyleError(`pair_modify table is not supported for pair style ${this.name}`);
    if (!this.file) throw new StyleError('All pair coeffs are not set');
    this.tabs = buildTables(this.file, this.map, this.ntypes, false);
    this.cutmax = this.file.cut;
    this.cutforcesq = this.cutmax * this.cutmax;
  }

  initOne(i: number, j: number): number {
    if (!this.file) throw new StyleError('All pair coeffs are not set');
    if (this.map[i] < 0 || this.map[j] < 0) {
      if (i === j && !this.hybridSub) throw new StyleError('All pair coeffs are not set');
      return 0;
    }
    return this.cutmax;
  }

  compute(pc: PairCompute): void {
    const half = pc.half;
    if (!half) throw new Error(`pair style ${this.name} needs a half neighbor list`);
    const { x, f, type, nlocal, nall } = pc;
    const T = this.tables();
    const nt = this.ntypes + 1;
    const cutsq = this.cutforcesq;
    const owner = pc.nb.owner;
    if (this.rho.length !== nall) {
      this.rho = new Float64Array(nall);
      this.fp = new Float64Array(nall);
      this.mu = new Float64Array(3 * nall);
      this.lam = new Float64Array(6 * nall);
    }
    const rho = this.rho, fp = this.fp, mu = this.mu, lam = this.lam;
    rho.fill(0);
    mu.fill(0);
    lam.fill(0);

    // pass over the half list: rho, mu and lambda of i and (newton on) of j
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
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
          const o = m * 7;
          let c = T.rhor[T.type2rhor[jtype * nt + itype]];
          rho[i] += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          c = T.u2r[T.type2u2r[jtype * nt + itype]];
          let u2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          mu[3 * i] += u2 * delx;
          mu[3 * i + 1] += u2 * dely;
          mu[3 * i + 2] += u2 * delz;
          c = T.w2r[T.type2w2r[jtype * nt + itype]];
          let w2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          lam[6 * i] += w2 * delx * delx;
          lam[6 * i + 1] += w2 * dely * dely;
          lam[6 * i + 2] += w2 * delz * delz;
          lam[6 * i + 3] += w2 * dely * delz;
          lam[6 * i + 4] += w2 * delx * delz;
          lam[6 * i + 5] += w2 * delx * dely;

          c = T.rhor[T.type2rhor[itype * nt + jtype]];
          rho[j] += ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          c = T.u2r[T.type2u2r[itype * nt + jtype]];
          u2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          mu[3 * j] -= u2 * delx;
          mu[3 * j + 1] -= u2 * dely;
          mu[3 * j + 2] -= u2 * delz;
          c = T.w2r[T.type2w2r[itype * nt + jtype]];
          w2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          lam[6 * j] += w2 * delx * delx;
          lam[6 * j + 1] += w2 * dely * dely;
          lam[6 * j + 2] += w2 * delz * delz;
          lam[6 * j + 3] += w2 * dely * delz;
          lam[6 * j + 4] += w2 * delx * delz;
          lam[6 * j + 5] += w2 * delx * dely;
        }
      }
    }
    reverseAdd(owner, nlocal, nall, rho, 1);
    reverseAdd(owner, nlocal, nall, mu, 3);
    reverseAdd(owner, nlocal, nall, lam, 6);

    // embedding: fp = F'(rho), and the energy with the mu and lambda terms of each owned atom
    const eatom = pc.eatom;
    for (let i = 0; i < nlocal; i++) {
      let p = rho[i] * T.rdrho + 1.0;
      let m = Math.trunc(p);
      m = Math.max(1, Math.min(m, T.nrho - 1));
      p -= m;
      p = Math.min(p, 1.0);
      const c = T.frho[T.type2frho[type[i]]];
      const o = m * 7;
      fp[i] = (c[o] * p + c[o + 1]) * p + c[o + 2];
      let phi = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
      const l = 6 * i, q = 3 * i;
      phi += 0.5 * (mu[q] * mu[q] + mu[q + 1] * mu[q + 1] + mu[q + 2] * mu[q + 2]);
      phi += 0.5 * (lam[l] * lam[l] + lam[l + 1] * lam[l + 1] + lam[l + 2] * lam[l + 2]);
      phi += 1.0 * (lam[l + 3] * lam[l + 3] + lam[l + 4] * lam[l + 4] + lam[l + 5] * lam[l + 5]);
      phi -= 1.0 / 6.0 * (lam[l] + lam[l + 1] + lam[l + 2]) * (lam[l] + lam[l + 1] + lam[l + 2]);
      pc.acc.evdwl += phi;
      if (eatom) eatom[i] += phi;
    }
    forwardCopy(owner, nlocal, nall, fp, 1);
    forwardCopy(owner, nlocal, nall, mu, 3);
    forwardCopy(owner, nlocal, nall, lam, 6);

    // forces over the half list
    for (let i = 0; i < nlocal; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const itype = type[i];
      const k0 = half.firstneigh[i], k1 = k0 + half.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = half.neighbors[k] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
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
          c = T.u2r[T.type2u2r[itype * nt + jtype]];
          const u2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
          const u2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          c = T.w2r[T.type2w2r[itype * nt + jtype]];
          const w2p = (c[o] * p + c[o + 1]) * p + c[o + 2];
          const w2 = ((c[o + 3] * p + c[o + 4]) * p + c[o + 5]) * p + c[o + 6];
          const recip = 1.0 / r;
          const phi = z2 * recip;
          const phip = z2p * recip - phi * recip;
          const psip = fp[i] * rhojp + fp[j] * rhoip + phip;
          const fpair = -psip * recip;
          const delmux = mu[3 * i] - mu[3 * j];
          const delmuy = mu[3 * i + 1] - mu[3 * j + 1];
          const delmuz = mu[3 * i + 2] - mu[3 * j + 2];
          const trdelmu = delmux * delx + delmuy * dely + delmuz * delz;
          const sumlamxx = lam[6 * i] + lam[6 * j];
          const sumlamyy = lam[6 * i + 1] + lam[6 * j + 1];
          const sumlamzz = lam[6 * i + 2] + lam[6 * j + 2];
          const sumlamyz = lam[6 * i + 3] + lam[6 * j + 3];
          const sumlamxz = lam[6 * i + 4] + lam[6 * j + 4];
          const sumlamxy = lam[6 * i + 5] + lam[6 * j + 5];
          const tradellam = sumlamxx * delx * delx + sumlamyy * dely * dely +
            sumlamzz * delz * delz + 2.0 * sumlamxy * delx * dely +
            2.0 * sumlamxz * delx * delz + 2.0 * sumlamyz * dely * delz;
          const nu = sumlamxx + sumlamyy + sumlamzz;
          let adpx = delmux * u2 + trdelmu * u2p * delx * recip +
            2.0 * w2 * (sumlamxx * delx + sumlamxy * dely + sumlamxz * delz) +
            w2p * delx * recip * tradellam - 1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * delx;
          let adpy = delmuy * u2 + trdelmu * u2p * dely * recip +
            2.0 * w2 * (sumlamxy * delx + sumlamyy * dely + sumlamyz * delz) +
            w2p * dely * recip * tradellam - 1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * dely;
          let adpz = delmuz * u2 + trdelmu * u2p * delz * recip +
            2.0 * w2 * (sumlamxz * delx + sumlamyz * dely + sumlamzz * delz) +
            w2p * delz * recip * tradellam - 1.0 / 3.0 * nu * (w2p * r + 2.0 * w2) * delz;
          adpx *= -1.0;
          adpy *= -1.0;
          adpz *= -1.0;
          const fx = delx * fpair + adpx;
          const fy = dely * fpair + adpy;
          const fz = delz * fpair + adpz;
          f[3 * i] += fx;
          f[3 * i + 1] += fy;
          f[3 * i + 2] += fz;
          f[3 * j] -= fx;
          f[3 * j + 1] -= fy;
          f[3 * j + 2] -= fz;
          tallyXyz(pc, i, j, phi, fx, fy, fz, delx, dely, delz);
        }
      }
    }
  }

  /** The table set of the last initStyle (pair_adp.cpp file2array + array2spline). */
  private tables(): EamTables {
    if (!this.tabs) throw new StyleError(`pair style ${this.name} is not initialised`);
    return this.tabs;
  }
}

/**
 * ev_tally_xyz (pair.cpp): the per-atom energy is half the pair energy on each atom and the per-atom virial is
 * half of (del * f) per component; the global virial comes from the fdotr sum.
 */
const tallyXyz = (pc: PairCompute, i: number, j: number, evdwl: number, fx: number, fy: number, fz: number, dx: number, dy: number, dz: number): void => {
  pc.acc.evdwl += evdwl;
  if (pc.eatom) {
    const e = 0.5 * evdwl;
    pc.eatom[i] += e;
    pc.eatom[j] += e;
  }
  if (pc.vatom) {
    const va = pc.vatom;
    const v0 = dx * fx, v1 = dy * fy, v2 = dz * fz, v3 = dx * fy, v4 = dx * fz, v5 = dy * fz;
    for (const k of [i, j]) {
      va[6 * k] += 0.5 * v0; va[6 * k + 1] += 0.5 * v1; va[6 * k + 2] += 0.5 * v2;
      va[6 * k + 3] += 0.5 * v3; va[6 * k + 4] += 0.5 * v4; va[6 * k + 5] += 0.5 * v5;
    }
  }
};
