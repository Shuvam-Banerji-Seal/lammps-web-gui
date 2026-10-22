/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/MANYBODY/pair_local_density.cpp and
 * pair_local_density.h (PairLocalDensity: pair_style local/density, pair_coeff * * <table>), including the
 * table reader, the clamped cubic spline of the tabulated embedding functions (interpolate_cbspl) and the
 * per-atom local densities. LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General
 * Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing authors (from
 * pair_local_density.cpp): Tanmoy Sanyal and M. Scott Shell (UC Santa Barbara), David Rosenberger (TU Darmstadt).
 *
 * Model (docs.lammps.org/pair_local_density.html): each atom i of a species with a potential gets the local density
 * rho_i = sum_j phi(r_ij) b_k(type_j), phi a smooth indicator that is 1 inside the lower cutoff, 0 beyond the upper
 * cutoff and a polynomial in r^2 between them; the energy is U_i = a_k(type_i) F_k(rho_i) with F_k a tabulated
 * function, extrapolated linearly outside the table. The pair force follows from dF/drho through the chain rule.
 *
 * Ghost atoms: the sums that native LAMMPS does on ghosts (reverse communication) and the forward copies of fp are
 * done through the owner index (nb.owner), which gives the same totals. The half neighbour list is used as the
 * source does (newton on).
 */
import { Pair, StyleError, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { tallyXYZ } from '../dpd/pair_dpd';

/** One tabulated local-density potential (k = 0 .. nLD-1). */
interface LDTable {
  nLD: number;
  nrho: number;
  /** Filter per k and atom type (1 / 0): a = central atoms with an embedding term, b = neighbours that contribute. */
  a: Uint8Array;
  b: Uint8Array;
  lower: Float64Array;
  upper: Float64Array;
  lowerSq: Float64Array;
  upperSq: Float64Array;
  c0: Float64Array;
  c2: Float64Array;
  c4: Float64Array;
  c6: Float64Array;
  rhoMin: Float64Array;
  rhoMax: Float64Array;
  delta: Float64Array;
  /** Clamped cubic spline coefficients: 7 per table point, index (k * nrho + m) * 7 + c. */
  spline: Float64Array;
  /** Largest upper cutoff over k (init_one). */
  cutmax: number;
}

/** interpolate_cbspl (pair_local_density.cpp): clamped cubic spline of f[0..n-1] (step delta) into 7 coefficients per point. */
const interpolateCbspl = (n: number, delta: number, f: Float64Array, fOff: number, out: Float64Array, oOff: number): void => {
  const coeffB = new Float64Array(n), coeffC = new Float64Array(n), coeffD = new Float64Array(n);
  const dl = new Float64Array(n), dd = new Float64Array(n), du = new Float64Array(n);
  const F = (i: number): number => f[fOff + i];
  const fpa = 0, fpb = 0;
  for (let i = 0; i < n - 1; i++) dl[i] = du[i] = delta;
  dd[0] = 2.0 * delta;
  dd[n - 1] = 2.0 * delta;
  coeffC[0] = (3.0 / delta) * (F(1) - F(0)) - 3.0 * fpa;
  coeffC[n - 1] = 3.0 * fpb - (3.0 / delta) * (F(n - 1) - F(n - 2));
  for (let i = 0; i < n - 2; i++) {
    dd[i + 1] = 4.0 * delta;
    coeffC[i + 1] = (3.0 / delta) * (F(i + 2) - F(i + 1)) - (3.0 / delta) * (F(i + 1) - F(i));
  }
  for (let i = 0; i < n - 1; i++) {
    du[i] /= dd[i];
    dd[i + 1] -= dl[i] * du[i];
  }
  coeffC[0] /= dd[0];
  for (let i = 1; i < n; i++) coeffC[i] = (coeffC[i] - dl[i - 1] * coeffC[i - 1]) / dd[i];
  for (let i = n - 2; i >= 0; i--) coeffC[i] -= coeffC[i + 1] * du[i];
  for (let i = 0; i < n - 1; i++) {
    coeffD[i] = (coeffC[i + 1] - coeffC[i]) / (3.0 * delta);
    coeffB[i] = (F(i + 1) - F(i)) / delta - delta * (coeffC[i + 1] + 2.0 * coeffC[i]) / 3.0;
  }
  for (let i = 0; i < n - 1; i++) {
    coeffB[i] = coeffB[i] * delta;
    coeffC[i] = coeffC[i] * delta * delta;
    coeffD[i] = coeffD[i] * delta * delta * delta;
  }
  for (let i = 0; i < n; i++) {
    const o = oOff + 7 * i;
    out[o + 3] = coeffD[i];
    out[o + 4] = coeffC[i];
    out[o + 5] = coeffB[i];
    out[o + 6] = F(i);
    out[o + 2] = out[o + 5] / delta;
    out[o + 1] = 2.0 * out[o + 4] / delta;
    out[o] = 3.0 * out[o + 3] / delta;
  }
};

/**
 * Reads the tabulated file (PairLocalDensity::parse_file). The file is a sequence of blank-separated fields: a header
 * "nLD nrho", then per potential: a line "lo up", a line of central types, a line of neighbour types, a line
 * "rho_min rho_max ..." (further values ignored), nrho values, and a blank line. Blank and comment lines are skipped.
 */
const parseLocalDensityFile = (text: string, ntypes: number): LDTable => {
  const rows: string[][] = [];
  for (const raw of text.split(/\r?\n/)) {
    const w = raw.replace(/#.*$/, '').trim();
    if (w !== '') rows.push(w.split(/\s+/));
  }
  let p = 0;
  const next = (): string[] => {
    if (p >= rows.length) throw new StyleError('Unexpected end of local/density table file');
    return rows[p++];
  };
  const head = next();
  if (head.length < 2) throw new StyleError('local/density table header must give nLD and nrho');
  const nLD = parseInt(head[0], 10), nrho = parseInt(head[1], 10);
  if (!(nLD > 0) || !(nrho > 1)) throw new StyleError('local/density table header must give nLD > 0 and nrho > 1');
  const nt = ntypes + 1;
  const t: LDTable = {
    nLD, nrho,
    a: new Uint8Array(nLD * nt), b: new Uint8Array(nLD * nt),
    lower: new Float64Array(nLD), upper: new Float64Array(nLD),
    lowerSq: new Float64Array(nLD), upperSq: new Float64Array(nLD),
    c0: new Float64Array(nLD), c2: new Float64Array(nLD), c4: new Float64Array(nLD), c6: new Float64Array(nLD),
    rhoMin: new Float64Array(nLD), rhoMax: new Float64Array(nLD), delta: new Float64Array(nLD),
    spline: new Float64Array(nLD * nrho * 7),
    cutmax: 0,
  };
  const ftmp = new Float64Array(nLD * nrho);
  const typeLine = (words: string[], dst: Uint8Array, k: number): void => {
    for (const w of words) {
      const at = parseInt(w, 10);
      if (!(at >= 1 && at <= ntypes)) throw new StyleError(`Invalid atom type filter value ${w} in local/density table`);
      dst[k * nt + at] = 1;
    }
  };
  for (let k = 0; k < nLD; k++) {
    const cut = next();
    if (cut.length < 2) throw new StyleError('local/density table: a potential needs its lower and upper cutoffs');
    t.lower[k] = parseFloat(cut[0]);
    t.upper[k] = parseFloat(cut[1]);
    typeLine(next(), t.a, k);
    typeLine(next(), t.b, k);
    const rho = next();
    if (rho.length < 2) throw new StyleError('local/density table: a potential needs rho_min and rho_max');
    t.rhoMin[k] = parseFloat(rho[0]);
    t.rhoMax[k] = parseFloat(rho[1]);
    t.delta[k] = (t.rhoMax[k] - t.rhoMin[k]) / (nrho - 1);
    for (let n = 0; n < nrho; n++) {
      const v = parseFloat(next()[0]);
      if (Number.isNaN(v)) throw new StyleError('local/density table: invalid value in the potential table');
      ftmp[k * nrho + n] = v;
    }
    const uc2 = t.upper[k] * t.upper[k], lc2 = t.lower[k] * t.lower[k];
    t.upperSq[k] = uc2;
    t.lowerSq[k] = lc2;
    const ratio = lc2 / uc2;
    let denom = 1.0 - ratio;
    denom = denom * denom * denom;
    t.c0[k] = (1 - 3.0 * ratio) / denom;
    t.c2[k] = (6.0 * ratio) / (uc2 * denom);
    t.c4[k] = -(3.0 + 3.0 * ratio) / (uc2 * uc2 * denom);
    t.c6[k] = 2.0 / (uc2 * uc2 * uc2 * denom);
    if (t.upper[k] > t.cutmax) t.cutmax = t.upper[k];
    interpolateCbspl(nrho, t.delta[k], ftmp, k * nrho, t.spline, k * nrho * 7);
  }
  return t;
};

export class PairLocalDensity extends Pair {
  readonly name: string = 'local/density';
  /** Forces and energies are summed in the pair loops; the virial comes from x.f (virial_fdotr_compute). */
  override virialFdotr = true;
  override needsHalf = true;
  private ldTable: LDTable | null = null;
  private nt = 0;
  /** Per-atom local densities and embedding derivatives, k * nlocal + i (owned atoms; ghosts map through owner). */
  private localrho = new Float64Array(0);
  private fp = new Float64Array(0);
  private nlocalCache = -1;

  settings(args: string[]): void {
    if (args.length > 0) throw new StyleError('Illegal pair_style command');
  }

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
  }

  /** pair_coeff * * <table>: the only form the style accepts (coeff is called once for all type pairs). */
  coeff(args: string[], ctx: StyleContext): void {
    if (args.length !== 3) throw new StyleError('Incorrect args for pair coefficients');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('Incorrect args for pair coefficients');
    const t = parseLocalDensityFile(ctx.readFile(args[2], potentialFile('local/density')), this.ntypes);
    if (t.nLD !== this.ntypes * this.ntypes) ctx.log(`WARNING: Expected ${this.ntypes * this.ntypes} local density potentials but got ${t.nLD}`);
    this.ldTable = t;
    this.nlocalCache = -1;
  }

  /** init_style: the table spline is built at read time, so nothing is left to do here. */
  override initStyle(): void {
    if (!this.ldTable) throw new StyleError('All pair local/density coeffs are not set');
  }

  /** init_one: every type pair gets the largest upper cutoff of the tables (cutforcesq is its square). */
  initOne(_i: number, _j: number): number {
    if (!this.ldTable) throw new StyleError('All pair local/density coeffs are not set');
    return this.ldTable.cutmax;
  }

  compute(pc: PairCompute): void {
    const t = this.ldTable;
    if (!t) throw new StyleError('pair local/density used before pair_coeff');
    const nLD = t.nLD, nl = pc.nlocal, nt = this.nt, nrho = t.nrho;
    if (this.nlocalCache !== nl || this.localrho.length !== nLD * nl) {
      this.localrho = new Float64Array(nLD * nl);
      this.fp = new Float64Array(nLD * nl);
      this.nlocalCache = nl;
    }
    const localrho = this.localrho, fp = this.fp;
    localrho.fill(0);
    fp.fill(0);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const list = pc.half!;
    const nb = list.neighbors;
    const inum = list.inum;
    const cutforcesq = t.cutmax * t.cutmax;

    // local densities: sum over the half list, both atoms of a pair get their contribution
    for (let i = 0; i < inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const it = type[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = nb[kk] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jt = type[j];
        const oj = owner[j];
        for (let k = 0; k < nLD; k++) {
          let phi: number;
          if (rsq < t.lowerSq[k]) phi = 1.0;
          else if (rsq > t.upperSq[k]) phi = 0.0;
          else phi = t.c0[k] + rsq * (t.c2[k] + rsq * (t.c4[k] + t.c6[k] * rsq));
          localrho[k * nl + i] += phi * t.b[k * nt + jt];
          localrho[k * nl + oj] += phi * t.b[k * nt + it];
        }
      }
    }

    // embedding energies of the owned atoms, linear extrapolation outside the table
    let evdwl = 0;
    for (let i = 0; i < inum; i++) {
      const it = type[i];
      let uLD = 0;
      for (let k = 0; k < nLD; k++) {
        if (!t.a[k * nt + it]) continue;
        const rho = localrho[k * nl + i];
        const base = k * nrho * 7;
        const aCoef = t.a[k * nt + it];
        if (rho <= t.rhoMin[k]) {
          const o = base;
          const fpk = t.spline[o + 2];
          fp[k * nl + i] = fpk;
          uLD += aCoef * (t.spline[o + 6] + fpk * (rho - t.rhoMin[k]));
        } else if (rho >= t.rhoMax[k]) {
          const o = base + (nrho - 2) * 7;
          const fpk = t.spline[o] + t.spline[o + 1] + t.spline[o + 2];
          fp[k * nl + i] = fpk;
          uLD += aCoef * ((t.spline[o + 3] + t.spline[o + 4] + t.spline[o + 5] + t.spline[o + 6]) + fpk * (rho - t.rhoMax[k]));
        } else {
          let p = (rho - t.rhoMin[k]) / t.delta[k];
          let m = Math.trunc(p);
          m = Math.max(0, Math.min(m, nrho - 2));
          p -= m;
          p = Math.min(p, 1.0);
          const o = base + m * 7;
          const fpk = (t.spline[o] * p + t.spline[o + 1]) * p + t.spline[o + 2];
          fp[k * nl + i] = fpk;
          uLD += aCoef * (((t.spline[o + 3] * p + t.spline[o + 4]) * p + t.spline[o + 5]) * p + t.spline[o + 6]);
        }
      }
      evdwl += uLD;
      if (pc.eatom) pc.eatom[i] += uLD;
    }
    pc.acc.evdwl += evdwl;

    // forces: the pair force is -(a_i b_j F'_i + a_j b_i F'_j) phi'(r^2) 2 r (the source's dphi / rsq)
    for (let i = 0; i < inum; i++) {
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const it = type[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = nb[kk] & NEIGHMASK;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (!(rsq < cutforcesq)) continue;
        const jt = type[j];
        const oj = owner[j];
        const rsqinv = 1.0 / rsq;
        let fpair = 0.0;
        for (let k = 0; k < nLD; k++) {
          if (rsq >= t.lowerSq[k] && rsq < t.upperSq[k]) {
            const dphi = rsq * (2.0 * t.c2[k] + rsq * (4.0 * t.c4[k] + 6.0 * t.c6[k] * rsq));
            fpair += -(t.a[k * nt + it] * t.b[k * nt + jt] * fp[k * nl + i] + t.a[k * nt + jt] * t.b[k * nt + it] * fp[k * nl + oj]) * dphi;
          }
        }
        fpair *= rsqinv;
        f[3 * i] += delx * fpair;
        f[3 * i + 1] += dely * fpair;
        f[3 * i + 2] += delz * fpair;
        f[3 * j] -= delx * fpair;
        f[3 * j + 1] -= dely * fpair;
        f[3 * j + 2] -= delz * fpair;
        if (pc.vatom) tallyXYZ(pc, i, j, 0, delx * fpair, dely * fpair, delz * fpair, delx, dely, delz);
      }
    }
  }
}
