/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BOCS/pair_ldd.cpp and pair_ldd.h
 * (PairLdd: pair_style ldd, pair_coeff * * <file> <species per type>), the local-density and gradient-density
 * sums, their embedding energies and the forces (LDD_calculate_LDs, LDD_calculate_energies, compute). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributed by Michael R. DeLyser and Maria C. Lesniewski (Pennsylvania
 * State University). Cited in the source: doi:10.1063/1.5128665 (local density potentials) and doi:10.1063/5.0075291
 * (the gradient keyword).
 *
 * Model (docs.lammps.org/pair_ldd.html): every atom of species X has, for each species Y, a local density
 * rho_Y = sum_j w_XY(r_ij) over its neighbours j of species Y, and a gradient density (the sum of w' r_hat). The
 * energy of an atom is sum_Y u_XY(rho_Y) (+ the gradient term when a gradient potential is given), and the forces
 * follow from the chain rule. Ghost atoms: the sums native LAMMPS does on ghosts (reverse communication) and the
 * forward copies are done through the owner index (nb.owner), which gives the same totals. The half neighbour list
 * is used as the source does (newton on).
 *
 * Scope: the indicators "lucy" and "dpd" and the potentials "quadratic", "table/spline" and "table/gradspline" (what
 * the BOCS examples use). The other indicators and potentials of the source are refused by name (potentials.ts,
 * indicators.ts). The per-atom outputs of "fix pair" are not provided (the engine has no such fix).
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { tallyXYZ } from '../dpd/pair_dpd';
import { makeIndicator, type LddIndicator } from './indicators';
import { LddPotentialQuadratic, LddPotentialTableGradSpline, LddPotentialTableSpline, type LddPotential } from './potentials';

/** Keywords of a potential file entry (pair_ldd.cpp coeff_ldd). */
const KEY_IND = 'indicator', KEY_POTL = 'potential', KEY_GRAD = 'gradient', KEY_SELF = 'self', KEY_IGNORE = 'ignore';

/** Builds a potential from "type args..." (the keyword's arguments start at kw[ipt + 1]). */
const makePotential = (name: string, kw: string[], ipt: number, label: string, read: (file: string) => string): { potl: LddPotential; used: number } => {
  const file = kw[ipt + 2];
  switch (name) {
    case 'quadratic': {
      const potl = new LddPotentialQuadratic(kw.slice(ipt + 2, ipt + 5));
      return { potl, used: potl.nCoeffs + 2 };
    }
    case 'table/spline':
    case 'table/gradspline': {
      if (file === undefined) throw new StyleError(`Missing filename following the ldd ${name} keyword (${label})`);
      const potl = name === 'table/spline' ? new LddPotentialTableSpline(read(file)) : new LddPotentialTableGradSpline(read(file));
      return { potl, used: potl.nCoeffs + 2 };
    }
    default:
      throw new StyleError(`ldd potential '${name}' is not supported by the browser engine (supported: quadratic, table/spline, table/gradspline)`);
  }
};

export class PairLdd extends Pair {
  readonly name: string = 'ldd';
  override virialFdotr = true;
  override needsHalf = true;
  /** Species names in order of first appearance in pair_coeff; map[type] = species index (-1 for NULL). */
  private elements: string[] = [];
  private map = new Int32Array(0);
  private nel = 0;
  private ignorePair = new Uint8Array(0);
  private ignoreMe = new Uint8Array(0);
  private bGradient = new Uint8Array(0);
  private selfInter = new Uint8Array(0);
  private inds: (LddIndicator | null)[] = [];
  private potls: (LddPotential | null)[] = [];
  private gradPotls: (LddPotential | null)[] = [];
  /** Per-atom local densities (k * nlocal + i for species k), gradient densities (3 * k + d) and energies. */
  private ld = new Float64Array(0);
  private grad = new Float64Array(0);
  private ttl = new Float64Array(0);
  private nlocalCache = -1;

  settings(args: string[]): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style ldd command: expected no arguments');
  }

  /** pair_coeff * * <file> <species of type 1> ... <species of type ntypes>. */
  coeff(args: string[], ctx: StyleContext): void {
    if (!ctx.s) throw new StyleError('Pair style ldd must be defined after the simulation box is created');
    if (args.length !== 3 + this.ntypes || args[0] !== '*' || args[1] !== '*') {
      throw new StyleError("Incorrect args for pair_coeff ldd: expected '* * <file> <species> ...' with one species name per atom type");
    }
    const species = args.slice(3);
    const elements: string[] = [];
    const map = new Int32Array(this.ntypes + 1).fill(-1);
    for (let t = 1; t <= this.ntypes; t++) {
      const s = species[t - 1];
      if (s === 'NULL') throw new StyleError('Pair style ldd does not allow a NULL species in the type map');
      let e = elements.indexOf(s);
      if (e < 0) { elements.push(s); e = elements.length - 1; }
      map[t] = e;
    }
    this.elements = elements;
    this.map = map;
    this.nel = elements.length;
    const nel = this.nel;
    this.ignorePair = new Uint8Array(nel * nel).fill(1);
    this.ignoreMe = new Uint8Array(nel).fill(1);
    this.bGradient = new Uint8Array(nel * nel);
    this.selfInter = new Uint8Array(nel * nel);
    this.inds = new Array<LddIndicator | null>(nel * nel).fill(null);
    this.potls = new Array<LddPotential | null>(nel * nel).fill(null);
    this.gradPotls = new Array<LddPotential | null>(nel * nel).fill(null);
    this.nlocalCache = -1;
    this.readFile(args[2], ctx);
  }

  /** read_file: one "<species_i> <species_j> keywords..." line per ordered species pair; '#' starts a comment. */
  private readFile(name: string, ctx: StyleContext): void {
    const nel = this.nel;
    const seen = new Uint8Array(nel * nel);
    const dim = ctx.s!.dimension;
    for (const raw of ctx.readFile(name).split(/\r?\n/)) {
      const words = raw.replace(/#.*$/, '').trim();
      if (words === '') continue;
      const w = words.split(/\s+/);
      if (w.length < 2) continue;
      const si = this.elements.indexOf(w[0]), sj = this.elements.indexOf(w[1]);
      if (si < 0 || sj < 0) continue;
      if (seen[si * nel + sj]) throw new StyleError(`Duplicate entry for species pair ${w[0]} ${w[1]} in ldd potential file ${name}`);
      seen[si * nel + sj] = 1;
      this.coeffLdd(si, sj, w.slice(2), dim, name, ctx);
    }
    for (let i = 0; i < nel; i++) {
      for (let j = 0; j < nel; j++) {
        if (!seen[i * nel + j]) throw new StyleError(`Missing entry for species pair ${this.elements[i]} ${this.elements[j]} in ldd potential file ${name}`);
      }
    }
    for (let i = 0; i < nel; i++) {
      let all = true;
      for (let j = 0; j < nel; j++) if (!this.ignorePair[i * nel + j]) all = false;
      this.ignoreMe[i] = all ? 1 : 0;
    }
  }

  /** coeff_ldd: the keyword part of one entry (the species names are consumed). */
  private coeffLdd(si: number, sj: number, kw: string[], dim: 2 | 3, file: string, ctx: StyleContext): void {
    const nel = this.nel, p = si * nel + sj;
    let iarg = 0;
    let bSelf = false, bIgnore = false;
    let bkInd = false, bkPotl = false, bkSelf = false, bkGrad = false;
    const label = `${this.elements[si]} ${this.elements[sj]} in ${file}`;
    while (iarg < kw.length) {
      const key = kw[iarg];
      if (key === KEY_IND) {
        if (bkInd) throw new StyleError(`Found ldd pair_coeff keyword ${KEY_IND} twice`);
        bkInd = true;
        if (iarg + 3 >= kw.length) throw new StyleError(`ldd pair_coeff keyword ${KEY_IND} must be followed by: wtype r0 rc`);
        const r0 = Number(kw[iarg + 2]), rc = Number(kw[iarg + 3]);
        if (!Number.isFinite(r0) || !Number.isFinite(rc)) throw new StyleError(`ldd indicator r0 and rc must be numbers (${label})`);
        if (rc <= 0.0 || r0 < 0.0 || r0 >= rc) {
          throw new StyleError(`ldd species pair ${this.elements[si]} ${this.elements[sj]}: indicator cutoff rc (${rc}) must be positive and the inner radius r0 (${r0}) must be non-negative and smaller than rc`);
        }
        this.inds[p] = makeIndicator(kw[iarg + 1], r0, rc, dim);
        iarg += 4;
      } else if (key === KEY_SELF) {
        if (bkSelf) throw new StyleError(`Found ldd pair_coeff keyword ${KEY_SELF} twice`);
        bkSelf = true;
        if (iarg + 1 >= kw.length) throw new StyleError(`ldd pair_coeff keyword ${KEY_SELF} must be followed by: yes/no`);
        if (kw[iarg + 1] === 'yes') bSelf = true;
        else if (kw[iarg + 1] === 'no') bSelf = false;
        else throw new StyleError(`Expected yes/no after ldd keyword ${KEY_SELF}, found ${kw[iarg + 1]}`);
        iarg += 2;
      } else if (key === KEY_POTL || key === KEY_GRAD) {
        const isGrad = key === KEY_GRAD;
        if (isGrad ? bkGrad : bkPotl) throw new StyleError(`Found ldd pair_coeff keyword ${key} twice`);
        if (isGrad) bkGrad = true; else bkPotl = true;
        if (iarg + 1 >= kw.length) throw new StyleError(`ldd pair_coeff keyword ${key} must be followed by: type *args*`);
        const made = makePotential(kw[iarg + 1], kw, iarg, label, (fn) => ctx.readFile(fn));
        if (isGrad) this.gradPotls[p] = made.potl; else this.potls[p] = made.potl;
        iarg += made.used;
      } else if (key === KEY_IGNORE) {
        bIgnore = true;
        iarg += 1;
      } else {
        throw new StyleError(`Unknown ldd pair_coeff keyword ${key} (valid: ${KEY_IND} ${KEY_SELF} ${KEY_POTL} ${KEY_GRAD} ${KEY_IGNORE})`);
      }
    }
    this.bGradient[p] = bkGrad ? 1 : 0;
    this.ignorePair[p] = bIgnore ? 1 : 0;
    if (!bIgnore) {
      if (!bkInd) throw new StyleError(`ldd species pair ${this.elements[si]} ${this.elements[sj]}: missing required keyword ${KEY_IND}`);
      if (!bkPotl) throw new StyleError(`ldd species pair ${this.elements[si]} ${this.elements[sj]}: missing required keyword ${KEY_POTL}`);
      if (!bkSelf) throw new StyleError(`ldd species pair ${this.elements[si]} ${this.elements[sj]}: missing required keyword ${KEY_SELF}`);
      if (bSelf) {
        if (si === sj) this.selfInter[p] = 1;
        else ctx.log(`WARNING: ldd self interaction requested for distinct species ${this.elements[si]} ${this.elements[sj]}; ignoring`);
      }
    }
  }

  override initStyle(ctx: StyleContext): void {
    if (ctx.newtonPair === false) throw new StyleError('Pair style ldd requires newton pair on');
  }

  /** init_one: the largest indicator cutoff of the two ordered species pairs this type pair maps to. */
  initOne(i: number, j: number): number {
    if (this.nel === 0) throw new StyleError('All pair ldd coeffs are not set');
    const si = this.map[i], sj = this.map[j];
    let cut = 0.0;
    const nel = this.nel;
    const a = this.inds[si * nel + sj], b = this.inds[sj * nel + si];
    if (!this.ignorePair[si * nel + sj] && a && a.rc > cut) cut = a.rc;
    if (!this.ignorePair[sj * nel + si] && b && b.rc > cut) cut = b.rc;
    return cut;
  }

  private ensure(nl: number): void {
    if (this.nlocalCache === nl) return;
    this.ld = new Float64Array(nl * this.nel);
    this.grad = new Float64Array(3 * nl * this.nel);
    this.ttl = new Float64Array(nl);
    this.nlocalCache = nl;
  }

  compute(pc: PairCompute): void {
    if (this.nel === 0) throw new StyleError('pair ldd used before pair_coeff');
    const nel = this.nel, nt = this.ntypes + 1;
    const nl = pc.nlocal;
    this.ensure(nl);
    const ld = this.ld, grad = this.grad, ttl = this.ttl;
    ld.fill(0);
    grad.fill(0);
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const list = pc.half!;
    const nb = list.neighbors;
    const inum = list.inum;
    const cutsq = this.cutsq;
    const map = this.map;

    this.calculateLDs(pc, inum, nb, list, nel, nt, cutsq, map, owner);
    this.calculateEnergies(inum, nel, map, type, ttl);

    let evdwl = 0;
    for (let i = 0; i < inum; i++) {
      const it = map[type[i]];
      if (this.ignoreMe[it]) continue;
      evdwl += ttl[i];
      if (pc.eatom) pc.eatom[i] += ttl[i];
    }
    pc.acc.evdwl += evdwl;

    // forces (compute() in pair_ldd.cpp)
    const eij = [0, 0, 0];
    const gf2 = [0, 0, 0];
    for (let i = 0; i < inum; i++) {
      const it = map[type[i]];
      if (this.ignoreMe[it]) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const iatype = type[i];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = nb[kk] & NEIGHMASK;
        const jatype = type[j];
        const jt = map[jatype];
        if (!(!this.ignorePair[it * nel + jt] || !this.ignorePair[jt * nel + it])) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (!(rsq < cutsq[iatype * nt + jatype])) continue;
        const rPair = Math.sqrt(rsq);
        const oj = owner[j];
        let fpairLD = 0.0;
        if (!this.ignorePair[it * nel + jt]) {
          fpairLD += this.inds[it * nel + jt]!.wp(rPair) * this.potls[it * nel + jt]!.f(ld[i * nel + jt]);
        }
        if (!this.ignorePair[jt * nel + it]) {
          fpairLD += this.inds[jt * nel + it]!.wp(rPair) * this.potls[jt * nel + it]!.f(ld[oj * nel + it]);
        }
        fpairLD /= rPair;
        f[3 * i] += delx * fpairLD;
        f[3 * i + 1] += dely * fpairLD;
        f[3 * i + 2] += delz * fpairLD;
        f[3 * j] -= delx * fpairLD;
        f[3 * j + 1] -= dely * fpairLD;
        f[3 * j + 2] -= delz * fpairLD;
        if (!(this.bGradient[it * nel + jt] || this.bGradient[jt * nel + it])) {
          if (pc.vatom) tallyXYZ(pc, i, j, 0, delx * fpairLD, dely * fpairLD, delz * fpairLD, delx, dely, delz);
          continue;
        }
        // gradient force: the source's gf1 (along r_ij) and gf2 (along the gradient densities)
        eij[0] = delx / rPair;
        eij[1] = dely / rPair;
        eij[2] = delz / rPair;
        let gf1 = 0.0;
        gf2[0] = gf2[1] = gf2[2] = 0.0;
        if (this.bGradient[it * nel + jt]) {
          const p = it * nel + jt;
          const gp = this.gradPotls[p]!, ind = this.inds[p]!;
          const ldv = ld[i * nel + jt];
          const gi = 3 * i * nel;
          const dotG = grad[gi + 3 * jt] * grad[gi + 3 * jt] + grad[gi + 3 * jt + 1] * grad[gi + 3 * jt + 1] + grad[gi + 3 * jt + 2] * grad[gi + 3 * jt + 2];
          const dotE = eij[0] * grad[gi + 3 * jt] + eij[1] * grad[gi + 3 * jt + 1] + eij[2] * grad[gi + 3 * jt + 2];
          gf1 += gp.f(ldv) * dotG * ind.wp(rPair) - 2.0 * gp.u(ldv) * (ind.wp2(rPair) - ind.wp(rPair) / rPair) * dotE;
          for (let d = 0; d < 3; d++) gf2[d] -= 2.0 * gp.u(ldv) * ind.wp(rPair) / rPair * grad[gi + 3 * jt + d];
        }
        if (this.bGradient[jt * nel + it]) {
          const p = jt * nel + it;
          const gp = this.gradPotls[p]!, ind = this.inds[p]!;
          const ldv = ld[oj * nel + it];
          const gj = 3 * oj * nel;
          const dotG = grad[gj + 3 * it] * grad[gj + 3 * it] + grad[gj + 3 * it + 1] * grad[gj + 3 * it + 1] + grad[gj + 3 * it + 2] * grad[gj + 3 * it + 2];
          const dotE = eij[0] * grad[gj + 3 * it] + eij[1] * grad[gj + 3 * it + 1] + eij[2] * grad[gj + 3 * it + 2];
          gf1 += gp.f(ldv) * dotG * ind.wp(rPair) + 2.0 * gp.u(ldv) * (ind.wp2(rPair) - ind.wp(rPair) / rPair) * dotE;
          for (let d = 0; d < 3; d++) gf2[d] += 2.0 * gp.u(ldv) * ind.wp(rPair) / rPair * grad[gj + 3 * it + d];
        }
        const fx = gf1 * eij[0] + gf2[0];
        const fy = gf1 * eij[1] + gf2[1];
        const fz = gf1 * eij[2] + gf2[2];
        f[3 * i] += fx;
        f[3 * i + 1] += fy;
        f[3 * i + 2] += fz;
        f[3 * j] -= fx;
        f[3 * j + 1] -= fy;
        f[3 * j + 2] -= fz;
        if (pc.vatom) {
          tallyXYZ(pc, i, j, 0, fx + fpairLD * delx, fy + fpairLD * dely, fz + fpairLD * delz, delx, dely, delz);
        }
      }
    }
  }

  /** LDD_calculate_LDs: local and gradient densities of the owned atoms (ghost contributions go to their owners). */
  private calculateLDs(pc: PairCompute, inum: number, nb: Int32Array, list: NonNullable<PairCompute['half']>, nel: number, nt: number,
    cutsq: Float64Array, map: Int32Array, owner: Int32Array): void {
    const { x, type } = pc;
    const ld = this.ld, grad = this.grad;
    for (let i = 0; i < inum; i++) {
      const it = map[type[i]];
      if (this.ignoreMe[it]) continue;
      const xi = x[3 * i], yi = x[3 * i + 1], zi = x[3 * i + 2];
      const iatype = type[i];
      if (this.selfInter[it * nel + it]) ld[i * nel + it] += this.inds[it * nel + it]!.invnorm;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let kk = k0; kk < k1; kk++) {
        const j = nb[kk] & NEIGHMASK;
        const jatype = type[j];
        const jt = map[jatype];
        if (!(!this.ignorePair[it * nel + jt] || !this.ignorePair[jt * nel + it])) continue;
        const delx = xi - x[3 * j], dely = yi - x[3 * j + 1], delz = zi - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq[iatype * nt + jatype] && !this.ignorePair[it * nel + jt]) {
          const r = Math.sqrt(rsq);
          const ind = this.inds[it * nel + jt]!;
          ld[i * nel + jt] += ind.w(r);
          const wprime = ind.wp(r);
          const g = 3 * i * nel + 3 * jt;
          grad[g] += wprime * delx / r;
          grad[g + 1] += wprime * dely / r;
          grad[g + 2] += wprime * delz / r;
        }
        const oj = owner[j];
        if (!this.ignorePair[jt * nel + it] && rsq < cutsq[jatype * nt + iatype]) {
          const r = Math.sqrt(rsq);
          const ind = this.inds[jt * nel + it]!;
          ld[oj * nel + it] += ind.w(r);
          const wprime = ind.wp(r);
          const g = 3 * oj * nel + 3 * it;
          grad[g] -= wprime * delx / r;
          grad[g + 1] -= wprime * dely / r;
          grad[g + 2] -= wprime * delz / r;
        }
      }
    }
  }

  /** LDD_calculate_energies: the embedding energy of each owned atom (ttl). */
  private calculateEnergies(inum: number, nel: number, map: Int32Array, type: Int32Array, ttl: Float64Array): void {
    const ld = this.ld, grad = this.grad;
    for (let i = 0; i < inum; i++) {
      const it = map[type[i]];
      ttl[i] = 0;
      if (this.ignoreMe[it]) continue;
      let total = 0;
      for (let tidx = 0; tidx < nel; tidx++) {
        if (this.ignorePair[it * nel + tidx]) continue;
        const p = it * nel + tidx;
        const ldv = ld[i * nel + tidx];
        let ldNrg = this.potls[p]!.u(ldv);
        if (this.bGradient[p]) {
          const g = 3 * i * nel + 3 * tidx;
          const ai2 = grad[g] * grad[g] + grad[g + 1] * grad[g + 1] + grad[g + 2] * grad[g + 2];
          ldNrg += ai2 * this.gradPotls[p]!.u(ldv);
        }
        total += ldNrg;
      }
      ttl[i] = total;
    }
  }
}
