/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_expand_sphere.cpp
 * and pair_lj_expand_sphere.h (PairLJExpandSphere: pair_style lj/expand/sphere). Contributing author (from the source):
 * Axel Kohlmeyer (Temple University). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The 12-6 Lennard-Jones potential of the distance r - Delta, with Delta = r_i + r_j from the per-atom radii
 * (docs.lammps.org/pair_lj_expand_sphere.html: "E = 4 epsilon [ (sigma/(r - Delta))^12 - (sigma/(r - Delta))^6 ]
 * r < r_c + Delta"; "r_c is the cutoff which does not include the distance Delta"; "supports the pair_modify shift
 * option"; "does *not* support the pair_modify tail option").
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import type { SimState } from '../../engine/types';
import { forEachTypePair, mixDistanceNative, mixEnergyNative, powint, radiusOfIndex } from './common';

export class PairLJExpandSphere extends Pair {
  readonly name: string = 'lj/expand/sphere';
  virialFdotr = true;
  cutGlobal = 0;
  private p!: PairParams;
  private rmax = new Float64Array(0);
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);
  private state: SimState | null = null;

  settings(args: string[]): void {
    // pair_lj_expand_sphere.cpp settings(): if (narg != 1) error->all(FLERR, "Illegal pair_style command");
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings(): every explicitly set pair takes the new global cutoff ("reset cutoffs that have been explicitly set")
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'cut']);
    const nt = ntypes + 1;
    this.rmax = new Float64Array(nt);
    this.lj1 = new Float64Array(nt * nt);
    this.lj2 = new Float64Array(nt * nt);
    this.lj3 = new Float64Array(nt * nt);
    this.lj4 = new Float64Array(nt * nt);
  }

  coeff(args: string[]): void {
    // pair_lj_expand_sphere.cpp coeff(): narg 4 or 5 (I J epsilon sigma [cutoff])
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const sig = parseNum(args[3], 'sigma');
    const cutOne = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [eps, sig, cutOne]));
  }

  initStyle(ctx: StyleContext): void {
    // init_style(): the radius attribute, then rmax per type from the atoms
    const s = ctx.s;
    if (!s?.radius) throw new StyleError('Pair style lj/expand/sphere requires atom attribute radius');
    this.state = s;
    // pair_modify tail: the native style has no tail term (pair_lj_expand_sphere.rst: "does *not* support" it)
    this.noTermLog(ctx, 'tail');
    this.rmax[0] = 0.0;
    for (let itype = 1; itype <= this.ntypes; itype++) {
      let rmaxOne = 0.0;
      for (let i = 0; i < s.n; i++) {
        if (s.type[i] === itype && s.radius[i] > rmaxOne) rmaxOne = s.radius[i];
      }
      this.rmax[itype] = rmaxOne;
    }
  }

  initOne(i: number, j: number): number {
    const p = this.p;
    if (!p.isSet(i, j)) {
      if (!p.isSet(i, i) || !p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair ${i} ${j})`);
      p.setMixed(i, j, 'epsilon', mixEnergyNative(this.mix, p.get('epsilon', i, i), p.get('epsilon', j, j), 0.0, 0.0));
      p.setMixed(i, j, 'sigma', mixDistanceNative(this.mix, p.get('sigma', i, i), p.get('sigma', j, j)));
      p.setMixed(i, j, 'cut', mixDistanceNative(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    const eps = p.get('epsilon', i, j), sig = p.get('sigma', i, j), cut = p.get('cut', i, j);
    const nt = this.ntypes + 1;
    const k1 = i * nt + j, k2 = j * nt + i;
    this.lj1[k1] = 48.0 * eps * powint(sig, 12);
    this.lj2[k1] = 24.0 * eps * powint(sig, 6);
    this.lj3[k1] = 4.0 * eps * powint(sig, 12);
    this.lj4[k1] = 4.0 * eps * powint(sig, 6);
    p.setMixed(j, i, 'epsilon', eps);
    p.setMixed(j, i, 'sigma', sig);
    p.setMixed(j, i, 'cut', cut);
    this.lj1[k2] = this.lj1[k1];
    this.lj2[k2] = this.lj2[k1];
    this.lj3[k2] = this.lj3[k1];
    this.lj4[k2] = this.lj4[k1];
    // "delta parameter is the sum of the radii"
    return cut + this.rmax[i] + this.rmax[j];
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const s = this.state!;
    const radius = s.radius!;
    const own = pc.nb.owner;
    const nt = this.ntypes + 1;
    const eps = this.p.p('epsilon'), sigma = this.p.p('sigma'), cutp = this.p.p('cut');
    const { cutsq, lj1, lj2, lj3, lj4 } = this;
    const sLJ = pc.specialLJ;
    const offset = this.shift;
    const tally = pc.eatom !== null || pc.vatom !== null;
    let evdwl = 0;
    const nb = list.neighbors;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const rtmp = radiusOfIndex(radius, own, i);
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = nb[k];
        const factorLJ = sLJ[jj >>> SBBITS];
        const j = jj & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        const t = itype * nt + jtype;
        if (rsq < cutsq[t]) {
          // cutsq is the maximum cutoff per type; the real cutoff applies to the shifted distance
          const rj = radiusOfIndex(radius, own, j);
          const r = Math.sqrt(rsq);
          const rshift = r - rtmp - rj;
          if (rshift < cutp[t]) {
            const rshiftsq = rshift * rshift;
            const r2inv = 1.0 / rshiftsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forcelj = r6inv * (lj1[t] * r6inv - lj2[t]);
            const fpair = factorLJ * forcelj / rshift / r;
            f[3 * i] += delx * fpair;
            f[3 * i + 1] += dely * fpair;
            f[3 * i + 2] += delz * fpair;
            f[3 * j] -= delx * fpair;
            f[3 * j + 1] -= dely * fpair;
            f[3 * j + 2] -= delz * fpair;
            let e = r6inv * (lj3[t] * r6inv - lj4[t]);
            if (offset && rshiftsq > 0.0) {
              const ratio6 = powint(sigma[t] / (cutp[t] + rtmp + rj), 6);
              e -= 4.0 * eps[t] * (ratio6 * ratio6 - ratio6);
            }
            e *= factorLJ;
            evdwl += e;
            if (tally) tallyAtom(pc, i, j, e, fpair, delx, dely, delz);
          }
        }
      }
    }
    pc.acc.evdwl += evdwl;
  }

  single(i: number, j: number, itype: number, jtype: number, rsq: number, _fc: number, factorLJ: number, _qi: number, _qj: number, _del?: PairSep) {
    const radius = this.state!.radius!;
    const t = itype * (this.ntypes + 1) + jtype;
    const cut = this.p.get('cut', itype, jtype);
    const eps = this.p.get('epsilon', itype, jtype);
    const sigma = this.p.get('sigma', itype, jtype);
    let fforce = 0, philj = 0;
    const r = Math.sqrt(rsq);
    const rshift = r - radius[i] - radius[j];
    if (rshift < cut) {
      const rshiftsq = rshift * rshift;
      const r2inv = 1.0 / rshiftsq;
      const r6inv = r2inv * r2inv * r2inv;
      const forcelj = r6inv * (this.lj1[t] * r6inv - this.lj2[t]);
      fforce = factorLJ * forcelj / rshift / r;
      philj = r6inv * (this.lj3[t] * r6inv - this.lj4[t]);
      if (this.shift && rshiftsq > 0.0) {
        const ratio6 = powint(sigma / (cut + radius[i] + radius[j]), 6);
        philj -= 4.0 * eps * (ratio6 * ratio6 - ratio6);
      }
    }
    return { eng: factorLJ * philj, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))} ${fmtCoeff(this.p.get('sigma', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('sigma', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'epsilon') return this.p.p('epsilon');
    if (name === 'sigma') return this.p.p('sigma');
    return undefined;
  }
}
