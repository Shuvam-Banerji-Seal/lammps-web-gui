/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-PAIR/pair_lj_cut_sphere.cpp
 * and pair_lj_cut_sphere.h (PairLJCutSphere: pair_style lj/cut/sphere). Contributing author (from the source): Axel
 * Kohlmeyer (Temple University). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The 12-6 Lennard-Jones potential with sigma_ij = 2 mix_distance(r_i, r_j) from the per-atom radii, and the cutoff a
 * ratio of sigma_ij (docs.lammps.org/pair_lj_cut_sphere.html: "E = 4 epsilon [ (sigma_ij/r)^12 - (sigma_ij/r)^6 ]
 * r < r_c * sigma_ij"; "The cutoff is not specified as a distance, but as ratio that is internally multiplied by
 * sigma_ij"; "The lj/cut/sphere pair style supports the pair_modify shift option" and "does *not* support the
 * pair_modify tail option"; "The lj/cut/sphere pair style does not support the *sixthpower* mixing rule").
 */
import { Pair, PairParams, StyleError, type PairCompute, type StyleContext, type PairSep } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { tallyAtom } from '../../engine/force/pair/lj_cut';
import { parseNum, fmtCoeff } from '../../engine/force/util';
import type { SimState } from '../../engine/types';
import { forEachTypePair, mixDistanceNative, mixEnergyNative, powint, radiusOfIndex } from './common';

export class PairLJCutSphere extends Pair {
  readonly name: string = 'lj/cut/sphere';
  virialFdotr = true;
  cutGlobal = 0;
  private p!: PairParams;
  /** 1 when the pair's cutoff ratio is the global one (settings resets those, as native does). */
  private rmax = new Float64Array(0);
  private state: SimState | null = null;

  settings(args: string[]): void {
    // pair_lj_cut_sphere.cpp settings(): if (narg != 1) error->all(FLERR, "Illegal pair_style command");
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // native settings(): every explicitly set pair takes the new global cutoff ("reset cutoffs that have been explicitly set")
    if (this.p) for (let i = 1; i <= this.ntypes; i++) for (let j = i; j <= this.ntypes; j++) if (this.p.isSet(i, j)) this.p.setMixed(i, j, 'cut', this.cutGlobal);
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'cut']);
      this.rmax = new Float64Array(ntypes + 1);
  }

  coeff(args: string[]): void {
    // pair_lj_cut_sphere.cpp coeff(): narg 3 or 4 (I J epsilon [cutoff ratio])
    if (args.length < 3 || args.length > 4) throw new StyleError('Incorrect args for pair coefficients');
    const eps = parseNum(args[2], 'epsilon');
    const cutOne = args.length === 4 ? parseNum(args[3], 'cutoff') : this.cutGlobal;
    forEachTypePair(args[0], args[1], this.ntypes, (i, j) => this.p.set(i, j, [eps, cutOne]));
  }

  initStyle(ctx: StyleContext): void {
    // init_style(): the radius attribute, the sixthpower refusal, then rmax per type from the atoms
    const s = ctx.s;
    if (!s?.radius) throw new StyleError('Pair style lj/cut/sphere requires atom attribute radius');
    if (this.mix === 'sixthpower') throw new StyleError('Pair_modify mix sixthpower is not compatible with pair style lj/cut/sphere');
    this.state = s;
    // pair_modify tail: the native style has no tail term (pair_lj_cut_sphere.rst: "does *not* support" it)
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
      p.setMixed(i, j, 'cut', mixDistanceNative(this.mix, p.get('cut', i, i), p.get('cut', j, j)));
    }
    p.setMixed(j, i, 'epsilon', p.get('epsilon', i, j));
    p.setMixed(j, i, 'cut', p.get('cut', i, j));
    // "since cut is a scaled by the mixed diameter, report maximum possible cutoff."
    return p.get('cut', i, j) * 2.0 * mixDistanceNative(this.mix, this.rmax[i], this.rmax[j]);
  }

  compute(pc: PairCompute): void {
    const list = pc.half!;
    const { x, f, type } = pc;
    const s = this.state!;
    const radius = s.radius!;
    const own = pc.nb.owner;
    const nt = this.ntypes + 1;
    const eps = this.p.p('epsilon'), cutp = this.p.p('cut');
    const { cutsq } = this;
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
          // cutsq is the maximum cutoff per type; the real cutoff follows from the mixed diameter
          const sigma = 2.0 * mixDistanceNative(this.mix, rtmp, radiusOfIndex(radius, own, j));
          const cs = cutp[t] * sigma;
          const rcutsq = cs * cs;
          if (rsq < rcutsq) {
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const sigma6 = powint(sigma, 6);
            const forcelj = r6inv * 24.0 * eps[t] * (2.0 * sigma6 * sigma6 * r6inv - sigma6);
            const fpair = factorLJ * forcelj * r2inv;
            f[3 * i] += delx * fpair;
            f[3 * i + 1] += dely * fpair;
            f[3 * i + 2] += delz * fpair;
            f[3 * j] -= delx * fpair;
            f[3 * j + 1] -= dely * fpair;
            f[3 * j + 2] -= delz * fpair;
            let e = r6inv * 4.0 * eps[t];
            e *= sigma6 * sigma6 * r6inv - sigma6;
            if (offset && rcutsq > 0.0) {
              const ratio6 = sigma6 / powint(rcutsq, 3);
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
    const s = this.state!;
    const radius = s.radius!;
    const eps = this.p.get('epsilon', itype, jtype);
    const cut = this.p.get('cut', itype, jtype);
    const sigma = 2.0 * mixDistanceNative(this.mix, radius[i], radius[j]);
    const cs = cut * sigma;
    const rcutsq = cs * cs;
    const sigma6 = powint(sigma, 6);
    const r2inv = 1.0 / rsq;
    const r6inv = r2inv * r2inv * r2inv;
    const forcelj = r6inv * 24.0 * eps * (2.0 * sigma6 * sigma6 * r6inv - sigma6);
    const fforce = factorLJ * forcelj * r2inv;
    let philj = r6inv * 4.0 * eps * (sigma6 * sigma6 * r6inv - sigma6);
    if (this.shift && rcutsq > 0.0) {
      const ratio6 = sigma6 / powint(rcutsq, 3);
      philj -= 4.0 * eps * (ratio6 * ratio6 - ratio6);
    }
    return { eng: factorLJ * philj, fforce };
  }

  dataCoeffs(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) out.push(`${i} ${fmtCoeff(this.p.get('epsilon', i, i))}`);
    return out;
  }

  dataCoeffsIJ(): string[] | null {
    const out: string[] = [];
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) {
        out.push(`${i} ${j} ${fmtCoeff(this.p.get('epsilon', i, j))} ${fmtCoeff(this.p.get('cut', i, j))}`);
      }
    }
    return out;
  }

  extract(name: string): unknown {
    if (name === 'epsilon') return this.p.p('epsilon');
    return undefined;
  }
}
