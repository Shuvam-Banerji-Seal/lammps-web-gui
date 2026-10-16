/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_line_lj.cpp and src/ASPHERE/pair_line_lj.h at commit b332f28 (pair_style line/lj: a
 * Lennard-Jones interaction between 2d line segments, discretized into sub-particles, and between segments and
 * point particles). The sources carry no contributing-author line; copyright Sandia Corporation (LAMMPS
 * development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_line_lj.html: "The set of non-overlapping spherical sub-particles that represent a
 * line segment are generated in the following manner. Their size is a function of the line segment length and the
 * specified sub-particle size for that particle type" (size_i, size_j per type).
 *
 * Browser-engine difference: the per-atom line data (length, theta) come from the core custom properties
 * (atoms.ts LINE_BONUS, LINE_FLAG) of the owner atom, not from atom->bonus; the forces and torques of ghost
 * sub-particles go to the owner through the neighbour list's reverse sum, and the virial is the fdotr sum.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { isLineStyle, LINE_BONUS, LINE_FLAG } from '../../engine/atoms';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

export class PairLineLJ extends Pair {
  readonly name = 'line/lj';
  /** The virial is the fdotr sum (pair_line_lj.cpp: virial_fdotr_compute). */
  virialFdotr = true;
  /** Global cutoff of pair_style line/lj (the cutoff of pair_coeff without its optional cutoff). */
  private cutGlobal = 0;
  /** Per type: size of the discretization sub-particles (pair_coeff size_i and size_j). */
  private subsize = new Float64Array(0);
  /** Per type pair (i*nt + j, i <= j): epsilon, sigma, sub-particle cutoff, cutoff, set flag. */
  private eps = new Float64Array(0);
  private sig = new Float64Array(0);
  private cutsub = new Float64Array(0);
  private cutIJ = new Float64Array(0);
  private setflag = new Uint8Array(0);
  private cutsubsq = new Float64Array(0);
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // pair_line_lj.cpp settings: the cutoffs of the pairs set explicitly are reset to the global cutoff
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) if (this.setflag[i * nt + j]) this.cutIJ[i * nt + j] = this.cutGlobal;
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nt = ntypes + 1;
    this.subsize = new Float64Array(nt);
    for (const name of ['eps', 'sig', 'cutsub', 'cutIJ', 'cutsubsq', 'lj1', 'lj2', 'lj3', 'lj4'] as const) this[name] = new Float64Array(nt * nt);
    this.setflag = new Uint8Array(nt * nt);
  }

  /** pair_coeff I J size_i size_j epsilon sigma cutsub [cutoff] (pair_line_lj.html). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 7 || args.length > 8) throw new StyleError('Incorrect args for pair coefficients');
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const sizeI = parseNum(args[2], 'size'), sizeJ = parseNum(args[3], 'size');
    const epsOne = parseNum(args[4], 'epsilon'), sigOne = parseNum(args[5], 'sigma');
    const cutsubOne = parseNum(args[6], 'cutsub');
    const cutOne = args.length === 8 ? parseNum(args[7], 'cutoff') : this.cutGlobal;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        this.subsize[i] = sizeI;
        this.subsize[j] = sizeJ;
        const k = i * nt + j;
        this.eps[k] = epsOne;
        this.sig[k] = sigOne;
        this.cutsub[k] = cutsubOne;
        this.cutIJ[k] = cutOne;
        this.setflag[k] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    if (!ctx.s || !isLineStyle(ctx.s.atomStyle)) throw new StyleError('Pair line/lj requires atom style line');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const k = i * nt + j, kr = j * nt + i;
    if (!this.setflag[k]) throw new StyleError('All pair coeffs are not set');
    const e = this.eps[k], s = this.sig[k];
    this.cutsubsq[k] = this.cutsub[k] * this.cutsub[k];
    this.lj1[k] = 48.0 * e * Math.pow(s, 12.0);
    this.lj2[k] = 24.0 * e * Math.pow(s, 6.0);
    this.lj3[k] = 4.0 * e * Math.pow(s, 12.0);
    this.lj4[k] = 4.0 * e * Math.pow(s, 6.0);
    this.eps[kr] = e;
    this.sig[kr] = s;
    this.cutsubsq[kr] = this.cutsubsq[k];
    this.lj1[kr] = this.lj1[k];
    this.lj2[kr] = this.lj2[k];
    this.lj3[kr] = this.lj3[k];
    this.lj4[kr] = this.lj4[k];
    return this.cutIJ[k];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const bonus = s.custom.get(LINE_BONUS)?.data;
    const flag = s.custom.get(LINE_FLAG)?.data;
    if (!bonus || !flag || !s.torque) throw new StyleError('Pair line/lj requires atom style line');
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const nbl = list.neighbors;
    const owner = nb.owner;
    const nt = this.ntypes + 1;
    const nlocal = pc.nlocal, nall = pc.nall;
    const tq = new Float64Array(3 * nall);

    // discretize (pair_line_lj.cpp discretize): a segment of length L is cut into n = int(L / size) + 1 pieces whose
    // centres sit at delta * L along the segment, delta = -1/2 + (2m + 1) / (2n); stored per owned atom
    const dStart = new Int32Array(nlocal), dCount = new Int32Array(nlocal);
    let ndisc = 0;
    for (let i = 0; i < nlocal; i++) {
      if (!flag[i]) continue;
      const size = this.subsize[type[i]];
      dCount[i] = Math.trunc(bonus[2 * i] / size) + 1;
      dStart[i] = ndisc;
      ndisc += dCount[i];
    }
    const ddx = new Float64Array(ndisc), ddy = new Float64Array(ndisc);
    for (let i = 0; i < nlocal; i++) {
      if (!flag[i]) continue;
      const L = bonus[2 * i], th = bonus[2 * i + 1], n = dCount[i];
      for (let m = 0; m < n; m++) {
        const delta = -0.5 + (2 * m + 1) / (2.0 * n);
        ddx[dStart[i] + m] = delta * L * libmCos(th);
        ddy[dStart[i] + m] = delta * L * libmSin(th);
      }
    }

    const { cutsq } = this;
    let evdwl = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = nbl[k] & NEIGHMASK;
        let delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1];
        const delz = ztmp - x[3 * j + 2];
        let rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        const kt = itype * nt + jtype;
        if (rsq >= cutsq[kt]) continue;
        const oi = owner[i], oj = owner[j];
        const lineI = flag[oi] !== 0, lineJ = flag[oj] !== 0;
        // term2 = 24 eps sigma^6 and term1 = 2 term2 sigma^6 (the sub-particle LJ coefficients of pair_line_lj.cpp)
        const cutsubsq = this.cutsubsq[kt];
        const sig = this.sig[kt], epsv = this.eps[kt];
        const sig3 = sig * sig * sig;
        const t2 = 24.0 * epsv * sig3 * sig3;
        const t1 = 2.0 * t2 * sig3 * sig3;
        if (lineI && lineJ) {
          // line/line: N x N sub-particle pairs
          const npi = dCount[oi], ifirst = dStart[oi], npj = dCount[oj], jfirst = dStart[oj];
          for (let ni = 0; ni < npi; ni++) {
            const dxi = ddx[ifirst + ni], dyi = ddy[ifirst + ni];
            for (let nj = 0; nj < npj; nj++) {
              const dxj = ddx[jfirst + nj], dyj = ddy[jfirst + nj];
              delx = (x[3 * i] + dxi) - (x[3 * j] + dxj);
              dely = (x[3 * i + 1] + dyi) - (x[3 * j + 1] + dyj);
              rsq = delx * delx + dely * dely;
              if (rsq >= cutsubsq) continue;
              const r2inv = 1.0 / rsq;
              const r6inv = r2inv * r2inv * r2inv;
              const forcelj = r6inv * (t1 * r6inv - t2);
              const fpair = forcelj * r2inv;
              evdwl += r6inv * (t1 / 12.0 * r6inv - t2 / 6.0);
              const fi0 = delx * fpair, fi1 = dely * fpair;
              f[3 * i] += fi0;
              f[3 * i + 1] += fi1;
              tq[3 * i + 2] += dxi * fi1 - dyi * fi0;
              f[3 * j] -= fi0;
              f[3 * j + 1] -= fi1;
              tq[3 * j + 2] -= dxj * fi1 - dyj * fi0;
            }
          }
        } else if (lineI) {
          // line/particle: the segment of i is discretized, j is a point
          const npi = dCount[oi], ifirst = dStart[oi];
          for (let ni = 0; ni < npi; ni++) {
            const dxi = ddx[ifirst + ni], dyi = ddy[ifirst + ni];
            delx = (x[3 * i] + dxi) - x[3 * j];
            dely = (x[3 * i + 1] + dyi) - x[3 * j + 1];
            rsq = delx * delx + dely * dely;
            if (rsq >= cutsubsq) continue;
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forcelj = r6inv * (t1 * r6inv - t2);
            const fpair = forcelj * r2inv;
            evdwl += r6inv * (t1 / 12.0 * r6inv - t2 / 6.0);
            const fi0 = delx * fpair, fi1 = dely * fpair;
            f[3 * i] += fi0;
            f[3 * i + 1] += fi1;
            tq[3 * i + 2] += dxi * fi1 - dyi * fi0;
            f[3 * j] -= fi0;
            f[3 * j + 1] -= fi1;
          }
        } else if (lineJ) {
          // particle/line: the point i against the discretized segment of j
          const npj = dCount[oj], jfirst = dStart[oj];
          for (let nj = 0; nj < npj; nj++) {
            const dxj = ddx[jfirst + nj], dyj = ddy[jfirst + nj];
            delx = x[3 * i] - (x[3 * j] + dxj);
            dely = x[3 * i + 1] - (x[3 * j + 1] + dyj);
            rsq = delx * delx + dely * dely;
            if (rsq >= cutsubsq) continue;
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forcelj = r6inv * (t1 * r6inv - t2);
            const fpair = forcelj * r2inv;
            evdwl += r6inv * (t1 / 12.0 * r6inv - t2 / 6.0);
            const fi0 = delx * fpair, fi1 = dely * fpair;
            f[3 * i] += fi0;
            f[3 * i + 1] += fi1;
            f[3 * j] -= fi0;
            f[3 * j + 1] -= fi1;
            tq[3 * j + 2] -= dxj * fi1 - dyj * fi0;
          }
        } else {
          // point/point: the plain LJ pair at the centres (lj1..lj4 of initOne)
          const r2inv = 1.0 / rsq;
          const r6inv = r2inv * r2inv * r2inv;
          const forcelj = r6inv * (this.lj1[kt] * r6inv - this.lj2[kt]);
          const fpair = forcelj * r2inv;
          evdwl += r6inv * (this.lj3[kt] * r6inv - this.lj4[kt]);
          f[3 * i] += delx * fpair;
          f[3 * i + 1] += dely * fpair;
          f[3 * i + 2] += delz * fpair;
          f[3 * j] -= delx * fpair;
          f[3 * j + 1] -= dely * fpair;
          f[3 * j + 2] -= delz * fpair;
        }
      }
    }
    nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
