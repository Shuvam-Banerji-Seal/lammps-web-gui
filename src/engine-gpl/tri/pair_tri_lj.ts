/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_tri_lj.cpp and src/ASPHERE/pair_tri_lj.h at commit b332f28 (pair_style tri/lj: a
 * Lennard-Jones interaction between 3d triangular particles, discretized into sub-particles, and between triangles
 * and point particles). The sources carry no contributing-author line; copyright Sandia Corporation (LAMMPS
 * development team, GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/pair_tri_lj.html: "Style *tri/lj* treats particles which are triangles as a set of small spherical particles that tile the triangle surface"; the triangle is recursively bisected until the sub-particles fit the
 * sigma of the pair coefficient, and each sub-particle carries its own size (the diameter of its circumcircle).
 *
 * Browser-engine difference: the corner points and principal frame of each triangle are the core custom properties
 * (atoms.ts TRI_BONUS, the corners in the principal frame; quat from the per-atom quaternion) of the owner atom.
 * The torque of a sub-particle goes to the owner through the neighbour list's reverse sum.
 */
import { Pair, StyleError, mixDistance, mixEpsilon, typeBounds, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { isTriStyle, TRI_BONUS, TRI_BONUS_COLS, TRI_FLAG } from '../../engine/atoms';
import { quatToMatrix } from '../../engine/compute/asphere';
import { pow as libmPow } from '../libm/pow';

interface Disc { dx: number; dy: number; dz: number; sigma: number }
type V3 = [number, number, number];

/**
 * Recursive discretization (pair_tri_lj.cpp discretize, the active version): when the sphere of radius sigma / 2
 * about the centroid covers all three corners, one sub-particle sits at the centroid with sigma = 2 * circumradius;
 * otherwise the triangle is split at the midpoint of its longest side into two triangles and each is discretized.
 */
const discretizeTri = (sigma: number, c1: V3, c2: V3, c3: V3, out: Disc[]): void => {
  const cen: V3 = [(c1[0] + c2[0] + c3[0]) / 3.0, (c1[1] + c2[1] + c3[1]) / 3.0, (c1[2] + c2[2] + c3[2]) / 3.0];
  const sigmasq = 0.25 * sigma * sigma;
  const lsq = (a: V3, b: V3): number => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
  const len1sq = lsq(c1, cen), len2sq = lsq(c2, cen), len3sq = lsq(c3, cen);
  if (len1sq <= sigmasq && len2sq <= sigmasq && len3sq <= sigmasq) {
    out.push({ dx: cen[0], dy: cen[1], dz: cen[2], sigma: 2.0 * Math.sqrt(Math.max(len1sq, len2sq, len3sq)) });
    return;
  }
  const mid = (a: V3, b: V3): V3 => [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
  const e1 = lsq(c2, c3), e2 = lsq(c1, c3), e3 = lsq(c1, c2);
  const maxsq = Math.max(e1, e2, e3);
  if (e1 === maxsq) {
    const m = mid(c2, c3);
    discretizeTri(sigma, c1, c2, m, out);
    discretizeTri(sigma, c1, c3, m, out);
  } else if (e2 === maxsq) {
    const m = mid(c1, c3);
    discretizeTri(sigma, c2, c1, m, out);
    discretizeTri(sigma, c2, c3, m, out);
  } else {
    const m = mid(c1, c2);
    discretizeTri(sigma, c3, c1, m, out);
    discretizeTri(sigma, c3, c2, m, out);
  }
};

export class PairTriLJ extends Pair {
  readonly name = 'tri/lj';
  /** The virial is the fdotr sum (pair_tri_lj.cpp: virial_fdotr_compute). */
  virialFdotr = true;
  private cutGlobal = 0;
  /** Per type pair (i*nt + j, i <= j): epsilon, sigma, cutoff, set flag; lj1..lj4 as in pair_tri_lj.cpp init_one. */
  private eps = new Float64Array(0);
  private sig = new Float64Array(0);
  private cutIJ = new Float64Array(0);
  private setflag = new Uint8Array(0);
  private lj1 = new Float64Array(0);
  private lj2 = new Float64Array(0);
  private lj3 = new Float64Array(0);
  private lj4 = new Float64Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
    // pair_tri_lj.cpp settings: the cutoffs of the pairs set explicitly are reset to the global cutoff
    const nt = this.ntypes + 1;
    for (let i = 1; i <= this.ntypes; i++) {
      for (let j = i; j <= this.ntypes; j++) if (this.setflag[i * nt + j]) this.cutIJ[i * nt + j] = this.cutGlobal;
    }
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    const nn = (ntypes + 1) * (ntypes + 1);
    for (const name of ['eps', 'sig', 'cutIJ', 'lj1', 'lj2', 'lj3', 'lj4'] as const) this[name] = new Float64Array(nn);
    this.setflag = new Uint8Array(nn);
  }

  /** pair_coeff I J epsilon sigma [cutoff] (pair_tri_lj.html). */
  coeff(args: string[], _ctx: StyleContext): void {
    if (args.length < 4 || args.length > 5) throw new StyleError('Incorrect args for pair coefficients');
    const nt = this.ntypes + 1;
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const epsOne = parseNum(args[2], 'epsilon'), sigOne = parseNum(args[3], 'sigma');
    const cutOne = args.length === 5 ? parseNum(args[4], 'cutoff') : this.cutGlobal;
    let count = 0;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const k = i * nt + j;
        this.eps[k] = epsOne;
        this.sig[k] = sigOne;
        this.cutIJ[k] = cutOne;
        this.setflag[k] = 1;
        count++;
      }
    }
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  initStyle(ctx: StyleContext): void {
    if (!ctx.s || !isTriStyle(ctx.s.atomStyle)) throw new StyleError('Pair tri/lj requires atom style tri');
  }

  initOne(i: number, j: number): number {
    const nt = this.ntypes + 1;
    const k = i * nt + j, kii = i * nt + i, kjj = j * nt + j, kr = j * nt + i;
    if (!this.setflag[k]) {
      // pair_tri_lj.cpp init_one: unset pairs mix (mix_energy and mix_distance of the diagonal values)
      if (i === j) throw new StyleError('All pair coeffs are not set');
      this.eps[k] = mixEpsilon(this.mix, this.eps[kii], this.eps[kjj], this.sig[kii], this.sig[kjj]);
      this.sig[k] = mixDistance(this.mix, this.sig[kii], this.sig[kjj]);
      this.cutIJ[k] = mixDistance(this.mix, this.cutIJ[kii], this.cutIJ[kjj]);
    }
    const e = this.eps[k], s = this.sig[k];
    this.lj1[k] = 48.0 * e * libmPow(s, 12.0);
    this.lj2[k] = 24.0 * e * libmPow(s, 6.0);
    this.lj3[k] = 4.0 * e * libmPow(s, 12.0);
    this.lj4[k] = 4.0 * e * libmPow(s, 6.0);
    this.eps[kr] = e;
    this.sig[kr] = s;
    this.lj1[kr] = this.lj1[k];
    this.lj2[kr] = this.lj2[k];
    this.lj3[kr] = this.lj3[k];
    this.lj4[kr] = this.lj4[k];
    return this.cutIJ[k];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const bonus = s.custom.get(TRI_BONUS)?.data;
    const flag = s.custom.get(TRI_FLAG)?.data;
    if (!bonus || !flag || !s.torque || !s.quat) throw new StyleError('Pair tri/lj requires atom style tri');
    const quat = s.quat;
    const list = pc.half!;
    const { x, f, type } = pc;
    const nb = pc.nb;
    const nbl = list.neighbors;
    const owner = nb.owner;
    const nt = this.ntypes + 1;
    const tq = new Float64Array(3 * pc.nall);

    // sub-particles of each owned triangle (pair_tri_lj.cpp: dnum/dfirst cache), built on first use; orientation
    // invariant, so a ghost image shares its owner's list. The corners in space are R(quat) times the body corners.
    const cache = new Map<number, Disc[]>();
    const points = (o: number): Disc[] => {
      let pts = cache.get(o);
      if (pts) return pts;
      pts = [];
      const q = [quat[4 * o], quat[4 * o + 1], quat[4 * o + 2], quat[4 * o + 3]];
      const R = quatToMatrix(q[0], q[1], q[2], q[3]);
      const b = (k: number): V3 => {
        const c = [bonus[TRI_BONUS_COLS * o + 3 * k], bonus[TRI_BONUS_COLS * o + 3 * k + 1], bonus[TRI_BONUS_COLS * o + 3 * k + 2]];
        return [0, 1, 2].map((r) => R[3 * r] * c[0] + R[3 * r + 1] * c[1] + R[3 * r + 2] * c[2]) as V3;
      };
      discretizeTri(this.sig[type[o] * nt + type[o]], b(0), b(1), b(2), pts);
      cache.set(o, pts);
      return pts;
    };

    const { cutsq } = this;
    let evdwl = 0;
    for (let i = list.ilo ?? 0; i < list.inum; i++) {
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const itype = type[i];
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const j = nbl[k] & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq0 = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        const kt = itype * nt + jtype;
        if (rsq0 >= cutsq[kt]) continue;
        const oi = owner[i], oj = owner[j];
        const triI = flag[oi] !== 0, triJ = flag[oj] !== 0;
        const epsv = this.eps[kt];
        if (triI && triJ) {
          // triangle/triangle: N x N sub-particle pairs (no sub-cutoff, as in the source)
          const pi = points(oi), pj = points(oj);
          for (const a of pi) {
            for (const b of pj) {
              const dxi = a.dx, dyi = a.dy, dzi = a.dz, dxj = b.dx, dyj = b.dy, dzj = b.dz;
              const ddx = (x[3 * i] + dxi) - (x[3 * j] + dxj);
              const ddy = (x[3 * i + 1] + dyi) - (x[3 * j + 1] + dyj);
              const ddz = (x[3 * i + 2] + dzi) - (x[3 * j + 2] + dzj);
              const rsq = ddx * ddx + ddy * ddy + ddz * ddz;
              const sg = 0.5 * (a.sigma + b.sigma);
              const sig3 = sg * sg * sg;
              const term2 = 24.0 * epsv * sig3 * sig3;
              const term1 = 2.0 * term2 * sig3 * sig3;
              const r2inv = 1.0 / rsq;
              const r6inv = r2inv * r2inv * r2inv;
              const forcelj = r6inv * (term1 * r6inv - term2);
              const fpair = forcelj * r2inv;
              evdwl += r6inv * (term1 / 12.0 * r6inv - term2 / 6.0);
              const fi0 = ddx * fpair, fi1 = ddy * fpair, fi2 = ddz * fpair;
              f[3 * i] += fi0; f[3 * i + 1] += fi1; f[3 * i + 2] += fi2;
              tq[3 * i] += dyi * fi2 - dzi * fi1;
              tq[3 * i + 1] += dzi * fi0 - dxi * fi2;
              tq[3 * i + 2] += dxi * fi1 - dyi * fi0;
              const fj0 = -ddx * fpair, fj1 = -ddy * fpair, fj2 = -ddz * fpair;
              f[3 * j] += fj0; f[3 * j + 1] += fj1; f[3 * j + 2] += fj2;
              tq[3 * j] += dyj * fj2 - dzj * fj1;
              tq[3 * j + 1] += dzj * fj0 - dxj * fj2;
              tq[3 * j + 2] += dxj * fj1 - dyj * fj0;
            }
          }
        } else if (triI) {
          // triangle/point: the sub-particles of i against the point j
          for (const a of points(oi)) {
            const dxi = a.dx, dyi = a.dy, dzi = a.dz;
            const ddx = (x[3 * i] + dxi) - x[3 * j];
            const ddy = (x[3 * i + 1] + dyi) - x[3 * j + 1];
            const ddz = (x[3 * i + 2] + dzi) - x[3 * j + 2];
            const rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            const sg = 0.5 * (a.sigma + this.sig[jtype * nt + jtype]);
            const sig3 = sg * sg * sg;
            const term2 = 24.0 * epsv * sig3 * sig3;
            const term1 = 2.0 * term2 * sig3 * sig3;
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forcelj = r6inv * (term1 * r6inv - term2);
            const fpair = forcelj * r2inv;
            evdwl += r6inv * (term1 / 12.0 * r6inv - term2 / 6.0);
            const fi0 = ddx * fpair, fi1 = ddy * fpair, fi2 = ddz * fpair;
            f[3 * i] += fi0; f[3 * i + 1] += fi1; f[3 * i + 2] += fi2;
            tq[3 * i] += dyi * fi2 - dzi * fi1;
            tq[3 * i + 1] += dzi * fi0 - dxi * fi2;
            tq[3 * i + 2] += dxi * fi1 - dyi * fi0;
            f[3 * j] -= fi0; f[3 * j + 1] -= fi1; f[3 * j + 2] -= fi2;
          }
        } else if (triJ) {
          // point/triangle: the point i against the sub-particles of j
          const sigI = this.sig[itype * nt + itype];
          for (const b of points(oj)) {
            const dxj = b.dx, dyj = b.dy, dzj = b.dz;
            const ddx = x[3 * i] - (x[3 * j] + dxj);
            const ddy = x[3 * i + 1] - (x[3 * j + 1] + dyj);
            const ddz = x[3 * i + 2] - (x[3 * j + 2] + dzj);
            const rsq = ddx * ddx + ddy * ddy + ddz * ddz;
            const sg = 0.5 * (sigI + b.sigma);
            const sig3 = sg * sg * sg;
            const term2 = 24.0 * epsv * sig3 * sig3;
            const term1 = 2.0 * term2 * sig3 * sig3;
            const r2inv = 1.0 / rsq;
            const r6inv = r2inv * r2inv * r2inv;
            const forcelj = r6inv * (term1 * r6inv - term2);
            const fpair = forcelj * r2inv;
            evdwl += r6inv * (term1 / 12.0 * r6inv - term2 / 6.0);
            const fi0 = ddx * fpair, fi1 = ddy * fpair, fi2 = ddz * fpair;
            f[3 * i] += fi0; f[3 * i + 1] += fi1; f[3 * i + 2] += fi2;
            const fj0 = -fi0, fj1 = -fi1, fj2 = -fi2;
            f[3 * j] += fj0; f[3 * j + 1] += fj1; f[3 * j + 2] += fj2;
            tq[3 * j] += dyj * fj2 - dzj * fj1;
            tq[3 * j + 1] += dzj * fj0 - dxj * fj2;
            tq[3 * j + 2] += dxj * fj1 - dyj * fj0;
          }
        } else {
          // point/point: the plain LJ pair at the centres (lj1..lj4 of initOne)
          const r2inv = 1.0 / rsq0;
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
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
