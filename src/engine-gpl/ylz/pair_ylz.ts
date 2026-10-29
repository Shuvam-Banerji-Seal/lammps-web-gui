/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ASPHERE/pair_ylz.cpp and src/ASPHERE/pair_ylz.h at commit b332f28 (contributing authors in the
 * C++ file: Hongyan Yuan (SUSTech), Zhaoyan Huang (SUSTech)); copyright (2003) Sandia Corporation (GNU GPL).
 * Reference in the C++ file: Yuan H, Huang C, Li J, Lykotrafitis G and Zhang S (2010), Phys. Rev. E 82, 011905.
 * PairYLZ::single (the same file at b332f28, for compute pair/local) is ported too: it recomputes the separation from the
 * positions (the engine passes it as del) and the analytic energy of the two ellipsoids' orientations.
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { Pair, PairParams, StyleError, mixDistance, mixEpsilon, type PairCompute, type PairSep, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { parseNum } from '../../engine/force/util';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

const MY_PI = 3.14159265358979323846;
const MY_PI2 = 1.57079632679489661923;
const MY_TWOBYSIXTH = 1.12246204830937298142;

/**
 * Rows of the transposed rotation matrices (MathExtra::quat_to_mat_trans): a[0] of an ellipsoid is its body x axis in
 * space, the first column of the body-to-space matrix R.
 */
const bodyX = (w: number, x: number, y: number, z: number, out: Float64Array): void => {
  const w2 = w * w, i2 = x * x, j2 = y * y, k2 = z * z;
  const twoij = 2.0 * x * y, twoik = 2.0 * x * z;
  const twojw = 2.0 * y * w, twokw = 2.0 * z * w;
  out[0] = w2 + i2 - j2 - k2;
  out[1] = twoij + twokw;
  out[2] = twoik - twojw;
};

/** The analytic energy, radial force and torques of ylz_analytic (pair_ylz.cpp), written into fforce, ttor, rtor. */
const ylzAnalytic = (eps: number, sig: number, zt: number, muu: number, sint: number, rcut: number,
  ni1: Float64Array, nj1: Float64Array, r12: Float64Array, rsq: number,
  fforce: Float64Array, ttor: Float64Array, rtor: Float64Array): number => {
  const r12hat = new Float64Array(3);
  const sc = 1.0 / Math.sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
  r12hat[0] = r12[0] * sc;
  r12hat[1] = r12[1] * sc;
  r12hat[2] = r12[2] * sc;
  const r = Math.sqrt(rsq);
  const energyWell = eps;
  const rmin = MY_TWOBYSIXTH * sig;
  const ninj = ni1[0] * nj1[0] + ni1[1] * nj1[1] + ni1[2] * nj1[2];
  const ni1rhat = ni1[0] * r12hat[0] + ni1[1] * r12hat[1] + ni1[2] * r12hat[2];
  const nj1rhat = nj1[0] * r12hat[0] + nj1[1] * r12hat[1] + nj1[2] * r12hat[2];
  const a = ninj + (sint - ni1rhat) * (sint + nj1rhat) - 2.0 * sint * sint;
  const phi = 1.0 + (a - 1.0) * muu;
  const dphiDrhat = new Float64Array(3), dphiDni1 = new Float64Array(3), dphiDnj1 = new Float64Array(3);
  for (let k = 0; k < 3; k++) {
    dphiDrhat[k] = muu * ((sint - ni1rhat) * nj1[k] - ni1[k] * (sint + nj1rhat));
    dphiDni1[k] = muu * (nj1[k] - r12hat[k] * (sint + nj1rhat));
    dphiDnj1[k] = muu * (ni1[k] + r12hat[k] * (sint - ni1rhat));
  }
  let U: number, dUdr: number, dUdphi: number;
  if (r < rmin) {
    const t = rmin / r;
    const t2 = t * t;
    const t4 = t2 * t2;
    const uR = (t4 - 2.0 * t2) * energyWell;
    U = uR + (1.0 - phi) * energyWell;
    dUdr = 4.0 * (t2 - t4) / r * energyWell;
    dUdphi = -energyWell;
  } else {
    const t = MY_PI2 * (r - rmin) / (rcut - rmin);
    const cosT = libmCos(t);
    let t1 = cosT;
    for (let k = 1; k <= 2 * zt - 2; k++) t1 *= cosT; // cos()^(2zt-1)
    const uA = -energyWell * t1 * cosT;
    U = uA * phi;
    dUdr = MY_PI * zt / (rcut - rmin) * t1 * libmSin(t) * phi * energyWell;
    dUdphi = uA;
  }
  const dUdrhat = new Float64Array(3);
  for (let k = 0; k < 3; k++) dUdrhat[k] = dUdphi * dphiDrhat[k];
  const dUdrhatrhat = dUdrhat[0] * r12hat[0] + dUdrhat[1] * r12hat[1] + dUdrhat[2] * r12hat[2];
  for (let k = 0; k < 3; k++) fforce[k] = dUdr * r12hat[k] + (dUdrhat[k] - dUdrhatrhat * r12hat[k]) / r;
  const dUdni1 = [dUdphi * dphiDni1[0], dUdphi * dphiDni1[1], dUdphi * dphiDni1[2]];
  // cross3(dUdni1, ni1) (the minus sign of the C++ comment is absorbed by the operand order)
  ttor[0] = dUdni1[1] * ni1[2] - dUdni1[2] * ni1[1];
  ttor[1] = dUdni1[2] * ni1[0] - dUdni1[0] * ni1[2];
  ttor[2] = dUdni1[0] * ni1[1] - dUdni1[1] * ni1[0];
  const dUdnj1 = [dUdphi * dphiDnj1[0], dUdphi * dphiDnj1[1], dUdphi * dphiDnj1[2]];
  rtor[0] = dUdnj1[1] * nj1[2] - dUdnj1[2] * nj1[1];
  rtor[1] = dUdnj1[2] * nj1[0] - dUdnj1[0] * nj1[2];
  rtor[2] = dUdnj1[0] * nj1[1] - dUdnj1[1] * nj1[0];
  return U;
};

/**
 * Pair style ylz: the one-particle-thick membrane model of ellipsoids (pair_ylz.cpp). pair_style ylz cutoff;
 * pair_coeff I J epsilon sigma zeta mu beta cutoff. Every atom must be an ellipsoid (atom_style ellipsoid).
 */
export class PairYLZ extends Pair {
  readonly name = 'ylz';
  needsHalf = true;
  needsFull = false;
  virialFdotr = true;
  private p = new PairParams(0, ['epsilon', 'sigma', 'zeta', 'mu', 'beta', 'cut']);
  private cutGlobal = 0;
  private allocatedTypes = false;
  /** the simulation state whose quaternions single() reads (set at init and at each compute) */
  private simState: PairCompute['s'] | null = null;

  override allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.p = new PairParams(ntypes, ['epsilon', 'sigma', 'zeta', 'mu', 'beta', 'cut']);
    this.allocatedTypes = true;
  }

  override settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Illegal pair_style ylz command');
    this.cutGlobal = parseNum(args[0], 'cutoff');
  }

  override coeff(args: string[], _ctx: StyleContext): void {
    if (!this.allocatedTypes) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 8) throw new StyleError('Incorrect args for pair coefficients');
    const vals = [2, 3, 4, 5, 6, 7].map((k) => parseNum(args[k], 'pair_coeff value'));
    // ylz sets the pairs of the I and J ranges with I <= J (pair_coeff in pair_ylz.cpp)
    const count = this.p.setRange(args[0], args[1], vals);
    if (count === 0) throw new StyleError('Incorrect args for pair coefficients');
  }

  override initStyle(ctx: StyleContext): void {
    if (!ctx.s) throw new StyleError('pair_style ylz needs a box');
    this.simState = ctx.s;
  }

  /** The separation vector is needed (del = x_i - x_j), as for the ellipsoid styles (pair_gayberne.ts). */
  override singleNeedsSeparation(): boolean { return true; }

  /** Native single_extra = 6 (pair_ylz.cpp constructor): svector holds the force and the torque on i (pN 1..6). */
  override singleExtra(): number { return 6; }
  private readonly svec = new Float64Array(6);

  /**
   * PairYLZ::single (pair_ylz.cpp at b332f28): r12 = x[j] - x[i] (here -del), the two body matrices from the quaternions
   * of i and j, and ylz_analytic; "svector[0] = factor_lj * fvec[0]" ... "svector[5] = factor_lj * ttor[2]",
   * "fforce = factor_lj * (-MathExtra::dot3(fvec, r12) / rsq)"
   * and "return factor_lj * one_eng".
   */
  override single(i: number, j: number, itype: number, jtype: number, rsq: number, _factorCoul: number, factorLJ: number, _qi = 0, _qj = 0, del?: PairSep): { eng: number; fforce: number; svector: Float64Array } {
    const s = this.simState;
    if (!s || !s.quat) throw new StyleError('pair ylz single() used before pair_style');
    if (!del) throw new StyleError('pair ylz single() needs the separation vector');
    const quat = s.quat;
    const a1 = new Float64Array(3), a2 = new Float64Array(3);
    bodyX(quat[4 * i], quat[4 * i + 1], quat[4 * i + 2], quat[4 * i + 3], a1);
    bodyX(quat[4 * j], quat[4 * j + 1], quat[4 * j + 2], quat[4 * j + 3], a2);
    const r12 = new Float64Array([-del.dx, -del.dy, -del.dz]);
    const eps = this.p.get('epsilon', itype, jtype), sig = this.p.get('sigma', itype, jtype);
    const zt = this.p.get('zeta', itype, jtype), muu = this.p.get('mu', itype, jtype);
    const sint = this.p.get('beta', itype, jtype), rcut = this.p.get('cut', itype, jtype);
    const fvec = new Float64Array(3), ttor = new Float64Array(3), rtor = new Float64Array(3);
    const oneEng = ylzAnalytic(eps, sig, zt, muu, sint, rcut, a1, a2, r12, rsq, fvec, ttor, rtor);
    const sv = this.svec;
    sv[0] = factorLJ * fvec[0]; sv[1] = factorLJ * fvec[1]; sv[2] = factorLJ * fvec[2];
    sv[3] = factorLJ * ttor[0]; sv[4] = factorLJ * ttor[1]; sv[5] = factorLJ * ttor[2];
    const dot = fvec[0] * r12[0] + fvec[1] * r12[1] + fvec[2] * r12[2];
    return { eng: factorLJ * oneEng, fforce: factorLJ * (-dot / rsq), svector: sv };
  }

  override initOne(i: number, j: number): number {
    if (!this.p.isSet(i, j)) {
      if (!this.p.isSet(i, i) || !this.p.isSet(j, j)) throw new StyleError(`All pair coeffs are not set (pair_style ylz, types ${i} ${j})`);
      const ei = this.p.get('epsilon', i, i), ej = this.p.get('epsilon', j, j);
      const si = this.p.get('sigma', i, i), sj = this.p.get('sigma', j, j);
      const ci = this.p.get('cut', i, i), cj = this.p.get('cut', j, j);
      this.p.setMixed(i, j, 'epsilon', mixEpsilon(this.mix, ei, ej, si, sj));
      this.p.setMixed(i, j, 'sigma', mixDistance(this.mix, si, sj));
      this.p.setMixed(i, j, 'cut', mixDistance(this.mix, ci, cj));
    }
    return this.p.get('cut', i, j);
  }

  override compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.shape || !s.quat || !s.torque) throw new StyleError('Pair style ylz requires atom style ellipsoid');
    this.simState = s;
    const nlocal = pc.nlocal, nall = pc.nall;
    const shape = s.shape, quat = s.quat;
    const owner = pc.nb.owner;
    const { x, f, type } = pc;
    const list = pc.half!;
    const tq = new Float64Array(3 * nall);
    const sL = pc.specialLJ;
    const cutsq = this.cutsq;
    const nt = this.ntypes + 1;
    // every atom must be an ellipsoid (pair_ylz.cpp: "All atoms for pair style ylz must be ellipsoids")
    for (let a = 0; a < nlocal; a++) {
      if (!(shape[3 * owner[a]] > 0)) throw new StyleError('All atoms for pair style ylz must be ellipsoids');
    }
    const a1 = new Float64Array(3), a2 = new Float64Array(3);
    const fforce = new Float64Array(3), ttor = new Float64Array(3), rtor = new Float64Array(3);
    const r12 = new Float64Array(3);
    let evdwl = 0;
    for (let ii = 0; ii < list.inum; ii++) {
      const i = ii;
      const oi = owner[i];
      const itype = type[i];
      bodyX(quat[4 * oi], quat[4 * oi + 1], quat[4 * oi + 2], quat[4 * oi + 3], a1);
      for (let k = list.firstneigh[i], k1 = k + list.numneigh[i]; k < k1; k++) {
        const jj = list.neighbors[k];
        const j = jj & NEIGHMASK;
        const oj = owner[j];
        const factorLJ = sL[jj >>> SBBITS];
        r12[0] = x[3 * j] - x[3 * i];
        r12[1] = x[3 * j + 1] - x[3 * i + 1];
        r12[2] = x[3 * j + 2] - x[3 * i + 2];
        const rsq = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        const jtype = type[j];
        if (!(rsq < cutsq[itype * nt + jtype])) continue;
        bodyX(quat[4 * oj], quat[4 * oj + 1], quat[4 * oj + 2], quat[4 * oj + 3], a2);
        const eps = this.p.get('epsilon', itype, jtype), sig = this.p.get('sigma', itype, jtype);
        const zt = this.p.get('zeta', itype, jtype), muu = this.p.get('mu', itype, jtype);
        const sint = this.p.get('beta', itype, jtype), rcut = this.p.get('cut', itype, jtype);
        const oneEng = ylzAnalytic(eps, sig, zt, muu, sint, rcut, a1, a2, r12, rsq, fforce, ttor, rtor);
        for (let d = 0; d < 3; d++) {
          fforce[d] *= factorLJ;
          ttor[d] *= factorLJ;
          rtor[d] *= factorLJ;
        }
        f[3 * i] += fforce[0]; f[3 * i + 1] += fforce[1]; f[3 * i + 2] += fforce[2];
        tq[3 * i] += ttor[0]; tq[3 * i + 1] += ttor[1]; tq[3 * i + 2] += ttor[2];
        f[3 * j] -= fforce[0]; f[3 * j + 1] -= fforce[1]; f[3 * j + 2] -= fforce[2];
        tq[3 * j] += rtor[0]; tq[3 * j + 1] += rtor[1]; tq[3 * j + 2] += rtor[2];
        const e = factorLJ * oneEng;
        evdwl += e;
        if (pc.eatom) {
          pc.eatom[i] += 0.5 * e;
          pc.eatom[j] += 0.5 * e;
        }
        // ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0, fforce, -r12): half of (-r12) (x) fforce to each atom
        if (pc.vatom) {
          const va = pc.vatom;
          const dx = -r12[0], dy = -r12[1], dz = -r12[2];
          const w = [dx * fforce[0], dy * fforce[1], dz * fforce[2], dx * fforce[1], dx * fforce[2], dy * fforce[2]];
          for (let c = 0; c < 6; c++) { va[6 * i + c] += 0.5 * w[c]; va[6 * j + c] += 0.5 * w[c]; }
        }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
