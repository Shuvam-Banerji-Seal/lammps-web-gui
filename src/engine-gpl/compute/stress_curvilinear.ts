/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-COMPUTE/compute_stress_cylinder.cpp and src/EXTRA-COMPUTE/compute_stress_spherical.cpp at commit
 * b332f28. Authors credited in the C++ files: Cody K. Addington (North Carolina State University; cylinder), Olav
 * Galteland (Norwegian University of Science and Technology; cylinder kinetic part and spherical). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * compute stress/cylinder and compute stress/spherical (registry/stress_curvilinear.ts). The loops follow the C++
 * statement by statement, including the bin assignment of each pair (a pair is split at the shell edges it crosses,
 * as the C++ intersects the segment with each shell) and the spherical kinetic rules (no mass, no nktv2p factor).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { massOf } from '../../engine/atoms';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import type { System } from '../../engine/system';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** One pair of the half list that passes the group test and the cutoff: positions of i and j (image), fpair. */
interface PairRow { xi: number; yi: number; zi: number; xj: number; yj: number; zj: number; fpair: number }

/**
 * Pairs with both atoms in the group inside the cutoff, in native's pair order: the C++ takes the pair as (i, j) of its
 * half list, and the segment split of the spherical integral depends on which atom is i (the loop is over ilist and
 * firstneigh). The order comes from nativeOrder.ts.
 */
function groupPairs(sys: System, groupBit: number, order: NativePairOrder): PairRow[] {
  const pair = sys.ff.pair;
  if (!pair || !pair.single) throw new StyleError('compute stress profile needs a pair_style that supports single() (the engine\'s pairwise evaluation)');
  sys.forces();
  const s = sys.state;
  const nb = sys.nb;
  if (!nb.half) throw new StyleError('compute stress profile: no half neighbour list is available');
  const specialLJ = (sys.ff as unknown as { specialLJ: Float64Array }).specialLJ;
  const specialCoul = (sys.ff as unknown as { specialCoul: Float64Array }).specialCoul;
  const nt = pair.ntypes + 1;
  const xa = nb.xall;
  const entries = order.pairs(s, nb, sys.geom, nb.half);
  const out: PairRow[] = [];
  for (let p = 0; p < entries.length; p += 2) {
    const i = entries[p];
    const jj = entries[p + 1];
    const j = jj & NEIGHMASK;
    if (!((s.mask[i] & groupBit) && (s.mask[nb.owner[j]] & groupBit))) continue;
    const dx = xa[3 * i] - xa[3 * j], dy = xa[3 * i + 1] - xa[3 * j + 1], dz = xa[3 * i + 2] - xa[3 * j + 2];
    const rsq = dx * dx + dy * dy + dz * dz;
    const ti = nb.typeall[i], tj = nb.typeall[j];
    if (rsq >= pair.cutsq[ti * nt + tj]) continue;
    const sb = jj >>> SBBITS;
    const fLJ = sb === 0 ? 1 : specialLJ[sb];
    const fCoul = sb === 0 ? 1 : specialCoul[sb];
    const res = pair.single(nb.owner[i], nb.owner[j], ti, tj, rsq, fCoul, fLJ, nb.qall[i], nb.qall[j]);
    out.push({ xi: xa[3 * i], yi: xa[3 * i + 1], zi: xa[3 * i + 2], xj: xa[3 * j], yj: xa[3 * j + 1], zj: xa[3 * j + 2], fpair: res.fforce });
  }
  return out;
}

/** Native phi table size (compute_stress_cylinder.cpp: nphi = 360). */
const NPHI = 360;

/**
 * compute stress/cylinder ID group-ID stress/cylinder zlo zhi Rmax bin_width [ke yes|no] (native arguments:
 * arg[3..6], optional ke). Global array: one row per radial bin; columns r, number density, then (with ke) the
 * kinetic pkr, pkphi, pkz, then the virial pvr, pvphi, pvz, all in pressure units (times nktv2p).
 */
export class ComputeStressCylinder extends Compute {
  readonly style = 'stress/cylinder';
  private readonly zlo: number;
  private readonly zhi: number;
  private readonly Rmax: number;
  private readonly binWidth: number;
  private readonly kinetic: boolean;
  private nbins = 0;
  private nzbins = 0;
  private R: Float64Array = new Float64Array(0);
  private Rinv: Float64Array = new Float64Array(0);
  private R2: Float64Array = new Float64Array(0);
  private R2kin: Float64Array = new Float64Array(0);
  private PrAinv: Float64Array = new Float64Array(0);
  private PzAinv: Float64Array = new Float64Array(0);
  private invVbin: Float64Array = new Float64Array(0);
  private binz: Float64Array = new Float64Array(0);
  private PphiAinv = 0;
  private readonly pairOrder = new NativePairOrder();
  private readonly tangent = new Float64Array(NPHI);
  private readonly ephiX = new Float64Array(NPHI);
  private readonly ephiY = new Float64Array(NPHI);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 4 && args.length !== 6) throw new StyleError(`compute ${id} stress/cylinder: illegal command (usage: zlo zhi Rmax bin_width [ke yes|no])`);
    this.zlo = Number(args[0]);
    this.zhi = Number(args[1]);
    this.Rmax = Number(args[2]);
    this.binWidth = Number(args[3]);
    this.kinetic = true;
    if (args.length === 6) {
      if (args[4] !== 'ke') throw new StyleError(`compute ${id} stress/cylinder: unknown keyword '${args[4]}' (use ke)`);
      const v = args[5];
      if (v !== 'yes' && v !== 'no' && v !== 'on' && v !== 'off' && v !== 'true' && v !== 'false') throw new StyleError(`compute ${id} stress/cylinder: ke expects yes or no (got '${v}')`);
      this.kinetic = v === 'yes' || v === 'on' || v === 'true';
    }
    if (!(this.binWidth > 0) || this.binWidth > this.Rmax) throw new StyleError(`compute ${id} stress/cylinder: illegal bin width ${this.binWidth} for Rmax ${this.Rmax}`);
    if (this.zhi < this.zlo || this.zhi - this.zlo < this.binWidth) throw new StyleError(`compute ${id} stress/cylinder: illegal z range ${this.zlo} ${this.zhi}`);
    if (sys.hasBox) this.setBins();
    this.arrayFlag = true;
    this.sizeArrayCols = this.kinetic ? 8 : 5;
  }

  /** Bin tables of the C++ constructor and init(); the box must bound the z range. */
  private setBins(): void {
    const s = this.sys.state;
    if (this.zhi > s.box.hi[2] || this.zlo < s.box.lo[2]) throw new StyleError(`compute ${this.id} stress/cylinder: the z range ${this.zlo} ${this.zhi} is outside the box`);
    const nb = Math.trunc(this.Rmax / this.binWidth);
    const nz = Math.trunc((this.zhi - this.zlo) / this.binWidth);
    if (nb < 1 || nz < 1) throw new StyleError(`compute ${this.id} stress/cylinder: illegal bin count`);
    this.nbins = nb;
    this.nzbins = nz;
    const w = this.binWidth, zl = this.zlo, zh = this.zhi;
    this.R = new Float64Array(nb);
    this.Rinv = new Float64Array(nb);
    this.R2 = new Float64Array(nb);
    this.R2kin = new Float64Array(nb);
    this.PrAinv = new Float64Array(nb);
    this.PzAinv = new Float64Array(nb);
    this.invVbin = new Float64Array(nb);
    this.binz = new Float64Array(nz);
    for (let iq = 0; iq < nb; iq++) {
      this.R[iq] = (iq + 0.5) * w;
      this.Rinv[iq] = 1.0 / this.R[iq];
      this.R2[iq] = this.R[iq] * this.R[iq];
      this.R2kin[iq] = (iq + 1.0) * w;
      this.R2kin[iq] *= this.R2kin[iq];
      this.PrAinv[iq] = 1.0 / (2.0 * Math.PI * (zh - zl) * this.R[iq]);
    }
    this.PphiAinv = 1.0 / ((zh - zl) * w * 2.0 * NPHI);
    this.invVbin[0] = 1.0 / ((zh - zl) * Math.PI * this.R2kin[0]);
    this.PzAinv[0] = 1.0 / (Math.PI * this.R2kin[0] * nz);
    for (let jq = 1; jq < nb; jq++) {
      this.invVbin[jq] = 1.0 / ((zh - zl) * Math.PI * (this.R2kin[jq] - this.R2kin[jq - 1]));
      this.PzAinv[jq] = 1.0 / (Math.PI * (this.R2kin[jq] - this.R2kin[jq - 1]) * nz);
    }
    for (let iphi = 0; iphi < NPHI; iphi++) {
      const phi = (iphi * Math.PI) / 180.0;
      this.tangent[iphi] = Math.tan(phi);
      this.ephiX[iphi] = -libmSin(phi);
      this.ephiY[iphi] = libmCos(phi);
    }
    for (let zz = 0; zz < nz; zz++) this.binz[zz] = (zz + 0.5) * w + zl;
    this.sizeArrayRows = nb;
    this.array = new Float64Array(nb * this.sizeArrayCols);
  }

  init(): void {
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError(`No pair style is defined for compute ${this.id} stress/cylinder`);
    if (!pair.single) throw new StyleError(`Pair style does not support compute ${this.id} stress/cylinder`);
    this.setBins();
  }

  protected computeArray(): void {
    const sys = this.sys;
    const s = sys.state;
    if (!this.nbins) this.setBins();
    const nb = this.nbins, nz = this.nzbins;
    const zlo = this.zlo, zhi = this.zhi;
    const R2kin = this.R2kin, R2 = this.R2, invV = this.invVbin;
    const density = new Float64Array(nb);
    const Pkr = new Float64Array(nb), Pkphi = new Float64Array(nb), Pkz = new Float64Array(nb);
    const Pvr = new Float64Array(nb), Pvphi = new Float64Array(nb), Pvz = new Float64Array(nb);
    const x = s.x, v = s.v;
    // number density and kinetic contribution, by radius (every owned atom, whatever the group)
    for (let i = 0; i < s.n; i++) {
      const zi = x[3 * i + 2];
      if (!(zi < zhi && zi > zlo)) continue;
      const tR2 = x[3 * i] * x[3 * i] + x[3 * i + 1] * x[3 * i + 1];
      if (tR2 > R2kin[nb - 1]) continue;
      let j = 0;
      while (j < nb && !(tR2 < R2kin[j])) j++;
      if (j >= nb) continue;
      density[j] += invV[j];
      if (this.kinetic && tR2 !== 0.0 && x[3 * i] !== 0.0) {
        const m = massOf(s, i);
        const xx = x[3 * i], yy = x[3 * i + 1];
        const vr = xx * v[3 * i] + yy * v[3 * i + 1];
        const vp = (v[3 * i + 1] / xx - yy * v[3 * i] / (xx * xx)) / ((yy / xx) * (yy / xx) + 1.0);
        Pkr[j] += (m * vr * vr) / tR2;
        Pkphi[j] += m * tR2 * vp * vp;
        Pkz[j] += m * v[3 * i + 2] * v[3 * i + 2];
      }
    }
    // configurational part: each pair of the group, split over the shells it crosses
    const pairs = groupPairs(sys, this.groupBit, this.pairOrder);
    const tangent = this.tangent, ephiX = this.ephiX, ephiY = this.ephiY, binz = this.binz;
    for (const p of pairs) {
      const ff = p.fpair;
      // the atom nearer the axis is "i" (C++: ri is the smaller of r1 and r2; ties keep i)
      const r1 = p.xi * p.xi + p.yi * p.yi;
      const r2 = p.xj * p.xj + p.yj * p.yj;
      let risq: number, rjsq: number, xi: number, yi: number, zi: number, dx: number, dy: number, dz: number;
      if (r2 < r1) {
        risq = r2; rjsq = r1;
        xi = p.xj; yi = p.yj; zi = p.zj;
        dx = p.xi - p.xj; dy = p.yi - p.yj; dz = p.zi - p.zj;
      } else {
        risq = r1; rjsq = r2;
        xi = p.xi; yi = p.yi; zi = p.zi;
        dx = p.xj - p.xi; dy = p.yj - p.yi; dz = p.zj - p.zi;
      }
      const A = dx * dx + dy * dy;
      const B = 2.0 * (xi * dx + yi * dy);
      // normal pressure contribution P_rhorho: the segment crosses each cylinder R
      for (let ib = 0; ib < nb; ib++) {
        if (rjsq < R2[ib]) continue;
        const C = risq - R2[ib];
        const D = B * B - 4.0 * A * C;
        if (D < 0.0 || A === 0.0) continue;
        const sqrtD = Math.sqrt(D);
        const alpha1 = 0.5 * (-B + sqrtD) / A;
        const alpha2 = 0.5 * (-B - sqrtD) / A;
        if (alpha1 > 0.0 && alpha1 < 1.0) {
          const zR = zi + alpha1 * dz;
          if (zR < zhi && zR > zlo) {
            const xR = xi + alpha1 * dx, yR = yi + alpha1 * dy;
            Pvr[ib] += ff * Math.abs(xR * dx + yR * dy);
          }
        }
        if (alpha2 > 0.0 && alpha2 < 1.0) {
          const zR = zi + alpha2 * dz;
          if (zR < zhi && zR > zlo) {
            const xR = xi + alpha2 * dx, yR = yi + alpha2 * dy;
            Pvr[ib] += ff * Math.abs(xR * dx + yR * dy);
          }
        }
      }
      // azimuthal pressure contribution (P_phiphi): the segment crosses each half-plane phi
      for (let iphi = 0; iphi < NPHI; iphi++) {
        if (dx * tangent[iphi] - dy === 0.0) continue;
        const alpha = (yi - xi * tangent[iphi]) / (dx * tangent[iphi] - dy);
        if (alpha >= 1.0 || alpha <= 0.0) continue;
        const zL = zi + alpha * dz;
        if (zL > zhi || zL < zlo) continue;
        const xL = xi + alpha * dx, yL = yi + alpha * dy;
        const L2 = xL * xL + yL * yL;
        if (L2 > R2kin[nb - 1]) continue;
        const ftphi = Math.abs(dx * ephiX[iphi] + dy * ephiY[iphi]) * ff;
        for (let ib = 0; ib < nb; ib++) {
          if (L2 < R2kin[ib]) {
            Pvphi[ib] += ftphi;
            break;
          }
        }
      }
      // z pressure contribution (P_zz): the segment crosses each z plane of binz
      for (let zb = 0; zb < nz; zb++) {
        if (p.zi > binz[zb] && p.zj > binz[zb]) continue;
        if (p.zi < binz[zb] && p.zj < binz[zb]) continue;
        const alpha = (binz[zb] - zi) / dz;
        const xL = xi + alpha * dx, yL = yi + alpha * dy;
        const L2 = xL * xL + yL * yL;
        if (L2 > R2kin[nb - 1]) continue;
        const ftz = Math.abs(dz) * ff;
        for (let ib = 0; ib < nb; ib++) {
          if (L2 < R2kin[ib]) {
            Pvz[ib] += ftz;
            break;
          }
        }
      }
    }
    // pressures (force over area), then nktv2p
    const nktv2p = s.units.nktv2p;
    const ncol = this.sizeArrayCols;
    const arr = this.array;
    for (let ib = 0; ib < nb; ib++) {
      if (this.kinetic) {
        Pkr[ib] *= invV[ib];
        Pkphi[ib] *= invV[ib];
        Pkz[ib] *= invV[ib];
      }
      Pvr[ib] *= this.PrAinv[ib] * this.Rinv[ib];
      Pvphi[ib] *= this.PphiAinv;
      Pvz[ib] *= this.PzAinv[ib];
      arr[ib * ncol] = this.R[ib];
      arr[ib * ncol + 1] = density[ib];
      if (this.kinetic) {
        arr[ib * ncol + 2] = Pkr[ib] * nktv2p;
        arr[ib * ncol + 3] = Pkphi[ib] * nktv2p;
        arr[ib * ncol + 4] = Pkz[ib] * nktv2p;
        arr[ib * ncol + 5] = Pvr[ib] * nktv2p;
        arr[ib * ncol + 6] = Pvphi[ib] * nktv2p;
        arr[ib * ncol + 7] = Pvz[ib] * nktv2p;
      } else {
        arr[ib * ncol + 2] = Pvr[ib] * nktv2p;
        arr[ib * ncol + 3] = Pvphi[ib] * nktv2p;
        arr[ib * ncol + 4] = Pvz[ib] * nktv2p;
      }
    }
  }
}

/**
 * compute stress/spherical ID group-ID stress/spherical x0 y0 z0 bin_width Rmax. Native bin width: nbins =
 * int(Rmax / bin_width) + 1 and the width becomes Rmax / nbins. Global array of 8 columns: r, number density, the
 * kinetic pkrr, pktt, pkpp, then the configurational pcrr, pctt, pcpp. Native gives no mass and no nktv2p factor.
 */
export class ComputeStressSpherical extends Compute {
  readonly style = 'stress/spherical';
  private readonly x0: number;
  private readonly y0: number;
  private readonly z0: number;
  private readonly Rmax: number;
  private nbins = 0;
  private width = 0;
  private invV: Float64Array = new Float64Array(0);
  private readonly pairOrder = new NativePairOrder();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 5) throw new StyleError(`compute ${id} stress/spherical: illegal number of arguments (usage: x0 y0 z0 bin_width Rmax)`);
    this.x0 = Number(args[0]);
    this.y0 = Number(args[1]);
    this.z0 = Number(args[2]);
    const bw = Number(args[3]);
    this.Rmax = Number(args[4]);
    this.nbins = Math.trunc(this.Rmax / bw) + 1;
    this.width = this.Rmax / this.nbins;
    if (!(this.width > 0)) throw new StyleError(`compute ${id} stress/spherical: bin width must be > 0`);
    this.arrayFlag = true;
    this.sizeArrayCols = 8;
    this.sizeArrayRows = this.nbins;
    this.array = new Float64Array(this.nbins * 8);
  }

  init(): void {
    const pair = this.sys.ff.pair;
    if (!pair) throw new StyleError(`No pair style is defined for compute ${this.id} stress/spherical`);
    if (!pair.single) throw new StyleError(`Pair style does not support compute ${this.id} stress/spherical`);
    const w = this.width;
    this.invV = new Float64Array(this.nbins);
    for (let bin = 0; bin < this.nbins; bin++) {
      const a = (bin + 1) * w, b = bin * w;
      this.invV[bin] = 0.75 / (Math.PI * (a * a * a - b * b * b));
    }
  }

  protected computeArray(): void {
    const sys = this.sys;
    const s = sys.state;
    const nb = this.nbins, w = this.width, Rmax = this.Rmax;
    const SMALL = 1.0e-10;
    const x0 = this.x0, y0 = this.y0, z0 = this.z0;
    const box = s.box;
    const L = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]];
    /** The C++ wraps a displacement by one box length when it exceeds half a box. */
    const wrap = (d: number[]): void => {
      for (let k = 0; k < 3; k++) {
        if (d[k] > 0.5 * L[k]) d[k] -= L[k];
        else if (d[k] < -0.5 * L[k]) d[k] += L[k];
      }
    };
    const tdens = new Float64Array(nb);
    const tpkrr = new Float64Array(nb), tpktt = new Float64Array(nb), tpkpp = new Float64Array(nb);
    const tpcrr = new Float64Array(nb), tpctt = new Float64Array(nb), tpcpp = new Float64Array(nb);
    const x = s.x, v = s.v;
    // kinetic part: every owned atom, positions relative to the centre
    for (let i = 0; i < s.n; i++) {
      const ri = [x[3 * i] - x0, x[3 * i + 1] - y0, x[3 * i + 2] - z0];
      wrap(ri);
      const r = Math.sqrt(ri[0] * ri[0] + ri[1] * ri[1] + ri[2] * ri[2]);
      if (r >= Rmax) continue;
      const bin = Math.trunc(r / w);
      if (r !== 0.0 && ri[0] !== 0.0) {
        const theta = Math.acos(ri[2] / r);
        tdens[bin] += 1.0;
        const vr = (ri[0] * v[3 * i] + ri[1] * v[3 * i + 1] + ri[2] * v[3 * i + 2]) / r;
        const q = ri[1] / ri[0];
        const vt = (r * libmSin(theta)) / (q * q + 1.0) * ((ri[0] * v[3 * i + 1] - ri[1] * v[3 * i]) / (ri[0] * ri[0]));
        const vp = (ri[2] * vr - r * v[3 * i + 2]) / (r * Math.sqrt(1.0 - (ri[2] / r) * (ri[2] / r)));
        tpkrr[bin] += vr * vr;
        tpktt[bin] += vt * vt;
        tpkpp[bin] += vp * vp;
      }
    }
    // configurational part: each group pair, its segment integrated along the sphere shells it crosses
    const pairs = groupPairs(sys, this.groupBit, this.pairOrder);
    for (const p of pairs) {
      // ri = x_j - x_i (raw image difference, as in the C++), qi = x_i - centre (wrapped)
      const ri = [p.xj - p.xi, p.yj - p.yi, p.zj - p.zi];
      const rsq = ri[0] * ri[0] + ri[1] * ri[1] + ri[2] * ri[2];
      const f0 = p.fpair;
      const qi = [p.xi - x0, p.yi - y0, p.zi - z0];
      wrap(qi);
      const rij = Math.sqrt(rsq);
      const f = -rij * f0;
      const ririj = qi[0] * ri[0] + qi[1] * ri[1] + qi[2] * ri[2];
      const sqr = qi[0] * qi[0] + qi[1] * qi[1] + qi[2] * qi[2];
      let la = 0.0, lb = 0.0, lSum = 0.0;
      const sql0 = sqr - (ririj * ririj) / rsq;
      const lambda0 = -ririj / rsq;
      const rsqxy = ri[0] * ri[0] + ri[1] * ri[1];
      const ririjxy = qi[0] * ri[0] + qi[1] * ri[1];
      const sqrixy = qi[0] * qi[0] + qi[1] * qi[1];
      const sqlxy0 = rsqxy !== 0.0 ? sqrixy - (ririjxy * ririjxy) / rsqxy : 0.0;
      const A = (qi[0] * ri[1] - qi[1] * ri[0]) * (qi[0] * ri[1] - qi[1] * ri[0]);
      let C = 0.0, B = 0.0;
      if (sqlxy0 > SMALL) C = Math.sqrt(rsqxy * sqrixy - ririjxy * ririjxy);
      if (sql0 > SMALL) B = Math.sqrt(rsq * sqr - ririj * ririj);
      const endBin = Math.trunc(Math.sqrt(rsq + 2.0 * ririj + sqr) / w);
      while (lb < 1.0) {
        let l1 = lb + SMALL;
        const bin = Math.floor(Math.sqrt(rsq * l1 * l1 + 2.0 * ririj * l1 + sqr) / w);
        const R1 = bin * w;
        const R2 = (bin + 1) * w;
        if (R1 * R1 < sql0 || R2 * R2 < sql0 || rsq <= 0.0) {
          lb = 1.0;
        } else {
          l1 = lambda0 + Math.sqrt((R1 * R1 - sql0) / rsq);
          const l2 = lambda0 - Math.sqrt((R1 * R1 - sql0) / rsq);
          const l3 = lambda0 + Math.sqrt((R2 * R2 - sql0) / rsq);
          const l4 = lambda0 - Math.sqrt((R2 * R2 - sql0) / rsq);
          if (l4 >= 0.0 && l4 <= 1.0 && l4 > la) lb = l4;
          else if (l3 >= 0.0 && l3 <= 1.0 && l3 > la) lb = l3;
          else if (l2 >= 0.0 && l2 <= 1.0 && l2 > la) lb = l2;
          else if (l1 >= 0.0 && l1 <= 1.0 && l1 > la) lb = l1;
          else lb = 1.0;
        }
        if (bin === endBin) lb = 1.0;
        if (la > lb) throw new StyleError(`compute ${this.id} stress/spherical: Error: la > lb`);
        if (bin >= 0 && bin < nb) {
          if (sql0 > SMALL) {
            const Fa = -B * Math.atan2(rsq * la + ririj, B);
            const Fb = -B * Math.atan2(rsq * lb + ririj, B);
            tpcrr[bin] -= (f / rij) * (rsq * (lb - la) + Fb - Fa);
            tpcpp[bin] -= ((f / rij) * (Fa - Fb)) / 2.0;
          } else {
            tpcrr[bin] -= f * rij * (lb - la);
          }
          if (sqlxy0 > SMALL) {
            tpctt[bin] -= ((f / rij) * A / C) * (Math.atan2(rsqxy * lb + ririjxy, C) - Math.atan2(rsqxy * la + ririjxy, C));
          }
        }
        lSum += lb - la;
        la = lb;
      }
      if (Math.abs(lSum - 1.0) > SMALL) throw new StyleError(`compute ${this.id} stress/spherical: the sum of the fractional line segments is not 1.0`);
    }
    const inv = this.invV;
    const ncol = 8;
    const arr = this.array;
    for (let bin = 0; bin < nb; bin++) {
      arr[bin * ncol] = (bin + 0.5) * w;
      arr[bin * ncol + 1] = tdens[bin] * inv[bin];
      arr[bin * ncol + 2] = tpkrr[bin] * inv[bin];
      arr[bin * ncol + 3] = tpktt[bin] * inv[bin];
      arr[bin * ncol + 4] = tpkpp[bin] * inv[bin];
      arr[bin * ncol + 5] = tpcrr[bin] * inv[bin];
      arr[bin * ncol + 6] = tpctt[bin] * inv[bin];
      arr[bin * ncol + 7] = tpcpp[bin] * inv[bin];
    }
  }
}
