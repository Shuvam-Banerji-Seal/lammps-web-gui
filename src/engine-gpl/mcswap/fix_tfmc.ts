/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/MC/fix_tfmc.cpp and
 * src/MC/fix_tfmc.h (Kristof Bal, University of Antwerp, Belgium). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2. This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix tfmc (docs.lammps.org/fix_tfmc.html): the force-biased (time-stamped force-bias) Monte Carlo displacement of the
 * group atoms, done in initial_integrate. Each atom moves by xi d_i along each Cartesian direction, with d_i =
 * d_max (m_min / m_i)^(1/4) and xi drawn from the biased distribution of the force f d_i / (2 k T) (the rejection loop
 * of native, with the RanMars stream seed). The keywords com (remove the centre-of-mass displacement, per direction)
 * and rot (remove the rigid rotation the displacements produce) follow native.
 *
 * Native structure kept here, for one processor: the atoms are visited in native local order (nativeOrder), the
 * random draws follow native's sequence (xi, then the acceptance uniform, until accepted), and the group sums use
 * native's order. The minimum mass over the group is taken in init. The rot keyword uses Group::xcm, Group::inertia
 * and Group::omega (src/group.cpp) with the unwrapped positions.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import { RanMars } from '../random_mars';
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { groupOmega } from '../group/group_omega';

/** utils::numeric (utils.cpp). */
const numeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isFinite(v)) {
    throw new StyleError(`Expected floating point parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};
/** utils::inumeric (utils.cpp). */
const inumeric = (w: string | undefined): number => {
  const v = Number(w);
  if (w === undefined || w.trim() === '' || !Number.isInteger(v)) {
    throw new StyleError(`Expected integer parameter instead of '${w ?? ''}' in input script or data file`);
  }
  return v;
};

export class FixTFMC extends Fix {
  readonly style = 'tfmc';
  private readonly dMax: number;
  private readonly tSet: number;
  private readonly seed: number;
  private comflag = false;
  private rotflag = false;
  private xflag = 0;
  private yflag = 0;
  private zflag = 0;
  private readonly rng: RanMars;
  private massMin = Number.MAX_VALUE;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 3) throw new StyleError('Illegal fix tfmc command');
    this.timeIntegrate = true;
    this.dMax = numeric(args[0]);
    this.tSet = numeric(args[1]);
    this.seed = inumeric(args[2]);
    if (this.dMax <= 0) throw new StyleError('Fix tfmc displacement length must be > 0');
    if (this.tSet <= 0) throw new StyleError('Fix tfmc temperature must be > 0');
    if (this.seed <= 0) throw new StyleError('Illegal fix tfmc random seed');
    let a = 3;
    while (a < args.length) {
      if (args[a] === 'com') {
        if (a + 4 > args.length) throw new StyleError('Illegal fix tfmc command');
        this.comflag = true;
        this.xflag = inumeric(args[a + 1]);
        this.yflag = inumeric(args[a + 2]);
        this.zflag = inumeric(args[a + 3]);
        a += 4;
      } else if (args[a] === 'rot') {
        this.rotflag = true;
        a += 1;
      } else {
        throw new StyleError('Illegal fix tfmc command');
      }
    }
    if (this.comflag) {
      if (this.xflag < 0 || this.xflag > 1 || this.yflag < 0 || this.yflag > 1 || this.zflag < 0 || this.zflag > 1) {
        throw new StyleError('Illegal fix tfmc command');
      }
    }
    if (this.xflag + this.yflag + this.zflag === 0) this.comflag = false;
    this.rng = new RanMars(this.seed);
  }

  /** FixTFMC::init: incompatible fixes, and the smallest mass of the group (per-type or per-atom). */
  init(): void {
    const sys = this.sys;
    const s = sys.state;
    for (const pre of ['shake', 'rattle', 'ilves']) {
      if (sys.fixes.some((f) => f.style.startsWith(pre))) throw new StyleError(`Fix tfmc is not compatible with fix ${pre}`);
    }
    let mmin = Number.MAX_VALUE;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      if (mmin > m) mmin = m;
    }
    this.massMin = mmin;
  }

  /** initial_integrate: the force-biased displacements of the group atoms, then the com and rot corrections. */
  initialIntegrate(): void {
    const s = this.sys.state;
    const boltz = s.units.boltz;
    const order = nativeOrder(s);
    const gb = this.groupBit;
    const xcmD = [0.0, 0.0, 0.0];
    const xd = this.rotflag ? new Float64Array(3 * s.n) : null;
    let count = 0;
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & gb)) continue;
      count++;
      const massone = massOf(s, i);
      const dI = this.dMax * libmPow(this.massMin / massone, 0.25);
      for (let j = 0; j < 3; j++) {
        let Pacc = 0.0;
        let Pran = 1.0;
        const gamma = s.f[3 * i + j] * dI / (2.0 * boltz * this.tSet);
        const gammaExp = libmExp(gamma);
        const gammaExpi = 1.0 / gammaExp;
        let xi = 0.0;
        while (Pacc < Pran) {
          xi = 2.0 * this.rng.uniform() - 1.0;
          Pran = this.rng.uniform();
          if (xi < 0) {
            Pacc = libmExp(2.0 * xi * gamma) * gammaExp - gammaExpi;
            Pacc = Pacc / (gammaExp - gammaExpi);
          } else if (xi > 0) {
            Pacc = gammaExp - libmExp(2.0 * xi * gamma) * gammaExpi;
            Pacc = Pacc / (gammaExp - gammaExpi);
          } else {
            Pacc = 1.0;
          }
        }
        s.x[3 * i + j] += xi * dI;
        if (this.comflag) xcmD[j] += xi * dI * massone;
        if (xd) xd[3 * i + j] = xi * dI;
      }
    }
    // masstotal = group->mass(igroup) when com or rot is set
    let masstotal = 0;
    if (this.comflag || this.rotflag) {
      for (let k = 0; k < s.n; k++) {
        const i = order[k];
        if (s.mask[i] & gb) masstotal += massOf(s, i);
      }
    }
    if (this.comflag && count !== 0) {
      const xcmDall = [0.0, 0.0, 0.0];
      for (let d = 0; d < 3; d++) xcmDall[d] = xcmD[d];
      if (masstotal > 0.0) {
        xcmDall[0] /= masstotal;
        xcmDall[1] /= masstotal;
        xcmDall[2] /= masstotal;
      } else {
        xcmDall[0] = xcmDall[1] = xcmDall[2] = 0.0;
      }
      for (let k = 0; k < s.n; k++) {
        const i = order[k];
        if (!(s.mask[i] & gb)) continue;
        if (this.xflag) s.x[3 * i] -= xcmDall[0];
        if (this.yflag) s.x[3 * i + 1] -= xcmDall[1];
        if (this.zflag) s.x[3 * i + 2] -= xcmDall[2];
      }
    }
    if (this.rotflag && xd && count !== 0) this.removeRotation(xd, masstotal);
  }

  /** The rigid rotation of the group's displacements (fix_tfmc.cpp rot): Group::xcm, inertia and omega. */
  private removeRotation(xd: Float64Array, masstotal: number): void {
    const s = this.sys.state;
    const gb = this.groupBit;
    const order = nativeOrder(s);
    const geom = this.sys.geom;
    const u = [0.0, 0.0, 0.0];
    // Group::xcm of the unwrapped positions
    const cm = [0.0, 0.0, 0.0];
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & gb)) continue;
      geom.unwrap(s.x, s.image, i, u);
      const m = massOf(s, i);
      cm[0] += u[0] * m; cm[1] += u[1] * m; cm[2] += u[2] * m;
    }
    if (masstotal > 0.0) { cm[0] /= masstotal; cm[1] /= masstotal; cm[2] /= masstotal; }
    // angular momentum of the displacements about cm
    const p = [0.0, 0.0, 0.0];
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & gb)) continue;
      geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - cm[0], dy = u[1] - cm[1], dz = u[2] - cm[2];
      const m = massOf(s, i);
      p[0] += m * (dy * xd[3 * i + 2] - dz * xd[3 * i + 1]);
      p[1] += m * (dz * xd[3 * i] - dx * xd[3 * i + 2]);
      p[2] += m * (dx * xd[3 * i + 1] - dy * xd[3 * i]);
    }
    // Group::inertia about cm
    const ione = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & gb)) continue;
      geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - cm[0], dy = u[1] - cm[1], dz = u[2] - cm[2];
      const m = massOf(s, i);
      ione[0][0] += m * (dy * dy + dz * dz);
      ione[1][1] += m * (dx * dx + dz * dz);
      ione[2][2] += m * (dx * dx + dy * dy);
      ione[0][1] -= m * dx * dy;
      ione[1][2] -= m * dy * dz;
      ione[0][2] -= m * dx * dz;
    }
    ione[1][0] = ione[0][1];
    ione[2][1] = ione[1][2];
    ione[2][0] = ione[0][2];
    const omega = groupOmega([p[0], p[1], p[2]], ione);
    for (let k = 0; k < s.n; k++) {
      const i = order[k];
      if (!(s.mask[i] & gb)) continue;
      geom.unwrap(s.x, s.image, i, u);
      const dx = u[0] - cm[0], dy = u[1] - cm[1], dz = u[2] - cm[2];
      s.x[3 * i] -= omega[1] * dz - omega[2] * dy;
      s.x[3 * i + 1] -= omega[2] * dx - omega[0] * dz;
      s.x[3 * i + 2] -= omega[0] * dy - omega[1] * dx;
    }
  }
}
