/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/LATBOLTZ/fix_lb_momentum.cpp and src/LATBOLTZ/fix_lb_momentum.h at commit b332f28 (fix lb/momentum:
 * every N steps the net linear momentum of the particles plus the fluid is set to zero, by subtracting the centre-of-
 * mass velocity from the particles and from the fluid populations and velocities). Contributing authors: Frances
 * Mackay, Santtu Ollila, Colin Denniston (UWO); Naveen Michaud-Agrawal (Johns Hopkins U). Copyright (2003) Sandia
 * Corporation, GNU General Public License. This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_lb_momentum.html: "Zero the linear momentum of the particles and fluid ... linear flags select
 * the components."
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';
import type { FixLbFluid } from './fix_lb_fluid';
import { MG15, NG15, W15 } from './lb_tables';

const intArg = (w: string | undefined): number => {
  const n = Number(w);
  if (w === undefined || !Number.isInteger(n)) throw new StyleError(`Illegal fix lb/momentum command: ${w ?? ''}`);
  return n;
};

export class FixLbMomentum extends Fix {
  readonly style = 'lb/momentum';
  private xflag = 1;
  private yflag = 1;
  private zflag = 1;
  private count = 0;
  private masstotal = 0;
  private fluid: FixLbFluid | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError('Illegal fix lb/momentum command');
    const nevery = intArg(args[0]);
    if (nevery <= 0) throw new StyleError('Illegal fix lb/momentum command');
    this.nevery = nevery;
    let iarg = 1;
    while (iarg < args.length) {
      if (args[iarg] === 'linear') {
        if (iarg + 4 > args.length) throw new StyleError('Illegal fix lb/momentum command');
        this.xflag = intArg(args[iarg + 1]);
        this.yflag = intArg(args[iarg + 2]);
        this.zflag = intArg(args[iarg + 3]);
        if (this.xflag < 0 || this.xflag > 1 || this.yflag < 0 || this.yflag > 1 || this.zflag < 0 || this.zflag > 1) {
          throw new StyleError('Illegal fix lb/momentum command');
        }
        iarg += 4;
      } else {
        throw new StyleError('Illegal fix lb/momentum command');
      }
    }
  }

  init(): void {
    const s = this.sys.state;
    this.count = 0;
    this.masstotal = 0;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) { this.count++; this.masstotal += massOf(s, i); }
    const fl = this.sys.fixes.filter((f): f is FixLbFluid => f.style === 'lb/fluid');
    if (fl.length === 0) throw new StyleError('fix lb/momentum requires a fix lb/fluid');
    this.fluid = fl[0];
  }

  /** fix_lb_momentum.cpp end_of_step(): remove the centre-of-mass velocity of particles plus fluid. */
  endOfStep(): void {
    const s = this.sys.state;
    const fluid = this.fluid!;
    const vcmp = [0.0, 0.0, 0.0];
    if (this.count) {
      const order = nativeOrder(s);
      for (let k = 0; k < s.n; k++) {
        const i = order[k];
        if (!(s.mask[i] & this.groupBit)) continue;
        const m = massOf(s, i);
        vcmp[0] += s.v[3 * i] * m;
        vcmp[1] += s.v[3 * i + 1] * m;
        vcmp[2] += s.v[3 * i + 2] * m;
      }
      vcmp[0] /= this.masstotal;
      vcmp[1] /= this.masstotal;
      vcmp[2] /= this.masstotal;
    }
    const fm = fluid.massMomentum();
    const denom = fm.mass + this.masstotal;
    const vcmall = [
      (this.masstotal * vcmp[0] + fm.momentum[0]) / denom,
      (this.masstotal * vcmp[1] + fm.momentum[1]) / denom,
      (this.masstotal * vcmp[2] + fm.momentum[2]) / denom,
    ];
    if (this.count) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        if (this.xflag) s.v[3 * i] -= vcmall[0];
        if (this.yflag) s.v[3 * i + 1] -= vcmall[1];
        if (this.zflag) s.v[3 * i + 2] -= vcmall[2];
      }
    }
    // fluid: convert to lattice units
    const dx = fluid.dxLb, dt = fluid.dtLb;
    vcmall[0] *= dt / dx;
    vcmall[1] *= dt / dx;
    vcmall[2] *= dt / dx;
    const ucmx = this.xflag ? vcmall[0] : 0.0;
    const ucmy = this.yflag ? vcmall[1] : 0.0;
    const ucmz = this.zflag ? vcmall[2] : 0.0;
    const sx = fluid.subNbx, sy = fluid.subNby, sz = fluid.subNbz;
    const QY = fluid.QY, QZ = fluid.QZ;
    const etacov = new Float64Array(14);
    const dens = fluid.densityField(), u = fluid.velocityField(), f = fluid.populationField();
    for (let i = 0; i < sx; i++)
      for (let j = 0; j < sy; j++)
        for (let k = 0; k < sz; k++) {
          const q = (i * QY + j) * QZ + k;
          const fnode = ((i * sy + j) * sz + k) * 15;
          const rho = dens[q];
          const u0 = u[3 * q], u1 = u[3 * q + 1], u2 = u[3 * q + 2];
          etacov[0] = etacov[10] = etacov[11] = etacov[12] = etacov[13] = 0.0;
          etacov[1] = rho * ucmx;
          etacov[2] = rho * ucmy;
          etacov[3] = rho * ucmz;
          etacov[4] = rho * (2.0 * u0 * ucmx - ucmx * ucmx);
          etacov[5] = rho * (2.0 * u1 * ucmy - ucmy * ucmy);
          etacov[6] = rho * (2.0 * u2 * ucmz - ucmz * ucmz);
          etacov[7] = rho * (u0 * ucmy + u1 * ucmx - ucmx * ucmy);
          etacov[8] = rho * (u1 * ucmz + u2 * ucmy - ucmy * ucmz);
          etacov[9] = rho * (u0 * ucmz + u2 * ucmx - ucmx * ucmz);
          etacov[13] = rho * (u0 * u1 * ucmz + u0 * ucmy * u2 - u0 * ucmy * ucmz + ucmx * u1 * u2 - ucmx * u1 * ucmz - ucmx * ucmy * u2 + ucmx * ucmy * ucmz);
          for (let l = 0; l < 15; l++)
            for (let ii = 1; ii < 14; ii++) f[fnode + l] -= W15[l] * MG15[ii * 15 + l] * etacov[ii] * NG15[ii];
          if (this.xflag) u[3 * q] -= vcmall[0];
          if (this.yflag) u[3 * q + 1] -= vcmall[1];
          if (this.zflag) u[3 * q + 2] -= vcmall[2];
        }
  }
}
