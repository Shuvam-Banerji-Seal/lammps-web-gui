/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_nvk.cpp and src/EXTRA-FIX/fix_nvk.h at commit b332f28 (class FixNVK, fix nvk; contributing
 * author Efrem Braun, UC Berkeley). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Microcanonical kinetic-energy integrator (Minary, Tuckerman and Martyna, PRL 2003, equations 4.12-4.17). The
 * target kinetic energy K_target is taken from the velocities at the start of each run (FixNVK::init).
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, nativeOrder } from '../../engine/atoms';

export class FixNVK extends Fix {
  readonly style = 'nvk';
  private dtv = 0;
  private dtf = 0;
  private kTarget = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (group !== 'all') throw new StyleError('Fix nvk only supports group all');
    this.timeIntegrate = true;
  }

  init(): void {
    this.resetDt();
    // compute initial kinetic energy (fix_nvk.cpp init: a copy of compute ke over the group)
    const s = this.sys.state;
    const { v, mask } = s;
    const pfactor = 0.5 * s.units.mvv2e;
    let ke = 0.0;
    for (const i of nativeOrder(s)) {
      if (!(mask[i] & this.groupBit)) continue;
      ke += massOf(s, i) * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
    }
    this.kTarget = ke * pfactor;
  }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt;
  }

  /** Minary 2003 equations 4.12 and 4.13: the a and b sums over the group, then s and sdot for half a step. */
  private sAndSdot(): { s: number; sdot: number } {
    const st = this.sys.state;
    const { v, f, mask } = st;
    let a = 0.0;
    let b = 0.0;
    for (const i of nativeOrder(st)) {
      if (!(mask[i] & this.groupBit)) continue;
      a += f[3 * i] * v[3 * i] + f[3 * i + 1] * v[3 * i + 1] + f[3 * i + 2] * v[3 * i + 2];
      b += (f[3 * i] * f[3 * i] + f[3 * i + 1] * f[3 * i + 1] + f[3 * i + 2] * f[3 * i + 2]) / massOf(st, i);
    }
    a /= 2.0 * this.kTarget;
    b /= 2.0 * this.kTarget * st.units.mvv2e;
    const sqtb = Math.sqrt(b);
    const s = (a / b) * (Math.cosh(this.dtf * sqtb) - 1.0) + Math.sinh(this.dtf * sqtb) / sqtb;
    const sdot = (a / b) * sqtb * Math.sinh(this.dtf * sqtb) + Math.cosh(this.dtf * sqtb);
    return { s, sdot };
  }

  /** Equations 4.15-4.17: v = (v + F s / m ftm2v) / sdot; x += dt v (x only in the initial half). */
  private kick(updateX: boolean): void {
    const st = this.sys.state;
    const { x, v, f, mask } = st;
    const { s, sdot } = this.sAndSdot();
    const ftm2v = st.units.ftm2v;
    for (const i of nativeOrder(st)) {
      if (!(mask[i] & this.groupBit)) continue;
      const sm = s / massOf(st, i);
      v[3 * i] = (v[3 * i] + f[3 * i] * sm * ftm2v) / sdot;
      v[3 * i + 1] = (v[3 * i + 1] + f[3 * i + 1] * sm * ftm2v) / sdot;
      v[3 * i + 2] = (v[3 * i + 2] + f[3 * i + 2] * sm * ftm2v) / sdot;
      if (updateX) {
        x[3 * i] += this.dtv * v[3 * i];
        x[3 * i + 1] += this.dtv * v[3 * i + 1];
        x[3 * i + 2] += this.dtv * v[3 * i + 2];
      }
    }
  }

  initialIntegrate(): void { this.kick(true); }

  finalIntegrate(): void { this.kick(false); }
}
