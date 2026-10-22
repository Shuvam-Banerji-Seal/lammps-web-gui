/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/EXTRA-FIX/fix_electron_stopping_fit.cpp and fix_electron_stopping_fit.h (FixElectronStoppingFit: the
 * constructor, init, setup, post_force, compute_scalar) at commit b332f28. Copyright Sandia Corporation (GNU General
 * Public License). Author of the source: James A. Stewart (SNL). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * Electronic stopping with a fitted friction (Stewart et al., J. Appl. Phys. 123, 165902 (2018); Lee et al., Phys.
 * Rev. B 102, 024107 (2020)): each atom of every type feels the friction gamma v, with gamma linear in v between
 * v_min^2 = E_coh / m and v_max^2 = 2 E_coh / m (the cohesive energy E_coh of the type), and gamma = d1 + d2 |v_i|
 * components above v_max. The scalar is the electronic energy loss, integrated with the trapezoid rule over steps.
 * Unlike fix electron/stopping (src/engine/fix/electron_stopping.ts) this fix takes no group, region or table: all
 * atoms are subject to it, and it applies the friction in setup as the source does.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

export class FixElectronStoppingFit extends Fix {
  readonly style = 'electron/stopping/fit';
  /** Per type (1-based index): the squared speeds bounding the fitted region, and the two drag coefficients. */
  private readonly vMinSq: Float64Array;
  private readonly vMaxSq: Float64Array;
  private readonly dragFac1: Float64Array;
  private readonly dragFac2: Float64Array;
  private loss = 0;
  private fDotVPrior = 0;
  private fDotVCurrent = 0;
  private lastStep = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const ntypes = sys.state.ntypes;
    if (args.length < 3 * ntypes) throw new StyleError('Incorrect number of fix electron/stopping/fit arguments');
    this.scalarFlag = true;
    this.extscalar = 0;
    this.vMinSq = new Float64Array(ntypes + 1);
    this.vMaxSq = new Float64Array(ntypes + 1);
    this.dragFac1 = new Float64Array(ntypes + 1);
    this.dragFac2 = new Float64Array(ntypes + 1);
    const mvv2e = sys.state.units.mvv2e;
    for (let t = 1; t <= ntypes; t++) {
      const eCoh = Number(args[3 * (t - 1)]);
      const d1 = Number(args[3 * (t - 1) + 1]);
      const d2 = Number(args[3 * (t - 1) + 2]);
      if (![eCoh, d1, d2].every((x) => Number.isFinite(x))) throw new StyleError(`fix ${id} (electron/stopping/fit): non-numeric argument for type ${t}`);
      // the source: mvv = 2 E_coh / mvv2e; v_min^2 = mvv / m, v_max^2 = 2 mvv / m
      const mvv = (2.0 * eCoh) / mvv2e;
      const mass = sys.state.massByType[t];
      this.vMinSq[t] = (1.0 * mvv) / mass;
      this.vMaxSq[t] = (2.0 * mvv) / mass;
      this.dragFac1[t] = d1;
      this.dragFac2[t] = d2;
    }
  }

  init(): void {
    this.loss = 0;
    this.fDotVPrior = 0;
    this.fDotVCurrent = 0;
    this.lastStep = this.sys.state.step;
  }

  /** The source's setup: post_force for verlet, so the friction is part of the first step's forces. */
  setup(): void { this.postForce(); }

  postForce(): void {
    const s = this.sys.state;
    const { v, f, type } = s;
    this.fDotVCurrent = 0.0;
    for (let i = 0; i < s.n; i++) {
      const vx = v[3 * i], vy = v[3 * i + 1], vz = v[3 * i + 2];
      const vv = vx * vx + vy * vy + vz * vz;
      const t = type[i];
      if (!(vv > this.vMinSq[t])) continue;
      const vMag = Math.sqrt(vv);
      const d1 = this.dragFac1[t], d2 = this.dragFac2[t];
      let gx: number, gy: number, gz: number;
      if (vv < this.vMaxSq[t]) {
        const frac = (vv - this.vMinSq[t]) / (this.vMaxSq[t] - this.vMinSq[t]);
        gx = frac * (d2 * vx + d1);
        gy = frac * (d2 * vy + d1);
        gz = frac * (d2 * vz + d1);
      } else {
        gx = d2 * vx + d1;
        gy = d2 * vy + d1;
        gz = d2 * vz + d1;
      }
      f[3 * i] -= gx * vx;
      f[3 * i + 1] -= gy * vy;
      f[3 * i + 2] -= gz * vz;
      const a = gx * vx, b = gy * vy, c = gz * vz;
      this.fDotVCurrent += vMag * Math.sqrt(a * a + b * b + c * c);
    }
    // trapezoid rule over the steps since the last force evaluation
    const thisStep = s.step;
    this.loss += ((thisStep - this.lastStep) * s.dt * 0.5) * (this.fDotVPrior + this.fDotVCurrent);
    this.lastStep = thisStep;
    this.fDotVPrior = this.fDotVCurrent;
  }

  computeScalar(): number { return this.loss; }
}
