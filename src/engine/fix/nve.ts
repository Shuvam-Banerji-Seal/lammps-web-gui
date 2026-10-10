import { Fix } from './fix';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { massOf } from '../atoms';

/*
 * fix ID group nve — docs.lammps.org/fix_nve.html: "Perform plain time
 * integration to update position and velocity for atoms in the group each
 * timestep. ... This fix invokes the velocity form of the Stoermer-Verlet
 * time integration algorithm (velocity-Verlet)."
 *   initial_integrate: v += dt/2 F/m (ftm2v); x += dt v
 *   final_integrate:   v += dt/2 F/m
 */

export class FixNVE extends Fix {
  readonly style: string = 'nve';
  protected dtv = 0;
  protected dtf = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length && new.target === FixNVE) throw new StyleError('fix nve takes no arguments');
    this.timeIntegrate = true;
  }

  init(): void { this.resetDt(); }

  resetDt(): void {
    const s = this.sys.state;
    this.dtv = s.dt;
    this.dtf = 0.5 * s.dt * s.units.ftm2v;
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    const { x, v, f, mask, type } = s;
    const bit = this.groupBit;
    const dtf = this.dtf, dtv = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
      x[3 * i] += dtv * v[3 * i]; x[3 * i + 1] += dtv * v[3 * i + 1]; x[3 * i + 2] += dtv * v[3 * i + 2];
    }
  }

  finalIntegrate(): void {
    const s = this.sys.state;
    const { v, f, mask, type } = s;
    const bit = this.groupBit;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = dtf / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
  }

  /**
   * run_style respa (run/respa.ts): true only for plain fix nve; the sphere and asphere variants
   * also integrate rotation, which the respa level kicks do not cover.
   */
  get respaOK(): boolean { return this.constructor === FixNVE; }

  /** Level kick v += h F/m over the group, with that level's force array f (h = half the level's timestep). */
  respaKick(f: Float64Array, h: number): void {
    const s = this.sys.state;
    const { v, mask } = s;
    const bit = this.groupBit;
    const ftm2v = h * s.units.ftm2v;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = ftm2v / massOf(s, i);
      v[3 * i] += c * f[3 * i]; v[3 * i + 1] += c * f[3 * i + 1]; v[3 * i + 2] += c * f[3 * i + 2];
    }
  }

  /** Innermost drift x += dt v over the group (run_style respa). */
  respaDrift(dt: number): void {
    const s = this.sys.state;
    const { x, v, mask } = s;
    const bit = this.groupBit;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      x[3 * i] += dt * v[3 * i]; x[3 * i + 1] += dt * v[3 * i + 1]; x[3 * i + 2] += dt * v[3 * i + 2];
    }
  }
}
