import type { System } from '../system';
import { Fix } from './fix';
import { StyleError } from '../force/types';
import { massOf } from '../atoms';

/*
 * fix dt/reset — docs.lammps.org/fix_dt_reset.html. Written from the doc page
 * and from black-box runs of the native binary, never from LAMMPS source.
 *
 * Doc: "Reset the timestep size every N steps during a run, so that no atom
 * moves further than the specified *Xmax* distance, based on current atom
 * velocities and forces." The doc gives no closed form for the per-atom
 * estimate ("a cheaper estimate is generated ... conservative"), so the rule
 * below was fitted to native measurements (see the comments on atomDt).
 *
 * Doc: "Given this putative timestep for each atom, the minimum timestep value
 * across all atoms is computed.  Then the *Tmin* and *Tmax* bounds are
 * applied, if specified."
 *
 * Doc (the scalar): "The scalar stores the last timestep on which the timestep
 * was reset to a new value." Measured: the scalar changes only when the new
 * value differs from the previous timestep size.
 */

/** Timestep reported when no atom in the group limits it (measured: a group with no motion gives 1e20). */
const UNLIMITED_DT = 1.0e20;

/**
 * Sets the timestep size and keeps the elapsed time ("thermo_style time") exact: the time already
 * accumulated at the previous size is banked before the change. Every fix that caches dt is told
 * through resetDt. Same bookkeeping as the timestep command.
 */
export const setTimestep = (sys: System, dt: number): void => {
  const s = sys.state;
  s.time += (s.step - s.timeStep) * s.dt;
  s.timeStep = s.step;
  s.dt = dt;
  for (const f of sys.fixes) f.resetDt?.();
};

const posInt = (w: string | undefined, what: string): number => {
  const v = Number(w);
  if (w === undefined || !Number.isInteger(v) || v < 1) throw new StyleError(`${what} must be a positive integer, got '${w ?? ''}'`);
  return v;
};

const bound = (w: string | undefined, what: string): number | null => {
  if (w === 'NULL') return null;
  const v = Number(w);
  if (w === undefined || !Number.isFinite(v) || !(v > 0)) throw new StyleError(`fix dt/reset: ${what} must be NULL or a positive time, got '${w ?? ''}'`);
  return v;
};

export class FixDtReset extends Fix {
  readonly style = 'dt/reset';
  private readonly tmin: number | null;
  private readonly tmax: number | null;
  private readonly xmax: number;
  private readonly emax: number | null;
  private readonly unitsBox: boolean;
  /** Last timestep on which dt changed (f_ID). */
  private lastReset = 0;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 4) throw new StyleError(`usage: fix ${id} group-ID dt/reset N Tmin Tmax Xmax [emax value] [units lattice|box]`);
    this.nevery = posInt(args[0], `fix ${id} (dt/reset): N`);
    this.tmin = bound(args[1], 'Tmin');
    this.tmax = bound(args[2], 'Tmax');
    if (this.tmin !== null && this.tmax !== null && this.tmin > this.tmax) {
      throw new StyleError(`fix ${id} (dt/reset): Tmin must not exceed Tmax`);
    }
    const xmax = Number(args[3]);
    if (!Number.isFinite(xmax) || !(xmax > 0)) throw new StyleError(`fix ${id} (dt/reset): Xmax must be a positive distance, got '${args[3]}'`);
    this.xmax = xmax;
    let emax: number | null = null;
    let unitsBox = false;
    for (let k = 4; k < args.length; k += 2) {
      const key = args[k];
      // As native LAMMPS does (src/fix_dt_reset.cpp FixDtReset::FixDtReset at b332f28): only emax and units are
      // keywords, any other word is "Illegal fix dt/reset command".
      if (key !== 'emax' && key !== 'units') throw new StyleError('Illegal fix dt/reset command');
      const value = args[k + 1];
      if (value === undefined) throw new StyleError(`fix ${id} (dt/reset): keyword '${key}' needs a value`);
      if (key === 'emax') {
        const e = Number(value);
        if (!Number.isFinite(e) || !(e > 0)) throw new StyleError(`fix ${id} (dt/reset): emax must be a positive energy, got '${value}'`);
        emax = e;
      } else if (key === 'units') {
        if (value !== 'lattice' && value !== 'box') throw new StyleError(`fix ${id} (dt/reset): units must be lattice or box, got '${value}'`);
        unitsBox = value === 'box';
      }
    }
    this.emax = emax;
    this.unitsBox = unitsBox;
    this.scalarFlag = true;
    this.extscalar = 0;
  }

  /** The timestep is chosen in setup (before the first step of a run), not by the force hook. */
  setup(): void { this.reset(); }

  /** Called by the run loop on steps that are multiples of N (fix dt/reset: "re-compute dt every N timesteps"). */
  endOfStep(): void { this.reset(); }

  /** Xmax in box units: the lattice spacing scales it under "units lattice" (the default). */
  private xmaxBox(): number {
    if (this.unitsBox) return this.xmax;
    return this.xmax * (this.sys.lattice ? this.sys.lattice.spacing[0] : 1);
  }

  private reset(): void {
    const s = this.sys.state;
    const xmax = this.xmaxBox();
    let best = Infinity;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const dti = this.atomDt(i, xmax);
      if (dti < best) best = dti;
    }
    let dt = best === Infinity ? UNLIMITED_DT : best;
    if (this.tmin !== null && dt < this.tmin) dt = this.tmin;
    if (this.tmax !== null && dt > this.tmax) dt = this.tmax;
    if (dt !== s.dt) {
      setTimestep(this.sys, dt);
      this.lastReset = s.step;
    }
  }

  /**
   * Largest timestep for atom i. With acceleration a = F ftm2v / m and velocity v, the displacement
   * over one step is estimated as (v + a T / 2) dt, where T is the time the acceleration acts:
   * T = min(X/|v|, sqrt(2X/|a|), dte), with dte = emax / (|F| |v|) when emax is given. The estimate
   * is conservative (dt <= X / |v + a T / 2|). The acceleration enters only when a.v >= 0 (a
   * decelerating atom keeps X / |v|). The emax criterion: the kinetic-energy change over one step
   * is estimated as |F| |v| dt, so dt <= dte.
   * Measured with native LAMMPS (black box, one atom, pair zero, lj): v=(0.5,0), a=(1,0), X=0.1 gives
   * 1/6 = 0.1 / (0.5 + 0.1); a=(0,1) gives 0.19612 = 0.1 / |(0.5, 0.1)|; v=0 gives sqrt(2X/|a|);
   * a opposing v gives X/|v|; emax 0.01 with v=(0.5,0), F=(1,0) gives 0.02 = dte; and a single
   * atom with v=(2.00,-2.75,-0.47), F=(37.6,-51.2,-252.0), X=0.1, emax 20 gives
   * X / |v + a T / 2| with T = dte = 0.0224025 (the acceleration time is bounded by dte).
   */
  private atomDt(i: number, xmax: number): number {
    const s = this.sys.state;
    const k = 3 * i;
    const v = s.v, f = s.f;
    const vx = v[k], vy = v[k + 1], vz = v[k + 2];
    const c = s.units.ftm2v / massOf(s, i);
    const ax = f[k] * c, ay = f[k + 1] * c, az = f[k + 2] * c;
    const vv = Math.sqrt(vx * vx + vy * vy + vz * vz);
    const aa = Math.sqrt(ax * ax + ay * ay + az * az);
    const av = ax * vx + ay * vy + az * vz;
    const fm = Math.sqrt(f[k] * f[k] + f[k + 1] * f[k + 1] + f[k + 2] * f[k + 2]);
    const dte = this.emax !== null && fm * vv > 0 ? this.emax / (fm * vv) : Infinity;
    const dtv = vv > 0 ? xmax / vv : Infinity;
    const dtf = aa > 0 ? Math.sqrt((2 * xmax) / aa) : Infinity;
    const T = Math.min(dtv, dtf, dte);
    let wx = vx, wy = vy, wz = vz;
    if (aa > 0 && av >= 0) {
      wx += 0.5 * T * ax;
      wy += 0.5 * T * ay;
      wz += 0.5 * T * az;
    }
    const ww = Math.sqrt(wx * wx + wy * wy + wz * wz);
    const dtx = ww > 0 ? xmax / ww : Infinity;
    return Math.min(dtx, dte);
  }

  computeScalar(): number { return this.lastReset; }
}
