import type { SimState } from './types';
import { massOf } from './atoms';

/*
 * Velocity Verlet (Swope et al., 1982; Allen & Tildesley §3.2) split into the
 * stages a fix can hook, in this order every step:
 *
 *   1. initialIntegrate   half kick v += dt/2 · F/m, then drift x += dt · v
 *   2. (positions wrapped into the box, image flags updated)
 *   3. forces computed by the ForceBackend
 *   4. postForce          e.g. Langevin friction + noise, enforce2d
 *   5. finalIntegrate     half kick v += dt/2 · F/m
 *   6. endOfStep          e.g. velocity-rescaling thermostats
 *
 * docs.lammps.org/fix_nve.html: "Perform plain time integration to update
 * position and velocity for atoms in the group each timestep. This creates a
 * system trajectory consistent with the microcanonical ensemble (NVE)".
 * docs.lammps.org/fix_enforce2d.html: "Zero out the z-dimension velocity and
 * force on each atom in the group."
 */

/** Where a run is, for fixes whose target ramps from start to stop. */
export interface RunContext {
  runStart: number;
  runStop: number;
}

/** Fraction of the current run completed, as fixes use it for T ramps. */
export const rampFraction = (s: SimState, ctx: RunContext): number =>
  ctx.runStop > ctx.runStart ? (s.step - ctx.runStart) / (ctx.runStop - ctx.runStart) : 0;

export interface Fix {
  readonly id: string;
  readonly style: string;
  /** True for fixes that move atoms (nve, nvt): at most one per run. */
  readonly integrates: boolean;
  setup?(s: SimState, ctx: RunContext): void;
  initialIntegrate?(s: SimState, ctx: RunContext): void;
  postForce?(s: SimState, ctx: RunContext): void;
  finalIntegrate?(s: SimState, ctx: RunContext): void;
  endOfStep?(s: SimState, ctx: RunContext): void;
}

/** Half kick: v += dt/2 · F/m (with the units' force-to-velocity factor). */
export const halfKick = (s: SimState): void => {
  const h = 0.5 * s.dt * s.units.ftm2v;
  const { v, f, type } = s;
  for (let i = 0; i < s.n; i++) {
    const c = h / massOf(s, i);
    v[3 * i] += c * f[3 * i];
    v[3 * i + 1] += c * f[3 * i + 1];
    v[3 * i + 2] += c * f[3 * i + 2];
  }
};

export const drift = (s: SimState): void => {
  const { x, v, dt } = s;
  for (let k = 0; k < 3 * s.n; k++) x[k] += dt * v[k];
};

/** Wraps positions into [lo, hi) on periodic dimensions, tracking images. */
export const wrapPositions = (s: SimState): void => {
  const { x, image, box } = s;
  for (let d = 0; d < 3; d++) {
    if (!box.periodic[d]) continue;
    const lo = box.lo[d];
    const L = box.hi[d] - lo;
    for (let i = 0; i < s.n; i++) {
      const k = 3 * i + d;
      if (x[k] >= lo && x[k] < lo + L) continue;
      const shift = Math.floor((x[k] - lo) / L);
      x[k] -= shift * L;
      if (x[k] >= lo + L) x[k] = lo;   // floating-point edge
      image[k] += shift;
    }
  }
};

export class FixNve implements Fix {
  readonly style = 'nve';
  readonly integrates = true;
  constructor(readonly id: string) {}
  initialIntegrate(s: SimState): void {
    halfKick(s);
    drift(s);
  }
  finalIntegrate(s: SimState): void {
    halfKick(s);
  }
}

export class FixEnforce2d implements Fix {
  readonly style = 'enforce2d';
  readonly integrates = false;
  constructor(readonly id: string) {}
  private zero(s: SimState): void {
    for (let i = 0; i < s.n; i++) {
      s.v[3 * i + 2] = 0;
      s.f[3 * i + 2] = 0;
    }
  }
  setup(s: SimState): void { this.zero(s); }
  postForce(s: SimState): void { this.zero(s); }
}
