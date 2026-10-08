import type { SimState } from './types';
import { Rng } from './rng';
import { kineticEnergy, temperature } from './observables';
import { massOf } from './atoms';

/*
 * docs.lammps.org/velocity.html:
 *   create "generates an ensemble of velocities using a random number
 *   generator with the specified seed at the specified temperature";
 *   scale "computes the current temperature of the group of atoms and then
 *   rescales the velocities to the specified temperature";
 *   "The keyword defaults are dist = uniform, sum = no, mom = yes, rot = no,
 *   bias = no, loop = all, and units = lattice."
 *
 * Our generator is not LAMMPS's (rng.ts), so a given seed gives different
 * velocities; after `create` the temperature is exactly the requested one,
 * as in LAMMPS, so step-0 thermo output is reproducible.
 */

export interface CreateOptions {
  dist?: 'uniform' | 'gaussian';
  mom?: boolean;
  rot?: boolean;
}

/** velocity all create T seed [dist ...] [mom ...] [rot ...] */
export const createVelocities = (s: SimState, temp: number, seed: number, opts: CreateOptions = {}): void => {
  const rng = new Rng(seed);
  const gaussian = opts.dist === 'gaussian';
  const { v, type } = s;
  for (let i = 0; i < s.n; i++) {
    // per-component variance kT/m: scale the deviate by 1/sqrt(m)
    const c = 1 / Math.sqrt(massOf(s, i));
    for (let d = 0; d < 3; d++) {
      const r = gaussian ? rng.gaussian() : rng.uniform() - 0.5;
      v[3 * i + d] = d === 2 && s.dimension === 2 ? 0 : r * c;
    }
  }
  if (opts.mom ?? true) zeroMomentum(s);
  if (opts.rot ?? false) zeroAngularMomentum(s);
  scaleVelocities(s, temp);
};

/** velocity all scale T */
export const scaleVelocities = (s: SimState, temp: number): void => {
  const current = temperature(s, kineticEnergy(s));
  if (!(current > 0)) return;
  const factor = Math.sqrt(temp / current);
  for (let k = 0; k < 3 * s.n; k++) s.v[k] *= factor;
};

/** velocity all set vx vy vz (null leaves a component unchanged). */
export const setVelocities = (s: SimState, vx: number | null, vy: number | null, vz: number | null): void => {
  for (let i = 0; i < s.n; i++) {
    if (vx !== null) s.v[3 * i] = vx;
    if (vy !== null) s.v[3 * i + 1] = vy;
    if (vz !== null) s.v[3 * i + 2] = vz;
  }
};

/** Removes the centre-of-mass velocity. */
export const zeroMomentum = (s: SimState): void => {
  const p = [0, 0, 0];
  let mtot = 0;
  for (let i = 0; i < s.n; i++) {
    const m = massOf(s, i);
    mtot += m;
    for (let d = 0; d < 3; d++) p[d] += m * s.v[3 * i + d];
  }
  if (!(mtot > 0)) return;
  for (let i = 0; i < s.n; i++) {
    for (let d = 0; d < 3; d++) s.v[3 * i + d] -= p[d] / mtot;
  }
};

/**
 * Removes rigid-body rotation about the centre of mass, using unwrapped
 * positions: ω = I⁻¹ L, then v -= ω × (r - r_com).
 */
export const zeroAngularMomentum = (s: SimState): void => {
  const L = [0, 1, 2].map((d) => s.box.hi[d] - s.box.lo[d]);
  const pos = (i: number, d: number) => s.x[3 * i + d] + s.image[3 * i + d] * L[d];
  const com = [0, 0, 0];
  let mtot = 0;
  for (let i = 0; i < s.n; i++) {
    const m = massOf(s, i);
    mtot += m;
    for (let d = 0; d < 3; d++) com[d] += m * pos(i, d);
  }
  if (!(mtot > 0)) return;
  for (let d = 0; d < 3; d++) com[d] /= mtot;
  const ang = [0, 0, 0];
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < s.n; i++) {
    const m = massOf(s, i);
    const r = [pos(i, 0) - com[0], pos(i, 1) - com[1], pos(i, 2) - com[2]];
    const vv = [s.v[3 * i], s.v[3 * i + 1], s.v[3 * i + 2]];
    ang[0] += m * (r[1] * vv[2] - r[2] * vv[1]);
    ang[1] += m * (r[2] * vv[0] - r[0] * vv[2]);
    ang[2] += m * (r[0] * vv[1] - r[1] * vv[0]);
    const r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) I[a][b] += m * ((a === b ? r2 : 0) - r[a] * r[b]);
    }
  }
  let omega: number[];
  if (s.dimension === 2) {
    omega = [0, 0, I[2][2] > 0 ? ang[2] / I[2][2] : 0];
  } else {
    const inv = invert3(I);
    if (!inv) return;
    omega = [0, 1, 2].map((a) => inv[a][0] * ang[0] + inv[a][1] * ang[1] + inv[a][2] * ang[2]);
  }
  for (let i = 0; i < s.n; i++) {
    const r = [pos(i, 0) - com[0], pos(i, 1) - com[1], pos(i, 2) - com[2]];
    s.v[3 * i] -= omega[1] * r[2] - omega[2] * r[1];
    s.v[3 * i + 1] -= omega[2] * r[0] - omega[0] * r[2];
    s.v[3 * i + 2] -= omega[0] * r[1] - omega[1] * r[0];
  }
};

const invert3 = (m: number[][]): number[][] | null => {
  const [[a, b, c], [d, e, f], [g, h, k]] = m;
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) return null;
  const s = 1 / det;
  return [
    [A * s, -(b * k - c * h) * s, (b * f - c * e) * s],
    [B * s, (a * k - c * g) * s, -(a * f - c * d) * s],
    [C * s, -(a * h - b * g) * s, (a * e - b * d) * s],
  ];
};
