import { FixNVE } from './nve';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { isEllipsoid } from '../atoms';
import { ellipsoidInertia, quatToMatrix, rotateTransposeVec, rotateVec } from '../compute/asphere';

/*
 * fix ID group nve/asphere — docs.lammps.org/fix_nve_asphere.html: "Perform
 * constant NVE integration to update position, velocity, orientation, and
 * angular velocity for aspherical particles in the group each timestep."
 * "This creates a system trajectory consistent with the microcanonical
 * ensemble." "This fix differs from the fix nve command, which assumes point
 * particles and only updates their position and velocity." "All particles in
 * the group must be finite-size." "They cannot be point particles, but they can
 * be aspherical or spherical as defined by their shape attribute."
 *
 * Per timestep (free rigid-body integration of an ellipsoid):
 *   initial: v += dt/2 F/m; x += dt v; L += dt/2 tau;
 *            q <- Richardson-extrapolated update of dq/dt = 1/2 (0, omega) q,
 *            omega = R I^-1 R^T L: one full step q_full, two half steps to
 *            q_half2 (omega recomputed at the half-step orientation), and
 *            q = 2 q_half2 - q_full, normalised after every stage;
 *   final:   v += dt/2 F/m; L += dt/2 tau.
 * The principal moments of a solid ellipsoid are I = m/5 (b^2+c^2),
 * m/5 (a^2+c^2), m/5 (a^2+b^2) in the body frame (textbook). The combination
 * 2 q_half2 - q_full (not the 4/3 Richardson weights) is the one that matches
 * native LAMMPS (black box): with the same 0.002 step, the rotational energy of
 * one ellipsoid drifts 2.7e-12 over 20 steps in both, see oracle w19asphere_top.
 *
 * fix nve/asphere/noforce (docs.lammps.org/fix_nve_asphere_noforce.html): "Perform updates of
 * position and orientation, but not velocity or angular momentum for atoms in the group each timestep."
 * The same orientation update with the stored velocity and angular momentum, no force or torque kicks.
 *
 * Errors: a point particle in the group stops the run (measured with native LAMMPS,
 * black box: the fix requires extended particles).
 */

const DT_HALF_ORIENT = 0.5;

/** One Richardson step of the orientation quaternion q (w i j k) over the full time step h. */
const richardsonStep = (q: Float64Array, off: number, L: Float64Array, lo: number, I: [number, number, number], h: number): void => {
  const q0 = q[off], q1 = q[off + 1], q2 = q[off + 2], q3 = q[off + 3];
  const omega = (w0: number, w1: number, w2: number, w3: number): [number, number, number] => {
    const R = quatToMatrix(w0, w1, w2, w3);
    const wb = rotateTransposeVec(R, L[lo], L[lo + 1], L[lo + 2]);
    return rotateVec(R, wb[0] / I[0], wb[1] / I[1], wb[2] / I[2]);
  };
  /** (0,w) * q: the quaternion derivative times 2. */
  const wq = (w: [number, number, number], a0: number, a1: number, a2: number, a3: number): [number, number, number, number] => [
    -(w[0] * a1 + w[1] * a2 + w[2] * a3),
    a0 * w[0] + (w[1] * a3 - w[2] * a2),
    a0 * w[1] + (w[2] * a1 - w[0] * a3),
    a0 * w[2] + (w[0] * a2 - w[1] * a1),
  ];
  const normalise = (a: number[]): number[] => {
    const n = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
    return [a[0] / n, a[1] / n, a[2] / n, a[3] / n];
  };
  // full step
  const w = omega(q0, q1, q2, q3);
  const d = wq(w, q0, q1, q2, q3);
  const qfull = normalise([0, 1, 2, 3].map((k) => [q0, q1, q2, q3][k] + DT_HALF_ORIENT * h * d[k]));
  // first half step
  const qh = normalise([0, 1, 2, 3].map((k) => [q0, q1, q2, q3][k] + DT_HALF_ORIENT * 0.5 * h * d[k]));
  // second half step, omega recomputed at the half-step orientation
  const wh = omega(qh[0], qh[1], qh[2], qh[3]);
  const dh = wq(wh, qh[0], qh[1], qh[2], qh[3]);
  const qh2 = normalise([0, 1, 2, 3].map((k) => qh[k] + DT_HALF_ORIENT * 0.5 * h * dh[k]));
  const qn = normalise([0, 1, 2, 3].map((k) => 2 * qh2[k] - qfull[k]));
  q[off] = qn[0]; q[off + 1] = qn[1]; q[off + 2] = qn[2]; q[off + 3] = qn[3];
};

export class FixNVEAsphere extends FixNVE {
  readonly style: string = 'nve/asphere';
  /** fix nve/asphere/noforce: no velocity or angular momentum update (see initialIntegrate). */
  protected noforce = false;

  constructor(sys: System, id: string, group: string, args: string[], noforce = false) {
    super(sys, id, group, args);
    this.noforce = noforce;
    if (args.length) throw new StyleError(`Illegal fix ${noforce ? 'nve/asphere/noforce' : 'nve/asphere'} keyword ${args[0]}`);
    if (!sys.state.shape || !sys.state.quat || !sys.state.angmom || !sys.state.torque) {
      throw new StyleError('Fix nve/asphere requires atom style ellipsoid');
    }
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (!isEllipsoid(s, i)) throw new StyleError('Fix nve/asphere requires extended particles');
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    if (this.noforce) {
      // fix nve/asphere/noforce: positions and orientation move with the stored velocity and angular momentum
      const { x, v } = s;
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        x[3 * i] += this.dtv * v[3 * i]; x[3 * i + 1] += this.dtv * v[3 * i + 1]; x[3 * i + 2] += this.dtv * v[3 * i + 2];
      }
    } else {
      super.initialIntegrate();
      this.kickAngmom();
    }
    const dtq = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      richardsonStep(s.quat!, 4 * i, s.angmom!, 3 * i, ellipsoidInertia(s, i), dtq);
    }
  }

  finalIntegrate(): void {
    if (this.noforce) return;
    super.finalIntegrate();
    this.kickAngmom();
  }

  /** L += dtf * torque for the group (the half-step angular momentum kick). */
  private kickAngmom(): void {
    const s = this.sys.state;
    const L = s.angmom!, tq = s.torque!;
    const dtf = this.dtf;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      L[3 * i] += dtf * tq[3 * i];
      L[3 * i + 1] += dtf * tq[3 * i + 1];
      L[3 * i + 2] += dtf * tq[3 * i + 2];
    }
  }
}

