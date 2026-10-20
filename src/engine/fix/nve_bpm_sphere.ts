import { FixNVE } from './nve';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { updateDipoles } from './nve_sphere';

/*
 * fix ID group nve/bpm/sphere [disc] — docs.lammps.org/fix_nve_bpm_sphere.html. The doc says: "Perform
 * constant NVE integration to update position, velocity, angular velocity, and quaternion orientation for
 * finite-size spherical particles in the group each timestep." The page contrasts it with fix nve/sphere, which
 * "does not update a particle's orientation or quaternion" (that is the doc wording of fix nve/sphere).
 * The disc keyword: "If the disc keyword is used, then each particle is treated as a 2d disc (circle) instead
 * of as a sphere." The only difference is "their moment of inertia, as used in the time integration."
 * The doc says the fix "requires that atoms store torque, angular velocity (omega), a radius, and a quaternion
 * as defined by the atom_style bpm/sphere command", and that "They cannot be point particles."
 *
 * Per timestep (velocity-Verlet, as fix nve; the angular kicks use the torque and I = c m r^2 with c = 2/5
 * for a sphere and c = 1/2 for a disc, the textbook moments; the doc names only the moment of inertia):
 *   initial: v += dt/2 F/m; x += dt v; omega += dt/2 tau / I;
 *            q <- one Richardson step of dq/dt = 1/2 (0, omega) q over dt;
 *   final:   v += dt/2 F/m; omega += dt/2 tau / I.
 * The Richardson scheme is the one measured with native LAMMPS (black box): a single sphere with a constant
 * lab-frame angular velocity (0.3, 0.4, 1.0), dt 0.01, 200 steps: the quaternion of every step agrees with the
 * scheme below to 5e-16, while the exact rotation differs by 5e-6 at step 200 and the first-order update by
 * 1.6e-5 (oracle case w53bpm_sphere_spin). The scheme: q_full = q + h/2 d, q_half = q + h/4 d,
 * q_half2 = q_half + h/4 d(q_half), q_new = normalise(2 q_half2 - q_full), with d(q) = (0, omega) q and every
 * stage normalised; omega is the lab-frame angular velocity (a sphere's inertia is isotropic).
 *
 * The doc (fix_nve_bpm_sphere.html) says: "If the *update* keyword is used with the *dipole* value, then the
 * orientation of the dipole moment of each particle is also updated during the time integration, similar to the
 * fix nve/sphere command." The dipole follows the lab-frame angular velocity as in fix nve/sphere (see
 * rotateDipoles; the rule is measured for nve/sphere and the same form is checked for nve/bpm/sphere in
 * oracle case w59bpm_dipole). The dipole needs atom_style hybrid bpm/sphere dipole (the mu attribute).
 * The dipole/dlm integrator (the doc's Dullweber-Leimkuhler-McLachlan scheme) is refused by name.
 * Errors: a point particle in the group (native stops with the same requirement).
 */

/** One Richardson step of the orientation quaternion q (w i j k) over the full time step h, lab-frame omega. */
export const orientStep = (q: Float64Array, off: number, w: [number, number, number], h: number): void => {
  const dOf = (a0: number, a1: number, a2: number, a3: number): [number, number, number, number] => [
    -(w[0] * a1 + w[1] * a2 + w[2] * a3),
    a0 * w[0] + (w[1] * a3 - w[2] * a2),
    a0 * w[1] + (w[2] * a1 - w[0] * a3),
    a0 * w[2] + (w[0] * a2 - w[1] * a1),
  ];
  const normalise = (a: number[]): number[] => {
    const n = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
    return [a[0] / n, a[1] / n, a[2] / n, a[3] / n];
  };
  const q0 = q[off], q1 = q[off + 1], q2 = q[off + 2], q3 = q[off + 3];
  const d = dOf(q0, q1, q2, q3);
  const qfull = normalise([0, 1, 2, 3].map((k) => [q0, q1, q2, q3][k] + 0.5 * h * d[k]));
  const qh = normalise([0, 1, 2, 3].map((k) => [q0, q1, q2, q3][k] + 0.25 * h * d[k]));
  const dh = dOf(qh[0], qh[1], qh[2], qh[3]);
  const qh2 = normalise([0, 1, 2, 3].map((k) => qh[k] + 0.25 * h * dh[k]));
  const qn = normalise([0, 1, 2, 3].map((k) => 2 * qh2[k] - qfull[k]));
  q[off] = qn[0]; q[off + 1] = qn[1]; q[off + 2] = qn[2]; q[off + 3] = qn[3];
};

export class FixNVEBpmSphere extends FixNVE {
  readonly style: string = 'nve/bpm/sphere';
  private inertia = 0.4;
  /** update dipole: the dipole orientation follows the angular velocity after the quaternion step. */
  private updateDipole = false;
  /** update dipole/dlm: the DLM splitting of fix nve/sphere (updateDipoles in nve_sphere.ts). */
  private dlm = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    for (let k = 0; k < args.length; k++) {
      if (args[k] === 'disc') {
        if (sys.dimension !== 2) throw new StyleError('Fix nve/bpm/sphere disc requires 2d simulation');
        this.inertia = 0.5;
      } else if (args[k] === 'update') {
        const v = args[k + 1];
        if (v === 'dipole' || v === 'dipole/dlm') {
          this.updateDipole = true;
          this.dlm = v === 'dipole/dlm';
        } else throw new StyleError(`Unknown keyword in fix nve/bpm/sphere update command: ${v ?? '(missing)'}`);
        k++;
      } else throw new StyleError(`Unknown keyword in fix nve/bpm/sphere command: ${args[k]}`);
    }
    const s = sys.state;
    if (!s.omega || !s.quat || !s.torque || !s.radius) throw new StyleError('Fix nve/bpm/sphere requires atom style bpm/sphere');
    if (this.updateDipole && !s.mu) throw new StyleError('Fix nve/bpm/sphere update dipole requires atom attribute mu');
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !(s.radius![i] > 0)) throw new StyleError('Fix nve/bpm/sphere requires extended particles');
    }
  }

  /**
   * omega += dt/2 torque / I (I = c m r^2, c = inertia), the angular half-step kick. As native LAMMPS does
   * (src/BPM/fix_nve_bpm_sphere.cpp at b332f28), the factor is dtf / (c (r^2 m)), not dtf / c / (r^2 m) as in fix
   * nve/sphere: the two round differently.
   */
  private kickOmega(): void {
    const s = this.sys.state;
    const omega = s.omega!, torque = s.torque!, radius = s.radius!, rmass = s.rmass!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const c = this.dtf / (this.inertia * (radius[i] * radius[i] * rmass[i]));
      omega[3 * i] += c * torque[3 * i];
      omega[3 * i + 1] += c * torque[3 * i + 1];
      omega[3 * i + 2] += c * torque[3 * i + 2];
    }
  }

  initialIntegrate(): void {
    super.initialIntegrate();
    this.kickOmega();
    const s = this.sys.state;
    const omega = s.omega!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      orientStep(s.quat!, 4 * i, [omega[3 * i], omega[3 * i + 1], omega[3 * i + 2]], this.dtv);
    }
    if (this.updateDipole) updateDipoles(this.sys, this.groupBit, this.dlm, this.dtv, this.dtf);
  }

  finalIntegrate(): void {
    super.finalIntegrate();
    this.kickOmega();
  }
}
