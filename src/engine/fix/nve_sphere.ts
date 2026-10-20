import { FixNVE } from './nve';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { gplHook } from '../gplHooks';

/*
 * fix ID group nve/sphere [update dipole|dipole/dlm] [disc] —
 * docs.lammps.org/fix_nve_sphere.html: "Perform constant NVE integration to
 * update position, velocity, and angular velocity for finite-size spherical
 * particles in the group each timestep." "If the *disc* keyword is used,
 * then each particle is treated as a 2d disc (circle) instead of as a
 * sphere.  This is only possible for 2d simulations ... The only difference
 * between discs and spheres in this context is their moment of inertia, as
 * used in the time integration."
 *
 * Positions and velocities follow fix nve (velocity-Verlet); the angular
 * velocity takes the same half-step kicks from the torque,
 *   omega += dt/2 * ftm2v * torque / I,   I = c m r^2,
 * with c = 2/5 for a solid sphere and 1/2 for a disc (textbook moments of
 * inertia). Errors as measured with native LAMMPS: point particles in the
 * group ("Fix nve/sphere requires extended particles"), disc outside 2d,
 * update dipole without a dipole moment (atom_style sphere has none, so the
 * engine rejects both update values), and an atom style without omega.
 *
 * update dipole/dlm: "The default dipole orientation integrator can be changed to the
 * Dullweber-Leimkuhler-McLachlan integration scheme (Dullweber) when using *update* with the value
 * *dipole/dlm*." The fix page names the scheme only; the DLM loop is ported from native LAMMPS into
 * src/engine-gpl/dipole/dlm.ts and reached through the lazy HOOKS word 'dipole/dlm'.
 */

/** The DLM orientation update of update dipole/dlm (lazy HOOKS word 'dipole/dlm', src/engine-gpl/registry/dipole_dlm.ts). */
export interface DipoleDlmHook {
  /** Rotates omega (3 per atom) and mu (4 per atom) of the group's atoms with |mu| > 0 over dtf / ftm2v. */
  dlmUpdate(n: number, mask: ArrayLike<number>, groupBit: number, omega: Float64Array, mu: Float64Array, dtf: number, ftm2v: number): void;
}

/**
 * update dipole and update dipole/dlm of fix nve/sphere, nve/bpm/sphere and nvt/npt/nph/sphere, after the position
 * and angular-velocity updates of the step. update dipole turns mu by the full step, measured with native LAMMPS
 * (black box): with a constant angular velocity the dipole after one step is |mu| (mu + dt w x mu) / |mu + dt w x mu|
 * (first step and the step after it checked to 6 digits, a 2 and 3 step run, |mu| = 1 and 2, w along z and along
 * (1,1,0)). As native LAMMPS does (src/fix_nve_sphere.cpp at b332f28), a dipole of zero length is left alone by both.
 */
export const updateDipoles = (sys: System, groupBit: number, dlm: boolean, dtv: number, dtf: number): void => {
  const s = sys.state;
  const { mask } = s;
  const mu = s.mu!, omega = s.omega!;
  if (dlm) {
    gplHook<DipoleDlmHook>('dipole/dlm').dlmUpdate(s.n, mask, groupBit, omega, mu, dtf, s.units.ftm2v);
    return;
  }
  for (let i = 0; i < s.n; i++) {
    if (!(mask[i] & groupBit) || !(mu[4 * i + 3] > 0)) continue;
    const px = mu[4 * i], py = mu[4 * i + 1], pz = mu[4 * i + 2], len = mu[4 * i + 3];
    const wx = omega[3 * i], wy = omega[3 * i + 1], wz = omega[3 * i + 2];
    const nx = px + dtv * (wy * pz - wz * py);
    const ny = py + dtv * (wz * px - wx * pz);
    const nz = pz + dtv * (wx * py - wy * px);
    const sc = len / Math.sqrt(nx * nx + ny * ny + nz * nz);
    mu[4 * i] = nx * sc; mu[4 * i + 1] = ny * sc; mu[4 * i + 2] = nz * sc;
  }
};

export class FixNVESphere extends FixNVE {
  readonly style: string = 'nve/sphere';
  private inertia = 0.4;
  /** update dipole: the dipole orientation follows the angular velocity (updateDipoles). */
  private updateDipole = false;
  /** update dipole/dlm: the orientation and omega take the DLM splitting instead. */
  private dlm = false;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    for (let k = 0; k < args.length;) {
      if (args[k] === 'disc') {
        if (sys.dimension !== 2) throw new StyleError('Fix nve/sphere disc requires 2d simulation');
        this.inertia = 0.5;
        k++;
      } else if (args[k] === 'update') {
        const w = args[k + 1];
        if (w === 'dipole' || w === 'dipole/dlm') {
          if (!sys.state.mu) throw new StyleError('Fix nve/sphere update dipole requires atom attribute mu');
          this.updateDipole = true;
          this.dlm = w === 'dipole/dlm';
        } else throw new StyleError(`Unknown keyword in fix nve/sphere update command: ${w ?? ''}`.trim());
        k += 2;
      } else throw new StyleError(`Unknown keyword in fix nve/sphere command: ${args[k]}`);
    }
    if (!sys.state.omega) throw new StyleError('Fix nve/sphere requires atom attribute omega');
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !(s.radius![i] > 0)) throw new StyleError('Fix nve/sphere requires extended particles');
    }
  }

  private kickOmega(): void {
    const s = this.sys.state;
    const { mask } = s;
    const radius = s.radius!, rmass = s.rmass!, omega = s.omega!, torque = s.torque!;
    const bit = this.groupBit;
    const dtfr = this.dtf / this.inertia;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & bit)) continue;
      const c = dtfr / (radius[i] * radius[i] * rmass[i]);
      omega[3 * i] += c * torque[3 * i];
      omega[3 * i + 1] += c * torque[3 * i + 1];
      omega[3 * i + 2] += c * torque[3 * i + 2];
    }
  }

  initialIntegrate(): void {
    super.initialIntegrate();
    this.kickOmega();
    if (this.updateDipole) updateDipoles(this.sys, this.groupBit, this.dlm, this.dtv, this.dtf);
  }

  finalIntegrate(): void {
    super.finalIntegrate();
    this.kickOmega();
  }
}
