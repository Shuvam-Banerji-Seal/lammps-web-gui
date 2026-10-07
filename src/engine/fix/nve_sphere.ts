import { FixNVE } from './nve';
import { StyleError } from '../force/types';
import type { System } from '../system';

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
 */

export class FixNVESphere extends FixNVE {
  readonly style: string = 'nve/sphere';
  private inertia = 0.4;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    for (let k = 0; k < args.length;) {
      if (args[k] === 'disc') {
        if (sys.dimension !== 2) throw new StyleError('Fix nve/sphere disc requires 2d simulation');
        this.inertia = 0.5;
        k++;
      } else if (args[k] === 'update') {
        const w = args[k + 1];
        if (w !== 'dipole' && w !== 'dipole/dlm') throw new StyleError(`Unknown keyword in fix nve/sphere command: update ${w ?? ''}`.trim());
        throw new StyleError('Fix nve/sphere update dipole requires atom attribute mu');
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
  }

  finalIntegrate(): void {
    super.finalIntegrate();
    this.kickOmega();
  }
}
