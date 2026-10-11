import { FixNVE } from './nve';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { BODY_RECORD } from '../bodies';
import { richardsonStep } from './nve_asphere';

/*
 * fix ID group nve/body — docs.lammps.org/fix_nve_body.html: "Perform constant NVE integration to update position,
 * velocity, orientation, and angular velocity for body particles in the group each timestep." "This fix differs
 * from the fix nve command, which assumes point particles and only updates their position and velocity."
 * "All particles in the group must be body particles. They cannot be point particles."
 *
 * Per timestep, the same free rigid-body update as fix nve/asphere (nve_asphere.ts), with the principal moments
 * of the body (bodies.ts) in place of the ellipsoid's:
 *   initial: v += dt/2 F/m; x += dt v; L += dt/2 tau; q <- Richardson step; (see richardsonStep)
 *   final:   v += dt/2 F/m; L += dt/2 tau.
 * The body's translation and angular momentum are the COM values of the atom (the forces and torques come from
 * pair body/nparticle).
 */
export class FixNVEBody extends FixNVE {
  readonly style: string = 'nve/body';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError(`Illegal fix nve/body keyword ${args[0]}`);
    if (!sys.state.quat || !sys.state.angmom || !sys.state.torque || !sys.state.custom.has(BODY_RECORD)) {
      throw new StyleError('Fix nve/body requires atom style body');
    }
  }

  init(): void {
    super.init();
    const s = this.sys.state;
    const rec = s.custom.get(BODY_RECORD)!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (rec.data[rec.cols * i] === 0) throw new StyleError('Fix nve/body requires body particles (all particles in the group must be bodies)');
    }
  }

  initialIntegrate(): void {
    const s = this.sys.state;
    super.initialIntegrate();
    this.kickAngmom();
    const rec = s.custom.get(BODY_RECORD)!;
    const dtq = this.dtv;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const b = rec.cols * i;
      richardsonStep(s.quat!, 4 * i, s.angmom!, 3 * i, [rec.data[b + 1], rec.data[b + 2], rec.data[b + 3]], dtq);
    }
  }

  finalIntegrate(): void {
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
