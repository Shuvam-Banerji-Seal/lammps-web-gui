import { FixNH } from './nh';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { ownCompute } from './util';

/*
 * fix nvt/sphere, npt/sphere, nph/sphere — docs.lammps.org/fix_nvt_sphere.html,
 * docs.lammps.org/fix_npt_sphere.html, docs.lammps.org/fix_nph_sphere.html.
 *
 * "Perform constant NVT integration to update position, velocity, and
 * angular velocity each timestep for finite-size spherical particles in
 * the group using a Nose/Hoover temperature thermostat." The same wording
 * with "NPT ... Nose/Hoover temperature thermostat and Nose/Hoover pressure
 * barostat" and "NPH ... using a Nose/Hoover pressure barostat".
 *
 * "The thermostat is applied to both the translational and rotational
 * degrees of freedom for the spherical particles, assuming a compute is used
 * which calculates a temperature that includes the rotational degrees of
 * freedom (see below)."
 *
 * "This fix computes a temperature each timestep. To do this, the fix creates
 * its own compute of style "temp/sphere", as if this command had been
 * issued:" with "compute fix-ID_temp group-ID temp/sphere" (nvt; the group is
 * the fix group). For npt and nph the docs give "compute fix-ID_temp all
 * temp/sphere" and "compute fix-ID_press all pressure fix-ID_temp".
 *
 * Implementation: the subclass reuses the whole fix nvt/npt/nph machinery
 * (src/engine/fix/nh.ts) and changes three things:
 *   1. the temperature compute is temp/sphere (translational plus rotational
 *      dof, "Point particles do not rotate" per compute_temp_sphere.html);
 *   2. the thermostat's velocity scaling also scales omega by the same factor
 *      (the docs' "applied to both the translational and rotational degrees of
 *      freedom");
 *   3. the half-step velocity kick also takes a half-step kick of the angular
 *      velocity, omega += dt/2 * ftm2v * torque / I, with I = 2/5 m r^2 for a
 *      sphere and 1/2 m r^2 for a disc (textbook moments of inertia, as in
 *      nve/sphere, see src/engine/fix/nve_sphere.ts).
 * The barostat's strain-rate scaling is left to fix nh, i.e. omega is not
 * scaled by the barostat. This is NOT yet verified: the npt/nph oracle cases
 * differ from native LAMMPS by a small drift that the point-particle fix
 * npt shows too (see the report accompanying this file).
 *
 * "This fix requires that atoms store torque and angular velocity (omega)
 * and a radius as defined by the :doc:`atom_style sphere <atom_style>`
 * command." "All particles in the group must be finite-size spheres. They
 * cannot be point particles." "Use of the *disc* keyword is only allowed for
 * 2d simulations, as defined by the :doc:`dimension <dimension>` keyword."
 * "The only difference between discs and spheres in this context is their
 * moment of inertia, as used in the time integration."
 *
 * update dipole / dipole/dlm: the docs list "update" as a keyword of these
 * fixes (dipole related keyword/value pairs from fix nh). The engine's
 * atom_style sphere has no dipole, so these are rejected with a StyleError
 * naming the keyword. Measured with native LAMMPS (black box): update dipole
 * without a dipole attribute errors with "requires atom attribute mu"; disc
 * in 3d errors with "disc option requires 2d simulation"; point particles in
 * the group error with "requires extended particles".
 */

type Hooks = {
  kick(): void;
  scaleVelocities(factor: number | number[]): void;
};

export class FixNHSphere extends FixNH {
  readonly style: string;
  private inertia = 0.4;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'nvt' | 'npt' | 'nph') {
    const name = `${variant}/sphere`;
    if (!sys.state.omega || !sys.state.radius || !sys.state.torque) {
      throw new StyleError(`Fix ${name} requires atom attribute omega`);
    }
    // keywords of this variant are stripped; everything else goes to fix nh
    const rest: string[] = [];
    let disc = false;
    for (let k = 0; k < args.length;) {
      if (args[k] === 'disc') {
        if (sys.dimension !== 2) throw new StyleError(`Fix ${name} disc option requires 2d simulation`);
        disc = true;
        k++;
      } else if (args[k] === 'update') {
        const w = args[k + 1];
        if (w === 'dipole' || w === 'dipole/dlm') {
          throw new StyleError(`Using update dipole flag requires atom attribute mu (fix ${name} update ${w}; atom_style sphere has no dipole)`);
        }
        throw new StyleError(`Illegal fix ${name} update value '${w ?? ''}'`);
      } else {
        rest.push(args[k]);
        k++;
      }
    }
    super(sys, id, group, rest, variant);
    this.style = name;
    this.inertia = disc ? 0.5 : 0.4;

    // "compute fix-ID_temp group-ID temp/sphere" (nvt) or "... all temp/sphere" (npt, nph)
    const tempGroup = variant === 'nvt' ? group : 'all';
    const tempId = `${id}_temp`;
    const tempCompute = ownCompute(sys, tempId, tempGroup, 'temp/sphere', []);
    (this as unknown as { tempCompute: unknown }).tempCompute = tempCompute;

    // The fix's hooks in fix nh call these two private methods on `this`;
    // shadow them on the instance so the sphere terms are applied in the same
    // places and order as the point-particle ones.
    const base = FixNH.prototype as unknown as Hooks;
    const self = this as unknown as Hooks;
    const baseKick = base.kick, baseScale = base.scaleVelocities;
    self.kick = () => {
      baseKick.call(this);
      this.kickOmega();
    };
    self.scaleVelocities = (factor) => {
      baseScale.call(this, factor);
      this.scaleOmega(factor);
    };
  }

  init(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & this.groupBit) && !(s.radius![i] > 0)) {
        throw new StyleError(`Fix ${this.style} requires extended particles`);
      }
    }
    super.init();
  }

  /** omega += dt/2 * ftm2v * torque / I for atoms in the group. */
  private kickOmega(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const dtfr = dtf / this.inertia;
    const { mask, radius, omega, torque } = s;
    const rmass = s.rmass!;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const c = dtfr / (radius![i] * radius![i] * rmass[i]);
      omega![3 * i] += c * torque![3 * i];
      omega![3 * i + 1] += c * torque![3 * i + 1];
      omega![3 * i + 2] += c * torque![3 * i + 2];
    }
  }

  /** The thermostat factor applied to velocities is applied to omega too. */
  private scaleOmega(factor: number | number[]): void {
    const s = this.sys.state;
    const fx = typeof factor === 'number' ? factor : factor[0];
    const fy = typeof factor === 'number' ? factor : factor[1];
    const fz = typeof factor === 'number' ? factor : factor[2];
    const omega = s.omega!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      omega[3 * i] *= fx; omega[3 * i + 1] *= fy; omega[3 * i + 2] *= fz;
    }
  }
}
