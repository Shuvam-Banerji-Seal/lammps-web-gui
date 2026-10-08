import { FixNH } from './nh';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { ownCompute } from './util';
import { isEllipsoid } from '../atoms';
import { ellipsoidInertia } from '../compute/asphere';
import { richardsonStep } from './nve_asphere';

/*
 * fix nvt/asphere, npt/asphere, nph/asphere — docs.lammps.org/fix_nvt_asphere.html,
 * docs.lammps.org/fix_npt_asphere.html, docs.lammps.org/fix_nph_asphere.html.
 *
 * "Perform constant NVT integration to update position, velocity,
 * orientation, and angular velocity each timestep for aspherical or
 * ellipsoidal particles in the group using a Nose/Hoover temperature
 * thermostat." NPT and NPH differ in the barostat, as in the sphere variants
 * (src/engine/fix/nh_sphere.ts).
 *
 * "This fix differs from the :doc:`fix nvt <fix_nh>` command, which
 * assumes point particles and only updates their position and velocity."
 * "The thermostat is applied to both the translational and rotational
 * degrees of freedom for the aspherical particles, assuming a compute is
 * used which calculates a temperature that includes the rotational
 * degrees of freedom (see below)."
 *
 * The docs say that the fix creates its own compute of style temp/asphere.
 * The code lines are: compute fix-ID_temp group-ID temp/asphere (nvt, with
 * the fix group), and compute fix-ID_temp all temp/asphere together with
 * compute fix-ID_press all pressure fix-ID_temp (npt and nph; the pressure
 * compute is built by fix nh). The temperature compute therefore counts the
 * 6N-3 degrees of freedom of compute temp/asphere with dof all (see
 * compute/asphere.ts).
 *
 * "This fix requires that atoms store torque and angular momentum and a
 * quaternion as defined by the :doc:`atom_style ellipsoid <atom_style>`
 * command." "All particles in the group must be finite-size." "They cannot be
 * point particles, but they can be aspherical or spherical as defined by
 * their shape attribute."
 *
 * Implementation: the subclass reuses the fix nvt/npt/nph machinery of
 * src/engine/fix/nh.ts and changes three things, in the same places in the
 * step as the sphere variant:
 *   1. the temperature compute is temp/asphere;
 *   2. the thermostat's velocity scaling also scales the angular momentum by
 *      the same factor (the thermostat acts on the rotational degrees of
 *      freedom too); L is the stored variable of atom_style ellipsoid, so the
 *      body angular velocity R I^-1 R^T L scales with it;
 *   3. the half-step kick also takes L += dt/2 * ftm2v * torque, and the drift
 *      updates the orientation with the same Richardson step as fix
 *      nve/asphere (richardsonStep in nve_asphere.ts) using the stored L.
 * The barostat's strain-rate scaling of velocities is left to fix nh: the
 * angular momenta are not scaled by it. Measured with native LAMMPS (black
 * box, oracle w19nhasphere_npt and w19nhasphere_nph): the npt and nph rows
 * match only without that scaling.
 *
 * Keywords: disc is accepted and changes nothing for ellipsoids in 3d
 * (measured with native LAMMPS, black box: identical thermo rows with and
 * without it over 20 steps); it is dropped before fix nh, which would refuse
 * it. An update keyword (dipoles) is refused by name, as a StyleError.
 * Measured with native LAMMPS (black box): a point particle in the temperature
 * group stops the run with the compute temp/asphere message, and an unknown
 * keyword is named in the error.
 */

type Hooks = {
  kick(): void;
  drift(): void;
  scaleVelocities(factor: number | number[]): void;
};

export class FixNHAsphere extends FixNH {
  readonly style: string;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'nvt' | 'npt' | 'nph') {
    const name = `${variant}/asphere`;
    if (!sys.state.shape || !sys.state.quat || !sys.state.angmom || !sys.state.torque) {
      throw new StyleError(`Fix ${name} requires atom style ellipsoid`);
    }
    // disc is accepted and changes nothing for ellipsoids in 3d (measured with
    // native LAMMPS, black box: the same thermo rows with and without it over 20
    // steps); it is dropped before fix nh, which would refuse it. The docs' keyword
    // list for these fixes is the fix nh one, so an update keyword is refused here by name.
    const rest: string[] = [];
    for (let k = 0; k < args.length; k++) {
      if (args[k] === 'update') {
        throw new StyleError(`Fix ${name} update ${args[k + 1] ?? ''}: updating dipoles needs a /sphere Nose-Hoover fix style (atom_style ellipsoid has no dipole)`);
      }
      if (args[k] !== 'disc') rest.push(args[k]);
    }
    super(sys, id, group, rest, variant);
    this.style = name;

    // compute fix-ID_temp group-ID temp/asphere (nvt) or compute fix-ID_temp all temp/asphere (npt, nph)
    const tempGroup = variant === 'nvt' ? group : 'all';
    const tempCompute = ownCompute(sys, `${id}_temp`, tempGroup, 'temp/asphere', []);
    (this as unknown as { tempCompute: unknown }).tempCompute = tempCompute;

    // The hooks of fix nh call these methods on `this`; shadow them on the
    // instance so the aspherical terms run in the same places as the point ones.
    const base = FixNH.prototype as unknown as Hooks;
    const self = this as unknown as Hooks;
    const baseKick = base.kick, baseDrift = base.drift, baseScale = base.scaleVelocities;
    self.kick = () => {
      baseKick.call(this);
      this.kickAngmom();
    };
    self.drift = () => {
      baseDrift.call(this);
      this.rotate();
    };
    self.scaleVelocities = (factor) => {
      baseScale.call(this, factor);
      this.scaleAngmom(factor);
    };
  }

  init(): void {
    const s = this.sys.state;
    // the temperature compute's group: the fix group for nvt, all atoms for npt and nph
    const tempBit = this.style === 'nvt/asphere' ? this.groupBit : this.sys.groupBit('all');
    for (let i = 0; i < s.n; i++) {
      if ((s.mask[i] & tempBit) && !isEllipsoid(s, i)) {
        // measured with native LAMMPS (black box): a point particle in the group stops the run in compute temp/asphere
        throw new StyleError('Compute temp/asphere requires all extended particles');
      }
    }
    super.init();
  }

  /** L += dt/2 * ftm2v * torque for atoms in the group. */
  private kickAngmom(): void {
    const s = this.sys.state;
    const dtf = 0.5 * s.dt * s.units.ftm2v;
    const L = s.angmom!, tq = s.torque!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      L[3 * i] += dtf * tq[3 * i];
      L[3 * i + 1] += dtf * tq[3 * i + 1];
      L[3 * i + 2] += dtf * tq[3 * i + 2];
    }
  }

  /** Orientation of atoms in the group over the full step, as in fix nve/asphere. */
  private rotate(): void {
    const s = this.sys.state;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      richardsonStep(s.quat!, 4 * i, s.angmom!, 3 * i, ellipsoidInertia(s, i), s.dt);
    }
  }

  /** The thermostat factor applied to velocities is applied to the angular momentum too. */
  private scaleAngmom(factor: number | number[]): void {
    const s = this.sys.state;
    const fx = typeof factor === 'number' ? factor : factor[0];
    const fy = typeof factor === 'number' ? factor : factor[1];
    const fz = typeof factor === 'number' ? factor : factor[2];
    const L = s.angmom!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      L[3 * i] *= fx; L[3 * i + 1] *= fy; L[3 * i + 2] *= fz;
    }
  }
}
