/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/BODY/fix_nh_body.cpp and fix_nh_body.h
 * (FixNHBody: the angular momentum and orientation updates of the body thermostats and barostats), src/BODY/
 * fix_nvt_body.cpp, fix_npt_body.cpp and fix_nph_body.cpp (the temperature and pressure computes they create). LAMMPS
 * is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0);
 * this file is GPL-2.0 too (src/engine-gpl/LICENSE). Contributing author (from the source): Trung Dac Nguyen
 * (ndactrung@gmail.com), based on FixNHAsphere.
 *
 * docs.lammps.org/fix_nvt_body.html: "Perform constant NVT integration to update position, velocity, orientation, and
 * angular velocity each timestep for body particles in the group using a Nose/Hoover temperature thermostat."
 * docs.lammps.org/fix_npt_body.html and fix_nph_body.html: the NPT and NPH variants with the barostat of fix nh.
 *
 * Native FixNHBody (fix_nh_body.cpp): nve_v adds dtf torque to the angular momentum after the fix nh velocity half
 * step; nve_x applies a Richardson orientation step (MathExtra::richardson with mq_to_omega) to each body after the
 * position update; nh_v_temp multiplies the angular momentum by factor_eta with the velocities. The barostat's
 * velocity scaling is not applied to the angular momenta (fix nh does not scale them), as for the asphere variants.
 * The temperature compute is temp/body (compute_temp_body.ts) on the fix group for nvt and on all atoms for npt
 * and nph. A point particle (no body) in the group stops the run: "Fix nvt/nph/npt body requires bodies".
 *
 * The body's principal moments and orientation are the atom_style body record (bodies.ts) and the atom quaternion,
 * as for fix nve/body (src/engine/fix/nve_body.ts). The implementation shadows the fix nh hooks on the instance, as
 * the asphere variant does (src/engine/fix/nh_asphere.ts).
 */
import { FixNH } from '../../engine/fix/nh';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { ownCompute } from '../../engine/fix/util';
import { BODY_RECORD } from '../../engine/bodies';
import { richardson } from '../asphere/math_extra';

type Hooks = {
  kick(): void;
  drift(): void;
  scaleVelocities(factor: number | number[]): void;
};

export class FixNHBody extends FixNH {
  readonly style: string;

  constructor(sys: System, id: string, group: string, args: string[], variant: 'nvt' | 'npt' | 'nph') {
    const name = `${variant}/body`;
    super(sys, id, group, args, variant, name);
    this.style = name;

    // compute fix-ID_temp group-ID temp/body (nvt) or compute fix-ID_temp all temp/body (npt, nph)
    const tempGroup = variant === 'nvt' ? group : 'all';
    const tempCompute = ownCompute(sys, `${id}_temp`, tempGroup, 'temp/body', []);
    (this as unknown as { tempCompute: unknown }).tempCompute = tempCompute;

    // The hooks of fix nh call these methods on `this`; shadow them on the instance so the body terms run in the
    // same places as the asphere ones.
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
    // fix_nh_body.cpp init: the atom style (its message says Compute), then bodies in the group
    if (!s.quat || !s.angmom || !s.torque || !s.custom.has(BODY_RECORD)) throw new StyleError('Compute nvt/nph/npt body requires atom style body');
    const rec = s.custom.get(BODY_RECORD)!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      if (rec.data[rec.cols * i] === 0) throw new StyleError('Fix nvt/nph/npt body requires bodies');
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

  /** Orientation of the bodies in the group over the full step (native FixNHBody::nve_x). */
  private rotate(): void {
    const s = this.sys.state;
    const rec = s.custom.get(BODY_RECORD)!;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const b = rec.cols * i;
      richardson(s.quat!, 4 * i, s.angmom!, 3 * i, [rec.data[b + 1], rec.data[b + 2], rec.data[b + 3]], s.dt);
    }
  }

  /** The thermostat factor applied to velocities is applied to the angular momentum too (native nh_v_temp). */
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
