/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/DPD-SMOOTH/fix_rigid_meso.cpp and
 * src/DPD-SMOOTH/fix_rigid_meso.h (class FixRigidMeso: setup, initial_integrate, final_integrate, set_xv, set_v).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 * Contributing author of the source: Morteza Jalalvand (IASBS), an extension of fix rigid/nve to SPH/SDPD particles.
 *
 * The rigid-body integrator is the no_squish scheme of the Nose-Hoover base (fix_rigid_nh.ts: the conjugate quaternion
 * momentum conjqm), without thermostat or barostat (the source errors on both). On top of the rigid-body update the
 * atoms of each body carry their SPH state: set_xv and set_v add dtf * desph to esph and dtf * drho to rho (a half
 * step), and set_xv also sets the velocity estimate vest = 2 v - v_old.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isSphStyle, SPH_DESPH, SPH_DRHO, SPH_ESPH, SPH_RHO, SPH_VEST } from '../../engine/atoms';
import { FixRigidBase } from './fix_rigid';
import { invquatvec, matvec, noSquishRotate, qToExyz, quatvec, transposeMatvec, angmomToOmega, type Q4, type V3 } from './math_extra';

export class FixRigidMeso extends FixRigidBase {
  constructor(sys: System, id: string, group: string, args: string[], style: string) {
    super(sys, id, group, args, style);
    // fix_rigid_meso.cpp: "requires atom_style with both energy and density"; no thermostat or barostat
    if (!isSphStyle(sys.state.atomStyle)) {
      throw new StyleError(`fix ${style} command requires atom_style with both energy and density`);
    }
    if (this.thermo.tstat) throw new StyleError(`Can not use thermostat with fix ${style}`);
    if (this.press.pstat) throw new StyleError(`Can not use barostat with fix ${style}`);
    this.scalarFlag = false;
  }

  /**
   * fix_rigid.cpp (dimension 2): the body flags of the default (fflag z and tflag x, y are 0, set when the bodies
   * are built) and enforce2d after every body force and torque sum (FixRigid::setup and final_integrate).
   */
  init(): void {
    super.init();
    if (this.sys.dimension === 2) {
      for (const b of this.bodies) {
        b.fflag[2] = 0;
        b.tflag[0] = 0;
        b.tflag[1] = 0;
      }
    }
  }

  protected computeForcesAndTorques(): void {
    super.computeForcesAndTorques();
    if (this.sys.dimension !== 2) return;
    for (const b of this.bodies) {
      b.xcm[2] = 0;
      b.vcm[2] = 0;
      b.fcm[2] = 0;
      b.torque[0] = 0;
      b.torque[1] = 0;
      b.angmom[0] = 0;
      b.angmom[1] = 0;
      b.omega[0] = 0;
      b.omega[1] = 0;
    }
  }

  /** FixRigidMeso::setup: FixRigid::setup, then conjqm = 2 quatvec(quat, transpose(ex, ey, ez) angmom). */
  setup(): void {
    super.setup();
    const mbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      transposeMatvec(b.ex, b.ey, b.ez, b.angmom, mbody);
      const q: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, mbody, q);
      b.conjqm = q.map((v) => v * 2.0);
    }
  }

  /** FixRigidMeso::initial_integrate: half kick of vcm, the full position step, the no_squish rotation, then set_xv. */
  initialIntegrate(): void {
    const dtv = this.dtv, dtf = this.dtf, dtq = this.dtq;
    const dtf2 = dtf * 2.0;
    const mbody: V3 = [0, 0, 0], tbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      // step 1.1 - vcm by a half step; step 1.2 - xcm by a full step
      const dtfm = dtf / b.mass;
      for (let d = 0; d < 3; d++) b.vcm[d] += dtfm * b.fcm[d] * b.fflag[d];
      for (let d = 0; d < 3; d++) b.xcm[d] += dtv * b.vcm[d];
      // step 1.3 - torque (body frame) to the quaternion momentum
      for (let d = 0; d < 3; d++) b.torque[d] *= b.tflag[d];
      transposeMatvec(b.ex, b.ey, b.ez, b.torque, tbody);
      const fquat: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, tbody, fquat);
      for (let k = 0; k < 4; k++) b.conjqm[k] += dtf2 * fquat[k];
      // steps 1.4 to 1.13 - no_squish rotations
      noSquishRotate(3, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(2, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(1, b.conjqm, b.quat, b.inertia, dtv);
      noSquishRotate(2, b.conjqm, b.quat, b.inertia, dtq);
      noSquishRotate(3, b.conjqm, b.quat, b.inertia, dtq);
      // exyz from the quaternion, angular momentum and angular velocity from the conjugate momentum
      const e = qToExyz(b.quat);
      b.ex = e.ex;
      b.ey = e.ey;
      b.ez = e.ez;
      invquatvec(b.quat, b.conjqm, mbody);
      matvec(b.ex, b.ey, b.ez, mbody, b.angmom);
      for (let d = 0; d < 3; d++) b.angmom[d] *= 0.5;
      angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
    }
    this.virial.fill(0);
    this.setXV(true, 0.5);
  }

  /** FixRigidMeso::final_integrate: forces and torques, the second half kicks, then set_v. */
  finalIntegrate(): void {
    const dtf = this.dtf;
    const dtf2 = dtf * 2.0;
    this.computeForcesAndTorques();
    const mbody: V3 = [0, 0, 0], tbody: V3 = [0, 0, 0];
    for (const b of this.bodies) {
      const dtfm = dtf / b.mass;
      for (let d = 0; d < 3; d++) b.vcm[d] += dtfm * b.fcm[d] * b.fflag[d];
      for (let d = 0; d < 3; d++) b.torque[d] *= b.tflag[d];
      transposeMatvec(b.ex, b.ey, b.ez, b.torque, tbody);
      const fquat: Q4 = [0, 0, 0, 0];
      quatvec(b.quat, tbody, fquat);
      for (let k = 0; k < 4; k++) b.conjqm[k] += dtf2 * fquat[k];
      invquatvec(b.quat, b.conjqm, mbody);
      matvec(b.ex, b.ey, b.ez, mbody, b.angmom);
      for (let d = 0; d < 3; d++) b.angmom[d] *= 0.5;
      angmomToOmega(b.angmom, b.ex, b.ey, b.ez, b.inertia, b.omega);
    }
    this.setXV(false, 0.5);
  }

  /**
   * FixRigidMeso::set_xv (setX) and set_v: the FixRigid velocities, positions and virial, plus for every body atom
   * esph += dtf desph and rho += dtf drho (both calls) and, for set_xv, vest = 2 v - v_old.
   */
  protected setXV(setX: boolean, virialFactor: number): void {
    const s = this.sys.state;
    const idx = this.indexById();
    const esph = s.custom.get(SPH_ESPH)!.data, desph = s.custom.get(SPH_DESPH)!.data;
    const rho = s.custom.get(SPH_RHO)!.data, drho = s.custom.get(SPH_DRHO)!.data;
    const members: number[] = [];
    for (const b of this.bodies) for (const id of b.atoms) members.push(idx[id]);
    const vOld = new Float64Array(3 * members.length);
    members.forEach((i, n) => {
      esph[i] += this.dtf * desph[i];
      rho[i] += this.dtf * drho[i];
      for (let d = 0; d < 3; d++) vOld[3 * n + d] = s.v[3 * i + d];
    });
    super.setXV(setX, virialFactor);
    if (setX) {
      const vest = s.custom.get(SPH_VEST)!.data;
      members.forEach((i, n) => {
        for (let d = 0; d < 3; d++) vest[3 * i + d] = 2 * s.v[3 * i + d] - vOld[3 * n + d];
      });
    }
  }
}
