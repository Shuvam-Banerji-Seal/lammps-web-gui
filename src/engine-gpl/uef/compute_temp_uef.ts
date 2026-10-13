/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/UEF/compute_temp_uef.cpp and src/UEF/compute_temp_uef.h at commit b332f28 (ComputeTempUef: the
 * kinetic tensor rotated into the lab frame while rot_flag is set; the scalar is unchanged).
 * Copyright (2003) Sandia Corporation (GNU General Public License); contributing author David Nicholson (MIT).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { ComputeTemp } from '../../engine/compute/temp';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { virialRot, type M3 } from './uef_utils';

/** The fix that owns the rotation (fix nvt/uef, npt/uef): its current rotation matrix. */
export interface UefRotationSource {
  getRot(out: M3): void;
}

/**
 * compute temp/uef (docs.lammps.org/compute_temp_uef.html, the fix nvt/uef page: "compute ID group temp/uef" is
 * created by the fix; the compute must be used with a fix nvt/npt/uef). Only the vector is rotated.
 */
export class ComputeTempUef extends ComputeTemp {
  readonly style = 'temp/uef';
  /** rot_flag: the kinetic tensor is reported in the lab frame (on) or in the frame the velocities are held in (off). */
  private rotFlag = true;
  private source: UefRotationSource | null = null;
  private readonly rotBuf: M3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  /** the timestep of the last invokeVector() and the tensor it produced (invoked_vector) */
  private stamp = -1;
  private readonly stamped = new Float64Array(6);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
  }

  /** Set by the owning fix: where the rotation matrix comes from. */
  setSource(src: UefRotationSource): void {
    this.source = src;
  }

  /** yes_rot (the fix's integrator turns the rotation on again after its own updates). */
  yesRot(): void {
    this.rotFlag = true;
    this.invalidate();
  }

  /** no_rot (the integrator works on velocities that are already in the lab frame). */
  noRot(): void {
    this.rotFlag = false;
    this.invalidate();
  }

  protected computeVector(): void {
    super.computeVector();
    if (this.rotFlag) {
      if (!this.source) throw new StyleError(`compute ${this.id} (temp/uef) has no owning fix nvt/npt/uef`);
      this.source.getRot(this.rotBuf);
      virialRot(this.vector, this.rotBuf);
    }
  }

  /**
   * compute_vector as the integrator invokes it: the tensor is computed now and stamped with the timestep
   * (LAMMPS sets invoked_vector = ntimestep). The fix's pressure calls it directly before its own pressure
   * compute in the anisotropic case (fix_nh.cpp: "temperature->compute_vector(); pressure->compute_vector();").
   */
  invokeVector(): void {
    this.computeVector();
    this.stamped.set(this.vector);
    this.stamp = this.sys.state.step;
  }

  /**
   * The kinetic tensor a pressure compute uses (compute_pressure_uef.cpp compute_vector: "if (temperature->invoked_vector
   * != update->ntimestep) temperature->compute_vector();"): the tensor stamped on this timestep, else a fresh one.
   * Within a timestep the initial-step tensor is therefore reused by the final-step pressure, as native does.
   */
  tensorForPressure(): Float64Array {
    if (this.stamp !== this.sys.state.step) this.invokeVector();
    return this.stamped;
  }
}
