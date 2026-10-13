/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/UEF/compute_pressure_uef.cpp and src/UEF/compute_pressure_uef.h at commit b332f28 (ComputePressureUef:
 * the virial rotated with the cached rotation matrix, the scalar averaged over the ext directions, no pair tail in
 * the tensor).
 * Copyright (2003) Sandia Corporation (GNU General Public License); contributing author David Nicholson (MIT).
 * This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 */
import { ComputePressure } from '../../engine/compute/pressure';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { virialRot, zeroRot, type M3 } from './uef_utils';
import { ComputeTempUef, type UefRotationSource } from './compute_temp_uef';

/**
 * compute pressure/uef: the pressure tensor in the lab frame, P_rot = Q P Q^t, with the rotation matrix the fix holds.
 * Used as "compute ID group pressure/uef temp-ID" by fix nvt/npt/uef (its ID_press compute).
 */
export class ComputePressureUef extends ComputePressure {
  readonly style = 'pressure/uef';
  /** ext: directions averaged into the scalar (the fix sets them from its ext keyword). */
  extFlags: [boolean, boolean, boolean] = [true, true, true];
  /** in_fix: the fix's own evaluations use the rotation cached at its last update (update_rot). */
  inFix = false;
  private source: UefRotationSource | null = null;
  private readonly rot: M3 = zeroRot();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
  }

  setSource(src: UefRotationSource): void {
    this.source = src;
  }

  /** update_rot: caches the fix's current rotation matrix (used by the next evaluations). */
  updateRot(): void {
    if (!this.source) throw new StyleError(`compute ${this.id} (pressure/uef) has no owning fix nvt/npt/uef`);
    this.source.getRot(this.rot);
  }

  protected computeScalar(): number {
    // all directions external: the stock scalar (trace of the virial and the temperature)
    if (this.extFlags[0] && this.extFlags[1] && this.extFlags[2]) return super.computeScalar();
    // otherwise average the tensor components of the external directions
    this.computeVector();
    let k = 0;
    let scalar = 0;
    if (this.extFlags[0]) { scalar += this.vector[0]; k++; }
    if (this.extFlags[1]) { scalar += this.vector[1]; k++; }
    if (this.extFlags[2]) { scalar += this.vector[2]; k++; }
    if (k > 1) scalar /= k;
    return scalar;
  }

  protected computeVector(): void {
    const s = this.sys.state;
    const w = Float64Array.from(this.rawVirial());
    let r: M3 = this.rot;
    if (!this.inFix) {
      if (!this.source) throw new StyleError(`compute ${this.id} (pressure/uef) has no owning fix nvt/npt/uef`);
      r = zeroRot();
      this.source.getRot(r);
    }
    virialRot(w, r);
    const vol = this.sys.geom.volume(this.sys.dimension);
    const invVolume = 1.0 / vol;
    const ke = this.keOverride ?? this.kineticTensor();
    for (let i = 0; i < 6; i++) {
      this.vector[i] = ((ke ? ke[i] : 0) + w[i]) * invVolume * s.units.nktv2p;
    }
  }

  /** The temperature's kinetic tensor (the stamped one of this timestep for temp/uef, see tensorForPressure). */
  private kineticTensor(): Float64Array | null {
    if (!this.tempId) return null;
    const t = this.sys.compute(this.tempId);
    return t instanceof ComputeTempUef ? t.tensorForPressure() : t.vectorValues();
  }
}
