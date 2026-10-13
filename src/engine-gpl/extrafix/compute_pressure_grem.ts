/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/REPLICA/compute_pressure_grem.cpp and compute_pressure_grem.h (ComputePressureGrem: init,
 * compute_scalar, compute_vector) at commit b332f28. Copyright Sandia Corporation (GNU General Public License).
 * Contributing authors of the gREM code: Edyta Malolepsza (Broad Institute), David Stelter and Tom Keyes (Boston
 * University). This file is GPL-2.0 (see src/engine-gpl/LICENSE).
 *
 * The pressure that fix grem creates for a barostat: the kinetic term uses the temperature of the gREM's own
 * temperature compute divided by the force scale factor of the fix, as the source does; the virial is unchanged.
 * Three dimensions only (the source's 2d branch is not ported).
 */
import { ComputePressure } from '../../engine/compute/pressure';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';

export class ComputePressureGrem extends ComputePressure {
  /**
   * scale(): the current force scale factor of the gREM fix (scale_grem). initialKinetic(): the kinetic tensor of the
   * temperature compute stored by the barostat at its initial pressure evaluation, or null before it (the scalar's
   * kinetic term uses it, as measured with native LAMMPS: the thermo pressure of step n carries the temperature of
   * the start of step n, see the header of tests/oracle/w108xf_grem_npt.in).
   */
  constructor(sys: System, id: string, tempId: string, private readonly scale: () => number, private readonly initialKinetic: () => Float64Array | null) {
    super(sys, id, 'all', [tempId]);
  }

  protected override computeScalar(): number {
    const sys = this.sys;
    const d = sys.dimension;
    if (d !== 3) throw new StyleError('compute PRESSURE/GREM is only supported in 3d');
    const s = sys.state;
    const vol = sys.geom.volume(d);
    const temp = sys.compute(this.tempId!);
    const kin = this.initialKinetic();
    // dof k_B T / scale of the source, with T = (trace of the kinetic tensor) / (dof k_B) at the stored evaluation
    const kinetic = kin ? (kin[0] + kin[1] + kin[2]) / this.scale() : (temp.dof * s.units.boltz * (temp.scalarValue() / this.scale()));
    const w = this.virial();
    const trace = w[0] + w[1] + w[2];
    return ((kinetic + trace) / 3.0) * (1.0 / vol) * s.units.nktv2p;
  }

  protected override computeVector(): void {
    const sys = this.sys;
    const d = sys.dimension;
    if (d !== 3) throw new StyleError('compute PRESSURE/GREM is only supported in 3d');
    const s = sys.state;
    const vol = sys.geom.volume(d);
    const w = this.virial();
    const scale = this.scale();
    const ke = this.keOverride ?? sys.compute(this.tempId!).vectorValues();
    for (let c = 0; c < 6; c++) this.vector[c] = ((ke[c] / scale + w[c]) / vol) * s.units.nktv2p;
  }
}
