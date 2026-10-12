/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/compute_temp_eff.cpp and
 * compute_temp_eff.h (class ComputeTempEff, compute temp/eff): the temperature of an electron force field system.
 * The kinetic energy of an electron (|espin| = 1) includes its radial motion, mefactor m ervel^2 with
 * mefactor = dimension / 4, and each electron removes dimension degrees of freedom. Contributing author (from the
 * source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { ComputeTemp } from '../../engine/compute/temp';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';

export class ComputeTempEff extends ComputeTemp {
  override readonly style: string = 'temp/eff';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, []);
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Compute temp/eff requires atom style electron');
    if (args.length) throw new StyleError(`compute temp/eff takes no arguments (got ${args.join(' ')})`);
  }

  /** compute_temp_eff.cpp dof_compute: d N - extra - fix, minus d for each electron. */
  override dofCompute(): void {
    const s = this.sys.state;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    let n = 0, electrons = 0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      n++;
      if (Math.abs(spin[i]) === 1) electrons++;
    }
    const d = this.sys.dimension;
    this.dof = d * n - this.extraDof - this.sys.dofRemoved(this.groupBit);
    this.dof -= d * electrons;
  }

  /** compute_temp_eff.cpp compute_scalar: sum m v^2 + mefactor m ervel^2 (electrons), times mvv2e / (dof boltz). */
  protected override computeScalar(): number {
    if (this.dynamicDof) this.dofCompute();
    const s = this.sys.state;
    const { v } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    let t = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      t += (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]) * m;
      if (Math.abs(spin[i]) === 1) t += mefactor * m * ervel[i] * ervel[i];
    }
    const tfactor = this.dof > 0.0 ? s.units.mvv2e / (this.dof * s.units.boltz) : 0.0;
    return t * tfactor;
  }

  /** compute_temp_eff.cpp compute_vector: the six components, the electron radial kinetic energy on the diagonal. */
  protected override computeVector(): void {
    const s = this.sys.state;
    const { v } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    const t = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      const m = massOf(s, i);
      t[0] += m * v[3 * i] * v[3 * i];
      t[1] += m * v[3 * i + 1] * v[3 * i + 1];
      t[2] += m * v[3 * i + 2] * v[3 * i + 2];
      t[3] += m * v[3 * i] * v[3 * i + 1];
      t[4] += m * v[3 * i] * v[3 * i + 2];
      t[5] += m * v[3 * i + 1] * v[3 * i + 2];
      if (Math.abs(spin[i]) === 1) {
        t[0] += mefactor * m * ervel[i] * ervel[i];
        t[1] += mefactor * m * ervel[i] * ervel[i];
        t[2] += mefactor * m * ervel[i] * ervel[i];
      }
    }
    for (let c = 0; c < 6; c++) this.vector[c] = t[c] * s.units.mvv2e;
  }
}
