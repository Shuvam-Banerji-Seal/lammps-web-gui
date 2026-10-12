/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/compute_ke_eff.cpp and
 * compute_ke_eff.h (class ComputeKEEff, compute ke/eff): the kinetic energy of an electron force field system,
 * including the radial kinetic energy mefactor m ervel^2 of each electron (|espin| = 1). Contributing author
 * (from the source): Andres Jaramillo-Botero. LAMMPS is Copyright (2003) Sandia Corporation and distributed under
 * the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';

export class ComputeKEEff extends Compute {
  readonly style: string = 'ke/eff';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length) throw new StyleError('Illegal compute ke/eff command');
    this.scalarFlag = true;
    this.extscalar = 1;
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Compute ke/eff requires atom style electron');
  }

  /** compute_ke_eff.cpp compute_scalar: 0.5 mvv2e times the sum of m v^2 and the electron radial terms. */
  protected computeScalar(): number {
    const s = this.sys.state;
    const { v, mask } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    let ke = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!(mask[i] & this.groupBit)) continue;
      const m = massOf(s, i);
      ke += m * (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]);
      if (Math.abs(spin[i]) === 1) ke += mefactor * m * ervel[i] * ervel[i];
    }
    return ke * (0.5 * s.units.mvv2e);
  }
}
