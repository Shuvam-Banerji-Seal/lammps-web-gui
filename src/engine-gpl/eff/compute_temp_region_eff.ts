/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/EFF/compute_temp_region_eff.cpp and
 * compute_temp_region_eff.h (class ComputeTempRegionEff, compute temp/region/eff): the temperature of the atoms of a
 * group inside a region, for an electron force field system. The degrees of freedom are those of the nuclei in the
 * region (the electrons do not count), and the kinetic energy includes the radial motion of the electrons in the
 * region. Contributing author (from the source): Andres Jaramillo-Botero (Caltech). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * The velocity bias of the region variant (remove_bias) is not used by any eFF fix here; the inherited methods keep
 * the core behaviour and are not called by the eFF fixes.
 */
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf, ELECTRON_RVEL, ELECTRON_SPIN, isElectronStyle } from '../../engine/atoms';
import { ComputeTempRegion } from '../../engine/compute/temp_bias';

export class ComputeTempRegionEff extends ComputeTempRegion {
  // the style name is the one of the base class (ComputeTempRegion types it as a literal); the registry name is temp/region/eff

  constructor(sys: System, id: string, group: string, args: string[]) {
    // compute_temp_region_eff.cpp constructor: the atom style, then exactly one word, then the region
    if (!isElectronStyle(sys.state.atomStyle)) throw new StyleError('Compute temp/region/eff requires atom style electron');
    if (args.length !== 1) throw new StyleError('Illegal compute temp/region/eff command');
    if (!sys.regions.has(args[0])) throw new StyleError(`Region ${args[0]} for compute temp/region/eff does not exist`);
    super(sys, id, group, args);
  }

  /** compute_temp_region_eff.cpp compute_scalar: dof = dim (nuclei in region) - extra; electrons add their KE, not their dof. */
  protected override computeScalar(): number {
    const s = this.sys.state;
    const { v } = s;
    const spin = s.custom.get(ELECTRON_SPIN)!.data;
    const ervel = s.custom.get(ELECTRON_RVEL)!.data;
    const mefactor = this.sys.dimension / 4.0;
    let count = 0, ecount = 0;
    let t = 0.0;
    for (let i = 0; i < s.n; i++) {
      if (!this.counted(i)) continue;
      count++;
      const m = massOf(s, i);
      t += (v[3 * i] * v[3 * i] + v[3 * i + 1] * v[3 * i + 1] + v[3 * i + 2] * v[3 * i + 2]) * m;
      if (Math.abs(spin[i]) === 1) {
        t += mefactor * m * ervel[i] * ervel[i];
        ecount++;
      }
    }
    this.dof = this.sys.dimension * (count - ecount) - this.extraDof;
    if (this.dof < 0.0 && count > 0) throw new StyleError('Temperature compute degrees of freedom < 0');
    return this.dof > 0.0 ? s.units.mvv2e * t / (this.dof * s.units.boltz) : 0.0;
  }

  /** compute_temp_region_eff.cpp compute_vector: the six components of the region atoms, with the electron radial terms. */
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
