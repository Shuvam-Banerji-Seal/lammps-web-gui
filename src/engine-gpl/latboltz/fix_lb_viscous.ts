/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/LATBOLTZ/fix_lb_viscous.cpp and src/LATBOLTZ/fix_lb_viscous.h at commit b332f28 (fix lb/viscous: the
 * hydrodynamic force of the lattice-Boltzmann fluid, computed by fix lb/fluid, is added to the force of each atom of
 * its group, with the mass fraction m / (m + m_fluid)). Contributing authors: Frances Mackay, Santtu Ollila, Colin
 * Denniston (UWO). Copyright (2003) Sandia Corporation, GNU General Public License. This file is GPL-2.0 (see
 * src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_lb_viscous.html: "The lb/viscous fix ... applies the fluid force to the particles ... this fix
 * must be used with fix lb/fluid and the atoms must be in the fix lb/fluid group."
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { massOf } from '../../engine/atoms';
import type { FixLbFluid } from './fix_lb_fluid';

export class FixLbViscous extends Fix {
  readonly style = 'lb/viscous';
  private readonly fluid: FixLbFluid;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length > 0) throw new StyleError('Illegal fix lb/viscous command');
    const fluids = sys.fixes.filter((f): f is FixLbFluid => f.style === 'lb/fluid');
    if (fluids.length === 0) throw new StyleError('the lb/fluid fix must also be used if using the lb/viscous fix');
    this.fluid = fluids[0];
    const s = sys.state;
    for (let j = 0; j < s.n; j++) {
      if ((s.mask[j] & this.groupBit) && !(s.mask[j] & this.fluid.groupBit)) {
        throw new StyleError('Atoms must be in the fix lb/fluid group to apply a fluid force to them');
      }
    }
  }

  /** fix_lb_viscous.cpp post_force(): f = hydroF + m / (m + m_fluid) * f for the group's atoms. */
  postForce(): void {
    const s = this.sys.state;
    const f = s.f;
    for (let i = 0; i < s.n; i++) if (s.mask[i] & this.groupBit) this.fluid.slot(s.id[i]);
    const hyd = this.fluid.hydroF, mp = this.fluid.massp;
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const id = s.id[i];
      const massfactor = massOf(s, i) / (massOf(s, i) + mp[id]);
      f[3 * i] = hyd[3 * id] + massfactor * f[3 * i];
      f[3 * i + 1] = hyd[3 * id + 1] + massfactor * f[3 * i + 1];
      f[3 * i + 2] = hyd[3 * id + 2] + massfactor * f[3 * i + 2];
    }
  }
}
