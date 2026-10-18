/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/GRANULAR/compute_contact_atom.cpp
 * and compute_contact_atom.h (ComputeContactAtom: compute contact/atom). LAMMPS is Copyright (2003) Sandia Corporation and
 * distributed under the GNU General Public License, version 2 (GPL-2.0). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_contact_atom.html: "compute ID group-ID contact/atom group2-ID" counts, for each atom, the number
 * of particles that touch it: "contact if distance <= sum of radii". The optional group2-ID "restricts which atoms to
 * consider for contacts". Each contact is tallied for both of its atoms, when one is in the compute group and the other
 * in group2 (as native update_i_flag / update_j_flag). The counts are exact integers, so the order of the pairs does not
 * change the result. Ghost counts go to their owners (native reverse communication). The pairs come from the engine's
 * half neighbour list, which holds every pair within the pair cutoff, as the occasional list of the source does.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import type { System } from '../../engine/system';

export class ComputeContactAtom extends Compute {
  readonly style = 'contact/atom';
  peratomFlag = true;
  sizePeratomCols = 0;
  private readonly jgroupbit: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 0 && args.length !== 1) throw new StyleError('Illegal compute contact/atom command');
    if (args.length === 1) {
      if (sys.groups.find(args[0]) < 0) throw new StyleError(`Compute contact/atom group2 ID ${args[0]} does not exist`);
      this.jgroupbit = sys.groups.bit(args[0]);
    } else {
      this.jgroupbit = sys.groups.bit('all');
    }
    if (!sys.state.radius) throw new StyleError('Compute contact/atom requires atom attribute radius');
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('Compute contact/atom requires a pair style be defined');
  }

  protected computePeratom(): void {
    const sys = this.sys;
    sys.forces();
    const s = sys.state;
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError('compute contact/atom: no neighbour list is available');
    const radius = s.radius!;
    const x = nb.xall, owner = nb.owner, nbr = list.neighbors;
    const contact = new Float64Array(s.n);
    const gbit = this.groupBit, jbit = this.jgroupbit;
    for (let i = 0; i < list.inum; i++) {
      const mi = s.mask[i];
      if (!(mi & gbit) && !(mi & jbit)) continue;
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      for (let k = k0; k < k1; k++) {
        const j = nbr[k] & NEIGHMASK;
        const oj = owner[j];
        const mj = s.mask[oj];
        const updateI = (mi & gbit) !== 0 && (mj & jbit) !== 0;
        const updateJ = (mj & gbit) !== 0 && (mi & jbit) !== 0;
        if (!updateI && !updateJ) continue;
        const delx = x[3 * i] - x[3 * j], dely = x[3 * i + 1] - x[3 * j + 1], delz = x[3 * i + 2] - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const radsum = radius[i] + radius[oj];
        const radsumsq = radsum * radsum;
        if (rsq <= radsumsq) {
          if (updateI) contact[i] += 1.0;
          if (updateJ) contact[oj] += 1.0;
        }
      }
    }
    this.vectorAtom = contact;
  }
}
