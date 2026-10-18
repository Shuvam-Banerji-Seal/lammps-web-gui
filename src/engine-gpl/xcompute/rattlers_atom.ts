/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_rattlers_atom.cpp
 * and compute_rattlers_atom.h (ComputeRattlersAtom: compute rattlers/atom, the atoms that have fewer than a given number
 * of contacts and so do not take part in the force network of a granular packing). Contributing authors: Joel Clemmer
 * (SNL) and Ishan Srivastava (LBNL). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_rattlers_atom.html: "compute ID group-ID rattlers/atom style ncontacts max_tries" with style
 * type (contacts within the pair cutoff of the two types) or radius (contacts when the centres are closer than the sum
 * of the radii). The per-atom vector is 1 for a rattler, and the scalar is the number of rattlers.
 * Contacts come from the pair style's half list; a contact of two group atoms counts for both atoms (through the
 * ghosts, as the native reverse communication does). Each pass flags the atoms with too few contacts, and the passes
 * repeat until none is flagged or max_tries is reached.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { argInt } from './common';

export class ComputeRattlersAtom extends Compute {
  readonly style = 'rattlers/atom';
  private readonly cutByRadius: boolean;
  private readonly ncontactsRattler: number;
  private readonly maxTries: number;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length !== 3) throw new StyleError('Illegal compute rattlers/atom command');
    if (args[0] === 'type') this.cutByRadius = false;
    else if (args[0] === 'radius') this.cutByRadius = true;
    else throw new StyleError('Illegal compute rattlers/atom command');
    if (this.cutByRadius && !sys.state.radius) throw new StyleError('Compute rattlers/atom radius style requires atom attribute radius');
    this.ncontactsRattler = argInt(args[1]);
    this.maxTries = argInt(args[2]);
    this.scalarFlag = true;
    this.extscalar = 1;
    this.peratomFlag = true;
    this.vectorFlag = false;
    this.sizePeratomCols = 0;
  }

  init(): void {
    if (!this.sys.ff.pair) throw new StyleError('No pair style is defined for compute rattlers');
    // the granular pair styles that extend beyond contact (JKR, DMT) set this flag in native LAMMPS
    if ((this.sys.ff.pair as unknown as { beyondContact?: boolean }).beyondContact) {
      throw new StyleError('Compute rattlers does not currently support pair styles that extend beyond contact');
    }
  }

  protected computePeratom(): void {
    const sys = this.sys;
    const s = sys.state;
    const pair = sys.ff.pair;
    if (!pair) throw new StyleError('No pair style is defined for compute rattlers');
    sys.forces();
    const nb = sys.nb;
    const list = nb.half;
    if (!list) throw new StyleError('compute rattlers/atom: no half neighbour list is available');
    const n = s.n;
    const nt = pair.ntypes + 1;
    const cutsq = pair.cutsq;
    const x = nb.xall, owner = nb.owner, mask = s.mask, typeAll = nb.typeall;
    const radius = s.radius;
    const bit = this.groupBit;
    const rattler = new Uint8Array(n);
    const ncontacts = new Int32Array(n);
    let changeFlag = 1;
    let ntry = 0;
    while (ntry < this.maxTries) {
      changeFlag = 0;
      ncontacts.fill(0);
      for (const i of nativeOrder(s)) {
        if (!(mask[i] & bit)) continue;
        if (rattler[i] === 1) continue;
        const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
        const itype = typeAll[i];
        const jnum = list.numneigh[i];
        const k0: number = list.firstneigh[i];
        for (let jj = 0; jj < jnum; jj++) {
          const j: number = list.neighbors[k0 + jj] & NEIGHMASK;
          const o: number = owner[j];
          if (!(mask[o] & bit)) continue;
          if (rattler[o] === 1) continue;
          const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
          const rsq = delx * delx + dely * dely + delz * delz;
          if (!this.cutByRadius) {
            if (rsq >= cutsq[itype * nt + typeAll[j]]) continue;
          } else {
            const radsum = radius![i] + radius![o];
            if (rsq >= radsum * radsum) continue;
          }
          ncontacts[i] += 1;
          ncontacts[o] += 1;
        }
      }
      // atoms with too few contacts become rattlers (every local atom is tested, as the native loop does)
      for (let i = 0; i < n; i++) {
        if (ncontacts[i] < this.ncontactsRattler && rattler[i] === 0) {
          rattler[i] = 1;
          changeFlag = 1;
        }
      }
      if (changeFlag === 0) break;
      ntry += 1;
    }
    if (changeFlag === 1) sys.warn('Rattler calculation failed to converge within max tries');
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = rattler[i];
    this.vectorAtom = out;
  }

  protected computeScalar(): number {
    const v = this.peratomValues();
    let total = 0;
    for (let i = 0; i < this.sys.state.n; i++) if (v[i] === 1) total += 1;
    return total;
  }
}
