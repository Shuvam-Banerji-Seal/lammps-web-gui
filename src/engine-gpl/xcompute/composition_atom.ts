/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/EXTRA-COMPUTE/compute_composition_atom.cpp
 * and compute_composition_atom.h (ComputeCompositionAtom: compute composition/atom, the local composition of each atom's
 * neighbourhood). Contributing author: Megan McCarthy (SNL). Copyright (2003) Sandia Corporation (GNU General Public
 * License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/compute_composition_atom.html: "compute ID group-ID composition/atom keyword value ..." with the keyword
 * "cutoff". The per-atom array has 1 + ntypes columns: column 1 is the number of atoms in the sphere (the atom itself
 * included), columns 2.. are the fractions of each atom type among them.
 */
import { Compute } from '../../engine/compute/compute';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { nativeOrder } from '../../engine/atoms';
import { NEIGHMASK } from '../../engine/neighbor';
import { argNumeric, OccasionalSnapshot, occasionalCutoff, pairCutforce } from './common';
import { topologyRefusal } from './topology';

export class ComputeCompositionAtom extends Compute {
  readonly style = 'composition/atom';
  private readonly cutoffArg: number;
  private readonly snap = new OccasionalSnapshot();

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length > 2) throw new StyleError('Illegal compute composition/atom command');
    let cutoff = 0.0;
    for (let i = 0; i < args.length;) {
      if (args[i] === 'cutoff') {
        if (i + 2 > args.length) throw new StyleError('Illegal compute composition/atom command');
        cutoff = argNumeric(args[i + 1]);
        if (cutoff <= 0.0) throw new StyleError('Illegal compute composition/atom command');
        i += 2;
      } else {
        throw new StyleError('Illegal compute composition/atom command');
      }
    }
    this.cutoffArg = cutoff;
    this.peratomFlag = true;
    this.sizePeratomCols = 1 + sys.state.ntypes;
  }

  /** The checks of ComputeCompositionAtom::init(): a cutoff or a pair style is needed. */
  init(): void {
    if (!this.sys.ff.pair && this.cutoffArg === 0.0) {
      throw new StyleError('Compute composition/atom requires a cutoff be specified or a pair style be defined');
    }
  }

  protected computePeratom(): void {
    const sys = this.sys;
    const s = sys.state;
    const pair = sys.ff.pair;
    if (!pair && this.cutoffArg === 0.0) {
      throw new StyleError('Compute composition/atom requires a cutoff be specified or a pair style be defined');
    }
    sys.forces();
    topologyRefusal(sys, 'composition/atom');
    const nb = sys.nb;
    let cutoff = this.cutoffArg;
    if (pair && cutoff === 0.0) cutoff = pairCutforce(pair.cut);
    if (cutoff > nb.cutghost) {
      throw new StyleError('Compute composition/atom cutoff exceeds ghost atom range - use comm_modify cutoff command');
    }
    const cutsq = cutoff * cutoff;
    const ntypes = s.ntypes;
    const cols = 1 + ntypes;
    const result = new Float64Array(s.n * cols);
    this.arrayAtom = result;
    const cutflag = !pair || cutoff > pairCutforce(pair.cut) + nb.skin;
    const list = this.snap.get(sys, occasionalCutoff(sys, cutoff, cutflag));
    const x = nb.xall, typeall = nb.typeall;

    for (const i of nativeOrder(s)) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const jnum = list.numneigh[i];
      const k0 = list.firstneigh[i];
      // the atom itself counts once in its own type's column
      let count = 1;
      const itype = s.type[i];
      result[i * cols + itype] += 1;
      for (let jj = 0; jj < jnum; jj++) {
        const j = list.neighbors[k0 + jj] & NEIGHMASK;
        const jtype = typeall[j];
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < cutsq) {
          count++;
          result[i * cols + jtype] += 1;
        }
      }
      // column 0 is the total count; the others become fractions
      result[i * cols] = count;
      const lfac = 1.0 / count;
      for (let c = 1; c < cols; c++) result[i * cols + c] *= lfac;
    }
  }
}
