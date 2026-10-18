/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/fix_store_force.cpp and
 * src/fix_store_force.h at commit b332f28 (fix store/force). Copyright (2003) Sandia Corporation (LAMMPS development
 * team). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_store_force.html: "fix ID group-ID store/force"; "This fix stores the force on each atom, before
 * other fixes modify it, as a per-atom array with 3 columns (fx, fy, fz)"; the values of atoms outside the group are 0.
 * The stored forces are read by a per-atom reference such as f_ID[1] (dump, compute reduce, compute global/atom).
 * Forces are stored in post_force (and at setup and in minimization), as native.
 */
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';

export class FixStoreForce extends Fix {
  readonly style = 'store/force';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.peratomFlag = true;
    this.sizePeratomCols = 3;
    // zeroed at creation: a dump or a variable may read it before the first run
    this.arrayAtom = new Float64Array(3 * sys.state.n);
  }

  postForce(): void {
    const s = this.sys.state;
    if (this.arrayAtom.length !== 3 * s.n) this.arrayAtom = new Float64Array(3 * s.n);
    const out = this.arrayAtom;
    for (let i = 0; i < s.n; i++) {
      if (s.mask[i] & this.groupBit) {
        out[3 * i] = s.f[3 * i];
        out[3 * i + 1] = s.f[3 * i + 1];
        out[3 * i + 2] = s.f[3 * i + 2];
      } else {
        out[3 * i] = 0;
        out[3 * i + 1] = 0;
        out[3 * i + 2] = 0;
      }
    }
  }

  minPostForce(): void { this.postForce(); }
}
