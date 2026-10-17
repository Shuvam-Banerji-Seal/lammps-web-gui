/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/MISC/fix_srp_react.cpp and fix_srp_react.h at commit b332f28 (fix SRPREACT: the bond particles of pair
 * style srp/react; Vaibhav Palkar and Olga Kuksenok, Clemson University). LAMMPS is Copyright (2003) Sandia Corporation
 * and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * Fix SRP (fix_srp.ts) whose bond particles are made again after a neighbor build in which fix bond/break broke or fix
 * bond/create made bonds (post_neighbor runs setup_pre_force); the count of the bond fix is then reset to zero, as native
 * does ("not reseting breakcount would lead to redundant rebuilds"), which its output vector shows.
 */
import type { System } from '../../engine/system';
import type { FixBondBreak } from '../mc/fix_bond_break';
import type { FixBondCreate } from '../mc/fix_bond_create';
import { FixSRP } from './fix_srp';

export class FixSRPREACT extends FixSRP {
  readonly style: string = 'SRPREACT';
  idbreak: string | null = null;
  idcreate: string | null = null;
  private fbb: FixBondBreak | null = null;
  private fbc: FixBondCreate | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    this.pairName = 'srp/react';
  }

  override init(): void {
    super.init();
    const find = (id: string | null) => (id ? this.sys.fixes.find((f) => f.id === id) ?? null : null);
    this.fbb = find(this.idbreak) as FixBondBreak | null;
    this.fbc = find(this.idcreate) as FixBondCreate | null;
  }

  /** post_neighbor: rebuild the bond particles after bonds were broken or made. */
  postNeighbor(): void {
    if (this.fbb && this.fbb.breakcount) {
      this.setupPreForce();
      this.fbb.breakcount = 0;
    }
    if (this.fbc && this.fbc.createcount) {
      this.setupPreForce();
      this.fbc.createcount = 0;
    }
  }
}
