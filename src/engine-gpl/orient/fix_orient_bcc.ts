/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ORIENT/fix_orient_bcc.cpp and
 * fix_orient_bcc.h (fix orient/bcc). Contributing authors: Koenraad Janssens and David Olmsted (SNL); the bcc
 * modification by Tegar Wicaksono (UBC). Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix orient/bcc: the 8 nearest neighbours of a bcc crystal (docs.lammps.org/fix_orient.html), 4 reference vectors per
 * grain, cutsq = 0.75 * alat^2 * 1.57^2. The shared code is in ./fix_orient_lattice.ts.
 */
import type { System } from '../../engine/system';
import { FixOrientLattice } from './fix_orient_lattice';

export class FixOrientBCC extends FixOrientLattice {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, { style: 'orient/bcc', nNb: 8, halfNn: 4, cutFactor: 0.75 });
  }
}
