/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0: src/ORIENT/fix_orient_fcc.cpp and
 * fix_orient_fcc.h (fix orient/fcc). Contributing authors: Koenraad Janssens and David Olmsted (SNL). Copyright (2003)
 * Sandia Corporation (GNU General Public License). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * fix orient/fcc: the 12 nearest neighbours of an fcc crystal (docs.lammps.org/fix_orient.html), 6 reference vectors per
 * grain, cutsq = 0.5 * alat^2 * 1.57^2. The shared code is in ./fix_orient_lattice.ts.
 */
import type { System } from '../../engine/system';
import { FixOrientLattice } from './fix_orient_lattice';

export class FixOrientFCC extends FixOrientLattice {
  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args, { style: 'orient/fcc', nNb: 12, halfNn: 6, cutFactor: 0.5 });
  }
}
