/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/GRANSURF/fix_surface_local.cpp and src/GRANSURF/fix_surface_local.h (FixSurfaceLocal:
 * constructor argument handling, connectivity2d_complete for the lines of the surface), and src/GRANSURF/fix_surface.cpp.
 * Contributing authors of the GRANSURF package: Joel Clemmer (SNL), Dan Bolintineanu (SNL).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/fix_surface_local.html: "The fix surface/local command ... stores the connectivity of the line or
 * triangle particles that make up the surface, used by pair styles such as surf/granular." The connectivity itself
 * is built by the pair style (gransurf/connect2d.ts) at its first compute after the run is set up; this fix checks
 * the atom style and the arguments, and it has no other effect.
 *
 * Browser-engine difference: the keywords of fix surface/local (flat) are refused with a StyleError.
 */
import { Fix } from '../../engine/fix/fix';
import { StyleError } from '../../engine/force/types';
import type { System } from '../../engine/system';
import { isLineStyle, isTriStyle } from '../../engine/atoms';

export class FixSurfaceLocal extends Fix {
  readonly style: string = 'surface/local';

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    // fix_surface_local.cpp: the optional keywords are not supported in this engine
    if (args.length) throw new StyleError(`fix surface/local keyword '${args[0]}' is not supported`);
  }

  init(): void {
    const s = this.sys.state;
    if (this.sys.dimension === 2 && !isLineStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style line in 2d');
    if (this.sys.dimension === 3 && !isTriStyle(s.atomStyle)) throw new StyleError('Fix surface/local requires atom style tri in 3d');
  }
}
