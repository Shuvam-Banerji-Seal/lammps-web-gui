/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source files (commit b332f28): src/stl_reader.cpp (STLReader::parse_text: the triangles are read up to the first
 * "endsolid" line; the reader returns after it, so a second solid in the same file is not read) and
 * src/GRANSURF/fix_surface.cpp (FixSurface::extract_from_stlfile, which reads the file with STLReader).
 * Copyright (2003) Sandia Corporation (LAMMPS development team, GNU General Public License). This file is GPL-2.0
 * (see src/engine-gpl/LICENSE).
 *
 * Browser-engine note: the mesh parser of the core (engine/stl.ts, used by create_atoms) reads every vertex of the file;
 * the surface fixes keep the first solid only, as native LAMMPS does (the screwfeeder example's container file holds
 * two copies of the same 13589 facets: native reads 13589 triangles from it).
 */

/** The text of an ASCII STL file up to and including its first endsolid line. */
export function stlFirstSolid(text: string): string {
  const m = /^[ \t]*endsolid[^\n]*$/m.exec(text);
  return m ? text.slice(0, m.index + m[0].length) : text;
}
