/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/dump_image.cpp (DumpImage::create_image, the AUTO branch "render dynamic bonds for my atoms",
 * and the autobond keyword of DumpImage::DumpImage) and src/GRAPHICS/image.cpp (Image::draw_cylinder, through draw.ts).
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_image.html, the autobond keyword: "The implicit bonds are found by searching the pair-wise
 * neighbor list for pairs of atoms that are closer than the bond cutoff; the second argument is the bond diameter. The
 * color of the bond is derived from the color of the atoms forming the implicit bond. For unit styles metal and real an
 * additional condition is applied: if the mass of both atoms of a pair within the bond cutoff is lower than 3 atomic
 * mass units, a bond is **not** drawn".
 *
 * The core (dump_image.ts) hands over the atoms of the dump group, their positions, masses (metal and real units only),
 * their atom colours and the minimum-image rule. Native walks the neighbor list; with a full pair search every pair
 * within the cutoff is seen once, which is the same set for a current list. Each bond is two cylinders, from each end
 * to the middle of the bond (draw_cylinder with sflag 3), in the colour of its own atom.
 */
import type { AutoBondSpec, RasterFrame, Vec3 } from '../../engine/output/image_objects';
import { drawCylinder } from './draw';

/** DumpImage::create_image (AUTO branch): bonds between selected atom pairs closer than the cutoff, each drawn as two halves. */
export function drawAutoBonds(f: RasterFrame, b: AutoBondSpec): void {
  const cutsq = b.cutoff * b.cutoff;
  const x = b.x;
  const del: Vec3 = [0, 0, 0];
  for (let i = 0; i < b.n; i++) {
    if (!b.selected(i)) continue;
    for (let j = i + 1; j < b.n; j++) {
      if (!b.selected(j)) continue;
      // pairs the neighbor list does not hold (special_bonds weights 0 0) are not seen by native's search
      if (b.excluded && b.excluded(i, j)) continue;
      // the hydrogen rule of the docs: both masses below 3 means no bond (only in units metal and real)
      if (b.mass && b.mass(i) < 3.0 && b.mass(j) < 3.0) continue;
      del[0] = x[3 * j] - x[3 * i];
      del[1] = x[3 * j + 1] - x[3 * i + 1];
      del[2] = x[3 * j + 2] - x[3 * i + 2];
      b.minimumImage(del);
      const rsq = del[0] * del[0] + del[1] * del[1] + del[2] * del[2];
      if (!(rsq < cutsq)) continue;
      // native: xmid = x1 + 0.5 dx for atom 1, xmid = x2 - 0.5 dx for atom 2; each half in its atom's colour
      const x1: Vec3 = [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
      const x2: Vec3 = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
      const mid1: Vec3 = [x1[0] + 0.5 * del[0], x1[1] + 0.5 * del[1], x1[2] + 0.5 * del[2]];
      const mid2: Vec3 = [x2[0] - 0.5 * del[0], x2[1] - 0.5 * del[1], x2[2] - 0.5 * del[2]];
      drawCylinder(f, x1, mid1, b.colorOf(i), b.width, 3, 1);
      drawCylinder(f, mid2, x2, b.colorOf(j), b.width, 3, 1);
    }
  }
}
