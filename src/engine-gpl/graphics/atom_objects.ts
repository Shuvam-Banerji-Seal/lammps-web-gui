/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/dump_image.cpp at b332f28, the lineflag and triflag branches of DumpImage::create_image (the
 * per-atom loops that draw atom_style line segments as cylinders and atom_style tri triangles as flat faces and/or
 * edges), with the lighting of the tri faces (reset_lighting(image, 0.3, 0.8, 0.45, 0.8) and restore_lighting) and
 * Image::draw_triangle and draw_cylinder of src/GRAPHICS/image.cpp (in draw.ts). The triangle corners come from the
 * engine's triangle body frame (src/engine/lineTri.ts triangleCorner, MathExtra::quat_to_mat and matvec in native).
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The core (src/engine/output/dump_image.ts) picks the atoms (group, line, tri or ellipsoid flag), their colours and
 * bonus data, and calls these loops in the order of native create_image (lines, triangles, then ellipsoids: ellipsoid_mesh.ts).
 */
import type { RasterFrame, Rgb255, AtomObjectsHook, LineAtom, TriAtom } from '../../engine/output/image_objects';
import { triangleCorner } from '../../engine/lineTri';
import { drawCylinder, drawTriangle, transparentAt, type PixelLight } from './draw';
import { drawEllipsoidAtoms } from './ellipsoid_mesh';
import { drawBodyAtoms } from './body_image';

type V = [number, number, number];

const unit = (c: Rgb255): V => [c[0] / 255, c[1] / 255, c[2] / 255];

/**
 * dump_image.cpp, lineflag branch: each segment is the centre plus and minus half its length along theta, in the
 * plane z = 0 (pt1[2] = pt2[2] = 0), drawn with draw_cylinder of the line width and both end caps (sflag 3).
 */
export function drawLineAtoms(f: RasterFrame, lines: readonly LineAtom[], diameter: number, opacity: number): void {
  for (const l of lines) {
    const dx = 0.5 * l.length * Math.cos(l.theta);
    const dy = 0.5 * l.length * Math.sin(l.theta);
    const pt1: V = [l.x + dx, l.y + dy, 0];
    const pt2: V = [l.x - dx, l.y - dy, 0];
    drawCylinder(f, pt1, pt2, unit(l.color), diameter, 3, opacity);
  }
}

/**
 * dump_image.cpp, triflag branch: tstyle 1 draws the faces only, 2 the edges only, any other value both (tridraw is
 * off for 2, edgedraw off for 1). A face is drawn with draw_triangle under the tri lighting (ambient 0.3, key 0.8,
 * fill 0.45, back 0.8: "brighten flat surfaces a bit"), the edges as three cylinders of diameter `diameter` with caps.
 */
export function drawTriAtoms(f: RasterFrame, tris: readonly TriAtom[], tstyle: number, diameter: number, opacity: number): void {
  const tridraw = tstyle !== 2;
  const edgedraw = tstyle !== 1;
  const faceLight: PixelLight = { ambient: 0.3, lights: { key: 0.8, fill: 0.45, back: 0.8 } };
  for (const t of tris) {
    const c = unit(t.color);
    const p1 = triangleCorner(t.center, t.quat, t.corners, 0);
    const p2 = triangleCorner(t.center, t.quat, t.corners, 1);
    const p3 = triangleCorner(t.center, t.quat, t.corners, 2);
    if (tridraw) drawTriangle(f, p1, p2, p3, c, opacity, faceLight);
    if (edgedraw) {
      drawCylinder(f, p1, p2, c, diameter, 3, opacity);
      drawCylinder(f, p2, p3, c, diameter, 3, opacity);
      drawCylinder(f, p3, p1, c, diameter, 3, opacity);
    }
  }
}

/** The lazy HOOKS entry 'atoms' (registry/graphics.ts): the line and triangle loops dump image calls through gplHook. */
export const atomObjectsHook: AtomObjectsHook = { drawLineAtoms, drawTriAtoms, transparentPixel: transparentAt, drawEllipsoidAtoms, drawBodyAtoms };
