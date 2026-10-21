/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/compute_chunk_atom.cpp (ComputeChunkAtom::compute_image, and the bin bounds of setup_xyz_bins) and
 * src/GRAPHICS/dump_image.cpp (the CYLINDER and TRIANGLE branches of the objects of create_image).
 * Copyright (2003) Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file headers). This
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_image.html, the compute keyword: "compute values = ID color cflag1 cflag2 ... graphics objects
 * provided by the compute". compute chunk/atom with bin/1d, bin/2d or bin/3d gives the walls between its bins: in 2d
 * one line per wall; in 3d the four edges of each wall for bin/3d and the wall as two triangles for bin/1d and bin/2d.
 * Each object has type 1 and diameter 0 (dump image adds cflag2 to the diameter and passes cflag1 as the cylinder caps).
 */
import type { ChunkBinImageSpec, ImageObject, ObjectStyle, RasterFrame, Vec3 } from '../../engine/output/image_objects';
import { drawCylinder, drawTriangle } from './draw';

/** A CYLINDER object: draw_cylinder(p1, p2, color, 0 + flag2, (int) flag1, opacity). */
const cylinder = (p1: Vec3, p2: Vec3): ImageObject => ({
  kind: 'draw',
  draw: (f: RasterFrame, st: ObjectStyle) => drawCylinder(f, p1, p2, st.colorOf(1), 0.0 + st.flag2, Math.trunc(st.flag1), st.opacity),
});

/** A TRIANGLE object: draw_triangle(p1, p2, p3, color, opacity). */
const triangle = (p1: Vec3, p2: Vec3, p3: Vec3): ImageObject => ({
  kind: 'draw',
  draw: (f: RasterFrame, st: ObjectStyle) => drawTriangle(f, p1, p2, p3, st.colorOf(1), st.opacity),
});

/** ComputeChunkAtom::compute_image for the bin/1d, bin/2d and bin/3d styles. */
export function chunkBinObjects(spec: ChunkBinImageSpec): ImageObject[] {
  const { binlo, binhi } = spec;
  // setup_xyz_bins: lo = the bin boundary at or below binlo, hi = the one at or above binhi, nlayers between them
  const layers = spec.bins.map((b) => {
    const idim = b.dim;
    const delta = b.delta, invdelta = 1.0 / delta, origin = b.origin;
    let lo: number, hi: number, n: number;
    if (origin < binlo[idim]) {
      n = Math.trunc((binlo[idim] - origin) * invdelta);
      lo = origin + n * delta;
    } else {
      n = Math.trunc((origin - binlo[idim]) * invdelta);
      lo = origin - n * delta;
      if (lo > binlo[idim]) lo -= delta;
    }
    if (origin < binhi[idim]) {
      n = Math.trunc((binhi[idim] - origin) * invdelta);
      hi = origin + n * delta;
      if (hi < binhi[idim]) hi += delta;
    } else {
      n = Math.trunc((origin - binhi[idim]) * invdelta);
      hi = origin - n * delta;
    }
    // std::lround: halves away from zero
    const q = (hi - lo) * invdelta;
    return { idim, offset: lo, delta, nlayers: Math.sign(q) * Math.round(Math.abs(q)) };
  });
  const map = (p: Vec3): Vec3 => (spec.reduced ? spec.lamda2x(p) : p);
  const out: ImageObject[] = [];
  for (const m of layers) {
    const idim = m.idim, idim1 = (idim + 1) % 3, idim2 = (idim + 2) % 3;
    for (let i = 0; i <= m.nlayers; i++) {
      const c = m.offset + i * m.delta;
      if (spec.dimension === 2) {
        const p1: Vec3 = [0, 0, 0], p2: Vec3 = [0, 0, 0];
        p1[idim] = p2[idim] = c;
        const other = 1 - idim;
        p1[other] = binlo[other];
        p1[2] = 1.0;
        p2[other] = binhi[other];
        p2[2] = 1.0;
        out.push(cylinder(map(p1), map(p2)));
      } else if (spec.ndim === 3) {
        const p: Vec3[] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
        for (let j = 0; j < 4; j++) p[j][idim] = c;
        p[0][idim1] = binlo[idim1]; p[0][idim2] = binlo[idim2];
        p[1][idim1] = binhi[idim1]; p[1][idim2] = binlo[idim2];
        p[2][idim1] = binhi[idim1]; p[2][idim2] = binhi[idim2];
        p[3][idim1] = binlo[idim1]; p[3][idim2] = binhi[idim2];
        const q = p.map(map);
        for (let j = 0; j < 4; j++) out.push(cylinder(q[j], q[(j + 1) % 4]));
      } else {
        const p1: Vec3 = [0, 0, 0], p2: Vec3 = [0, 0, 0], p3: Vec3 = [0, 0, 0], p4: Vec3 = [0, 0, 0];
        p1[idim] = p2[idim] = p3[idim] = p4[idim] = c;
        p1[idim1] = binlo[idim1]; p1[idim2] = binlo[idim2];
        p2[idim1] = binhi[idim1]; p2[idim2] = binlo[idim2];
        p3[idim1] = binhi[idim1]; p3[idim2] = binhi[idim2];
        p4[idim1] = binlo[idim1]; p4[idim2] = binhi[idim2];
        const [q1, q2, q3, q4] = [p1, p2, p3, p4].map(map);
        out.push(triangle(q1, q2, q3));
        out.push(triangle(q1, q3, q4));
      }
    }
  }
  return out;
}
