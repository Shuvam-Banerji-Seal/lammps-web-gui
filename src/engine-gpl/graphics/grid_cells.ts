/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/GRAPHICS/dump_image.cpp (DumpImage::create_image "render my grid cells", grid_cell_corners_2d,
 * grid_cell_corners_3d, reset_lighting and restore_lighting) and src/GRAPHICS/image.cpp (Image::draw_triangle, through
 * draw.ts). Copyright (2003) Sandia Corporation, distributed under the GNU General Public License (the LAMMPS file
 * headers). This file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * docs.lammps.org/dump_image.html, the grid keyword: "grid value = per-grid quantity to use for coloring grid cells"; the
 * cells are coloured by the grid colour map (dump_modify gmap) with the opacity of dump_modify gtrans. Each cell is drawn
 * as triangles in the order of its cells (x fastest, then y, then z): two for a 2d rectangle, twelve for the surface of a
 * 3d cell, with the lighting reset for flat surfaces (ambient 0.9, key, fill and back lights 0.3).
 */
import type { GridCellsSpec, RasterFrame, Vec3 } from '../../engine/output/image_objects';
import { drawTriangle, type PixelLight } from './draw';

/** reset_lighting(image, 0.9, 0.3, 0.3, 0.3): "reset lighting for flat surfaces to make them brighter". */
const GRID_LIGHT: PixelLight = { ambient: 0.9, lights: { key: 0.3, fill: 0.3, back: 0.3 } };

/** DumpImage::create_image, the grid cells: corners of each cell and its triangles. */
export function drawGridCells(f: RasterFrame, g: GridCellsSpec): void {
  const [nx, ny, nz] = g.size;
  const { boxlo, prd } = g;
  const corners: Vec3[] = [];
  // grid_cell_corners_2d / grid_cell_corners_3d: corner n = x + 2 y (+ 4 z); orthogonal boxes from boxlo and the cell
  // widths prd/N, triclinic boxes from the fractional corners through Domain::lamda2x
  const cell = (ix: number, iy: number, iz: number, three: boolean): void => {
    corners.length = 0;
    if (!g.triclinic) {
      const xdelta = prd[0] / nx, ydelta = prd[1] / ny, zdelta = prd[2] / nz;
      for (let z = 0; z < (three ? 2 : 1); z++) {
        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) {
            corners.push([boxlo[0] + (ix + x) * xdelta, boxlo[1] + (iy + y) * ydelta, three ? boxlo[2] + (iz + z) * zdelta : 0.0]);
          }
        }
      }
    } else {
      const dx = 1.0 / nx, dy = 1.0 / ny, dz = 1.0 / nz;
      for (let z = 0; z < (three ? 2 : 1); z++) {
        for (let y = 0; y < 2; y++) {
          for (let x = 0; x < 2; x++) corners.push(g.lamda2x([(ix + x) * dx, (iy + y) * dy, three ? (iz + z) * dz : 0.0]));
        }
      }
    }
  };
  const c = corners;
  const op = g.opacity;
  let n = 0;
  if (g.dimension === 2) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        cell(ix, iy, 0, false);
        const color = g.colorOf(g.values[n++]);
        drawTriangle(f, c[0], c[1], c[3], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[3], c[2], color, op, GRID_LIGHT);
      }
    }
    return;
  }
  for (let iz = 0; iz < nz; iz++) {
    for (let iy = 0; iy < ny; iy++) {
      for (let ix = 0; ix < nx; ix++) {
        cell(ix, iy, iz, true);
        const color = g.colorOf(g.values[n++]);
        // lower x face, upper x face, lower y, upper y, lower z, upper z
        drawTriangle(f, c[0], c[4], c[6], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[6], c[2], color, op, GRID_LIGHT);
        drawTriangle(f, c[1], c[5], c[7], color, op, GRID_LIGHT);
        drawTriangle(f, c[1], c[7], c[3], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[1], c[5], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[5], c[4], color, op, GRID_LIGHT);
        drawTriangle(f, c[2], c[6], c[7], color, op, GRID_LIGHT);
        drawTriangle(f, c[2], c[7], c[3], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[2], c[3], color, op, GRID_LIGHT);
        drawTriangle(f, c[0], c[3], c[1], color, op, GRID_LIGHT);
        drawTriangle(f, c[4], c[5], c[7], color, op, GRID_LIGHT);
        drawTriangle(f, c[4], c[7], c[6], color, op, GRID_LIGHT);
      }
    }
  }
}
