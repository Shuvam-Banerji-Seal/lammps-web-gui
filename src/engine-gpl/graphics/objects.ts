/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the object records that src/graphics.h
 * (enum SPHERE, CYLINDER, TRINORM, PIXMAP) and src/GRAPHICS/fix_graphics_*.cpp hand to dump image through
 * Fix::image(objs, parms). Here they are plain records, one per object, instead of a parms row of doubles.
 * LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2
 * (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */

export type Vec3 = [number, number, number];
export type Rgb = [number, number, number];

/** A sphere: atom type, centre and diameter (a negative diameter asks the renderer for its default size). */
export interface SphereObject {
  kind: 'sphere';
  type: number;
  x: number;
  y: number;
  z: number;
  diameter: number;
}

/** A cylinder between two end points. */
export interface CylinderObject {
  kind: 'cylinder';
  type: number;
  x1: number;
  y1: number;
  z1: number;
  x2: number;
  y2: number;
  z2: number;
  diameter: number;
}

/** A bond drawn between two atom positions; the two halves take the colour of their atom types. */
export interface BondObject {
  kind: 'bond';
  types: [number, number];
  x1: number;
  y1: number;
  z1: number;
  x2: number;
  y2: number;
  z2: number;
}

/** A triangle with per-vertex normals and the atom type of each vertex (fix graphics/chunk). */
export interface TriNormObject {
  kind: 'trinorm';
  types: [number, number, number];
  vertices: [Vec3, Vec3, Vec3];
  normals: [Vec3, Vec3, Vec3];
}

/** A raster image placed in space (fix graphics/labels image), with caller-supplied RGB rows (bottom row first). */
export interface PixmapObject {
  kind: 'pixmap';
  pos: Vec3;
  width: number;
  height: number;
  scale: number;
  transcolor: Rgb | null;
  rgb: Uint8Array;
}

export type GraphicsObject =
  | SphereObject
  | CylinderObject
  | BondObject
  | TriNormObject
  | PixmapObject;
