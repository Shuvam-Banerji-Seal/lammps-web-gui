/*
 * Objects that fixes hand to dump image (docs.lammps.org/dump_image.html, keyword "fix ID type|element|const
 * flag1 flag2": "the fix must provide graphics objects; flag1 and flag2 are added to the object's sizes").
 * A fix that draws graphics (fix graphics/objects, graphics/lines, graphics/arrows) implements
 * Fix.imageObjects(); the renderer in dump_image.ts draws each object with its depth test.
 *
 * Geometry conventions as LAMMPS states them in the fix docs: a sphere's diameter, a cylinder's and a cone's end
 * diameters are full widths; an arrow is given by its centre, a unit direction, a length and the shaft diameter,
 * and its head takes the last `ratio` fraction of the length with a head diameter twice the shaft diameter
 * (the ArrowObj template of src/GRAPHICS/image_objects.cpp at b332f28, radius 0.1 and tip width 0.1 in its unit
 * length). The object shapes here are drawn by the browser renderer, which is not pixel-identical to native.
 */

export type Vec3 = [number, number, number];

/** An 8-bit RGB colour. */
export type Rgb255 = [number, number, number];

/**
 * The renderer state one snapshot's objects draw into (dump_image.ts builds it; the drawing follows image.cpp, see
 * src/engine-gpl/graphics/draw.ts). `rgb` is the image with row 0 at the top and `depth` the depth buffer with larger
 * values nearer the viewer and -Infinity for an empty pixel. The camera is Image::view_params' orthographic camera:
 * camRight, camUp and camDir are unit vectors in world space, ctr is the box centre (xctr, yctr, zctr), zdist the
 * camera distance and pixelWidth the world length of one pixel.
 */
export interface RasterFrame {
  w: number;
  h: number;
  rgb: Uint8Array;
  depth: Float64Array;
  camRight: Vec3;
  camUp: Vec3;
  camDir: Vec3;
  ctr: Vec3;
  zdist: number;
  pixelWidth: number;
}

/** What dump image's "fix ID style flag1 flag2" keyword gives a graphics object (dump_image.html). */
export interface ObjectStyle {
  /** flag1 and flag2 of the keyword (added to lengths and diameters, see dump_image.html). */
  flag1: number;
  flag2: number;
  /** Colour of an object of atom type `type` (the type, element or const style of the keyword). */
  colorOf(type: number): Rgb255;
  /** Default diameter of a sphere of type `type` (the dump's adiam setting). */
  diameterOf(type: number): number;
  /** Diameter of a bond between two atom types (the dump's bond width setting). */
  bondDiameter(t1: number, t2: number): number;
}

export type ImageObject =
  /** graphics objects drawn by the GPL drawing code in src/engine-gpl/graphics (their own kinds and shading). */
  | { kind: 'draw'; draw: (frame: RasterFrame, style: ObjectStyle) => void }
  /** sphere: centre and full diameter. */
  | { kind: 'sphere'; type: number; center: Vec3; diameter: number }
  /** cylinder: end points and full diameter (dump image's flag1 chooses sphere caps at the ends). */
  | { kind: 'cylinder'; type: number; p1: Vec3; p2: Vec3; diameter: number }
  /** cone: end points with their full diameters (the round cone; polygonal cones are refused by the fix). */
  | { kind: 'cone'; type: number; p1: Vec3; p2: Vec3; d1: number; d2: number }
  /** arrow: centre, unit direction, length, shaft diameter, tip fraction (0.1 .. 0.5). */
  | { kind: 'arrow'; type: number; center: Vec3; dir: Vec3; length: number; diameter: number; ratio: number };

/** Shaft and head of an arrow in world coordinates. */
export interface ArrowParts {
  shaft: { p1: Vec3; p2: Vec3; diameter: number };
  head: { p1: Vec3; p2: Vec3; d1: number; d2: number };
}

/**
 * Split an arrow into its shaft (diameter D) and head (base diameter 2D, length ratio * length). The arrow runs
 * from centre - length/2 * dir to centre + length/2 * dir.
 */
export const arrowParts = (a: { center: Vec3; dir: Vec3; length: number; diameter: number; ratio: number }): ArrowParts => {
  const n = Math.hypot(a.dir[0], a.dir[1], a.dir[2]) || 1;
  const u = [a.dir[0] / n, a.dir[1] / n, a.dir[2] / n] as Vec3;
  const h = 0.5 * a.length;
  const bot: Vec3 = [a.center[0] - h * u[0], a.center[1] - h * u[1], a.center[2] - h * u[2]];
  const tip: Vec3 = [a.center[0] + h * u[0], a.center[1] + h * u[1], a.center[2] + h * u[2]];
  const frac = 1 - a.ratio;
  const mid: Vec3 = [bot[0] + frac * (tip[0] - bot[0]), bot[1] + frac * (tip[1] - bot[1]), bot[2] + frac * (tip[2] - bot[2])];
  return {
    shaft: { p1: bot, p2: mid, diameter: a.diameter },
    head: { p1: mid, p2: tip, d1: 2 * a.diameter, d2: 0 },
  };
};
