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

/** The specular highlight of Image::draw_pixel: pow(n . half, hardness) * intensity (dump_modify specular). */
export interface ImageSpecular { hardness: number; intensity: number }

/** The key, fill and back light colours of Image (keyLightColor, fillLightColor, backLightColor), each 0..1. */
export interface ImageLights { key: number; fill: number; back: number }

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
  /** dump image shiny (specular intensity; hardness 16 * shiny). Default 1. */
  shiny?: number;
  /** Ambient colour of the surfaces: 0 without ssao, 0.5 with it (Image::view_params), or the dump_modify lights ambient. Default 0. */
  ambient?: number;
  /** The key, fill and back light colours (0..1) of dump_modify lights; the image defaults 0.9, 0.45, 0.9 when absent. */
  lights?: ImageLights;
  /**
   * Native's depth buffer (Image::depthBuffer, zdist - camera offset, -1 for an empty pixel) for the pixels the GPL
   * routines wrote; a pixel counts only while depth still holds zdist - dist (the write of draw.ts drawPixel). Set by
   * dump image's render; the composite passes read native depths from it so that no round trip changes a bit.
   */
  dist?: Float64Array;
  /** dump_modify specular: the highlight hardness and intensity (undefined: 16 * shiny and shiny, the defaults). */
  spec?: ImageSpecular;
  /** The ssao tangent buffer (2 values per pixel, rows from the top), set when ssao is on; null otherwise. */
  surface?: Float64Array;
}

/**
 * The post-processing settings of dump image (the Image fields of outline, depthcue and defocus; dump_image.html).
 * Values are the parsed keyword values; colours are 0..255.
 */
export interface PostProcessOptions {
  /** maxdel of Image::view_params (2 * longest box edge): the outline threshold is 0.02 * maxdel. */
  maxdel: number;
  /** Image::fsaa: the outline width is doubled with the doubled internal image. */
  fsaa: boolean;
  outline?: { width: number; color: Rgb255 };
  depthcue?: {
    strength: number;
    /** depthcue color: null for "auto" (the background, gradient rows included). */
    color: Rgb255 | null;
    /** depthcue start fraction between the near and far side of the box; null for "auto". */
    start: number | null;
    background: Rgb255;
    /** dump image backcolor2 gradient (null: a single background colour). */
    background2: Rgb255 | null;
  };
  defocus?: { strength: number; start: number | null };
  /** The simulation box bounds (Image::boxbounds): lo and hi corners. */
  boxLo: Vec3;
  boxHi: Vec3;
}

/**
 * The shading helpers of the lazily loaded GPL graphics module (engine-gpl/graphics/image_shading.ts), read by the core
 * dump image through gplHook('image'): the draw_pixel colour of a surface and the ssao occlusion pass.
 */
export interface ImageShading {
  /** draw_pixel's colour (0..1 per channel) of a camera-space unit normal and a surface colour (0..1). */
  shadeSurface(surface: Vec3, col: Vec3, shiny: number, ambient: number, lights?: ImageLights, spec?: ImageSpecular): Vec3;
  /** compute_SSAO over a finished frame (radius in world units, samples, seed). */
  applySsao(f: RasterFrame, radius: number, samples: number, seed: number): void;
  /** Image::draw_sphere of an atom: world centre x, colour 0..1, world diameter, opacity (draw.ts). */
  drawSphere(f: RasterFrame, x: Vec3, col: Vec3, diameter: number, opacity: number): void;
  /** Image::draw_cylinder between world points x1 and x2: full diameter, sflag caps (bit 0 at x1, bit 1 at x2), colour 0..1. */
  drawCylinder(f: RasterFrame, x1: Vec3, x2: Vec3, col: Vec3, diameter: number, sflag: number, opacity: number): void;
  /**
   * Image::draw_axes (axes.ts): the four axes points (origin and the ends of x, y and z; the z end is unused in 2d), the
   * diameter, opacity, the background colour (bytes, for the letters' colours) and whether fsaa doubles the frame.
   */
  drawAxes(f: RasterFrame, axes: Vec3[], diameter: number, opacity: number, background: Rgb255, fsaa: boolean, dimension: 2 | 3): void;
  /** Image::color2rgb (native rgbcolors, 0..1 per channel): null for "none" and an unknown name. */
  color2rgb(name: string): Vec3 | null;
  /** dump_image.cpp default_colors: the colour of atom type `type` (type 1 is red), 0..1. */
  defaultTypeColor(type: number): Vec3;
  /** Image::element2color (the element table, 0..1) and Image::element2diam (the covalent diameter, 0 when unknown). */
  element2color(name: string): Vec3 | null;
  element2diam(name: string): number;
  /** Image::merge's composite passes after SSAO: outline, depthcue, defocus (in place, rows from the top). */
  postProcess(f: RasterFrame, o: PostProcessOptions): void;
}

/** What dump image's "fix ID style flag1 flag2" keyword gives a graphics object (dump_image.html). */
export interface ObjectStyle {
  /** flag1 and flag2 of the keyword (added to lengths and diameters, see dump_image.html). */
  flag1: number;
  flag2: number;
  /** Transparency of the fix's objects (dump_modify ftrans; 1 = opaque, the default). */
  opacity: number;
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

/** An atom_style line particle for dump image's "line" keyword: centre, length, theta (LINE_BONUS) and colour (0..255). */
export interface LineAtom { x: number; y: number; length: number; theta: number; color: Rgb255; opacity: number }

/**
 * An atom_style tri particle for dump image's "tri" keyword: centroid, orientation quaternion (w, i, j, k), the nine
 * body-frame corner values (lineTri.ts triangleCorner) and colour (0..255).
 */
export interface TriAtom { center: Vec3; quat: readonly number[]; corners: readonly number[]; color: Rgb255; opacity: number }

/**
 * The lazily loaded line and triangle loops of dump image (src/engine-gpl/graphics/atom_objects.ts), read through
 * gplHook('atoms'). The core passes the atoms it draws; the GPL side draws them in native order.
 */
export interface AtomObjectsHook {
  /** atom_style line segments: each a cylinder of `diameter` with both caps, at the atom's opacity (dump_modify atrans). */
  drawLineAtoms(f: RasterFrame, lines: readonly LineAtom[], diameter: number): void;
  /** atom_style tri triangles: tstyle 1 faces, 2 edges, other values both; edges of `diameter`; each at its atom's opacity. */
  drawTriAtoms(f: RasterFrame, tris: readonly TriAtom[], tstyle: number, diameter: number): void;
  /** The screen-door mask of Image's draw loops: true when a pixel (column ix, native row iy) is dropped at this opacity. */
  transparentPixel(opacity: number, ix: number, iy: number): boolean;
  /** atom_style ellipsoid particles: rounded triangles (estyle 1) or a wireframe (2) of the given refinement level. */
  drawEllipsoidAtoms(f: RasterFrame, atoms: readonly EllipsoidAtom[], estyle: number, level: number, diameter: number): void;
  /** atom_style body particles (rounded/polyhedron): the elements of body_image.ts with the flags of dump image's "body" keyword. */
  drawBodyAtoms(f: RasterFrame, atoms: readonly BodyAtom[], flag1: number, flag2: number): void;
}

/**
 * An atom_style ellipsoid particle for dump image's "ellipsoid" keyword: centre, half-axes, orientation quaternion (w, i,
 * j, k), the superellipsoid block exponents (null for a plain ellipsoid), the colour (0..255) and the transparency.
 */
export interface EllipsoidAtom { center: Vec3; shape: Vec3; quat: readonly number[]; block: [number, number] | null; color: Rgb255; opacity: number }

/**
 * The drawing style of a dump image "region" keyword (dump_image.html: "region-ID color drawstyle"): the colour (0..255),
 * the style (filled, frame, transparent or points), the frame diameter (frame, and the point diameter of points) and the
 * opacity (transparent).
 */
export interface RegionDrawSpec {
  color: Rgb255;
  drawstyle: 'filled' | 'frame' | 'transparent' | 'points';
  framediam: number;
  opacity: number;
}

/**
 * A cylinder or cone region for dump image's "region" keyword: the axis (0 x, 1 y, 2 z), the two coordinates other than
 * the axis (c1 and c2 in the region command's order), the radii at the lower and upper end, the extent along the axis,
 * the colour (0..255) and the drawing style with its frame diameter or opacity (dump_image.html).
 */
export interface RegionConeSpec {
  style: 'cylinder' | 'cone';
  axis: 0 | 1 | 2;
  c1: number;
  c2: number;
  radlo: number;
  radhi: number;
  lo: number;
  hi: number;
  color: Rgb255;
  drawstyle: 'filled' | 'frame' | 'transparent';
  framediam: number;
  opacity: number;
}

/** A block or prism region (the box corners of dump_image.cpp): b = xlo, xhi, ylo, yhi, zlo, zhi and, for a prism, xy, xz, yz. */
export interface BoxRegionSpec extends RegionDrawSpec {
  kind: 'block' | 'prism';
  b: number[];
  /** The region's "open" face numbers (1 xlo, 2 xhi, 3 ylo, 4 yhi, 5 zlo, 6 zhi as the region command gives them). */
  openFaces: number[];
}

/** A sphere region: centre and radius (static regions only). */
export interface SphereRegionSpec extends RegionDrawSpec { center: Vec3; radius: number }

/** An ellipsoid region: centre and half-axes (static regions only). */
export interface EllipsoidRegionSpec extends RegionDrawSpec { center: Vec3; shape: Vec3 }

/** A plane region: a point of the plane, its unit normal and the box diagonal (PlaneObj's scale). */
export interface PlaneRegionSpec extends RegionDrawSpec { point: Vec3; normal: Vec3; scale: number }

/** The points style: npoints tries in the box; framediam is the sphere diameter of each point. */
export interface PointsRegionSpec extends RegionDrawSpec { npoints: number }

/**
 * The lazily loaded region drawing of dump image (src/engine-gpl/graphics/region_box.ts, region_cone.ts), read through
 * gplHook('region'). Each draws its region clipped to the box [boxlo, boxhi] as the native branch of dump_image.cpp does.
 */
export interface RegionHook {
  /** Cylinder and cone regions (lo and hi limited to the box along the axis). */
  drawConeRegion(f: RasterFrame, spec: RegionConeSpec, boxlo: Vec3, boxhi: Vec3): void;
  drawBoxRegion(f: RasterFrame, spec: BoxRegionSpec, boxlo: Vec3, boxhi: Vec3): void;
  drawSphereRegion(f: RasterFrame, spec: SphereRegionSpec): void;
  drawEllipsoidRegion(f: RasterFrame, spec: EllipsoidRegionSpec): void;
  drawPlaneRegion(f: RasterFrame, spec: PlaneRegionSpec, boxlo: Vec3, boxhi: Vec3): void;
  /** The points style for any region: `match` tells whether a position lies in the region. */
  drawPointsRegion(f: RasterFrame, spec: PointsRegionSpec, boxlo: Vec3, boxhi: Vec3, match: (x: number, y: number, z: number) => boolean): void;
}

/**
 * An atom_style body particle of the rounded/polyhedron style for dump image's "body" keyword (the record of
 * src/engine/bodies.ts): centre, orientation quaternion, the number of vertices, edges and faces, the body-frame vertex
 * displacements (3 per vertex), the edge end vertices (2 per edge) and the face vertices (4 per face, the fourth negative
 * for a triangle), the rounded radius and the colour (0..255).
 */
export interface BodyAtom {
  /** The body style of the record: rounded/polyhedron (faces and edges), rounded/polygon (planar vertices) or nparticle (spheres). */
  style: 'rounded/polyhedron' | 'rounded/polygon' | 'nparticle';
  center: Vec3;
  quat: readonly number[];
  nsub: number;
  nedge: number;
  nface: number;
  vertices: readonly number[];
  edges: readonly number[];
  faces: readonly number[];
  /** The rounded radius (rounded/polyhedron), or the diameter stored by rounded/polygon (its diameter is 2 * radius when flag1 is 0). */
  radius: number;
  color: Rgb255;
  opacity: number;
}
