import type { System } from '../system';
import type { SimState } from '../types';
import { StyleError } from '../force/types';
import { buildAtomMap, buildSpecial, isEllipsoid, isLineAtom, isTriAtom, massOf, LINE_BONUS, TRI_BONUS, TRI_BONUS_COLS } from '../atoms';
import { termRows } from '../compute/local';
import { SUPER_BLOCK } from '../superellipsoid';
import { BODY_RECORD, bodyRecordOffset, parseBodyStyle, POLY_OFF } from '../bodies';
import { encodePng } from './png';
import { encodeJpeg } from './jpeg';
import { ColorMap } from './image_colormap';
import { type AtomObjectsHook, type BodyAtom, type EllipsoidAtom, type ImageObject, type ImageShading, type LineAtom, type ObjectStyle, type RasterFrame, type RegionConeSpec, type RegionHook, type Rgb255, type TriAtom, type Vec3 } from './image_objects';
import { gplHook } from '../gplHooks';
import { BlockRegion, ConeRegion, EllipsoidRegion, PlaneRegion, PrismRegion, SphereRegion } from '../region';

/*
 * dump image — docs.lammps.org/dump_image.html.
 *
 * "dump ID group-ID style N file color diameter keyword value ..." with
 * "style = *image* or *movie*"; the engine implements *image* only ("To convert images
 * into movies, LAMMPS has to be compiled with the -DLAMMPS_FFMPEG
 * flag.";
 * a browser build has no video encoder, so *dump movie* stays refused).
 *
 * "The filename suffix determines whether a JPEG, PNG, or PPM file is
 * created with the *image* dump style.  If the suffix is ".jpg" or ".jpeg",
 * then a JPEG format file is created, if the suffix is ".png", then a PNG
 * format is created, else a PPM (aka NETPBM) format file is created."
 * "Dump *image* filenames must contain a wildcard character "*" so that one
 * image file per snapshot is written.  The "*" character is replaced with the
 * timestep value."
 *
 * The browser engine's file store holds text, so each snapshot is stored under
 * its file name as a `data:` URL (base64) for the same bytes the image would
 * have on disk.
 *
 * The *view*, *center*, *up* and *zoom* keywords map 3d simulation space to
 * the image plane (orthographic): "The *view* keyword determines the viewpoint
 * from which the simulation box is viewed, looking towards the *center*
 * point." "*theta* value is the vertical angle from the +z axis"; "*phi* ...
 * azimuthal angle around the z axis ... A value of 0.0 is a view along the +x
 * axis". "up_internal = view cross (up cross view)"; "The *zoom* keyword
 * scales the size of the simulation box as it appears in the image."
 *
 * Camera scale and the projection centre were measured with native LAMMPS
 * (black box): a box of longest edge L is scaled so that L spans h/2 pixels
 * (scale = h / (2 L) * zoom) and the projected box centre lands at pixel
 * column floor(w/2); a +y offset moves the atom towards larger columns, a
 * +z offset towards smaller rows. The sphere radius is
 * 0.5 * diameter * scale.
 */

/** An 8-bit colour as the flat raster routines write it. */
type RGB = [number, number, number];
/**
 * A colour as native keeps it: red, green and blue in 0..1 (Image::rgbcolors, image.cpp at b332f28). Shaded objects
 * (atoms) keep it until the pixel write (int(c * 255), Image::draw_pixel); the flat routines take rgbOf(c).
 */
type Col = [number, number, number];

/** static_cast<int>(c * 255.0) for each channel (native backcolor, depthcue and outline colours). */
const rgbOf = (c: Col): RGB => [Math.trunc(c[0] * 255), Math.trunc(c[1] * 255), Math.trunc(c[2] * 255)];

/** The default box colour, 0..1 per channel as native keeps it: color2rgb("gold") (image.cpp, Image::Image). */
const boxcolorDefault = (): Col => gplHook<ImageShading>('image').color2rgb('gold')!;

/** What the object drawing needs from one snapshot's camera. */
interface DrawCtx {
  rgb: Uint8Array;
  depth: Float64Array;
  project: (p: number[]) => [number, number, number];
  scale: number;
  /** The same camera as the frame the GPL graphics objects draw into (image_objects.ts RasterFrame). */
  frame: RasterFrame;
  /** The shading helpers of the lazily loaded GPL graphics module (gplHook('image')). */
  shading: ImageShading;
}

/** dump image ssao ambient colour (Image::view_params: ambientColor 0.5 with ssao; 0 otherwise). */
const SSAO_AMBIENT = 0.5;


/**
 * dump_image.html Default: "acolor = \* red/green/blue/yellow/aqua/cyan". Measured with native
 * LAMMPS (black box): types 1..6 render red, green, blue, yellow, aqua, magenta, so the sixth
 * entry is magenta (255,0,255) here; the mapping "repeats itself for types > 6".
 */

/** Atom attributes dump image can colour/size by (dump.html attribute names + c_/f_/v_); type and element are special. */
const SUPPORTED_ATTRS = new Set([
  'type', 'element', 'id', 'mass', 'x', 'y', 'z', 'xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz',
  'q', 'mol', 'radius', 'diameter',
]);


interface Attr {
  color: string;
  diameter: string;
}


/** The pieces of Dump.write() the image renderer reads. */
export interface ImageOwner {
  readonly id: string;
  element: string[];
}

const DEG = Math.PI / 180;

/** dump image axes locations; null is "no", "yes" is the lower left corner. */
type AxesLocation = 'center' | 'lowerleft' | 'lowerright' | 'upperleft' | 'upperright';
const AXES_LOCATIONS = new Map<string, AxesLocation | null>([
  ['no', null], ['center', 'center'], ['lowerleft', 'lowerleft'], ['yes', 'lowerleft'], ['lowerright', 'lowerright'], ['upperleft', 'upperleft'], ['upperright', 'upperright'],
]);
/**
 * The origin's offset from the lower corner of the box, in units of the offset (dump_image.cpp, the axes block of
 * DumpImage::create_image: 1 = +offset, -1 = -offset, 8 = 8 * offset), per x, y, z. The 2d and 3d tables are the
 * native ones, which differ for the right and upper corners.
 */
const AXES_SHIFT: Record<2 | 3, Record<AxesLocation, [number, number, number]>> = {
  3: { center: [1, 1, 1], lowerleft: [-1, -1, -1], lowerright: [-1, 8, -1], upperleft: [-1, -1, 8], upperright: [-1, 8, 8] },
  2: { center: [1, 1, 1], lowerleft: [-1, -1, -1], lowerright: [8, -1, -1], upperleft: [-1, 8, -1], upperright: [8, 8, -1] },
};
const cross = (a: number[], b: number[]): number[] => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a: number[]): number[] => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
/** Image::view_params' EPSILON (image.cpp at b332f28: constexpr double EPSILON = 1.0e-6). */
const VIEW_EPSILON = 1.0e-6;
/**
 * The bounds Image::view_params is given (dump_image.cpp box_bounds, domain->boxlo_bound and boxhi_bound for a triclinic
 * box, domain.cpp set_global_box): the lower bound of x and of y is the smaller of the box bound and its tilted copy,
 * updated in turn for xz, as native does; the upper bounds are the larger. An orthogonal box is its own bounds.
 */
const viewBounds = (b: SimState['box']): { lo: number[]; hi: number[] } => {
  const [xy, xz, yz] = b.tilt;
  let lo0 = Math.min(b.lo[0], b.lo[0] + xy);
  lo0 = Math.min(lo0, lo0 + xz);
  const lo1 = Math.min(b.lo[1], b.lo[1] + yz);
  let hi0 = Math.max(b.hi[0], b.hi[0] + xy);
  hi0 = Math.max(hi0, hi0 + xz);
  const hi1 = Math.max(b.hi[1], b.hi[1] + yz);
  return { lo: [lo0, lo1, b.lo[2]], hi: [hi0, hi1, b.hi[2]] };
};

/**
 * base64 of the bytes as one flat string. Built by repeated `+=`, the result is a rope of about 30-byte nodes, one per
 * character, and a frame kept in the session's file store (dump image writes one per snapshot) then retains about 30
 * times its text (measured: a 1.44 MB base64 frame kept 44 MB of heap). The characters go into a byte array first and
 * are decoded in chunks, so the string is flat; its content is unchanged.
 */
export const toBase64 = (bytes: Uint8Array): string => {
  const T = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const chars = new Uint8Array(Math.ceil(bytes.length / 3) * 4);
  let o = 0;
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i], b1 = bytes[i + 1], b2 = bytes[i + 2];
    chars[o++] = T.charCodeAt(b0 >> 2);
    chars[o++] = T.charCodeAt(((b0 & 3) << 4) | (b1 === undefined ? 0 : b1 >> 4));
    chars[o++] = b1 === undefined ? 0x3d : T.charCodeAt(((b1 & 15) << 2) | (b2 === undefined ? 0 : b2 >> 6));
    chars[o++] = b2 === undefined ? 0x3d : T.charCodeAt(b2 & 63);
  }
  const parts: string[] = [];
  for (let i = 0; i < chars.length; i += 8192) parts.push(String.fromCharCode(...chars.subarray(i, i + 8192)));
  return parts.join('');
};

/** Build a PPM (P6 binary, 8-bit RGB) file. */
const encodePpm = (w: number, h: number, rgb: Uint8Array): Uint8Array => {
  const header = `P6\n${w} ${h}\n255\n`;
  const out = new Uint8Array(header.length + rgb.length);
  for (let i = 0; i < header.length; i++) out[i] = header.charCodeAt(i);
  out.set(rgb, header.length);
  return out;
};

export class ImageDump {
  private attr: Attr;
  private w = 512;
  private h = 512;
  private theta: number;
  private phi: number;
  private up: number[];
  private centerFrac: number[] = [0.5, 0.5, 0.5];
  private zoom = 1;
  private atomOn = true;
  private boxOn = true;
  private boxDiam = 0.02;
  private adiamAll: number | null = null;
  private acolorByType = new Map<number, Col>();
  private adiamByType = new Map<number, number>();
  /** dump_modify atrans: the transparency (0..1, 1 opaque) of the atoms of each type, native aopacity (default 1). */
  private aopacityByType = new Map<number, number>();
  /** dump_modify lights: ambient, key, fill and back light colours (0..1); null keeps the Image defaults. */
  private lightSet: { ambient: number; key: number; fill: number; back: number } | null = null;
  private bcolorByType = new Map<number, Col>();
  private bdiamByType = new Map<number, number>();
  private backcolor: RGB = [0, 0, 0];
  /** dump_modify backcolor2: the top of a background gradient (null: a single background colour). */
  private backcolor2: RGB | null = null;
  /** The box colour default: color2rgb("gold") (image.cpp, Image::Image; gold is 1.0, 0.843, 0.0), see boxcolorDefault(). */
  private boxcolor: Col = boxcolorDefault();
  /** dump_modify color definitions (name -> RGB), local to this dump. */
  private userColors = new Map<string, Col>();
  /**
   * Colour names as native's Image::rgbcolors holds them: a name's entry is one array, the colours that use it point at
   * it (color2rgb returns the stored double*), and dump_modify color redefines it in place (Image::addcolor), so every
   * earlier use of the name, in acolor, bcolor, amap, bmap and the like, takes the new value. Measured with native
   * LAMMPS: a name set to 0x0d0887, used by acolor, then redefined as 0.050 0.030 0.528, renders as the redefined colour.
   */
  private bond: { color: string; width: string } | null;
  /** dump image bond color = c_ID or c_ID[I]: the per-bond /local compute and its column (0 = its vector). */
  private bondCompute: { id: string; column: number } | null = null;
  /** dump_modify bmap: the colour map of the per-bond compute values (null = the dump_image.html default). */
  private bmap: ColorMap | null = null;
  private readonly defaultBondMap = ColorMap.default();
  /** The bond colour map before the current frame (fix graphics/labels colorscale map bond reads it). */
  private priorBondMap: ColorMap | null = null;
  /** dump image autobond (cutoff, width): the bonds are found from the atom distances; the bond keyword replaces them. */
  private autobond: { cutoff: number; width: number } | null = null;
  /** dump_modify amap (colour map of an atom colour value); null = the dump_image.html default. */
  private amap: ColorMap | null = null;
  private readonly defaultMap = ColorMap.default();
  /** dump image shiny (specular intensity; the highlight hardness is 16 * shiny), fsaa, and ssao (seed, intensity). */
  private shiny = 1;
  private fsaa = false;
  private ssao = false;
  private ssaoSeed = 1;
  private ssaoInt = 0;
  /** dump_modify ssaosamples (0: the number derived from the ssao intensity) and specular (see parse()). */
  private ssaoSamples = 0;
  private specularSet = false;
  private specularNone = false;
  private specularHardness = 16;
  /** dump image "outline" keyword: the width (1 to 16 pixels) and the colour; null when off. */
  private outlineOn: { width: number; color: RGB } | null = null;
  /** dump image "depthcue" keyword: the strength, the fog colour (null: auto) and the start (null: auto); null when off. */
  private depthcueOn: { strength: number; color: RGB | null; start: number | null } | null = null;
  /** dump image "defocus" keyword: the blur strength and the start (null: auto); null when off. */
  private defocusOn: { strength: number; start: number | null } | null = null;
  /** dump image "axes" keyword: the location, the length and the diameter fraction (null: no axes). */
  private axes: { loc: AxesLocation; len: number; diam: number } | null = null;
  /** dump image "fix ID style flag1 flag2" keywords: the fixes whose graphics objects are drawn. */
  private fixObjects: { id: string; source: 'fix' | 'compute'; style: 'type' | 'element' | 'const'; flag1: number; flag2: number; rgb: RGB; opacity: number }[] = [];
  /** dump image "line" keyword: colour style (type, atom or index) and the cylinder diameter (null: keyword off). */
  private lineOn: { color: 'type' | 'atom' | 'index'; diam: number } | null = null;
  /** dump image "tri" keyword: colour style, tstyle (1 faces, 2 edges, other both) and the edge diameter (null: off). */
  private triOn: { color: 'type' | 'atom' | 'index'; tstyle: number; diam: number } | null = null;
  /** dump image "ellipsoid" keyword: colour style, estyle (1 rounded triangles, 2 wireframe), refinement level, width. */
  private ellipsoidOn: { color: 'type' | 'atom' | 'index'; estyle: number; level: number; diam: number } | null = null;
  /** dump image "body" keyword: colour style and the flag1 (diameter) and flag2 (face style) values. */
  private bodyOn: { color: 'type' | 'atom' | 'index'; flag1: number; flag2: number } | null = null;
  /** While one fix object of ftrans opacity below 1 is drawn: keeps a pixel (column, row from the top) when true. */
  private gate: ((xx: number, yy: number) => boolean) | null = null;
  /** dump image "region" keywords: region ID, colour, drawing style, frame diameter and opacity (dump_image.html). */
  private regions: { id: string; color: RGB; drawstyle: 'filled' | 'frame' | 'transparent' | 'points'; framediam: number; opacity: number; npoints: number }[] = [];

  /**
   * multifileOverride: write_dump only. write_dump.html: "Normally, dump image files require a filename with a "*"
   * character for the timestep. That is not the case for the write_dump command; no wildcard "*" character is
   * necessary." (As native LAMMPS does, src/write_dump.cpp at b332f28: dump->multifile_override = 1 for image.)
   */
  constructor(private sys: System, readonly id: string, readonly group: string, file: string, args: string[], multifileOverride = false) {
    if (!multifileOverride && !file.includes('*')) throw new StyleError(`dump ${id}: dump image requires a file name with '*' requesting one snapshot per file`);
    const s = sys.state;
    const color = args[0];
    const diameter = args[1];
    if (!color || !diameter) throw new StyleError(`dump ${id}: dump image needs a color and a diameter attribute, e.g. type type`);
    for (const [label, a] of [['color', color], ['diameter', diameter]] as const) {
      if (!SUPPORTED_ATTRS.has(a) && !/^[cfv]_[A-Za-z0-9_]+$/.test(a)) {
        throw new StyleError(`dump ${id}: dump image ${label} attribute '${a}' is not supported by the browser engine`);
      }
    }
    this.attr = { color, diameter };
    // dump_image.html Default: "view = 60 30 (for 3d)", "view = 0 0 (for 2d)", "up = 0 0 1 (for 3d)", "up = 0 1 0 (for 2d)"
    this.theta = s.dimension === 2 ? 0 : 60;
    this.phi = s.dimension === 2 ? 0 : 30;
    this.up = s.dimension === 2 ? [0, 1, 0] : [0, 0, 1];
    // dump_image.html Default: "bond = none none (if no bonds in system)", "bond = atom 0.5 (if bonds in system)"
    this.bond = s.topo.bonds.n > 0 ? { color: 'atom', width: '0.5' } : null;
    this.parse(args.slice(2));
  }

  /**
   * A logical keyword value as utils::logical reads it (utils.cpp at b332f28): yes, on, true or 1 are on; no, off,
   * false or 0 are off; anything else is an error ("Expected boolean parameter instead of ...").
   */
  private logical(w: string | undefined): boolean {
    if (w === 'yes' || w === 'on' || w === 'true' || w === '1') return true;
    if (w === 'no' || w === 'off' || w === 'false' || w === '0') return false;
    throw new StyleError(`dump ${this.id}: expected boolean parameter instead of '${w ?? ''}' in dump image`);
  }

  /**
   * A colour as native's Image::color2rgb reads it: a dump_modify color definition first, then the table of
   * Image::rgbcolors (engine-gpl/graphics/colors.ts, through the 'image' hook). Null for "none" and an unknown name.
   */
  private namedColor(w: string): Col | null {
    const live = this.userColors.get(w);
    if (live) return live;
    const builtin = gplHook<ImageShading>('image').color2rgb(w);
    if (!builtin) return null;
    // the first use of a built-in name creates its entry, so a later dump_modify color redefines the same array
    const ref: Col = [builtin[0], builtin[1], builtin[2]];
    this.userColors.set(w, ref);
    return ref;
  }

  /** Image::addcolor: sets the RGB values of a colour name in place (an existing entry keeps its identity). */
  private setColor(name: string, c: readonly [number, number, number]): void {
    const ref = this.userColors.get(name);
    if (ref) {
      ref[0] = c[0];
      ref[1] = c[1];
      ref[2] = c[2];
    } else {
      this.userColors.set(name, [c[0], c[1], c[2]]);
    }
  }

  /** A colour name for the image keywords; null for "none" or an unknown name (the caller raises its error). */
  private colorOrNull(w: string | undefined): Col | null {
    if (w === undefined || w === 'none') return null;
    return this.namedColor(w);
  }

  private num(w: string, key: string): number {
    if (w.startsWith('v_')) return this.sys.equalVariable(w.slice(2));
    const n = Number(w);
    if (!Number.isFinite(n)) throw new StyleError(`dump ${this.id}: dump image ${key} needs a number`);
    return n;
  }

  private color(w: string): Col {
    const c = this.namedColor(w);
    if (!c) throw new StyleError(`dump ${this.id}: unknown colour '${w}'`);
    return c;
  }

  private parse(a: string[]): void {
    for (let k = 0; k < a.length;) {
      const key = a[k];
      switch (key) {
        case 'atom': {
          const v = a[k + 1];
          if (v !== 'yes' && v !== 'no') throw new StyleError(`dump ${this.id}: dump image atom must be yes or no`);
          this.atomOn = v === 'yes';
          k += 2;
          break;
        }
        case 'adiam': this.adiamAll = this.num(a[k + 1], 'adiam'); k += 2; break;
        case 'size': {
          const w = this.num(a[k + 1], 'size'), h = this.num(a[k + 2], 'size');
          if (!(w > 0) || !(h > 0)) throw new StyleError(`dump ${this.id}: dump image size needs positive width and height`);
          this.w = Math.trunc(w); this.h = Math.trunc(h);
          k += 3;
          break;
        }
        case 'view': this.theta = this.num(a[k + 1], 'view'); this.phi = this.num(a[k + 2], 'view'); k += 3; break;
        case 'up': this.up = [this.num(a[k + 1], 'up'), this.num(a[k + 2], 'up'), this.num(a[k + 3], 'up')]; k += 4; break;
        case 'center': {
          // center values = flag Cx Cy Cz ; flag is *s* or *d* (static/dynamic)
          const flag = a[k + 1];
          if (flag !== 's' && flag !== 'd') throw new StyleError(`dump ${this.id}: dump image center flag must be s or d`);
          this.centerFrac = [this.num(a[k + 2], 'center'), this.num(a[k + 3], 'center'), this.num(a[k + 4], 'center')];
          k += 5;
          break;
        }
        case 'zoom': {
          const z = this.num(a[k + 1], 'zoom');
          if (!(z > 0)) throw new StyleError(`dump ${this.id}: dump image zoom must be > 0`);
          this.zoom = z;
          k += 2;
          break;
        }
        case 'box': {
          const v = a[k + 1];
          if (v !== 'yes' && v !== 'no') throw new StyleError(`dump ${this.id}: dump image box must be yes or no`);
          this.boxOn = v === 'yes';
          this.boxDiam = this.num(a[k + 2], 'box');
          k += 3;
          break;
        }
        case 'bond': {
          const color = a[k + 1], width = a[k + 2];
          if (!color || !width) throw new StyleError(`dump ${this.id}: dump image bond needs a color and a width`);
          // dump_image.html: "color = atom or type or none or c_ID or c_ID[I]" (c_ID: per-bond vector of a local compute)
          const ref = /^c_([^[\]]+)(?:\[(\d+)\])?$/.exec(color);
          const computeRef = ref ? { id: ref[1], column: ref[2] === undefined ? 0 : Number(ref[2]) } : null;
          if (computeRef && computeRef.column < 0) throw new StyleError(`dump ${this.id}: invalid dump image bond compute reference ${color}`);
          if (!computeRef && color !== 'atom' && color !== 'type' && color !== 'none') throw new StyleError(`dump ${this.id}: dump image bond color '${color}' is not supported (atom, type, none or c_ID)`);
          if (width !== 'atom' && width !== 'type' && width !== 'none' && !Number.isFinite(Number(width))) throw new StyleError(`dump ${this.id}: dump image bond width '${width}' is not supported`);
          if (color === 'none' && width === 'none') this.bond = null;
          else this.bond = { color: computeRef ? 'compute' : color, width };
          this.bondCompute = computeRef;
          this.autobond = null;
          k += 3;
          break;
        }
        case 'autobond': {
          // dump_image.html: "autobond values = cutoff width = bond cutoff and width of bonds"; dump_image.cpp rejects a
          // cutoff or width that is not positive ("Illegal dump image autobond cutoff value", "... diameter value")
          if (k + 3 > a.length) throw new StyleError(`dump ${this.id}: dump image autobond needs a cutoff and a width`);
          const cutoff = this.num(a[k + 1], 'autobond'), width = this.num(a[k + 2], 'autobond');
          if (!(cutoff > 0)) throw new StyleError(`dump ${this.id}: illegal dump image autobond cutoff value ${cutoff}`);
          if (!(width > 0)) throw new StyleError(`dump ${this.id}: illegal dump image autobond diameter value ${width}`);
          this.autobond = { cutoff, width };
          this.bond = null;
          k += 3;
          break;
        }
        case 'fix': {
          // dump_image.html: "fix ID style flag1 flag2" with "style = type, element or const"; "flag1 and flag2 are
          // added to the object sizes" (the objects of fix graphics/lines, objects and arrows).
          const id = a[k + 1], style = a[k + 2];
          if (!id || !style || a.length < k + 5) throw new StyleError(`dump ${this.id}: dump image fix needs an ID, a style and two values`);
          if (!this.sys.fixes.some((f) => f.id === id)) throw new StyleError(`dump ${this.id}: dump image fix ID '${id}' does not exist`);
          if (style !== 'type' && style !== 'element' && style !== 'const') throw new StyleError(`dump ${this.id}: unsupported colour style '${style}' for dump image fix ${id}`);
          this.fixObjects.push({ id, source: 'fix', style, flag1: this.num(a[k + 3], 'fix'), flag2: this.num(a[k + 4], 'fix'), rgb: [255, 255, 255], opacity: 1 });
          k += 5;
          break;
        }
        // dump_image.html: "line = color width", color = type, atom or index (dump_image.cpp: "Dump image line only
        // supports color by type, atom, or index"); the width is numeric. Line particles are drawn as cylinders (lineflag).
        case 'line': {
          const c = a[k + 1];
          if (c !== 'type' && c !== 'atom' && c !== 'index') throw new StyleError(`dump ${this.id}: dump image line only supports color by type, atom, or index`);
          this.lineOn = { color: c, diam: this.num(a[k + 2], 'line') };
          k += 3;
          break;
        }
        // dump_image.html: "tri = color tflag width"; tflag 1 draws the triangles, 2 the edges, 3 both (dump_image.cpp:
        // tstyle is read as an integer; tridraw is off for 2 and edgedraw off for 1).
        case 'tri': {
          const c = a[k + 1];
          if (c !== 'type' && c !== 'atom' && c !== 'index') throw new StyleError(`dump ${this.id}: dump image tri only supports color by type, atom, or index`);
          const ts = this.num(a[k + 2], 'tri');
          if (!Number.isInteger(ts)) throw new StyleError(`dump ${this.id}: dump image tri tflag must be an integer`);
          this.triOn = { color: c, tstyle: ts, diam: this.num(a[k + 3], 'tri') };
          k += 4;
          break;
        }
        // dump_image.html: "region values = region-ID color drawstyle [opacity (optional) npoints (optional) diameter (optional)]
        // [hull_points npoints (optional)]"; drawstyle = filled, frame, points or transparent (dump_image.cpp: "Unknown region
        // draw style"). The frame diameter must be > 0 and the opacity lies in 0.0 to 1.0. Only the filled, frame and transparent
        // styles are drawn by drawRegions (points for any region; the box, sphere, ellipsoid, plane and cone shapes).
        case 'region': {
          const id = a[k + 1], color = a[k + 2], style = a[k + 3];
          if (!id || !color || !style) throw new StyleError(`dump ${this.id}: dump image region needs an ID, a colour and a draw style`);
          this.sys.region(id);
          const rgb = rgbOf(this.color(color));
          let drawstyle: 'filled' | 'frame' | 'transparent' | 'points' = 'filled', framediam = 0.5, opacity = 1, npoints = 0;
          let used = 4;
          if (style === 'filled') { /* no value */ }
          else if (style === 'points') {
            // dump_image.cpp: "npoints" (>= 1) then the point "diameter" (> 0.0), six words in all with the style
            npoints = this.num(a[k + 4], 'region');
            if (!Number.isInteger(npoints) || npoints < 1) throw new StyleError(`dump ${this.id}: dump image region number of points must be > 0`);
            framediam = this.num(a[k + 5], 'region');
            if (!(framediam > 0)) throw new StyleError(`dump ${this.id}: dump image region point diameter must be > 0.0`);
            drawstyle = 'points'; used = 6;
          }
          else if (style === 'frame') {
            framediam = this.num(a[k + 4], 'region');
            if (!(framediam > 0)) throw new StyleError(`dump ${this.id}: dump image region frame diameter must be > 0.0`);
            drawstyle = 'frame'; used = 5;
          } else if (style === 'transparent') {
            opacity = this.num(a[k + 4], 'region');
            if (opacity < 0 || opacity > 1) throw new StyleError(`dump ${this.id}: dump image region opacity must be in the range 0.0 to 1.0`);
            drawstyle = 'transparent'; used = 5;
          } else throw new StyleError(`dump ${this.id}: unknown dump image region draw style '${style}'`);
          if (a[k + used] === 'hull_points') throw new StyleError(`dump ${this.id}: dump image region hull_points is not supported by the browser engine`);
          this.regions.push({ id, color: rgb, drawstyle, framediam, opacity, npoints });
          k += used;
          break;
        }
        // dump_image.html: "ellipsoid = color eflag level width"; eflag 1 draws rounded triangles and 2 the edges, other values
        // select rounded triangles (dump_image.cpp: "if (estyle !=2) estyle = 1"); level 0 is the default 4 and the level must
        // lie in 0 to 6 ("invalid mesh refinement level"). Only the color styles type, atom and index are accepted.
        case 'ellipsoid': {
          const c = a[k + 1];
          if (c !== 'type' && c !== 'atom' && c !== 'index') throw new StyleError(`dump ${this.id}: dump image ellipsoid only supports color by type, atom, or index`);
          const eflag = this.num(a[k + 2], 'ellipsoid'), level0 = this.num(a[k + 3], 'ellipsoid');
          if (!Number.isInteger(eflag) || !Number.isInteger(level0)) throw new StyleError(`dump ${this.id}: dump image ellipsoid eflag and level must be integers`);
          const level = level0 === 0 ? 4 : level0;
          if (level < 0 || level > 6) throw new StyleError(`dump ${this.id}: dump image ellipsoid invalid mesh refinement level ${level}`);
          this.ellipsoidOn = { color: c, estyle: eflag === 2 ? 2 : 1, level, diam: this.num(a[k + 4], 'ellipsoid') };
          k += 5;
          break;
        }
        // dump_image.html: "compute = computeID color cflag1 cflag2" with color = type, element or const (dump_image.cpp:
        // "Unsupported color style for dump image compute"). The compute must exist and provide graphics objects.
        case 'compute': {
          const id = a[k + 1], style = a[k + 2];
          if (!id || !style || a.length < k + 5) throw new StyleError(`dump ${this.id}: dump image compute needs an ID, a style and two values`);
          this.sys.compute(id);
          if (style !== 'type' && style !== 'element' && style !== 'const') throw new StyleError(`dump ${this.id}: unsupported colour style '${style}' for dump image compute ${id}`);
          this.fixObjects.push({ id, source: 'compute', style, flag1: this.num(a[k + 3], 'compute'), flag2: this.num(a[k + 4], 'compute'), rgb: [255, 255, 255], opacity: 1 });
          k += 5;
          break;
        }
        // dump_image.html: "body = color bflag1 bflag2" with color = type, atom or index (dump_image.cpp: "Dump image body only
        // supports color by type, atom, or index"). The image() methods of the body styles rounded/polyhedron
        // (body_rounded_polyhedron.cpp), nparticle (body_nparticle.cpp) and rounded/polygon (body_rounded_polygon.cpp) are drawn.
        case 'body': {
          const c = a[k + 1];
          if (c !== 'type' && c !== 'atom' && c !== 'index') throw new StyleError(`dump ${this.id}: dump image body only supports color by type, atom, or index`);
          const bs = parseBodyStyle(this.sys.state.atomStyle);
          if (!bs) throw new StyleError(`dump ${this.id}: dump image body needs atom_style body`);
          this.bodyOn = { color: c, flag1: this.num(a[k + 2], 'body'), flag2: this.num(a[k + 3], 'body') };
          k += 4;
          break;
        }
        // The remaining keyword/value pairs are accepted by the grammar but need renderers the engine does not
        // have (or only change image quality). Named so the stop is never silent (dump_image.html keyword list).
        case 'axes': {
          // dump_image.html: "axes = no or location, length, diameter"; the location is center, lowerleft (yes),
          // lowerright, upperleft or upperright; native reads four words in every case (dump_image.cpp, iarg += 4).
          const loc = AXES_LOCATIONS.get(a[k + 1] ?? '');
          if (loc === undefined) throw new StyleError(`dump ${this.id}: unknown dump image axes location '${a[k + 1] ?? ''}'`);
          if (loc === null) this.axes = null;
          else {
            const len = this.num(a[k + 2], 'axes'), diam = this.num(a[k + 3], 'axes');
            if (!(len > 0)) throw new StyleError(`dump ${this.id}: invalid dump image axes length ${len}`);
            if (!(diam > 0)) throw new StyleError(`dump ${this.id}: invalid dump image axes diameter ${diam}`);
            this.axes = { loc, len, diam };
          }
          k += 4;
          break;
        }
        case 'shiny': {
          // dump_image.html: "shiny value = shiny value from 0.0 to 1.0"; native checks 0 <= shiny <= 1 (dump_image.cpp)
          const v = this.num(a[k + 1], 'shiny');
          if (v < 0 || v > 1) throw new StyleError(`dump ${this.id}: invalid dump image shiny value ${v}`);
          this.shiny = v;
          k += 2;
          break;
        }
        case 'fsaa': {
          // dump_image.html: "fsaa value = *yes* or *no*"; with yes the image is rendered at twice the size and averaged
          const v = a[k + 1];
          if (v !== 'yes' && v !== 'no') throw new StyleError(`dump ${this.id}: dump image fsaa must be yes or no`);
          this.fsaa = v === 'yes';
          k += 2;
          break;
        }
        case 'ssao': {
          // dump_image.html: "ssao value = *yes* or *no* seed intensity"; seed is a positive integer, intensity 0.0 to 1.0
          const v = a[k + 1];
          if (v !== 'yes' && v !== 'no') throw new StyleError(`dump ${this.id}: dump image ssao must be yes or no`);
          const seed = this.num(a[k + 2], 'ssao'), inten = this.num(a[k + 3], 'ssao');
          if (!(seed > 0) || !Number.isInteger(seed)) throw new StyleError(`dump ${this.id}: invalid dump image ssao seed ${seed}`);
          if (inten < 0 || inten > 1) throw new StyleError(`dump ${this.id}: invalid dump image ssao intensity value ${inten}`);
          this.ssao = v === 'yes';
          this.ssaoSeed = seed;
          this.ssaoInt = inten;
          k += 4;
          break;
        }
        // dump_image.html: "depthcue values = cueing cfactor color start"; the four words are read in every case
        // (dump_image.cpp: iarg+5 > narg, then iarg += 5). cueing is a logical (see logical()).
        case 'depthcue': {
          if (k + 5 > a.length) throw new StyleError(`dump ${this.id}: dump image depthcue needs cueing, cfactor, color and start`);
          const on = this.logical(a[k + 1]);
          const strength = this.num(a[k + 2], 'depthcue');
          // dump_image.html: "cfactor = strength of fading from 0.0 to 1.0" (dump_image.cpp: cfactor < 0.0 || cfactor > 1.0)
          if (strength < 0 || strength > 1) throw new StyleError(`dump ${this.id}: invalid dump image depthcue strength value ${strength}`);
          // dump_image.html: "color = fog color name or *auto* = fade toward the background color"
          let color: Col | null = null;
          if (a[k + 3] !== 'auto') {
            color = this.colorOrNull(a[k + 3]);
            if (!color) throw new StyleError(`dump ${this.id}: invalid dump image depthcue color ${a[k + 3]}`);
          }
          // dump_image.html: "start = box fraction along the view direction where fading starts, or *auto*"; "With *auto*
          // it begins at the nearest rendered object, so the front of the scene never fades." and "The fading always ends
          // at the most distant rendered object."
          const start = a[k + 4] === 'auto' ? null : this.num(a[k + 4], 'depthcue');
          this.depthcueOn = on ? { strength, color: color ? rgbOf(color) : null, start } : null;
          k += 5;
          break;
        }
        // dump_image.html: "defocus values = blurring bfactor start" (dump_image.cpp: iarg+4 > narg, iarg += 4); "Objects in
        // front of the start position are not blurred, and the blur always reaches its maximum at the most distant rendered
        // object."

        case 'defocus': {
          if (k + 4 > a.length) throw new StyleError(`dump ${this.id}: dump image defocus needs blurring, bfactor and start`);
          const on = this.logical(a[k + 1]);
          const strength = this.num(a[k + 2], 'defocus');
          // dump_image.html: "bfactor = strength of the blur from 0.0 to 1.0" (dump_image.cpp: bfactor < 0.0 || bfactor > 1.0)
          if (strength < 0 || strength > 1) throw new StyleError(`dump ${this.id}: invalid dump image defocus strength value ${strength}`);
          const start = a[k + 3] === 'auto' ? null : this.num(a[k + 3], 'defocus');
          this.defocusOn = on ? { strength, start } : null;
          k += 4;
          break;
        }
        // dump_image.html: "outline values = flag width color = outlines at depth jumps" and "width = width of the outlines in
        // pixels (from 1 to 16)"; width is an integer from 1 to
        // 16 (dump_image.cpp: utils::inumeric, owidth < 1 || owidth > 16), color a colour name (not none)
        case 'outline': {
          if (k + 4 > a.length) throw new StyleError(`dump ${this.id}: dump image outline needs flag, width and color`);
          const on = this.logical(a[k + 1]);
          const wText = a[k + 2];
          const width = /^[+-]?\d+$/.test(wText) ? Number(wText) : NaN;
          if (!(width >= 1 && width <= 16)) throw new StyleError(`dump ${this.id}: invalid dump image outline width ${wText}`);
          const color = this.colorOrNull(a[k + 3]);
          if (!color) throw new StyleError(`dump ${this.id}: invalid dump image outline color ${a[k + 3]}`);
          this.outlineOn = on ? { width, color: rgbOf(color) } : null;
          k += 4;
          break;
        }
        // ellipsoid draws ellipsoids (and superellipsoids, atom_style ellipsoid superellipsoid): no renderer here
        case 'grid':
        case 'subbox': case 'persp':
          throw new StyleError(`dump ${this.id}: dump image keyword '${key}' is not supported by the browser engine`);
        default:
          throw new StyleError(`dump ${this.id}: unknown dump image keyword '${key}'`);
      }
    }
  }

  /**
   * Image-specific dump_modify keywords (dump_image.html "dump_modify options for dump image/movie").
   * Returns the number of tokens consumed (key included), or 0 when the keyword is not image-specific
   * and the caller's general dump_modify switch should handle it.
   */
  modifyKeyword(a: string[], k: number): number {
    const key = a[k];
    const typeRange = (w: string): number[] => {
      const m = /^(?:(\d+)?\*(\d+)?|\*|(\d+))$/.exec(w);
      if (!m) throw new StyleError(`dump ${this.id}: dump_modify ${key} needs an atom type or a type range`);
      const n = this.sys.state.ntypes;
      if (m[3] !== undefined) return [Number(m[3])];
      const lo = m[1] ? Number(m[1]) : 1;
      const hi = m[2] ? Number(m[2]) : n;
      const out: number[] = [];
      for (let t = lo; t <= hi; t++) out.push(t);
      return out;
    };
    const colors = (w: string): Col[] => w.split('/').map((c) => this.color(c));
    switch (key) {
      case 'acolor': {
        const types = typeRange(a[k + 1]);
        const cs = colors(a[k + 2]);
        types.forEach((t, i) => this.acolorByType.set(t, cs[i % cs.length]));
        return 3;
      }
      case 'adiam': {
        for (const t of typeRange(a[k + 1])) this.adiamByType.set(t, this.num(a[k + 2], 'adiam'));
        return 3;
      }
      // dump_modify atrans: "atrans itypes value" with the value 0.0 to 1.0 (dump_modify.cpp: "Illegal transparency ... in
      // dump_modify atrans command"); the screen-door mask of the draw loops is the one of ftrans.
      case 'atrans': {
        const types = typeRange(a[k + 1]);
        const v = this.num(a[k + 2], 'atrans');
        if (v < 0 || v > 1) throw new StyleError(`dump ${this.id}: illegal transparency ${a[k + 2]} in dump_modify atrans`);
        for (const t of types) this.aopacityByType.set(t, v);
        return 3;
      }
      // dump_modify lights: "lights ambient key fill back" with each value in 0.0 to 1.0 (dump_image.cpp: "Illegal ambient
      // light value", "Illegal key light value", ...). The ambient colour replaces the 0.5 that ssao sets at construction.
      case 'lights': {
        const v = [1, 2, 3, 4].map((o) => this.num(a[k + o], 'lights'));
        const names = ['ambient', 'key', 'fill', 'back'];
        v.forEach((x, i) => { if (x < 0 || x > 1) throw new StyleError(`dump ${this.id}: illegal ${names[i]} light value ${a[k + i + 1]} in dump_modify lights`); });
        this.lightSet = { ambient: v[0], key: v[1], fill: v[2], back: v[3] };
        return 5;
      }
      case 'bcolor': {
        const types = typeRange(a[k + 1]);
        const cs = colors(a[k + 2]);
        types.forEach((t, i) => this.bcolorByType.set(t, cs[i % cs.length]));
        return 3;
      }
      case 'bdiam': {
        for (const t of typeRange(a[k + 1])) this.bdiamByType.set(t, this.num(a[k + 2], 'bdiam'));
        return 3;
      }
      // dump_image.html (dump_modify options): "*ssaosamples* arg = nsamples" and "nsamples = number of SSAO sampling
      // directions per pixel (from 4 to 64)"; dump_modify.cpp checks 4 <= nsamples <= 64 (image.cpp ssaosamples)
      case 'ssaosamples': {
        const v = a[k + 1] === undefined ? NaN : Number(a[k + 1]);
        if (!Number.isInteger(v) || v < 4 || v > 64) throw new StyleError(`dump ${this.id}: illegal ssaosamples value ${a[k + 1] ?? ''} in dump_modify`);
        this.ssaoSamples = v;
        return 2;
      }
      // dump_image.html (dump_modify options): "*specular* arg = style" and "style = *none* or *wide* or *narrow* or *tight*
      // = specular highlights off or their width"; "The *specular* keyword adjusts the specular highlights independently from
      // the *shiny* keyword of the dump image command." Hardness wide 10, narrow 50, tight 250 (dump_modify.cpp, image.cpp)
      case 'specular': {
        const st = a[k + 1];
        if (st === 'none') { this.specularNone = true; this.specularSet = true; return 2; }
        const h = st === 'wide' ? 10 : st === 'narrow' ? 50 : st === 'tight' ? 250 : null;
        if (h === null) throw new StyleError(`dump ${this.id}: unknown specular setting ${st ?? ''}`);
        this.specularHardness = h;
        this.specularNone = false;
        this.specularSet = true;
        return 2;
      }
      case 'backcolor': this.backcolor = rgbOf(this.color(a[k + 1])); return 2;
      case 'backcolor2': this.backcolor2 = a[k + 1] === 'none' ? null : rgbOf(this.color(a[k + 1])); return 2;
      case 'boxcolor': this.boxcolor = this.color(a[k + 1]); return 2;
      case 'color': {
        const name = a[k + 1];
        // dump_modify.html: "color = 0xRRGGBB" is also accepted (dump_image.cpp: the regular expression ^0x[0-9a-fA-F]+$
        // and the 8-bit channels divided by 255); a hex colour takes 3 words, a numeric one 5
        if (/^0x[0-9a-fA-F]+$/.test(a[k + 2] ?? '')) {
          const v = parseInt(a[k + 2], 16);
          this.setColor(name, [((v & 0xff0000) >> 16) / 255, ((v & 0x00ff00) >> 8) / 255, (v & 0x0000ff) / 255]);
          return 3;
        }
        // dump_modify color: rgbcolors[name] = {r, g, b} with the values as given (image.cpp add_color), no rounding
        const c: Col = [this.num(a[k + 2], 'color'), this.num(a[k + 3], 'color'), this.num(a[k + 4], 'color')];
        this.setColor(name, c);
        return 5;
      }
      case 'amap': {
        const { map, used } = ColorMap.parse(a.slice(k + 1), (c) => this.namedColor(c));
        this.amap = map;
        return used + 1;
      }
      case 'ccolor': {
        // dump_modify ccolor: "computeID color" sets the colour of the objects of a dump image compute (native: "Compute ID is
        // not included in dump"; the colour is read for the const style)
        const f = this.fixObjects.find((x) => x.id === a[k + 1] && x.source === 'compute');
        if (!f) throw new StyleError(`dump ${this.id}: dump_modify ccolor: compute ID '${a[k + 1] ?? ''}' is not included in dump ${this.id}`);
        f.rgb = rgbOf(this.color(a[k + 2] ?? ''));
        return 3;
      }
      case 'fcolor': {
        // dump_modify fcolor: "fcolor ID color" sets the colour of the objects of a dump image "fix" keyword that
        // use the const style (the colour is only read for that style, as in native dump_image.cpp)
        const f = this.fixObjects.find((x) => x.id === a[k + 1] && x.source === 'fix');
        if (!f) throw new StyleError(`dump ${this.id}: dump_modify fcolor: fix ID '${a[k + 1] ?? ''}' is not a dump image fix`);
        f.rgb = rgbOf(this.color(a[k + 2] ?? ''));
        return 3;
      }
      case 'ftrans': {
        // dump_modify ftrans: "fixID transparency" sets the opacity of the objects of that dump image fix (native
        // dump_modify: "Illegal transparency" outside 0..1; "Fix ID is not included in dump" when no such fix)
        const v = this.num(a[k + 2], 'ftrans');
        if (v < 0 || v > 1) throw new StyleError(`dump ${this.id}: illegal transparency ${a[k + 2]} for dump_modify ftrans`);
        const f = this.fixObjects.find((x) => x.id === a[k + 1] && x.source === 'fix');
        if (!f) throw new StyleError(`dump ${this.id}: fix ID ${a[k + 1] ?? ''} is not included in dump ${this.id}`);
        f.opacity = v;
        return 3;
      }
      case 'bmap': {
        // dump_modify bmap: "identical to amap args" (dump_image.html)
        const { map, used } = ColorMap.parse(a.slice(k + 1), (c) => this.namedColor(c));
        this.bmap = map;
        return used + 1;
      }
      case 'gmap': case 'bitrate': case 'framerate': case 'region': case 'thresh':
        throw new StyleError(`dump ${this.id}: dump_modify keyword '${key}' is not supported for dump image by the browser engine`);
      default:
        return 0;
    }
  }

  /** The transparency of the atoms of a type (dump_modify atrans; 1 when not set). */
  private atomOpacity(type: number): number {
    return this.aopacityByType.get(type) ?? 1;
  }

  private typeColor(s: SimState, i: number, colorVals: Float64Array | null, cmap: ColorMap | null): RGB {
    return rgbOf(this.atomColor(s, i, colorVals, cmap));
  }

  /**
   * The colour of an atom as native's DumpImage::create_image sets it: colortype (acolor, else default_colors[type % 16]),
   * colorelement (the element table), or the colour map of an atom value (ColorMap keeps native's double colours).
   */
  private atomColor(s: SimState, i: number, colorVals: Float64Array | null, cmap: ColorMap | null): Col {
    const type = s.type[i];
    const hook = gplHook<ImageShading>('image');
    if (this.attr.color === 'type') return this.acolorByType.get(type) ?? hook.defaultTypeColor(type);
    if (this.attr.color === 'element') return hook.element2color(this.elementName(s, type)) ?? hook.element2color('C')!;
    return cmap!.color(colorVals![i]);
  }

  /**
   * The colour map of an atom colour value: dump_modify amap, or the dump_image.html default
   * "amap = min max cf 0.0 2 min blue max red" (blue at the group minimum, red at the group maximum).
   * The range is the group's minimum and maximum of the values (ColorMap.bounds).
   */
  /** The atom colour map before the current frame (null before the first frame). */
  private priorAtomMap: ColorMap | null = null;

  /**
   * The atom colour map as fix graphics/labels colorscale sees it: the range and colours of the map at the end of the
   * previous dump output (native draws the scale in end_of_step, before the dump's own minmax of this frame).
   */
  colorScaleMap(): ColorMap {
    return this.priorAtomMap ?? (this.amap ?? this.defaultMap).snapshot();
  }

  private atomColorMap(colorVals: Float64Array | null, s: SimState): ColorMap | null {
    if (!colorVals) return null;
    const [lo, hi] = this.rangeOf(colorVals, s);
    const cm = this.amap ?? this.defaultMap;
    cm.bounds(lo, hi);
    return cm;
  }

  private elementName(s: SimState, type: number): string {
    void s;
    return this.ownerElement[type - 1] ?? 'C';
  }

  /** dump_modify element mapping, set by Dump.modify (default C for every type). */
  private ownerElement: string[] = [];

  private diameter(s: SimState, i: number, diamVals: Float64Array | null): number {
    if (this.adiamAll !== null) return this.adiamAll;
    const type = s.type[i];
    if (this.attr.diameter === 'type') return this.adiamByType.get(type) ?? 1.0;
    if (this.attr.diameter === 'element') return gplHook<ImageShading>('image').element2diam(this.elementName(s, type));
    return diamVals![i];
  }

  /** Per-atom values of a colour/diameter attribute (dump.html attribute names, c_ID, f_ID, v_name). */
  private attributeValues(attr: string, s: SimState): Float64Array {
    const n = s.n;
    const out = new Float64Array(n);
    const comp = 'xyz'.indexOf(attr[attr.length - 1]);
    switch (attr) {
      case 'id': for (let i = 0; i < n; i++) out[i] = s.id[i]; return out;
      case 'type': for (let i = 0; i < n; i++) out[i] = s.type[i]; return out;
      case 'mass': for (let i = 0; i < n; i++) out[i] = massOf(s, i); return out;
      case 'q': for (let i = 0; i < n; i++) out[i] = s.q[i]; return out;
      case 'mol': for (let i = 0; i < n; i++) out[i] = s.molecule[i]; return out;
      case 'radius': for (let i = 0; i < n; i++) out[i] = s.radius ? s.radius[i] : 0; return out;
      case 'diameter': for (let i = 0; i < n; i++) out[i] = s.radius ? 2 * s.radius[i] : 0; return out;
      case 'x': case 'y': case 'z': for (let i = 0; i < n; i++) out[i] = s.x[3 * i + comp]; return out;
      case 'xu': case 'yu': case 'zu': {
        const u: number[] = [0, 0, 0];
        for (let i = 0; i < n; i++) { this.sys.geom.unwrap(s.x, s.image, i, u); out[i] = u[comp]; }
        return out;
      }
      case 'vx': case 'vy': case 'vz': for (let i = 0; i < n; i++) out[i] = s.v[3 * i + comp]; return out;
      case 'fx': case 'fy': case 'fz': for (let i = 0; i < n; i++) out[i] = s.f[3 * i + comp]; return out;
    }
    if (attr.startsWith('v_')) return this.sys.atomVariable(attr.slice(2));
    const m = /^([cf])_([A-Za-z0-9_]+)(?:\[(\d+)\])?$/.exec(attr);
    if (!m) throw new StyleError(`dump ${this.id}: unknown dump image attribute '${attr}'`);
    const cols = m[1] === 'c' ? this.sys.compute(m[2]).sizePeratomCols : this.sys.fix(m[2]).sizePeratomCols;
    const vals = m[1] === 'c' ? this.sys.compute(m[2]).peratomValues() : (cols ? this.sys.fix(m[2]).arrayAtom : this.sys.fix(m[2]).vectorAtom);
    if (cols === 0) { out.set(vals.subarray(0, n)); return out; }
    const col = m[3] ? Number(m[3]) : 1;
    for (let i = 0; i < n; i++) out[i] = vals[i * cols + col - 1];
    return out;
  }

  private rangeOf(vals: Float64Array, s: SimState): [number, number] {
    let lo = Infinity, hi = -Infinity;
    const g = this.sys.groupBit(this.group);
    for (let i = 0; i < s.n; i++) {
      if (!(s.mask[i] & g)) continue;
      lo = Math.min(lo, vals[i]);
      hi = Math.max(hi, vals[i]);
    }
    return lo <= hi ? [lo, hi] : [0, 0];
  }

  /** Render one snapshot and store it under `name` as a data: URL. */
  write(s: SimState, name: string, owner: ImageOwner): void {
    this.ownerElement = owner.element;
    const rgb = this.fsaa ? this.renderSupersampled(s) : this.render(s);
    const lower = name.toLowerCase();
    let url: string;
    if (lower.endsWith('.png')) url = 'data:image/png;base64,' + toBase64(encodePng(this.w, this.h, rgb));
    else if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) url = 'data:image/jpeg;base64,' + toBase64(encodeJpeg(this.w, this.h, rgb));
    else url = 'data:image/x-portable-pixmap;base64,' + toBase64(encodePpm(this.w, this.h, rgb));
    this.sys.writeFile(name, url, false);
  }

  // ------------------------------------------------------------------ renderer

  private render(s: SimState): Uint8Array {
    const { w, h } = this;
    const rgb = new Uint8Array(w * h * 3);
    this.clearBackground(rgb);
    const depth = new Float64Array(w * h).fill(-Infinity);

    const box = s.box;
    // Image::view_params works on the bounds of the box (dump_image.cpp box_bounds: domain->boxlo_bound and boxhi_bound
    // for a triclinic box), so the extents, the centre and the scale all use them
    const vb = viewBounds(box);
    const lx = vb.hi[0] - vb.lo[0], ly = vb.hi[1] - vb.lo[1], lz = vb.hi[2] - vb.lo[2];
    const maxdim = Math.max(lx, ly, lz, 1e-12);
    const scale = (h / (2 * maxdim)) * this.zoom;

    // box centre: fractions of the box dimensions (dump_image.cpp, DumpImage::set_image_center: xctr = boxxlo + cx*(boxxhi-boxxlo))
    const center: number[] = [0, 1, 2].map((d) => vb.lo[d] + this.centerFrac[d] * (vb.hi[d] - vb.lo[d]));

    const th = this.theta * DEG, ph = this.phi * DEG;
    let V = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    const up = normalize(this.up);
    // Image::view_params (image.cpp at b332f28): when the view direction is (nearly) parallel to the up vector, camDir is
    // tilted by EPSILON = 1e-6 (theta = 0 and theta = 180 are the degenerate cases), so that camRight stays defined
    if (Math.abs(up[0] * V[0] + up[1] * V[1] + up[2] * V[2]) > 1 - VIEW_EPSILON) {
      if (th === 0) V = [Math.sin(VIEW_EPSILON) * Math.cos(ph), Math.sin(VIEW_EPSILON) * Math.sin(ph), Math.cos(VIEW_EPSILON)];
      else if (th === Math.PI) V = [Math.sin(th - VIEW_EPSILON) * Math.cos(ph), Math.sin(th - VIEW_EPSILON) * Math.sin(ph), Math.cos(th - VIEW_EPSILON)];
      else V = [Math.sin(th + VIEW_EPSILON) * Math.cos(ph), Math.sin(th + VIEW_EPSILON) * Math.sin(ph), Math.cos(th + VIEW_EPSILON)];
    }
    // camRight = up x camDir, camUp = camDir x camRight (both normalised, as Image::view_params does)
    const right = normalize(cross(up, V));
    const ui = normalize(cross(V, right));
    const cx0 = Math.floor(w / 2);
    const cy0 = h % 2 === 0 ? h / 2 - 1.5 : (h - 1) / 2;
    const project = (p: number[]): [number, number, number] => {
      const dx = p[0] - center[0], dy = p[1] - center[1], dz = p[2] - center[2];
      const sx = cx0 + scale * (dx * right[0] + dy * right[1] + dz * right[2]);
      const sy = cy0 - scale * (dx * ui[0] + dy * ui[1] + dz * ui[2]);
      const dep = dx * V[0] + dy * V[1] + dz * V[2];
      return [sx, sy, dep];
    };

    // Image::view_params (image.cpp at b332f28: FOV = MY_PI / 6.0, so tan(FOV) is taken of MY_PI / 6.0 exactly): pixelWidth = maxdel / (height * zoom) with maxdel = 2 * longest edge, and
    // zdist = (maxdel / tan(30 deg) + 0.5 * del . camDir) / zoom with del = 2 * edges (image.cpp at b332f28)
    const maxdel = 2 * maxdim;
    const zdist = (maxdel / Math.tan(Math.PI / 6) + 0.5 * (2 * lx * V[0] + 2 * ly * V[1] + 2 * lz * V[2])) / this.zoom;
    const frame: RasterFrame = {
      w, h, rgb, depth,
      camRight: [right[0], right[1], right[2]], camUp: [ui[0], ui[1], ui[2]], camDir: [V[0], V[1], V[2]],
      ctr: [center[0], center[1], center[2]], zdist, pixelWidth: maxdel / h / this.zoom,
      // dump image shiny; with ssao the ambient colour is 0.5 and the tangents of the surfaces are kept (view_params)
      shiny: this.shiny,
      // Image::view_params sets the ssao ambient at construction; dump_modify lights (later) replaces it
      ambient: this.lightSet ? this.lightSet.ambient : this.ssao ? SSAO_AMBIENT : 0,
      lights: this.lightSet ? { key: this.lightSet.key, fill: this.lightSet.fill, back: this.lightSet.back } : undefined,
      surface: this.ssao ? new Float64Array(2 * w * h) : undefined,
      dist: new Float64Array(w * h).fill(-1),
      spec: this.specularSet ? { hardness: this.specularHardness, intensity: this.specularNone ? 0 : this.shiny } : undefined,
      fsaa: this.fsaa,
    };
    const ctx: DrawCtx = { rgb, depth, project, scale, frame, shading: gplHook<ImageShading>('image') };

    // per-atom colour/diameter attributes (computed once per snapshot)
    const colorVals = this.attr.color !== 'type' && this.attr.color !== 'element' ? this.attributeValues(this.attr.color, s) : null;
    // the colour scale of fix graphics/labels reads the map as the previous output left it (ColorMap::info)
    this.priorAtomMap = (this.amap ?? this.defaultMap).snapshot();
    const cmap = this.atomColorMap(colorVals, s);
    const diamVals = this.attr.diameter !== 'type' && this.attr.diameter !== 'element' ? this.attributeValues(this.attr.diameter, s) : null;

    if (this.boxOn) this.drawBox(s, ctx, this.boxDiam);
    if (this.bond) this.drawBonds(s, ctx, diamVals);
    if (this.autobond) this.drawAutoBonds(s, ctx, this.autobond);

    if (this.atomOn) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.sys.groupBit(this.group))) continue;
        // dump_image.cpp: an atom of a line or tri particle is not drawn as a sphere while those keywords are on
        if (this.lineOn && isLineAtom(s, i)) continue;
        if (this.triOn && isTriAtom(s, i)) continue;
        if (this.ellipsoidOn && isEllipsoid(s, i)) continue;
        if (this.bodyOn) continue;
        const diam = this.diameter(s, i, diamVals);
        if (!(diam > 0)) continue;
        const r = 0.5 * diam * scale;
        if (r < 0.5) continue;
        const base = this.atomColor(s, i, colorVals, cmap);
        const opacity = this.atomOpacity(s.type[i]);
        if (!(opacity > 0)) continue;
        // Image::draw_sphere (image.cpp at b332f28) in world units through the GPL module (draw.ts); the colour stays a
        // double until the shaded pixel write, as native's double colours do
        ctx.shading.drawSphere(frame, [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]], base, diam, opacity);
      }
    }
    if (this.lineOn || this.triOn) this.drawParticles(s, ctx, colorVals, cmap);
    if (this.ellipsoidOn) this.drawEllipsoids(s, ctx, colorVals, cmap);
    if (this.bodyOn) this.drawBodies(s, ctx, colorVals, cmap);
    if (this.fixObjects.length) this.drawFixObjects(s, ctx);
    if (this.regions.length) this.drawRegions(s, ctx);
    if (this.axes) {
      // Image::draw_axes (GPL axes.ts) with the points dump_image.cpp sets up (axesPoints)
      const { axes, diameter } = this.axesPoints(s);
      ctx.shading.drawAxes(frame, axes, diameter, 1, this.backcolor, this.fsaa, s.dimension === 2 ? 2 : 3);
    }
    // Image::compute_SSAO runs on the finished image, before the fsaa reduction (Image::merge)
    if (this.ssao) ctx.shading.applySsao(frame, maxdel * 0.05 * this.ssaoInt, this.ssaoSamples > 0 ? this.ssaoSamples : Math.trunc(8 + 32 * this.ssaoInt), this.ssaoSeed);
    // Image::merge then runs compute_outline, compute_depthcue and compute_defocus on the same image, in that order,
    // still before the fsaa reduction (the image is the doubled one with fsaa). docs: dump_image.html "outline",
    // "depthcue" and "defocus" (2Sep2026): see the keyword comments in parse().
    if (this.outlineOn || this.depthcueOn || this.defocusOn) {
      ctx.shading.postProcess(frame, {
        maxdel,
        fsaa: this.fsaa,
        outline: this.outlineOn ?? undefined,
        depthcue: this.depthcueOn ? { ...this.depthcueOn, background: this.backcolor, background2: this.backcolor2 } : undefined,
        defocus: this.defocusOn ?? undefined,
        boxLo: [s.box.lo[0], s.box.lo[1], s.box.lo[2]],
        boxHi: [s.box.hi[0], s.box.hi[1], s.box.hi[2]],
      });
    }
    return rgb;
  }

  /**
   * Image::clear: the background is the backcolor, or with backcolor2 a vertical gradient. Native rows count from the
   * bottom: row iy has fraction iy / height of backcolor2 and the rest of backcolor, each channel truncated to an int.
   */
  private clearBackground(rgb: Uint8Array): void {
    const { w, h } = this;
    const b1 = this.backcolor, b2 = this.backcolor2;
    for (let iy = 0; iy < h; iy++) {
      const row = h - 1 - iy;
      const f = iy / h;
      const c: RGB = b2
        ? [Math.trunc(f * b2[0] + (1 - f) * b1[0]), Math.trunc(f * b2[1] + (1 - f) * b1[1]), Math.trunc(f * b2[2] + (1 - f) * b1[2])]
        : b1;
      for (let x = 0; x < w; x++) {
        const o = (row * w + x) * 3;
        rgb[o] = c[0]; rgb[o + 1] = c[1]; rgb[o + 2] = c[2];
      }
    }
  }

  /**
   * dump image fsaa: the image is rendered at twice the width and height and every output pixel is the average of
   * a 2x2 block (Image::merge: "0.25 * (sum of the four bytes)", truncated to a byte).
   */
  private renderSupersampled(s: SimState): Uint8Array {
    const w = this.w, h = this.h;
    this.w = 2 * w;
    this.h = 2 * h;
    let big: Uint8Array;
    try {
      big = this.render(s);
    } finally {
      this.w = w;
      this.h = h;
    }
    const out = new Uint8Array(w * h * 3);
    const bw = 2 * w;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i1 = (2 * y * bw + 2 * x) * 3, i2 = i1 + 3, i3 = i1 + 3 * bw, i4 = i3 + 3;
      for (let c = 0; c < 3; c++) {
        out[(y * w + x) * 3 + c] = Math.trunc(0.25 * (big[i1 + c] + big[i2 + c] + big[i3 + c] + big[i4 + c]));
      }
    }
    return out;
  }

  /**
   * Draws the graphics objects of the dump's "fix" keywords (image_objects.ts). Sizes get the keyword's flags
   * added (flag1 to lengths, flag2 to diameters, as dump_image.html states); a shape of zero or negative size is
   * skipped, like native draw_cylinder does. Colour: type and element styles take the object's type through the
   * atom colour or element tables (native: itype = int(type - 1) % ntypes + 1, white when itype <= 0).
   */
  private drawFixObjects(s: SimState, ctx: DrawCtx): void {
    for (const fo of this.fixObjects) {
      const provider = fo.source === 'compute' ? this.sys.compute(fo.id) : this.sys.fixes.find((f) => f.id === fo.id);
      const objs = provider?.imageObjects?.();
      if (!objs) throw new StyleError(`dump ${this.id}: dump image ${fo.source} ${fo.id} provides no graphics objects`);
      for (const o of objs) {
        // objects drawn by the GPL renderer get the keyword's colour, diameter and flag rules through this style
        if (o.kind === 'draw') {
          o.draw(ctx.frame, this.objectStyle(s, fo));
          continue;
        }
        const itype = Math.trunc(o.type - 1) % s.ntypes + 1;
        let base: RGB = [255, 255, 255];
        if (fo.style === 'const') base = fo.rgb;
        else if (itype > 0 && fo.style === 'type') base = rgbOf(this.acolorByType.get(itype) ?? gplHook<ImageShading>('image').defaultTypeColor(itype));
        else if (itype > 0) base = rgbOf(gplHook<ImageShading>('image').element2color(this.elementName(s, itype))!);
        this.drawImageObjectAt(o, base, fo.flag1, fo.flag2, ctx, fo.opacity);
      }
    }
  }

  /**
   * The "region" keywords (dump_image.cpp, "render regions, if supported"). Block, prism, sphere, ellipsoid and plane
   * regions and the points style of any region are drawn by the GPL region module (src/engine-gpl/graphics/region_box.ts),
   * cylinder and cone regions by region_cone.ts. Moving or rotating regions, hull_points, compound regions in the
   * filled, frame or transparent styles, cone regions with open faces and planes of a triclinic box are refused by name.
   */
  private drawRegions(s: SimState, ctx: DrawCtx): void {
    const hook = gplHook<RegionHook>('region');
    const boxlo: [number, number, number] = [s.box.lo[0], s.box.lo[1], s.box.lo[2]];
    const boxhi: [number, number, number] = [s.box.hi[0], s.box.hi[1], s.box.hi[2]];
    for (const r of this.regions) {
      const reg = this.sys.region(r.id);
      if (reg.dynamic) throw new StyleError(`dump ${this.id}: dump image region ${r.id} with move or rotate is not supported by the browser engine`);
      const base = { color: r.color, drawstyle: r.drawstyle, framediam: r.framediam, opacity: r.opacity };
      if (r.drawstyle === 'points') {
        // the points style is the same for every region style (dump_image.cpp, POINTS branch: match() with open faces off)
        hook.drawPointsRegion(ctx.frame, { ...base, npoints: r.npoints }, boxlo, boxhi, (x, y, z) => reg.match(x, y, z));
        continue;
      }
      if (reg instanceof BlockRegion || reg instanceof PrismRegion) {
        const b = reg instanceof BlockRegion ? reg.bounds() : reg.values();
        hook.drawBoxRegion(ctx.frame, { ...base, kind: reg instanceof BlockRegion ? 'block' : 'prism', b, openFaces: reg.openFaces }, boxlo, boxhi);
      } else if (reg instanceof SphereRegion) {
        hook.drawSphereRegion(ctx.frame, { ...base, center: reg.center(), radius: reg.radius() });
      } else if (reg instanceof EllipsoidRegion) {
        hook.drawEllipsoidRegion(ctx.frame, { ...base, center: reg.center(), shape: reg.axes() });
      } else if (reg instanceof PlaneRegion) {
        // PlaneObj is scaled by the box diagonal (dump_image.cpp: len3(domain->prd)); a triclinic box is not drawn
        if (s.box.tilt.some((t) => t !== 0)) throw new StyleError(`dump ${this.id}: dump image plane region ${r.id} in a triclinic box is not supported by the browser engine`);
        const n = reg.normal();
        const len = Math.hypot(n[0], n[1], n[2]);
        if (!(len > 0)) throw new StyleError(`dump ${this.id}: illegal normal vector of plane region ${r.id}`);
        const diag = Math.hypot(s.box.hi[0] - s.box.lo[0], s.box.hi[1] - s.box.lo[1], s.box.hi[2] - s.box.lo[2]);
        hook.drawPlaneRegion(ctx.frame, { ...base, point: reg.point(), normal: [n[0] / len, n[1] / len, n[2] / len], scale: diag }, boxlo, boxhi);
      } else if (reg instanceof ConeRegion) {
        if (reg.openFaces.length > 0) throw new StyleError(`dump ${this.id}: dump image region ${r.id} with open faces is not supported by the browser engine`);
        const g = reg.geometry();
        const spec: RegionConeSpec = { style: reg.style === 'cone' ? 'cone' : 'cylinder', axis: g.axis, c1: g.c1, c2: g.c2, radlo: g.radlo, radhi: g.radhi, lo: g.lo, hi: g.hi, ...base, drawstyle: r.drawstyle as RegionConeSpec['drawstyle'] };
        hook.drawConeRegion(ctx.frame, spec, boxlo, boxhi);
      } else {
        throw new StyleError(`dump ${this.id}: dump image region ${r.id} of style ${reg.style} is not supported by the browser engine`);
      }
    }
  }

  /** The colour of a particle in a "type", "index" or "atom" colour style (dump_image.cpp: index uses the ordinal). */
  private particleRGB(s: SimState, i: number, mode: 'type' | 'atom' | 'index', ordinal: number, colorVals: Float64Array | null, cmap: ColorMap | null): Rgb255 {
    if (mode === 'type') return this.typeRGB(s.type[i]);
    if (mode === 'index') return this.typeRGB((ordinal % s.ntypes) + 1);
    return this.typeColor(s, i, colorVals, cmap);
  }

  /**
   * The atom_style ellipsoid particles of the "ellipsoid" keyword (dump_image.cpp, ellipsoidflag branch): the group's
   * ellipsoids in atom order with their half-axes, orientation and superellipsoid block (when the style has one), drawn by
   * the GPL loop (ellipsoid_mesh.ts). The index colour uses the particle's ordinal among the ellipsoids, as native does.
   */
  private drawEllipsoids(s: SimState, ctx: DrawCtx, colorVals: Float64Array | null, cmap: ColorMap | null): void {
    const e = this.ellipsoidOn!;
    const hook = gplHook<AtomObjectsHook>('atoms');
    const bit = this.sys.groupBit(this.group);
    const sh = s.shape!, q = s.quat!;
    const block = s.custom.get(SUPER_BLOCK)?.data ?? null;
    const atoms: EllipsoidAtom[] = [];
    let ord = 0;
    for (let i = 0; i < s.n; i++) {
      if (!isEllipsoid(s, i)) continue;
      const o = ord++;
      if (!(s.mask[i] & bit)) continue;
      atoms.push({
        center: [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]],
        shape: [sh[3 * i], sh[3 * i + 1], sh[3 * i + 2]],
        quat: [q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]],
        block: block ? [block[2 * i], block[2 * i + 1]] : null,
        color: this.particleRGB(s, i, e.color, o, colorVals, cmap),
        opacity: this.atomOpacity(s.type[i]),
      });
    }
    if (atoms.length) hook.drawEllipsoidAtoms(ctx.frame, atoms, e.estyle, e.level, e.diam);
  }

  /**
   * The atom_style body particles of the "body" keyword (dump_image.cpp, bodyflag branch): every particle of the group, in
   * atom order, with its record (vertices, edges, faces, rounded radius) and colour; the GPL loop draws them (body_image.ts).
   * Their atom spheres are not drawn while the keyword is on.
   */
  private drawBodies(s: SimState, ctx: DrawCtx, colorVals: Float64Array | null, cmap: ColorMap | null): void {
    const b = this.bodyOn!;
    const hook = gplHook<AtomObjectsHook>('atoms');
    const bit = this.sys.groupBit(this.group);
    const rec = s.custom.get(BODY_RECORD);
    if (!rec || !s.quat) return;
    const bstyle = parseBodyStyle(s.atomStyle)!.bstyle as BodyAtom['style'];
    const atoms: BodyAtom[] = [];
    let ord = 0;
    for (let i = 0; i < s.n; i++) {
      const o = ord++;
      if (!(s.mask[i] & bit)) continue;
      const base = rec.cols * i;
      const nsub = rec.data[base];
      // a point particle (no body) draws nothing (dump_image.cpp: "if (body[j] < 0) continue")
      if (nsub === 0) continue;
      const quat = [s.quat[4 * i], s.quat[4 * i + 1], s.quat[4 * i + 2], s.quat[4 * i + 3]];
      const center: [number, number, number] = [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]];
      const color = this.particleRGB(s, i, b.color, o, colorVals, cmap);
      const opacity = this.atomOpacity(s.type[i]);
      if (bstyle === 'nparticle') {
        // BodyNparticle::image: the sub-particle displacements only
        const o0 = bodyRecordOffset(bstyle);
        atoms.push({ style: bstyle, center, quat, nsub, nedge: 0, nface: 0, vertices: Array.from(rec.data.subarray(base + o0, base + o0 + 3 * nsub)), edges: [], faces: [], radius: 0, color, opacity });
      } else if (bstyle === 'rounded/polygon') {
        // BodyRoundedPolygon::image: the vertices, and the stored diameter (rec[4]) in radius
        const o0 = bodyRecordOffset(bstyle);
        atoms.push({ style: bstyle, center, quat, nsub, nedge: 0, nface: 0, vertices: Array.from(rec.data.subarray(base + o0, base + o0 + 3 * nsub)), edges: [], faces: [], radius: rec.data[base + 4], color, opacity });
      } else {
        const nedge = rec.data[base + 4], nface = rec.data[base + 5];
        const v0 = base + POLY_OFF, e0 = v0 + 3 * nsub, f0 = e0 + 2 * nedge;
        atoms.push({
          style: 'rounded/polyhedron', center, quat, nsub, nedge, nface,
          vertices: Array.from(rec.data.subarray(v0, e0)),
          edges: Array.from(rec.data.subarray(e0, f0)),
          faces: Array.from(rec.data.subarray(f0, f0 + 4 * nface)),
          radius: rec.data[base + 7],
          color, opacity,
        });
      }
    }
    if (atoms.length) hook.drawBodyAtoms(ctx.frame, atoms, b.flag1, b.flag2);
  }

  /** The dump's colour of a type (acolor, or the default type colours): the "type" colour style of line and tri. */
  private typeRGB(t: number): RGB {
    return rgbOf(this.acolorByType.get(t) ?? gplHook<ImageShading>('image').defaultTypeColor(t));
  }

  /**
   * The atom_style line and tri particles of the "line" and "tri" keywords, drawn by the GPL loops (atom_objects.ts) in
   * native order: all line segments, then all triangles. The colour style follows dump_image.cpp: type uses the atom
   * type's colour; index uses the particle's ordinal among the particles of that kind, as the native bonus index
   * (type = ordinal % ntypes + 1); atom uses the atom colour. Only particles in the dump's group are drawn.
   */
  private drawParticles(s: SimState, ctx: DrawCtx, colorVals: Float64Array | null, cmap: ColorMap | null): void {
    const hook = gplHook<AtomObjectsHook>('atoms');
    const bit = this.sys.groupBit(this.group);
    const colorOf = (i: number, mode: 'type' | 'atom' | 'index', ordinal: number): Rgb255 => this.particleRGB(s, i, mode, ordinal, colorVals, cmap);
    const lb = this.lineOn ? s.custom.get(LINE_BONUS)?.data : undefined;
    const tb = this.triOn ? s.custom.get(TRI_BONUS)?.data : undefined;
    const lines: LineAtom[] = [];
    const tris: TriAtom[] = [];
    let nl = 0, nt = 0;
    for (let i = 0; i < s.n; i++) {
      const inGroup = (s.mask[i] & bit) !== 0;
      if (this.lineOn && isLineAtom(s, i)) {
        const ord = nl++;
        if (inGroup) lines.push({ x: s.x[3 * i], y: s.x[3 * i + 1], length: lb![2 * i], theta: lb![2 * i + 1], color: colorOf(i, this.lineOn.color, ord), opacity: this.atomOpacity(s.type[i]) });
      }
      if (this.triOn && isTriAtom(s, i)) {
        const ord = nt++;
        if (inGroup) {
          const q = s.quat!;
          tris.push({
            center: [s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]],
            quat: [q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]],
            corners: Array.from(tb!.subarray(TRI_BONUS_COLS * i, TRI_BONUS_COLS * i + 9)),
            color: colorOf(i, this.triOn.color, ord),
            opacity: this.atomOpacity(s.type[i]),
          });
        }
      }
    }
    if (this.lineOn && lines.length) hook.drawLineAtoms(ctx.frame, lines, this.lineOn.diam);
    if (this.triOn && tris.length) hook.drawTriAtoms(ctx.frame, tris, this.triOn.tstyle, this.triOn.diam);
  }

  /** The style of one "fix" keyword for the GPL graphics objects (ObjectStyle in image_objects.ts). */
  private objectStyle(s: SimState, fo: { style: 'type' | 'element' | 'const'; flag1: number; flag2: number; opacity: number; rgb: RGB }): ObjectStyle {
    const colorOf = (type: number): Rgb255 => {
      const itype = Math.trunc(type - 1) % s.ntypes + 1;
      if (fo.style === 'const') return [fo.rgb[0], fo.rgb[1], fo.rgb[2]];
      if (itype <= 0) return [255, 255, 255];
      if (fo.style === 'type') return rgbOf(this.acolorByType.get(itype) ?? gplHook<ImageShading>('image').defaultTypeColor(itype));
      return rgbOf(gplHook<ImageShading>('image').element2color(this.elementName(s, itype))!);
    };
    // default sphere size of a type: the dump's adiam setting (numeric, per type, or per element)
    const diameterOf = (type: number): number => {
      if (this.adiamAll !== null) return this.adiamAll;
      if (this.attr.diameter === 'type') return this.adiamByType.get(type) ?? 1.0;
      if (this.attr.diameter === 'element') return gplHook<ImageShading>('image').element2diam(this.elementName(s, type));
      return 1.0;
    };
    // bond width of the dump's bond keyword (a number, or atom / type as the default sizes of the two atoms)
    const bondDiameter = (t1: number, t2: number): number => {
      const w = this.bond?.width;
      if (w === undefined) return 0.5;
      if (w === 'atom' || w === 'type') return Math.min(diameterOf(t1), diameterOf(t2));
      if (w === 'none') return 0;
      return Number(w);
    };
    return { flag1: fo.flag1, flag2: fo.flag2, opacity: fo.opacity, colorOf, diameterOf, bondDiameter };
  }

  /**
   * One graphics object of a fix at its ftrans opacity: below 1 the object's pixels go through the screen-door mask of
   * Image's draw loops (atoms.ts hook transparentPixel, image.cpp transthresh), otherwise the object is drawn as it is.
   */
  private drawImageObjectAt(o: ImageObject, base: RGB, flag1: number, flag2: number, c: DrawCtx, opacity: number): void {
    if (!(opacity > 0)) return;
    this.withOpacity(opacity, () => this.drawImageObject(o, base, flag1, flag2, c));
  }

  /**
   * Runs one draw at an opacity below 1: the pixels of the screen-door mask (image.cpp transthresh, the same test as
   * ftrans and dump_modify atrans) are skipped. At opacity 1 the draw runs as it is.
   */
  private withOpacity(opacity: number, draw: () => void): void {
    if (opacity >= 1) {
      draw();
      return;
    }
    const mask = gplHook<AtomObjectsHook>('atoms');
    const h = this.h;
    this.gate = (xx, yy) => !mask.transparentPixel(opacity, xx, h - 1 - yy);
    try {
      draw();
    } finally {
      this.gate = null;
    }
  }

  /** One graphics object in colour `base`; flag1 and flag2 as dump image's "fix" keyword gives them. */
  private drawImageObject(o: ImageObject, base: RGB, flag1: number, flag2: number, c: DrawCtx): void {
    const { rgb, depth, project, scale } = c;
    if (o.kind === 'draw') throw new StyleError(`dump ${this.id}: internal: GPL graphics object reached the core object drawing`);
    if (o.kind === 'sphere') {
      const r = 0.5 * (o.diameter + flag2) * scale;
      if (!(r > 0)) return;
      const [px, py, pd] = project(o.center);
      this.drawSphere(c, px, py, pd, r, base);
    } else if (o.kind === 'cylinder') {
      const d = o.diameter + flag2;
      if (!(d > 0)) return;
      this.drawStroke(rgb, depth, project(o.p1), project(o.p2), base, Math.max(1, d * scale));
      // flag1 is the sflag of native draw_cylinder: bit 0 caps p1, bit 1 caps p2, each with a sphere
      const caps = Math.trunc(flag1);
      const r = 0.5 * d * scale;
      if (caps & 1) { const [px, py, pd] = project(o.p1); this.drawSphere(c, px, py, pd, r, base); }
      if (caps & 2) { const [px, py, pd] = project(o.p2); this.drawSphere(c, px, py, pd, r, base); }
    } else if (o.kind === 'cone') {
      this.drawCone(rgb, depth, project, o.p1, o.p2, o.d1 + flag2, o.d2 + flag2, scale, base);
    } else {
      // dump_image.cpp (the ARROW branch of create_image, b332f28): ImageObjects::ArrowObj(ratio).draw with the object's
      // length + flag1 and diameter + flag2; the template is drawn by the GPL graphics module (axes.ts drawArrowObj)
      const d = o.diameter + flag2;
      c.shading.drawArrow(c.frame, [base[0] / 255, base[1] / 255, base[2] / 255], o.center, o.length + flag1, o.dir, d, 1, o.ratio);
    }
  }

  /**
   * The axes of dump image "axes" (dump_image.cpp, create_image, "render XYZ axes" at b332f28): the diameter is the shortest
   * box edge (x and y, and z in 3d) times the axes diameter; the origin is the lower corner of the box and the x, y and z
   * points are its upper corners in those directions; all four are moved by the offset (10% of the longest edge, half of it
   * for center, times 8 for the upper and right positions of the location); the arrows then reach axes-length times the
   * edges from the origin. Native forms each arrow end as origin + len * (point - origin), and so does this. Triclinic
   * boxes are refused.
   */
  private axesPoints(s: SimState): { axes: Vec3[]; diameter: number } {
    const b = s.box;
    if (b.tilt.some((t) => t !== 0)) throw new StyleError(`dump ${this.id}: dump image axes for a triclinic box are not supported by the browser engine`);
    const dim = s.dimension === 2 ? 2 : 3;
    const axes = this.axes!;
    const lo = b.lo, hi = b.hi;
    let diameter = Math.min(hi[0] - lo[0], hi[1] - lo[1]);
    if (dim === 3) diameter = Math.min(diameter, hi[2] - lo[2]);
    diameter *= axes.diam;
    const pts: Vec3[] = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]]];
    let offset = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
    if (dim === 3) offset = Math.max(offset, hi[2] - lo[2]);
    offset *= axes.loc === 'center' ? 0.5 : 0.1;
    const m = AXES_SHIFT[dim][axes.loc];
    for (const p of pts) for (let c = 0; c < 3; c++) {
      // native: "+= offset", "-= offset" or "+= 8.0*offset"
      if (m[c] === 1) p[c] = p[c] + offset;
      else if (m[c] === -1) p[c] = p[c] - offset;
      else p[c] = p[c] + m[c] * offset;
    }
    for (let k = 1; k < 4; k++) for (let c = 0; c < 3; c++) pts[k][c] = pts[0][c] + axes.len * (pts[k][c] - pts[0][c]);
    return { axes: pts, diameter };
  }

  /**
   * A cone as chords perpendicular to its projected axis (the silhouette of a side-on cone); a cone that points at the
   * viewer (its axis projects to a point) is a flat disc of its largest diameter.
   */
  private drawCone(rgb: Uint8Array, depth: Float64Array, project: (p: number[]) => [number, number, number], p1: Vec3, p2: Vec3, d1: number, d2: number, scale: number, color: RGB): void {
    const a = project(p1), b = project(p2);
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    if (len < 1) {
      const r = 0.5 * Math.max(d1, d2) * scale;
      if (r > 0) this.drawDisc(rgb, depth, a[0], a[1], a[2], Math.max(0.5, r), color);
      return;
    }
    const nx = -dy / len, ny = dx / len;
    const steps = Math.ceil(len);
    for (let t = 0; t <= steps; t++) {
      const f = t / steps;
      const d = d1 + (d2 - d1) * f;
      if (!(d > 0)) continue;
      const r = 0.5 * d * scale;
      const x = a[0] + dx * f, y = a[1] + dy * f, z = a[2] + (b[2] - a[2]) * f;
      this.drawLine(rgb, depth, [x + nx * r, y + ny * r, z], [x - nx * r, y - ny * r, z], color, 1);
    }
  }

  private drawDisc(rgb: Uint8Array, depth: Float64Array, x: number, y: number, z: number, r: number, color: RGB): void {
    const w = this.w, h = this.h;
    const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(w - 1, Math.ceil(x + r));
    const y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(h - 1, Math.ceil(y + r));
    for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
      if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > r * r) continue;
      if (this.gate && !this.gate(xx, yy)) continue;
      const o = yy * w + xx;
      if (z <= depth[o]) continue;
      depth[o] = z;
      rgb[3 * o] = color[0]; rgb[3 * o + 1] = color[1]; rgb[3 * o + 2] = color[2];
    }
  }

  /** A straight stroke of a given pixel width between two projected points (the drawLine rule). */
  private drawStroke(rgb: Uint8Array, depth: Float64Array, p1: [number, number, number], p2: [number, number, number], color: RGB, width: number): void {
    this.drawLine(rgb, depth, p1, p2, color, width);
  }

  /**
   * Image::draw_sphere: the pixels of a sphere of radius r (pixels) centred at (px, py). Each pixel is shaded by the
   * draw_pixel light sum of its camera-space normal (x along the image right, y up, z towards the viewer) with the
   * frame's shiny and ambient values, from the shading helpers of the GPL graphics module. A pixel's depth is the depth
   * of the sphere centre plus its surface offset (native draw_sphere: dist - sz).
   */
  private drawSphere(c: DrawCtx, px: number, py: number, pd: number, r: number, base: RGB): void {
    const w = this.w, h = this.h;
    const { rgb, depth, frame, shading } = c;
    const x0 = Math.max(0, Math.floor(px - r - 1)), x1 = Math.min(w - 1, Math.ceil(px + r + 1));
    const y0 = Math.max(0, Math.floor(py - r - 1)), y1 = Math.min(h - 1, Math.ceil(py + r + 1));
    const r2 = r * r;
    const col: [number, number, number] = [base[0] / 255, base[1] / 255, base[2] / 255];
    for (let y = y0; y <= y1; y++) {
      const dy = y + 0.5 - py;
      for (let x = x0; x <= x1; x++) {
        const dx = x + 0.5 - px;
        const q = (dx * dx + dy * dy) / r2;
        if (q > 1) continue;
        if (this.gate && !this.gate(x, y)) continue;
        const idx = y * w + x;
        const nz = Math.sqrt(1 - q);
        // camera space: right = dx / r, up = -dy / r (rows grow downwards), towards the viewer = nz
        const normal: [number, number, number] = [dx / r, -dy / r, nz];
        // the depth buffer is in world units (Image::draw_pixel's dist - sz, image.cpp at b332f28): the radius r is in
        // pixels here, so the world radius is r / scale
        const surfDepth = pd + (r / c.scale) * nz;
        if (surfDepth <= depth[idx]) continue;
        depth[idx] = surfDepth;
        if (frame.surface) {
          frame.surface[2 * idx] = normal[1];
          frame.surface[2 * idx + 1] = -normal[0];
        }
        const sh = shading.shadeSurface(normal, col, frame.shiny ?? 1, frame.ambient ?? 0, frame.lights, frame.spec);
        const o = idx * 3;
        rgb[o] = clamp(sh[0] * 255);
        rgb[o + 1] = clamp(sh[1] * 255);
        rgb[o + 2] = clamp(sh[2] * 255);
      }
    }
  }

  /**
   * The simulation box of dump image "box" (dump_image.cpp, create_image, "render outline of simulation box" at b332f28):
   * the diameter is the shortest box edge (x and y, and z in 3d) times the box keyword's value, and Image::draw_box
   * (image.cpp at b332f28) draws the twelve edges in its fixed order as cylinders with caps at both ends, through the
   * GPL drawCylinder. The corners follow Domain::lamda_box_corners (x changing fastest); an orthogonal box uses the
   * bounds themselves, a triclinic one lamda2x (x = h0 l0 + h5 l1 + h4 l2 + lo).
   */
  private drawBox(s: SimState, ctx: DrawCtx, boxDiam: number): void {
    const b = s.box;
    const lens = [b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], b.hi[2] - b.lo[2]];
    let diameter = Math.min(lens[0], lens[1]);
    if (s.dimension === 3) diameter = Math.min(diameter, lens[2]);
    diameter *= boxDiam;
    if (!(diameter > 0)) return;
    const tri = b.tilt.some((t) => t !== 0);
    const corners: Vec3[] = [];
    for (let k = 0; k < 2; k++) for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) {
      if (tri) {
        const xy = b.tilt[0], xz = b.tilt[1], yz = b.tilt[2];
        corners.push([lens[0] * i + xy * j + xz * k + b.lo[0], lens[1] * j + yz * k + b.lo[1], lens[2] * k + b.lo[2]]);
      } else {
        corners.push([i ? b.hi[0] : b.lo[0], j ? b.hi[1] : b.lo[1], k ? b.hi[2] : b.lo[2]]);
      }
    }
    // Image::draw_box: the edges between corners (index bit 0 = x, bit 1 = y, bit 2 = z)
    const edges: [number, number][] = [[0, 1], [2, 3], [0, 2], [1, 3], [0, 4], [1, 5], [2, 6], [3, 7], [4, 5], [6, 7], [4, 6], [5, 7]];
    for (const [a, c] of edges) ctx.shading.drawCylinder(ctx.frame, corners[a], corners[c], this.boxcolor, diameter, 3, 1);
  }

  /**
   * The per-bond values of a bond colour compute (dump_image.cpp, the LOCALVALUE checks of init and create_image at
   * b332f28): the compute's local rows are the bonds of its group in native order (termRows), keyed by the bond's index.
   * The colour map of the values is set from their range when it is dynamic (native map_minmax over all rows).
   */
  private bondValues(bc: { id: string; column: number }): { values: Map<number, number>; lo: number; hi: number } {
    const comp = this.sys.compute(bc.id);
    if (!comp.localFlag) throw new StyleError(`dump ${this.id}: dump image bond compute ${bc.id} does not compute local info`);
    if (bc.column === 0 && comp.sizeLocalCols !== 0) throw new StyleError(`dump ${this.id}: dump image bond compute ${bc.id} does not compute a local vector`);
    if (bc.column > 0 && comp.sizeLocalCols === 0) throw new StyleError(`dump ${this.id}: dump image bond compute ${bc.id} does not compute a local array`);
    if (bc.column > comp.sizeLocalCols) throw new StyleError(`dump ${this.id}: dump image bond compute ${bc.id} local array has no column ${bc.column}`);
    const ncol = Math.max(1, comp.sizeLocalCols);
    const col = Math.max(0, bc.column - 1);
    const data = comp.localValues();
    const rows = termRows(this.sys, 'bond', comp.groupBit);
    const values = new Map<number, number>();
    let lo = Infinity, hi = -Infinity;
    rows.forEach(({ k }, r) => {
      const v = data[r * ncol + col];
      values.set(k, v);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    return { values, lo, hi };
  }

  /** The bond colour map as fix graphics/labels colorscale map bond sees it: the previous output's (ColorMap::info). */
  bondColorScaleMap(): ColorMap {
    return this.priorBondMap ?? (this.bmap ?? this.defaultBondMap).snapshot();
  }

  private drawBonds(s: SimState, ctx: DrawCtx, diamVals: Float64Array | null): void {
    const bonds = s.topo.bonds;
    if (!bonds || bonds.n === 0) return;
    // native draws the halves of an atom-coloured bond with the atoms' transparency (aopacity); the engine draws a bond in
    // one colour, so with any transparent type the bonds are named as unsupported instead of drawn opaque
    for (const v of this.aopacityByType.values()) {
      if (v < 1) throw new StyleError(`dump ${this.id}: dump image bonds with dump_modify atrans below 1 are not supported by the browser engine`);
    }
    const map = buildAtomMap(s);
    const bond = this.bond!;
    const colorVals = this.attr.color !== 'type' && this.attr.color !== 'element' ? this.attributeValues(this.attr.color, s) : null;
    const cmap = this.atomColorMap(colorVals, s);
    const hook = gplHook<ImageShading>('image');
    // bonds coloured by a compute: the map of the values (dynamic range from all rows) gives each bond's colour
    let bvals: Map<number, number> | null = null;
    let bmapOf: ColorMap | null = null;
    if (this.bondCompute) {
      const bv = this.bondValues(this.bondCompute);
      bvals = bv.values;
      this.priorBondMap = (this.bmap ?? this.defaultBondMap).snapshot();
      bmapOf = this.bmap ?? this.defaultBondMap;
      bmapOf.bounds(bv.lo, bv.hi);
    }
    for (let bi = 0; bi < bonds.n; bi++) {
      const i1 = map[bonds.atoms[2 * bi]], i2 = map[bonds.atoms[2 * bi + 1]];
      if (i1 < 0 || i2 < 0) continue;
      if (!(s.mask[i1] & this.sys.groupBit(this.group)) || !(s.mask[i2] & this.sys.groupBit(this.group))) continue;
      let width: number;
      if (bond.width === 'none') continue;
      if (bond.width === 'atom') width = Math.min(this.diameter(s, i1, diamVals), this.diameter(s, i2, diamVals));
      else if (bond.width === 'type') width = this.bdiamByType.get(bonds.type[bi]) ?? 0.5;
      else width = Number(bond.width);
      if (!(width > 0)) continue;
      let col: Col;
      if (bond.color === 'compute') {
        const v = bvals!.get(bi);
        if (v === undefined) throw new StyleError(`dump ${this.id}: dump image bond compute ${this.bondCompute!.id} produced fewer values than rendered bonds; its group must select the same bonds as the dump`);
        col = bmapOf!.color(v);
      } else {
        col = bond.color === 'type' ? (this.bcolorByType.get(bonds.type[bi]) ?? hook.defaultTypeColor(bonds.type[bi])) : this.atomColor(s, i1, colorVals, cmap);
      }
      const x1: Vec3 = [s.x[3 * i1], s.x[3 * i1 + 1], s.x[3 * i1 + 2]];
      const x2: Vec3 = [s.x[3 * i2], s.x[3 * i2 + 1], s.x[3 * i2 + 2]];
      // dump_image.cpp (DumpImage::create_image, bond loop at b332f28): "draw cylinder in 2 pieces if bcolor = ATOM
      // or bond crosses periodic boundary". A crossing bond takes the minimum image of its vector (Domain::minimum_image_check,
      // the half box length per periodic dimension) and is drawn from each end to the midpoint; an atom-coloured bond is
      // drawn in two halves, each in the colour of its own atom. Image::draw_cylinder (sflag 3) is the GPL drawCylinder.
      const g = this.sys.geom;
      const del = [x2[0] - x1[0], x2[1] - x1[1], x2[2] - x1[2]];
      const crosses = (g.periodic[0] && Math.abs(del[0]) > 0.5 * g.lx) || (g.periodic[1] && Math.abs(del[1]) > 0.5 * g.ly) || (g.periodic[2] && Math.abs(del[2]) > 0.5 * g.lz);
      if (!crosses && bond.color !== 'atom') {
        ctx.shading.drawCylinder(ctx.frame, x1, x2, col, width, 3, 1);
        continue;
      }
      if (crosses) g.minimumImage(del);
      const mid1: Vec3 = [x1[0] + 0.5 * del[0], x1[1] + 0.5 * del[1], x1[2] + 0.5 * del[2]];
      const mid2: Vec3 = [x2[0] - 0.5 * del[0], x2[1] - 0.5 * del[1], x2[2] - 0.5 * del[2]];
      const col2: Col = bond.color === 'atom' ? this.atomColor(s, i2, colorVals, cmap) : col;
      ctx.shading.drawCylinder(ctx.frame, x1, mid1, col, width, 3, 1);
      ctx.shading.drawCylinder(ctx.frame, mid2, x2, col2, width, 3, 1);
    }
  }

  /**
   * dump image autobond (dump_image.cpp, the AUTO branch of create_image): the pair search and the two-halves drawing are
   * the GPL autobond loop (engine-gpl/graphics/autobond.ts); the core gives it the atoms, their colours and the rules.
   */
  private drawAutoBonds(s: SimState, ctx: DrawCtx, ab: { cutoff: number; width: number }): void {
    // the same atrans rule as the bond keyword: the halves of a bond would need the atoms' transparency
    for (const v of this.aopacityByType.values()) {
      if (v < 1) throw new StyleError(`dump ${this.id}: dump image bonds with dump_modify atrans below 1 are not supported by the browser engine`);
    }
    const group = this.sys.groupBit(this.group);
    const colorVals = this.attr.color !== 'type' && this.attr.color !== 'element' ? this.attributeValues(this.attr.color, s) : null;
    const cmap = this.atomColorMap(colorVals, s);
    // Domain::minimum_image; the hydrogen rule (both masses below 3) applies to units metal and real only
    const g = this.sys.geom;
    const hydrogenRule = s.units.style === 'metal' || s.units.style === 'real';
    // native's search runs over the neighbor list, which drops the special pairs whose lj and coul weights are both 0
    // (neighbor.ts encode; keepExcluded when a kspace or coul/long style needs them). The special lists are the ones
    // the force field builds from the bonds.
    const ff = this.sys.ff;
    const sp = s.topo.bonds.n > 0 ? buildSpecial(s, { angle: ff.special.angle, dihedral: ff.special.dihedral }) : null;
    const keepExcluded = !!ff.kspace || !!ff.pair?.coulLong || !!ff.pair?.keepExcluded;
    const ljw = [1, ...ff.special.lj], clw = [1, ...ff.special.coul];
    const excluded = sp && !keepExcluded ? (i: number, j: number): boolean => {
      for (let k = sp.offset[i]; k < sp.offset[i + 1]; k++) {
        if (sp.partner[k] !== s.id[j]) continue;
        const o = sp.order[k];
        return ljw[o] === 0 && clw[o] === 0;
      }
      return false;
    } : undefined;
    ctx.shading.drawAutoBonds(ctx.frame, {
      n: s.n,
      x: s.x,
      selected: (i) => (s.mask[i] & group) !== 0,
      mass: hydrogenRule ? (i) => massOf(s, i) : null,
      minimumImage: (d) => g.minimumImage(d),
      excluded,
      colorOf: (i) => this.atomColor(s, i, colorVals, cmap),
      cutoff: ab.cutoff,
      width: ab.width,
    });
  }

  private drawLine(rgb: Uint8Array, depth: Float64Array, p1: [number, number, number], p2: [number, number, number], color: RGB, width: number): void {
    const w = this.w, h = this.h;
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
    const len = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(len));
    const rad = Math.max(0.5, width / 2);
    for (let t = 0; t <= steps; t++) {
      const x = p1[0] + (dx * t) / steps;
      const y = p1[1] + (dy * t) / steps;
      const d = p1[2] + ((p2[2] - p1[2]) * t) / steps;
      const x0 = Math.max(0, Math.floor(x - rad)), x1 = Math.min(w - 1, Math.ceil(x + rad));
      const y0 = Math.max(0, Math.floor(y - rad)), y1 = Math.min(h - 1, Math.ceil(y + rad));
      for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
        if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > rad * rad) continue;
        if (this.gate && !this.gate(xx, yy)) continue;
        const o = (yy * w + xx) * 3;
        if (d <= depth[yy * w + xx]) continue;
        depth[yy * w + xx] = d;
        rgb[o] = color[0]; rgb[o + 1] = color[1]; rgb[o + 2] = color[2];
      }
    }
  }
}

const clamp = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v | 0);
