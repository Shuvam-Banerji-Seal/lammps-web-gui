import type { System } from '../system';
import type { SimState } from '../types';
import { StyleError } from '../force/types';
import { buildAtomMap, massOf } from '../atoms';
import { encodePng } from './png';
import { encodeJpeg } from './jpeg';
import { ColorMap } from './image_colormap';
import { arrowParts, type ImageObject, type ImageShading, type ObjectStyle, type RasterFrame, type Rgb255, type Vec3 } from './image_objects';
import { gplHook } from '../gplHooks';

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

type RGB = [number, number, number];

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

/** A useful subset of the 140 pre-defined colour names (dump_image.html). */
const NAMED_COLORS: Record<string, RGB> = {
  red: [255, 0, 0], green: [0, 255, 0], blue: [0, 0, 255], yellow: [255, 255, 0],
  aqua: [0, 255, 255], cyan: [0, 255, 255], magenta: [255, 0, 255], fuchsia: [255, 0, 255],
  white: [255, 255, 255], black: [0, 0, 0], gray: [128, 128, 128], grey: [128, 128, 128],
  silver: [192, 192, 192], maroon: [128, 0, 0], olive: [128, 128, 0], lime: [0, 255, 0],
  teal: [0, 128, 128], navy: [0, 0, 128], purple: [128, 0, 128], orange: [255, 165, 0],
  pink: [255, 192, 203], brown: [165, 42, 42], gold: [255, 215, 0], violet: [238, 130, 238],
  darkred: [139, 0, 0], darkgreen: [0, 100, 0], darkblue: [0, 0, 139], lightgray: [211, 211, 211],
  lightgrey: [211, 211, 211], indigo: [75, 0, 130], turquoise: [64, 224, 208], salmon: [250, 128, 114],
  khaki: [240, 230, 140], crimson: [220, 20, 60], orchid: [218, 112, 214], tan: [210, 180, 140],
  beige: [245, 245, 220], coral: [255, 127, 80], plum: [221, 160, 221], azure: [240, 255, 255],
  // the colour names the GRAPHICS examples use (dump image colour table of src/GRAPHICS/image.cpp at b332f28,
  // the values in 0..1 times 255 rounded)
  forestgreen: [34, 139, 34], goldenrod: [218, 165, 32], indianred: [205, 92, 92],
  darkgray: [69, 69, 69], steelblue: [70, 130, 180], slategray: [112, 128, 144], lightslategray: [119, 136, 153],
  midnightblue: [25, 25, 112], firebrick: [178, 34, 34], darkslategray: [47, 79, 79],
};

/**
 * dump_image.html Default: "acolor = \* red/green/blue/yellow/aqua/cyan". Measured with native
 * LAMMPS (black box): types 1..6 render red, green, blue, yellow, aqua, magenta, so the sixth
 * entry is magenta (255,0,255) here; the mapping "repeats itself for types > 6".
 */
const TYPE_COLORS: RGB[] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]];

/** Atom attributes dump image can colour/size by (dump.html attribute names + c_/f_/v_); type and element are special. */
const SUPPORTED_ATTRS = new Set([
  'type', 'element', 'id', 'mass', 'x', 'y', 'z', 'xu', 'yu', 'zu', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz',
  'q', 'mol', 'radius', 'diameter',
]);

/** Common element colours/radii (AtomEye-like); every type defaults to C (dump_image.html). */
const ELEMENTS: Record<string, [RGB, number]> = {
  H: [[255, 255, 255], 0.5], He: [[217, 255, 255], 0.6], Li: [[204, 128, 255], 1.5], Be: [[194, 255, 0], 1.1],
  B: [[255, 181, 181], 0.9], C: [[144, 144, 144], 1.5], N: [[48, 80, 248], 1.4], O: [[255, 13, 13], 1.3],
  F: [[144, 224, 80], 1.2], Ne: [[179, 227, 245], 1.2], Na: [[171, 92, 242], 1.9], Mg: [[138, 255, 0], 1.6],
  Al: [[191, 166, 166], 1.5], Si: [[240, 200, 160], 1.5], P: [[255, 128, 0], 1.4], S: [[255, 255, 48], 1.4],
  Cl: [[31, 240, 31], 1.4], Ar: [[128, 209, 227], 1.3], K: [[143, 64, 212], 2.2], Ca: [[61, 255, 0], 2.0],
  Fe: [[224, 102, 51], 1.6], Cu: [[200, 128, 51], 1.5], Zn: [[125, 128, 176], 1.5], Au: [[255, 209, 35], 1.6],
};

interface Attr {
  color: string;
  diameter: string;
}

type Line = { a: RGB; b: RGB; x1: number; y1: number; d1: number; x2: number; y2: number; d2: number; w: number };

/** The pieces of Dump.write() the image renderer reads. */
export interface ImageOwner {
  readonly id: string;
  element: string[];
}

const DEG = Math.PI / 180;

/** dump image axes locations; null is "no", "yes" is the lower left corner. */
type AxesLocation = 'center' | 'lowerleft' | 'lowerright' | 'upperleft' | 'upperright';
/** The colours of the x, y and z axes of dump image "axes" (red, green, blue). */
const AXIS_COLORS: RGB[] = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
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

const toBase64 = (bytes: Uint8Array): string => {
  const T = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let out = '';
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i], b1 = bytes[i + 1], b2 = bytes[i + 2];
    out += T[b0 >> 2];
    out += T[((b0 & 3) << 4) | (b1 === undefined ? 0 : b1 >> 4)];
    out += b1 === undefined ? '=' : T[((b1 & 15) << 2) | (b2 === undefined ? 0 : b2 >> 6)];
    out += b2 === undefined ? '=' : T[b2 & 63];
  }
  return out;
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
  private acolorByType = new Map<number, RGB>();
  private adiamByType = new Map<number, number>();
  private bcolorByType = new Map<number, RGB>();
  private bdiamByType = new Map<number, number>();
  private backcolor: RGB = [0, 0, 0];
  /** dump_modify backcolor2: the top of a background gradient (null: a single background colour). */
  private backcolor2: RGB | null = null;
  private boxcolor: RGB = [255, 255, 0];
  /** dump_modify color definitions (name -> RGB), local to this dump. */
  private userColors = new Map<string, RGB>();
  private bond: { color: string; width: string } | null;
  /** dump_modify amap (colour map of an atom colour value); null = the dump_image.html default. */
  private amap: ColorMap | null = null;
  private readonly defaultMap = ColorMap.default();
  /** dump image shiny (specular intensity; the highlight hardness is 16 * shiny), fsaa, and ssao (seed, intensity). */
  private shiny = 1;
  private fsaa = false;
  private ssao = false;
  private ssaoSeed = 1;
  private ssaoInt = 0;
  /** dump image "axes" keyword: the location, the length and the diameter fraction (null: no axes). */
  private axes: { loc: AxesLocation; len: number; diam: number } | null = null;
  /** dump image "fix ID style flag1 flag2" keywords: the fixes whose graphics objects are drawn. */
  private fixObjects: { id: string; style: 'type' | 'element' | 'const'; flag1: number; flag2: number; rgb: RGB }[] = [];

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

  private num(w: string, key: string): number {
    if (w.startsWith('v_')) return this.sys.equalVariable(w.slice(2));
    const n = Number(w);
    if (!Number.isFinite(n)) throw new StyleError(`dump ${this.id}: dump image ${key} needs a number`);
    return n;
  }

  private color(w: string): RGB {
    const c = this.userColors.get(w) ?? NAMED_COLORS[w];
    if (!c) throw new StyleError(`dump ${this.id}: unknown colour '${w}'`);
    return [c[0], c[1], c[2]];
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
          if (color !== 'atom' && color !== 'type' && color !== 'none') throw new StyleError(`dump ${this.id}: dump image bond color '${color}' is not supported (atom, type or none)`);
          if (width !== 'atom' && width !== 'type' && width !== 'none' && !Number.isFinite(Number(width))) throw new StyleError(`dump ${this.id}: dump image bond width '${width}' is not supported`);
          if (color === 'none' && width === 'none') this.bond = null;
          else this.bond = { color, width };
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
          this.fixObjects.push({ id, style, flag1: this.num(a[k + 3], 'fix'), flag2: this.num(a[k + 4], 'fix'), rgb: [255, 255, 255] });
          k += 5;
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
            // the letters X, Y and Z are bitmaps in native image.cpp (Image::draw_axes), not ported: named, not silent
            this.sys.log(`WARNING: dump ${this.id} image axes: the X, Y and Z letters are not drawn by the browser engine`);
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
        // ellipsoid draws ellipsoids (and superellipsoids, atom_style ellipsoid superellipsoid): no renderer here
        case 'ellipsoid': case 'autobond': case 'grid': case 'line': case 'tri': case 'body': case 'region':
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
    const colors = (w: string): RGB[] => w.split('/').map((c) => this.color(c));
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
      case 'backcolor': this.backcolor = this.color(a[k + 1]); return 2;
      case 'backcolor2': this.backcolor2 = a[k + 1] === 'none' ? null : this.color(a[k + 1]); return 2;
      case 'boxcolor': this.boxcolor = this.color(a[k + 1]); return 2;
      case 'color': {
        const name = a[k + 1];
        const c: RGB = [this.num(a[k + 2], 'color'), this.num(a[k + 3], 'color'), this.num(a[k + 4], 'color')];
        this.userColors.set(name, [Math.round(c[0] * 255), Math.round(c[1] * 255), Math.round(c[2] * 255)]);
        return 5;
      }
      case 'amap': {
        const { map, used } = ColorMap.parse(a.slice(k + 1), (c) => this.userColors.get(c) ?? NAMED_COLORS[c] ?? null);
        this.amap = map;
        return used + 1;
      }
      case 'fcolor': {
        // dump_modify fcolor: "fcolor ID color" sets the colour of the objects of a dump image "fix" keyword that
        // use the const style (the colour is only read for that style, as in native dump_image.cpp)
        const f = this.fixObjects.find((x) => x.id === a[k + 1]);
        if (!f) throw new StyleError(`dump ${this.id}: dump_modify fcolor: fix ID '${a[k + 1] ?? ''}' is not a dump image fix`);
        f.rgb = this.color(a[k + 2] ?? '');
        return 3;
      }
      case 'gmap': case 'bitrate': case 'framerate': case 'region': case 'thresh':
        throw new StyleError(`dump ${this.id}: dump_modify keyword '${key}' is not supported for dump image by the browser engine`);
      default:
        return 0;
    }
  }

  private typeColor(s: SimState, i: number, colorVals: Float64Array | null, cmap: ColorMap | null): RGB {
    const type = s.type[i];
    if (this.attr.color === 'type') {
      const custom = this.acolorByType.get(type);
      return custom ?? TYPE_COLORS[(type - 1) % TYPE_COLORS.length];
    }
    if (this.attr.color === 'element') {
      const el = this.elementName(s, type);
      return (ELEMENTS[el] ?? ELEMENTS.C)[0];
    }
    return cmap!.color(colorVals![i]);
  }

  /**
   * The colour map of an atom colour value: dump_modify amap, or the dump_image.html default
   * "amap = min max cf 0.0 2 min blue max red" (blue at the group minimum, red at the group maximum).
   * The range is the group's minimum and maximum of the values (ColorMap.bounds).
   */
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
    if (this.attr.diameter === 'element') {
      const el = this.elementName(s, type);
      return (ELEMENTS[el] ?? ELEMENTS.C)[1];
    }
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
    const lx = box.hi[0] - box.lo[0], ly = box.hi[1] - box.lo[1], lz = box.hi[2] - box.lo[2];
    const maxdim = Math.max(lx, ly, lz, 1e-12);
    const scale = (h / (2 * maxdim)) * this.zoom;

    // box centre: fractions of the box dimensions (dump_image.html center keyword)
    const center: number[] = [0, 1, 2].map((d) => box.lo[d] + this.centerFrac[d] * (box.hi[d] - box.lo[d]));

    const th = this.theta * DEG, ph = this.phi * DEG;
    const V = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
    let ui = cross(V, cross(this.up, V));
    if (Math.hypot(ui[0], ui[1], ui[2]) < 1e-9) ui = this.dimension2Fallback(V);
    ui = normalize(ui);
    const right = normalize(cross(ui, V));
    const cx0 = Math.floor(w / 2);
    const cy0 = h % 2 === 0 ? h / 2 - 1.5 : (h - 1) / 2;
    const project = (p: number[]): [number, number, number] => {
      const dx = p[0] - center[0], dy = p[1] - center[1], dz = p[2] - center[2];
      const sx = cx0 + scale * (dx * right[0] + dy * right[1] + dz * right[2]);
      const sy = cy0 - scale * (dx * ui[0] + dy * ui[1] + dz * ui[2]);
      const dep = dx * V[0] + dy * V[1] + dz * V[2];
      return [sx, sy, dep];
    };

    // Image::view_params (orthographic): pixelWidth = maxdel / (height * zoom) with maxdel = 2 * longest edge, and
    // zdist = (maxdel / tan(30 deg) + 0.5 * del . camDir) / zoom with del = 2 * edges (image.cpp at b332f28)
    const maxdel = 2 * maxdim;
    const zdist = (maxdel / Math.tan(30 * DEG) + 0.5 * (2 * lx * V[0] + 2 * ly * V[1] + 2 * lz * V[2])) / this.zoom;
    const frame: RasterFrame = {
      w, h, rgb, depth,
      camRight: [right[0], right[1], right[2]], camUp: [ui[0], ui[1], ui[2]], camDir: [V[0], V[1], V[2]],
      ctr: [center[0], center[1], center[2]], zdist, pixelWidth: maxdel / (h * this.zoom),
      // dump image shiny; with ssao the ambient colour is 0.5 and the tangents of the surfaces are kept (view_params)
      shiny: this.shiny,
      ambient: this.ssao ? SSAO_AMBIENT : 0,
      surface: this.ssao ? new Float64Array(2 * w * h) : undefined,
    };
    const ctx: DrawCtx = { rgb, depth, project, scale, frame, shading: gplHook<ImageShading>('image') };

    // per-atom colour/diameter attributes (computed once per snapshot)
    const colorVals = this.attr.color !== 'type' && this.attr.color !== 'element' ? this.attributeValues(this.attr.color, s) : null;
    const cmap = this.atomColorMap(colorVals, s);
    const diamVals = this.attr.diameter !== 'type' && this.attr.diameter !== 'element' ? this.attributeValues(this.attr.diameter, s) : null;

    if (this.boxOn) this.drawBox(s, rgb, depth, project, scale, this.boxDiam);
    if (this.bond) this.drawBonds(s, rgb, depth, project, scale, diamVals);

    if (this.atomOn) {
      for (let i = 0; i < s.n; i++) {
        if (!(s.mask[i] & this.sys.groupBit(this.group))) continue;
        const diam = this.diameter(s, i, diamVals);
        if (!(diam > 0)) continue;
        const [px, py, pd] = project([s.x[3 * i], s.x[3 * i + 1], s.x[3 * i + 2]]);
        const r = 0.5 * diam * scale;
        if (r < 0.5) continue;
        const base = this.typeColor(s, i, colorVals, cmap);
        this.drawSphere(ctx, px, py, pd, r, base);
      }
    }
    if (this.fixObjects.length) this.drawFixObjects(s, ctx);
    if (this.axes) {
      // the origin is grey, the x, y and z arrows red, green and blue (dump_image.cpp, Image::draw_axes)
      const [origin, ...arrows] = this.axesObjects(s);
      this.drawImageObject(origin, [128, 128, 128], 0, 0, ctx);
      arrows.forEach((o, d) => this.drawImageObject(o, AXIS_COLORS[d], 0, 0, ctx));
    }
    // Image::compute_SSAO runs on the finished image, before the fsaa reduction (Image::merge)
    if (this.ssao) ctx.shading.applySsao(frame, maxdel * 0.05 * this.ssaoInt, Math.trunc(8 + 32 * this.ssaoInt), this.ssaoSeed);
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
      const fix = this.sys.fixes.find((f) => f.id === fo.id);
      const objs = fix?.imageObjects?.();
      if (!objs) throw new StyleError(`dump ${this.id}: dump image fix ${fo.id} provides no graphics objects`);
      for (const o of objs) {
        // objects drawn by the GPL renderer get the keyword's colour, diameter and flag rules through this style
        if (o.kind === 'draw') {
          o.draw(ctx.frame, this.objectStyle(s, fo));
          continue;
        }
        const itype = Math.trunc(o.type - 1) % s.ntypes + 1;
        let base: RGB = [255, 255, 255];
        if (fo.style === 'const') base = fo.rgb;
        else if (itype > 0 && fo.style === 'type') base = this.acolorByType.get(itype) ?? TYPE_COLORS[(itype - 1) % TYPE_COLORS.length];
        else if (itype > 0) base = (ELEMENTS[this.elementName(s, itype)] ?? ELEMENTS.C)[0];
        this.drawImageObject(o, base, fo.flag1, fo.flag2, ctx);
      }
    }
  }

  /** The style of one "fix" keyword for the GPL graphics objects (ObjectStyle in image_objects.ts). */
  private objectStyle(s: SimState, fo: { style: 'type' | 'element' | 'const'; flag1: number; flag2: number; rgb: RGB }): ObjectStyle {
    const colorOf = (type: number): Rgb255 => {
      const itype = Math.trunc(type - 1) % s.ntypes + 1;
      if (fo.style === 'const') return [fo.rgb[0], fo.rgb[1], fo.rgb[2]];
      if (itype <= 0) return [255, 255, 255];
      if (fo.style === 'type') return this.acolorByType.get(itype) ?? TYPE_COLORS[(itype - 1) % TYPE_COLORS.length];
      return (ELEMENTS[this.elementName(s, itype)] ?? ELEMENTS.C)[0];
    };
    // default sphere size of a type: the dump's adiam setting (numeric, per type, or per element)
    const diameterOf = (type: number): number => {
      if (this.adiamAll !== null) return this.adiamAll;
      if (this.attr.diameter === 'type') return this.adiamByType.get(type) ?? 1.0;
      if (this.attr.diameter === 'element') return (ELEMENTS[this.elementName(s, type)] ?? ELEMENTS.C)[1];
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
    return { flag1: fo.flag1, flag2: fo.flag2, colorOf, diameterOf, bondDiameter };
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
      const d = o.diameter + flag2;
      if (!(d > 0)) return;
      const parts = arrowParts({ center: o.center, dir: o.dir, length: o.length + flag1, diameter: d, ratio: o.ratio });
      this.drawStroke(rgb, depth, project(parts.shaft.p1), project(parts.shaft.p2), base, Math.max(1, d * scale));
      this.drawCone(rgb, depth, project, parts.head.p1, parts.head.p2, parts.head.d1, parts.head.d2, scale, base);
    }
  }

  /**
   * The axes of dump image "axes" (dump_image.cpp, DumpImage::create_image): the origin is the lower corner of the box
   * moved by the offset (10% of the longest edge, half of it for center), the arrows reach axes-length times the box
   * edges, and the diameter is axes-diameter times the shortest edge. The origin is a grey sphere (its radius is passed
   * as the diameter, as native does), the x, y and z arrows are red, green and blue (z only in 3d). The letters are
   * not drawn (the log line in the parser). Triclinic boxes are refused.
   */
  private axesObjects(s: SimState): ImageObject[] {
    const b = s.box;
    if (b.tilt.some((t) => t !== 0)) throw new StyleError(`dump ${this.id}: dump image axes for a triclinic box are not supported by the browser engine`);
    const dim = s.dimension === 2 ? 2 : 3;
    const edges = [0, 1, 2].slice(0, dim).map((d) => b.hi[d] - b.lo[d]);
    const axes = this.axes!;
    const diameter = Math.min(...edges) * axes.diam;
    const offset = Math.max(...edges) * (axes.loc === 'center' ? 0.5 : 0.1);
    const m = AXES_SHIFT[dim][axes.loc];
    const origin: Vec3 = [b.lo[0] + m[0] * offset, b.lo[1] + m[1] * offset, b.lo[2] + m[2] * offset];
    const objs: ImageObject[] = [{ kind: 'sphere', type: 0, center: origin, diameter: 0.5 * diameter }];
    const colour = [0, 1, 2];
    for (const d of colour.slice(0, dim)) {
      const tip: Vec3 = [origin[0], origin[1], origin[2]];
      tip[d] += axes.len * edges[d];
      const dir: Vec3 = [tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2]];
      const center: Vec3 = [0.5 * (origin[0] + tip[0]), 0.5 * (origin[1] + tip[1]), 0.5 * (origin[2] + tip[2])];
      objs.push({ kind: 'arrow', type: d, center, dir, length: axes.len * edges[d], diameter, ratio: 0.2 });
    }
    return objs;
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

  private dimension2Fallback(V: number[]): number[] {
    // When up is parallel to the view (e.g. 3d view 0 0 with the default up), pick any perpendicular.
    const alt = Math.abs(V[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
    return cross(V, cross(alt, V));
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
        const idx = y * w + x;
        const nz = Math.sqrt(1 - q);
        // camera space: right = dx / r, up = -dy / r (rows grow downwards), towards the viewer = nz
        const normal: [number, number, number] = [dx / r, -dy / r, nz];
        const surfDepth = pd + r * nz;
        if (surfDepth <= depth[idx]) continue;
        depth[idx] = surfDepth;
        if (frame.surface) {
          frame.surface[2 * idx] = normal[1];
          frame.surface[2 * idx + 1] = -normal[0];
        }
        const sh = shading.shadeSurface(normal, col, frame.shiny ?? 1, frame.ambient ?? 0);
        const o = idx * 3;
        rgb[o] = clamp(sh[0] * 255);
        rgb[o + 1] = clamp(sh[1] * 255);
        rgb[o + 2] = clamp(sh[2] * 255);
      }
    }
  }

  private drawBox(s: SimState, rgb: Uint8Array, depth: Float64Array, project: (p: number[]) => [number, number, number], scale: number, diam: number): void {
    const b = s.box;
    const corners: number[][] = [];
    const dim = s.dimension;
    // the 8 corners (4 in 2d) of the (possibly tilted) box: each fractional coordinate is 0 or 1
    const kz = dim === 2 ? [0] : [0, 1];
    const idx = (i: number, j: number, k: number) => [b.lo[0] + i * (b.hi[0] - b.lo[0]) + j * b.tilt[0] + k * b.tilt[1], b.lo[1] + j * (b.hi[1] - b.lo[1]) + k * b.tilt[2], b.lo[2] + k * (b.hi[2] - b.lo[2])];
    for (const i of [0, 1]) for (const j of [0, 1]) for (const k of kz) corners.push(idx(i, j, k));
    const width = Math.max(1, diam * Math.min(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1]) * scale);
    const edges: [number, number][] = [];
    // connect corners differing in exactly one axis
    for (let a = 0; a < corners.length; a++) for (let c = a + 1; c < corners.length; c++) {
      let d = 0;
      for (let t = 0; t < 3; t++) if (Math.abs(corners[a][t] - corners[c][t]) > 1e-9) d++;
      if (d === 1) edges.push([a, c]);
    }
    const seen = new Set<string>();
    for (const [a, c] of edges) {
      const key = `${Math.min(a, c)}-${Math.max(a, c)}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const p1 = project(corners[a]), p2 = project(corners[c]);
      this.drawLine(rgb, depth, p1, p2, this.boxcolor, width);
    }
  }

  private drawBonds(s: SimState, rgb: Uint8Array, depth: Float64Array, project: (p: number[]) => [number, number, number], scale: number, diamVals: Float64Array | null): void {
    const bonds = s.topo.bonds;
    if (!bonds || bonds.n === 0) return;
    const map = buildAtomMap(s);
    const bond = this.bond!;
    const colorVals = this.attr.color !== 'type' && this.attr.color !== 'element' ? this.attributeValues(this.attr.color, s) : null;
    const cmap = this.atomColorMap(colorVals, s);
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
      const col = bond.color === 'type' ? (this.bcolorByType.get(bonds.type[bi]) ?? TYPE_COLORS[(bonds.type[bi] - 1) % TYPE_COLORS.length]) : this.typeColor(s, i1, colorVals, cmap);
      const p1 = project([s.x[3 * i1], s.x[3 * i1 + 1], s.x[3 * i1 + 2]]);
      const p2 = project([s.x[3 * i2], s.x[3 * i2 + 1], s.x[3 * i2 + 2]]);
      this.drawLine(rgb, depth, p1, p2, col, Math.max(1, width * scale));
    }
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
        const o = (yy * w + xx) * 3;
        if (d <= depth[yy * w + xx]) continue;
        depth[yy * w + xx] = d;
        rgb[o] = color[0]; rgb[o + 1] = color[1]; rgb[o + 2] = color[2];
      }
    }
  }
}

const clamp = (v: number): number => (v < 0 ? 0 : v > 255 ? 255 : v | 0);
