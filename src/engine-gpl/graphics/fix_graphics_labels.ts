/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_labels.cpp
 * (FixGraphicsLabels: keywords image, text, colorscale; the colour names and r/g/b parsing of get_color(); the
 * size rule of text labels (2x rendering for anti-aliasing, scale 0.5, sizes above 128 rendered at 128 and scaled);
 * end_of_step() builds one PIXMAP per image, text or colorscale object, in the order image, text, colorscale). LAMMPS is
 * Copyright (2003) Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this
 * file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Text labels and colour scales are the create_label() and create_colorscale() pixmaps of the native renderer
 * (scalable_font.ts). Image labels take their RGB rows from the caller (readImage: PNG files through
 * engine/output/png_decode.ts). The colorscale keyword reads the atom colour map of its dump image as the previous
 * output left it (Dump.colorScaleMap); the grid and bond maps are refused by name. Variables (v_name) and
 * $-substitution are supplied by the caller through LabelEnv.
 */
import { ScalableFont, createLabel, createColorscale, type ColorScaleMap } from './scalable_font';
import type { GraphicsObject, PixmapObject, Rgb, Vec3 } from './objects';
import { Fix } from '../../engine/fix/fix';
import type { System } from '../../engine/system';
import type { FixFactory } from '../../engine/styles';
import { StyleError } from '../../engine/force/types';
import type { ImageObject } from '../../engine/output/image_objects';
import { parseInteger } from './common';
import { drawGraphicsObject } from './draw';
import { decodePng } from '../../engine/output/png_decode';
import type { ColorMap } from '../../engine/output/image_colormap';

/** A number given directly or a reference to an equal-style variable. */
export type LabelValue = number | { variable: string };

export interface ImageData {
  width: number;
  height: number;
  /** RGB rows, 3 bytes per pixel, bottom row first (the native pixmap layout). */
  rgb: Uint8Array;
}

export interface ImageLabelSpec {
  kind: 'image';
  file: string;
  pos: [LabelValue, LabelValue, LabelValue];
  scale: LabelValue;
  /** Transparent colour; null for none. "auto" is resolved to the first pixel at parse time. */
  transcolor: Rgb | null;
  data: ImageData;
}

export interface TextLabelSpec {
  kind: 'text';
  text: string;
  pos: [LabelValue, LabelValue, LabelValue];
  /** Font size in pixels as the native renderer gets it (after the 2x rule). */
  size: LabelValue;
  /** Display scale of the rendered label (0.5 for sizes up to 128). */
  scale: number;
  fontcolor: Rgb;
  backcolor: Rgb;
  framecolor: Rgb;
  transcolor: Rgb | null;
  horizontal: boolean;
}

/** A colour scale of a dump image's atom colour map (fix graphics/labels colorscale). */
export interface ScaleLabelSpec {
  kind: 'scale';
  /** The dump image whose colour map is drawn. */
  dumpid: string;
  /** Which colour map of the dump is drawn: the atom map or the bond map (map keyword). */
  map: 'atom' | 'bond';
  text: string;
  pos: [LabelValue, LabelValue, LabelValue];
  /** Font size in pixels as the native renderer gets it (after the 2x rule). */
  size: LabelValue;
  scale: number;
  fontcolor: Rgb;
  backcolor: Rgb;
  framecolor: Rgb;
  transcolor: Rgb | null;
  horizontal: boolean;
  /** Bar length in pixels (2 * the length keyword, as the native code stores it); 0 for the text width. */
  length: number;
  tics: number;
}

export type LabelSpec = ImageLabelSpec | TextLabelSpec | ScaleLabelSpec;

/** What the labels need from the engine at each snapshot. */
export interface LabelEnv {
  variable(name: string): number;
  /** Replace $-references in label text, as Input::substitute does. */
  substitute(text: string): string;
  /** The colour map of a dump image as the previous output left it (ColorMap::info, value2color). */
  colorScale?(dumpid: string, map: 'atom' | 'bond'): ColorMap;
}

const COLOR_NAMES: Record<string, Rgb> = {
  white: [255, 255, 255],
  black: [0, 0, 0],
  silver: [192, 192, 192],
  darkgray: [64, 64, 64],
};

/** get_color(): a colour name, or "r/g/b" with each value in 0..255. */
export function parseColor(text: string): Rgb {
  if (text in COLOR_NAMES) return [...COLOR_NAMES[text]] as Rgb;
  const parts = text.split('/');
  if (parts.length !== 3) throw new Error(`Error parsing RGB color value ${text}: Invalid RGB value`);
  const rgb = parts.map((p) => {
    if (!/^[+-]?\d+$/.test(p)) throw new Error(`Error parsing RGB color value ${text}: Invalid RGB value`);
    const v = Number(p);
    if (v < 0 || v > 255) throw new Error(`Error parsing RGB color value ${text}: Invalid RGB value`);
    return v;
  });
  return [rgb[0], rgb[1], rgb[2]];
}

/** An integer keyword value (utils::inumeric). */
function parseIntegerValue(text: string, what: string): number {
  if (!/^[+-]?\d+$/.test(text)) throw new Error(`Expected integer parameter instead of '${text}' in ${what}`);
  return Number(text);
}

/** A value is "v_name" for a variable, otherwise a number. */
function parseValue(text: string): LabelValue {
  if (text.startsWith('v_')) return { variable: text.slice(2) };
  const v = Number(text);
  if (!Number.isFinite(v)) throw new Error(`Expected floating point parameter instead of '${text}'`);
  return v;
}

function valueOf(v: LabelValue, env: LabelEnv): number {
  return typeof v === 'number' ? v : env.variable(v.variable);
}

/** The text size rule applied at parse time (a constant size). */
function constantTextSize(size: number): { size: number; scale: number } {
  const s = size * 2.0;
  if (s < 8.0 || s > 1024.0) throw new Error(`Invalid fix graphics/labels text size value: ${s * 0.5}`);
  if (s > 128.0) return { size: 128.0, scale: s / 256.0 };
  return { size: s, scale: 0.5 };
}

/**
 * Parse the arguments after "fix graphics/labels nevery". `readImage` decodes an image file into RGB rows (bottom row
 * first); it is called once per image keyword.
 */
export function parseLabelArgs(args: string[], readImage: (file: string) => ImageData): LabelSpec[] {
  const specs: LabelSpec[] = [];
  let i = 0;
  const need = (n: number, what: string) => {
    if (i + n > args.length) throw new Error(`Illegal ${what}: missing arguments`);
  };
  while (i < args.length) {
    const key = args[i];
    if (key === 'image') {
      need(5, 'fix graphics/labels image');
      const file = args[i + 1];
      const data = readImage(file);
      const pos: [LabelValue, LabelValue, LabelValue] = [parseValue(args[i + 2]), parseValue(args[i + 3]), parseValue(args[i + 4])];
      i += 5;
      const img: ImageLabelSpec = { kind: 'image', file, pos, scale: 1.0, transcolor: null, data };
      while (i < args.length && !['image', 'text', 'colorscale'].includes(args[i])) {
        if (args[i] === 'scale') {
          need(2, 'fix graphics/labels image scale');
          img.scale = parseValue(args[i + 1]);
          i += 2;
        } else if (args[i] === 'transcolor') {
          need(2, 'fix graphics/labels image transcolor');
          const v = args[i + 1];
          if (v === 'auto') img.transcolor = [data.rgb[0], data.rgb[1], data.rgb[2]];
          else if (v === 'none') img.transcolor = null;
          else img.transcolor = parseColor(v);
          i += 2;
        } else {
          throw new Error(`Unknown fix graphics/labels image keyword: ${args[i]}`);
        }
      }
      specs.push(img);
    } else if (key === 'text') {
      need(5, 'fix graphics/labels text');
      const pos: [LabelValue, LabelValue, LabelValue] = [parseValue(args[i + 2]), parseValue(args[i + 3]), parseValue(args[i + 4])];
      const txt: TextLabelSpec = {
        kind: 'text', text: args[i + 1], pos,
        size: 48.0, scale: 0.5,
        fontcolor: [255, 255, 255], backcolor: [192, 192, 192], framecolor: [192, 192, 192],
        transcolor: [192, 192, 192], horizontal: true,
      };
      i += 5;
      while (i < args.length && !['image', 'text', 'colorscale'].includes(args[i])) {
        const k = args[i];
        if (k === 'size') {
          need(2, 'fix graphics/labels text size');
          const v = parseValue(args[i + 1]);
          if (typeof v === 'number') {
            const c = constantTextSize(v);
            txt.size = c.size;
            txt.scale = c.scale;
          } else {
            txt.size = v;
          }
          i += 2;
        } else if (k === 'fontcolor') {
          need(2, 'fix graphics/labels text fontcolor');
          txt.fontcolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'backcolor') {
          need(2, 'fix graphics/labels text backcolor');
          txt.backcolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'framecolor') {
          need(2, 'fix graphics/labels text framecolor');
          txt.framecolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'transcolor') {
          need(2, 'fix graphics/labels text transcolor');
          txt.transcolor = args[i + 1] === 'none' ? null : parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'horizontal' || k === 'vertical') {
          txt.horizontal = k === 'horizontal';
          i += 1;
        } else {
          throw new Error(`Unknown fix graphics/labels text keyword: ${k}`);
        }
      }
      specs.push(txt);
    } else if (key === 'colorscale') {
      need(6, 'fix graphics/labels colorscale');
      const pos: [LabelValue, LabelValue, LabelValue] = [parseValue(args[i + 3]), parseValue(args[i + 4]), parseValue(args[i + 5])];
      const sc: ScaleLabelSpec = {
        kind: 'scale', dumpid: args[i + 1], text: args[i + 2], pos, map: 'atom',
        size: 48.0, scale: 0.5,
        fontcolor: [255, 255, 255], backcolor: [192, 192, 192], framecolor: [192, 192, 192],
        transcolor: [192, 192, 192], horizontal: true, length: 0, tics: 0,
      };
      i += 6;
      while (i < args.length && !['image', 'text', 'colorscale'].includes(args[i])) {
        const k = args[i];
        if (k === 'size') {
          need(2, 'fix graphics/labels colorscale size');
          const v = parseValue(args[i + 1]);
          if (typeof v === 'number') {
            const c = constantTextSize(v);
            sc.size = c.size;
            sc.scale = c.scale;
          } else {
            sc.size = v;
          }
          i += 2;
        } else if (k === 'length') {
          need(2, 'fix graphics/labels colorscale length');
          sc.length = 2 * parseIntegerValue(args[i + 1], 'fix graphics/labels colorscale length');
          i += 2;
        } else if (k === 'tics') {
          need(2, 'fix graphics/labels colorscale tics');
          sc.tics = parseIntegerValue(args[i + 1], 'fix graphics/labels colorscale tics');
          if (sc.tics < 0) throw new Error('Invalid tics value');
          i += 2;
        } else if (k === 'map') {
          need(2, 'fix graphics/labels colorscale map');
          // the atom and bond colour maps are drawn; the grid map is not (the browser engine has no grid colouring)
          if (args[i + 1] !== 'atom' && args[i + 1] !== 'bond') throw new Error(`fix graphics/labels colorscale map ${args[i + 1]} is not supported by the browser engine (only atom and bond)`);
          sc.map = args[i + 1] === 'bond' ? 'bond' : 'atom';
          i += 2;
        } else if (k === 'fontcolor') {
          need(2, 'fix graphics/labels colorscale fontcolor');
          sc.fontcolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'backcolor') {
          need(2, 'fix graphics/labels colorscale backcolor');
          sc.backcolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'framecolor') {
          need(2, 'fix graphics/labels colorscale framecolor');
          sc.framecolor = parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'transcolor') {
          need(2, 'fix graphics/labels colorscale transcolor');
          sc.transcolor = args[i + 1] === 'none' ? null : parseColor(args[i + 1]);
          i += 2;
        } else if (k === 'horizontal' || k === 'vertical') {
          sc.horizontal = k === 'horizontal';
          i += 1;
        } else {
          throw new Error(`Unknown fix graphics/labels colorscale keyword: ${k}`);
        }
      }
      specs.push(sc);
    } else {
      throw new Error(`Unknown fix graphics/labels keyword: ${key}`);
    }
  }
  return specs;
}

/** The label objects of one snapshot, images first then texts (end_of_step order). */
export class FixGraphicsLabels {
  private readonly font = new ScalableFont();

  constructor(private readonly specs: LabelSpec[]) {}

  objects(env: LabelEnv): GraphicsObject[] {
    const out: GraphicsObject[] = [];
    for (const s of this.specs) {
      if (s.kind === 'image') {
        const pix: PixmapObject = {
          kind: 'pixmap',
          pos: [valueOf(s.pos[0], env), valueOf(s.pos[1], env), valueOf(s.pos[2], env)],
          width: s.data.width,
          height: s.data.height,
          scale: valueOf(s.scale, env),
          transcolor: s.transcolor,
          rgb: s.data.rgb,
        };
        out.push(pix);
      }
    }
    for (const s of this.specs) {
      if (s.kind !== 'text') continue;
      const pos: Vec3 = [valueOf(s.pos[0], env), valueOf(s.pos[1], env), valueOf(s.pos[2], env)];
      let size = 0;
      let scale = s.scale;
      if (typeof s.size === 'number') {
        size = s.size;
      } else {
        // variable size: 2x rendering, clamped to the renderable range
        const v = 2.0 * env.variable(s.size.variable);
        if (v > 128.0) {
          scale = v / 256.0;
          size = 128.0;
        } else {
          size = Math.max(v, 8.0);
          scale = 0.5;
        }
      }
      const text = s.text.includes('$') ? env.substitute(s.text) : s.text;
      this.font.selectFont(Math.trunc(size));
      const layout = createLabel(this.font, text, s.horizontal, { font: s.fontcolor, frame: s.framecolor, back: s.backcolor });
      const lab: PixmapObject = {
        kind: 'pixmap',
        pos,
        width: layout.width,
        height: layout.height,
        scale,
        transcolor: s.transcolor,
        rgb: layout.rgb,
      };
      out.push(lab);
    }
    for (const s of this.specs) {
      if (s.kind !== 'scale') continue;
      const pos: Vec3 = [valueOf(s.pos[0], env), valueOf(s.pos[1], env), valueOf(s.pos[2], env)];
      let size = 0;
      let scale = s.scale;
      if (typeof s.size === 'number') {
        size = s.size;
      } else {
        const v = 2.0 * env.variable(s.size.variable);
        if (v > 128.0) {
          scale = v / 256.0;
          size = 128.0;
        } else {
          size = Math.max(v, 8.0);
          scale = 0.5;
        }
      }
      if (!env.colorScale) throw new Error(`fix graphics/labels colorscale ${s.dumpid}: no dump image colour map`);
      const map: ColorScaleMap = env.colorScale(s.dumpid, s.map);
      const text = s.text.includes('$') ? env.substitute(s.text) : s.text;
      this.font.selectFont(Math.trunc(size));
      const layout = createColorscale(this.font, text, s.horizontal, s.length, map, s.tics,
        { font: s.fontcolor, frame: s.framecolor, back: s.backcolor });
      const bar: PixmapObject = { kind: 'pixmap', pos, width: layout.width, height: layout.height, scale, transcolor: s.transcolor, rgb: layout.rgb };
      out.push(bar);
    }
    return out;
  }
}

/** The fix: "fix ID group-ID graphics/labels N keyword values ..." (fix_graphics_labels.cpp constructor). */
export class FixGraphicsLabelsFix extends Fix {
  readonly style = 'graphics/labels';
  private readonly labels: FixGraphicsLabels;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    if (args.length < 1) throw new StyleError(`usage: fix ${id} group-ID graphics/labels N keyword values ...`);
    this.nevery = parseInteger(args[0], `fix ${id} graphics/labels N`);
    if (this.nevery <= 0) throw new StyleError(`fix ${id} graphics/labels: N must be > 0`);
    // fix graphics/labels image: the file is read from the engine's file store (a PNG; the RGB rows are kept, the
    // alpha channel is dropped), stored bottom row first as the native pixmap is
    const readImage = (file: string): ImageData => {
      if (!/\.png$/i.test(file)) throw new StyleError(`fix ${id} graphics/labels image ${file}: only PNG image files are read by the browser engine`);
      const png = decodePng(sys.readFile(file));
      const rgb = new Uint8Array(png.width * png.height * 3);
      for (let y = 0; y < png.height; y++) {
        const dst = (png.height - 1 - y) * png.width * 3;
        for (let x = 0; x < png.width; x++) {
          rgb[dst + 3 * x] = png.rgba[4 * (y * png.width + x)];
          rgb[dst + 3 * x + 1] = png.rgba[4 * (y * png.width + x) + 1];
          rgb[dst + 3 * x + 2] = png.rgba[4 * (y * png.width + x) + 2];
        }
      }
      return { width: png.width, height: png.height, rgb };
    };
    let specs: LabelSpec[];
    try {
      specs = parseLabelArgs(args.slice(1), readImage);
    } catch (e) {
      if (e instanceof StyleError) throw e;
      throw new StyleError(`fix ${id} graphics/labels: ${(e as Error).message}`);
    }
    if (specs.length === 0) throw new StyleError(`fix ${id} graphics/labels needs at least one image or text label`);
    this.labels = new FixGraphicsLabels(specs);
  }

  /** The label pixmaps of the snapshot (end_of_step: images first, then texts). */
  imageObjects(): ImageObject[] {
    const sys = this.sys;
    const env: LabelEnv = {
      variable: (name) => sys.equalVariable(name),
      // Input::substitute on the label text: $x, ${name} and $(formula), with the variables of the run now
      substitute: (t) => sys.substituteText(t),
      colorScale: (dumpid, which) => {
        const d = sys.dumps.find((x) => x.id === dumpid) as unknown as { colorScaleMap?: (w: 'atom' | 'bond') => ColorMap | null } | undefined;
        const map = d?.colorScaleMap?.(which) ?? null;
        if (!map) throw new StyleError(`Dump ID ${dumpid} for colorscale not found or not dump style image`);
        return map;
      },
    };
    return this.labels.objects(env).map((o): ImageObject => ({ kind: 'draw', draw: (f, st) => drawGraphicsObject(f, st, o) }));
  }
}

export const makeFixGraphicsLabels: FixFactory = (sys, id, group, args) => new FixGraphicsLabelsFix(sys, id, group, args);
