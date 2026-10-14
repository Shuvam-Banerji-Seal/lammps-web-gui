/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/fix_graphics_labels.cpp
 * (FixGraphicsLabels: keywords image, text, colorscale; the colour names and r/g/b parsing of get_color(); the
 * size rule of text labels (2x rendering for anti-aliasing, scale 0.5, sizes above 128 rendered at 128 and scaled);
 * end_of_step() builds one PIXMAP per image or text label, in the order image, text). LAMMPS is Copyright (2003)
 * Sandia Corporation and distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0
 * too (src/engine-gpl/LICENSE).
 *
 * Text labels are vector strokes (see scalable_font.ts) instead of the bitmap of the native renderer. Image labels
 * take their RGB rows from the caller (readImage), since file decoding is not part of the engine. The colorscale
 * keyword needs the colour map of a dump image, which this engine does not have yet, so it is refused by name.
 * Variables (v_name) and $-substitution are supplied by the caller through LabelEnv.
 */
import { ScalableFont, createLabel, type TextLayout } from './scalable_font';
import type { GraphicsObject, LabelObject, PixmapObject, Rgb, Vec3 } from './objects';

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

export type LabelSpec = ImageLabelSpec | TextLabelSpec;

/** What the labels need from the engine at each snapshot. */
export interface LabelEnv {
  variable(name: string): number;
  /** Replace $-references in label text, as Input::substitute does. */
  substitute(text: string): string;
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
      throw new Error('fix graphics/labels colorscale is not supported: it needs the colour map of a dump image');
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
      const layout: TextLayout = createLabel(this.font, text, s.horizontal);
      const lab: LabelObject = {
        kind: 'label',
        pos,
        width: layout.width,
        height: layout.height,
        scale,
        transcolor: s.transcolor,
        fontcolor: s.fontcolor,
        backcolor: s.backcolor,
        framecolor: s.framecolor,
        frame: layout.frame,
        strokes: layout.strokes,
      };
      out.push(lab);
    }
    return out;
  }
}
