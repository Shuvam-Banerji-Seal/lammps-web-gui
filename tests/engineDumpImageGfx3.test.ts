import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image graphics of the third batch against native LAMMPS PPMs: autobond (dump_image.cpp, the AUTO branch of
 * create_image, engine-gpl/graphics/autobond.ts), fix graphics/objects and fix graphics/arrows drawn as ArrowObj
 * (image_objects.cpp, axes.ts drawArrowObj), dump image bond coloured by a local compute (c_ID, bmap, dump_modify bmap
 * with the dynamic range of the values), fix graphics/labels colorscale map bond, and fsaa yes with a label pixmap
 * (image.cpp draw_pixmap, both overloads with the fsaa branches).
 *
 * Fixtures (tests/fixtures/image_gfx3): <case>.in is the input (the dump line is the case's subject) and <case>.ppm is
 * native's output: /home/roy/.local/bin/lmp, b332f28, built with the GRAPHICS package, 128 x 128 px. Each case writes
 * img.*.ppm; a case that writes two frames (colorscale_bond) is compared with its last frame. colalias sets a colour
 * name, uses it in acolor, then redefines it: native renders the redefined colour for the earlier use too (Image::addcolor
 * writes the stored array; color2rgb returns that pointer).
 *
 * Before this batch the engine differed from native on arrows_fix (1922 bytes, flat arrow strokes), fsaa_label (3084
 * bytes, label pixmap scale under fsaa), and autobond, bond_compute and colorscale_bond were refused. Every case is
 * byte-exact now; the expected counts are 0 and any change shows up here.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_gfx3');

const CASES = ['autobond', 'arrow_obj', 'arrows_fix', 'bond_compute', 'colorscale_bond', 'fsaa_label', 'colalias'];

interface Raster { w: number; h: number; rgb: Uint8Array }

/** Reads a binary P6 PPM (maxval 255; '#' comment lines allowed). */
const readPpm = (b: Uint8Array): Raster => {
  let i = 0;
  const fields: string[] = [];
  while (fields.length < 4) {
    while (i < b.length && /\s/.test(String.fromCharCode(b[i]))) i++;
    if (b[i] === 0x23) { while (b[i] !== 0x0a) i++; continue; }
    const s = i;
    while (i < b.length && !/\s/.test(String.fromCharCode(b[i]))) i++;
    fields.push(String.fromCharCode(...b.subarray(s, i)));
  }
  i++;
  expect(fields[0]).toBe('P6');
  const w = Number(fields[1]), h = Number(fields[2]);
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

const decodeUrl = (url: string): Uint8Array => {
  const bin = atob(url.slice(url.indexOf(',') + 1));
  const out = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) out[k] = bin.charCodeAt(k);
  return out;
};

/** Runs a case's input and returns its last image (a case may write two frames). */
const render = async (name: string): Promise<Raster> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(readFileSync(join(DIR, `${name}.in`), 'utf8'));
  const last = [...files.keys()].filter((k) => k.endsWith('.ppm')).pop();
  expect(last, `${name} writes a PPM`).toBeDefined();
  return readPpm(decodeUrl(files.get(last!)!));
};

const native = (name: string): Raster => readPpm(new Uint8Array(readFileSync(join(DIR, `${name}.ppm`))));

/** Bytes that differ from native, and the largest difference. */
const diff = (a: Raster, b: Raster): { count: number; max: number } => {
  expect(a.w).toBe(b.w);
  expect(a.h).toBe(b.h);
  let count = 0, max = 0;
  for (let k = 0; k < a.rgb.length; k++) {
    const d = Math.abs(a.rgb[k] - b.rgb[k]);
    if (d) { count++; if (d > max) max = d; }
  }
  return { count, max };
};

describe('dump image graphics batch 3 against native LAMMPS', () => {
  for (const name of CASES) {
    it(`${name}: engine image vs native PPM is byte-exact`, async () => {
      expect(diff(await render(name), native(name))).toEqual({ count: 0, max: 0 });
    });
  }
});
