import { describe, expect, it } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * fix graphics/labels text labels and colorscale bars in dump image output, compared byte for byte with native LAMMPS
 * (src/GRAPHICS/scalable_font.cpp create_label and create_colorscale, fix_graphics_labels.cpp end_of_step, dump_image.cpp
 * draw of PIXMAP objects).
 *
 * Fixtures (tests/fixtures/image_labels): <case>.in is the input (its dump line names the fix with "fix lab const 0 0"
 * or "fix cs const 0 0", which dump image needs to draw the fix's objects) and <case>.ppm is native's frame:
 * /home/roy/.local/bin/lmp, b332f28, run 0, written as PPM. Label cases: lab_s10 (size 10), lab_s16 (size 16, two words
 * and digits), lab_s24_col (size 24, font, back and frame colours), lab_s200 (size 200: rendered at 128 and scaled by
 * 200/256), lab_vert (vertical), lab_trans (back and frame colours, transcolor none). Colour scale cases: cs_def (the
 * default map, whose range on the first output is the default range of the native colour map), cs_amap (dump_modify
 * amap 0 3 cf: a custom map with a min and max), cs_tics (tics 3, size 16, length 120), cs_col (fontcolor, backcolor,
 * framecolor, transcolor none), cs_vert (vertical, size 10, length 60, tics 2).
 *
 * Before the create_label bitmap rasteriser was ported (the stroke rendering of the earlier engine), the counts of
 * differing bytes (count, largest difference) were: lab_s10 125 (250), lab_s16 560 (255), lab_s200 43604 (255),
 * lab_s24_col 212 (255), lab_trans 86 (215), lab_vert 6 (255). The colour scale was refused. Now every case is
 * byte-identical (0, 0) to native.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_labels');

interface Raster { w: number; h: number; rgb: Uint8Array }

/** Reads a binary P6 PPM (maxval 255; '#' comment lines allowed). */
const readPpm = (b: Uint8Array): Raster => {
  let i = 0;
  const fields: string[] = [];
  while (fields.length < 4) {
    while (i < b.length && /\s/.test(String.fromCharCode(b[i]))) i++;
    if (b[i] === 0x23) {
      while (b[i] !== 0x0a) i++;
      continue;
    }
    const s = i;
    while (i < b.length && !/\s/.test(String.fromCharCode(b[i]))) i++;
    fields.push(String.fromCharCode(...b.subarray(s, i)));
  }
  i++;
  expect(fields[0]).toBe('P6');
  const w = Number(fields[1]);
  const h = Number(fields[2]);
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

const decodeUrl = (url: string): Uint8Array => {
  const bin = atob(url.slice(url.indexOf(',') + 1));
  const out = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) out[k] = bin.charCodeAt(k);
  return out;
};

/** Runs a case's input and returns the frame it wrote. */
const render = async (file: string): Promise<Raster> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(readFileSync(join(DIR, file), 'utf8'));
  const name = [...files.keys()].find((k) => k.endsWith('.ppm'));
  expect(name, `${file} writes a PPM`).toBeDefined();
  return readPpm(decodeUrl(files.get(name!)!));
};

const diff = (a: Raster, b: Raster): { count: number; max: number } => {
  expect(a.w).toBe(b.w);
  expect(a.h).toBe(b.h);
  let count = 0;
  let max = 0;
  for (let k = 0; k < a.rgb.length; k++) {
    const d = Math.abs(a.rgb[k] - b.rgb[k]);
    if (d) {
      count++;
      if (d > max) max = d;
    }
  }
  return { count, max };
};

const CASES = readdirSync(DIR)
  .filter((f) => f.endsWith('.in'))
  .map((f) => f.slice(0, -3))
  .sort();

describe('fix graphics/labels text and colorscale pixels against native LAMMPS (dump image PPM)', () => {
  it('has the expected fixture set', () => {
    expect(CASES).toEqual(['cs_amap', 'cs_col', 'cs_def', 'cs_tics', 'cs_vert', 'lab_s10', 'lab_s16', 'lab_s200', 'lab_s24_col', 'lab_trans', 'lab_vert']);
  });

  for (const name of CASES) {
    it(`${name}: engine frame vs native PPM (0 bytes differ)`, async () => {
      const native = readPpm(new Uint8Array(readFileSync(join(DIR, `${name}.ppm`))));
      expect(diff(await render(`${name}.in`), native)).toEqual({ count: 0, max: 0 });
    });
  }
});
