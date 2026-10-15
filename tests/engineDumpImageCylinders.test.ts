import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image box, bonds, axes and triangle edges as shaded cylinders (Image::draw_cylinder, image.cpp at b332f28, through
 * the GPL drawCylinder in src/engine-gpl/graphics/draw.ts), the axes arrows (Image::draw_axes and ImageObjects::ArrowObj,
 * image_objects.cpp, in src/engine-gpl/graphics/axes.ts), the view setup of Image::view_params (EPSILON tilt, triclinic
 * bounds) and the axis letters, compared with native LAMMPS PPMs.
 *
 * Fixtures (tests/fixtures/image_cyl): <case>.in is the input (the dump line is the case's subject) and <case>.ppm is
 * native's output: /home/roy/.local/bin/lmp, b332f28, run 0, written as PPM. Every case is 64x64 to 200x200 px.
 *
 * Pinned differences (count of bytes that differ, largest difference) against native. Every case is byte-exact except
 * axes_b: 16 bytes, each by up to 47 levels, all in the blue (z) arrow's head on the image column that carries the
 * origin's projection (x = 80 of 160). The cause is not identified yet (the arrow head is draw_trinorm, image.cpp, and the
 * column is the sampling line of the axis). Those values are pinned, not loosened, so any change in them shows up here.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_cyl');

/** The pinned [count, max] of each case against native. */
const EXPECTED: Record<string, { count: number; max: number }> = {
  box_d1: { count: 0, max: 0 },
  box_d2: { count: 0, max: 0 },
  box_d3: { count: 0, max: 0 },
  box_tri: { count: 0, max: 0 },
  axes_a: { count: 0, max: 0 },
  axes_b: { count: 16, max: 47 },
  axes_c: { count: 0, max: 0 },
  axes_t: { count: 0, max: 0 },
  acolor_a: { count: 0, max: 0 },
  bond_a: { count: 0, max: 0 },
  bond_b: { count: 0, max: 0 },
  combo_a: { count: 0, max: 0 },
  combo_b: { count: 0, max: 0 },
  view0_atoms: { count: 0, max: 0 },
  tri_atoms: { count: 0, max: 0 },
};

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

/** Runs a case's input and returns the image it wrote. */
const render = async (file: string): Promise<Raster> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(readFileSync(join(DIR, file), 'utf8'));
  const name = [...files.keys()].find((k) => k.endsWith('.ppm'));
  expect(name, `${file} writes a PPM`).toBeDefined();
  return readPpm(decodeUrl(files.get(name!)!));
};

const native = (file: string): Raster => readPpm(new Uint8Array(readFileSync(join(DIR, file))));

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

describe('dump image cylinders, axes and view setup against native LAMMPS', () => {
  for (const name of Object.keys(EXPECTED)) {
    it(`${name}: engine image vs native PPM`, async () => {
      const d = diff(await render(`${name}.in`), native(`${name}.ppm`));
      expect(d).toEqual(EXPECTED[name]);
    });
  }
});
