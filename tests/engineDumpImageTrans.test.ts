import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image subbox and the transparencies of the box, subbox, axes and compute objects (dump_modify boxtrans,
 * subboxtrans, axestrans, ctrans), with the drawing order of DumpImage::create_image (src/GRAPHICS/dump_image.cpp at
 * b332f28: atoms, lines and triangles, ellipsoids, bodies, bonds, autobonds, objects, regions, subbox, box, axes), which
 * decides how transparent pieces blend (order_atrans: transparent atoms under bonds and a transparent box).
 *
 * compute chunk/atom bin/1d, bin/2d and bin/3d give the walls between their bins as objects (chunk_*: the GPL port of
 * ComputeChunkAtom::compute_image in src/engine-gpl/graphics/chunk_bins.ts).
 *
 * Fixtures (tests/fixtures/image_trans): <case>.in is the input and <case>.ppm is native's output:
 * /home/roy/.local/bin/lmp, b332f28, run 0, one process (the subbox is the box), written as PPM. 128x128 px.
 *
 * Pinned differences (count of bytes that differ, largest difference) against native. Every case is byte-exact except
 * chunk_2d_reduced: 3 bytes (one pixel) on the edge of a y wall. The same walls without the reduced units and bound
 * keyword differ in one column of 11 pixels at the edge where two walls meet in the image: the engine's corner of the
 * y = 2 wall projects to xf = 15.999999999999998 (camRight x = -0.49999999999999994, the sincos value native also uses)
 * and its raster stops one column short of native's. The cause is not identified yet; the values are pinned, not
 * loosened, so any change in them shows up here.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_trans');

/** The pinned [count, max] of each case against native. */
const EXPECTED: Record<string, { count: number; max: number }> = {
  subbox_a: { count: 0, max: 0 },
  subbox_trans: { count: 0, max: 0 },
  order_atrans: { count: 0, max: 0 },
  axestrans: { count: 0, max: 0 },
  ctrans: { count: 0, max: 0 },
  chunk_1d: { count: 0, max: 0 },
  chunk_2d_reduced: { count: 3, max: 141 },
  chunk_3d_bound: { count: 0, max: 0 },
  chunk_2d_dim2: { count: 0, max: 0 },
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
  expect(name, 'the input writes a PPM').toBeDefined();
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

describe('dump image subbox and transparencies against native LAMMPS', () => {
  for (const name of Object.keys(EXPECTED)) {
    it(`${name}: engine image vs native PPM`, async () => {
      expect(diff(await render(`${name}.in`), native(`${name}.ppm`))).toEqual(EXPECTED[name]);
    });
  }
});
