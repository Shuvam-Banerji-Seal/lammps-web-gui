import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image draws the atoms DumpCustom::count() chooses: the dump group, dump_modify region and dump_modify thresh
 * (src/GRAPHICS/dump_image.cpp at b332f28, create_image). The choice also bounds a dynamic atom colour map, and a bond
 * (explicit or autobond) is drawn only when both of its atoms are chosen.
 *
 * Fixtures (tests/fixtures/image_select): <case>.in is the input and <case>.ppm is native's output:
 * /home/roy/.local/bin/lmp, b332f28, run 0, written as PPM. Every case is 96x96 px and byte-exact.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_select');

const CASES = ['region_a', 'region_empty', 'thresh_color', 'thresh_bonds', 'thresh_autobond', 'thresh_ellipsoid'];

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

describe('dump image region and thresh against native LAMMPS', () => {
  for (const name of CASES) {
    it(`${name}: engine image vs native PPM`, async () => {
      expect(diff(await render(`${name}.in`), native(`${name}.ppm`))).toEqual({ count: 0, max: 0 });
    });
  }

  const BONDS = readFileSync(join(DIR, 'thresh_bonds.in'), 'utf8');

  it('stops when a bond colour compute gives more values than the bonds drawn', async () => {
    // Measured with native LAMMPS: ERROR on proc 0: Dump image bond compute bl produced 4 values but 2 bonds were
    // rendered; its group must select the same bonds as the dump (src/GRAPHICS/dump_image.cpp:1780)
    const input = BONDS.replace('bond atom 0.25', 'bond c_bl 0.25').replace('dump d all image', 'compute bl all bond/local dist\ndump d all image');
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    await expect(s.execute(input)).rejects.toThrow('Dump image bond compute bl produced 4 values but 2 bonds were rendered; its group must select the same bonds as the dump');
  });

  it('stops when no atom is chosen and the atom colour map is dynamic', async () => {
    // Measured with native LAMMPS: ERROR: Invalid atom color map min/max values (src/GRAPHICS/dump_image.cpp:1010)
    const input = readFileSync(join(DIR, 'thresh_color.in'), 'utf8').replace('thresh x > 1.5', 'thresh x > 100');
    const s = new Session({ emit: () => {}, writeFile: () => {} });
    await expect(s.execute(input)).rejects.toThrow(/colour map range is empty/);
  });

});
