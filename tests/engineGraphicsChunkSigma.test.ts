import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * fix graphics/chunk without a radius keyword (src/GRAPHICS/fix_graphics_chunk.cpp end_of_step): the radius of an atom
 * is half the pair style's sigma[type][type] ("sigma = (double **) force->pair->extract("sigma", dim)"), or the default
 * 0.5 * xlattice when there is no pair style. Fixtures (tests/fixtures/image_chunk): <case>.in and native's 128 x 128
 * img.0.ppm from /home/roy/.local/bin/lmp (b332f28, GRAPHICS package): chunk_sigma takes the lj/cut sigmas 1.6 and 0.8
 * of the two types, chunk_default has no pair style, chunk_flag0 draws no triangles (flag1 = 0 of the dump image fix
 * keyword). A pair style whose sigma the engine does not expose is refused. region_forestgreen draws a filled sphere region
 * in forestgreen (0.133, 0.545, 0.133): the shaded objects take their colour in 0..1 as native does, not rounded to 8 bits
 * first (which darkened the chunk hulls, whose type 2 default colour is forestgreen, by one level).
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_chunk');

const render = async (input: string): Promise<{ files: Map<string, string>; error?: string }> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  try {
    await s.execute(input);
  } catch (e) {
    return { files, error: e instanceof Error ? e.message : String(e) };
  }
  return { files };
};

const bytesOf = (url: string): Buffer => Buffer.from(url.slice(url.indexOf(',') + 1), 'base64');

/** The pixels of a binary P6 PPM (maxval 255; native writes '#' comment lines in the header). */
const pixels = (b: Uint8Array): { w: number; h: number; rgb: Uint8Array } => {
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
  const w = Number(fields[1]), h = Number(fields[2]);
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

describe('fix graphics/chunk radius from the pair style sigma', () => {
  for (const c of ['chunk_sigma', 'chunk_default', 'chunk_flag0', 'region_forestgreen']) {
    it(`${c}: the image is native's pixel for pixel`, async () => {
      const r = await render(readFileSync(join(DIR, `${c}.in`), 'utf8'));
      expect(r.error).toBeUndefined();
      const got = pixels(bytesOf(r.files.get('img.0.ppm')!));
      const want = pixels(readFileSync(join(DIR, `${c}.ppm`)));
      expect([got.w, got.h]).toEqual([want.w, want.h]);
      let diff = 0;
      for (let k = 0; k < want.rgb.length; k++) if (got.rgb[k] !== want.rgb[k]) diff++;
      expect(diff).toBe(0);
    });
  }

  it('refuses a pair style whose sigma the engine does not expose, naming it', async () => {
    const input = readFileSync(join(DIR, 'chunk_sigma.in'), 'utf8')
      .replace(/pair_style lj\/cut 2\.5\n(pair_coeff .*\n)+/, 'pair_style soft 2.5\npair_coeff * * 1.0\n');
    const r = await render(input);
    expect(r.error).toMatch(/pair style soft does not expose/);
  });
});
