import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image depthcue, defocus and outline (src/engine-gpl/graphics/image_postprocess.ts, ported from
 * src/GRAPHICS/image.cpp at b332f28: compute_outline, compute_depthcue, compute_defocus, box_depth_minmax; the order
 * of Image::merge), and the dump_modify ssaosamples and specular keywords (dump_image.html; image.cpp draw_pixel and
 * compute_SSAO), against native LAMMPS PPMs.
 *
 * Fixtures (tests/fixtures/image_depth): <case>.in is the input with the keyword, <case>.base.in the same input
 * without it. <case>.ppm and <case>.base.ppm are native's output (/home/roy/.local/bin/lmp, b332f28; the native PPM
 * header is "# CREATOR" then the size). Cases are 64x64, the defocus cases 200x200 (the blur radius is 1% of the image
 * height, so a smaller image blurs by less than one pixel and native skips the pass).
 *
 * Every case is byte-exact with native except dc_named (depthcue with a white fog, start 0.3): 1 byte, by one
 * level in the red channel, where the fade lands just below an integer (see the notes in tests/engineDumpImageDepth
 * and the report). That byte is pinned, not loosened. (It was 35 bytes before the camera basis followed native's
 * view_params arithmetic, camRight = up x camDir and camUp = camDir x camRight, in dump_image.ts; the pin moved with it.)
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_depth');

/** Bytes that differ from native (count) and the largest difference (max) for each fixture: [count, max]. */
const EXACT = ['dc_auto', 'df_auto', 'df_num', 'outline', 'combo', 'ssaos', 'specT', 'specN'];
const BASE: Record<string, { count: number; max: number }> = {
  dc_auto: { count: 0, max: 0 }, dc_named: { count: 0, max: 0 }, df_auto: { count: 0, max: 0 },
  df_num: { count: 0, max: 0 }, outline: { count: 0, max: 0 }, combo: { count: 0, max: 0 },
  ssaos: { count: 0, max: 0 }, specT: { count: 0, max: 0 }, specN: { count: 0, max: 0 },
};
const WITH: Record<string, { count: number; max: number }> = {
  dc_auto: { count: 0, max: 0 }, dc_named: { count: 1, max: 1 }, df_auto: { count: 0, max: 0 },
  df_num: { count: 0, max: 0 }, outline: { count: 0, max: 0 }, combo: { count: 0, max: 0 },
  ssaos: { count: 0, max: 0 }, specT: { count: 0, max: 0 }, specN: { count: 0, max: 0 },
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

/** Runs an input and returns the image it wrote (out.0.ppm style file, the only .ppm). */
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

const CASES = Object.keys(WITH);
const cache = new Map<string, Promise<{ kw: Raster; base: Raster; nkw: Raster; nbase: Raster }>>();
const load = (name: string) => {
  if (!cache.has(name)) {
    cache.set(name, (async () => ({
      kw: await render(name + '.in'),
      base: await render(name + '.base.in'),
      nkw: native(name + '.ppm'),
      nbase: native(name + '.base.ppm'),
    }))());
  }
  return cache.get(name)!;
};

describe('dump image depthcue, defocus and outline against native LAMMPS', () => {
  for (const name of CASES) {
    it(`${name}: without the new keyword the engine matches native's output`, async () => {
      const { base, nbase } = await load(name);
      const d = diff(base, nbase);
      expect(d).toEqual(BASE[name]);
    });

    it(`${name}: with the keyword the engine matches native's output (dc_named: the pinned 1 byte)`, async () => {
      const { kw, nkw } = await load(name);
      const d = diff(kw, nkw);
      expect(d).toEqual(WITH[name]);
    });
  }

  it('where the native neighbourhood of a pixel agrees with the engine, the keyword output is exact', async () => {
    // Pixels whose 7x7 neighbourhood agrees in the no-keyword pair: the keyword's own output must match native there.
    // (dc_named is the one fixture that is not byte-exact; its fade is checked below to one level.)
    const R = 3;
    for (const name of EXACT) {
      const { kw, base, nkw, nbase } = await load(name);
      const { w, h } = base;
      const agree = (x: number, y: number) => {
        const o = (y * w + x) * 3;
        return base.rgb[o] === nbase.rgb[o] && base.rgb[o + 1] === nbase.rgb[o + 1] && base.rgb[o + 2] === nbase.rgb[o + 2];
      };
      let checked = 0, bad = 0;
      for (let y = R; y < h - R; y++) for (let x = R; x < w - R; x++) {
        let ok = true;
        for (let yy = y - R; yy <= y + R && ok; yy++) for (let xx = x - R; xx <= x + R && ok; xx++) ok = agree(xx, yy);
        if (!ok) continue;
        checked++;
        const o = (y * w + x) * 3;
        for (let c = 0; c < 3; c++) if (kw.rgb[o + c] !== nkw.rgb[o + c]) bad++;
      }
      expect(checked, `${name}: pixels with an agreeing neighbourhood`).toBeGreaterThan(0);
      expect(bad, `${name}: differing bytes on agreeing neighbourhoods`).toBe(0);
    }
    const dc = await load('dc_named');
    const { w, h } = dc.base;
    let worst = 0;
    for (let i = 0; i < w * h * 3; i++) {
      const o = i;
      if (dc.base.rgb[o] === dc.nbase.rgb[o] && dc.base.rgb[o + 1] === dc.nbase.rgb[o + 1] && dc.base.rgb[o + 2] === dc.nbase.rgb[o + 2]) {
        worst = Math.max(worst, Math.abs(dc.kw.rgb[o] - dc.nkw.rgb[o]));
      }
    }
    expect(worst).toBeLessThanOrEqual(1);
  });
});
