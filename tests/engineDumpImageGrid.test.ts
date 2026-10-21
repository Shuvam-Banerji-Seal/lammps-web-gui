import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image grid (src/GRAPHICS/dump_image.cpp at b332f28, create_image "render my grid cells", in
 * src/engine-gpl/graphics/grid_cells.ts): the cells of a per-grid quantity of a compute or fix, coloured by the grid
 * colour map (dump_modify gmap, dynamic range over the cells) with the opacity of dump_modify gtrans and the brightened
 * lighting of flat surfaces; 2d rectangles, 3d cell surfaces, triclinic cells through Domain::lamda2x; fix
 * graphics/labels colorscale map grid draws the grid map as the previous output left it.
 *
 * Fixtures (tests/fixtures/image_grid): <case>.in is the input and <case>.ppm is native's output:
 * /home/roy/.local/bin/lmp, b332f28, written as PPM, 128x128 px (grid_ave: the image of step 2 of a run 2).
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_grid');

/** Each case and the image file of the engine run that is compared. */
const CASES: Record<string, string> = { grid_2d: 'img.0.ppm', grid_3d: 'img.0.ppm', grid_ave: 'img.2.ppm', grid_tri: 'img.0.ppm', grid_colorscale: 'img.1.ppm' };

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
const render = async (file: string, out?: string): Promise<Raster> => {
  const files = new Map<string, string>();
  const s = new Session({ emit: () => {}, writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(readFileSync(join(DIR, file), 'utf8'));
  const name = out ?? [...files.keys()].find((k) => k.endsWith('.ppm'));
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

describe('dump image grid against native LAMMPS', () => {
  for (const [name, out] of Object.entries(CASES)) {
    it(`${name}: engine image vs native PPM`, async () => {
      expect(diff(await render(`${name}.in`, out), native(`${name}.ppm`))).toEqual({ count: 0, max: 0 });
    });
  }

  const GRID2D = readFileSync(join(DIR, 'grid_2d.in'), 'utf8');
  const run = async (input: string): Promise<{ logs: string[]; files: Map<string, string> }> => {
    const logs: string[] = [];
    const files = new Map<string, string>();
    const s = new Session({ emit: (e) => { if (e.kind === 'log') logs.push(e.text); }, writeFile: (n, t) => files.set(n, t) });
    await s.execute(input);
    return { logs, files };
  };

  // Measured with native LAMMPS (src/GRAPHICS/dump_image.cpp:3041 and src/utils.cpp at b332f28):
  // ERROR: Invalid gtrans opacity in dump_modify command
  // ERROR: Dump image compute pg data data is not per-grid vector
  // ERROR: Dump image compute pg array data is accessed out-of-range
  it.each([
    ['gtrans outside 0..1', GRID2D.replace('adiam * 0.3', 'adiam * 0.3 gtrans 1.5'), 'Invalid gtrans opacity in dump_modify command'],
    ['a per-grid array named as a vector', GRID2D.replace('grid c_pg:grid:data[1]', 'grid c_pg:grid:data'), 'Dump image compute pg data data is not per-grid vector'],
    ['a column past the array', GRID2D.replace('grid c_pg:grid:data[1]', 'grid c_pg:grid:data[3]'), 'Dump image compute pg array data is accessed out-of-range'],
  ])('stops as native does: %s', async (_name, input, text) => {
    await expect(run(input)).rejects.toThrow(text);
  });

  it('ignores gtrans when no grid is drawn, as native does', async () => {
    // Measured with native LAMMPS: the run completes (DumpImage::modify_param returns before reading the value)
    const { files } = await run(GRID2D.replace(' grid c_pg:grid:data[1]', '').replace('adiam * 0.3', 'adiam * 0.3 gtrans 1.5'));
    expect([...files.keys()].some((k) => k.endsWith('.ppm'))).toBe(true);
  });

  it('draws a 3d per-grid array column with its values and says once that native does not', async () => {
    // native never loads the column in 3d (see dump_image.ts drawGrid); the engine's image equals the vector form's
    const vec = readFileSync(join(DIR, 'grid_3d.in'), 'utf8');
    const arr = vec.replace('property/grid 3 3 3 z', 'property/grid 3 3 3 id z').replace('grid c_pg:grid:data', 'grid c_pg:grid:data[2]');
    const a = await run(arr), b = await run(vec);
    expect(a.files.get('img.0.ppm')).toBe(b.files.get('img.0.ppm'));
    expect(a.logs.filter((l) => l.includes('does not load a per-grid array column in 3d'))).toHaveLength(1);
    expect(b.logs.some((l) => l.includes('per-grid array column'))).toBe(false);
  });
});
