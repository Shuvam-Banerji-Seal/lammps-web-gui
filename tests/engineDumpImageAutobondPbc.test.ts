import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Session } from '../src/engine/interpreter';

/*
 * dump image autobond in periodic boxes (src/GRAPHICS/dump_image.cpp at b332f28, the AUTO branch of create_image, in
 * src/engine-gpl/graphics/autobond.ts). Native walks the pair neighbor list; a pair whose second atom is a ghost is drawn
 * whole at the ghost, so a bond across a periodic boundary reaches out of the box on the side of the atom whose list
 * holds the pair. That side comes from native's list build: the bins and the half stencil (neighbor bin, orthogonal box,
 * also with neigh_modify binsize), the parity of the two IDs (neighbor nsq, triclinic box), both sides (newton off, then
 * the smaller ID draws) or every pair from both sides (a full list, pair sw). Only the pairs of the list are drawn: a
 * type pair with a shorter pair cutoff has no bonds past it (pbc_typecut).
 *
 * Fixtures (tests/fixtures/image_autobond): <case>.in is the input and <case>.ppm is native's output:
 * /home/roy/.local/bin/lmp, b332f28, run 0, written as PPM. Every case is 128x128 px and byte-exact.
 */

const DIR = join(process.cwd(), 'tests', 'fixtures', 'image_autobond');

const CASES = ['pbc_lattice', 'pbc_liquid', 'pbc_binsize', 'pbc_nsq', 'pbc_newtoff', 'pbc_tri', 'pbc_full', 'pbc_typecut'];

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
  s.addFile('Si.sw', readFileSync(join(process.cwd(), 'third_party', 'lammps', 'potentials', 'Si.sw'), 'utf8'));
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

describe('dump image autobond across periodic boundaries against native LAMMPS', () => {
  for (const name of CASES) {
    it(`${name}: engine image vs native PPM`, async () => {
      expect(diff(await render(`${name}.in`), native(`${name}.ppm`))).toEqual({ count: 0, max: 0 });
    });
  }

  const LATTICE = readFileSync(join(DIR, 'pbc_lattice.in'), 'utf8');
  const run = async (input: string): Promise<string[]> => {
    const logs: string[] = [];
    const s = new Session({ emit: (e) => { if (e.kind === 'log') logs.push(e.text); }, writeFile: () => {} });
    await s.execute(input);
    return logs;
  };

  // Measured with native LAMMPS (src/GRAPHICS/dump_image.cpp:903, 906, 908 and 914, DumpImage::init_style):
  // ERROR: Dump image autobond requires neighbor style 'bin' or 'nsq'
  // ERROR: Dump image autobond requires a pair style to be defined
  // ERROR: Dump image autobond cutoff > neighbor cutoff
  // ERROR: Dump image autobond cutoff is larger than periodic domain
  it.each([
    ['neighbor multi', LATTICE.replace('pair_style lj/cut 2.5', 'neighbor 0.3 multi\npair_style lj/cut 2.5'), "Dump image autobond requires neighbor style 'bin' or 'nsq'"],
    ['no pair style', LATTICE.replace(/^pair_(style|coeff) .*\n/gm, ''), 'Dump image autobond requires a pair style to be defined'],
    ['cutoff past the neighbor cutoff', LATTICE.replace('autobond 1.1 0.15', 'autobond 3.0 0.15'), 'Dump image autobond cutoff > neighbor cutoff'],
    ['cutoff past the periodic box', LATTICE.replace('autobond 1.1 0.15', 'autobond 4.5 0.15').replace('pair_style lj/cut 2.5', 'pair_style lj/cut 5.0'), 'Dump image autobond cutoff is larger than periodic domain'],
  ])('stops as native does: %s', async (_name, input, text) => {
    await expect(run(input)).rejects.toThrow(text);
  });

  it('warns once when the cutoff is past the smallest neighbor cutoff', async () => {
    // Measured with native LAMMPS: WARNING: Dump image autobond cutoff > minimum neighbor cutoff
    // (src/GRAPHICS/dump_image.cpp:910)
    const logs = await run(readFileSync(join(DIR, 'pbc_typecut.in'), 'utf8'));
    expect(logs.filter((l) => l.includes('WARNING: Dump image autobond cutoff > minimum neighbor cutoff'))).toHaveLength(1);
  });
});
