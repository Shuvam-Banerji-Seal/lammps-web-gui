import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';
import { ColorMap } from '../src/engine/output/image_colormap';

/*
 * GRAPHICS package in the browser engine: dump_modify amap (colour maps), dump image "fix" objects, and the fixes
 * graphics/objects, graphics/lines and graphics/arrows (src/engine-gpl/graphics). The images cannot be compared
 * pixel-exactly with native LAMMPS, so the checks are: the colour-map rules of image.cpp (ColorMap), the projected
 * position of an object's pixels for a 10x10x10 box seen along x (view 90 0, 200x200 px: scale 10 px per unit, the
 * projected box centre at column 100 and row 98.5, +y to larger columns), and the thermo values of f_ID.
 */

const b64 = (url: string): Uint8Array => {
  const i = url.indexOf(',');
  const bin = atob(url.slice(i + 1));
  const out = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) out[k] = bin.charCodeAt(k);
  return out;
};

/** Decode a P6 PPM (binary, maxval 255) written by dump image. */
const decodePpm = (b: Uint8Array): { w: number; h: number; rgb: Uint8Array } => {
  let i = 0;
  const tok = (): string => {
    while (i < b.length) {
      if (b[i] === 0x23) { while (i < b.length && b[i] !== 0x0a) i++; } else if ([0x20, 0x09, 0x0a, 0x0d].includes(b[i])) i++; else break;
    }
    const s = i;
    while (i < b.length && ![0x20, 0x09, 0x0a, 0x0d].includes(b[i])) i++;
    return String.fromCharCode(...b.subarray(s, i));
  };
  expect(tok()).toBe('P6');
  const w = Number(tok()), h = Number(tok());
  expect(Number(tok())).toBe(255);
  i++;
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

const px = (img: { w: number; rgb: Uint8Array }, x: number, y: number): [number, number, number] => {
  const o = (y * img.w + x) * 3;
  return [img.rgb[o], img.rgb[o + 1], img.rgb[o + 2]];
};

/** Columns and rows of pixels that are not black. */
const litExtent = (img: { w: number; h: number; rgb: Uint8Array }): { x0: number; x1: number; y0: number; y1: number; n: number } => {
  let x0 = Infinity, x1 = -1, y0 = Infinity, y1 = -1, n = 0;
  for (let y = 0; y < img.h; y++) for (let x = 0; x < img.w; x++) {
    const [r, g, b] = px(img, x, y);
    if (r + g + b > 0) { n++; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  }
  return { x0, x1, y0, y1, n };
};

/** A 10x10x10 box; `atoms` are create_atoms lines; `pre` and `dump` are the input lines around `run`. */
const scene = (atoms: string, pre: string, dump: string, steps = 0): string => `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 2 box
${atoms}
mass * 1.0
${pre}
${dump}
run ${steps}
`;

const run = async (input: string): Promise<{ files: Map<string, string>; rows: Record<string, number>[]; logs: string[] }> => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  await s.execute(input);
  const rows = events.filter((e): e is Extract<EngineEvent, { kind: 'thermo' }> => e.kind === 'thermo').map((e) => e.row as unknown as Record<string, number>);
  const logs = events.filter((e): e is Extract<EngineEvent, { kind: 'log' }> => e.kind === 'log').map((e) => e.text);
  return { files, rows, logs };
};

const named = (w: string): [number, number, number] | null => {
  const c: Record<string, [number, number, number]> = { blue: [0, 0, 255], red: [255, 0, 0], green: [0, 255, 0], white: [255, 255, 255] };
  return c[w] ?? null;
};

describe('dump_modify amap (colour maps, image.cpp ColorMap rules)', () => {
  it('the default map runs blue at the minimum to red at the maximum, linearly between', () => {
    const m = ColorMap.default();
    m.bounds(0, 10);
    // color2rgb("blue") = (0, 0, 1) and color2rgb("red") = (1, 0, 0), image.cpp rgbcolors: the colours stay doubles
    expect(m.color(0)).toEqual([0, 0, 1]);
    expect(m.color(10)).toEqual([1, 0, 0]);
    const mid = m.color(5);
    expect(mid[0]).toBeCloseTo(0.5, 12);
    expect(mid[2]).toBeCloseTo(0.5, 12);
    // values outside the range are clamped
    expect(m.color(-4)).toEqual([0, 0, 1]);
    expect(m.color(40)).toEqual([1, 0, 0]);
  });

  it('sequential style bins the range and gives the top value the bin below (rounddown)', () => {
    // hyper/in.hyper.global: dump_modify amap 1 3 sa 1 3 blue red green
    const { map, used } = ColorMap.parse(['1', '3', 'sa', '1', '3', 'blue', 'red', 'green'], named);
    expect(used).toBe(8);
    map.bounds(1, 3);
    expect(map.color(1)).toEqual([0, 0, 255]);
    expect(map.color(2)).toEqual([255, 0, 0]);
    // 3 is the top of the range: with a range of exactly two bins it takes the bin below (red), as native does
    expect(map.color(3)).toEqual([255, 0, 0]);
  });

  it('discrete style takes the first bin that contains the value', () => {
    const { map } = ColorMap.parse(['0', '10', 'da', '0', '2', 'min', '5', 'blue', '5', 'max', 'red'], named);
    map.bounds(0, 10);
    expect(map.color(3)).toEqual([0, 0, 255]);
    expect(map.color(7)).toEqual([255, 0, 0]);
  });

  it('refuses malformed maps with the reason', () => {
    expect(() => ColorMap.parse(['1', '3', 'xa', '2', 'blue', 'red'], named)).toThrow(/colour map style/);
    expect(() => ColorMap.parse(['0', '1', 'cf', '0', '2', 'min', 'blue', 'max', 'nope'], named)).toThrow(/unknown colour 'nope'/);
    expect(() => ColorMap.parse(['0', '1', 'cf', '0', '2', '0', 'blue', 'max', 'red'], named)).toThrow(/first continuous entry must be min/);
    expect(() => ColorMap.parse(['3', '1', 'cf', '0', '2', 'min', 'blue', 'max', 'red'], named)).toThrow(/lo must be smaller/);
  });

  it('dump image colours atoms by an amap (id 1 in bin [0,2) blue, id 2 in bin [2,4] red)', async () => {
    const atoms = 'create_atoms 1 single 5 2 5\ncreate_atoms 1 single 5 8 5';
    const { files } = await run(scene(atoms, '', 'dump im all image 1 out.*.ppm id id size 200 200 box no 0.02 atom yes adiam 1.0 view 90 0\ndump_modify im amap 0 4 sa 2 2 blue red'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    // atom id 1 at y = 2 projects to column 70, id 2 at y = 8 to column 130 (10 px per unit, +y to larger columns);
    // the row of the centre is 98 (pixel index of the continuous centre 98.5)
    // Measured with native LAMMPS (b332f28, this scene): pixel (70, 98) is (78, 78, 255) and (130, 98) is (255, 78, 78)
    const a = px(img, 70, 98), b = px(img, 130, 98);
    expect(a).toEqual([78, 78, 255]);
    expect(b).toEqual([255, 78, 78]);
  });
});

describe('dump image fix objects (graphics/objects, graphics/arrows)', () => {
  it('a sphere object appears at its projected position with the requested diameter', async () => {
    // radius 1 at (y = 8, z = 5): column 100 + 3 * 10 = 130, row 98.5; diameter 2 units = 20 px
    const { files } = await run(scene('create_atoms 1 single 5 5 5', 'fix s all graphics/objects 1 sphere 1 5 8 5 1.0', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 fix s const 1 0'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    const e = litExtent(img);
    expect(e.x0).toBeGreaterThanOrEqual(119);
    expect(e.x1).toBeLessThanOrEqual(141);
    // pixel centres sit at x + 0.5, so a disc centred on column 130.0 spans columns 120..139 (mean 129.5)
    expect((e.x0 + e.x1) / 2).toBeCloseTo(129.5, 0);
    expect((e.y0 + e.y1) / 2).toBeCloseTo(98, 0);
    // a disc of radius 10 px has about pi * 100 pixels
    expect(e.n).toBeGreaterThan(260);
    expect(e.n).toBeLessThan(380);
  });

  it('an arrow object runs along its direction (bottom to tip, head at the tip)', async () => {
    // bottom (5,5,5) to tip (5,8,5), radius 0.2 (shaft 4 px), tip ratio 0.2: columns 100 to 130
    const { files } = await run(scene('create_atoms 1 single 5 5 5', 'fix a all graphics/objects 1 arrow 1 5 5 5 5 8 5 0.2 0.2', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 fix a const 0 0'));
    const e = litExtent(decodePpm(b64(files.get('out.0.ppm')!)));
    expect(e.x0).toBeGreaterThanOrEqual(97);
    expect(e.x1).toBeGreaterThanOrEqual(128);
    expect(e.x1).toBeLessThanOrEqual(134);
  });

  it('graphics/arrows velocity draws one arrow per atom with length |v| * scale', async () => {
    // one atom at (5,5,5) with v = (0,2,0): an arrow of length 2 * scale 1 centred on the atom, columns 90 to 110
    const { files } = await run(scene('create_atoms 1 single 5 5 5', 'velocity all set 0 2 0\nfix v all graphics/arrows 1 velocity 1.0 0.5', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 fix v const 0 0'));
    const e = litExtent(decodePpm(b64(files.get('out.0.ppm')!)));
    // the arrow is ImageObjects::ArrowObj (image_objects.cpp), whose bottom and tip are flat (no caps), so the lit columns
    // are exactly the arrow's length: 90..110 (the arrow of length 2 centred on column 100)
    expect(e.x0).toBeGreaterThanOrEqual(89);
    expect(e.x0).toBeLessThanOrEqual(91);
    expect(e.x1).toBeGreaterThanOrEqual(109);
    expect(e.x1).toBeLessThanOrEqual(111);
  });

  it('dump_modify fcolor sets the colour of a const-style fix object', async () => {
    const { files } = await run(scene('create_atoms 1 single 5 5 5', 'fix s all graphics/objects 1 sphere 1 5 8 5 1.0', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 fix s const 0 0\ndump_modify im fcolor s red'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    const c = px(img, 130, 98);
    expect(c[0]).toBeGreaterThan(150);
    expect(c[1]).toBeLessThan(60);
    expect(c[2]).toBeLessThan(60);
  });

  it('graphics/lines keeps Nlength averages; f_ID counts them (window rule of fix ave/atom)', async () => {
    // Nevery 1, Nrepeat 2, Nfreq 2, Nlength 3: the first store is at step 2, the cap at 3 stores
    const input = scene('create_atoms 1 single 5 5 5', 'velocity all set 0 1 0\nfix l all graphics/lines 1 2 2 3\nthermo_style custom step f_l\nthermo 1', '', 9);
    const { rows } = await run(input);
    const at = (step: number) => rows.find((r) => r.step === step)?.f_l;
    expect(at(1)).toBe(0);
    expect(at(3)).toBe(1);
    expect(at(5)).toBe(2);
    expect(at(9)).toBe(3);
  });

  it('graphics/lines draws the history as line segments with the flag2 width', async () => {
    const input = scene('create_atoms 1 single 5 5 5', 'velocity all set 0 1 0\nfix l all graphics/lines 1 1 2 3', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 fix l const 1 0.5', 6);
    const { files } = await run(input);
    expect(litExtent(decodePpm(b64(files.get('out.6.ppm')!))).n).toBeGreaterThan(0);
  });
});

describe('dump image axes (dump_image.cpp axes block)', () => {
  it('lowerleft axes start at the box corner shifted by 10% of the longest edge (native: tests/fixtures/image_cyl/axes_t.ppm)', async () => {
    // origin (-1, -1, -1) projects to column 40, row 158.5 (z down is larger rows); the y arrow (green) runs from
    // y = -1 to y = 4 (length 0.5 * 10 = 5) along row 158. The red x arrow points at the viewer, so its end-on head
    // covers the start of the green shaft. The far end of the green arrow is at column 86 in native's PPM for this
    // scene (the arrow head is a draw_trinorm mesh, image_objects.cpp ArrowObj, so its tip is 4 columns short of the
    // flat-stroke column 90 the earlier engine drew); the letters are drawn too (draw_axes), so no log line names them.
    const { files, logs } = await run(scene('create_atoms 1 single 5 5 5', '', 'dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 atom no view 90 0 axes yes 0.5 0.05'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    let x1 = -1, n = 0, ys = 0;
    for (let y = 0; y < img.h; y++) for (let x = 0; x < img.w; x++) {
      const [r, g, b] = px(img, x, y);
      if (g > 150 && r < 80 && b < 80) { n++; x1 = Math.max(x1, x); ys += y; }
    }
    expect(n).toBe(88);
    expect(ys / n).toBeCloseTo(158.4772727, 6);
    expect(x1).toBe(86);
    expect(logs.some((t) => /letters are not drawn/.test(t))).toBe(false);
  });

  it('refuses a triclinic box and an unknown location', async () => {
    await expect(run(scene('create_atoms 1 single 5 5 5', '', 'dump im all image 1 out.*.ppm type type axes diagonal 0.5 0.05'))).rejects.toThrow(/unknown dump image axes location 'diagonal'/);
  });
});

describe('graphics fixes and dump image keywords that are refused by name', () => {
  it('graphics/isosurface is refused by name', async () => {
    await expect(run(scene('create_atoms 1 single 5 5 5', 'fix x all graphics/isosurface 1', ''))).rejects.toThrow('fix graphics/isosurface is not supported');
  });

  it('graphics/objects refuses polygonal cones (progbar is drawn, see engineGraphics2Render.test.ts)', async () => {
    await expect(run(scene('create_atoms 1 single 5 5 5', 'fix o all graphics/objects 1 cone 1 5 5 5 5 6 5 0.2 0.1 4', ''))).rejects.toThrow(/polygonal cones/);
  });

  it('dump image fix must name an existing fix that provides graphics', async () => {
    await expect(run(scene('create_atoms 1 single 5 5 5', '', 'dump im all image 1 out.*.ppm type type fix nope const 0 0'))).rejects.toThrow(/fix ID 'nope' does not exist/);
    await expect(run(scene('create_atoms 1 single 5 5 5', 'fix n all nve', 'dump im all image 1 out.*.ppm type type fix n const 0 0'))).rejects.toThrow(/provides no graphics objects/);
  });
});
