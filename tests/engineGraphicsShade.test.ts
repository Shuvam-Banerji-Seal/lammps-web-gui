import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import { applySsao, shadeSurface } from '../src/engine-gpl/graphics/image_shading';
import type { RasterFrame } from '../src/engine/output/image_objects';

/*
 * dump image shading (src/engine-gpl/graphics/image_shading.ts: Image::draw_pixel with shiny, Image::compute_SSAO),
 * and the fsaa reduction in src/engine/output/dump_image.ts (Image::merge). The light constants are the Image
 * constructor defaults (setup_lights): the key light is at theta 30, phi -45, the fill at theta 0, phi 30, the back
 * light at theta 15, phi 180, with colours 0.9, 0.45 and 0.9. The half vector of the key light is normalise(key + z).
 */

const DEG = Math.PI / 180;
const dirOf = (theta: number, phi: number): number[] => [Math.cos(theta) * Math.sin(phi), Math.sin(theta), Math.cos(theta) * Math.cos(phi)];
const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sat = (v: number) => Math.min(1, Math.max(0, v));

describe('draw_pixel light sum (shiny and ambient)', () => {
  it('a surface facing the viewer, red, shiny 1: key diffuse, half-vector highlight and fill', () => {
    // n = (0, 0, 1): key diffuse 0.9 * cos(key) with cos = 0.6124, the highlight 0.9 * (n.half)^16 with
    // n.half = (1 + cos key) / |key + z|, fill 0.45 * 0.8660 and back 0 (the back light points away from the viewer)
    const key = dirOf(30 * DEG, -45 * DEG), fill = dirOf(0, 30 * DEG);
    const hk = [key[0], key[1], 1 + key[2]];
    const half = Math.hypot(hk[0], hk[1], hk[2]);
    const nh = hk[2] / half;
    const expect0 = 1 * 0.9 * sat(key[2]) + 0.9 * Math.pow(nh, 16) + 0 + 1 * 0.45 * sat(fill[2]);
    const out = shadeSurface([0, 0, 1], [1, 0, 0], 1, 0);
    expect(out[0]).toBeCloseTo(sat(expect0), 12);
    expect(out[1]).toBeCloseTo(sat(0.9 * Math.pow(nh, 16)), 12);
    expect(out[2]).toBeCloseTo(sat(0.9 * Math.pow(nh, 16)), 12);
    // the same numbers as the 0..255 bytes dump image writes (int(c * 255)): green and blue 0.16 * 255 = 40
    expect(Math.trunc(out[1] * 255)).toBe(Math.trunc(0.9 * Math.pow(nh, 16) * 255));
  });

  it('shiny scales the highlight intensity and sharpens it (hardness 16 * shiny)', () => {
    const n = [0.3, 0.2, Math.sqrt(1 - 0.13)];
    const key = dirOf(30 * DEG, -45 * DEG), fill = dirOf(0, 30 * DEG), back = dirOf(15 * DEG, 180 * DEG);
    const hk = [key[0], key[1], 1 + key[2]];
    const hl = Math.hypot(hk[0], hk[1], hk[2]);
    const nh = sat(dot(n, hk) / hl);
    const col = 0.2;
    // unsaturated sums, so the highlight is the only term that depends on shiny
    const diffuse = col * 0.9 * sat(dot(n, key)) + col * 0.45 * sat(dot(n, fill)) + col * 0.9 * sat(dot(n, back));
    const d = (s: number) => shadeSurface(n as [number, number, number], [col, col, col], s, 0)[0];
    expect(d(0)).toBeCloseTo(diffuse, 12);
    expect(d(0.5)).toBeCloseTo(diffuse + 0.9 * Math.pow(nh, 8) * 0.5, 12);
    expect(d(1)).toBeCloseTo(diffuse + 0.9 * Math.pow(nh, 16), 12);
  });

  it('ambient 0.5 (ssao on) adds half the surface colour to every channel', () => {
    const out = shadeSurface([0, 0, 1], [0.4, 0, 0], 0, 0.5);
    const base = shadeSurface([0, 0, 1], [0.4, 0, 0], 0, 0);
    expect(out[0] - base[0]).toBeCloseTo(0.2, 12);
    expect(out[1]).toBeCloseTo(base[1], 12);
  });
});

/** A flat frame of depth-buffer values (buf = zdist - depth) with a per-pixel surface. */
const frameOf = (w: number, h: number, dep: (x: number, y: number) => number | null, normal: (x: number, y: number) => number[]): RasterFrame => {
  const rgb = new Uint8Array(w * h * 3).fill(200);
  const depth = new Float64Array(w * h).fill(-Infinity);
  const surface = new Float64Array(2 * w * h);
  const zdist = 100;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const d = dep(x, y);
    if (d === null) continue;
    const idx = y * w + x;
    depth[idx] = zdist - d;
    const n = normal(x, y);
    surface[2 * idx] = n[1];
    surface[2 * idx + 1] = -n[0];
  }
  return { w, h, rgb, depth, camRight: [1, 0, 0], camUp: [0, 1, 0], camDir: [0, 0, 1], ctr: [0, 0, 0], zdist, pixelWidth: 0.5, surface };
};

describe('compute_SSAO', () => {
  it('leaves a flat surface facing the viewer unchanged (no horizon, no occlusion)', () => {
    const f = frameOf(6, 6, () => 10, () => [0, 0, 1]);
    const before = Uint8Array.from(f.rgb);
    applySsao(f, 2, 12, 315123);
    expect([...f.rgb]).toEqual([...before]);
  });

  it('never brightens a pixel and leaves empty pixels alone', () => {
    const f = frameOf(12, 8,
      (x, y) => (x === 0 || y === 7 ? null : 10 + ((x * 3 + y) % 4) * 1.5),
      (x, y) => { const a = 0.2 * ((x + y) % 3); return [Math.sin(a), 0, Math.sqrt(1 - Math.sin(a) ** 2)]; });
    const before = Uint8Array.from(f.rgb);
    applySsao(f, 3, 16, 315123);
    for (let i = 0; i < before.length; i++) expect(f.rgb[i]).toBeLessThanOrEqual(before[i]);
    // the empty pixels (depth -Infinity) keep their colour
    const empty = 0; // column 0 has no depth: the top-left pixel
    expect(f.rgb[empty]).toBe(before[empty]);
  });

  it('darkens a pixel that sits at the foot of a much nearer wall', () => {
    // a wall at column 2 (native depth 30) is nearer than the floor around it (depth 60); the floor next to it is occluded
    const floorDepth = (x: number) => (x === 2 ? 30 : 60);
    const f = frameOf(9, 3, (x) => floorDepth(x), () => [0.5, 0, Math.sqrt(0.75)]);
    const before = Uint8Array.from(f.rgb);
    applySsao(f, 4, 24, 315123);
    // the pixel of the floor just beside the wall (column 3, row 1) is darker than before
    const o = (1 * 9 + 3) * 3;
    expect(f.rgb[o]).toBeLessThan(before[o]);
  });
});

const run = async (body: string) => {
  const files = new Map<string, string>();
  const session = new Session({ emit: () => {}, writeFile: (n, t) => files.set(n, t) });
  await session.execute(`units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 4.0 5.0
create_atoms 1 single 5.0 6.0 5.0
mass * 1.0
${body}
run 0
`);
  return files;
};

describe('dump image keywords shiny, fsaa and ssao', () => {
  it('accepts the keywords; the values are checked (shiny 0..1, ssao seed > 0, intensity 0..1)', async () => {
    await expect(run('dump im all image 1 out.*.ppm type type size 64 64 shiny 0.4 fsaa yes ssao yes 315123 0.4')).resolves.toBeDefined();
    await expect(run('dump im all image 1 out.*.ppm type type shiny 1.5')).rejects.toThrow(/shiny/);
    await expect(run('dump im all image 1 out.*.ppm type type ssao yes 0 0.4')).rejects.toThrow(/seed/);
    await expect(run('dump im all image 1 out.*.ppm type type ssao yes 5 1.5')).rejects.toThrow(/intensity/);
  });

  it('fsaa output is the 2x2 average of the same image rendered at twice the size', async () => {
    const small = await run('dump im all image 1 out.*.ppm type type size 40 30 zoom 1.2 view 70 20 fsaa yes');
    const big = await run('dump im all image 1 out.*.ppm type type size 80 60 zoom 1.2 view 70 20');
    const dec = (url: string) => {
      const i = url.indexOf(',');
      const bin = atob(url.slice(i + 1));
      // skip the PPM header "P6\n<w> <h>\n255\n"
      const head = /^P6\s+(\d+)\s+(\d+)\s+255\s/.exec(bin)!;
      const out = new Uint8Array(bin.length - head[0].length);
      for (let k = 0; k < out.length; k++) out[k] = bin.charCodeAt(head[0].length + k);
      return { w: Number(head[1]), h: Number(head[2]), px: out };
    };
    const a = dec(small.get('out.0.ppm')!), b = dec(big.get('out.0.ppm')!);
    expect(a.w).toBe(40);
    expect(a.h).toBe(30);
    expect(b.w).toBe(80);
    for (let y = 0; y < a.h; y++) for (let x = 0; x < a.w; x++) for (let c = 0; c < 3; c++) {
      const s = b.px[((2 * y) * 80 + 2 * x) * 3 + c] + b.px[((2 * y) * 80 + 2 * x + 1) * 3 + c]
        + b.px[((2 * y + 1) * 80 + 2 * x) * 3 + c] + b.px[((2 * y + 1) * 80 + 2 * x + 1) * 3 + c];
      expect(a.px[(y * 40 + x) * 3 + c]).toBe(Math.trunc(0.25 * s));
    }
  });

  it('shiny changes the highlight: shiny 0 and shiny 1 renders differ', async () => {
    const s0 = await run('dump im all image 1 out.*.ppm type type size 48 48 shiny 0');
    const s1 = await run('dump im all image 1 out.*.ppm type type size 48 48 shiny 1');
    expect(s0.get('out.0.ppm')).not.toBe(s1.get('out.0.ppm'));
  });
});
