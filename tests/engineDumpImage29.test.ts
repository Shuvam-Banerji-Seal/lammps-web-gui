import { describe, expect, it } from 'vitest';
import { Session } from '../src/engine/interpreter';
import type { EngineEvent } from '../src/engine/types';

/*
 * dump image (src/engine/output/dump_image.ts). Each snapshot is stored under its file name as a
 * `data:` URL; the tests decode the PNG/PPM bytes themselves and check the projection geometry.
 *
 * Projection facts measured with native LAMMPS (black box) for a box of longest edge L and an
 * image of height h: scale = h / (2 L) * zoom, the projected box centre is at pixel
 * floor(w/2), (h even ? h/2 - 1.5 : (h-1)/2), +y moves towards larger columns and +z towards
 * smaller rows for view 90 0.
 */

const b64 = (url: string): Uint8Array => {
  const i = url.indexOf(',');
  const bin = atob(url.slice(i + 1));
  const out = new Uint8Array(bin.length);
  for (let k = 0; k < bin.length; k++) out[k] = bin.charCodeAt(k);
  return out;
};

const u32 = (b: Uint8Array, o: number): number => ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (b: Uint8Array): number => { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC_TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const adler32 = (b: Uint8Array): number => { let a = 1, s = 0; for (let i = 0; i < b.length; i++) { a = (a + b[i]) % 65521; s = (s + a) % 65521; } return ((s << 16) | a) >>> 0; };

interface Raster { w: number; h: number; rgb: Uint8Array }

/** Decode a P6 PPM (binary, maxval 255). */
const decodePpm = (b: Uint8Array): Raster => {
  let i = 0;
  const tok = () => {
    while (i < b.length) { if (b[i] === 0x23) { while (i < b.length && b[i] !== 0x0a) i++; } else if (b[i] === 0x20 || b[i] === 0x09 || b[i] === 0x0a || b[i] === 0x0d) i++; else break; }
    const s = i; while (i < b.length && ![0x20, 0x09, 0x0a, 0x0d].includes(b[i])) i++; return String.fromCharCode(...b.subarray(s, i));
  };
  expect(tok()).toBe('P6');
  const w = Number(tok()), h = Number(tok());
  expect(Number(tok())).toBe(255);
  i++;
  return { w, h, rgb: b.subarray(i, i + w * h * 3) };
};

/** Decode a PNG made by png.ts: verify chunk CRCs and the zlib Adler-32, inflate stored blocks. */
const decodePng = (b: Uint8Array): Raster => {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let k = 0; k < 8; k++) expect(b[k]).toBe(sig[k]);
  let off = 8, w = 0, h = 0;
  const idat: Uint8Array[] = [];
  while (off < b.length) {
    const len = u32(b, off);
    const type = String.fromCharCode(b[off + 4], b[off + 5], b[off + 6], b[off + 7]);
    const data = b.subarray(off + 8, off + 8 + len);
    const crc = u32(b, off + 8 + len);
    expect(crc, `CRC of ${type}`).toBe(crc32(b.subarray(off + 4, off + 8 + len)));
    if (type === 'IHDR') { w = u32(data, 0); h = u32(data, 4); expect(data[8]).toBe(8); expect(data[9]).toBe(2); }
    if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  // concatenate IDAT
  let n = 0; for (const d of idat) n += d.length;
  const z = new Uint8Array(n); let p = 0; for (const d of idat) { z.set(d, p); p += d.length; }
  expect(z[0]).toBe(0x78);
  expect(((z[0] << 8) | z[1]) % 31).toBe(0);
  // stored deflate blocks
  const raw: number[] = [];
  let q = 2;
  for (;;) {
    const final = z[q] & 1;
    expect((z[q] >> 1) & 3).toBe(0); // stored
    const len = z[q + 1] | (z[q + 2] << 8);
    const nlen = z[q + 3] | (z[q + 4] << 8);
    expect((len ^ 0xffff)).toBe(nlen);
    for (let k = 0; k < len; k++) raw.push(z[q + 5 + k]);
    q += 5 + len;
    if (final) break;
  }
  const stored = Uint8Array.from(raw);
  const ad = u32(z, q);
  expect(ad, 'Adler-32').toBe(adler32(stored));
  // unfilter (filter 0)
  const rgb = new Uint8Array(w * h * 3);
  const stride = w * 3;
  for (let y = 0; y < h; y++) {
    expect(stored[y * (stride + 1)]).toBe(0);
    rgb.set(stored.subarray(y * (stride + 1) + 1, y * (stride + 1) + 1 + stride), y * stride);
  }
  return { w, h, rgb };
};

/** Centroid of pixels where `pred` holds (background is black by default). */
const centroid = (r: Raster, pred: (r: number, g: number, b: number) => boolean): { x: number; y: number; n: number } => {
  let sx = 0, sy = 0, n = 0;
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const o = (y * r.w + x) * 3;
    if (pred(r.rgb[o], r.rgb[o + 1], r.rgb[o + 2])) { sx += x; sy += y; n++; }
  }
  return { x: sx / n, y: sy / n, n };
};

const base = (dump: string): string => `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 5.0 5.0 5.0
mass * 1.0
${dump}
run 0
`;

const run = async (input: string, extra: Record<string, string> = {}): Promise<Map<string, string>> => {
  const files = new Map<string, string>();
  const events: EngineEvent[] = [];
  const s = new Session({ emit: (e) => events.push(e), writeFile: (n, t, ap) => files.set(n, (ap ? files.get(n) ?? '' : '') + t) });
  for (const [n, t] of Object.entries(extra)) s.addFile(n, t);
  await s.execute(input);
  return files;
};

describe('dump image', () => {
  it('stores each snapshot as a data URL under its wildcard file name', async () => {
    const files = await run(base('dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 view 90 0'));
    const url = files.get('out.0.ppm');
    expect(url?.startsWith('data:image/x-portable-pixmap;base64,')).toBe(true);
  });

  it('PPM: one atom at the box centre lands at the image centre', async () => {
    const files = await run(base('dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 view 90 0 zoom 1.0'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    expect([img.w, img.h]).toEqual([200, 200]);
    const c = centroid(img, (r) => r > 0); // red atom on black
    expect(c.n).toBeGreaterThan(20);
    expect(Math.abs(c.x - 100)).toBeLessThanOrEqual(2);
    expect(Math.abs(c.y - 100)).toBeLessThanOrEqual(2);
  });

  it('PNG: decodes (CRC + Adler-32) and the atom projects to the same place', async () => {
    const files = await run(base('dump im all image 1 out.*.png type type size 200 200 box no 0.02 view 90 0 zoom 1.0'));
    const img = decodePng(b64(files.get('out.0.png')!));
    expect([img.w, img.h]).toEqual([200, 200]);
    const c = centroid(img, (r) => r > 0);
    expect(c.n).toBeGreaterThan(20);
    expect(Math.abs(c.x - 100)).toBeLessThanOrEqual(2);
    expect(Math.abs(c.y - 100)).toBeLessThanOrEqual(2);
  });

  it('default type colours: type 1 red, type 2 green', async () => {
    const input = `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 5.0 4.0 5.0
create_atoms 2 single 5.0 6.0 5.0
mass * 1.0
dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 view 90 0 zoom 1.0
run 0
`;
    const img = decodePpm(b64((await run(input)).get('out.0.ppm')!));
    const red = centroid(img, (r, g, b) => r > g && r > b && r > 20);
    const green = centroid(img, (r, g, b) => g > r && g > b && g > 20);
    expect(red.n).toBeGreaterThan(10);
    expect(green.n).toBeGreaterThan(10);
    expect(red.x).toBeLessThan(green.x); // +y to the right, atom 1 is at smaller y
  });

  it('view rotation moves an off-centre atom to the side the doc predicts', async () => {
    // +x atom: depth for view 90 0 (along +x) -> centre; for view 90 90 (along +y) -> to the left.
    const atom = (view: string) => base(`dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 view ${view} zoom 1.0`)
      .replace('create_atoms 1 single 5.0 5.0 5.0', 'create_atoms 1 single 7.0 5.0 5.0');
    const alongX = centroid(decodePpm(b64((await run(atom('90 0'))).get('out.0.ppm')!)), (r) => r > 0);
    const alongY = centroid(decodePpm(b64((await run(atom('90 90'))).get('out.0.ppm')!)), (r) => r > 0);
    expect(Math.abs(alongX.x - 100)).toBeLessThanOrEqual(2);
    expect(alongY.x).toBeLessThan(alongX.x - 10);
  });

  it('JPEG starts with FFD8 and ends with FFD9', async () => {
    const files = await run(base('dump im all image 1 out.*.jpg type type size 64 64 box no 0.02 view 90 0'));
    const url = files.get('out.0.jpg')!;
    expect(url.startsWith('data:image/jpeg;base64,')).toBe(true);
    const b = b64(url);
    expect([b[0], b[1]]).toEqual([0xff, 0xd8]);
    expect([b[b.length - 2], b[b.length - 1]]).toEqual([0xff, 0xd9]);
  });

  it('colours by a per-atom attribute through the default blue->red map', async () => {
    const input = `units lj
atom_style atomic
boundary f f f
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5.0 3.0 5.0
create_atoms 1 single 5.0 7.0 5.0
mass * 1.0
variable c atom y
dump im all image 1 out.*.ppm v_c type size 200 200 box no 0.02 view 90 0 zoom 1.0
run 0
`;
    const img = decodePpm(b64((await run(input)).get('out.0.ppm')!));
    // y=3 -> blue end, y=7 -> red end; both drawn
    const any = centroid(img, (r, g, b) => r + g + b > 20);
    expect(any.n).toBeGreaterThan(40);
    let blueish = 0, reddish = 0;
    for (let o = 0; o < img.rgb.length; o += 3) { if (img.rgb[o + 2] > img.rgb[o] + 20) blueish++; if (img.rgb[o] > img.rgb[o + 2] + 20) reddish++; }
    expect(blueish).toBeGreaterThan(10);
    expect(reddish).toBeGreaterThan(10);
  });

  it('honours dump_modify acolor and backcolor', async () => {
    const input = base('dump im all image 1 out.*.ppm type type size 100 100 box no 0.02 view 90 0\ndump_modify im acolor 1 blue backcolor white');
    const img = decodePpm(b64((await run(input)).get('out.0.ppm')!));
    const c = centroid(img, (r, g, b) => b > r + 20 && b > g + 20);
    expect(c.n).toBeGreaterThan(10);
    // background is white: a corner pixel is (255,255,255)
    expect([img.rgb[0], img.rgb[1], img.rgb[2]]).toEqual([255, 255, 255]);
  });

  it('draws bonds by default when the system has bonds', async () => {
    const data = `bonded

2 atoms
1 bonds
1 atom types
1 bond types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1.0

Atoms # molecular

1 1 1 5.0 4.0 5.0
2 1 1 5.0 6.0 5.0

Bonds

1 1 1 2
`;
    const input = `units lj
atom_style molecular
read_data d.data
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
bond_style harmonic
bond_coeff 1 100.0 2.0
dump im all image 1 out.*.ppm type type size 200 200 box no 0.02 view 90 0 zoom 1.0
run 0
`;
    const img = decodePpm(b64((await run(input, { 'd.data': data })).get('out.0.ppm')!));
    const c = centroid(img, (r, g, b) => r + g + b > 20);
    expect(c.n).toBeGreaterThan(40);
    // two atoms separated in y span more columns than a single sphere
    let minx = 1e9, maxx = -1;
    for (let y = 0; y < img.h; y++) for (let x = 0; x < img.w; x++) { const o = (y * img.w + x) * 3; if (img.rgb[o] + img.rgb[o + 1] + img.rgb[o + 2] > 20) { minx = Math.min(minx, x); maxx = Math.max(maxx, x); } }
    expect(maxx - minx).toBeGreaterThan(15);
  });

  it('colours and sizes by element (every type defaults to C)', async () => {
    const files = await run(base('dump im all image 1 out.*.ppm element element size 200 200 box no 0.02 view 90 0'));
    const img = decodePpm(b64(files.get('out.0.ppm')!));
    const c = centroid(img, (r, g, b) => r + g + b > 20);
    expect(c.n).toBeGreaterThan(20);
    // carbon is neutral grey: the three channels are equal
    let equal = 0, total = 0;
    for (let o = 0; o < img.rgb.length; o += 3) { if (img.rgb[o] + img.rgb[o + 1] + img.rgb[o + 2] > 20) { total++; if (img.rgb[o] === img.rgb[o + 1] && img.rgb[o] === img.rgb[o + 2]) equal++; } }
    expect(equal).toBe(total);
  });

  it('names an unsupported keyword instead of ignoring it', async () => {
    await expect(run(base('dump im all image 1 out.*.ppm type type ssao yes 5 0.5'))).rejects.toThrow(/ssao/);
    await expect(run(base('dump im all image 1 out.*.ppm type type axes yes 0.5 0.1'))).rejects.toThrow(/axes/);
  });

  it('requires a wildcard in the file name', async () => {
    await expect(run(base('dump im all image 1 out.ppm type type'))).rejects.toThrow(/\*/);
  });

  it('dump movie stays refused', async () => {
    await expect(run(base('dump im all movie 1 out.mpg type type'))).rejects.toThrow(/movie/);
  });
});
