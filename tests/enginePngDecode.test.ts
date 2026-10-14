import { describe, expect, it } from 'vitest';
import { decodePng } from '../src/engine/output/png_decode';
import { encodePng } from '../src/engine/output/png';

/*
 * PNG reader for fix graphics/labels "image" (src/engine/output/png_decode.ts). The round trip uses the engine's own
 * encoder (src/engine/output/png.ts); the filter cases are hand-built PNG files whose decoded pixels are worked out
 * from the PNG filter rules (W3C PNG, section 9): Sub adds the left byte, Up the byte above, Paeth the predictor.
 */

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

const crc = (b: Uint8Array): number => {
  let c = 0xffffffff;
  for (const x of b) c = CRC[(c ^ x) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};

const be32 = (v: number): number[] => [(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff];

const chunk = (type: string, data: number[]): number[] => {
  const td = [...type].map((c) => c.charCodeAt(0)).concat(data);
  return [...be32(data.length), ...td, ...be32(crc(Uint8Array.from(td)))];
};

/** A zlib stream of stored (uncompressed) DEFLATE blocks, the form the engine's encoder writes. */
const zlibStored = (raw: number[]): number[] => {
  const out = [0x78, 0x01];
  for (let p = 0; p < raw.length || p === 0; p += 65535) {
    const part = raw.slice(p, p + 65535);
    const last = p + 65535 >= raw.length ? 1 : 0;
    out.push(last, part.length & 0xff, part.length >> 8, (~part.length) & 0xff, ((~part.length) >> 8) & 0xff, ...part);
    if (last) break;
  }
  let a = 1, b = 0;
  for (const x of raw) { a = (a + x) % 65521; b = (b + a) % 65521; }
  return [...out, ...be32(((b << 16) | a) >>> 0)];
};

/** A PNG file from rows: each row is [filter type, ...filtered bytes]. */
const pngOf = (w: number, h: number, colourType: number, rows: number[][], extra: { plte?: number[]; trns?: number[]; depth?: number } = {}): Uint8Array => {
  const ihdr = [...be32(w), ...be32(h), extra.depth ?? 8, colourType, 0, 0, 0];
  const raw = rows.flat();
  const parts = [
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
    ...chunk('IHDR', ihdr),
    ...(extra.plte ? chunk('PLTE', extra.plte) : []),
    ...(extra.trns ? chunk('tRNS', extra.trns) : []),
    ...chunk('IDAT', zlibStored(raw)),
    ...chunk('IEND', []),
  ];
  return Uint8Array.from(parts);
};

describe('PNG decoder', () => {
  it('reads back the RGB file the engine encodes (all pixels, alpha 255)', () => {
    const w = 3, h = 2;
    const rgb = Uint8Array.from([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30, 40, 50, 60]);
    const png = decodePng(encodePng(w, h, rgb));
    expect(png.width).toBe(3);
    expect(png.height).toBe(2);
    for (let i = 0; i < w * h; i++) {
      expect([...png.rgba.subarray(4 * i, 4 * i + 4)]).toEqual([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255]);
    }
  });

  it('accepts a string of the file bytes (one character per byte), as the engine file store holds it', () => {
    const bytes = encodePng(1, 1, Uint8Array.from([7, 8, 9]));
    const text = String.fromCharCode(...bytes);
    expect([...decodePng(text).rgba]).toEqual([7, 8, 9, 255]);
  });

  it('undoes the Sub, Up and Paeth filters with the PNG rules (RGB, 2 rows)', () => {
    // row 0 (Sub): raw (1,2,3) then (4,5,6); the left neighbour adds: (1,2,3), (1+4,2+5,3+6) = (5,7,9)
    // row 1 (Up): raw (1,1,1) then (2,2,2) added to the pixel above: (2,3,4), (7,9,11)
    const png = decodePng(pngOf(2, 2, 2, [[1, 1, 2, 3, 4, 5, 6], [2, 1, 1, 1, 2, 2, 2]]));
    expect([...png.rgba]).toEqual([1, 2, 3, 255, 5, 7, 9, 255, 2, 3, 4, 255, 7, 9, 11, 255]);
  });

  it('undoes the Paeth filter with a predictor that picks the upper neighbour', () => {
    // 1 x 2 grey: row 0 is 100 (None); row 1 is raw 1 with Paeth: left 0, up 100, upper-left 0, so the predictor is 100
    const png = decodePng(pngOf(1, 2, 0, [[0, 100], [4, 1]]));
    expect([...png.rgba]).toEqual([100, 100, 100, 255, 101, 101, 101, 255]);
  });

  it('reads grey + alpha, RGBA and palette images with transparency', () => {
    const ga = decodePng(pngOf(1, 1, 4, [[0, 30, 200]]));
    expect([...ga.rgba]).toEqual([30, 30, 30, 200]);
    const rgba = decodePng(pngOf(1, 1, 6, [[0, 1, 2, 3, 4]]));
    expect([...rgba.rgba]).toEqual([1, 2, 3, 4]);
    const pal = decodePng(pngOf(2, 1, 3, [[0, 1, 0]], { plte: [255, 0, 0, 0, 0, 255], trns: [0] }));
    expect([...pal.rgba]).toEqual([0, 0, 255, 255, 255, 0, 0, 0]);
  });

  it('refuses a bad signature, a CRC error, interlacing, other bit depths and a palette with no PLTE', () => {
    const good = pngOf(1, 1, 2, [[0, 1, 2, 3]]);
    expect(() => decodePng(good.map((b, k) => (k === 0 ? 0 : b)))).toThrow(/bad signature/);
    const bad = Uint8Array.from(good);
    bad[41] ^= 0xff; // the first byte of the IDAT data (the signature, IHDR chunk of 25 bytes, then 8 bytes of IDAT header)
    expect(() => decodePng(bad)).toThrow(/CRC error/);
    const interlaced = pngOf(1, 1, 2, [[0, 1, 2, 3]]);
    // IHDR interlace byte is the last byte of the IHDR data: change it and rebuild the CRC
    const ihdrStart = 8 + 8;
    const ihdrData = [...interlaced.subarray(ihdrStart, ihdrStart + 13)];
    ihdrData[12] = 1;
    const rebuilt = Uint8Array.from([...interlaced.subarray(0, 8), ...chunk('IHDR', ihdrData), ...interlaced.subarray(ihdrStart + 13 + 4)]);
    expect(() => decodePng(rebuilt)).toThrow(/interlaced/);
    expect(() => decodePng(pngOf(1, 1, 2, [[0, 1, 2, 3, 0, 0, 0]], { depth: 16 }))).toThrow(/bit depth 16/);
    expect(() => decodePng(pngOf(1, 1, 3, [[0, 0]]))).toThrow(/without a PLTE/);
  });
});
