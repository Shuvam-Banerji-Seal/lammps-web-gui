import { describe, it, expect } from 'vitest';
import { GifWriter, lzwEncode, medianCutPalette, paletteBits } from '../src/components/workbench/notebook/render/gif';

/** Reads the blocks of a GIF89a file (no decoding of pixels): header, logical screen, and each frame's size. */
const parseGif = (bytes: Uint8Array) => {
  const txt = String.fromCharCode(...bytes.subarray(0, 6));
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = dv.getUint16(6, true);
  const height = dv.getUint16(8, true);
  let p = 13;
  let loops: number | null = null;
  const frames: { w: number; h: number; delay: number; minCode: number; lzw: Uint8Array; bits: number; table: Uint8Array }[] = [];
  let pendingDelay = 0;
  while (p < bytes.length) {
    const b = bytes[p];
    if (b === 0x3b) { p++; break; }
    if (b === 0x21) {
      const label = bytes[p + 1];
      p += 2;
      if (label === 0xf9) {
        pendingDelay = dv.getUint16(p + 2, true);
        p += 1 + bytes[p];
        p += 1; // terminator
      } else {
        // application or other extension: sub-blocks
        if (label === 0xff) {
          const app = String.fromCharCode(...bytes.subarray(p + 1, p + 12));
          const len = bytes[p];
          if (app === 'NETSCAPE2.0') loops = dv.getUint16(p + len + 3, true);
          p += 1 + len;
        }
        while (bytes[p] !== 0) p += 1 + bytes[p];
        p++;
      }
    } else if (b === 0x2c) {
      const w = dv.getUint16(p + 5, true);
      const h = dv.getUint16(p + 7, true);
      const packed = bytes[p + 9];
      expect(packed & 0x80).toBe(0x80); // local colour table
      const bits = (packed & 7) + 1;
      const table = bytes.slice(p + 10, p + 10 + 3 * (1 << bits));
      p += 10 + 3 * (1 << bits);
      const minCode = bytes[p];
      p++;
      const parts: number[] = [];
      while (bytes[p] !== 0) {
        const n = bytes[p];
        for (let i = 1; i <= n; i++) parts.push(bytes[p + i]);
        p += 1 + n;
      }
      p++;
      frames.push({ w, h, delay: pendingDelay, minCode, lzw: Uint8Array.from(parts), bits, table });
    } else {
      throw new Error(`unexpected block 0x${b.toString(16)} at ${p}`);
    }
  }
  return { sig: txt, width, height, loops, frames, trailer: bytes[p - 1] === 0x3b, end: p };
};

/** GIF LZW decoder (the reverse of the encoder's code stream), for the round-trip checks. */
const lzwDecode = (data: Uint8Array, minCode: number, count: number): Uint8Array => {
  const clear = 1 << minCode;
  const eoi = clear + 1;
  let width = minCode + 1;
  let next = eoi + 1;
  const prefix = new Int32Array(4096);
  const suffix = new Uint8Array(4096);
  const lenOf = new Int32Array(4096);
  for (let i = 0; i < clear; i++) suffix[i] = i;
  let bitPos = 0;
  const readCode = (): number => {
    let code = 0;
    for (let i = 0; i < width; i++) {
      const byte = data[bitPos >> 3];
      code |= ((byte >> (bitPos & 7)) & 1) << i;
      bitPos++;
    }
    return code;
  };
  const out = new Uint8Array(count);
  let o = 0;
  const emitCode = (code: number): number => {
    // writes the string of `code` at out[o..], returns its first byte
    let len = 0;
    let c = code;
    while (c >= clear) { len++; c = prefix[c]; }
    const first = c;
    len++;
    let k = code;
    for (let i = len - 1; i >= 0; i--) {
      if (k < clear) { out[o + i] = k; break; }
      out[o + i] = suffix[k];
      k = prefix[k];
    }
    o += len;
    lenOf[next] = len;
    return first;
  };
  let prev = -1;
  let prevFirst = 0;
  for (;;) {
    if (bitPos + width > data.length * 8) break;
    const code = readCode();
    if (code === clear) { width = minCode + 1; next = eoi + 1; prev = -1; continue; }
    if (code === eoi) break;
    if (prev < 0) {
      const first = emitCode(code);
      prevFirst = first;
      prev = code;
      continue;
    }
    if (code < next) {
      // known code: output it, add prev + first byte of it
      const first = emitCode(code);
      prefix[next] = prev;
      suffix[next] = first;
      next++;
      prevFirst = first;
    } else {
      // KwKwK: the code being defined now
      prefix[next] = prev;
      suffix[next] = prevFirst;
      next++;
      const first = emitCode(code);
      prevFirst = first;
    }
    prev = code;
    if (next === (1 << width) && width < 12) width++;
  }
  return out.subarray(0, o);
};

const gradientFrame = (w: number, h: number, f: number): Uint8ClampedArray => {
  const px = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = 4 * (y * w + x);
    px[o] = (x * 255 / w + f * 40) & 255;
    px[o + 1] = (y * 255 / h) & 255;
    px[o + 2] = ((x * y + f * 7) * 3) & 255;
    px[o + 3] = 255;
  }
  return px;
};

describe('GIF89a writer', () => {
  it('writes the header, the screen size, the loop extension and one image per frame', () => {
    const w = new GifWriter(40, 30, { loops: 0 });
    for (let f = 0; f < 4; f++) w.addFrame(gradientFrame(40, 30, f), 4);
    expect(w.frames).toBe(4);
    const g = parseGif(w.finish());
    expect(g.sig).toBe('GIF89a');
    expect(g.width).toBe(40);
    expect(g.height).toBe(30);
    expect(g.loops).toBe(0);
    expect(g.frames.length).toBe(4);
    expect(g.trailer).toBe(true);
    for (const fr of g.frames) {
      expect(fr.w).toBe(40);
      expect(fr.h).toBe(30);
      expect(fr.delay).toBe(4);
    }
  });

  it('keeps the loop count and the per-frame delay', () => {
    const w = new GifWriter(8, 8, { loops: 3 });
    w.addFrame(gradientFrame(8, 8, 0), 12);
    w.addFrame(gradientFrame(8, 8, 1), 1);
    const g = parseGif(w.finish());
    expect(g.loops).toBe(3);
    expect(g.frames.map((f) => f.delay)).toEqual([12, 1]);
  });

  it('refuses frames after finish and bad sizes', () => {
    const w = new GifWriter(4, 4);
    w.addFrame(gradientFrame(4, 4, 0), 1);
    w.finish();
    expect(() => w.addFrame(gradientFrame(4, 4, 0), 1)).toThrow();
    expect(() => new GifWriter(0, 4)).toThrow(RangeError);
    expect(() => new GifWriter(70000, 4)).toThrow(RangeError);
    expect(() => new GifWriter(4, 4).addFrame(new Uint8ClampedArray(8), 1)).toThrow(RangeError);
  });

  it('decodes back to the exact pixels when the frame has few colours (LZW round trip, codes past 4096)', () => {
    const W = 301;
    const H = 211;
    const cols = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 10]];
    let seed = 12345;
    const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed; };
    const w = new GifWriter(W, H);
    const want: number[] = [];
    for (let f = 0; f < 2; f++) {
      const px = new Uint8ClampedArray(W * H * 4);
      for (let p = 0; p < W * H; p++) {
        const c = cols[rnd() % 4];
        px[4 * p] = c[0]; px[4 * p + 1] = c[1]; px[4 * p + 2] = c[2]; px[4 * p + 3] = 255;
        want.push(c[0], c[1], c[2]);
      }
      w.addFrame(px, 5);
    }
    const g = parseGif(w.finish());
    expect(g.frames.length).toBe(2);
    for (let f = 0; f < 2; f++) {
      const fr = g.frames[f];
      const idx = lzwDecode(fr.lzw, fr.minCode, W * H);
      expect(idx.length).toBe(W * H);
      // each pixel is its palette entry in the frame's own colour table: compare with the source colours
      let bad = 0;
      for (let p = 0; p < W * H; p++) {
        const j = idx[p];
        for (let c = 0; c < 3; c++) if (fr.table[3 * j + c] !== want[3 * (f * W * H + p) + c]) bad++;
      }
      expect(bad).toBe(0);
    }
  });

  it('round-trips a palette stream through lzwEncode and lzwDecode for every code width', () => {
    for (const bits of [2, 3, 5, 8]) {
      const n = 20000;
      const idx = new Uint8Array(n);
      let seed = 7 + bits;
      for (let i = 0; i < n; i++) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        idx[i] = (seed >>> 24) % (1 << bits);
      }
      const enc = lzwEncode(idx, bits);
      expect(Array.from(lzwDecode(enc, bits, n))).toEqual(Array.from(idx));
    }
  });
});

describe('median-cut palette', () => {
  it('finds the exact colours when there are few of them', () => {
    const cols = [[10, 20, 30], [200, 0, 0], [0, 255, 128]];
    const px = new Uint8ClampedArray(3 * 4 * 6);
    for (let p = 0; p < 18; p++) {
      const c = cols[p % 3];
      px[4 * p] = c[0]; px[4 * p + 1] = c[1]; px[4 * p + 2] = c[2]; px[4 * p + 3] = 255;
    }
    const pal = medianCutPalette(px, 18, 256);
    const got = new Set<string>();
    for (let j = 0; j < pal.length / 3; j++) got.add(`${pal[3 * j]},${pal[3 * j + 1]},${pal[3 * j + 2]}`);
    for (const c of cols) expect(got.has(c.join(','))).toBe(true);
  });

  it('never returns more than the maximum, and the bit width covers the palette', () => {
    const px = gradientFrame(64, 64, 0);
    const pal = medianCutPalette(px, 64 * 64, 256);
    expect(pal.length / 3).toBeLessThanOrEqual(256);
    expect(paletteBits(1)).toBe(2);
    expect(paletteBits(4)).toBe(2);
    expect(paletteBits(5)).toBe(3);
    expect(paletteBits(256)).toBe(8);
  });
});
