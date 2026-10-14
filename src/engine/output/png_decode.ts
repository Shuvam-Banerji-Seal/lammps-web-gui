import { StyleError } from '../force/types';
import { inflateRaw } from '../gunzip';

/*
 * PNG reader for image labels (fix graphics/labels "image file.png", docs.lammps.org/fix_graphics_labels.html), and
 * the PNG format of the file layout of W3C PNG (https://www.w3.org/TR/png/): an 8-byte signature, then chunks of
 * length (4 bytes, big-endian), type (4 letters), data and CRC-32 over type and data. The image data is a zlib
 * stream (RFC 1950: two header bytes, a DEFLATE stream, an Adler-32 trailer) whose decompressed rows each start
 * with a filter type byte (PNG section 9: 0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth).
 *
 * Supported: 8-bit grey (colour type 0), RGB (2), palette (3, with optional tRNS alpha), grey + alpha (4) and RGBA (6),
 * not interlaced. Other bit depths and Adam7 interlacing throw a StyleError naming them. The result is always RGBA,
 * rows from the top. This file contains no LAMMPS source code.
 */

export interface DecodedPng {
  width: number;
  height: number;
  /** 4 bytes per pixel (R, G, B, A), rows from the top. */
  rgba: Uint8Array;
}

const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

const crc32 = (bytes: Uint8Array, start: number, end: number): number => {
  let c = 0xffffffff;
  for (let k = start; k < end; k++) c = CRC_TABLE[(c ^ bytes[k]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};

const u32 = (b: Uint8Array, p: number): number => ((b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]) >>> 0;

/** Paeth predictor (PNG section 9.4). */
const paeth = (a: number, b: number, c: number): number => {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  return pb <= pc ? b : c;
};

/** Decodes a PNG file given as its bytes (one character per byte, or a Uint8Array). */
export const decodePng = (input: Uint8Array | string): DecodedPng => {
  const bytes = typeof input === 'string' ? Uint8Array.from(input, (c) => c.charCodeAt(0) & 0xff) : input;
  if (bytes.length < 8 || SIGNATURE.some((v, k) => bytes[k] !== v)) throw new StyleError('PNG: not a PNG file (bad signature)');
  let width = 0, height = 0, depth = 0, ctype = -1, interlace = 0;
  let palette: Uint8Array | null = null;
  let trns: Uint8Array | null = null;
  const idat: Uint8Array[] = [];
  let idatLen = 0;
  let p = 8;
  let sawEnd = false;
  while (p + 8 <= bytes.length && !sawEnd) {
    const len = u32(bytes, p);
    const type = String.fromCharCode(bytes[p + 4], bytes[p + 5], bytes[p + 6], bytes[p + 7]);
    const dataStart = p + 8;
    const dataEnd = dataStart + len;
    if (dataEnd + 4 > bytes.length) throw new StyleError(`PNG: truncated ${type} chunk`);
    if (crc32(bytes, p + 4, dataEnd) !== u32(bytes, dataEnd)) throw new StyleError(`PNG: CRC error in ${type} chunk`);
    const data = bytes.subarray(dataStart, dataEnd);
    if (type === 'IHDR') {
      if (len !== 13) throw new StyleError('PNG: bad IHDR length');
      width = u32(data, 0);
      height = u32(data, 4);
      depth = data[8];
      ctype = data[9];
      interlace = data[12];
    } else if (type === 'PLTE') {
      palette = data.slice();
    } else if (type === 'tRNS') {
      trns = data.slice();
    } else if (type === 'IDAT') {
      idat.push(data);
      idatLen += data.length;
    } else if (type === 'IEND') {
      sawEnd = true;
    }
    p = dataEnd + 4;
  }
  if (width === 0 || height === 0) throw new StyleError('PNG: missing or empty IHDR');
  if (depth !== 8) throw new StyleError(`PNG: bit depth ${depth} is not supported (8-bit only)`);
  if (interlace !== 0) throw new StyleError('PNG: interlaced images are not supported');
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 } as Record<number, number>;
  if (!(ctype in channels)) throw new StyleError(`PNG: unknown colour type ${ctype}`);
  if (ctype === 3 && !palette) throw new StyleError('PNG: palette image without a PLTE chunk');
  if (idatLen < 6) throw new StyleError('PNG: no image data');

  // zlib stream: CMF/FLG header (CM 8, no preset dictionary), the DEFLATE data, then an Adler-32 trailer
  const z = new Uint8Array(idatLen);
  let zp = 0;
  for (const c of idat) { z.set(c, zp); zp += c.length; }
  if ((z[0] & 0x0f) !== 8 || ((z[0] << 8) | z[1]) % 31 !== 0 || z[1] & 0x20) throw new StyleError('PNG: bad zlib header');
  const raw = inflateRaw(z, 2);

  const bpp = channels[ctype];
  const stride = width * bpp;
  if (raw.length !== height * (stride + 1)) throw new StyleError(`PNG: image data has ${raw.length} bytes, expected ${height * (stride + 1)}`);
  const rows = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    for (let x = 0; x < stride; x++) {
      const left = x >= bpp ? rows[dst + x - bpp] : 0;
      const up = y > 0 ? rows[dst - stride + x] : 0;
      const upLeft = y > 0 && x >= bpp ? rows[dst - stride + x - bpp] : 0;
      let pred: number;
      switch (ft) {
        case 0: pred = 0; break;
        case 1: pred = left; break;
        case 2: pred = up; break;
        case 3: pred = (left + up) >> 1; break;
        case 4: pred = paeth(left, up, upLeft); break;
        default: throw new StyleError(`PNG: unknown filter type ${ft} on row ${y}`);
      }
      rows[dst + x] = (raw[src + x] + pred) & 0xff;
    }
  }

  // expand to RGBA
  const rgba = new Uint8Array(width * height * 4);
  const n = width * height;
  for (let i = 0; i < n; i++) {
    const o = 4 * i;
    if (ctype === 0) {
      rgba[o] = rgba[o + 1] = rgba[o + 2] = rows[i];
      rgba[o + 3] = 255;
    } else if (ctype === 2) {
      rgba[o] = rows[3 * i]; rgba[o + 1] = rows[3 * i + 1]; rgba[o + 2] = rows[3 * i + 2];
      rgba[o + 3] = 255;
    } else if (ctype === 3) {
      const idx = rows[i];
      if (3 * idx + 2 >= palette!.length) throw new StyleError(`PNG: palette index ${idx} is out of range`);
      rgba[o] = palette![3 * idx]; rgba[o + 1] = palette![3 * idx + 1]; rgba[o + 2] = palette![3 * idx + 2];
      rgba[o + 3] = trns && idx < trns.length ? trns[idx] : 255;
    } else if (ctype === 4) {
      rgba[o] = rgba[o + 1] = rgba[o + 2] = rows[2 * i];
      rgba[o + 3] = rows[2 * i + 1];
    } else {
      rgba[o] = rows[4 * i]; rgba[o + 1] = rows[4 * i + 1]; rgba[o + 2] = rows[4 * i + 2]; rgba[o + 3] = rows[4 * i + 3];
    }
  }
  return { width, height, rgba };
};
