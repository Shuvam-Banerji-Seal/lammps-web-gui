/*
 * Zstandard writer for dump files whose name ends in .zst and for the COMPRESS zstd dump styles (atom/zstd,
 * custom/zstd, ...). Any Zstandard decoder reads the files (the zstd tool, for one).
 *
 * RFC 8878 (Zstandard Compression and the 'application/zstd' Media Type): a frame is the magic number 0xFD2FB528, a
 * frame header, blocks and an optional content checksum (section 3.1.1). This writer uses Raw blocks only
 * (Block_Type 0, section 3.1.1.2.2: "the block contains Block_Size bytes" stored as they are), which are valid
 * Zstandard and keep it small and dependency-free; the file is larger than a compressed one. The checksum is "the
 * lowest 4 bytes of the XXH64 digest of the original data, with a seed of zero" (section 3.1.1, Content_Checksum),
 * XXH64 as specified at https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md.
 *
 * The engine keeps files as strings, one character per byte, so the result is returned as a latin1 string.
 * This file contains no LAMMPS source code.
 */

// XXH64 on unsigned 64-bit values held as two uint32 halves; each operation leaves its result in (H, L).
let H = 0, L = 0;

const P1H = 0x9e3779b1, P1L = 0x85ebca87;
const P2H = 0xc2b2ae3d, P2L = 0x27d4eb4f;
const P3H = 0x165667b1, P3L = 0x9e3779f9;
const P4H = 0x85ebca77, P4L = 0xc2b2ae63;
const P5H = 0x27d4eb2f, P5L = 0x165667c5;

const add = (ah: number, al: number, bh: number, bl: number): void => {
  const lo = al + bl;
  L = lo >>> 0;
  H = (ah + bh + (lo > 0xffffffff ? 1 : 0)) >>> 0;
};

/** (a * b) mod 2^64. */
const mul = (ah: number, al: number, bh: number, bl: number): void => {
  const a0 = al & 0xffff, a1 = al >>> 16, b0 = bl & 0xffff, b1 = bl >>> 16;
  const mid = a0 * b1 + a1 * b0;
  const lo = a0 * b0 + (mid % 0x10000) * 0x10000;
  L = lo >>> 0;
  H = (a1 * b1 + Math.floor(mid / 0x10000) + Math.floor(lo / 0x100000000) + Math.imul(ah, bl) + Math.imul(al, bh)) >>> 0;
};

/** Rotate left by 0 < r < 32. */
const rotl = (h: number, l: number, r: number): void => {
  H = ((h << r) | (l >>> (32 - r))) >>> 0;
  L = ((l << r) | (h >>> (32 - r))) >>> 0;
};

/** XXH64 round: acc = rotl(acc + input * PRIME64_2, 31) * PRIME64_1. */
const round = (ah: number, al: number, ih: number, il: number): void => {
  mul(ih, il, P2H, P2L);
  add(ah, al, H, L);
  rotl(H, L, 31);
  mul(H, L, P1H, P1L);
};

/** XXH64 merge: acc = (acc ^ round(0, val)) * PRIME64_1 + PRIME64_4. */
const merge = (ah: number, al: number, vh: number, vl: number): void => {
  round(0, 0, vh, vl);
  const xh = (ah ^ H) >>> 0, xl = (al ^ L) >>> 0;
  mul(xh, xl, P1H, P1L);
  add(H, L, P4H, P4L);
};

const u32 = (b: Uint8Array, i: number): number => (b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (b[i + 3] << 24)) >>> 0;

/** XXH64 of the bytes with seed 0, as [high 32 bits, low 32 bits]. */
export const xxh64 = (b: Uint8Array): [number, number] => {
  const n = b.length;
  let i = 0;
  let hh: number, hl: number;
  if (n >= 32) {
    // v1 = seed + P1 + P2, v2 = seed + P2, v3 = seed, v4 = seed - P1
    add(P1H, P1L, P2H, P2L); let v1h = H, v1l = L;
    let v2h = P2H, v2l = P2L;
    let v3h = 0, v3l = 0;
    add(~P1H >>> 0, ~P1L >>> 0, 0, 1); let v4h = H, v4l = L;
    for (; i + 32 <= n; i += 32) {
      round(v1h, v1l, u32(b, i + 4), u32(b, i)); v1h = H; v1l = L;
      round(v2h, v2l, u32(b, i + 12), u32(b, i + 8)); v2h = H; v2l = L;
      round(v3h, v3l, u32(b, i + 20), u32(b, i + 16)); v3h = H; v3l = L;
      round(v4h, v4l, u32(b, i + 28), u32(b, i + 24)); v4h = H; v4l = L;
    }
    rotl(v1h, v1l, 1); hh = H; hl = L;
    rotl(v2h, v2l, 7); add(hh, hl, H, L); hh = H; hl = L;
    rotl(v3h, v3l, 12); add(hh, hl, H, L); hh = H; hl = L;
    rotl(v4h, v4l, 18); add(hh, hl, H, L); hh = H; hl = L;
    merge(hh, hl, v1h, v1l); merge(H, L, v2h, v2l); merge(H, L, v3h, v3l); merge(H, L, v4h, v4l);
    hh = H; hl = L;
  } else {
    hh = P5H; hl = P5L;
  }
  // h64 += length (below 2^53)
  add(hh, hl, Math.floor(n / 0x100000000), n >>> 0); hh = H; hl = L;
  for (; i + 8 <= n; i += 8) {
    round(0, 0, u32(b, i + 4), u32(b, i));
    const xh = (hh ^ H) >>> 0, xl = (hl ^ L) >>> 0;
    rotl(xh, xl, 27); mul(H, L, P1H, P1L); add(H, L, P4H, P4L); hh = H; hl = L;
  }
  if (i + 4 <= n) {
    mul(0, u32(b, i), P1H, P1L);
    const xh = (hh ^ H) >>> 0, xl = (hl ^ L) >>> 0;
    rotl(xh, xl, 23); mul(H, L, P2H, P2L); add(H, L, P3H, P3L); hh = H; hl = L;
    i += 4;
  }
  for (; i < n; i++) {
    mul(0, b[i], P5H, P5L);
    const xh = (hh ^ H) >>> 0, xl = (hl ^ L) >>> 0;
    rotl(xh, xl, 11); mul(H, L, P1H, P1L); hh = H; hl = L;
  }
  // avalanche: h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32
  hl = (hl ^ (hh >>> 1)) >>> 0;
  mul(hh, hl, P2H, P2L); hh = H; hl = L;
  hl = (hl ^ ((hl >>> 29) | (hh << 3))) >>> 0; hh = (hh ^ (hh >>> 29)) >>> 0;
  mul(hh, hl, P3H, P3L); hh = H; hl = L;
  hl = (hl ^ hh) >>> 0;
  return [hh, hl];
};

/** RFC 8878 section 3.1.1.2.4: Block_Maximum_Size is "the smallest of Window_Size or 128 KB". */
const MAX_BLOCK = 128 * 1024;

/** One Zstandard frame holding `data` in Raw blocks, with the content checksum when asked. */
export const zstdRawFrame = (data: Uint8Array, checksum: boolean): Uint8Array => {
  const blocks = Math.max(1, Math.ceil(data.length / MAX_BLOCK));
  const out = new Uint8Array(4 + 2 + 3 * blocks + data.length + (checksum ? 4 : 0));
  // magic number 0xFD2FB528, little-endian
  out.set([0x28, 0xb5, 0x2f, 0xfd], 0);
  // Frame_Header_Descriptor: no content size, not single-segment, Content_Checksum_flag (bit 2), no dictionary;
  // Window_Descriptor: Exponent 7, Mantissa 0, a 2^(10+7) = 128 KB window
  out[4] = checksum ? 0x04 : 0x00;
  out[5] = 7 << 3;
  let o = 6;
  for (let k = 0; k < blocks; k++) {
    const start = k * MAX_BLOCK;
    const len = Math.min(MAX_BLOCK, data.length - start);
    // Block_Header (3 bytes, little-endian): Last_Block (bit 0), Block_Type 0 = Raw (bits 1-2), Block_Size (bits 3-23)
    const hdr = (len << 3) | (k === blocks - 1 ? 1 : 0);
    out[o] = hdr & 0xff; out[o + 1] = (hdr >>> 8) & 0xff; out[o + 2] = (hdr >>> 16) & 0xff;
    o += 3;
    out.set(data.subarray(start, start + len), o);
    o += len;
  }
  if (checksum) {
    const low = xxh64(data)[1];
    out[o] = low & 0xff; out[o + 1] = (low >>> 8) & 0xff; out[o + 2] = (low >>> 16) & 0xff; out[o + 3] = low >>> 24;
  }
  return out;
};

/** Text (ASCII or UTF-8) as one Zstandard frame, returned as a latin1 string of its bytes. */
export const zstdText = (text: string, checksum = true): string => {
  const bytes = zstdRawFrame(new TextEncoder().encode(text), checksum);
  let s = '';
  for (let k = 0; k < bytes.length; k += 0x8000) s += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
  return s;
};
