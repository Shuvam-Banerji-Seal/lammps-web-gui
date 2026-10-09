import { StyleError } from './force/types';

/**
 * gzip (RFC 1952) members with DEFLATE (RFC 1951) data -> the decompressed bytes. read_data and the other
 * file readers accept gzipped text "(detected by a .gz suffix)" (docs.lammps.org/read_data.html). Pure
 * TypeScript, no dependency.
 *
 * Reference: RFC 1952 (https://www.rfc-editor.org/rfc/rfc1952) and RFC 1951
 * (https://www.rfc-editor.org/rfc/rfc1951). This file contains no LAMMPS source.
 */

// ---------------------------------------------------------------- CRC-32 (RFC 1952 §8, poly 0xEDB88320)

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array, start: number, end: number): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------- DEFLATE tables (RFC 1951 §3.2.5)

const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/** RFC 1951 §3.2.7: the code-length alphabet is sent in this order. */
const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

// ---------------------------------------------------------------- output buffer (doubling, per task)

class Out {
  buf: Uint8Array;
  len = 0;
  constructor(cap: number) { this.buf = new Uint8Array(Math.max(cap, 1024)); }
  ensure(n: number): void {
    if (this.len + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.len + n) cap *= 2;
    const nb = new Uint8Array(cap);
    nb.set(this.buf.subarray(0, this.len));
    this.buf = nb;
  }
  /** Copy `length` bytes from `dist` back, handling a distance smaller than the run (overlap). */
  copy(dist: number, length: number): void {
    this.ensure(length);
    const b = this.buf;
    if (dist >= length) {
      b.copyWithin(this.len, this.len - dist, this.len - dist + length);
      this.len += length;
    } else {
      for (let i = 0; i < length; i++) { b[this.len] = b[this.len - dist]; this.len++; }
    }
  }
}

// ---------------------------------------------------------------- bit reader (LSB-first, RFC 1951 §3.1.1)

class BitReader {
  pos: number;
  bitBuf = 0;
  bitCount = 0;
  constructor(readonly data: Uint8Array, start: number, readonly end: number) { this.pos = start; }

  private fill(n: number): void {
    while (this.bitCount < n) {
      if (this.pos >= this.end) throw new StyleError('gzip: truncated DEFLATE stream');
      this.bitBuf |= this.data[this.pos++] << this.bitCount;
      this.bitCount += 8;
    }
  }

  bits(n: number): number {
    if (n === 0) return 0;
    this.fill(n);
    const v = this.bitBuf & ((1 << n) - 1);
    this.bitBuf >>>= n;
    this.bitCount -= n;
    return v;
  }

  /** Canonical-table decode: index by the next `maxBits` stream bits (zero-padded at end of stream). */
  decode(table: Int32Array, maxBits: number): number {
    while (this.bitCount < maxBits && this.pos < this.end) {
      this.bitBuf |= this.data[this.pos++] << this.bitCount;
      this.bitCount += 8;
    }
    const entry = table[this.bitBuf & ((1 << maxBits) - 1)];
    if (entry < 0) throw new StyleError('gzip: invalid Huffman code');
    const len = entry >>> 16;
    if (len > this.bitCount) throw new StyleError('gzip: truncated DEFLATE stream');
    this.bitBuf >>>= len;
    this.bitCount -= len;
    return entry & 0xffff;
  }

  /** Discard the bits of the current partial byte (block header / trailer alignment). */
  align(): void { const r = this.bitCount & 7; this.bitBuf >>>= r; this.bitCount -= r; }

  /** Byte offset of the next unread byte once `align()` has been called. */
  bytePos(): number { return this.pos - (this.bitCount >> 3); }
}

// ---------------------------------------------------------------- canonical Huffman (RFC 1951 §3.2.2)

function buildHuffman(lengths: Uint8Array, maxBits: number): Int32Array {
  const table = new Int32Array(1 << maxBits).fill(-1);
  const count = new Int32Array(maxBits + 1);
  for (let i = 0; i < lengths.length; i++) {
    const l = lengths[i];
    if (l) { count[l]++; if (l > maxBits) throw new StyleError('gzip: Huffman code length out of range'); }
  }
  const nextCode = new Int32Array(maxBits + 1);
  let code = 0;
  for (let bits = 1; bits <= maxBits; bits++) { code = (code + count[bits - 1]) << 1; nextCode[bits] = code; }
  for (let sym = 0; sym < lengths.length; sym++) {
    const len = lengths[sym];
    if (!len) continue;
    const c = nextCode[len]++;
    let rev = 0;
    for (let x = c, b = 0; b < len; b++) { rev = (rev << 1) | (x & 1); x >>>= 1; }
    const entry = sym | (len << 16);
    for (let i = rev; i < (1 << maxBits); i += (1 << len)) table[i] = entry;
  }
  return table;
}

function maxLen(lengths: Uint8Array): number {
  let m = 0;
  for (let i = 0; i < lengths.length; i++) if (lengths[i] > m) m = lengths[i];
  return m;
}

interface Huffman { table: Int32Array; maxBits: number; }

let fixedLit: Huffman | null = null;
let fixedDist: Huffman | null = null;

function fixedTables(): [Huffman, Huffman] {
  if (!fixedLit || !fixedDist) {
    // RFC 1951 §3.2.6 fixed code lengths
    const lit = new Uint8Array(288);
    for (let i = 0; i <= 143; i++) lit[i] = 8;
    for (let i = 144; i <= 255; i++) lit[i] = 9;
    for (let i = 256; i <= 279; i++) lit[i] = 7;
    for (let i = 280; i <= 287; i++) lit[i] = 8;
    fixedLit = { table: buildHuffman(lit, 9), maxBits: 9 };
    const dist = new Uint8Array(32).fill(5);
    fixedDist = { table: buildHuffman(dist, 5), maxBits: 5 };
  }
  return [fixedLit, fixedDist];
}

function dynamicTables(r: BitReader): [Huffman, Huffman] {
  const hlit = r.bits(5) + 257;
  const hdist = r.bits(5) + 1;
  const hclen = r.bits(4) + 4;
  const clLen = new Uint8Array(19);
  for (let i = 0; i < hclen; i++) clLen[CL_ORDER[i]] = r.bits(3);
  const clTable = buildHuffman(clLen, 7);
  const lengths = new Uint8Array(hlit + hdist);
  let i = 0;
  while (i < hlit + hdist) {
    const sym = r.decode(clTable, 7);
    if (sym < 16) {
      lengths[i++] = sym;
    } else if (sym === 16) {
      if (i === 0) throw new StyleError('gzip: code-length repeat with no previous length');
      const prev = lengths[i - 1];
      const rep = 3 + r.bits(2);
      for (let k = 0; k < rep; k++) lengths[i++] = prev;
    } else if (sym === 17) {
      const rep = 3 + r.bits(3);
      for (let k = 0; k < rep; k++) lengths[i++] = 0;
    } else {
      const rep = 11 + r.bits(7);
      for (let k = 0; k < rep; k++) lengths[i++] = 0;
    }
    if (i > hlit + hdist) throw new StyleError('gzip: code-length repeat overruns the table');
  }
  const litLen = lengths.subarray(0, hlit);
  const distLen = lengths.subarray(hlit);
  const litMax = maxLen(litLen);
  if (litMax === 0) throw new StyleError('gzip: empty literal/length Huffman tree');
  const distMax = Math.max(1, maxLen(distLen));
  return [
    { table: buildHuffman(litLen, litMax), maxBits: litMax },
    { table: buildHuffman(distLen, distMax), maxBits: distMax },
  ];
}

// ---------------------------------------------------------------- DEFLATE blocks (RFC 1951 §3.2.3)

function inflateBlock(r: BitReader, out: Out): void {
  const type = r.bits(2);
  if (type === 0) {
    r.align();
    const len = r.bits(8) | (r.bits(8) << 8);
    const nlen = r.bits(8) | (r.bits(8) << 8);
    if ((((len ^ 0xffff) & 0xffff)) !== nlen) throw new StyleError('gzip: stored block LEN/NLEN mismatch');
    out.ensure(len);
    let left = len;
    while (left > 0 && r.bitCount >= 8) { out.buf[out.len++] = r.bitBuf & 0xff; r.bitBuf >>>= 8; r.bitCount -= 8; left--; }
    if (left > 0) {
      if (r.pos + left > r.end) throw new StyleError('gzip: truncated DEFLATE stream');
      out.buf.set(r.data.subarray(r.pos, r.pos + left), out.len);
      out.len += left;
      r.pos += left;
    }
    return;
  }
  if (type === 3) throw new StyleError('gzip: invalid DEFLATE block type 3 (reserved)');
  const [lit, dist] = type === 1 ? fixedTables() : dynamicTables(r);
  for (;;) {
    const sym = r.decode(lit.table, lit.maxBits);
    if (sym < 256) { out.ensure(1); out.buf[out.len++] = sym; continue; }
    if (sym === 256) break;
    const li = sym - 257;
    if (li >= LENGTH_BASE.length) throw new StyleError('gzip: invalid length code');
    const length = LENGTH_BASE[li] + r.bits(LENGTH_EXTRA[li]);
    const dsym = r.decode(dist.table, dist.maxBits);
    if (dsym >= DIST_BASE.length) throw new StyleError('gzip: invalid distance code');
    const d = DIST_BASE[dsym] + r.bits(DIST_EXTRA[dsym]);
    if (d > out.len) throw new StyleError('gzip: distance too far back');
    out.copy(d, length);
  }
}

function inflate(r: BitReader, out: Out): void {
  for (;;) {
    const last = r.bits(1);
    inflateBlock(r, out);
    if (last) break;
  }
}

// ---------------------------------------------------------------- RFC 1952 member

function skipZeroTerminated(data: Uint8Array, p: number): number {
  while (p < data.length && data[p] !== 0) p++;
  if (p >= data.length) throw new StyleError('gzip: truncated header string');
  return p + 1;
}

/** Inflates one member starting at `pos`; returns the offset just past its trailer. */
function inflateMember(data: Uint8Array, pos: number, out: Out): number {
  if (data.length - pos < 10) throw new StyleError('gzip: truncated header');
  if (data[pos] !== 0x1f || data[pos + 1] !== 0x8b) throw new StyleError('gzip: bad magic number (not a gzip stream)');
  const cm = data[pos + 2];
  if (cm !== 8) throw new StyleError(`gzip: unsupported compression method ${cm} (only 8 = deflate)`);
  const flg = data[pos + 3];
  if (flg & 0xe0) throw new StyleError('gzip: reserved FLG bits are set');
  let p = pos + 10;
  if (flg & 0x04) { // FEXTRA
    if (p + 2 > data.length) throw new StyleError('gzip: truncated FEXTRA header');
    const xlen = data[p] | (data[p + 1] << 8);
    p += 2;
    if (p + xlen > data.length) throw new StyleError('gzip: truncated FEXTRA field');
    p += xlen;
  }
  if (flg & 0x08) p = skipZeroTerminated(data, p); // FNAME
  if (flg & 0x10) p = skipZeroTerminated(data, p); // FCOMMENT
  if (flg & 0x02) { // FHCRC
    if (p + 2 > data.length) throw new StyleError('gzip: truncated FHCRC');
    p += 2;
  }
  const startLen = out.len;
  const r = new BitReader(data, p, data.length);
  inflate(r, out);
  r.align();
  const trailer = r.bytePos();
  if (trailer + 8 > data.length) throw new StyleError('gzip: truncated stream (missing CRC-32/ISIZE trailer)');
  const wantCrc = (data[trailer] | (data[trailer + 1] << 8) | (data[trailer + 2] << 16) | (data[trailer + 3] << 24)) >>> 0;
  const wantSize = (data[trailer + 4] | (data[trailer + 5] << 8) | (data[trailer + 6] << 16) | (data[trailer + 7] << 24)) >>> 0;
  if (crc32(out.buf, startLen, out.len) !== wantCrc) throw new StyleError('gzip: CRC-32 mismatch (corrupted stream)');
  if (((out.len - startLen) >>> 0) !== wantSize) throw new StyleError('gzip: ISIZE mismatch (corrupted stream)');
  return trailer + 8;
}

/**
 * Decompresses one or more concatenated gzip (RFC 1952) members into the decompressed bytes. A bad
 * member header, an invalid/truncated DEFLATE stream, or a failed CRC-32/ISIZE check throws a
 * StyleError naming the problem.
 */
export const gunzip = (data: Uint8Array): Uint8Array => {
  const out = new Out(Math.max(1024, data.length * 3));
  let pos = 0;
  let members = 0;
  while (pos < data.length) {
    // gzip members may be followed by zero padding; a member never starts with 0x00
    if (data[pos] === 0) {
      let p = pos;
      while (p < data.length && data[p] === 0) p++;
      if (p === data.length) break;
      throw new StyleError('gzip: invalid header (unexpected zero byte between members)');
    }
    pos = inflateMember(data, pos, out);
    members++;
  }
  if (members === 0) throw new StyleError('gzip: empty input (not a gzip stream)');
  return out.buf.slice(0, out.len);
};
