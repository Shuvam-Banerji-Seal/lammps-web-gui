/*
 * gzip writer for write_data and other file names that end in .gz. The reader is gunzip.ts.
 *
 * write_data.rst (versionadded 11Feb2026): the file may be a compressed text file, detected by its suffix.
 * The output is a gzip member (RFC 1952) whose DEFLATE data (RFC 1951) uses stored blocks only: BTYPE 00,
 * blocks of at most 65535 bytes (RFC 1951 section 3.2.4). Stored blocks are valid gzip and keep this writer
 * small and dependency-free; they are larger than a compressed file.
 *
 * The engine keeps files as strings, one character per byte, so the result is returned as a latin1 string.
 * This file contains no LAMMPS source code.
 */

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 (RFC 1952 section 8) of a byte array. */
export const crc32 = (bytes: Uint8Array): number => {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};

const MAX_STORED = 65535;

/** A gzip member holding `data` in stored DEFLATE blocks. */
export const gzipStored = (data: Uint8Array): Uint8Array => {
  const blocks = Math.max(1, Math.ceil(data.length / MAX_STORED));
  const out = new Uint8Array(10 + data.length + 5 * blocks + 8);
  const dv = new DataView(out.buffer);
  // RFC 1952 section 2.3: ID1 ID2 CM FLG MTIME(4) XFL OS. No name, no comment, no extra field, MTIME 0,
  // XFL 0, OS 3 (Unix).
  out.set([0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0, 0x00, 0x03], 0);
  let o = 10;
  for (let b = 0; b < blocks; b++) {
    const start = b * MAX_STORED;
    const len = Math.min(MAX_STORED, data.length - start);
    const final = b === blocks - 1;
    // RFC 1951 section 3.2.3: BFINAL then BTYPE (2 bits, 00 = stored); the rest of the byte is skipped.
    out[o++] = final ? 1 : 0;
    // RFC 1951 section 3.2.4: LEN then its one's complement NLEN, least significant byte first.
    dv.setUint16(o, len, true);
    dv.setUint16(o + 2, len ^ 0xffff, true);
    o += 4;
    out.set(data.subarray(start, start + len), o);
    o += len;
  }
  // RFC 1952 section 2.3.1: CRC32 and ISIZE (size modulo 2^32), least significant byte first.
  dv.setUint32(o, crc32(data), true);
  dv.setUint32(o + 4, data.length >>> 0, true);
  return out.subarray(0, o + 8);
};

/** Text (ASCII or UTF-8) gzipped, returned as a latin1 string of the gzip bytes. */
export const gzipText = (text: string): string => {
  const bytes = gzipStored(new TextEncoder().encode(text));
  let s = '';
  for (let k = 0; k < bytes.length; k += 0x8000) s += String.fromCharCode(...bytes.subarray(k, k + 0x8000));
  return s;
};
