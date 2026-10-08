/*
 * Minimal PNG writer for dump image (docs.lammps.org/dump_image.html:
 * "if the suffix is ".png", then a PNG format is created"; PNG is
 * "lossless"). The browser engine has no compression library, so the zlib
 * stream uses stored (uncompressed) deflate blocks; the chunk CRCs and the
 * stream Adler-32 are still correct, so any PNG reader accepts the file.
 *
 * PNG layout (Portable Network Graphics spec, section 5): 8-byte signature,
 * then length/type/data/CRC chunks. IHDR is 8-bit truecolour (colour type 2).
 */

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

const crc32 = (bytes: Uint8Array): number => {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};

const adler32 = (bytes: Uint8Array): number => {
  let a = 1, b = 0;
  for (let i = 0; i < bytes.length; i++) {
    a += bytes[i];
    if (a >= 65521) a -= 65521;
    b += a;
    if (b >= 65521) b -= 65521;
  }
  return (((b << 16) | a) >>> 0);
};

const u32be = (v: number): Uint8Array => new Uint8Array([(v >>> 24) & 0xff, (v >>> 16) & 0xff, (v >>> 8) & 0xff, v & 0xff]);

/** zlib wrapper (RFC 1950) around stored deflate blocks (RFC 1951, BTYPE=00). */
const zlibStored = (data: Uint8Array): Uint8Array => {
  const nblocks = Math.max(1, Math.ceil(data.length / 65535));
  const out = new Uint8Array(2 + nblocks * 5 + data.length + 4);
  out[0] = 0x78; // CMF: deflate, 32K window
  out[1] = 0x01; // FLG: no dict, check bits make (CMF<<8|FLG) a multiple of 31
  let o = 2, off = 0;
  for (let b = 0; b < nblocks; b++) {
    const len = Math.min(65535, data.length - off);
    const final = b === nblocks - 1 ? 1 : 0;
    out[o++] = final; // BFINAL + BTYPE 00 (stored)
    out[o++] = len & 0xff;
    out[o++] = (len >>> 8) & 0xff;
    out[o++] = ~len & 0xff;
    out[o++] = (~len >>> 8) & 0xff;
    out.set(data.subarray(off, off + len), o);
    o += len;
    off += len;
  }
  const ad = adler32(data);
  out[o++] = (ad >>> 24) & 0xff;
  out[o++] = (ad >>> 16) & 0xff;
  out[o++] = (ad >>> 8) & 0xff;
  out[o++] = ad & 0xff;
  return out;
};

const chunk = (type: string, data: Uint8Array): Uint8Array => {
  const out = new Uint8Array(12 + data.length);
  out.set(u32be(data.length), 0);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crcInput = new Uint8Array(4 + data.length);
  for (let i = 0; i < 4; i++) crcInput[i] = out[4 + i];
  crcInput.set(data, 4);
  out.set(u32be(crc32(crcInput)), 8 + data.length);
  return out;
};

/** Encode 8-bit RGB pixels (length w*h*3) as a PNG file. */
export const encodePng = (w: number, h: number, rgb: Uint8Array): Uint8Array => {
  const ihdr = new Uint8Array(13);
  ihdr.set(u32be(w), 0);
  ihdr.set(u32be(h), 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // colour type: truecolour RGB
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace
  // filtered scanlines: one 0 filter byte in front of every RGB row
  const stride = w * 3;
  const raw = new Uint8Array((stride + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (stride + 1)] = 0;
    raw.set(rgb.subarray(y * stride, y * stride + stride), y * (stride + 1) + 1);
  }
  const ihdrChunk = chunk('IHDR', ihdr);
  const idatChunk = chunk('IDAT', zlibStored(raw));
  const iendChunk = chunk('IEND', new Uint8Array(0));
  const out = new Uint8Array(SIG.length + ihdrChunk.length + idatChunk.length + iendChunk.length);
  out.set(SIG, 0);
  out.set(ihdrChunk, SIG.length);
  out.set(idatChunk, SIG.length + ihdrChunk.length);
  out.set(iendChunk, SIG.length + ihdrChunk.length + idatChunk.length);
  return out;
};
