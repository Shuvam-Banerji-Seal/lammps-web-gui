/*
 * Minimal baseline JPEG writer for dump image (docs.lammps.org/dump_image.html:
 * "if the suffix is ".jpg" or ".jpeg", then a JPEG format file is created";
 * "JPEG images have lossy compression"). The browser engine ships no codec, so
 * this is a from-scratch baseline sequential DCT encoder: RGB -> YCbCr, 8x8
 * forward DCT, the JPEG Annex K luminance/chrominance quantisation and Huffman
 * tables (scaled for quality 85), 4:4:4 sampling. It emits SOI ... EOI.
 */

// JPEG Annex K.1 quantisation tables in natural (row) order.
const Q_LUMA = [
  16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
  14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
  49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const Q_CHROMA = [
  17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
  24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
  99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

// Zig-zag order: position k of the sequence is natural index ZIGZAG[k].
const ZIGZAG = [
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
  12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

// JPEG Annex K Huffman specifications: bit-length counts (index 1..16) + values.
const DC_LUMA_BITS = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_LUMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const DC_CHROMA_BITS = [0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_CHROMA_VALS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const AC_LUMA_BITS = [0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
const AC_LUMA_VALS = [
  0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
  0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
  0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
  0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
  0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
  0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
  0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
  0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
  0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
  0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];
const AC_CHROMA_BITS = [0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
const AC_CHROMA_VALS = [
  0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
  0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
  0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
  0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
  0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
  0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
  0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
  0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
  0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
  0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
  0xf9, 0xfa,
];

interface Huff {
  code: Int32Array; // code value per symbol (value byte)
  size: Int32Array; // code length per symbol
}

const buildHuff = (bits: number[], vals: number[]): Huff => {
  const code = new Int32Array(256).fill(-1);
  const size = new Int32Array(256);
  let huffcode = 0, k = 0;
  for (let len = 1; len <= 16; len++) {
    for (let i = 0; i < bits[len]; i++) {
      code[vals[k]] = huffcode;
      size[vals[k]] = len;
      huffcode++;
      k++;
    }
    huffcode <<= 1;
  }
  return { code, size };
};

const DC_LUMA = buildHuff(DC_LUMA_BITS, DC_LUMA_VALS);
const DC_CHROMA = buildHuff(DC_CHROMA_BITS, DC_CHROMA_VALS);
const AC_LUMA = buildHuff(AC_LUMA_BITS, AC_LUMA_VALS);
const AC_CHROMA = buildHuff(AC_CHROMA_BITS, AC_CHROMA_VALS);

// Cosine basis: base[u][x] = cos((2x+1) u pi / 16) * (u == 0 ? 1/sqrt2 : 1).
const COS = (() => {
  const c = new Float64Array(64);
  for (let u = 0; u < 8; u++) for (let x = 0; x < 8; x++) {
    c[u * 8 + x] = Math.cos(((2 * x + 1) * u * Math.PI) / 16) * (u === 0 ? Math.SQRT1_2 : 1);
  }
  return c;
})();

/** Separable 8x8 forward DCT of a block (natural order). */
const fdct = (block: Float64Array): Float64Array => {
  const tmp = new Float64Array(64);
  const out = new Float64Array(64);
  for (let y = 0; y < 8; y++) {
    for (let u = 0; u < 8; u++) {
      let s = 0;
      for (let x = 0; x < 8; x++) s += block[y * 8 + x] * COS[u * 8 + x];
      tmp[y * 8 + u] = s;
    }
  }
  for (let u = 0; u < 8; u++) {
    for (let v = 0; v < 8; v++) {
      let s = 0;
      for (let y = 0; y < 8; y++) s += tmp[y * 8 + u] * COS[v * 8 + y];
      out[v * 8 + u] = 0.25 * s;
    }
  }
  return out;
};

const qualityScale = (quality: number): number => (quality < 50 ? 5000 / quality : 200 - 2 * quality);

const scaleTable = (table: number[], scale: number): Int32Array => {
  const out = new Int32Array(64);
  for (let i = 0; i < 64; i++) out[i] = Math.max(1, Math.min(255, Math.floor((table[i] * scale + 50) / 100)));
  return out;
};

class ByteWriter {
  private buf: number[] = [];
  push(v: number): void { this.buf.push(v & 0xff); }
  /** Entropy-coded data only: a 0xFF byte is followed by a stuffed 0x00. */
  pushEntropy(v: number): void {
    this.buf.push(v & 0xff);
    if ((v & 0xff) === 0xff) this.buf.push(0);
  }
  pushU16(v: number): void { this.push((v >>> 8) & 0xff); this.push(v & 0xff); }
  bytes(): Uint8Array { return Uint8Array.from(this.buf); }
}

class BitWriter {
  private out: ByteWriter;
  private acc = 0;
  private nbits = 0;
  constructor(out: ByteWriter) { this.out = out; }
  write(value: number, size: number): void {
    for (let i = size - 1; i >= 0; i--) {
      this.acc = (this.acc << 1) | ((value >>> i) & 1);
      this.nbits++;
      if (this.nbits === 8) { this.out.pushEntropy(this.acc); this.acc = 0; this.nbits = 0; }
    }
  }
  flush(): void {
    while (this.nbits > 0) { this.acc = (this.acc << 1) | 1; this.nbits++; if (this.nbits === 8) { this.out.pushEntropy(this.acc); this.acc = 0; this.nbits = 0; } }
  }
}

const category = (v: number): number => {
  let a = Math.abs(v), c = 0;
  while (a) { c++; a >>= 1; }
  return c;
};

const encodeBlock = (bw: BitWriter, q: Int32Array, coef: Float64Array, dc: Huff, ac: Huff, prev: { dc: number }): void => {
  const zz = new Int32Array(64);
  for (let k = 0; k < 64; k++) {
    const nat = ZIGZAG[k];
    zz[k] = Math.round(coef[nat] / q[nat]);
  }
  const diff = zz[0] - prev.dc;
  prev.dc = zz[0];
  const cat = category(diff);
  bw.write(dc.code[cat], dc.size[cat]);
  if (cat) bw.write(diff >= 0 ? diff : diff + (1 << cat) - 1, cat);
  let run = 0;
  for (let k = 1; k < 64; k++) {
    const v = zz[k];
    if (v === 0) { run++; continue; }
    while (run > 15) { bw.write(ac.code[0xf0], ac.size[0xf0]); run -= 16; }
    const c = category(v);
    const sym = (run << 4) | c;
    bw.write(ac.code[sym], ac.size[sym]);
    bw.write(v >= 0 ? v : v + (1 << c) - 1, c);
    run = 0;
  }
  if (run > 0) bw.write(ac.code[0x00], ac.size[0x00]);
};

/** Encode 8-bit RGB pixels (length w*h*3) as a baseline JPEG file. */
export const encodeJpeg = (w: number, h: number, rgb: Uint8Array, quality = 85): Uint8Array => {
  const scale = qualityScale(quality);
  const qLuma = scaleTable(Q_LUMA, scale);
  const qChroma = scaleTable(Q_CHROMA, scale);
  const mw = w, mh = h;
  const yPlane = new Uint8Array(mw * mh);
  const cbPlane = new Uint8Array(mw * mh);
  const crPlane = new Uint8Array(mw * mh);
  for (let i = 0; i < mw * mh; i++) {
    const r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    yPlane[i] = Math.max(0, Math.min(255, Math.round(y)));
    cbPlane[i] = Math.max(0, Math.min(255, Math.round(-0.168736 * r - 0.331264 * g + 0.5 * b + 128)));
    crPlane[i] = Math.max(0, Math.min(255, Math.round(0.5 * r - 0.418688 * g - 0.081312 * b + 128)));
  }

  const out = new ByteWriter();
  out.push(0xff); out.push(0xd8); // SOI

  // APP0 / JFIF
  out.push(0xff); out.push(0xe0);
  out.pushU16(16);
  for (const ch of 'JFIF\0') out.push(ch.charCodeAt(0));
  out.push(1); out.push(1); out.push(0); out.pushU16(1); out.pushU16(1); out.push(0); out.push(0);

  // DQT: two tables in zig-zag order
  out.push(0xff); out.push(0xdb);
  out.pushU16(2 + 65 * 2);
  out.push(0x00);
  for (let k = 0; k < 64; k++) out.push(qLuma[ZIGZAG[k]]);
  out.push(0x01);
  for (let k = 0; k < 64; k++) out.push(qChroma[ZIGZAG[k]]);

  // SOF0: baseline, 8-bit, 3 components 4:4:4
  out.push(0xff); out.push(0xc0);
  out.pushU16(8 + 3 * 3);
  out.push(8);
  out.pushU16(mh); out.pushU16(mw);
  out.push(3);
  out.push(1); out.push(0x11); out.push(0); // Y, 1x1, qtable 0
  out.push(2); out.push(0x11); out.push(1); // Cb
  out.push(3); out.push(0x11); out.push(1); // Cr

  // DHT: four tables
  const writeDht = (cls: number, id: number, bits: number[], vals: number[]): void => {
    out.push(0xff); out.push(0xc4);
    out.pushU16(2 + 1 + 16 + vals.length);
    out.push((cls << 4) | id);
    for (let i = 1; i <= 16; i++) out.push(bits[i]);
    for (const v of vals) out.push(v);
  };
  writeDht(0, 0, DC_LUMA_BITS, DC_LUMA_VALS);
  writeDht(1, 0, AC_LUMA_BITS, AC_LUMA_VALS);
  writeDht(0, 1, DC_CHROMA_BITS, DC_CHROMA_VALS);
  writeDht(1, 1, AC_CHROMA_BITS, AC_CHROMA_VALS);

  // SOS
  out.push(0xff); out.push(0xda);
  out.pushU16(6 + 2 * 3);
  out.push(3);
  out.push(1); out.push(0x00);
  out.push(2); out.push(0x11);
  out.push(3); out.push(0x11);
  out.push(0); out.push(63); out.push(0);

  const bw = new BitWriter(out);
  const block = new Float64Array(64);
  const prevY = { dc: 0 }, prevCb = { dc: 0 }, prevCr = { dc: 0 };
  const blockAt = (plane: Uint8Array, bx: number, by: number): void => {
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const px = Math.min(mw - 1, bx * 8 + x);
      const py = Math.min(mh - 1, by * 8 + y);
      const v = plane[py * mw + px];
      block[y * 8 + x] = v - 128;
    }
  };
  const bwX = Math.ceil(mw / 8), bwY = Math.ceil(mh / 8);
  for (let by = 0; by < bwY; by++) {
    for (let bx = 0; bx < bwX; bx++) {
      blockAt(yPlane, bx, by);
      encodeBlock(bw, qLuma, fdct(block), DC_LUMA, AC_LUMA, prevY);
      blockAt(cbPlane, bx, by);
      encodeBlock(bw, qChroma, fdct(block), DC_CHROMA, AC_CHROMA, prevCb);
      blockAt(crPlane, bx, by);
      encodeBlock(bw, qChroma, fdct(block), DC_CHROMA, AC_CHROMA, prevCr);
    }
  }
  bw.flush();
  out.push(0xff); out.push(0xd9); // EOI
  return out.bytes();
};
