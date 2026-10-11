/**
 * Animated GIF (GIF89a) writer for the Render panel: a streaming encoder with a palette per frame (median cut
 * over a sample of the frame's pixels, at most 256 colours), nearest-colour mapping through a 15-bit lookup, and
 * GIF LZW compression. Frames are written as they are added, so memory holds the compressed bytes only.
 * Written from the GIF89a specification (the Graphics Interchange Format, CompuServe 1989 and 1990 revision); no
 * third-party code. Pure: no DOM.
 */

const MAX_COLOURS = 256;
/** Pixels sampled per frame for the palette; larger frames are sampled on a regular stride. */
const PALETTE_SAMPLES = 40_000;
const HASH_BITS = 14;
const HASH_SIZE = 1 << HASH_BITS;
const MAX_CODES = 4096;

const pushLE16 = (out: number[], v: number): void => {
  out.push(v & 0xff, (v >> 8) & 0xff);
};

/**
 * Palette of up to `max` colours for the given RGBA pixels: median cut on a sample. Each step splits the box with
 * the largest (population × widest channel range) at the median of its widest channel; a box's colour is the
 * mean of its samples.
 */
export const medianCutPalette = (rgba: ArrayLike<number>, pixels: number, max = MAX_COLOURS): Uint8Array => {
  const stride = Math.max(1, Math.floor(pixels / PALETTE_SAMPLES));
  const m = Math.ceil(pixels / stride);
  const R = new Uint8Array(m);
  const G = new Uint8Array(m);
  const B = new Uint8Array(m);
  let k = 0;
  for (let p = 0; p < pixels; p += stride) {
    R[k] = rgba[4 * p];
    G[k] = rgba[4 * p + 1];
    B[k] = rgba[4 * p + 2];
    k++;
  }
  const idx = new Uint32Array(k);
  for (let i = 0; i < k; i++) idx[i] = i;
  const chan = [R, G, B];

  type Box = { lo: number; hi: number; axis: number; score: number };
  const extent = (lo: number, hi: number): { axis: number; range: number } => {
    let best = { axis: 0, range: -1 };
    for (let c = 0; c < 3; c++) {
      const arr = chan[c];
      let mn = 255;
      let mx = 0;
      for (let i = lo; i < hi; i++) {
        const v = arr[idx[i]];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const range = mx - mn;
      if (range > best.range) best = { axis: c, range };
    }
    return best;
  };
  const boxOf = (lo: number, hi: number): Box => {
    const e = extent(lo, hi);
    return { lo, hi, axis: e.axis, score: (hi - lo) * e.range };
  };

  const boxes: Box[] = k > 0 ? [boxOf(0, k)] : [];
  while (boxes.length < max) {
    let pick = -1;
    let best = 0;
    for (let b = 0; b < boxes.length; b++) {
      if (boxes[b].hi - boxes[b].lo >= 2 && boxes[b].score > best) {
        best = boxes[b].score;
        pick = b;
      }
    }
    if (pick < 0) break;
    const { lo, hi, axis } = boxes[pick];
    const arr = chan[axis];
    idx.subarray(lo, hi).sort((a, b) => arr[a] - arr[b]);
    const mid = lo + ((hi - lo) >> 1);
    boxes[pick] = boxOf(lo, mid);
    boxes.push(boxOf(mid, hi));
  }

  if (boxes.length === 0) return new Uint8Array(3);
  const out = new Uint8Array(3 * boxes.length);
  boxes.forEach((b, j) => {
    let r = 0;
    let g = 0;
    let bl = 0;
    for (let i = b.lo; i < b.hi; i++) {
      const s = idx[i];
      r += R[s];
      g += G[s];
      bl += B[s];
    }
    const n = Math.max(1, b.hi - b.lo);
    out[3 * j] = Math.round(r / n);
    out[3 * j + 1] = Math.round(g / n);
    out[3 * j + 2] = Math.round(bl / n);
  });
  return out;
};

/** Index of the nearest palette colour to (r, g, b) by squared distance. */
const nearest = (pal: Uint8Array, r: number, g: number, b: number): number => {
  let best = 0;
  let bestD = Infinity;
  for (let j = 0; j < pal.length / 3; j++) {
    const dr = pal[3 * j] - r;
    const dg = pal[3 * j + 1] - g;
    const db = pal[3 * j + 2] - b;
    const d = dr * dr + dg * dg + db * db;
    if (d < bestD) {
      bestD = d;
      best = j;
    }
  }
  return best;
};

/** Colour-table size bits for a palette of n colours: at least 2 (4 entries), at most 8 (256). */
export const paletteBits = (n: number): number => {
  let bits = 2;
  while ((1 << bits) < n && bits < 8) bits++;
  return bits;
};

/** LZW-compresses palette indices into GIF code bytes (the image data before sub-blocking). */
export const lzwEncode = (indices: Uint8Array, minCodeSize: number): Uint8Array => {
  const clear = 1 << minCodeSize;
  const eoi = clear + 1;
  const out: number[] = [];
  let acc = 0;
  let nbits = 0;
  let width = minCodeSize + 1;
  const emit = (code: number): void => {
    acc |= code << nbits;
    nbits += width;
    while (nbits >= 8) {
      out.push(acc & 0xff);
      acc >>>= 8;
      nbits -= 8;
    }
  };
  // dictionary: open-addressed hash of (prefix code << 8 | byte) -> code
  const keys = new Int32Array(HASH_SIZE);
  const vals = new Uint16Array(HASH_SIZE);
  keys.fill(-1);
  const slot = (key: number): number => {
    let h = Math.imul(key, 0x9e3779b1) >>> (32 - HASH_BITS);
    while (keys[h] !== -1 && keys[h] !== key) h = (h + 1) & (HASH_SIZE - 1);
    return h;
  };
  let next = eoi + 1;
  const reset = (): void => {
    keys.fill(-1);
    next = eoi + 1;
    width = minCodeSize + 1;
  };

  emit(clear);
  if (indices.length === 0) {
    emit(eoi);
  } else {
    let prefix = indices[0];
    for (let i = 1; i < indices.length; i++) {
      const c = indices[i];
      const key = (prefix << 8) | c;
      const h = slot(key);
      if (keys[h] === key) {
        prefix = vals[h];
        continue;
      }
      emit(prefix);
      if (next < MAX_CODES) {
        keys[h] = key;
        vals[h] = next;
        next++;
        // the decoder adds each entry one code late, so the width grows when the code after this one needs it
        if (next > (1 << width) && width < 12) width++;
      } else {
        emit(clear);
        reset();
      }
      prefix = c;
    }
    emit(prefix);
    emit(eoi);
  }
  if (nbits > 0) out.push(acc & 0xff);
  return Uint8Array.from(out);
};

/** Image data as GIF sub-blocks: length-prefixed chunks of at most 255 bytes, then the zero terminator. */
const subBlocks = (data: Uint8Array, out: number[]): void => {
  for (let p = 0; p < data.length; p += 255) {
    const n = Math.min(255, data.length - p);
    out.push(n);
    for (let i = 0; i < n; i++) out.push(data[p + i]);
  }
  out.push(0);
};

export interface GifWriterOptions {
  /** Times the animation repeats; 0 repeats forever (the default). */
  loops?: number;
}

/** Streaming GIF89a writer: create, add frames (RGBA, width × height), then finish. */
export class GifWriter {
  readonly width: number;
  readonly height: number;
  private readonly parts: Uint8Array[] = [];
  private frameCount = 0;
  private done = false;

  constructor(width: number, height: number, opts: GifWriterOptions = {}) {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width > 0xffff || height > 0xffff) {
      throw new RangeError(`GIF size ${width} × ${height} is out of range`);
    }
    this.width = width;
    this.height = height;
    const head: number[] = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]; // GIF89a
    pushLE16(head, width);
    pushLE16(head, height);
    head.push(0x70, 0, 0); // no global colour table, colour resolution 8, background 0, aspect 0
    // NETSCAPE2.0 application extension: the loop count
    const loops = Math.max(0, Math.floor(opts.loops ?? 0));
    head.push(0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30, 0x03, 0x01);
    pushLE16(head, loops);
    head.push(0x00);
    this.parts.push(Uint8Array.from(head));
  }

  /** Frames written so far. */
  get frames(): number {
    return this.frameCount;
  }

  /** Adds one frame from RGBA pixels (alpha is ignored; the frame is opaque) shown for delayCs hundredths of a second. */
  addFrame(rgba: ArrayLike<number>, delayCs: number): void {
    if (this.done) throw new Error('GIF already finished');
    const pixels = this.width * this.height;
    if (rgba.length < 4 * pixels) throw new RangeError('GIF frame has fewer pixels than its size');
    const pal = medianCutPalette(rgba, pixels, MAX_COLOURS);
    const bits = paletteBits(pal.length / 3);
    const lut = new Int16Array(1 << 15).fill(-1);
    const indices = new Uint8Array(pixels);
    for (let p = 0; p < pixels; p++) {
      const r = rgba[4 * p];
      const g = rgba[4 * p + 1];
      const b = rgba[4 * p + 2];
      const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
      let j = lut[key];
      if (j < 0) {
        j = nearest(pal, (key >> 10) * 8 + 4, ((key >> 5) & 31) * 8 + 4, (key & 31) * 8 + 4);
        lut[key] = j;
      }
      indices[p] = j;
    }
    const head: number[] = [0x21, 0xf9, 0x04, 0x04]; // graphic control: do not dispose, no transparency
    pushLE16(head, Math.max(1, Math.round(delayCs)));
    head.push(0x00, 0x00); // transparent index (unused), block terminator
    head.push(0x2c, 0, 0, 0, 0);
    pushLE16(head, this.width);
    pushLE16(head, this.height);
    head.push(0x80 | (bits - 1)); // local colour table of 2^bits entries
    const table = new Uint8Array(3 * (1 << bits));
    table.set(pal.subarray(0, Math.min(pal.length, table.length)));
    const body: number[] = [];
    body.push(bits); // LZW minimum code size
    subBlocks(lzwEncode(indices, bits), body);
    this.parts.push(Uint8Array.from(head), table, Uint8Array.from(body));
    this.frameCount++;
  }

  /** The complete file (the trailer is written here). Further frames are refused. */
  finish(): Uint8Array<ArrayBuffer> {
    if (!this.done) {
      this.parts.push(Uint8Array.from([0x3b]));
      this.done = true;
    }
    let total = 0;
    for (const p of this.parts) total += p.length;
    const out = new Uint8Array(total);
    let at = 0;
    for (const p of this.parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }
}
