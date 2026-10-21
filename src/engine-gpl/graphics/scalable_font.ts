/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/scalable_font.cpp (the SSFN
 * renderer's _ssfn_c, _ssfn_l, _ssfn_b, _ssfn_r, _ssfn_g and _ssfn_render in bitmap mode, ScalableFont::select_font,
 * create_label and create_colorscale) and src/GRAPHICS/scalable_font.h. The SSFN renderer is by Mateusz Viste / bztsrc
 * (MIT, see the header of scalable_font.cpp). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the
 * GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Port scope: the bundled sans font (VeraR.sfn: SSFN family SANS, style regular, quality 4, no feature flags), so the
 * bold, italic, underline, strike-through, absolute-size and outline paths are not reached and are not ported. The
 * contour fragments are walked by _ssfn_g with the auto-hinting pass and scan-converted by _ssfn_r (even-odd fill,
 * anti-aliasing off), and the bitmap fragments of the format are drawn as _ssfn_g does in bitmap mode. A pixmap glyph
 * fragment (colour map) is refused by name, since no bundled glyph has one. Integer arithmetic follows the C code:
 * the grid values are the uint16 values of the C arrays (stored with & 0xffff), the uint8 fields of the glyph are
 * masked with & 0xff, and C's truncating division is Math.trunc.
 */
import { SANS_FONT_BASE64 } from './scalable_sans_font';
import type { Rgb } from './objects';

/** Thrown for a font problem; the fix turns it into an error naming the label. */
export class SsfnError extends Error {}

/** SSFN constants used by the bitmap path (scalable_font.cpp). */
const FEAT_HBIGCRD = 64;
const FRAG_BITMAP = 0;
const FRAG_LBITMAP = 1;
const FRAG_PIXMAP = 2;
const FRAG_HINTING = 3;
const HINTING_THRESHOLD = 16;
const MIN_SIZE = 8;
const MAX_SIZE = 255;
/** 4095 << 4: grid values from here on are not looked up in the hinting table (_ssfn_g2ix and friends). */
const GRID_TOP = 4095 << 4;
/** (4096 << 4) - 16: the clamp of _ssfn_l. */
const GRID_CLAMP = (4096 << 4) - 16;
/** The uint16 value of the pen-up point (the 0xffff of the C code). */
const PEN_UP = 0xffff;

/** Decode the bundled sans font once. */
function decodeBase64(text: string): Uint8Array {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(text, 'base64'));
  const bin = atob(text);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

let cachedFont: Uint8Array | null = null;
/** The bundled sans font bytes (the VeraR.sfn file). */
export function sansFontBytes(): Uint8Array {
  if (!cachedFont) cachedFont = decodeBase64(SANS_FONT_BASE64);
  return cachedFont;
}

/** One rasterised glyph (the ssfn_glyph_t of a bitmap-mode render): pixels are bits, rows of `pitch` bytes. */
export interface Glyph {
  /** Advance in pixels (the uint8 adv_x). */
  advX: number;
  /** Baseline row, counted from the top of the glyph (uint8). */
  baseline: number;
  /** Width in pixels (uint8 w: the last inked column + 1). */
  w: number;
  /** Height in pixels (uint8 h). */
  h: number;
  /** Bytes per row. */
  pitch: number;
  /** h * pitch bytes, bit x of row y at data[y * pitch + (x >> 3)] bit (x & 7). */
  data: Uint8Array;
}

/** Parsed SSFN font header (the ssfn_font_t fields the bitmap path reads). */
interface FontHeader {
  quality: number;
  features: number;
  baseline: number;
  bboxTop: number;
  charactersOffs: number;
}

function readHeader(F: Uint8Array): FontHeader {
  const u16 = (o: number) => F[o] | (F[o + 1] << 8);
  const u32 = (o: number) => (F[o] | (F[o + 1] << 8) | (F[o + 2] << 16) | (F[o + 3] << 24)) >>> 0;
  const magic = String.fromCharCode(F[0], F[1], F[2], F[3]);
  const endMagic = String.fromCharCode(F[F.length - 4], F[F.length - 3], F[F.length - 2], F[F.length - 1]);
  const size = u32(4);
  if (magic !== 'SSFN' || endMagic !== 'NFSS' || size !== F.length) {
    throw new SsfnError('bundled sans font is not a valid SSFN file');
  }
  return {
    quality: F[10],
    features: F[11],
    baseline: u16(16),
    bboxTop: u16(22),
    charactersOffs: u32(32),
  };
}

/** The renderer context for one font: the ssfn_t fields the bitmap path uses. */
class BitmapRenderer {
  private readonly F: Uint8Array;
  private readonly font: FontHeader;
  /** Shift from the 4095.15 grid to the 16-bit grid: 16 - (4 + quality). */
  private readonly s: number;
  /** Grid shift of the C code (g = 4 + quality). */
  private readonly g: number;
  /** Character table entry size (the l of _ssfn_c and the ol of _ssfn_g). */
  private readonly ol: number;
  /** Auto-hinting table: 0..4095 grid to pixel map, 4096.. the stem histogram (uint16 as in C). */
  private readonly hint = new Uint16Array(8192);
  /** Pixels per em of the glyph being rendered (ctx->m). */
  private m = 0;
  /** Outline points as uint16 values, x and y alternating (ctx->p, ctx->np). */
  private p: number[] = [];
  private lx = -1;
  private ly = -1;
  private mx = -1;
  private my = -1;
  /** The glyph being rendered: its pixel bits, pitch and inked width (ctx->ret). */
  private pix: Uint8Array = new Uint8Array(0);
  private pitch = 0;
  private retW = 0;
  private retH = 0;

  constructor(F: Uint8Array, font: FontHeader) {
    this.F = F;
    this.font = font;
    this.g = 4 + font.quality;
    this.s = 16 - this.g;
    this.ol = font.quality < 5 && font.charactersOffs < 65536 ? 4 : font.charactersOffs < 1048576 ? 5 : 6;
  }

  /** _ssfn_c: the glyph header of a code point, or -1 when the font has none. */
  findGlyph(code: number): number {
    const F = this.F;
    const off = this.font.charactersOffs;
    if (!off) return -1;
    let ptr = off;
    const l = this.ol;
    for (let i = 0; i < 0x110000; i++) {
      if (F[ptr] & 0x80) {
        if (F[ptr] & 0x40) {
          i += F[ptr + 1] | ((F[ptr] & 0x3f) << 8);
          ptr += 2;
        } else {
          i += F[ptr] & 0x3f;
          ptr++;
        }
      } else {
        if (i === code) return ptr;
        ptr += F[ptr] * l + 10;
      }
    }
    return -1;
  }

  // the grid macros of scalable_font.cpp, with ctx->m as the scale
  private g2o(x: number): number {
    return (x * this.m + (1 << 7)) >> 8;
  }
  private g2i(x: number): number {
    return (x * this.m + (1 << 15)) >> 16;
  }
  private g2ic(x: number): number {
    return (x * this.m + (1 << 16) - 1) >> 16;
  }
  private o2i(x: number): number {
    return (x + (1 << 7)) >> 8;
  }
  private o2ic(x: number): number {
    return (x + (1 << 8) - 1) >> 8;
  }
  /** _ssfn_g2ox */
  private g2ox(x: number): number {
    if (x >= GRID_TOP) return this.g2o(x);
    return this.hint[x >> 4];
  }
  /** _ssfn_g2ixc */
  private g2ixc(x: number): number {
    return x >= GRID_TOP ? this.g2ic(x) : this.o2ic(this.hint[x >> 4]);
  }

  /** _ssfn_l: add a point to the contour (grid coordinates; -1 starts a new contour). */
  private line(x0: number, y0: number, l: number): void {
    let x = x0;
    let y = y0;
    if (x > GRID_CLAMP) x = GRID_CLAMP;
    if (y > GRID_CLAMP) y = GRID_CLAMP;
    if (x < -1 || y < -1 || (x === this.lx && y === this.ly)) return;
    const np = this.p.length;
    if (!np || !l || this.g2i(this.p[np - 2]) !== this.g2i(x) || this.g2i(this.p[np - 1]) !== this.g2i(y)) {
      this.p.push(x & 0xffff, y & 0xffff);
      this.lx = x;
      this.ly = y;
    }
  }

  /** _ssfn_b: cubic Bezier flattened by recursive halving (depth limit 8). */
  private bezier(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
                 x3: number, y3: number, l: number): void {
    if (l < 8 && (x0 !== x3 || y0 !== y3)) {
      const m0x = Math.trunc((x1 - x0) / 2) + x0;
      const m0y = Math.trunc((y1 - y0) / 2) + y0;
      const m1x = Math.trunc((x2 - x1) / 2) + x1;
      const m1y = Math.trunc((y2 - y1) / 2) + y1;
      const m2x = Math.trunc((x3 - x2) / 2) + x2;
      const m2y = Math.trunc((y3 - y2) / 2) + y2;
      const m3x = Math.trunc((m1x - m0x) / 2) + m0x;
      const m3y = Math.trunc((m1y - m0y) / 2) + m0y;
      const m4x = Math.trunc((m2x - m1x) / 2) + m1x;
      const m4y = Math.trunc((m2y - m1y) / 2) + m1y;
      const m5x = Math.trunc((m4x - m3x) / 2) + m3x;
      const m5y = Math.trunc((m4y - m3y) / 2) + m3y;
      this.bezier(x0, y0, m0x, m0y, m3x, m3y, m5x, m5y, l + 1);
      this.bezier(m5x, m5y, m4x, m4y, m2x, m2y, x3, y3, l + 1);
    }
    this.line(x3, y3, l);
  }

  /**
   * _ssfn_g: walk the glyph fragments at gp. render=false is the auto-hinting pass (it fills the stem histogram and the
   * hinting table); render=true collects the outline points and draws the bitmap fragments.
   */
  private glyph(gp: number, render: boolean): void {
    const F = this.F;
    const s = this.s;
    const ol = this.ol;
    const quality = this.font.quality;
    const features = this.font.features;
    const hint = this.hint;
    this.lx = this.ly = this.mx = this.my = -1;
    let nf = F[gp];
    let rg = gp + 10;
    let hinted = 0;
    for (; nf--; rg += ol) {
      let o: number;
      let ox: number;
      let oy: number;
      if (ol === 4) {
        o = (F[rg + 1] << 8) | F[rg];
        ox = F[rg + 2];
        oy = F[rg + 3];
      } else if (ol === 5) {
        o = ((F[rg + 2] & 0xf) << 16) | (F[rg + 1] << 8) | F[rg];
        ox = (((F[rg + 2] >> 4) & 3) << 8) | F[rg + 3];
        oy = (((F[rg + 2] >> 6) & 3) << 8) | F[rg + 4];
      } else {
        o = ((F[rg + 3] & 0xf) << 24) | (F[rg + 2] << 16) | (F[rg + 1] << 8) | F[rg];
        ox = ((F[rg + 3] & 0xf) << 8) | F[rg + 4];
        oy = (((F[rg + 3] >> 4) & 0xf) << 8) | F[rg + 5];
      }
      ox <<= s;
      oy <<= s;
      let raw = o;
      if (F[raw] & 0x80) {
        const t = (F[raw] & 0x60) >> 5;
        if (!render && t !== FRAG_HINTING) break;
        if (t === FRAG_LBITMAP || t === FRAG_BITMAP) {
          let x: number;
          let y: number;
          if (t === FRAG_LBITMAP) {
            x = ((((F[raw] >> 2) & 3) << 8) + F[raw + 1]) + 1;
            y = (((F[raw] & 3) << 8) | F[raw + 2]) + 1;
            raw += 3;
          } else {
            x = (F[raw] & 0x1f) + 1;
            y = F[raw + 1] + 1;
            raw += 2;
          }
          // bitmap mode of the bitmap branch (scalable_font.cpp, _ssfn_g)
          const a = x << 3;
          const b = y << s;
          let c = this.g2i(oy);
          const n = this.g2i(ox);
          const w = this.g2i(x << (3 + s));
          const bh = this.g2i(b);
          if (c + bh >= this.retH) c = this.retH - bh;
          c *= this.pitch;
          for (let j = 0; j < bh; j++) {
            const row = Math.trunc(Math.trunc(((j << 8) * (y << 8)) / (bh << 8)) >> 8) * x;
            for (let i = 0; i < w; i++) {
              const mm = Math.trunc(Math.trunc(((i << 8) * (a << 8)) / (w << 8)) >> 8);
              if (F[raw + row + (mm >> 3)] & (1 << (mm & 7))) {
                let d = n + i;
                this.pix[c + (d >> 3)] |= 1 << (d & 7);
                d++;
                if (d > this.retW) this.retW = d;
              }
            }
            c += this.pitch;
          }
          hinted = 0;
          continue;
        }
        if (t === FRAG_PIXMAP) {
          throw new SsfnError('pixmap glyph fragments (colour maps) are not ported');
        }
        // SSFN_FRAG_HINTING
        let n: number;
        if (F[raw] & 0x10) {
          n = ((F[raw] & 0xf) << 8) | F[raw + 1];
          raw += 2;
        } else {
          n = F[raw] & 0xf;
          raw++;
        }
        if (render || !ox) {
          raw += n << (features & FEAT_HBIGCRD ? 1 : 0);
          continue;
        }
        let y = 4096;
        let x = ((ox >> s) - 1) << (s - 4);
        hint[y++] = x;
        for (n++; n-- !== 0 && x < 4096;) {
          x = F[raw];
          raw++;
          if (features & FEAT_HBIGCRD) {
            x |= F[raw] << 8;
            raw++;
          }
          x <<= s - 4;
          hint[y++] = x;
        }
        if (y < 4096) hint[y++] = 65535;
        hinted = 1;
        continue;
      }
      // contour fragment
      if (!render && hinted) break;
      let n: number;
      if (F[raw] & 0x40) {
        n = ((F[raw] & 0x3f) << 8) | F[raw + 1];
        raw += 2;
      } else {
        n = F[raw] & 0x3f;
        raw++;
      }
      let x: number;
      let y: number;
      if (quality < 5) {
        x = F[raw];
        y = F[raw + 1];
        raw += 2;
      } else {
        x = ((F[raw] & 3) << 8) | F[raw + 1];
        y = ((F[raw] & 0x30) << 4) | F[raw + 2];
        raw += 3;
      }
      x <<= s;
      y <<= s;
      y += oy;
      x += ox;
      if (render) {
        if (this.p.length) {
          this.line(this.mx, this.my, 0);
          this.line(-1, -1, 0);
        }
        this.line(x, y, 0);
      }
      this.lx = this.mx = x;
      this.ly = this.my = y;
      for (n++; n--;) {
        const t = this.g < 8 ? (F[raw] >> 7) | ((F[raw + 1] >> 6) & 2) : F[raw] & 3;
        let ex = 0;
        let ey = 0;
        let a = 0;
        let b = 0;
        let c = 0;
        let d = 0;
        if (this.g <= 7) {
          ex = F[raw] & 0x7f;
          ey = F[raw + 1] & 0x7f;
          if (t === 0) raw += F[raw] & 4 ? 5 : 2;
          else if (t === 1) raw += 2;
          else if (t === 2) {
            a = F[raw + 2] & 0x7f;
            b = F[raw + 3] & 0x7f;
            raw += 4;
          } else {
            a = F[raw + 2] & 0x7f;
            b = F[raw + 3] & 0x7f;
            c = F[raw + 4] & 0x7f;
            d = F[raw + 5] & 0x7f;
            raw += 6;
          }
        } else if (this.g === 8) {
          ex = F[raw + 1];
          ey = F[raw + 2];
          if (t === 0) raw += F[raw] & 4 ? 5 : 2;
          else if (t === 1) raw += 3;
          else if (t === 2) {
            a = F[raw + 3];
            b = F[raw + 4];
            raw += 5;
          } else {
            a = F[raw + 3];
            b = F[raw + 4];
            c = F[raw + 5];
            d = F[raw + 6];
            raw += 7;
          }
        } else if (this.g === 9) {
          ex = ((F[raw] & 4) << 6) | F[raw + 1];
          ey = ((F[raw] & 8) << 5) | F[raw + 2];
          if (t === 0) raw += F[raw] & 4 ? 5 : 2;
          else if (t === 1) raw += 3;
          else if (t === 2) {
            a = ((F[raw] & 16) << 4) | F[raw + 3];
            b = ((F[raw] & 32) << 3) | F[raw + 4];
            raw += 5;
          } else {
            a = ((F[raw] & 16) << 4) | F[raw + 3];
            b = ((F[raw] & 32) << 3) | F[raw + 4];
            c = ((F[raw] & 64) << 2) | F[raw + 5];
            d = ((F[raw] & 128) << 1) | F[raw + 6];
            raw += 7;
          }
        } else {
          ex = ((F[raw] & 12) << 6) | F[raw + 1];
          ey = ((F[raw] & 48) << 4) | F[raw + 2];
          if (t === 0) raw += F[raw] & 4 ? 5 : 2;
          else if (t === 1) raw += 3;
          else if (t === 2) {
            a = ((F[raw + 3] & 3) << 8) | F[raw + 4];
            b = ((F[raw + 3] & 12) << 6) | F[raw + 5];
            raw += 6;
          } else {
            a = ((F[raw + 3] & 3) << 8) | F[raw + 4];
            b = ((F[raw + 3] & 12) << 6) | F[raw + 5];
            c = ((F[raw + 3] & 48) << 4) | F[raw + 6];
            d = ((F[raw + 3] & 192) << 2) | F[raw + 7];
            raw += 8;
          }
        }
        const px = (ex << s) + ox;
        const py = (ey << s) + oy;
        const pa = (a << s) + ox;
        const pb = (b << s) + oy;
        const pc = (c << s) + ox;
        const pd = (d << s) + oy;
        if (render) {
          if (t === 1) this.line(px, py, 0);
          else if (t === 2) {
            this.bezier(this.lx, this.ly, ((pa - this.lx) >> 1) + this.lx, ((pb - this.ly) >> 1) + this.ly,
              ((px - pa) >> 1) + pa, ((py - pb) >> 1) + pb, px, py, 0);
          } else if (t === 3) this.bezier(this.lx, this.ly, pa, pb, pc, pd, px, py, 0);
        } else if (t === 1 && px >= 0 && py >= 0) {
          const da = ((this.lx < px) ? px - this.lx : this.lx - px) >> 4;
          const db = ((this.ly < py) ? py - this.ly : this.ly - py) >> 4;
          const dc = (this.lx + px) >> 5;
          if (da < 2) {
            const idx = !hint[4096 + dc] && dc && hint[4096 + dc - 1] ? dc - 1 : dc;
            hint[4096 + idx] += db;
          }
        }
        this.lx = px;
        this.ly = py;
      }
    }
    if (!render && !hinted) {
      // the stem widths of the histogram (the tail of _ssfn_g)
      let m = 0;
      let x = 0;
      for (let j = 0; j < 4096; j++) {
        if (hint[4096 + j] >= 4096 / HINTING_THRESHOLD) {
          if (!j) m++;
          else {
            hint[4096 + m++] = j - x;
            x = j;
          }
        }
      }
      if (m < 4096) hint[4096 + m] = 65535;
    }
  }

  /**
   * _ssfn_r: scan-convert the outline points into the bitmap, one row at a time (even-odd fill, the crossings of each
   * row sorted and paired). Returns the baseline after the rounding fix of the C code.
   */
  private raster(baseline: number): number {
    const P = this.p;
    const np = P.length;
    let M = 0;
    for (let y = 0; y < this.retH; y++) {
      const Y = y ? Math.trunc(((y << 16) - (1 << 15)) / this.m) : 0;
      const r: number[] = [];
      for (let i = 0; i < np - 3; i += 2) {
        if ((P[i] === PEN_UP && P[i + 1] === PEN_UP) || (P[i + 2] === PEN_UP && P[i + 3] === PEN_UP)) continue;
        if ((P[i + 1] < Y && P[i + 3] >= Y) || (P[i + 3] < Y && P[i + 1] >= Y)) {
          let x: number;
          if (this.g2i(P[i + 1]) === this.g2i(P[i + 3])) x = (P[i] + P[i + 2]) >> 1;
          else {
            // the product is an int in C: wrap it the way a 32-bit multiply does
            x = P[i] + Math.trunc(Math.imul(Y - P[i + 1], P[i + 2] - P[i]) / (P[i + 3] - P[i + 1]));
          }
          if (x < 0) throw new SsfnError('negative grid coordinate in glyph outline');
          x = this.g2ox(x);
          let k = 0;
          while (k < r.length && x > r[k]) k++;
          r.splice(k, 0, x & 0xffff);
        }
      }
      // an odd crossing count drops the last one (its pair is closed by the previous crossing)
      let n = r.length;
      if (n > 1 && n & 1) {
        r[n - 2] = r[n - 1];
        r.pop();
        n--;
      }
      if (n) {
        if (y > M) M = y;
        const k = y * this.pitch;
        for (let i = 0; i < n - 1; i += 2) {
          let l = (r[i] + 128) >> 8;
          const mm = (r[i + 1] + 128) >> 8;
          for (; l < mm; l++) this.pix[k + (l >> 3)] ^= 1 << (l & 7);
          if (l + 1 > this.retW) this.retW = l + 1;
        }
      }
    }
    // fix rounding errors: a baseline on the last inked row moves up one row
    return M + 1 === baseline ? baseline - 1 : baseline;
  }

  /** _ssfn_render in bitmap mode at `size` pixels for the glyph at gp. */
  render(size: number, gp: number): Glyph {
    const F = this.F;
    const s = this.s;
    const wRaw = ((F[gp + 2] & 0x0f) << 8) | F[gp + 6];
    // the glyph is scaled so that the distance from the font box top to the baseline is `size` pixels
    let h = !wRaw ? size : Math.trunc((4096 * 16 * size) / ((this.font.baseline - this.font.bboxTop) << s));
    let p = h;
    if (p > 255) {
      p = h = 255;
    }
    p = (p + 7) >> 3;
    this.m = h;
    const H = this.hint;
    H.fill(0, 4096);
    this.p = [];
    this.glyph(gp, false);

    // grid to pixel table (the loop after the hinting pass in _ssfn_render)
    H[0] = 0;
    let i = 0;
    let j = 0;
    let x = 0;
    let m = 0;
    while (i < 4096 && j < 4095) {
      let y = H[4096 + i] === 65535 ? 4095 - j : H[4096 + i];
      j += y;
      if (j === x) {
        H[j] = ((((j << 4) * h + (1 << 15)) >> 16) << 8);
        if (!y) j++;
      } else {
        y = this.g2o(y << 4);
        m += y;
        if ((i & 1) || y < 256) {
          i++;
          continue;
        }
        m &= ~0xff;
        const n = H[x];
        for (let l = 0; l + x <= j; l++) H[x + l] = n + Math.trunc(((m - n) * l) / (j - x));
      }
      x = j;
      i++;
    }
    // advance widths: the rounded-up grid extent (uix) and its vertical twin (uax, which sets the baseline below)
    const advX = this.g2ixc(wRaw << s) & 0xff;
    const advYRaw = this.g2ic(((((F[gp + 2] & 0xf0) << 4) | F[gp + 7])) << s);
    const bt = (((F[gp + 3] & 0xf0) << 4) | F[gp + 9]) << s;
    const baseline0 = ((((this.font.baseline << s) - bt) * h + (1 << 16) - 1) >> 16) & 0xff;

    this.pix = new Uint8Array(p * h);
    this.pitch = p;
    this.retW = 0;
    this.retH = h;
    this.p = [];
    this.glyph(gp, true);
    let baseline = this.raster(baseline0);
    if (advYRaw & 0xff) baseline = this.retW >> 1;
    return { advX, baseline: baseline & 0xff, w: this.retW & 0xff, h: h & 0xff, pitch: p, data: this.pix };
  }
}

/** The renderer for the bundled sans font: select a size, then ask for glyphs or labels. */
export class ScalableFont {
  private readonly renderer: BitmapRenderer;
  private readonly F: Uint8Array;
  private selected = 0;

  constructor(bytes: Uint8Array = sansFontBytes()) {
    this.F = bytes;
    this.renderer = new BitmapRenderer(bytes, readHeader(bytes));
  }

  /** select_font for SSFN::FAMILY_SANS, SSFN::STYLE_REGULAR (_ssfn_select: size 8 to 255). */
  selectFont(size: number): void {
    if (!Number.isInteger(size) || size < MIN_SIZE || size > MAX_SIZE) {
      throw new SsfnError(`invalid font size ${size}`);
    }
    this.selected = size;
  }

  /** The selected size in pixels (0 before selectFont). */
  get size(): number {
    return this.selected;
  }

  /** The bitmap of one character at the selected size (_ssfn_render); throws when the font has no glyph for it. */
  glyph(ch: string): Glyph {
    if (!this.selected) throw new SsfnError('no font size selected');
    const code = ch.codePointAt(0) ?? 0;
    // the native label loops read the characters as C chars: a code above 127 is never found in the table
    const gp = code > 127 ? -1 : this.renderer.findGlyph(code);
    if (gp < 0) throw new SsfnError(`Glyph not found: ${JSON.stringify(ch)}`);
    return this.renderer.render(this.selected, gp);
  }
}

/** The text as the glyph loops see it: '_' is drawn as a space (the "ugly hack" of create_label). */
const glyphChars = (text: string): string[] => [...text].map((c) => (c === '_' ? ' ' : c));

/** The colours of a label: font, frame and background, 0..255 per channel. */
export interface LabelColors {
  font: Rgb;
  frame: Rgb;
  back: Rgb;
}

/** A label pixmap: width x height pixels, RGB bytes, row 0 at the bottom (the native pixmap layout). */
export interface LabelPixmap {
  width: number;
  height: number;
  rgb: Uint8Array;
}

/**
 * ScalableFont::create_label(): the text in the font colour on the background, inside a frame of xhalf pixels. The
 * pen starts at 2 * xspace (xspace = advance of a space), the label is 3 * xspace taller than the glyph extent
 * (the extent of the text plus "gll", so that the descender and the ascender are always in), and a vertical label
 * is the horizontal one rotated: the first column becomes the bottom row.
 */
export function createLabel(font: ScalableFont, text: string, horizontal: boolean, colors: LabelColors): LabelPixmap {
  const chars = glyphChars(text);
  const xspace = font.glyph(' ').advX;
  const xhalf = Math.trunc(xspace / 2);

  // dry run to determine the size of the pixmap
  const { width, miny, maxy } = dryRun(font, chars);
  const height = maxy - miny + 1 + 3 * xspace;

  // background and frame
  const pix = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const c = y < xhalf || y >= height - xhalf || x < xhalf || x >= width - xhalf ? colors.frame : colors.back;
      const o = 3 * (y * width + x);
      pix[o] = c[0];
      pix[o + 1] = c[1];
      pix[o + 2] = c[2];
    }
  }

  // the glyphs, in the font colour
  let penx = 2 * xspace;
  for (const c of chars) {
    const g = font.glyph(c);
    for (let y = 0; y < g.h; y++) {
      const yoffs = (g.h - 1 - y + g.baseline - miny + xspace + Math.trunc(xhalf / 2)) * width * 3;
      for (let x = 0; x < g.w; x++) {
        const xoffs = (penx + x) * 3;
        if (g.data[y * g.pitch + (x >> 3)] & (1 << (x & 7))) {
          const o = yoffs + xoffs;
          if (o >= 0 && o + 2 < pix.length) {
            pix[o] = colors.font[0];
            pix[o + 1] = colors.font[1];
            pix[o + 2] = colors.font[2];
          }
        }
      }
    }
    penx += g.advX;
  }

  if (horizontal) return { width, height, rgb: pix };
  return rotatePixmap(pix, width, height);
}

/** The vertical copy of create_label: flipped[x][height-1-y] = pixmap[y][x], with the width and height swapped. */
function rotatePixmap(pix: Uint8Array, width: number, height: number): LabelPixmap {
  const flipped = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let i = 0; i < 3; i++) {
        flipped[x * 3 * height + 3 * (height - 1 - y) + i] = pix[y * 3 * width + 3 * x + i];
      }
    }
  }
  return { width: height, height: width, rgb: flipped };
}

/** The colour map a colour scale reads: ColorMap::info and ColorMap::value2color of the dump image (0..1 channels). */
export interface ColorScaleMap {
  info(): { lo: number; hi: number; sequential: boolean };
  color(value: number): [number, number, number];
}

/**
 * The number text of fmt::format("{:3.3}", v): fmt's general format with precision 3 (like %.3g: the mantissa has at
 * most three significant digits and no trailing zeros; scientific when the exponent is below -4 or not below 3), padded
 * to a width of 3 on the left.
 */
export function formatGeneral3(v: number): string {
  let text: string;
  if (v === 0) text = '0';
  else {
    const sci = v.toExponential(2);
    const x = Number(sci.slice(sci.indexOf('e') + 1));
    if (x < -4 || x >= 3) {
      const [mant, ex] = sci.split('e');
      const m = mant.includes('.') ? mant.replace(/0+$/, '').replace(/\.$/, '') : mant;
      const sign = ex[0] === '-' ? '-' : '+';
      const digits = ex.replace(/^[+-]/, '');
      text = `${m}e${sign}${digits.padStart(2, '0')}`;
    } else {
      const f = v.toFixed(Math.max(0, 2 - x));
      text = f.includes('.') ? f.replace(/0+$/, '').replace(/\.$/, '') : f;
    }
  }
  return text.padStart(3, ' ');
}

/** The dry run of create_label/create_colorscale: width, lowest and highest ink row over the text plus "gll". */
function dryRun(font: ScalableFont, chars: string[]): { width: number; miny: number; maxy: number } {
  let width = 0;
  let miny = 1073741824;
  let maxy = 0;
  for (const c of [...chars, 'g', 'l', 'l']) {
    const g = font.glyph(c);
    width += g.advX;
    for (let y = 0; y < g.h; y++) {
      const ypos = g.h - 1 - y + g.baseline;
      for (let x = 0; x < g.w; x++) {
        if (g.data[y * g.pitch + (x >> 3)] & (1 << (x & 7))) {
          if (ypos < miny) miny = ypos;
          if (ypos > maxy) maxy = ypos;
        }
      }
    }
  }
  return { width, miny, maxy };
}

/**
 * ScalableFont::create_colorscale(): the colour bar of the map with its range as tick text on both ends (fmt "{:3.3}"),
 * the text in the middle, padded with "<" and ">" to `minwidth` pixels, and optional tic marks. The bar's height comes
 * from the first layout pass (the unpadded text), its label offsets from the second, as the native code does.
 */
export function createColorscale(font: ScalableFont, text: string, horizontal: boolean, minwidth: number,
                                 map: ColorScaleMap, tics: number, colors: LabelColors): LabelPixmap {
  const xspace = font.glyph(' ').advX;
  const xfill = font.glyph('<').advX;
  const info = map.info();
  let lo = info.lo;
  let hi = info.hi;
  const addone = info.sequential;
  let newtext = `${formatGeneral3(lo)}  ${text}  ${formatGeneral3(hi)}`;

  // dry run to determine the height (and the width of the unpadded text)
  const first = dryRun(font, glyphChars(newtext));
  const xhalf = Math.trunc(xspace / 2);
  const height = first.maxy - first.miny + 1 + 6 * xspace;

  if (minwidth > first.width) {
    const wextra = Math.trunc(Math.trunc((minwidth - first.width) / xfill) / 4);
    // fmt "{:<<{}}" of the character '<' with width wextra: the character padded to wextra, so one at least (also for 0)
    const pad = Math.max(1, wextra);
    newtext = `${formatGeneral3(lo)} ${'<'.repeat(pad)} ${text} ${'>'.repeat(pad)} ${formatGeneral3(hi)}`;
  }
  // re-do the dry run for the padded text
  const second = dryRun(font, glyphChars(newtext));
  const width = second.width;
  const miny = second.miny;

  // background and frame
  const pix = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const c = y < xhalf || y >= height - xhalf || x < xhalf || x >= width - xhalf ? colors.frame : colors.back;
      const o = 3 * (y * width + x);
      pix[o] = c[0];
      pix[o + 1] = c[1];
      pix[o + 2] = c[2];
    }
  }

  // the colour bar
  if (addone) hi += 1.0;
  const delta = (hi - lo) / (width - 2 * xspace);
  const ticinc = tics ? Math.trunc((width - 2 * xspace) / (tics + 1)) : 0;
  let ticmin = ticinc + xspace + 1;
  let ticmax = ticmin + xhalf;
  const top = height - 4 * xspace - xhalf;
  for (let x = xspace; x < width - xspace; x++) {
    for (let y = xhalf + Math.trunc(xhalf / 2); y < top; y++) {
      const offs = 3 * y * width + 3 * x;
      const val = lo + delta * (x - xspace);
      const color = map.color(val);
      const onFont = tics > 0
        ? ((x >= ticmin && x <= ticmax) || x <= xspace + xhalf || x >= width - xspace - xhalf) ||
          y <= xspace + Math.trunc(xhalf / 2) || y >= height - 5 * xspace
        : x <= xspace + xhalf || x >= width - xspace - xhalf || y <= xspace + Math.trunc(xhalf / 2) ||
          y >= height - 5 * xspace;
      if (onFont) {
        pix[offs] = colors.font[0];
        pix[offs + 1] = colors.font[1];
        pix[offs + 2] = colors.font[2];
      } else {
        pix[offs] = Math.trunc(color[0] * 255.0);
        pix[offs + 1] = Math.trunc(color[1] * 255.0);
        pix[offs + 2] = Math.trunc(color[2] * 255.0);
      }
      if (tics > 0 && x === ticmax) {
        ticmin += ticinc;
        ticmax += ticinc;
      }
    }
  }

  // the text, drawn over the bar
  let penx = 2 * xspace;
  const peny = 3 * xspace + xhalf;
  for (const c of glyphChars(newtext)) {
    const g = font.glyph(c);
    for (let y = 0; y < g.h; y++) {
      const yoffs = (g.h - 1 - y + peny + g.baseline - miny + xspace + Math.trunc(xhalf / 2)) * width * 3;
      for (let x = 0; x < g.w; x++) {
        const xoffs = (penx + x) * 3;
        if (g.data[y * g.pitch + (x >> 3)] & (1 << (x & 7))) {
          const o = yoffs + xoffs;
          if (o >= 0 && o + 2 < pix.length) {
            pix[o] = colors.font[0];
            pix[o + 1] = colors.font[1];
            pix[o + 2] = colors.font[2];
          }
        }
      }
    }
    penx += g.advX;
  }

  if (horizontal) return { width, height, rgb: pix };
  return rotatePixmap(pix, width, height);
}
