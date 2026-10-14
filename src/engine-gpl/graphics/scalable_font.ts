/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/GRAPHICS/scalable_font.cpp (the SSFN
 * renderer: _ssfn_c, _ssfn_l, _ssfn_b, _ssfn_g, _ssfn_render, _ssfn_select and ScalableFont::create_label) and
 * src/GRAPHICS/scalable_font.h. The SSFN renderer is by Mateusz Viste (public domain, see the header of
 * scalable_font.cpp). LAMMPS is Copyright (2003) Sandia Corporation and distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * Port scope: only the vector (outline) path of the renderer, for the sans regular font that fix graphics/labels
 * selects (SSFN::FAMILY_SANS, SSFN::STYLE_REGULAR). The glyph outlines are decoded (with the auto-hinting pass, so the
 * coordinates are the ones the native renderer uses) and flattened into line segments instead of being scan-converted
 * into the bitmap that create_label() fills. The label layout (advance widths, the 2*xspace pen offset, the frame and
 * the 3*xspace margin, the vertical flip) follows create_label(); the label pixel rows are continuous coordinates, so
 * the stroke positions match the native glyph shapes but not the native anti-aliased pixels.
 */
import { SANS_FONT_BASE64 } from './scalable_sans_font';

/** Thrown for a font problem; the fix turns it into an error naming the label. */
export class SsfnError extends Error {}

/** Frame-level constants of the SSFN container (scalable_font.cpp). */
const SSFN_FEAT_HBIGCRD = 64;
const SSFN_FRAG_BITMAP = 0;
const SSFN_FRAG_LBITMAP = 1;
const SSFN_FRAG_PIXMAP = 2;
const SSFN_FRAG_HINTING = 3;
const SSFN_HINTING_THRESHOLD = 16;
const SSFN_MIN_SIZE = 8;
const SSFN_MAX_SIZE = 255;
const GRID_MAX = 4095 << 4;
/** Sentinel grid value for a pen-up point (the 0xffff of the C code). */
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

/** One glyph as native create_label() sees it: advance, baseline row and contours in glyph pixels. */
export interface GlyphOutline {
  /** Advance width in pixels (the uint8 adv_x of the native glyph). */
  advX: number;
  /** Glyph height in pixels at the scale of the native bitmap renderer (the h of _ssfn_render). */
  height: number;
  /** Baseline row of the glyph, counted from its top (uint8 baseline). */
  baseline: number;
  /** Contours; each is a list of [x, y] pixel points, y measured downwards from the glyph top. */
  contours: Array<Array<[number, number]>>;
}

/** Parsed SSFN font header (the ssfn_font_t fields the vector path reads). */
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

/** The vector renderer state (the ssfn_t fields the outline path uses). */
class OutlineRenderer {
  private readonly F: Uint8Array;
  private readonly font: FontHeader;
  /** Auto-hinting grid: entries 0..4095 map grid to pixels, 4096.. hold the stem histogram. */
  private readonly h = new Uint16Array(8192);
  private readonly g: number;
  private readonly s: number;
  private readonly ol: number;
  private m = 0;
  private p: number[] = [];
  private np = 0;
  private lx = -1;
  private ly = -1;
  private mx = -1;
  private my = -1;

  constructor(F: Uint8Array, font: FontHeader) {
    this.F = F;
    this.font = font;
    this.g = 4 + font.quality;
    this.s = 16 - this.g;
    // glyph offset size, as in _ssfn_g: 4, 5 or 6 bytes per fragment entry
    this.ol =
      font.quality < 5 && font.charactersOffs < 65536 ? 4 : font.charactersOffs < 1048576 ? 5 : 6;
  }

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
  /** _ssfn_g2ix: grid x to pixel x through the hinting table. */
  private g2ix(x: number): number {
    return x >= GRID_MAX ? this.g2i(x) : this.o2i(this.h[x >> 4]);
  }
  /** _ssfn_g2ixc: grid x to pixel x, rounded up (advance widths). */
  private g2ixc(x: number): number {
    return x >= GRID_MAX ? this.g2ic(x) : this.o2ic(this.h[x >> 4]);
  }

  /** _ssfn_l: add a point to the contour (grid coordinates; -1 starts a new contour). */
  private lineTo(x: number, y: number, lf: number): void {
    if (x > GRID_MAX - 16) x = GRID_MAX - 16;
    if (y > GRID_MAX - 16) y = GRID_MAX - 16;
    if (x < -1 || y < -1 || (x === this.lx && y === this.ly)) return;
    if (this.np === 0 || !lf || this.g2i(this.p[this.np - 2]) !== this.g2i(x) ||
        this.g2i(this.p[this.np - 1]) !== this.g2i(y)) {
      this.p[this.np++] = x & 0xffff;
      this.p[this.np++] = y & 0xffff;
      this.lx = x;
      this.ly = y;
    }
  }

  /** _ssfn_b: cubic Bezier flattened by recursive halving (the native subdivision depth limit is 8). */
  private bezier(x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
                 x3: number, y3: number, lv: number): void {
    if (lv < 8 && (x0 !== x3 || y0 !== y3)) {
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
      this.bezier(x0, y0, m0x, m0y, m3x, m3y, m5x, m5y, lv + 1);
      this.bezier(m5x, m5y, m4x, m4y, m2x, m2y, x3, y3, lv + 1);
    }
    this.lineTo(x3, y3, lv);
  }

  /** _ssfn_g: walk the glyph fragments; render=false is the hinting pass, render=true the outline pass. */
  private glyph(gp: number, render: boolean): void {
    const F = this.F;
    const s = this.s;
    const ol = this.ol;
    const features = this.font.features;
    const quality = this.font.quality;
    const h = this.h;
    this.lx = this.ly = this.mx = this.my = -1;
    const nfrag = F[gp];
    let rg = gp + 10;
    let hint = 0;
    for (let k = 0; k < nfrag; k++, rg += ol) {
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
        if (!render && t !== SSFN_FRAG_HINTING) break;
        if (t === SSFN_FRAG_LBITMAP || t === SSFN_FRAG_BITMAP || t === SSFN_FRAG_PIXMAP) {
          // the bundled font has only contour glyphs (checked by the tests); bitmap glyphs are not ported
          throw new SsfnError('bitmap glyph fragments are not supported by the vector port');
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
          raw += n << (features & SSFN_FEAT_HBIGCRD ? 1 : 0);
          continue;
        }
        let y = 4096;
        let x = ((ox >> s) - 1) << (s - 4);
        h[y++] = x;
        for (n++; n-- !== 0 && x < 4096;) {
          x = F[raw];
          raw++;
          if (features & SSFN_FEAT_HBIGCRD) {
            x |= F[raw] << 8;
            raw++;
          }
          x <<= s - 4;
          h[y++] = x;
        }
        if (y < 4096) h[y++] = 65535;
        hint = 1;
        continue;
      }
      // contour fragment
      if (!render && hint) break;
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
        if (this.np) {
          this.lineTo(this.mx, this.my, 0);
          this.lineTo(-1, -1, 0);
        }
        this.lineTo(x, y, 0);
      }
      this.lx = this.mx = x;
      this.ly = this.my = y;
      for (n++; n-- !== 0;) {
        const t = this.g < 8 ? (F[raw] >> 7) | ((F[raw + 1] >> 6) & 2) : F[raw] & 3;
        let ex = 0;
        let ey = 0;
        let a = 0;
        let b = 0;
        let c = 0;
        let d = 0;
        // point layout by quality (the g switch of _ssfn_g)
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
        let px = ex << s;
        let py = ey << s;
        a <<= s;
        b <<= s;
        c <<= s;
        d <<= s;
        px += ox;
        py += oy;
        a += ox;
        b += oy;
        c += ox;
        d += oy;
        if (render) {
          if (t === 1) this.lineTo(px, py, 0);
          else if (t === 2) {
            this.bezier(this.lx, this.ly, ((a - this.lx) >> 1) + this.lx, ((b - this.ly) >> 1) + this.ly,
              ((px - a) >> 1) + a, ((py - b) >> 1) + b, px, py, 0);
          } else if (t === 3) this.bezier(this.lx, this.ly, a, b, c, d, px, py, 0);
        } else if (t === 1 && px >= 0 && py >= 0) {
          const da = ((this.lx < px) ? px - this.lx : this.lx - px) >> 4;
          const db = ((this.ly < py) ? py - this.ly : this.ly - py) >> 4;
          const dc = (this.lx + px) >> 5;
          if (da < 2) {
            const idx = !h[4096 + dc] && dc && h[4096 + dc - 1] ? dc - 1 : dc;
            h[4096 + idx] += db;
          }
        }
        this.lx = px;
        this.ly = py;
      }
    }
    if (!render && !hint) {
      // build the list of stem widths from the histogram (the second half of _ssfn_g)
      let m = 0;
      let x = 0;
      for (let j = 0; j < 4096; j++) {
        if (h[4096 + j] >= 4096 / SSFN_HINTING_THRESHOLD) {
          if (!j) m++;
          else {
            h[4096 + m++] = j - x;
            x = j;
          }
        }
      }
      if (m < 4096) h[4096 + m] = 65535;
    }
  }

  /** Render one glyph outline (the vector path of _ssfn_render). */
  render(size: number, gp: number): GlyphOutline {
    const F = this.F;
    const s = this.s;
    // _ssfn_render in bitmap mode (the mode of ScalableFont::create_label): the glyph height is scaled so that the
    // distance from the top of the font box to the baseline is size pixels; zero-width glyphs keep the size
    const width0 = (((F[gp + 2] & 0x0f) << 8) | F[gp + 6]);
    const gh = !width0 ? size : Math.trunc(((4096 << 4) * size) / ((this.font.baseline - this.font.bboxTop) << s));
    this.m = Math.min(gh, 255);
    this.np = 0;
    this.p = [];
    this.h.fill(0, 4096);
    this.glyph(gp, false);
    // hinting table: grid to pixel map (the loop after the hinting pass in _ssfn_render)
    const h = this.h;
    h[0] = 0;
    const hh = this.m;
    let i = 0;
    let j = 0;
    let x = 0;
    let y = 0;
    let acc = 0;
    while (i < 4096 && j < 4095) {
      y = h[4096 + i] === 65535 ? 4095 - j : h[4096 + i];
      j += y;
      if (j === x) {
        h[j] = ((((j << 4) * hh + (1 << 15)) >> 16) << 8) & 0xffff;
        if (!y) j++;
      } else {
        y = this.g2o(y << 4);
        acc += y;
        if ((i & 1) || y < 256) {
          i++;
          continue;
        }
        acc &= ~0xff;
        const n = h[x];
        for (let l = 0; l + x <= j; l++) h[x + l] = (n + Math.trunc(((acc - n) * l) / (j - x))) & 0xffff;
      }
      x = j;
      i++;
    }
    const width = width0 << s;
    const advX = this.g2ixc(width) & 0xff;
    // glyph top offset (bt of _ssfn_render): the baseline is measured from the glyph top
    const bt = (((F[gp + 3] & 0xf0) << 4) | F[gp + 9]) << s;
    const baseline = (((((this.font.baseline << s) - bt) * hh + (1 << 16) - 1) >> 16)) & 0xff;

    this.np = 0;
    this.p = [];
    this.glyph(gp, true);
    const contours: Array<Array<[number, number]>> = [];
    let cur: Array<[number, number]> = [];
    const out: Array<[number, number]> = [];
    for (let k = 0; k < this.np; k += 2) {
      // no bearing offset: the native bitmap path places the points on the grid origin of the glyph as well
      const pc = this.p[k] === PEN_UP ? 0xff : this.g2ix(this.p[k]) & 0xff;
      const pd = this.p[k + 1] === PEN_UP ? 0xff : this.g2i(this.p[k + 1]) & 0xff;
      out.push([pc, pd]);
    }
    // consecutive duplicates are dropped as in the native output loop; 255,255 starts a new contour
    let last: [number, number] | null = null;
    for (const pt of out) {
      if (pt[0] === 0xff && pt[1] === 0xff) {
        if (cur.length) contours.push(cur);
        cur = [];
        last = null;
        continue;
      }
      if (last && last[0] === pt[0] && last[1] === pt[1]) continue;
      cur.push(pt);
      last = pt;
    }
    if (cur.length) contours.push(cur);
    return { advX, height: this.m, baseline, contours };
  }
}

/** The renderer for the bundled sans font: select a size, then ask for labelled text. */
export class ScalableFont {
  private readonly F: Uint8Array;
  private readonly font: FontHeader;
  private selected = 0;

  constructor(bytes: Uint8Array = sansFontBytes()) {
    this.F = bytes;
    this.font = readHeader(bytes);
  }

  /** _ssfn_select for SSFN::FAMILY_SANS, SSFN::STYLE_REGULAR. */
  selectFont(size: number): void {
    if (!Number.isInteger(size) || size < SSFN_MIN_SIZE || size > SSFN_MAX_SIZE) {
      throw new SsfnError(`invalid font size ${size}`);
    }
    this.selected = size;
  }

  /** The selected size in pixels (0 before selectFont). */
  get size(): number {
    return this.selected;
  }

  /** Outline of one character at the selected size; throws if the font has no glyph for it. */
  glyph(ch: string): GlyphOutline {
    if (!this.selected) throw new SsfnError('no font size selected');
    const code = ch.codePointAt(0) ?? 0;
    const gp = this.findGlyph(code);
    if (gp < 0) throw new SsfnError(`no glyph for character ${JSON.stringify(ch)}`);
    return new OutlineRenderer(this.F, this.font).render(this.selected, gp);
  }

  /** _ssfn_c: walk the character table (skip codes are encoded as in the native parser). */
  private findGlyph(code: number): number {
    const F = this.F;
    const off = this.font.charactersOffs;
    if (!off) return -1;
    let ptr = off;
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
        const l = this.font.quality < 5 && off < 65536 ? 4 : off < 1048576 ? 5 : 6;
        ptr += F[ptr] * l + 10;
      }
    }
    return -1;
  }
}

/** A label in its own pixel frame; row 0 is the bottom row, as in the native pixmap. */
export interface TextLayout {
  width: number;
  height: number;
  /** Frame thickness in pixels (xhalf of create_label). */
  frame: number;
  /** Flat [x0, y0, x1, y1, ...] segment list in label pixels, origin at the bottom-left. */
  strokes: number[];
}

/**
 * ScalableFont::create_label() with strokes instead of a bitmap. Text is laid out like the native label: the pen
 * starts at 2*xspace, xspace is the advance of a space, the label is 3*xspace taller than the glyph extent, and a
 * vertical label is the horizontal one rotated (x -> row, row -> column, flipped).
 */
export function createLabel(font: ScalableFont, text: string, horizontal: boolean): TextLayout {
  if (!font.size) throw new SsfnError('no font size selected');
  const space = font.glyph(' ');
  const xspace = space.advX;
  const xhalf = Math.trunc(xspace / 2);
  // dry run: width and vertical extent over the text plus "gll" (consistent spacing)
  let width = 0;
  let miny = Infinity;
  let maxy = -Infinity;
  const measure = (ch: string) => {
    const gl = font.glyph(ch === '_' ? ' ' : ch);
    width += gl.advX;
    for (const contour of gl.contours) {
      for (const [, d] of contour) {
        const yp = gl.baseline + gl.height - d;
        if (yp < miny) miny = yp;
        if (yp > maxy) maxy = yp;
      }
    }
  };
  for (const ch of [...text, 'g', 'l', 'l']) measure(ch);
  const iMin = Math.floor(miny);
  const iMax = Math.floor(maxy);
  const height = iMax - iMin + 1 + 3 * xspace;

  // second pass: place the strokes
  const strokes: number[] = [];
  let penx = 2 * xspace;
  const offsetY = xspace + Math.trunc(xhalf / 2) - iMin;
  for (const ch of [...text]) {
    const gl = font.glyph(ch === '_' ? ' ' : ch);
    const base = gl.baseline + gl.height;
    for (const contour of gl.contours) {
      for (let k = 0; k + 1 < contour.length; k++) {
        const [xa, da] = contour[k];
        const [xb, db] = contour[k + 1];
        strokes.push(penx + xa, base - da + offsetY, penx + xb, base - db + offsetY);
      }
    }
    penx += gl.advX;
  }

  if (horizontal) return { width, height, frame: xhalf, strokes };
  // vertical: rotate (x, row) to (height - row, x) with the swapped extent
  const rotated: number[] = [];
  for (let k = 0; k < strokes.length; k += 2) {
    rotated.push(height - strokes[k + 1], strokes[k]);
  }
  return { width: height, height: width, frame: xhalf, strokes: rotated };
}
