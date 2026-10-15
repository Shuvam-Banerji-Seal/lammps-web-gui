import { StyleError } from '../force/types';

/*
 * dump_modify amap colour maps for dump image (docs.lammps.org/dump_modify.html): "amap lo hi style delta N
 * entry1 entry2 ... entryN"; "lo" and "hi" are a number, or min or max; "style" is two letters, the first c
 * (continuous), d (discrete) or s (sequential), the second a (absolute values) or f (fractional, the range
 * scaled to 0..1); "delta" is the bin size of the sequential style. dump_image.html: "The default is
 * amap = min max cf 0.0 2 min blue max red".
 *
 * The decision rules follow the colour-map class of LAMMPS (src/GRAPHICS/image.cpp at b332f28, ColorMap::reset,
 * minmax and value2color), restated in engine terms: the value is clamped to the current [lo, hi]; the
 * continuous style interpolates linearly between the entries; the discrete style takes the first entry whose
 * [lo, hi] contains the value; the sequential style bins the value as trunc((value - lo) / delta) and cycles
 * through the N colours, a value exactly at the top of the range taking the colour of the bin below it when the
 * range is an exact multiple of delta.
 */

/**
 * A colour of the map as native keeps it: each channel as a double (ColorMap returns the colour of color2rgb, not a byte).
 * The caller takes the 8-bit value with int(c * 255) where a byte is needed (image.cpp draw_pixel and friends).
 */
export type RGB = [number, number, number];

type Bound = { kind: 'num'; v: number } | { kind: 'min' } | { kind: 'max' };

interface Entry {
  /** continuous style: the entry's position; unused otherwise. */
  pos: Bound;
  /** discrete style: the bin [lo, hi]; unused otherwise. */
  lo: Bound;
  hi: Bound;
  color: RGB;
}

const parseBound = (w: string | undefined, what: string): Bound => {
  if (w === 'min') return { kind: 'min' };
  if (w === 'max') return { kind: 'max' };
  const v = Number(w);
  if (w === undefined || w === '' || !Number.isFinite(v)) throw new StyleError(`dump_modify amap: ${what} '${w ?? ''}' must be a number, min or max`);
  return { kind: 'num', v };
};

const MIN: Bound = { kind: 'min' };
const MAX: Bound = { kind: 'max' };

export class ColorMap {
  private lo: Bound = MIN;
  private hi: Bound = MAX;
  private style: 'c' | 'd' | 's' = 'c';
  private absolute = false;
  private binsize = 1;
  private entries: Entry[] = [];
  /** Entry positions/bins as numbers in the value space of the current range (set by bounds()). */
  private pos: number[] = [];
  private bins: [number, number][] = [];
  /**
   * As native LAMMPS does before the first minmax (src/GRAPHICS/image.cpp at b332f28, ColorMap constructor:
   * "locurrent = mlo = MINVALUE" with the enum value 1, and "hicurrent = mhi = MAXVALUE" with 2): the range of a
   * dynamic map that no frame has bounded yet, which is what a colour scale of the first output reads.
   */
  private locurrent = 1;
  private hicurrent = 2;
  private roundDown = false;

  /** The dump_image.html default: amap = min max cf 0.0 2 min blue max red. */
  static default(): ColorMap {
    const m = new ColorMap();
    m.entries = [
      // amap = min max cf 0.0 2 min blue max red: color2rgb("blue") and color2rgb("red") (image.cpp rgbcolors, 0..1)
      { pos: MIN, lo: MIN, hi: MIN, color: [0, 0, 1] },
      { pos: MAX, lo: MIN, hi: MIN, color: [1, 0, 0] },
    ];
    return m;
  }

  /**
   * Parse the words after the amap keyword. `colorOf` resolves a colour name (null when unknown). Returns the
   * colour map and the number of words it used.
   */
  static parse(words: string[], colorOf: (name: string) => RGB | null): { map: ColorMap; used: number } {
    const m = new ColorMap();
    const id = 'dump_modify amap';
    m.lo = parseBound(words[0], 'lo');
    m.hi = parseBound(words[1], 'hi');
    if (m.lo.kind === 'num' && m.hi.kind === 'num' && m.lo.v >= m.hi.v) throw new StyleError(`${id}: lo must be smaller than hi`);
    const st = words[2] ?? '';
    if (st.length !== 2) throw new StyleError(`${id}: colour map style must be two letters (e.g. cf, ca, sa, da), got '${st}'`);
    if (st[0] !== 'c' && st[0] !== 'd' && st[0] !== 's') throw new StyleError(`${id}: unknown colour map style '${st[0]}' (c, d or s)`);
    if (st[1] !== 'a' && st[1] !== 'f') throw new StyleError(`${id}: unknown colour map range '${st[1]}' (a or f)`);
    m.style = st[0];
    m.absolute = st[1] === 'a';
    // dump_image.html: "amap = lo hi style delta N entry1 ..." with "delta = binsize (only used for style s, otherwise
    // ignored)"; the delta word is present for every style, so N is always the word after it
    let k = 3;
    if (m.style === 's') {
      m.binsize = Number(words[k]);
      if (!(m.binsize > 0)) throw new StyleError(`${id}: the sequential bin size must be > 0`);
    }
    k++;
    const n = Number(words[k]);
    if (!Number.isInteger(n) || n < 1) throw new StyleError(`${id}: needs at least 1 colour map entry`);
    k++;
    const colour = (w: string | undefined): RGB => {
      const c = w === undefined ? null : colorOf(w);
      if (!c) throw new StyleError(`${id}: unknown colour '${w ?? ''}'`);
      return c;
    };
    for (let i = 0; i < n; i++) {
      if (m.style === 'c') {
        m.entries.push({ pos: parseBound(words[k], 'entry value'), lo: MIN, hi: MIN, color: colour(words[k + 1]) });
        k += 2;
      } else if (m.style === 'd') {
        m.entries.push({ pos: MIN, lo: parseBound(words[k], 'entry lo'), hi: parseBound(words[k + 1], 'entry hi'), color: colour(words[k + 2]) });
        k += 3;
      } else {
        m.entries.push({ pos: MIN, lo: MIN, hi: MIN, color: colour(words[k]) });
        k += 1;
      }
    }
    if (m.style === 'c') {
      if (n < 2) throw new StyleError(`${id}: the continuous style needs at least 2 entries`);
      if (m.entries[0].pos.kind !== 'min') throw new StyleError(`${id}: the first continuous entry must be min`);
      if (m.entries[n - 1].pos.kind !== 'max') throw new StyleError(`${id}: the last continuous entry must be max`);
      for (let i = 1; i < n - 1; i++) {
        const p = m.entries[i].pos, q = m.entries[i - 1].pos;
        if (p.kind !== 'num') throw new StyleError(`${id}: only the first and last continuous entries may be min or max`);
        if (q.kind === 'num' && p.v <= q.v) throw new StyleError(`${id}: the continuous entries must increase`);
      }
    } else if (m.style === 'd') {
      if (m.entries[n - 1].hi.kind !== 'max') throw new StyleError(`${id}: the last discrete entry must end at max`);
    }
    // a static range is fixed when the map is set (ColorMap::reset calls minmax for a map that is not dynamic)
    if (m.lo.kind === 'num' && m.hi.kind === 'num') m.bounds(0, 0);
    return { map: m, used: k };
  }

  /** The map as it stands now, detached from later bounds() calls (see ImageDump.colorScaleMap). */
  snapshot(): ColorMap {
    const m = new ColorMap();
    Object.assign(m, this);
    return m;
  }

  /** ColorMap::info: the current range and whether the map is sequential (which adds one to the top of the scale). */
  info(): { lo: number; hi: number; sequential: boolean } {
    return { lo: this.locurrent, hi: this.hicurrent, sequential: this.style === 's' };
  }

  /**
   * Fix the range from the dynamic minimum and maximum of the values (ColorMap::minmax); explicit numbers
   * override them. Throws when the range is empty.
   */
  bounds(dynMin: number, dynMax: number): void {
    this.locurrent = this.lo.kind === 'num' ? this.lo.v : dynMin;
    this.hicurrent = this.hi.kind === 'num' ? this.hi.v : dynMax;
    if (this.locurrent > this.hicurrent) throw new StyleError('dump_modify amap: the colour map range is empty for these atom values');
    const value = (b: Bound, minFrac: number, maxFrac: number): number =>
      b.kind === 'num' ? b.v : b.kind === 'min' ? (this.absolute ? this.locurrent : minFrac) : (this.absolute ? this.hicurrent : maxFrac);
    if (this.style === 'c') {
      this.pos = this.entries.map((e) => value(e.pos, 0, 1));
    } else if (this.style === 'd') {
      this.bins = this.entries.map((e) => [value(e.lo, 0, 1), value(e.hi, 0, 1)] as [number, number]);
    } else {
      // the bin edge test of ColorMap::minmax (src/GRAPHICS/image.cpp at b332f28, EPSBIN_ABSOLUTE 1e-9 and
      // EPSBIN_FRACTIONAL 1e-6): a range that ends just on a bin edge gives its top value the bin below.
      const span = this.absolute ? this.hicurrent - this.locurrent : 1;
      const epsbin = this.absolute ? span * 1.0e-9 : 1.0e-6;
      const inv = 1 / this.binsize;
      this.roundDown = Math.trunc((span - epsbin) * inv) < Math.trunc(span * inv);
    }
  }

  /** The colour of one value (bounds() must have been called). */
  color(value: number): RGB {
    let v = Math.min(Math.max(value, this.locurrent), this.hicurrent);
    let lo = this.locurrent, hi = this.hicurrent;
    if (!this.absolute) {
      v = this.locurrent === this.hicurrent ? 0 : (v - this.locurrent) / (this.hicurrent - this.locurrent);
      lo = 0;
      hi = 1;
    }
    const n = this.entries.length;
    if (this.style === 'c') {
      for (let i = 0; i < n - 1; i++) {
        if (v >= this.pos[i] && v <= this.pos[i + 1]) {
          const f = (v - this.pos[i]) / (this.pos[i + 1] - this.pos[i]);
          const a = this.entries[i].color, b = this.entries[i + 1].color;
          return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
        }
      }
      return this.entries[0].color;
    }
    if (this.style === 'd') {
      for (let i = 0; i < n; i++) if (v >= this.bins[i][0] && v <= this.bins[i][1]) return this.entries[i].color;
      return this.entries[0].color;
    }
    let ibin = Math.trunc((v - lo) * (1 / this.binsize));
    if (this.roundDown && v === hi && ibin > 0) ibin--;
    return this.entries[((ibin % n) + n) % n].color;
  }
}
