import type { Atom, ColorBy, ColorMapName, VisualizationConfig } from '../types';

/**
 * Colour maps for per-atom quantities: nine sRGB stops each, sampled from matplotlib's viridis, plasma, inferno,
 * turbo and coolwarm (the perceptually uniform and diverging maps), a hue rainbow and a grey ramp, interpolated
 * linearly between stops as matplotlib's ListedColormap does with this many entries.
 */
const STOPS: Record<ColorMapName, string> = {
  viridis: '440154472c7a3b518b2c718e21908d27ad815cc863aadc32fde725',
  plasma: '0d08874c02a17e03a8a92395cc4778e56b5df89540fdc527f0f921',
  inferno: '0000041f0c48550f6d88226aba3655e35933f98e09f8c932fcffa4',
  turbo: '30123b4662d736aaf91ae4b672fe5ec7ef34fabb39f66b197a0403',
  coolwarm: '3b4cc06282ea8db0feb8d0f9ddddddf5c4acf49a7bde604db40426',
  rainbow: '0000ff0080ff00ffff00ff8000ff0080ff00ffff00ff8000ff0000',
  grayscale: '0000002020204040406060608080809f9f9fbfbfbfdfdfdfffffff',
};

export const COLOR_MAP_NAMES = Object.keys(STOPS) as ColorMapName[];

/** The colour modes and their labels (the viewer's picker and legend). */
export const COLOR_BY_LABELS: Record<ColorBy, string> = {
  type: 'Element / type', charge: 'Charge', speed: 'Speed |v|', id: 'Atom ID', molecule: 'Molecule', x: 'Position x', y: 'Position y', z: 'Position z',
};

const parsed = new Map<ColorMapName, Float32Array>();
/** The stops of a map as sRGB triples in 0..1. */
const stopsOf = (name: ColorMapName): Float32Array => {
  let s = parsed.get(name);
  if (!s) {
    const hex = STOPS[name] ?? STOPS.viridis;
    s = new Float32Array(hex.length / 2);
    for (let k = 0; k < s.length; k++) s[k] = parseInt(hex.slice(2 * k, 2 * k + 2), 16) / 255;
    parsed.set(name, s);
  }
  return s;
};

/** The sRGB colour (0..1) at t in [0, 1] of a map, written into out[0..2]. */
export const sampleColorMap = (name: ColorMapName, t: number, out: Float32Array | number[], reverse = false): void => {
  const s = stopsOf(name);
  const n = s.length / 3 - 1;
  let u = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  if (reverse) u = 1 - u;
  const x = u * n;
  const k = Math.min(n - 1, Math.floor(x));
  const f = x - k;
  for (let c = 0; c < 3; c++) out[c] = s[3 * k + c] * (1 - f) + s[3 * k + 3 + c] * f;
};

/** A CSS linear-gradient of a map, left to right (the legend bar). */
export const colorMapGradient = (name: ColorMapName, reverse = false): string => {
  const s = stopsOf(name);
  const n = s.length / 3;
  const parts: string[] = [];
  for (let k = 0; k < n; k++) {
    const j = reverse ? n - 1 - k : k;
    const rgb = [s[3 * j], s[3 * j + 1], s[3 * j + 2]].map((v) => Math.round(v * 255)).join(',');
    parts.push(`rgb(${rgb}) ${((100 * k) / (n - 1)).toFixed(1)}%`);
  }
  return `linear-gradient(to right, ${parts.join(', ')})`;
};

/** The quantity an atom is coloured by (NaN when the structure does not carry it: a speed without velocities). */
export const atomScalar = (a: Atom, by: ColorBy): number => {
  switch (by) {
    case 'charge': return a.charge;
    case 'speed': return a.vx === undefined ? NaN : Math.hypot(a.vx, a.vy ?? 0, a.vz ?? 0);
    case 'id': return a.id;
    case 'molecule': return a.molId;
    case 'x': return a.x;
    case 'y': return a.y;
    case 'z': return a.z;
    default: return NaN;
  }
};

/** Whether a colour mode maps a continuous quantity (and shows a legend); type and molecule are categorical. */
export const isContinuous = (by: ColorBy | undefined): boolean => by !== undefined && by !== 'type' && by !== 'molecule';

/** Min and max of the quantity over the given atom lists (a loop: no spread over data-sized arrays); null if none. */
export const scalarRange = (lists: readonly Atom[][], by: ColorBy): [number, number] | null => {
  let lo = Infinity, hi = -Infinity;
  for (const atoms of lists) {
    for (let i = 0; i < atoms.length; i++) {
      const v = atomScalar(atoms[i], by);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  return lo <= hi ? [lo, hi] : null;
};

/** A distinct colour per molecule id: hues a golden-ratio step apart, sRGB 0..1 into out. */
export const categoricalColor = (k: number, out: Float32Array | number[]): void => {
  const h = (((k * 0.6180339887) % 1) + 1) % 1;
  // HSL with s = 0.65, l = 0.55
  const l = 0.55, sat = 0.65;
  const q = l + sat - l * sat, p = 2 * l - q;
  const hue = (t: number): number => {
    const u = ((t % 1) + 1) % 1;
    if (u < 1 / 6) return p + (q - p) * 6 * u;
    if (u < 1 / 2) return q;
    if (u < 2 / 3) return p + (q - p) * (2 / 3 - u) * 6;
    return p;
  };
  out[0] = hue(h + 1 / 3);
  out[1] = hue(h);
  out[2] = hue(h - 1 / 3);
};

/** Grey for an atom whose quantity is missing. */
export const MISSING_COLOR = 0.5;

/** '#rrggbb' (or 'rrggbb') into sRGB 0..1; magenta for anything else. */
export const hexToRgb = (hex: string, out: Float32Array | number[]): void => {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex);
  const v = m ? parseInt(m[1], 16) : 0xff00ff;
  out[0] = (v >> 16) / 255;
  out[1] = ((v >> 8) & 255) / 255;
  out[2] = (v & 255) / 255;
};

/** Writes an atom's sRGB colour (0..1) into out[0..2]. */
export type AtomColorer = (a: Atom, out: Float32Array) => void;

/**
 * The colour of each atom under a view: its type colour (customColors, fallback when unset), a hue per molecule, or
 * its quantity through the colour map over colorMin..colorMax (grey when the structure lacks the quantity).
 */
export const makeAtomColorer = (c: VisualizationConfig, fallback: string): AtomColorer => {
  const by = c.colorBy ?? 'type';
  if (by === 'type') {
    const cache = new Map<number, Float32Array>();
    return (a, out) => {
      let v = cache.get(a.type);
      if (!v) {
        v = new Float32Array(3);
        hexToRgb(c.customColors[a.type] || fallback, v);
        cache.set(a.type, v);
      }
      out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
    };
  }
  if (by === 'molecule') return (a, out) => categoricalColor(a.molId, out);
  const lo = c.colorMin ?? 0, span = (c.colorMax ?? 1) - lo;
  const map = c.colorMap ?? 'viridis', rev = !!c.colorReverse;
  return (a, out) => {
    const v = atomScalar(a, by);
    if (!Number.isFinite(v)) {
      out[0] = out[1] = out[2] = MISSING_COLOR;
      return;
    }
    sampleColorMap(map, span > 0 ? (v - lo) / span : 0.5, out, rev);
  };
};

/** A key that changes exactly when makeAtomColorer would colour some atom differently. */
export const colorKeyOf = (c: VisualizationConfig): string => {
  const by = c.colorBy ?? 'type';
  if (by === 'molecule') return by;
  if (by !== 'type') return `${by}|${c.colorMap ?? 'viridis'}|${c.colorReverse ? 1 : 0}|${c.colorMin}|${c.colorMax}`;
  return Object.keys(c.customColors)
    .sort((x, y) => Number(x) - Number(y))
    .map((k) => `${k}:${c.customColors[Number(k)]}`)
    .join(';');
};
