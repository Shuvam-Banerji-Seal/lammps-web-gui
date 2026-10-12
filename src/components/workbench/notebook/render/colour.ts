/**
 * Atom colouring for the Render panel: which per-atom quantity colours the atoms (type, speed, kinetic energy,
 * height along an axis, atom id, one colour), the colour map for the scalar modes and the legend's labels.
 * Pure: the colours are sRGB triples in 0..1, and the scene writes them into the instanced colour buffer.
 */

export type ColourMode = 'type' | 'speed' | 'ke' | 'x' | 'y' | 'z' | 'id' | 'single';

export const COLOUR_LABEL: Record<ColourMode, string> = {
  type: 'Type (view colours)',
  speed: 'Speed',
  ke: 'Kinetic energy (1/2 m v²)',
  x: 'Height along x',
  y: 'Height along y',
  z: 'Height along z',
  id: 'Atom id',
  single: 'Single colour',
};

/** The modes that colour by a number and get a colour bar. */
export const SCALAR_MODES: readonly ColourMode[] = ['speed', 'ke', 'x', 'y', 'z', 'id'];

/** A recorded frame as the colour code needs it: step, wrapped positions, box bounds and atom ids. */
export interface ColourFrame {
  step: number;
  x: ArrayLike<number>;
  /** Velocities (x, y, z per atom) when the frame carries them; optional. */
  v?: ArrayLike<number>;
  /** Mass per atom when the frame carries it (kinetic energy uses it); absent: every atom has mass 1. */
  mass?: ArrayLike<number>;
  box: { lo: readonly number[]; hi: readonly number[] };
  topo: { id: Int32Array };
}

/** True when two frames hold the same atoms in the same order (the buffer shares topology between frames). */
export const sameAtoms = (a: ColourFrame, b: ColourFrame): boolean => {
  if (a.topo === b.topo) return true;
  const ia = a.topo.id;
  const ib = b.topo.id;
  if (ia.length !== ib.length) return false;
  for (let i = 0; i < ia.length; i++) if (ia[i] !== ib[i]) return false;
  return true;
};

/** Minimum-image displacement along one axis for a periodic box of length L (orthogonal approximation). */
const wrapDelta = (d: number, L: number): number => (L > 0 ? d - L * Math.round(d / L) : d);

/**
 * Speed of each atom in the frame, from its displacement between the neighbouring recorded frames (central
 * difference, one-sided at the ends), in length units per run time unit: dt is the timestep. Periodic
 * displacements use the minimum image. Null when the time between the frames is not known (no dt, or the
 * neighbours share the frame's step).
 */
export const displacementSpeeds = (
  prev: ColourFrame | undefined,
  cur: ColourFrame,
  next: ColourFrame | undefined,
  dt: number | null,
): Float64Array | null => {
  const from = prev && sameAtoms(prev, cur) ? prev : undefined;
  const to = next && sameAtoms(next, cur) ? next : undefined;
  const a = from ?? cur;
  const b = to ?? cur;
  const dStep = b.step - a.step;
  if (dt === null || !Number.isFinite(dt) || dt === 0 || dStep === 0) return null;
  const time = dStep * dt;
  const n = cur.topo.id.length;
  const out = new Float64Array(n);
  const L = [cur.box.hi[0] - cur.box.lo[0], cur.box.hi[1] - cur.box.lo[1], cur.box.hi[2] - cur.box.lo[2]];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < 3; k++) {
      const d = wrapDelta(Number(b.x[3 * i + k]) - Number(a.x[3 * i + k]), L[k]);
      s += d * d;
    }
    out[i] = Math.sqrt(s) / Math.abs(time);
  }
  return out;
};

/** Speeds from the frame's own velocities when it has them, else from the displacement fallback. */
export const frameSpeeds = (
  prev: ColourFrame | undefined,
  cur: ColourFrame,
  next: ColourFrame | undefined,
  dt: number | null,
): Float64Array | null => {
  if (cur.v && cur.v.length >= 3 * cur.topo.id.length) {
    const n = cur.topo.id.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const vx = Number(cur.v[3 * i]);
      const vy = Number(cur.v[3 * i + 1]);
      const vz = Number(cur.v[3 * i + 2]);
      out[i] = Math.sqrt(vx * vx + vy * vy + vz * vz);
    }
    return out;
  }
  return displacementSpeeds(prev, cur, next, dt);
};

/**
 * The per-atom number a scalar mode colours by. Speed and kinetic energy need the timestep (see frameSpeeds);
 * null when the mode cannot be computed for this frame.
 */
export const scalarValues = (
  mode: ColourMode,
  prev: ColourFrame | undefined,
  cur: ColourFrame,
  next: ColourFrame | undefined,
  dt: number | null,
): Float64Array | null => {
  const n = cur.topo.id.length;
  if (mode === 'id') {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = cur.topo.id[i];
    return out;
  }
  if (mode === 'x' || mode === 'y' || mode === 'z') {
    const axis = mode === 'x' ? 0 : mode === 'y' ? 1 : 2;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = Number(cur.x[3 * i + axis]);
    return out;
  }
  if (mode === 'speed' || mode === 'ke') {
    const sp = frameSpeeds(prev, cur, next, dt);
    if (!sp) return null;
    if (mode === 'speed') return sp;
    // kinetic energy 1/2 m v^2 per atom (mass 1 when the frame has no masses)
    for (let i = 0; i < n; i++) {
      const m = cur.mass && i < cur.mass.length ? Number(cur.mass[i]) : 1;
      sp[i] = 0.5 * m * sp[i] * sp[i];
    }
    return sp;
  }
  return null;
};

export interface ValueRange {
  min: number;
  max: number;
}

/** The smallest and largest value of a set of arrays (a loop, not a spread: the arrays can be large). */
export const rangeOf = (arrays: readonly (ArrayLike<number> | null)[]): ValueRange | null => {
  let min = Infinity;
  let max = -Infinity;
  for (const arr of arrays) {
    if (!arr) continue;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min <= max ? { min, max } : null;
};

/** Colour stops of the scalar map, from low to high, sRGB 0..1 (the map is this project's own). */
const STOPS: [number, number, number][] = [
  [0.13, 0.07, 0.36],
  [0.23, 0.42, 0.78],
  [0.18, 0.66, 0.55],
  [0.67, 0.84, 0.27],
  [0.98, 0.85, 0.21],
];

/** sRGB colour of the scalar map at t in 0..1 (clamped). */
export const heatColour = (t: number): [number, number, number] => {
  const c = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (STOPS.length - 1);
  const k = Math.min(STOPS.length - 2, Math.floor(c));
  const f = c - k;
  const a = STOPS[k];
  const b = STOPS[k + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
};

/** "#rrggbb" to sRGB 0..1; an invalid string gives magenta (the engine's unknown-type colour). */
export const hexToRgb = (hex: string): [number, number, number] => {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [1, 0, 1];
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
};

/**
 * Per-atom sRGB colours (3 floats per atom) for a scalar mode over the given range. A degenerate range (all
 * values equal) takes the middle of the map. `single` fills every atom with one colour.
 */
export const atomColours = (
  values: ArrayLike<number> | null,
  count: number,
  range: ValueRange | null,
  single: string,
): Float32Array => {
  const out = new Float32Array(3 * count);
  if (!values || !range) {
    const [r, g, b] = hexToRgb(single);
    for (let i = 0; i < count; i++) { out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b; }
    return out;
  }
  const span = range.max - range.min;
  for (let i = 0; i < count; i++) {
    const t = span > 0 ? (Number(values[i]) - range.min) / span : 0.5;
    const [r, g, b] = heatColour(t);
    out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b;
  }
  return out;
};

/** The two colour-bar labels for a range (three significant digits; a flat range shows one value). */
export const colourBarLabels = (range: ValueRange): { low: string; high: string } => {
  const f = (v: number) => (Number.isFinite(v) ? String(Number(v.toPrecision(3))) : '-');
  return { low: f(range.min), high: f(range.max) };
};
