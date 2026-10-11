/**
 * Pure data code for the notebook's graphs: thermo series extraction, statistics, linear fits, the
 * fluctuation formulas for heat capacity and compressibility, unit labels and CSV text.
 * No React, no DOM. Long series are never spread into Math.min/Math.max or function arguments.
 */
import type { ThermoRow } from '../../../engine/types';
import { thermoUnit } from '../../../lammps/thermoUnits';

export interface XY { x: number; y: number }

export type XMode = 'step' | 'time';
export type Ensemble = 'NVE' | 'NVT' | 'NPT' | 'unknown';

/** One thermo table of a cell's run: its columns, rows, units and the ensemble its cell's fixes imply. */
export interface ThermoSegment {
  label: string;
  keywords: string[];
  labels?: string[];
  units?: string;
  rows: ThermoRow[];
  /** Timestep in the units style's time unit, used when the rows carry no `time` column. */
  dt: number | null;
  ensemble: Ensemble;
}

/* ---------- units and constants ---------- */

/**
 * Boltzmann constant in the units of each style, derived from SI values (docs.lammps.org/units.html lists the
 * style constants; they are these values in the style's units). null for styles whose constant is not used here.
 */
const KB_SI = 1.380649e-23; // J/K
const NA = 6.02214076e23; // 1/mol
const EV_J = 1.602176634e-19; // J per eV
const KCAL_J = 4184; // J per kcal
const HARTREE_J = 4.3597447222071e-18; // J per Hartree
export const boltzmannConstant = (units: string | undefined): number | null => {
  switch (units) {
    case 'lj': return 1;
    case 'real': return (KB_SI * NA) / KCAL_J; // kcal/(mol K)
    case 'metal': return KB_SI / EV_J; // eV/K
    case 'si': return KB_SI; // J/K
    case 'cgs': return KB_SI * 1e7; // erg/K
    case 'electron': return KB_SI / HARTREE_J; // Hartree/K
    default: return null;
  }
};

/** Label of a column with its unit, e.g. "temp [Kelvin]"; the unit comes from the thermo unit table. */
export const columnLabel = (key: string, units: string | undefined): string => {
  const u = thermoUnit(units, key);
  return u ? `${key} [${u}]` : key;
};

/** Name of the time unit ("femtoseconds", ...) for a units style; "lj time" for lj (unitless). */
export const timeUnitName = (units: string | undefined): string => {
  if (!units || units === 'lj') return 'lj time';
  return thermoUnit(units, 'time') ?? 'time units';
};

/** Unit label of the x axis in time mode, e.g. "time [femtoseconds]". */
export const timeUnitLabel = (units: string | undefined): string =>
  units === undefined || units === 'lj' ? 'time (lj units)' : `time [${timeUnitName(units)}]`;

/** Which ensemble a cell's fix commands imply: fix <id> <group> nve|nvt|npt|nph, or a thermostat fix. */
export const ensembleOf = (script: string): Ensemble => {
  const fixes = Array.from(script.matchAll(/^\s*fix\s+\S+\s+\S+\s+(\S+)/gm), (m) => m[1].toLowerCase());
  if (fixes.some((s) => s.startsWith('npt') || s.startsWith('nph'))) return 'NPT';
  if (fixes.some((s) => s.startsWith('nvt') || s === 'langevin' || s.startsWith('temp/'))) return 'NVT';
  if (fixes.some((s) => s === 'nve')) return 'NVE';
  return 'unknown';
};

/* ---------- series ---------- */

/**
 * Timestep implied by the rows: time / step from a row that has both columns (exact, LAMMPS `time` is step x dt),
 * else the segment's dt. null when neither is known.
 */
export const segmentDt = (seg: ThermoSegment): number | null => {
  for (let i = seg.rows.length - 1; i >= 0; i--) {
    const r = seg.rows[i];
    const t = r.time, s = r.step;
    if (Number.isFinite(t) && Number.isFinite(s) && s > 0) return t / s;
  }
  return seg.dt !== null && Number.isFinite(seg.dt) && seg.dt > 0 ? seg.dt : null;
};

/** Numeric thermo columns (step excluded): every keyword with at least one finite value in any segment. */
export const numericColumns = (segments: readonly ThermoSegment[]): string[] => {
  const seen = new Map<string, boolean>();
  for (const seg of segments) {
    for (const k of seg.keywords) {
      if (k === 'step') continue;
      if (!seen.has(k)) seen.set(k, false);
      if (seen.get(k)) continue;
      for (let i = 0; i < seg.rows.length; i++) {
        if (Number.isFinite(seg.rows[i][k])) { seen.set(k, true); break; }
      }
    }
  }
  const out: string[] = [];
  for (const [k, ok] of seen) if (ok) out.push(k);
  return out;
};

/** Points of one column of one segment in the chosen x mode. Non-finite values and rows are skipped. */
export const seriesPoints = (seg: ThermoSegment, key: string, mode: XMode, dtOverride?: number | null): XY[] => {
  const dt = mode === 'time' ? (dtOverride ?? segmentDt(seg)) : 1;
  if (mode === 'time' && (dt === null || dt === undefined)) return [];
  const out: XY[] = [];
  for (let i = 0; i < seg.rows.length; i++) {
    const r = seg.rows[i];
    const y = r[key];
    const s = r.step;
    if (!Number.isFinite(y) || !Number.isFinite(s)) continue;
    out.push({ x: mode === 'time' ? s * (dt as number) : s, y });
  }
  return out;
};

/** Trailing running mean over `window` points (window < 2 returns the input). O(n). */
export const runningMean = (points: readonly XY[], window: number): XY[] => {
  if (window < 2 || points.length === 0) return points.slice();
  const out: XY[] = new Array(points.length);
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    sum += points[i].y;
    if (i >= window) sum -= points[i - window].y;
    const n = Math.min(i + 1, window);
    out[i] = { x: points[i].x, y: sum / n };
  }
  return out;
};

/** Evenly spaced indices 0..n-1 (at most `max` of them): long series are sampled, never copied whole. */
export const pickEvenly = (n: number, max: number): number[] => {
  if (n <= 0 || max <= 0) return [];
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const out: number[] = new Array(max);
  for (let k = 0; k < max; k++) out[k] = Math.floor((k * (n - 1)) / (max - 1));
  return out;
};

/* ---------- statistics ---------- */

export interface Summary { n: number; mean: number; std: number; min: number; max: number }

/** Mean, sample standard deviation (n-1), min and max of the finite values. NaN fields when there are none. */
export const summarize = (values: readonly number[]): Summary => {
  let n = 0, sum = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    n++;
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (n === 0) return { n: 0, mean: NaN, std: NaN, min: NaN, max: NaN };
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) ss += (v - mean) * (v - mean);
  }
  return { n, mean, std: n > 1 ? Math.sqrt(ss / (n - 1)) : NaN, min, max };
};

/** Population variance <x^2> - <x>^2 (computed about the mean, so it does not cancel), or NaN when empty. */
export const populationVariance = (values: readonly number[]): number => {
  let n = 0, sum = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) { n++; sum += values[i]; }
  if (n === 0) return NaN;
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) ss += (values[i] - mean) ** 2;
  return ss / n;
};

export interface LinearFit { slope: number; intercept: number; r2: number; n: number }

/** Least-squares line through the finite points. null with fewer than two points or no spread in x. */
export const linearFit = (points: readonly XY[]): LinearFit | null => {
  let n = 0, sx = 0, sy = 0;
  for (let i = 0; i < points.length; i++) {
    const { x, y } = points[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n++; sx += x; sy += y;
  }
  if (n < 2) return null;
  const mx = sx / n, my = sy / n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < points.length; i++) {
    const { x, y } = points[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    sxx += (x - mx) * (x - mx);
    sxy += (x - mx) * (y - my);
    syy += (y - my) * (y - my);
  }
  if (!(sxx > 0)) return null;
  const slope = sxy / sxx;
  const r2 = syy > 0 ? (sxy * sxy) / (sxx * syy) : 1;
  return { slope, intercept: my - slope * mx, r2, n };
};

/**
 * Heat capacity from energy fluctuations (canonical ensemble, NVT):
 *   C_V = (<E^2> - <E>^2) / (k_B T^2),
 * with E the total energy of the run's rows and T the mean temperature of the same rows. Returned for the whole
 * system and per atom. Only meaningful for an NVT run that has equilibrated; the caller shows that caveat.
 */
export const heatCapacity = (
  energies: readonly number[], temps: readonly number[], atoms: number, units: string | undefined,
): { total: number; perAtom: number | null; kB: number; meanT: number } | { error: string } => {
  const kB = boltzmannConstant(units);
  if (kB === null) return { error: `no Boltzmann constant for units ${units ?? 'unset'}` };
  const T = summarize(temps).mean;
  if (!(T > 0)) return { error: 'mean temperature is not positive' };
  const varE = populationVariance(energies);
  if (!Number.isFinite(varE)) return { error: 'no finite energy values' };
  const total = varE / (kB * T * T);
  return { total, perAtom: atoms > 0 ? total / atoms : null, kB, meanT: T };
};

/**
 * Isothermal compressibility from volume fluctuations (isothermal-isobaric ensemble, NPT):
 *   kappa_T = (<V^2> - <V>^2) / (k_B T <V>),   bulk modulus K = 1 / kappa_T.
 * Units: kappa in the inverse of the pressure unit, K in the pressure unit (lj: unitless).
 */
export const compressibility = (
  volumes: readonly number[], temps: readonly number[], units: string | undefined,
): { kappa: number; bulk: number; kB: number; meanT: number; meanV: number } | { error: string } => {
  const kB = boltzmannConstant(units);
  if (kB === null) return { error: `no Boltzmann constant for units ${units ?? 'unset'}` };
  const T = summarize(temps).mean;
  const V = summarize(volumes).mean;
  if (!(T > 0) || !(V > 0)) return { error: 'mean temperature and volume must be positive' };
  const varV = populationVariance(volumes);
  if (!Number.isFinite(varV) || !(varV > 0)) return { error: 'volume does not fluctuate' };
  const kappa = varV / (kB * T * V);
  return { kappa, bulk: 1 / kappa, kB, meanT: T, meanV: V };
};

/**
 * Energy drift per atom per unit time for an NVE run: the slope of total energy against time, divided by the
 * atom count. The caller passes points in time mode.
 */
export const energyDrift = (
  points: readonly XY[], atoms: number,
): { perAtomPerTime: number; r2: number } | null => {
  const fit = linearFit(points);
  if (!fit || !(atoms > 0)) return null;
  return { perAtomPerTime: fit.slope / atoms, r2: fit.r2 };
};

/* ---------- trajectory analysis helpers ---------- */

export interface DiffusionFit { D: number; slope: number; intercept: number; r2: number; tFrom: number; tTo: number; n: number; dim: number }

/**
 * Diffusion coefficient from the linear part of MSD(t): D = slope / (2 dim), the slope fitted over
 * t in [fromFrac, toFrac] x t_max (lag 0 excluded). null when the window has fewer than two points.
 */
export const fitDiffusion = (msd: readonly XY[], fromFrac: number, toFrac: number, dim: 2 | 3): DiffusionFit | null => {
  if (msd.length < 2) return null;
  const tMax = msd[msd.length - 1].x;
  const tFrom = fromFrac * tMax, tTo = toFrac * tMax;
  const window: XY[] = [];
  for (let i = 0; i < msd.length; i++) {
    const p = msd[i];
    if (p.x > 0 && p.x >= tFrom && p.x <= tTo) window.push(p);
  }
  const fit = linearFit(window);
  if (!fit) return null;
  return { D: fit.slope / (2 * dim), slope: fit.slope, intercept: fit.intercept, r2: fit.r2, tFrom, tTo, n: fit.n, dim };
};

/** Maxwell-Boltzmann speed density f(v) for mass m at temperature kT (kT in the same units as m v^2). */
export const maxwellBoltzmannSpeed = (v: number, kT: number, m: number, dim: 2 | 3): number => {
  if (!(kT > 0) || !(m > 0) || v < 0) return 0;
  const a = m / (2 * kT);
  if (dim === 3) return 4 * Math.PI * (a / Math.PI) ** 1.5 * v * v * Math.exp(-a * v * v);
  return 2 * a * v * Math.exp(-a * v * v);
};

/* ---------- CSV ---------- */

/** RFC 4180 field: quoted when it holds a comma, quote or newline. */
const csvField = (v: string | number | undefined): string => {
  if (v === undefined || (typeof v === 'number' && !Number.isFinite(v))) return '';
  const s = String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** CSV text: a header row and one line per row (missing values blank). */
export const toCsv = (header: string[], rows: (string | number | undefined)[][]): string => {
  const lines: string[] = [header.map(csvField).join(',')];
  for (const r of rows) lines.push(r.map(csvField).join(','));
  return lines.join('\n') + '\n';
};
