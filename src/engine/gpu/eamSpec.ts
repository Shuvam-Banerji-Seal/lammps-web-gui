import type { Pair } from '../force/types';
import type { EAMKernelTables } from '../force/pair/eamKernel';

/*
 * pair eam, eam/alloy and eam/fs on the GPU (gpu/eamKernels.ts, webgpuForces.ts computeEAM): the engine style's own
 * density and r*phi tables after its init (force/pair/eamKernel.ts, the seven coefficients per grid interval that its
 * loops read), packed once into one f32 array. The step is native's (src/MANYBODY/pair_eam.cpp compute at b332f28, as
 * eamKernel.ts follows it): densities rho_i summed over the neighbors, F'(rho_i) and the embedding energy from the
 * style's embedding spline (on the CPU, in fp64: eam.ts embedOwned), then the forces
 * -(F'_i rho'_ij(r) + F'_j rho'_ji(r) + phi'(r)) / r along each pair and the pair energy phi = z2r / r.
 * eamDensity and eamPair are the kernel's arithmetic in fp64, which the tests compare with the CPU style.
 */

/** The styles the GPU path runs this way (the engine classes of force/pair/eam.ts with these names). */
export const GPU_EAM_STYLES: ReadonlySet<string> = new Set(['eam', 'eam/alloy', 'eam/fs']);

/** Words per type pair at the start of coef: the density table's offset, the r*phi table's offset, cutsq, scale. */
export const EAM_HEADER = 4;

export interface GpuEamSpec {
  /** ntypes + 1: type pair (i, j) is entry i * stride + j. */
  stride: number;
  /** the tables' grid: nr points, 1/dr */
  nr: number;
  rdr: number;
  /**
   * EAM_HEADER words per type pair (offsets as f32 integers, exact below 2^24, 0 for none: as u32 bit patterns they
   * would be denormals, which a GPU may flush to zero), then the tables, seven coefficients per grid interval
   * (eamKernel.ts): the derivative's three, then the value's four.
   */
  coef: Float32Array;
  maxCutoff: number;
  /** Changes when the tables do (the backend caches the uploaded buffer by it). */
  key: string;
}

/** An EAM style of force/pair/eam.ts after its init: its tables and its embedding step. */
export interface EamPair extends Pair {
  gpuTables(): EAMKernelTables;
  embedOwned(type: ArrayLike<number>, rho: ArrayLike<number>, n: number, fp: Float64Array): number;
}

/** The style is one the GPU path runs as EAM (by name, and with the tables of eam.ts). */
export const isGpuEam = (p: Pair | null): p is EamPair =>
  !!p && GPU_EAM_STYLES.has(p.name) && typeof (p as Partial<EamPair>).gpuTables === 'function' && typeof (p as Partial<EamPair>).embedOwned === 'function';

let keyCounter = 0;
const tableKeys = new WeakMap<Float64Array, number>();
/** A stable id per table array (the style rebuilds its splines when its files or coefficients change). */
const tableKey = (c: Float64Array): number => {
  let k = tableKeys.get(c);
  if (k === undefined) { k = ++keyCounter; tableKeys.set(c, k); }
  return k;
};

/** The packed tables of an initialized EAM style, or why the GPU cannot run it. */
export const gpuEamSpec = (pair: EamPair): GpuEamSpec | string => {
  const t = pair.gpuTables();
  const nt = t.ntypes + 1;
  let grid: { nr: number; rdr: number } | null = null;
  for (const sp of [...t.rhoTab, ...t.gTab]) {
    if (!sp) continue;
    if (!grid) grid = { nr: sp.n, rdr: sp.rdx };
    else if (sp.n !== grid.nr || sp.rdx !== grid.rdr) return `pair style ${pair.name} with tables on different grids`;
  }
  if (!grid) return `pair style ${pair.name} without tables`;
  // each distinct table once, after the header
  const offsets = new Map<Float64Array, number>();
  let len = EAM_HEADER * nt * nt;
  const place = (c: Float64Array | undefined): number => {
    if (!c) return 0;
    let o = offsets.get(c);
    if (o === undefined) { o = len; offsets.set(c, o); len += c.length; }
    return o;
  };
  const rhoOff = new Uint32Array(nt * nt), z2Off = new Uint32Array(nt * nt);
  let maxCut = 0;
  for (let i = 1; i < nt; i++) {
    for (let j = 1; j < nt; j++) {
      const k = i * nt + j;
      rhoOff[k] = place(t.rhoTab[k]?.coef);
      z2Off[k] = place(t.gTab[k]?.coef);
      if (t.cutsq[k] > 0) maxCut = Math.max(maxCut, Math.sqrt(t.cutsq[k]));
    }
  }
  // the kernel indexes coefficients as f32 integers below 2^24 (exact)
  if (len >= 2 ** 24) return `pair style ${pair.name} with tables too large for the GPU kernel`;
  const coef = new Float32Array(len);
  for (let k = 0; k < nt * nt; k++) {
    const h = EAM_HEADER * k;
    coef[h] = rhoOff[k];
    coef[h + 1] = z2Off[k];
    coef[h + 2] = rhoOff[k] && z2Off[k] ? t.cutsq[k] : 0;
    coef[h + 3] = t.scale[k] ?? 1;
  }
  for (const [c, o] of offsets) coef.set(c, o);
  const ids = [...offsets.keys()].map(tableKey).join(',');
  return { stride: nt, nr: grid.nr, rdr: grid.rdr, coef, maxCutoff: maxCut, key: `${nt}|${grid.nr}|${grid.rdr}|${ids}|${Array.from(t.cutsq).join(',')}|${Array.from(t.scale).join(',')}` };
};

/** Table index and fraction of distance r on the grid, as eamKernel.ts takes them: p = r rdr + 1, m <= nr - 1, p <= 1. */
const gridOf = (spec: GpuEamSpec, r: number): { o: number; p: number } => {
  let p = r * spec.rdr + 1.0;
  let m = Math.trunc(p);
  if (m > spec.nr - 1) m = spec.nr - 1;
  p -= m;
  if (p > 1.0) p = 1.0;
  return { o: 7 * (m - 1), p };
};

/** The density at a type-ti atom from a type-tj neighbor at squared distance rsq (0 outside the cutoff), in fp64. */
export const eamDensity = (spec: GpuEamSpec, ti: number, tj: number, rsq: number): number => {
  const c = spec.coef, h = EAM_HEADER * (ti * spec.stride + tj);
  if (!(rsq < c[h + 2])) return 0;
  const base = c[h];
  const { o, p } = gridOf(spec, Math.sqrt(rsq));
  const k = base + o;
  return ((c[k + 3] * p + c[k + 4]) * p + c[k + 5]) * p + c[k + 6];
};

/**
 * One pair's force over r (fpair, along x_i - x_j) and its energy phi, with F' of both atoms, in fp64: the kernel's
 * arithmetic.
 */
export const eamPair = (spec: GpuEamSpec, ti: number, tj: number, rsq: number, fpi: number, fpj: number): { fpair: number; phi: number } => {
  const c = spec.coef;
  const hij = EAM_HEADER * (ti * spec.stride + tj), hji = EAM_HEADER * (tj * spec.stride + ti);
  if (!(rsq < c[hij + 2])) return { fpair: 0, phi: 0 };
  const r = Math.sqrt(rsq);
  const { o, p } = gridOf(spec, r);
  const d = (base: number) => { const k = base + o; return (c[k] * p + c[k + 1]) * p + c[k + 2]; };
  // rhojp: the density at i due to j (table ij), rhoip: at j due to i (table ji)
  const rhojp = d(c[hij]), rhoip = d(c[hji]);
  const z = c[hij + 1] + o;
  const z2p = (c[z] * p + c[z + 1]) * p + c[z + 2];
  const z2 = ((c[z + 3] * p + c[z + 4]) * p + c[z + 5]) * p + c[z + 6];
  const recip = 1.0 / r;
  const phi = z2 * recip;
  const phip = z2p * recip - phi * recip;
  const psip = fpi * rhojp + fpj * rhoip + phip;
  const sc = c[hij + 3];
  return { fpair: -sc * psip * recip, phi: sc * phi };
};
