import type { SimState } from '../types';
import type { Neighbor } from '../neighbor';

/*
 * Bookkeeping of the magnetic precession vectors (SimState.fm) when several spin contributors act on the same
 * atoms: a spin pair style of a hybrid/overlay (spin/exchange, spin/dmi, ...), fix precession/spin and fix
 * setforce/spin. Each contributor keeps its last contribution per atom id; fm is the sum of the contributions,
 * updated by the difference when a contributor is evaluated again. The contributions are kept by atom id, so
 * they survive the atom sorting that permutes fm.
 */

interface Record { buf: Float64Array; e: number }
const books = new WeakMap<SimState, Map<object, Record>>();

const recordsOf = (s: SimState): Map<object, Record> => {
  let m = books.get(s);
  if (!m) { m = new Map(); books.set(s, m); }
  return m;
};

const maxIdOf = (s: SimState): number => {
  let m = 0;
  for (let i = 0; i < s.n; i++) if (s.id[i] > m) m = s.id[i];
  return m;
};

const recordFor = (s: SimState, owner: object): Record => {
  const m = recordsOf(s);
  let rec = m.get(owner);
  if (!rec) { rec = { buf: new Float64Array(0), e: 0 }; m.set(owner, rec); }
  const need = 3 * (maxIdOf(s) + 1);
  if (rec.buf.length < need) {
    const nb = new Float64Array(need);
    nb.set(rec.buf);
    rec.buf = nb;
  }
  return rec;
};

/**
 * Sets the contribution of `owner` to the precession vectors: vals[3i..3i+2] is the contribution of local atom i
 * (length >= 3 s.n) and energy its magnetic energy (pair owners; 0 for the others). fm and spinEnergy are updated.
 */
export const spinContribute = (s: SimState, owner: object, vals: Float64Array, energy = 0): void => {
  const fm = s.fm!;
  const rec = recordFor(s, owner);
  for (let i = 0; i < s.n; i++) {
    const t = 3 * s.id[i];
    for (let k = 0; k < 3; k++) {
      const v = vals[3 * i + k];
      fm[3 * i + k] += v - rec.buf[t + k];
      rec.buf[t + k] = v;
    }
  }
  rec.e = energy;
  let e = 0;
  for (const r of recordsOf(s).values()) e += r.e;
  s.spinEnergy = e;
};

/**
 * Contribution of `owner` to the precession vector of local atom i as last set (zero before the first set).
 */
export const spinOwnerValue = (s: SimState, owner: object, i: number, k: number): number => {
  const rec = recordsOf(s).get(owner);
  if (!rec) return 0;
  const t = 3 * s.id[i] + k;
  return t < rec.buf.length ? rec.buf[t] : 0;
};

/** Spin pair style that contributes precession vectors and reports its atom-level magnetic terms (fix nve/spin). */
export interface SpinPairStyle {
  readonly name: string;
  /** Precession vector of owned atom i from the current spins (out[0..2]). */
  omegaOf(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void;
  /**
   * Optional: native compute_single_pair of the sweep (fix nve/spin), which accumulates into fmi and may reset it. When
   * defined, the sweep calls it on the shared accumulator instead of omegaOf (the pair order of a hybrid then matters).
   */
  singlePair?(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, fmi: number[]): void;
  /**
   * Optional: the precession vector that the force evaluation stores in fm (native compute of the pair style, which
   * sums over all neighbours and carries no kspace field), when it differs from the sweep vector omegaOf (native
   * compute_single_pair of fix nve/spin). Used for the fm of the output after a step (fix nve/spin refreshFm).
   */
  outputOf?(i: number, nb: Neighbor, sp: Float64Array, type: Int32Array, x: Float64Array, list: { firstneigh: Int32Array; numneigh: Int32Array; neighbors: Int32Array }, out: Float64Array): void;
  /** True when the style adds mechanical forces (only with fix nve/spin lattice moving). */
  readonly spinForce: boolean;
}
