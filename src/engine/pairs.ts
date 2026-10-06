import type { LJPair, PairTable } from './types';

/*
 * lj/cut coefficients and mixing.
 *
 * docs.lammps.org/pair_lj.html: "E = 4ε[(σ/r)^12 − (σ/r)^6]  r < rc";
 *   pair_coeff takes ε, σ and an optional cutoff that defaults to the global
 *   cutoff.
 * docs.lammps.org/pair_modify.html: mix geometric "ε_ij = √(ε_i ε_j)" and
 *   "σ_ij = √(σ_i σ_j)"; arithmetic "σ_ij = ½(σ_i + σ_j)"; sixthpower
 *   "ε_ij = (2√(ε_i ε_j) σ_i³ σ_j³)/(σ_i⁶ + σ_j⁶)", "σ_ij = (½(σ_i⁶ + σ_j⁶))^(1/6)";
 *   "the cutoff distance is mixed the same way as sigma."
 *   shift: "adds an energy term to each pairwise interaction ... but does not
 *   affect pair forces". Defaults "mix = geometric, shift = no".
 * docs.lammps.org/pair_coeff.html: wildcards "*", "*n", "n*", "m*n"; "only
 *   type pairs with I <= J are considered"; J,I is set to the I,J values.
 */

export const newPairTable = (ntypes: number, globalCutoff: number): PairTable => ({
  style: 'lj/cut',
  ntypes,
  globalCutoff,
  pairs: new Array((ntypes + 1) * (ntypes + 1)).fill(undefined),
  explicit: new Array((ntypes + 1) * (ntypes + 1)).fill(false),
  mix: 'geometric',
  shift: false,
});

const idx = (t: PairTable, i: number, j: number) => i * (t.ntypes + 1) + j;

/** Parses a pair_coeff type range ("2", "*", "*3", "2*", "1*3") into [lo, hi]. */
export const typeRange = (token: string, ntypes: number): [number, number] => {
  const star = token.indexOf('*');
  if (star < 0) {
    const n = Number(token);
    if (!Number.isInteger(n) || n < 1 || n > ntypes) throw new Error(`invalid atom type '${token}' (1..${ntypes})`);
    return [n, n];
  }
  const a = token.slice(0, star);
  const b = token.slice(star + 1);
  const lo = a === '' ? 1 : Number(a);
  const hi = b === '' ? ntypes : Number(b);
  if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 1 || hi > ntypes || lo > hi) {
    throw new Error(`invalid atom type range '${token}' (1..${ntypes})`);
  }
  return [lo, hi];
};

/** pair_coeff I J eps sigma [cutoff]. */
export const setPairCoeff = (
  t: PairTable, iTok: string, jTok: string, epsilon: number, sigma: number, cutoff?: number,
): number => {
  const [ilo, ihi] = typeRange(iTok, t.ntypes);
  const [jlo, jhi] = typeRange(jTok, t.ntypes);
  if (!(epsilon >= 0) || !(sigma > 0)) throw new Error('pair_coeff needs epsilon >= 0 and sigma > 0');
  const rc = cutoff ?? t.globalCutoff;
  if (!(rc > 0)) throw new Error('pair_coeff cutoff must be > 0');
  let count = 0;
  for (let i = ilo; i <= ihi; i++) {
    for (let j = Math.max(i, jlo); j <= jhi; j++) {
      const p: LJPair = { epsilon, sigma, cutoff: rc };
      t.pairs[idx(t, i, j)] = p;
      t.pairs[idx(t, j, i)] = p;
      t.explicit[idx(t, i, j)] = true;
      t.explicit[idx(t, j, i)] = true;
      count++;
    }
  }
  return count;
};

const mixPair = (mix: PairTable['mix'], a: LJPair, b: LJPair): LJPair => {
  const eps = Math.sqrt(a.epsilon * b.epsilon);
  if (mix === 'geometric') {
    return { epsilon: eps, sigma: Math.sqrt(a.sigma * b.sigma), cutoff: Math.sqrt(a.cutoff * b.cutoff) };
  }
  if (mix === 'arithmetic') {
    return { epsilon: eps, sigma: 0.5 * (a.sigma + b.sigma), cutoff: 0.5 * (a.cutoff + b.cutoff) };
  }
  const s6 = (x: number) => x ** 6;
  const sixth = (p: number, q: number) => Math.pow(0.5 * (s6(p) + s6(q)), 1 / 6);
  return {
    epsilon: (2 * eps * a.sigma ** 3 * b.sigma ** 3) / (s6(a.sigma) + s6(b.sigma)),
    sigma: sixth(a.sigma, b.sigma),
    cutoff: sixth(a.cutoff, b.cutoff),
  };
};

/**
 * Fills every I != J pair not set explicitly by mixing I,I and J,J. Returns
 * the list of missing pairs (empty = ready to run).
 */
export const resolvePairs = (t: PairTable): string[] => {
  const missing: string[] = [];
  for (let i = 1; i <= t.ntypes; i++) {
    if (!t.pairs[idx(t, i, i)]) missing.push(`${i} ${i}`);
  }
  for (let i = 1; i <= t.ntypes; i++) {
    for (let j = i + 1; j <= t.ntypes; j++) {
      if (t.explicit[idx(t, i, j)]) continue;
      const a = t.pairs[idx(t, i, i)];
      const b = t.pairs[idx(t, j, j)];
      if (!a || !b) { missing.push(`${i} ${j}`); continue; }
      const p = mixPair(t.mix, a, b);
      t.pairs[idx(t, i, j)] = p;
      t.pairs[idx(t, j, i)] = p;
    }
  }
  return missing;
};

export const getPair = (t: PairTable, i: number, j: number): LJPair | undefined => t.pairs[idx(t, i, j)];

/** Largest cutoff over all type pairs (the cell size). */
export const maxCutoff = (t: PairTable): number => {
  let m = 0;
  for (const p of t.pairs) if (p && p.cutoff > m) m = p.cutoff;
  return m;
};

/**
 * Flat per-pair coefficient arrays, index i*(ntypes+1)+j, for the force
 * loops. With u = 1/r^2 and s6 = sigma^6 u^3:
 *   E(r)      = 4 eps s6 (s6 - 1)
 *   F(r)/r    = 24 eps u s6 (2 s6 - 1)        (textbook -dE/dr / r)
 */
export interface PairArrays {
  stride: number;
  cutsq: Float64Array;
  /** 4 eps sigma^12 and 4 eps sigma^6 (energy). */
  e12: Float64Array;
  e6: Float64Array;
  /** 48 eps sigma^12 and 24 eps sigma^6 (force / r^2 numerator). */
  f12: Float64Array;
  f6: Float64Array;
  /** Energy at the cutoff, subtracted when shift is on (0 otherwise). */
  eshift: Float64Array;
  maxCutoff: number;
}

export const pairArrays = (t: PairTable): PairArrays => {
  const stride = t.ntypes + 1;
  const n = stride * stride;
  const a: PairArrays = {
    stride,
    cutsq: new Float64Array(n), e12: new Float64Array(n), e6: new Float64Array(n),
    f12: new Float64Array(n), f6: new Float64Array(n), eshift: new Float64Array(n),
    maxCutoff: maxCutoff(t),
  };
  for (let k = 0; k < n; k++) {
    const p = t.pairs[k];
    if (!p) continue;
    const s6 = p.sigma ** 6;
    const s12 = s6 * s6;
    a.cutsq[k] = p.cutoff * p.cutoff;
    a.e12[k] = 4 * p.epsilon * s12;
    a.e6[k] = 4 * p.epsilon * s6;
    a.f12[k] = 48 * p.epsilon * s12;
    a.f6[k] = 24 * p.epsilon * s6;
    if (t.shift) {
      const rc6 = p.cutoff ** 6;
      a.eshift[k] = a.e12[k] / (rc6 * rc6) - a.e6[k] / rc6;
    }
  }
  return a;
};
