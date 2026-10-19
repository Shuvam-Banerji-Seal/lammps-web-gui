/**
 * Statistics of one thermo column's time series: block averages for the standard error of a correlated mean,
 * the normalised autocorrelation function, and a histogram. Pure functions, no DOM. Non-finite values are dropped
 * first, so the caller can pass a column as it is read from the thermo rows.
 */

const finiteOnly = (values: readonly number[]): number[] => values.filter((v) => Number.isFinite(v));

export interface BlockPoint {
  /** Number of consecutive values in each block. */
  blockSize: number;
  /** Mean of the block means (the series mean over the blocks used). */
  mean: number;
  /** Standard error of the block means: sqrt( sum (m_b - mean)^2 / (nb - 1) / nb ). */
  stdError: number;
}

/**
 * Block averaging (Flyvbjerg-Petersen style): for block sizes 1, 2, 4, ... the series is cut into nb = floor(n / size)
 * blocks of `size` consecutive values, and the standard error of the mean is taken over the block means. The
 * error grows with the block size while the values are correlated and levels off once blocks are independent.
 * Sizes are kept while nb >= minBlocks (at least 2, so the sample variance exists).
 */
export const blockAverage = (values: readonly number[], minBlocks = 4): BlockPoint[] => {
  const v = finiteOnly(values);
  const n = v.length;
  const need = Math.max(2, minBlocks);
  const out: BlockPoint[] = [];
  for (let size = 1; Math.floor(n / size) >= need; size *= 2) {
    const nb = Math.floor(n / size);
    const means = new Float64Array(nb);
    for (let b = 0; b < nb; b++) {
      let s = 0;
      for (let i = b * size; i < (b + 1) * size; i++) s += v[i];
      means[b] = s / size;
    }
    let mean = 0;
    for (let b = 0; b < nb; b++) mean += means[b];
    mean /= nb;
    let ss = 0;
    for (let b = 0; b < nb; b++) ss += (means[b] - mean) * (means[b] - mean);
    out.push({ blockSize: size, mean, stdError: Math.sqrt(ss / (nb - 1) / nb) });
  }
  return out;
};

/**
 * Normalised autocorrelation with the mean removed: C(k) = (1/n) sum_{t=0}^{n-k-1} (x_t - m)(x_{t+k} - m), and the
 * returned value at lag k is C(k) / C(0), for k = 0 .. min(maxLag, n - 1). The first value is 1. A constant series
 * (C(0) = 0) gives 1 at lag 0 and 0 elsewhere. Empty with fewer than two values.
 */
export const autocorrelation = (values: readonly number[], maxLag: number): number[] => {
  const v = finiteOnly(values);
  const n = v.length;
  if (n < 2) return [];
  const lagMax = Math.max(0, Math.min(Math.floor(maxLag), n - 1));
  let m = 0;
  for (let t = 0; t < n; t++) m += v[t];
  m /= n;
  const d = new Float64Array(n);
  for (let t = 0; t < n; t++) d[t] = v[t] - m;
  const c = new Float64Array(lagMax + 1);
  for (let k = 0; k <= lagMax; k++) {
    let s = 0;
    for (let t = 0; t + k < n; t++) s += d[t] * d[t + k];
    c[k] = s / n;
  }
  const c0 = c[0];
  return Array.from(c, (x, k) => (c0 > 0 ? x / c0 : k === 0 ? 1 : 0));
};

export interface Histogram1D {
  /** Bin centres. */
  centers: number[];
  /** Bin width. */
  width: number;
  counts: number[];
  /** count / (n * width): integrates to 1 over the bins. */
  density: number[];
}

/**
 * Histogram of `nbins` equal bins from the minimum to the maximum value (the maximum falls in the last bin). A
 * constant series gets one bin of width 1 around it. Empty for no finite values.
 */
export const histogram = (values: readonly number[], nbins: number): Histogram1D => {
  const v = finiteOnly(values);
  const n = v.length;
  if (n === 0 || nbins < 1) return { centers: [], width: 0, counts: [], density: [] };
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (v[i] < lo) lo = v[i];
    if (v[i] > hi) hi = v[i];
  }
  const width = hi > lo ? (hi - lo) / nbins : 1;
  const counts = new Array<number>(nbins).fill(0);
  for (let i = 0; i < n; i++) {
    counts[Math.min(nbins - 1, Math.floor((v[i] - lo) / width))]++;
  }
  return {
    centers: counts.map((_, b) => lo + (b + 0.5) * width),
    width,
    counts,
    density: counts.map((c) => c / (n * width)),
  };
};
