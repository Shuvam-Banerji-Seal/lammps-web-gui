/*
 * Complex FFTs for the PPPM solver: mixed radix 2, 3, 5 (kspace_modify.html:
 * the PPPM mesh "must be factorizable into powers of 2, 3, and 5"), written
 * from the textbook Cooley-Tukey decimation-in-time recursion. Data are
 * interleaved (re, im) Float64Arrays. Forward transform uses exp(-2 pi i jk/n);
 * the inverse uses exp(+2 pi i jk/n) and is NOT normalized.
 */

export const isFFTSize = (n: number): boolean => {
  if (n < 1 || !Number.isInteger(n)) return false;
  for (const p of [2, 3, 5]) while (n % p === 0) n /= p;
  return n === 1;
};

/** Smallest size >= n that factors into 2, 3 and 5. */
export const nextFFTSize = (n: number): number => {
  let m = Math.max(1, Math.ceil(n));
  while (!isFFTSize(m)) m++;
  return m;
};

interface Plan {
  n: number;
  factors: number[];
  /** twiddles exp(-2 pi i k/n), k = 0..n-1 */
  cos: Float64Array;
  sin: Float64Array;
  scratch: Float64Array;
}

const plans = new Map<number, Plan>();

const plan = (n: number): Plan => {
  let p = plans.get(n);
  if (p) return p;
  if (!isFFTSize(n)) throw new Error(`FFT size ${n} must factor into 2, 3 and 5`);
  const factors: number[] = [];
  let m = n;
  for (const f of [5, 3, 2]) while (m % f === 0) { factors.push(f); m /= f; }
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    cos[k] = Math.cos((2 * Math.PI * k) / n);
    sin[k] = -Math.sin((2 * Math.PI * k) / n);
  }
  p = { n, factors, cos, sin, scratch: new Float64Array(2 * n) };
  plans.set(n, p);
  return p;
};

/**
 * In-place 1d FFT of `count` interleaved complex values starting at `off`
 * with element stride `stride` (in complex units). sign = -1 forward,
 * +1 inverse (unnormalized).
 */
export const fft1d = (data: Float64Array, off: number, stride: number, n: number, sign: -1 | 1): void => {
  if (n === 1) return;
  const p = plan(n);
  const buf = p.scratch;
  // gather
  for (let k = 0; k < n; k++) {
    const s = 2 * (off + k * stride);
    buf[2 * k] = data[s];
    buf[2 * k + 1] = data[s + 1];
  }
  const out = new Float64Array(2 * n);
  rec(buf, 0, 1, out, 0, n, p, 0, sign);
  for (let k = 0; k < n; k++) {
    const s = 2 * (off + k * stride);
    data[s] = out[2 * k];
    data[s + 1] = out[2 * k + 1];
  }
};

/**
 * Recursive mixed-radix DIT: transforms in[inOff + j*inStride], j < len, into
 * out[outOff .. outOff+len). `level` indexes p.factors.
 */
const rec = (
  inp: Float64Array, inOff: number, inStride: number, out: Float64Array, outOff: number,
  len: number, p: Plan, level: number, sign: number,
): void => {
  if (len === 1) {
    out[2 * outOff] = inp[2 * inOff];
    out[2 * outOff + 1] = inp[2 * inOff + 1];
    return;
  }
  const r = p.factors[level];
  const m = len / r;
  // sub-transforms of the r decimated sequences
  for (let q = 0; q < r; q++) rec(inp, inOff + q * inStride, inStride * r, out, outOff + q * m, m, p, level + 1, sign);
  // butterflies: X[k + s*m] = sum_q W_len^{q(k + s m)} Y_q[k]
  const step = p.n / len;   // twiddle index stride for this level
  const tmpRe = new Float64Array(r), tmpIm = new Float64Array(r);
  for (let k = 0; k < m; k++) {
    for (let q = 0; q < r; q++) {
      const idx = outOff + q * m + k;
      let re = out[2 * idx], im = out[2 * idx + 1];
      if (q > 0) {
        const t = ((q * k * step) % p.n);
        const c = p.cos[t], sn = sign < 0 ? p.sin[t] : -p.sin[t];
        const nr = re * c - im * sn;
        im = re * sn + im * c;
        re = nr;
      }
      tmpRe[q] = re; tmpIm[q] = im;
    }
    for (let s = 0; s < r; s++) {
      let accRe = 0, accIm = 0;
      for (let q = 0; q < r; q++) {
        const t = ((q * s * m * step) % p.n);
        const c = p.cos[t], sn = sign < 0 ? p.sin[t] : -p.sin[t];
        accRe += tmpRe[q] * c - tmpIm[q] * sn;
        accIm += tmpRe[q] * sn + tmpIm[q] * c;
      }
      const o = outOff + s * m + k;
      out[2 * o] = accRe;
      out[2 * o + 1] = accIm;
    }
  }
};

/**
 * In-place 3d FFT of an (nx, ny, nz) complex grid stored x-fastest:
 * index = (iz * ny + iy) * nx + ix.
 */
export const fft3d = (data: Float64Array, nx: number, ny: number, nz: number, sign: -1 | 1): void => {
  for (let iz = 0; iz < nz; iz++) for (let iy = 0; iy < ny; iy++) fft1d(data, (iz * ny + iy) * nx, 1, nx, sign);
  for (let iz = 0; iz < nz; iz++) for (let ix = 0; ix < nx; ix++) fft1d(data, iz * ny * nx + ix, nx, ny, sign);
  for (let iy = 0; iy < ny; iy++) for (let ix = 0; ix < nx; ix++) fft1d(data, iy * nx + ix, nx * ny, nz, sign);
};
