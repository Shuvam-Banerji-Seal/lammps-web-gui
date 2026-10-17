/*
 * Complex FFTs for the PPPM solver: mixed radix 2, 3, 5 (kspace_modify.html:
 * the PPPM mesh "must be factorizable into powers of 2, 3, and 5"), written
 * as the textbook Stockham autosort (iterative, self-sorting) with radix-4,
 * 2, 3 and 5 butterflies and per-size buffers, so a transform allocates nothing. Data are
 * interleaved (re, im) Float64Arrays. Forward transform uses exp(-2 pi i jk/n);
 * the inverse uses exp(+2 pi i jk/n) and is NOT normalized.
 */
import { libm } from '../libm';

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
  /** radices of the stages, in order (4s first, then 2, 3, 5) */
  radices: number[];
  /** cos and sin of 2 pi k / n, k = 0..n-1 */
  cos: Float64Array;
  sin: Float64Array;
  /** ping-pong buffers (interleaved re, im) */
  a: Float64Array;
  b: Float64Array;
  /** cos and sin of 2 pi / 3, 2 pi / 5 and 4 pi / 5 (the small butterflies) */
  c3: number; s3: number;
  c51: number; s51: number; c52: number; s52: number;
}

const plans = new Map<number, Plan>();

const plan = (n: number): Plan => {
  let p = plans.get(n);
  if (p) return p;
  if (!isFFTSize(n)) throw new Error(`FFT size ${n} must factor into 2, 3 and 5`);
  const radices: number[] = [];
  let m = n;
  while (m % 4 === 0) { radices.push(4); m /= 4; }
  while (m % 2 === 0) { radices.push(2); m /= 2; }
  while (m % 3 === 0) { radices.push(3); m /= 3; }
  while (m % 5 === 0) { radices.push(5); m /= 5; }
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    cos[k] = libm.cos((2 * Math.PI * k) / n);
    sin[k] = libm.sin((2 * Math.PI * k) / n);
  }
  p = {
    n, radices, cos, sin, a: new Float64Array(2 * n), b: new Float64Array(2 * n),
    c3: libm.cos((2 * Math.PI) / 3), s3: libm.sin((2 * Math.PI) / 3),
    c51: libm.cos((2 * Math.PI) / 5), s51: libm.sin((2 * Math.PI) / 5),
    c52: libm.cos((4 * Math.PI) / 5), s52: libm.sin((4 * Math.PI) / 5),
  };
  plans.set(n, p);
  return p;
};

/**
 * One Stockham autosort stage of radix R (textbook self-sorting decimation in time, no bit reversal): for each of
 * the n/R butterflies j, the inputs src[j + r n/R] times the twiddles w^(r k), k = j mod ns, go through a radix-R DFT
 * and land at dst[(j - k) R + k + r ns]; ns is the size of the transforms combined so far. sign = -1 forward.
 */
const stage = (p: Plan, src: Float64Array, dst: Float64Array, R: number, ns: number, sign: number): void => {
  const n = p.n, cos = p.cos, sin = p.sin;
  const stride = n / R;          // input distance between the R points of a butterfly
  const tw = n / (ns * R);       // twiddle table step: w^(r k) = exp(sign 2 pi i r k / (ns R))
  for (let j = 0; j < stride; j++) {
    const k = j % ns;
    const out = (j - k) * R + k;
    if (R === 4) {
      let x0r = src[2 * j], x0i = src[2 * j + 1];
      let x1r = src[2 * (j + stride)], x1i = src[2 * (j + stride) + 1];
      let x2r = src[2 * (j + 2 * stride)], x2i = src[2 * (j + 2 * stride) + 1];
      let x3r = src[2 * (j + 3 * stride)], x3i = src[2 * (j + 3 * stride) + 1];
      if (k !== 0) {
        let t = k * tw, c = cos[t], s = sign * sin[t];
        let re = x1r * c - x1i * s; x1i = x1r * s + x1i * c; x1r = re;
        t = 2 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x2r * c - x2i * s; x2i = x2r * s + x2i * c; x2r = re;
        t = 3 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x3r * c - x3i * s; x3i = x3r * s + x3i * c; x3r = re;
      }
      const t0r = x0r + x2r, t0i = x0i + x2i, t1r = x0r - x2r, t1i = x0i - x2i;
      const t2r = x1r + x3r, t2i = x1i + x3i;
      // (x1 - x3) times sign i
      const t3r = -sign * (x1i - x3i), t3i = sign * (x1r - x3r);
      dst[2 * out] = t0r + t2r; dst[2 * out + 1] = t0i + t2i;
      dst[2 * (out + ns)] = t1r + t3r; dst[2 * (out + ns) + 1] = t1i + t3i;
      dst[2 * (out + 2 * ns)] = t0r - t2r; dst[2 * (out + 2 * ns) + 1] = t0i - t2i;
      dst[2 * (out + 3 * ns)] = t1r - t3r; dst[2 * (out + 3 * ns) + 1] = t1i - t3i;
    } else if (R === 2) {
      const x0r = src[2 * j], x0i = src[2 * j + 1];
      let x1r = src[2 * (j + stride)], x1i = src[2 * (j + stride) + 1];
      if (k !== 0) {
        const t = k * tw, c = cos[t], s = sign * sin[t];
        const re = x1r * c - x1i * s; x1i = x1r * s + x1i * c; x1r = re;
      }
      dst[2 * out] = x0r + x1r; dst[2 * out + 1] = x0i + x1i;
      dst[2 * (out + ns)] = x0r - x1r; dst[2 * (out + ns) + 1] = x0i - x1i;
    } else if (R === 3) {
      const x0r = src[2 * j], x0i = src[2 * j + 1];
      let x1r = src[2 * (j + stride)], x1i = src[2 * (j + stride) + 1];
      let x2r = src[2 * (j + 2 * stride)], x2i = src[2 * (j + 2 * stride) + 1];
      if (k !== 0) {
        let t = k * tw, c = cos[t], s = sign * sin[t];
        let re = x1r * c - x1i * s; x1i = x1r * s + x1i * c; x1r = re;
        t = 2 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x2r * c - x2i * s; x2i = x2r * s + x2i * c; x2r = re;
      }
      const ar = x1r + x2r, ai = x1i + x2i, br = x1r - x2r, bi = x1i - x2i;
      const mr = x0r + p.c3 * ar, mi = x0i + p.c3 * ai;
      // (x1 - x2) times sign i sin(2 pi / 3)
      const s3 = sign * p.s3;
      const nr = -s3 * bi, ni = s3 * br;
      dst[2 * out] = x0r + ar; dst[2 * out + 1] = x0i + ai;
      dst[2 * (out + ns)] = mr + nr; dst[2 * (out + ns) + 1] = mi + ni;
      dst[2 * (out + 2 * ns)] = mr - nr; dst[2 * (out + 2 * ns) + 1] = mi - ni;
    } else {
      // R === 5
      const x0r = src[2 * j], x0i = src[2 * j + 1];
      let x1r = src[2 * (j + stride)], x1i = src[2 * (j + stride) + 1];
      let x2r = src[2 * (j + 2 * stride)], x2i = src[2 * (j + 2 * stride) + 1];
      let x3r = src[2 * (j + 3 * stride)], x3i = src[2 * (j + 3 * stride) + 1];
      let x4r = src[2 * (j + 4 * stride)], x4i = src[2 * (j + 4 * stride) + 1];
      if (k !== 0) {
        let t = k * tw, c = cos[t], s = sign * sin[t];
        let re = x1r * c - x1i * s; x1i = x1r * s + x1i * c; x1r = re;
        t = 2 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x2r * c - x2i * s; x2i = x2r * s + x2i * c; x2r = re;
        t = 3 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x3r * c - x3i * s; x3i = x3r * s + x3i * c; x3r = re;
        t = 4 * k * tw; c = cos[t]; s = sign * sin[t];
        re = x4r * c - x4i * s; x4i = x4r * s + x4i * c; x4r = re;
      }
      const a1r = x1r + x4r, a1i = x1i + x4i, b1r = x1r - x4r, b1i = x1i - x4i;
      const a2r = x2r + x3r, a2i = x2i + x3i, b2r = x2r - x3r, b2i = x2i - x3i;
      const c1 = p.c51, c2 = p.c52, s1 = sign * p.s51, s2 = sign * p.s52;
      const m1r = x0r + c1 * a1r + c2 * a2r, m1i = x0i + c1 * a1i + c2 * a2i;
      const m2r = x0r + c2 * a1r + c1 * a2r, m2i = x0i + c2 * a1i + c1 * a2i;
      // i (s1 b1 + s2 b2) and i (s2 b1 - s1 b2)
      const u1r = -(s1 * b1i + s2 * b2i), u1i = s1 * b1r + s2 * b2r;
      const u2r = -(s2 * b1i - s1 * b2i), u2i = s2 * b1r - s1 * b2r;
      dst[2 * out] = x0r + a1r + a2r; dst[2 * out + 1] = x0i + a1i + a2i;
      dst[2 * (out + ns)] = m1r + u1r; dst[2 * (out + ns) + 1] = m1i + u1i;
      dst[2 * (out + 2 * ns)] = m2r + u2r; dst[2 * (out + 2 * ns) + 1] = m2i + u2i;
      dst[2 * (out + 3 * ns)] = m2r - u2r; dst[2 * (out + 3 * ns) + 1] = m2i - u2i;
      dst[2 * (out + 4 * ns)] = m1r - u1r; dst[2 * (out + 4 * ns) + 1] = m1i - u1i;
    }
  }
};

/**
 * In-place 1d FFT of `count` interleaved complex values starting at `off`
 * with element stride `stride` (in complex units). sign = -1 forward,
 * +1 inverse (unnormalized).
 */
export const fft1d = (data: Float64Array, off: number, stride: number, n: number, sign: -1 | 1): void => {
  if (n === 1) return;
  const p = plan(n);
  let src = p.a, dst = p.b;
  for (let k = 0; k < n; k++) {
    const s = 2 * (off + k * stride);
    src[2 * k] = data[s];
    src[2 * k + 1] = data[s + 1];
  }
  let ns = 1;
  for (const R of p.radices) {
    stage(p, src, dst, R, ns, sign);
    ns *= R;
    const t = src; src = dst; dst = t;
  }
  for (let k = 0; k < n; k++) {
    const s = 2 * (off + k * stride);
    data[s] = src[2 * k];
    data[s + 1] = src[2 * k + 1];
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
