/*
 * PPPM's per-atom loops (force/kspace/pppm.ts): the assignment weights of each atom and the field interpolated back at
 * each charged atom. They are a module of their own so that the pair threads (cpu/pairThreads.ts, the worker's
 * cpu/pairThreadsCore.ts) run exactly the code the engine thread runs, on ranges of atoms: the results are the same
 * bit for bit whichever thread computes an atom.
 */

/**
 * The p weights M_p(x0 - k), k = 0 .. p - 1, into out[o + k]: the recursion of bspline() evaluated once for every
 * argument it reaches (t_m = x0 - m by successive subtraction, as its x - 1), level by level, instead of once per call.
 * a and b are scratch arrays of at least 2p entries.
 */
export const bsplineWeights = (p: number, x0: number, out: Float64Array, o: number, t: Float64Array, a: Float64Array, b: Float64Array): void => {
  const m1 = 2 * p - 1;
  t[0] = x0;
  for (let m = 1; m < m1; m++) t[m] = t[m - 1] - 1;
  for (let m = 0; m < m1; m++) a[m] = t[m] >= 0 && t[m] < 1 ? 1 : 0;
  let cur = a, nxt = b;
  for (let n = 2; n <= p; n++) {
    // level n needs the arguments t_0 .. t_(m1 - n)
    for (let m = 0; m <= m1 - n; m++) {
      const x = t[m];
      nxt[m] = x <= 0 || x >= n ? 0 : (x * cur[m] + (n - x) * cur[m + 1]) / (n - 1);
    }
    const sw = cur; cur = nxt; nxt = sw;
  }
  for (let k = 0; k < p; k++) out[o + k] = cur[k];
};

/** m mod n in [0, n). */
export const wrap = (m: number, n: number): number => ((m % n) + n) % n;

/**
 * What the per-atom loops need of the box and the mesh: the box origin and the upper-triangular inverse box matrix of
 * Geometry.toLamda, the slab factor (kspace_modify slab), the mesh and the order.
 */
export interface PppmGrid {
  lo0: number; lo1: number; lo2: number;
  i00: number; i01: number; i02: number; i11: number; i12: number; i22: number;
  slab: number;
  nx: number; ny: number; nz: number;
  order: number;
}

/**
 * PPPM's per-atom loops on the pair threads (cpu/pairThreads.ts): each returns null when the threads cannot take them
 * (not started yet, not cross-origin isolated, ...) and the engine thread runs the loop itself. The weights come back
 * in shared arrays that pppmField reads again; pppmMeshes gives the shared field meshes pppmField interpolates.
 */
export interface PppmThreads {
  readonly minAtoms: number;
  pppmWeights(g: PppmGrid, x: Float64Array, n: number): { first: Int32Array; wts: Float64Array } | null;
  pppmMeshes(len: number): [Float64Array, Float64Array, Float64Array] | null;
  pppmField(g: PppmGrid, n: number, q: ArrayLike<number>, first: Int32Array, wts: Float64Array,
    ex: Float64Array, ey: Float64Array, ez: Float64Array, C: number, V: number, zForce: boolean): Float64Array | null;
  /**
   * The charge mesh (len values, re and im interleaved) spread from the weights of pppmWeights, in a shared array that
   * pppmFft transforms; null when the threads cannot (the engine thread then spreads it).
   */
  pppmSpread?(g: PppmGrid, n: number, q: ArrayLike<number>, first: Int32Array, wts: Float64Array, len: number): Float64Array | null;
  /** fft3d of each mesh (the threads' shared ones: pppmSpread's and pppmMeshes') on the threads; false when it cannot. */
  pppmFft?(g: PppmGrid, meshes: Float64Array[], sign: -1 | 1): boolean;
}

/**
 * The parameters of one threaded loop in a shared array (cpu/pairThreads.ts writes them, the workers read them):
 * the grid, the atom count, the chunk count, C, V, zForce, and the count of the items the chunks split (the atoms; the
 * mesh z planes of the spread; the lines of an FFT pass). Words of the array.
 */
export const PPPM_PARAMS = 20;

export const writePppmParams = (a: Float64Array, g: PppmGrid, n: number, nchunks: number, C: number, V: number, zForce: boolean, items = n): void => {
  a[0] = g.lo0; a[1] = g.lo1; a[2] = g.lo2; a[3] = g.i00; a[4] = g.i01; a[5] = g.i02; a[6] = g.i11; a[7] = g.i12; a[8] = g.i22;
  a[9] = g.slab; a[10] = g.nx; a[11] = g.ny; a[12] = g.nz; a[13] = g.order; a[14] = n; a[15] = nchunks; a[16] = C; a[17] = V;
  a[18] = zForce ? 1 : 0; a[19] = items;
};

export const readPppmParams = (a: Float64Array): { g: PppmGrid; n: number; nchunks: number; C: number; V: number; zForce: boolean; items: number } => ({
  g: {
    lo0: a[0], lo1: a[1], lo2: a[2], i00: a[3], i01: a[4], i02: a[5], i11: a[6], i12: a[7], i22: a[8], slab: a[9],
    nx: a[10], ny: a[11], nz: a[12], order: a[13],
  },
  n: a[14], nchunks: a[15], C: a[16], V: a[17], zForce: a[18] !== 0, items: a[19],
});

/** The scratch arrays of one thread for pppmWeights. */
export const weightScratch = (P: number) => ({
  t: new Float64Array(2 * P), sa: new Float64Array(2 * P), sb: new Float64Array(2 * P),
});

/**
 * Atoms [i0, i1): the first mesh index in x, y and z (first[3i + d]) and the P weights of each direction
 * (wts[(3i + d) P + k]), from the fractional coordinates (Geometry.toLamda, z divided by the slab factor).
 */
export const pppmWeights = (
  g: PppmGrid, x: ArrayLike<number>, i0: number, i1: number, first: Int32Array, wts: Float64Array,
  sc: { t: Float64Array; sa: Float64Array; sb: Float64Array },
): void => {
  const P = g.order;
  const odd = P % 2 === 1;
  for (let i = i0; i < i1; i++) {
    const dx = x[3 * i] - g.lo0, dy = x[3 * i + 1] - g.lo1, dz = x[3 * i + 2] - g.lo2;
    const l0 = g.i00 * dx + g.i01 * dy + g.i02 * dz;
    const l1 = g.i11 * dy + g.i12 * dz;
    const l2 = (g.i22 * dz) / g.slab;
    for (let d = 0; d < 3; d++) {
      const u = (d === 0 ? l0 : d === 1 ? l1 : l2) * (d === 0 ? g.nx : d === 1 ? g.ny : g.nz);
      const f = odd ? Math.round(u) - (P - 1) / 2 : Math.floor(u) - P / 2 + 1;
      first[3 * i + d] = f;
      // the weights M_P(u - (f + k) + P/2) of the P mesh points from f
      bsplineWeights(P, u - f + P / 2, wts, (3 * i + d) * P, sc.t, sc.sa, sc.sb);
    }
  }
};

/**
 * Atoms [i0, i1): the force pre * E on each charged atom, E the field meshes ex, ey, ez (real parts at even indices)
 * interpolated with the atom's weights, pre = C q / V; out[3i .. 3i + 2] (z 0 without zForce). Uncharged atoms are left
 * as they are. xs is a scratch array of 3P entries.
 */
export const pppmField = (
  g: PppmGrid, i0: number, i1: number, q: ArrayLike<number>, first: Int32Array, wts: Float64Array,
  ex: Float64Array, ey: Float64Array, ez: Float64Array, C: number, V: number, zForce: boolean, out: Float64Array, xs: Int32Array,
): void => {
  const P = g.order, nx = g.nx, ny = g.ny, nz = g.nz;
  for (let i = i0; i < i1; i++) {
    const qi = q[i];
    if (qi === 0) continue;
    const fx = first[3 * i], fy = first[3 * i + 1], fz = first[3 * i + 2];
    for (let k = 0; k < P; k++) {
      xs[k] = wrap(fx + k, nx);
      xs[P + k] = wrap(fy + k, ny);
      xs[2 * P + k] = wrap(fz + k, nz);
    }
    let Ex = 0, Ey = 0, Ez = 0;
    for (let c = 0; c < P; c++) {
      const iz = xs[2 * P + c];
      const wz = wts[(3 * i + 2) * P + c];
      for (let b = 0; b < P; b++) {
        const iy = xs[P + b];
        const wyz = wz * wts[(3 * i + 1) * P + b];
        const row = (iz * ny + iy) * nx;
        for (let a = 0; a < P; a++) {
          const ww = wyz * wts[(3 * i) * P + a];
          const idx = 2 * (row + xs[a]);
          Ex += ww * ex[idx]; Ey += ww * ey[idx]; Ez += ww * ez[idx];
        }
      }
    }
    const pre = (C * qi) / V;
    out[3 * i] = pre * Ex; out[3 * i + 1] = pre * Ey;
    out[3 * i + 2] = zForce ? pre * Ez : 0;
  }
};

/**
 * The mesh z planes [z0, z1) of the charge mesh rho (re at even indices): every charged atom in index order adds
 * q w_z w_y w_x to the points of its stencil in those planes, with the arithmetic of KSpacePPPM.spread. Each mesh point
 * then receives the same additions in the same order as the engine thread's loop over all atoms gives it, so planes
 * spread by different threads add up to the same mesh bit for bit. xs is a scratch array of 3P entries.
 */
/** wrap(m, n) without the remainders when m is already in [0, n) (the usual case); the same integer. */
const wrapFast = (m: number, n: number): number => (m >= 0 && m < n ? m : wrap(m, n));

export const pppmSpread = (
  g: PppmGrid, n: number, q: ArrayLike<number>, first: Int32Array, wts: Float64Array, rho: Float64Array, z0: number, z1: number, xs: Int32Array,
): void => {
  const P = g.order, nx = g.nx, ny = g.ny, nz = g.nz, w = z1 - z0;
  for (let i = 0; i < n; i++) {
    const qi = q[i];
    if (qi === 0) continue;
    const fz = first[3 * i + 2];
    // the stencil's planes wrap(fz + c), c < P: skip the atom unless one of them is in [z0, z1)
    const rel = wrapFast(fz - z0, nz);
    if (rel >= w && rel + P - 1 < nz && P <= nz) continue;
    const fx = first[3 * i], fy = first[3 * i + 1];
    for (let k = 0; k < P; k++) {
      xs[k] = wrapFast(fx + k, nx);
      xs[P + k] = wrapFast(fy + k, ny);
    }
    for (let c = 0; c < P; c++) {
      const iz = wrapFast(fz + c, nz);
      if (iz < z0 || iz >= z1) continue;
      const wz = qi * wts[(3 * i + 2) * P + c];
      for (let b = 0; b < P; b++) {
        const iy = xs[P + b];
        const wyz = wz * wts[(3 * i + 1) * P + b];
        const row = (iz * ny + iy) * nx;
        for (let a = 0; a < P; a++) {
          rho[2 * (row + xs[a])] += wyz * wts[(3 * i) * P + a];
        }
      }
    }
  }
};
