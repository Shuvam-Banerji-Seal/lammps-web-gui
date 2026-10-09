import { StyleError } from '../force/types';
import { parseNum, parseInt_ } from '../force/util';

/*
 * Descriptor so3 of pair_style mliap (wave 37). Doc (docs.lammps.org/pair_mliap.html, plans/lammps-docs/
 * pair_mliap.rst): "The descriptor style *so3* is a descriptor that is derived from the the smooth SO(3)
 * power spectrum with the explicit inclusion of a radial basis" and "The available models are *linear* and
 * *nn*." Descriptor file (same doc): "The SO3 descriptor file is similar to the SNAP descriptor except that it
 * contains a few more arguments (e.g., *nmax* and *alpha*)." The file keywords are not listed on that page;
 * this engine accepts rcutfac, nmax, lmax, alpha, nelems, elems, radelems and welems. Measured with native
 * LAMMPS (black box): rfac0, rmin0 and switchflag are refused (Incorrect SO3 parameter file), and so is a
 * file without welems.
 *
 * Published physics: Bartok, Kondor, Csanyi, PRB 87 184115 (2013), and Zagaceta, Yanxon, Zhu, arXiv:2005.04332
 * (the smooth SO(3) power spectrum, Eqs. 4-8, and the radial basis Eqs. 5-6: phi_k(r) = (rcut - r)^(k+2)/N_k,
 * g_n = sum_k W_nk phi_k, W = S^(-1/2), with the r^2 measure).
 *
 * Measured with native LAMMPS (black box), in this engine's reading:
 *  - the cutoff of every pair is rcutfac (not rcutfac*(R_i+R_j)): dimers with radelems 0.5/0.7 and 0.9/0.1
 *    give the same values, and rcutfac 5 vs 7 changes them;
 *  - each neighbour j is weighted by w_j = welems of its element (both factors of a product, so single-element
 *    descriptors scale as w^2);
 *  - the cutoff function is f_c(r) = 0.5(cos(pi r/rcut)+1) on the neighbour density (f_c^2 per product);
 *  - the density is sum_j w_j f_c(r_j) exp(-alpha |r - r_j|^2); the expansion of exp(2 alpha r.r_j) uses the
 *    spherical modified Bessel i_l (i_0(x) = sinh(x)/x), which is the correct kernel for the Gaussian density;
 *  - components are ordered with n2 outer, n1 <= n2 inner, l innermost (nmax 3, lmax 2 measured);
 *  - the constant is B_{n1 n2 l} = (4 pi)^2 sqrt((2l+1)/2) sum_{j,k} q_{n1 l}(r_j) q_{n2 l}(r_k) P_l(cos g_jk),
 *    q_{nl}(r) = w f_c(r) exp(-alpha r^2) J_{nl}(r), J_{nl}(r) = int_0^rc r'^2 g_n(r') exp(-alpha r'^2)
 *    i_l(2 alpha r r') dr' (matched to 1e-8 for nmax 1-3, lmax 0-2, 2-element clusters, rcutfac 5).
 *  - alpha, nmax, lmax of the native code are the same as the descriptor file's (no scaling of alpha).
 * nmax 0 (a native descriptor count of 0) is a StyleError here.
 */

export interface So3Descriptor {
  elems: string[];
  /** Element radii are read but do not enter the cutoff (measured, see the header). */
  radius: Float64Array;
  weight: Float64Array;
  rcutfac: number;
  nmax: number;
  lmax: number;
  alpha: number;
  /** Number of descriptor components: nmax (nmax+1)/2 (lmax+1). */
  K: number;
}

const dataLines = (text: string): string[][] =>
  text.split('\n').map((l) => l.replace(/#.*$/, '').trim()).filter((l) => l.length > 0).map((l) => l.split(/\s+/));

export const parseMliapSo3Descriptor = (text: string, filename: string): So3Descriptor => {
  const kv = new Map<string, string[]>();
  for (const w of dataLines(text)) {
    const [kw, ...vals] = w;
    if (vals.length === 0) throw new StyleError(`mliap so3 descriptor file ${filename}: keyword '${kw}' has no value`);
    kv.set(kw, vals);
  }
  const allowed = ['rcutfac', 'nmax', 'lmax', 'alpha', 'nelems', 'elems', 'radelems', 'welems'];
  for (const kw of kv.keys()) {
    if (!allowed.includes(kw)) throw new StyleError(`mliap so3 descriptor file ${filename}: keyword '${kw}' is not implemented in this engine`);
  }
  for (const kw of allowed) {
    if (!kv.has(kw)) throw new StyleError(`mliap so3 descriptor file ${filename}: keyword '${kw}' is required`);
  }
  const one = (kw: string): string => kv.get(kw)![0];
  const nelems = parseInt_(one('nelems'), `${filename} nelems`);
  if (nelems < 1) throw new StyleError(`mliap so3 descriptor file ${filename}: nelems must be at least 1`);
  const elems = kv.get('elems')!;
  const rad = kv.get('radelems')!.map((v) => parseNum(v, `${filename} radelems`));
  const wt = kv.get('welems')!.map((v) => parseNum(v, `${filename} welems`));
  if (elems.length !== nelems || rad.length !== nelems || wt.length !== nelems) {
    throw new StyleError(`mliap so3 descriptor file ${filename}: elems, radelems and welems must each list nelems = ${nelems} entries`);
  }
  const rcutfac = parseNum(one('rcutfac'), `${filename} rcutfac`);
  if (!(rcutfac > 0)) throw new StyleError(`mliap so3 descriptor file ${filename}: rcutfac must be positive`);
  const nmax = parseInt_(one('nmax'), `${filename} nmax`);
  if (nmax < 1) throw new StyleError(`mliap so3 descriptor file ${filename}: nmax must be at least 1 (nmax 0 is not implemented in this engine)`);
  const lmax = parseInt_(one('lmax'), `${filename} lmax`);
  if (lmax < 0) throw new StyleError(`mliap so3 descriptor file ${filename}: lmax must be a non-negative integer`);
  const alpha = parseNum(one('alpha'), `${filename} alpha`);
  if (!(alpha > 0)) throw new StyleError(`mliap so3 descriptor file ${filename}: alpha must be positive`);
  return {
    elems, radius: Float64Array.from(rad), weight: Float64Array.from(wt), rcutfac, nmax, lmax, alpha,
    K: (nmax * (nmax + 1)) / 2 * (lmax + 1),
  };
};

/** Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration on P_n). */
const gaussLegendre = (n: number): { x: Float64Array; w: Float64Array } => {
  const x = new Float64Array(n), w = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let z = Math.cos((Math.PI * (i + 0.75)) / (n + 0.5));
    let dp = 1;
    for (let it = 0; it < 100; it++) {
      let p0 = 1, p1 = z;
      for (let k = 2; k <= n; k++) {
        const p2 = ((2 * k - 1) * z * p1 - (k - 1) * p0) / k;
        p0 = p1; p1 = p2;
      }
      // p1 = P_n(z), p0 = P_{n-1}(z)
      dp = (n * (z * p1 - p0)) / (z * z - 1);
      const dz = p1 / dp;
      z -= dz;
      if (Math.abs(dz) < 1e-16) break;
    }
    x[i] = z;
    w[i] = 2 / ((1 - z * z) * dp * dp);
  }
  return { x, w };
};

/** Eigen-decomposition by cyclic Jacobi rotations; returns S^(-1/2) of a symmetric positive matrix. */
const symInvSqrt = (S: number[][]): number[][] => {
  const n = S.length;
  const a = S.map((r) => r.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) as number));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-300) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (a[p][q] === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const out = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let k = 0; k < n; k++) {
    const lam = a[k][k] ** -0.5;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i][j] += V[i][k] * lam * V[j][k];
  }
  return out;
};

/** Quadrature nodes on the radial integral [0, rc]. */
const QUAD = 100;

/**
 * Scaled spherical modified Bessel functions e^{-x} i_l(x), l = 0..L+1 (i_0(x) = sinh(x)/x,
 * i_l'(x) = i_{l+1}(x) + (l/x) i_l(x)). Series for small x, upward recurrence for x >= 2(L+2) (stable there).
 */
const scaledIn = (L: number, x: number, out: Float64Array): void => {
  const T = 2 * (L + 2);
  if (x < T) {
    const ex = Math.exp(-x);
    const x2 = x * x;
    for (let l = 0; l <= L + 1; l++) {
      // term_0 = x^l / (2l+1)!!
      let t = 1;
      for (let k = 0; k < l; k++) t *= x;
      for (let k = 1; k <= 2 * l + 1; k += 2) t /= k;
      let sum = t;
      for (let k = 0; k < 400; k++) {
        t *= x2 / (2 * (k + 1) * (2 * (k + 1) + 2 * l + 1));
        sum += t;
        if (t < 1e-18 * sum) break;
      }
      out[l] = ex * sum;
    }
    return;
  }
  const e2 = Math.exp(-2 * x);
  out[0] = (1 - e2) / (2 * x);
  out[1] = ((1 + e2) / 2) / x - ((1 - e2) / 2) / (x * x);
  for (let l = 1; l <= L; l++) out[l + 1] = out[l - 1] - ((2 * l + 1) / x) * out[l];
};

/** Legendre P_l(c) and dP_l/dc for l = 0..L. */
const legendre = (L: number, c: number, P: Float64Array, dP: Float64Array): void => {
  P[0] = 1; dP[0] = 0;
  if (L === 0) return;
  P[1] = c; dP[1] = 1;
  for (let l = 1; l < L; l++) {
    P[l + 1] = ((2 * l + 1) * c * P[l] - l * P[l - 1]) / (l + 1);
    dP[l + 1] = dP[l - 1] + (2 * l + 1) * P[l];
  }
};

export class So3Engine {
  readonly K: number;
  readonly nmax: number;
  readonly lmax: number;
  readonly alpha: number;
  readonly rc: number;
  /** Radial quadrature: A[n*QUAD+q] = omega_q r_q^2 g_n(r_q); rq[q] = r_q. */
  private A: Float64Array;
  private rq: Float64Array;
  /** Per-component prefactor (4 pi)^2 sqrt((2l+1)/2) indexed by component. */
  private S: Float64Array;
  private lOf: Int32Array;
  private n1Of: Int32Array;
  private n2Of: Int32Array;
  // scratch
  private ib = new Float64Array(64);
  private Kv: Float64Array;
  private Kd: Float64Array;
  private qv: Float64Array;
  private qd: Float64Array;
  private Pab: Float64Array;
  private dPab: Float64Array;

  constructor(d: So3Descriptor) {
    this.nmax = d.nmax; this.lmax = d.lmax; this.alpha = d.alpha; this.rc = d.rcutfac;
    this.K = d.K;
    const nmax = d.nmax, rc = d.rcutfac;
    // orthonormal radial basis (Zagaceta et al., Eqs. 5-6, with the r^2 measure): S_pq closed form
    const D = (p: number) => (2 * p + 5) * (2 * p + 6) * (2 * p + 7);
    const Smat: number[][] = [];
    for (let p = 1; p <= nmax; p++) {
      const row: number[] = [];
      for (let q = 1; q <= nmax; q++) {
        row.push(Math.sqrt(D(p) * D(q)) / ((5 + p + q) * (6 + p + q) * (7 + p + q)));
      }
      Smat.push(row);
    }
    const W = symInvSqrt(Smat);
    const Nk = (k: number) => Math.sqrt((2 * rc ** (2 * k + 7)) / ((2 * k + 5) * (2 * k + 6) * (2 * k + 7)));
    const gl = gaussLegendre(QUAD);
    this.rq = new Float64Array(QUAD);
    this.A = new Float64Array(nmax * QUAD);
    for (let q = 0; q < QUAD; q++) {
      const r = (rc * (gl.x[q] + 1)) / 2;
      const om = (rc * gl.w[q]) / 2;
      this.rq[q] = r;
      for (let n = 0; n < nmax; n++) {
        let g = 0;
        for (let k = 1; k <= nmax; k++) g += W[n][k - 1] * (rc - r) ** (k + 2) / Nk(k);
        this.A[n * QUAD + q] = om * r * r * g;
      }
    }
    // component layout: n2 outer, n1 <= n2, l innermost (measured with native LAMMPS, black box)
    this.lOf = new Int32Array(this.K); this.n1Of = new Int32Array(this.K); this.n2Of = new Int32Array(this.K);
    this.S = new Float64Array(this.K);
    let c = 0;
    for (let n2 = 0; n2 < nmax; n2++) {
      for (let n1 = 0; n1 <= n2; n1++) {
        for (let l = 0; l <= d.lmax; l++) {
          this.lOf[c] = l; this.n1Of[c] = n1; this.n2Of[c] = n2;
          this.S[c] = (4 * Math.PI) ** 2 * Math.sqrt((2 * l + 1) / 2);
          c++;
        }
      }
    }
    const NL = nmax * (d.lmax + 1);
    this.Kv = new Float64Array(NL); this.Kd = new Float64Array(NL);
    this.qv = new Float64Array(0); this.qd = new Float64Array(0);
    this.Pab = new Float64Array(0); this.dPab = new Float64Array(0);
  }

  /**
   * Descriptor values B (length K) of one atom from its neighbours: nb = 3m displacements d_j = x_j - x_i,
   * wts = m neighbour weights (welems of the neighbour's element). Neighbours must satisfy 0 < r < rcutfac.
   * If gam is given (dE_i/dB, length K), the derivative sum_c gam_c dB_c/dd_j is written to grad (3m,
   * overwritten): F_j = -grad_j and F_i = +sum_j grad_j.
   */
  evaluate(nb: Float64Array, wts: Float64Array, B: Float64Array, gam?: Float64Array, grad?: Float64Array): void {
    const m = wts.length;
    const nmax = this.nmax, L = this.lmax, NL = nmax * (L + 1), rc = this.rc, alpha = this.alpha;
    if (this.qv.length < m * NL) {
      this.qv = new Float64Array(m * NL); this.qd = new Float64Array(m * NL);
      this.Pab = new Float64Array(m * m * (L + 1)); this.dPab = new Float64Array(m * m * (L + 1));
    }
    const uh = new Float64Array(3 * m), rr = new Float64Array(m);
    const fcv = new Float64Array(m), dfcv = new Float64Array(m);
    for (let a = 0; a < m; a++) {
      const dx = nb[3 * a], dy = nb[3 * a + 1], dz = nb[3 * a + 2];
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
      rr[a] = r;
      uh[3 * a] = dx / r; uh[3 * a + 1] = dy / r; uh[3 * a + 2] = dz / r;
      const th = (Math.PI * r) / rc;
      fcv[a] = 0.5 * (Math.cos(th) + 1);
      dfcv[a] = -0.5 * Math.sin(th) * (Math.PI / rc);
      // radial integrals K_nl(r) = e^{-alpha r^2} J_nl(r) and dK/dr
      this.Kv.fill(0); this.Kd.fill(0);
      for (let q = 0; q < QUAD; q++) {
        const rq = this.rq[q];
        const dr = r - rq;
        const gauss = Math.exp(-alpha * dr * dr);
        const x = 2 * alpha * r * rq;
        scaledIn(L, x, this.ib);
        for (let n = 0; n < nmax; n++) {
          const An = this.A[n * QUAD + q] * gauss;
          if (An === 0) continue;
          for (let l = 0; l <= L; l++) {
            const il = this.ib[l];
            const dil = this.ib[l + 1] + (l / x) * il - il; // d/dx (e^{-x} i_l(x))
            const idx = n * (L + 1) + l;
            this.Kv[idx] += An * il;
            this.Kd[idx] += An * (-2 * alpha * dr * il + 2 * alpha * rq * dil);
          }
        }
      }
      for (let k = 0; k < NL; k++) {
        this.qv[a * NL + k] = wts[a] * fcv[a] * this.Kv[k];
        this.qd[a * NL + k] = wts[a] * (dfcv[a] * this.Kv[k] + fcv[a] * this.Kd[k]);
      }
    }
    // Legendre of the angles between neighbours
    const Pl = new Float64Array(L + 1), dPl = new Float64Array(L + 1);
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) {
        const c = uh[3 * a] * uh[3 * b] + uh[3 * a + 1] * uh[3 * b + 1] + uh[3 * a + 2] * uh[3 * b + 2];
        legendre(L, a === b ? 1 : c, Pl, dPl);
        for (let l = 0; l <= L; l++) {
          this.Pab[(a * m + b) * (L + 1) + l] = Pl[l];
          this.dPab[(a * m + b) * (L + 1) + l] = dPl[l];
        }
      }
    }
    for (let c = 0; c < this.K; c++) {
      const n1 = this.n1Of[c], n2 = this.n2Of[c], l = this.lOf[c];
      let s = 0;
      for (let a = 0; a < m; a++) {
        const q1 = this.qv[a * NL + n1 * (L + 1) + l];
        for (let b = 0; b < m; b++) {
          s += q1 * this.qv[b * NL + n2 * (L + 1) + l] * this.Pab[(a * m + b) * (L + 1) + l];
        }
      }
      B[c] = this.S[c] * s;
    }
    if (!gam || !grad) return;
    grad.fill(0);
    for (let c = 0; c < this.K; c++) {
      const g = gam[c];
      if (g === 0) continue;
      const n1 = this.n1Of[c], n2 = this.n2Of[c], l = this.lOf[c];
      const gc = g * this.S[c];
      const i1 = n1 * (L + 1) + l, i2 = n2 * (L + 1) + l;
      for (let a = 0; a < m; a++) {
        const ua = [uh[3 * a], uh[3 * a + 1], uh[3 * a + 2]];
        // diagonal a = b: P_l(1) = 1, no angular derivative
        const dq = this.qd[a * NL + i1] * this.qv[a * NL + i2] + this.qv[a * NL + i1] * this.qd[a * NL + i2];
        for (let t = 0; t < 3; t++) grad[3 * a + t] += gc * dq * ua[t];
        for (let b = 0; b < m; b++) {
          if (b === a) continue;
          const ub = [uh[3 * b], uh[3 * b + 1], uh[3 * b + 2]];
          const cab = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2];
          const p = this.Pab[(a * m + b) * (L + 1) + l];
          const dp = this.dPab[(a * m + b) * (L + 1) + l];
          const q1 = this.qv[a * NL + i1], q2 = this.qv[b * NL + i2];
          const dq1 = this.qd[a * NL + i1], dq2 = this.qd[b * NL + i2];
          const ra = rr[a], rb = rr[b];
          for (let t = 0; t < 3; t++) {
            // d/dd_a of q1(a) q2(b) P_l(c_ab)
            grad[3 * a + t] += gc * (dq1 * q2 * p * ua[t] + q1 * q2 * dp * (ub[t] - cab * ua[t]) / ra);
            // d/dd_b of q1(a) q2(b) P_l(c_ab)
            grad[3 * b + t] += gc * (q1 * dq2 * p * ub[t] + q1 * q2 * dp * (ua[t] - cab * ub[t]) / rb);
          }
        }
      }
    }
  }
}
