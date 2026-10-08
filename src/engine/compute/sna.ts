import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNum, parseInt_ } from '../force/util';

/*
 * compute ID group-ID sna/atom rcutfac rfac0 twojmax R_1 ... R_N w_1 ... w_N keyword values ...
 * — docs.lammps.org/compute_sna_atom.html
 *
 * (source: plans/lammps-docs/compute_sna_atom.rst)
 *
 * "Define a computation that calculates a set of quantities related to the
 * bispectrum components of the atoms in a group."
 *
 * The neighbor density is mapped onto the 3-sphere: "The radial distance *r*
 * within *R_ii'* is mapped on to a third polar angle :math:`\theta_0` defined
 * by," with the formula
 *
 *   \theta_0 = \mathsf{rfac0} \frac{r-r_{min0}}{R_{ii'}-r_{min0}} \pi
 *
 * and "The position of a neighbor atom *i'* relative to a central atom *i* is a
 * point within the 3D ball of radius :math:`R_{ii'}` = *rcutfac*
 * :math:`(R_i + R_i')`". The switching function is
 *
 *   f_c(r)   = & \frac{1}{2}(\cos(\pi \frac{r-r_{min0}}{R_{ii'}-r_{min0}}) + 1), r \leq R_{ii'} \\
 *            = & 0,  r > R_{ii'}
 *
 * with the *switchflag* keyword: "The keyword *switchflag* can be used to turn
 * off the switching function :math:`f_c(r)`."
 *
 * Expansion coefficients "u^j_{m,m'} = U^j_{m,m'}(0,0,0) + \sum_{r_{ii'} < R_{ii'}}{f_c(r_{ii'}) w_{\mu_{i'}} U^j_{m,m'}(\theta_0,\theta,\phi)}"
 * where U^j is the SU(2) representation (Wigner D-matrix) on the 3-sphere.
 * Here U^j is built from the Cayley-Klein parameters of the Hopf map of the
 * point (theta0, theta, phi): a = (z0 + i z)/r0, b = (y + i x)/r0 with
 * z0 = r cot(theta0) and r0 = r / sin(theta0) (so |a|^2 + |b|^2 = 1), and the
 * spin-j representation in the orthonormal monomial basis x^k y^(2j-k) /
 * sqrt(k! (2j-k)!) of the substitution (x, y) -> (a x - conj(b) y, b x + conj(a) y).
 * Unitarity and the identity at (a, b) = (1, 0) are the textbook properties of
 * that representation (Varshalovich); the central atom contributes the identity.
 *
 * The bispectrum components B_{j_1,j_2,j} are the invariant triple products
 * of the expansion coefficients (doc: "scalar triple products of expansion
 * coefficients"), with H the coupling coefficients. Here H is the product of two orthonormal
 * Clebsch-Gordan coefficients (textbook, Racah formula), so
 * B = sum_{m,m'} conj(u^j_{m,m'}) sum CG(j1 m1 j2 m2|j m) CG(j1 m1' j2 m2'|j m') u^{j1}_{m1 m1'} u^{j2}_{m2 m2'}.
 * Measured with native LAMMPS (black box): this unscaled value reproduces the
 * columns for 2 atom types, rmin0 0.3, rfac0 0.8, weights 0.7 and 1.3, radii
 * 1.6 and 2.1, twojmax 3..6 and the bnormflag division by 2j+1 (the third index).
 *
 * "The keyword *bzeroflag* determines whether or not *B0*, the bispectrum
 * components of an atom with no neighbors, are subtracted from the
 * calculated bispectrum components." B0 is the value for the self term alone
 * (identity u matrices).
 *
 * "The keyword *bnormflag* determines whether or not the bispectrum
 * component :math:`B_{j_1,j_2,j}` is divided by a factor of :math:`2j+1`."
 *
 * "The keyword *quadraticflag* determines whether or not the quadratic
 * combinations of bispectrum quantities are generated. These are formed by
 * taking the outer product of the vector of bispectrum components with
 * itself." Measured with native LAMMPS (black box): the upper triangle of
 * B_i B_j with the diagonal entries halved (B_i B_i / 2).
 *
 * Column order (docs, Output info): "for j1 in range(0,twojmax+1): for j2 in
 * range(0,j1+1): for j in range(j1-j2,min(twojmax,j1+j2)+1,2): if (j>=j1):
 * print j1/2.,j2/2.,j/2." The number of columns is "K = m(m+1)(2m+1)/6"
 * for even twojmax, "K = m(m+1)(m+2)/3" for odd twojmax, with m = floor(twojmax/2)+1.
 *
 * Default: "The optional keyword defaults are *rmin0* = 0, *switchflag* = 1,
 * *bzeroflag* = 1, *quadraticflag* = 0, *bnormflag* = 0, *wselfallflag* = 0,
 * *switchinnerflag* = 0, *nnn* = -1, *wmode* = 0, *delta* = 1.e-3"
 *
 * Unsupported (StyleError): chem, switchinnerflag/sinner/dinner, nnn/wmode/delta,
 * and the compute snad/atom, snav/atom, snap, sna/grid families.
 */

/** Factorial of a non-negative integer (exact in doubles for the sizes used here). */
const factorial = (n: number): number => {
  let f = 1;
  for (let i = 2; i <= n; i++) f *= i;
  return f;
};

/** Factorial of a half-integer argument given doubled: f2(x2) = (x2/2)!. */
const f2 = (x2: number): number => factorial(x2 / 2);

/**
 * Clebsch-Gordan coefficient <j1 m1 j2 m2 | j m> (Racah formula, textbook).
 * All angular momenta and projections are passed doubled (J1 = 2 j1, M1 = 2 m1, ...).
 */
export const clebsch = (J1: number, M1: number, J2: number, M2: number, J: number, M: number): number => {
  if (M1 + M2 !== M) return 0;
  if (Math.abs(M1) > J1 || Math.abs(M2) > J2 || Math.abs(M) > J) return 0;
  if ((J1 + J2 + J) % 2 !== 0) return 0;
  if (J < Math.abs(J1 - J2) || J > J1 + J2) return 0;
  const pre = Math.sqrt(
    (J + 1) * f2(J1 + J2 - J) * f2(J1 - J2 + J) * f2(-J1 + J2 + J) / f2(J1 + J2 + J + 2),
  );
  const pre2 = Math.sqrt(
    f2(J1 + M1) * f2(J1 - M1) * f2(J2 + M2) * f2(J2 - M2) * f2(J + M) * f2(J - M),
  );
  const a0 = (J1 + J2 - J) / 2, b0 = (J1 - M1) / 2, c0 = (J2 + M2) / 2;
  const d0 = (J - J2 + M1) / 2, e0 = (J - J1 - M2) / 2;
  let sum = 0;
  for (let k = 0; k <= a0 && k <= b0 && k <= c0; k++) {
    const d = d0 + k, e = e0 + k;
    if (d < 0 || e < 0) continue;
    const denom = factorial(k) * factorial(a0 - k) * factorial(b0 - k) * factorial(c0 - k) * factorial(d) * factorial(e);
    sum += (k & 1 ? -1 : 1) / denom;
  }
  return pre * pre2 * sum;
};

/**
 * Wigner U^J (J = 2j) for the Cayley-Klein parameters (a, b), |a|^2 + |b|^2 = 1,
 * in the orthonormal monomial basis. Substituting (x, y) -> (a x - conj(b) y,
 * b x + conj(a) y) in x^k y^(J-k) / sqrt(k!(J-k)!) gives the matrix. Written
 * flat: row kp (power of x in the output) times J+1 plus column k.
 */
export const wignerU = (J: number, ar: number, ai: number, br: number, bi: number): { re: Float64Array; im: Float64Array } => {
  const n = J + 1;
  const re = new Float64Array(n * n);
  const im = new Float64Array(n * n);
  // linear forms as polynomials in x: index 0 = coefficient of y, index 1 = coefficient of x
  const L1 = [[-br, bi], [ar, ai]]; // a x - conj(b) y  (conj(b) = br - i bi, so -conj(b) = -br + i bi)
  const L2 = [[ar, -ai], [br, bi]]; // b x + conj(a) y
  // the y-coefficient of L1 is -conj(b) = (-br, +bi) and of L2 is conj(a) = (ar, -ai)
  const mul = (p: number[][], q: number[][]): number[][] => {
    const r: number[][] = Array.from({ length: p.length + q.length - 1 }, () => [0, 0]);
    for (let i = 0; i < p.length; i++) {
      for (let j = 0; j < q.length; j++) {
        r[i + j][0] += p[i][0] * q[j][0] - p[i][1] * q[j][1];
        r[i + j][1] += p[i][0] * q[j][1] + p[i][1] * q[j][0];
      }
    }
    return r;
  };
  for (let k = 0; k <= J; k++) {
    let p: number[][] = [[1, 0]];
    for (let t = 0; t < k; t++) p = mul(p, L1);
    for (let t = 0; t < J - k; t++) p = mul(p, L2);
    for (let kp = 0; kp <= J; kp++) {
      const c = p[kp] ?? [0, 0];
      const s = Math.sqrt(factorial(kp) * factorial(J - kp)) / Math.sqrt(factorial(k) * factorial(J - k));
      re[kp * n + k] = c[0] * s;
      im[kp * n + k] = c[1] * s;
    }
  }
  return { re, im };
};

/** One bispectrum component: doubled indices and its coupling table. */
interface Triple {
  J1: number;
  J2: number;
  J: number;
  /** T[k*(J1+1)+m1] = CG(j1 m1, j2 m2 | j m) with m = k - j; m2 = pick[...]. */
  coef: Float64Array;
  /** Index m2 for the same flat slot (or -1 when no CG term). */
  m2: Int32Array;
}

/** Bispectrum component list in the documented order, with coupling tables. */
const buildTriples = (twojmax: number): Triple[] => {
  const out: Triple[] = [];
  for (let J1 = 0; J1 <= twojmax; J1++) {
    for (let J2 = 0; J2 <= J1; J2++) {
      for (let J = Math.abs(J1 - J2); J <= Math.min(twojmax, J1 + J2); J += 2) {
        if (J < J1) continue;
        const coef = new Float64Array((J + 1) * (J1 + 1));
        const m2 = new Int32Array((J + 1) * (J1 + 1)).fill(-1);
        for (let k = 0; k <= J; k++) {
          for (let m1 = 0; m1 <= J1; m1++) {
            const m2i = k + (J1 + J2 - J) / 2 - m1;
            if (m2i < 0 || m2i > J2) continue;
            const M1 = 2 * m1 - J1, M2 = 2 * m2i - J2, M = 2 * k - J;
            const c = clebsch(J1, M1, J2, M2, J, M);
            if (c === 0) continue;
            coef[k * (J1 + 1) + m1] = c;
            m2[k * (J1 + 1) + m1] = m2i;
          }
        }
        out.push({ J1, J2, J, coef, m2 });
      }
    }
  }
  return out;
};

export class ComputeSnaAtom extends Compute {
  readonly style = 'sna/atom';
  peratomFlag = true;
  private readonly rcutfac: number;
  private readonly rfac0: number;
  private readonly twojmax: number;
  /** Per-type radius and neighbor weight, index t-1 for LAMMPS type t. */
  private readonly radius: Float64Array;
  private readonly weight: Float64Array;
  private readonly rmin0: number;
  private readonly switchflag: boolean;
  private readonly bzeroflag: boolean;
  private readonly quadraticflag: boolean;
  private readonly bnormflag: boolean;
  private readonly triples: Triple[];
  private readonly nbComps: number;
  private readonly ncols: number;
  /** Bispectrum of an atom with no neighbors, scaled as the output (for bzeroflag). */
  private b0: Float64Array | null = null;

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    const ntypes = sys.state.ntypes;
    const need = 3 + 2 * ntypes;
    if (args.length < need) {
      throw new StyleError(`compute ${id} (sna/atom): expects rcutfac rfac0 twojmax, ${ntypes} radii and ${ntypes} weights (${need} values), got ${args.length}`);
    }
    this.rcutfac = parseNum(args[0], `compute ${id} (sna/atom) rcutfac`);
    if (!(this.rcutfac > 0)) throw new StyleError(`compute ${id} (sna/atom): rcutfac must be positive (got ${args[0]})`);
    this.rfac0 = parseNum(args[1], `compute ${id} (sna/atom) rfac0`);
    const tj = parseInt_(args[2], `compute ${id} (sna/atom) twojmax`);
    if (tj < 0) throw new StyleError(`compute ${id} (sna/atom): twojmax must be a non-negative integer (got ${args[2]})`);
    this.twojmax = tj;
    this.radius = new Float64Array(ntypes);
    this.weight = new Float64Array(ntypes);
    for (let t = 0; t < ntypes; t++) {
      this.radius[t] = parseNum(args[3 + t], `compute ${id} (sna/atom) R_${t + 1}`);
      this.weight[t] = parseNum(args[3 + ntypes + t], `compute ${id} (sna/atom) w_${t + 1}`);
    }
    let rmin0 = 0;
    let switchflag = true, bzeroflag = true, quadraticflag = false, bnormflag = false;
    for (let k = need; k < args.length; k++) {
      const kw = args[k];
      const flag = (): boolean => {
        const w = args[++k];
        if (w !== '0' && w !== '1') throw new StyleError(`compute ${id} (sna/atom): ${kw} must be 0 or 1 (got '${w}')`);
        return w === '1';
      };
      if (kw === 'rmin0') rmin0 = parseNum(args[++k], `compute ${id} (sna/atom) rmin0`);
      else if (kw === 'switchflag') switchflag = flag();
      else if (kw === 'bzeroflag') bzeroflag = flag();
      else if (kw === 'quadraticflag') quadraticflag = flag();
      else if (kw === 'bnormflag') bnormflag = flag();
      else if (kw === 'wselfallflag') flag(); // only acts together with chem, which is not implemented
      else if (kw === 'chem' || kw === 'switchinnerflag' || kw === 'sinner' || kw === 'dinner' || kw === 'nnn' || kw === 'wmode' || kw === 'delta' || kw === 'bikflag' || kw === 'dgradflag') {
        throw new StyleError(`compute ${id} (sna/atom): keyword '${kw}' is not implemented in this engine`);
      } else {
        throw new StyleError(`compute ${id} (sna/atom): unknown keyword '${kw}'`);
      }
    }
    this.rmin0 = rmin0;
    this.switchflag = switchflag;
    this.bzeroflag = bzeroflag;
    this.quadraticflag = quadraticflag;
    this.bnormflag = bnormflag;
    this.triples = buildTriples(tj);
    this.nbComps = this.triples.length;
    this.ncols = this.nbComps + (quadraticflag ? (this.nbComps * (this.nbComps + 1)) / 2 : 0);
    this.sizePeratomCols = this.ncols;
  }

  /** Largest pair cutoff rcutfac (R_i + R_j) over the type pairs. */
  private maxCutoff(): number {
    let rmax = 0;
    for (let t = 0; t < this.radius.length; t++) if (this.radius[t] > rmax) rmax = this.radius[t];
    return this.rcutfac * 2 * rmax;
  }

  /** Bispectrum components (scaled) from the accumulated u matrices. */
  private bispectrum(u: { re: Float64Array; im: Float64Array }[], out: Float64Array): void {
    let c = 0;
    for (const t of this.triples) {
      const n1 = t.J1 + 1, n2 = t.J2 + 1, n = t.J + 1;
      const u1 = u[t.J1], u2 = u[t.J2], uj = u[t.J];
      let val = 0;
      // C_{k,kp} = sum_{m1,m1'} T[k,m1] T[kp,m1'] u1[m1,m1'] u2[m2(k,m1), m2(kp,m1')]
      for (let k = 0; k < n; k++) {
        for (let kp = 0; kp < n; kp++) {
          let cr = 0, ci = 0;
          for (let m1 = 0; m1 < n1; m1++) {
            const s1 = k * n1 + m1;
            const c1 = t.coef[s1];
            if (c1 === 0) continue;
            const m2 = t.m2[s1];
            for (let m1p = 0; m1p < n1; m1p++) {
              const s2 = kp * n1 + m1p;
              const c2 = t.coef[s2];
              if (c2 === 0) continue;
              const m2p = t.m2[s2];
              const ar = u1.re[m1 * n1 + m1p], ai = u1.im[m1 * n1 + m1p];
              const br = u2.re[m2 * n2 + m2p], bi = u2.im[m2 * n2 + m2p];
              const w = c1 * c2;
              cr += w * (ar * br - ai * bi);
              ci += w * (ar * bi + ai * br);
            }
          }
          // conj(u^j_{k,kp}) * C_{k,kp}, real part
          val += uj.re[k * n + kp] * cr + uj.im[k * n + kp] * ci;
        }
      }
      out[c++] = val;
    }
  }

  /** Output-column transform: B0 subtraction, bnorm, quadratic terms. */
  private finish(raw: Float64Array, row: Float64Array, off: number): void {
    const K = this.nbComps;
    for (let c = 0; c < K; c++) {
      let b = raw[c];
      if (this.bzeroflag) b -= this.b0![c];
      if (this.bnormflag) b /= this.triples[c].J + 1;
      row[off + c] = b;
    }
    if (this.quadraticflag) {
      let q = off + K;
      for (let i = 0; i < K; i++) {
        for (let j = i; j < K; j++) {
          row[q++] = i === j ? 0.5 * row[off + i] * row[off + i] : row[off + i] * row[off + j];
        }
      }
    }
  }

  protected computePeratom(): void {
    this.sys.forces();
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const cutmax = this.maxCutoff();
    if (nb.cutghost < cutmax - 1e-12) {
      throw new StyleError(`compute ${this.id} (sna/atom): cutoff rcutfac*(R_i+R_j) up to ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (set a pair style with a larger cutoff)`);
    }
    const n = s.n;
    const nall = nb.nall;
    const xa = nb.xall;
    const ta = nb.typeall;
    const out = (this.arrayAtom = new Float64Array(this.ncols * n));
    const tj = this.twojmax;
    if (this.bzeroflag && !this.b0) {
      const id: { re: Float64Array; im: Float64Array }[] = [];
      for (let J = 0; J <= tj; J++) {
        const m = J + 1;
        const re = new Float64Array(m * m), im = new Float64Array(m * m);
        for (let k = 0; k < m; k++) re[k * m + k] = 1;
        id.push({ re, im });
      }
      const raw = new Float64Array(this.nbComps);
      this.bispectrum(id, raw);
      this.b0 = raw.slice();
    }
    const u: { re: Float64Array; im: Float64Array }[] = [];
    for (let J = 0; J <= tj; J++) {
      const m = J + 1;
      u.push({ re: new Float64Array(m * m), im: new Float64Array(m * m) });
    }
    const raw = new Float64Array(this.nbComps);
    const rc = this.rcutfac;
    const rfac0 = this.rfac0;
    const rmin0 = this.rmin0;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ti = ta[i];
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      // self term: identity (weight 1 for the central atom)
      for (let J = 0; J <= tj; J++) {
        const m = J + 1;
        u[J].re.fill(0);
        u[J].im.fill(0);
        for (let k = 0; k < m; k++) u[J].re[k * m + k] = 1;
      }
      for (let k = 0; k < nall; k++) {
        if (k === i) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = rc * (this.radius[ti - 1] + this.radius[ta[k] - 1]);
        if (!(r < Rii) || r === 0) continue;
        const wj = this.weight[ta[k] - 1];
        const theta0 = (rfac0 * Math.PI * (r - rmin0)) / (Rii - rmin0);
        const z0 = r / Math.tan(theta0);
        const r0 = r / Math.sin(theta0);
        const fc = this.switchflag ? 0.5 * (Math.cos((Math.PI * (r - rmin0)) / (Rii - rmin0)) + 1) : 1;
        const sc = fc * wj;
        // Cayley-Klein parameters of the point on the 3-sphere
        const ar = z0 / r0, ai = dz / r0, br = dy / r0, bi = dx / r0;
        for (let J = 0; J <= tj; J++) {
          const U = wignerU(J, ar, ai, br, bi);
          const uj = u[J];
          for (let q = 0; q < U.re.length; q++) {
            uj.re[q] += sc * U.re[q];
            uj.im[q] += sc * U.im[q];
          }
        }
      }
      this.bispectrum(u, raw);
      this.finish(raw, out, i * this.ncols);
    }
  }
}
