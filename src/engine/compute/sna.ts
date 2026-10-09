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
 * The keyword *chem* activates the explicit multi-element form (docs: "The
 * keyword *chem* activates the explicit multi-element variant of the SNAP
 * bispectrum components. The argument *nelements* specifies the number of SNAP
 * elements that will be handled. This is followed by *elementlist*, a list of
 * integers of length *ntypes*, with values in the range [0, *nelements* ),
 * which maps each LAMMPS type to one of the SNAP elements."). The partial
 * density of element μ is
 * u^μ = wself_{μ_i μ} U(0,0,0) + sum_{j: elem(j)=μ} f_c w_{μ_j} U(θ0,θ,φ), the
 * bispectrum is indexed on ordered triplets B^{κλμ} = sum conj(u^μ) H u^κ u^λ,
 * and "the data is arranged into" N_elem^3 "sub-blocks, each sub-block
 * corresponding to a particular chemical labeling" κλμ "with the last label
 * changing fastest." For the self term, "If *wselfallflag* is on, then"
 * wself = 1; "If it is off then" wself = 0 "except in the case of" μ_i = μ.
 *
 * Unsupported (StyleError): switchinnerflag/sinner/dinner, nnn/wmode/delta,
 * chem on snad/atom and snav/atom, and the compute snap, sna/grid families.
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
export interface Triple {
  J1: number;
  J2: number;
  J: number;
  /** T[k*(J1+1)+m1] = CG(j1 m1, j2 m2 | j m) with m = k - j; m2 = pick[...]. */
  coef: Float64Array;
  /** Index m2 for the same flat slot (or -1 when no CG term). */
  m2: Int32Array;
}

/** Bispectrum component list in the documented order, with coupling tables. */
export const buildTriples = (twojmax: number): Triple[] => {
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
  readonly style: string = 'sna/atom';
  peratomFlag = true;
  protected readonly rcutfac: number;
  protected readonly rfac0: number;
  protected readonly twojmax: number;
  /** Per-type radius and neighbor weight, index t-1 for LAMMPS type t. */
  protected readonly radius: Float64Array;
  protected readonly weight: Float64Array;
  protected readonly rmin0: number;
  protected readonly switchflag: boolean;
  protected readonly bzeroflag: boolean;
  protected readonly quadraticflag: boolean;
  protected readonly bnormflag: boolean;
  /** chemflag: explicit multi-element bispectrum (docs.lammps.org/compute_sna_atom.html). */
  protected readonly chemflag: boolean;
  /** Number of SNAP elements when chemflag is set, else 1. */
  protected readonly nelements: number;
  /** LAMMPS type (t-1) -> SNAP element index; all 0 when chemflag is off. */
  protected readonly elemMap: Int32Array;
  protected readonly wselfallflag: boolean;
  protected readonly triples: Triple[];
  protected readonly nbComps: number;
  /** Number of linear bispectrum columns: K, or K*nelements^3 with chemflag. */
  protected readonly nbase: number;
  protected readonly ncols: number;
  /** Bispectrum of an atom with no neighbors (identity self term), per component. */
  protected b0: Float64Array | null = null;

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
    let chemflag = false, nelements = 1, wselfallflag = false;
    const elemMap = new Int32Array(ntypes);
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
      else if (kw === 'wselfallflag') wselfallflag = flag();
      else if (kw === 'chem') {
        // "chem values = nelements elementlist", "elementlist = ntypes integers in range [0, nelements)"
        const ne = parseInt_(args[++k], `compute ${id} (sna/atom) chem nelements`);
        if (ne < 1) throw new StyleError(`compute ${id} (sna/atom): chem nelements must be positive (got ${ne})`);
        for (let t = 0; t < ntypes; t++) {
          const e = parseInt_(args[++k], `compute ${id} (sna/atom) chem element ${t + 1}`);
          if (e < 0 || e >= ne) throw new StyleError(`compute ${id} (sna/atom): chem elementlist entry ${e} out of range [0, ${ne})`);
          elemMap[t] = e;
        }
        nelements = ne;
        chemflag = true;
      } else if (kw === 'switchinnerflag' || kw === 'sinner' || kw === 'dinner' || kw === 'nnn' || kw === 'wmode' || kw === 'delta' || kw === 'bikflag' || kw === 'dgradflag') {
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
    this.chemflag = chemflag;
    this.nelements = nelements;
    this.elemMap = elemMap;
    this.wselfallflag = wselfallflag;
    this.triples = buildTriples(tj);
    this.nbComps = this.triples.length;
    this.nbase = this.nbComps * (chemflag ? nelements * nelements * nelements : 1);
    this.ncols = this.nbase + (quadraticflag ? (this.nbase * (this.nbase + 1)) / 2 : 0);
    this.sizePeratomCols = this.ncols;
  }

  /** Largest pair cutoff rcutfac (R_i + R_j) over the type pairs. */
  protected maxCutoff(): number {
    let rmax = 0;
    for (let t = 0; t < this.radius.length; t++) if (this.radius[t] > rmax) rmax = this.radius[t];
    return this.rcutfac * 2 * rmax;
  }

  /**
   * Raw bispectrum of every chem block: block b = (κ,λ,μ) in row-major order
   * with μ fastest (docs: "each sub-block corresponding to a particular
   * chemical labeling" κλμ "with the last label changing fastest").
   */
  private rawBlocks(ue: Cmat[][], out: Float64Array): void {
    if (!this.chemflag) {
      rawBispectrum(this.triples, ue[0], out);
      return;
    }
    const Ne = this.nelements, K = this.nbComps;
    let b = 0;
    for (let k1 = 0; k1 < Ne; k1++) {
      for (let k2 = 0; k2 < Ne; k2++) {
        for (let k3 = 0; k3 < Ne; k3++) {
          for (let c = 0; c < K; c++) {
            const t = this.triples[c];
            out[b * K + c] = bispectrumComponent(t, ue[k1][t.J1], ue[k2][t.J2], ue[k3][t.J]);
          }
          b++;
        }
      }
    }
  }

  /** Output-column transform: B0 subtraction, bnorm, quadratic terms. */
  protected finish(raw: Float64Array, row: Float64Array, off: number, muI = 0): void {
    const K = this.nbComps, Ne = this.chemflag ? this.nelements : 1;
    const b0 = this.b0;
    let b = 0;
    for (let k1 = 0; k1 < Ne; k1++) {
      for (let k2 = 0; k2 < Ne; k2++) {
        for (let k3 = 0; k3 < Ne; k3++) {
          // B0 enters only for the self patterns: all elements with wselfallflag,
          // else only the block (μ_i,μ_i,μ_i)
          const useB0 = !this.chemflag || this.wselfallflag || (k1 === muI && k2 === muI && k3 === muI);
          const bo = b * K;
          for (let c = 0; c < K; c++) {
            let v = raw[bo + c];
            if (this.bzeroflag && useB0 && b0) v -= b0[c];
            if (this.bnormflag) v /= this.triples[c].J + 1;
            row[off + bo + c] = v;
          }
          b++;
        }
      }
    }
    if (this.quadraticflag) {
      const nb = this.nbase;
      let q = off + nb;
      for (let i = 0; i < nb; i++) {
        for (let j = i; j < nb; j++) {
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
    const Ne = this.chemflag ? this.nelements : 1;
    if (this.bzeroflag && !this.b0) {
      const id: Cmat[] = [];
      for (let J = 0; J <= tj; J++) {
        const m = J + 1;
        const re = new Float64Array(m * m), im = new Float64Array(m * m);
        for (let k = 0; k < m; k++) re[k * m + k] = 1;
        id.push({ re, im });
      }
      const raw = new Float64Array(this.nbComps);
      rawBispectrum(this.triples, id, raw);
      this.b0 = raw;
    }
    // one set of coefficient matrices per element for the explicit multi-element form
    const ue: Cmat[][] = [];
    for (let e = 0; e < Ne; e++) {
      const arr: Cmat[] = [];
      for (let J = 0; J <= tj; J++) {
        const m = J + 1;
        arr.push({ re: new Float64Array(m * m), im: new Float64Array(m * m) });
      }
      ue.push(arr);
    }
    const raw = new Float64Array(this.nbase);
    const rc = this.rcutfac;
    const rfac0 = this.rfac0;
    const rmin0 = this.rmin0;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const ti = ta[i];
      const muI = this.chemflag ? this.elemMap[ti - 1] : 0;
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      // self term: identity times w^self_{μ_i,e} (1 for all e with wselfallflag,
      // else 1 only for e = μ_i); docs.lammps.org/compute_sna_atom.html gives
      // u^μ = w^self U(0,0,0) + sum over neighbours of element μ
      for (let e = 0; e < Ne; e++) {
        const wself = this.wselfallflag || e === muI ? 1 : 0;
        for (let J = 0; J <= tj; J++) {
          const m = J + 1;
          ue[e][J].re.fill(0);
          ue[e][J].im.fill(0);
          if (wself !== 0) for (let k = 0; k < m; k++) ue[e][J].re[k * m + k] = 1;
        }
      }
      for (let k = 0; k < nall; k++) {
        if (k === i) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = rc * (this.radius[ti - 1] + this.radius[ta[k] - 1]);
        if (!(r < Rii) || r === 0) continue;
        const wj = this.weight[ta[k] - 1];
        const theta0 = (rfac0 * Math.PI * (r - rmin0)) / (Rii - rmin0);
        // positive radial normalization r0 = r / |sin theta0| (theta0 < 0 when r < rmin0)
        const sn = Math.sin(theta0), sg = sn < 0 ? -1 : 1;
        const r0 = r / Math.abs(sn);
        // f_c = 1 for r < rmin0 (measured with native LAMMPS; see the header of snap.ts)
        const fc = this.switchflag && r >= rmin0 ? 0.5 * (Math.cos((Math.PI * (r - rmin0)) / (Rii - rmin0)) + 1) : 1;
        const sc = fc * wj;
        // Cayley-Klein parameters of the point on the 3-sphere
        const ar = sg * Math.cos(theta0), ai = dz / r0, br = dy / r0, bi = dx / r0;
        const e = this.chemflag ? this.elemMap[ta[k] - 1] : 0;
        for (let J = 0; J <= tj; J++) {
          const U = wignerU(J, ar, ai, br, bi);
          const uj = ue[e][J];
          for (let q = 0; q < U.re.length; q++) {
            uj.re[q] += sc * U.re[q];
            uj.im[q] += sc * U.im[q];
          }
        }
      }
      this.rawBlocks(ue, raw);
      this.finish(raw, out, i * this.ncols, muI);
    }
  }
}

/*
 * Derivatives of the linear forms L1 = a x - conj(b) y and L2 = b x + conj(a) y
 * with respect to p = (Re a, Im a, Re b, Im b); [re, im] pairs. DL1Y/DL1X are the
 * y- and x-coefficients of dL1/dp, DL2Y/DL2X those of dL2/dp.
 */
const DL1Y: number[][] = [[0, 0], [0, 0], [-1, 0], [0, 1]];
const DL1X: number[][] = [[1, 0], [0, 1], [0, 0], [0, 0]];
const DL2Y: number[][] = [[1, 0], [0, -1], [0, 0], [0, 0]];
const DL2X: number[][] = [[0, 0], [0, 0], [1, 0], [0, 1]];


/** Real and imaginary flat (J+1)^2 matrices. */
export interface Cmat { re: Float64Array; im: Float64Array }

/**
 * One bispectrum component B_{j1,j2,j} = sum conj(u^j) H u^{j1} u^{j2}, with
 * the three expansion-coefficient matrices supplied explicitly (chemflag uses
 * a different element's partial density in each of the three slots).
 */
export const bispectrumComponent = (t: Triple, u1: Cmat, u2: Cmat, uj: Cmat): number => {
  const n1 = t.J1 + 1, n2 = t.J2 + 1, n = t.J + 1;
  let val = 0;
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
      val += uj.re[k * n + kp] * cr + uj.im[k * n + kp] * ci;
    }
  }
  return val;
};

/** Bispectrum components (raw, before B0/bnorm) and their adjoints, shared by the pair style. */
export const rawBispectrum = (triples: Triple[], u: Cmat[], out: Float64Array): void => {
  for (let c = 0; c < triples.length; c++) {
    const t = triples[c];
    out[c] = bispectrumComponent(t, u[t.J1], u[t.J2], u[t.J]);
  }
};

/** Gradient of a real function with respect to the real (r) and imaginary (i) parts of a matrix. */
export interface Grad { r: Float64Array; i: Float64Array }

/**
 * Adds sum_c g_c dB_c/d(u) to the gradients gr[J] (w.r.t. Re u^J) and gi[J]
 * (w.r.t. Im u^J), for the three roles of every component: the conjugated
 * u^J (dB/dRe = Re C, dB/dIm = Im C), and the holomorphic u^{J1}, u^{J2}
 * (dB/dRe = Re H, dB/dIm = -Im H), with the Clebsch-Gordan weights.
 */
export const adjointBispectrum = (triples: Triple[], u: Cmat[], g: Float64Array, gr: Grad[], gi: Grad[]): void => {
  for (let c = 0; c < triples.length; c++) {
    const gc = g[c];
    if (gc === 0) continue;
    const t = triples[c];
    const n1 = t.J1 + 1, n2 = t.J2 + 1, n = t.J + 1;
    const u1 = u[t.J1], u2 = u[t.J2], uj = u[t.J];
    for (let k = 0; k < n; k++) {
      for (let kp = 0; kp < n; kp++) {
        const cur = uj.re[k * n + kp], cui = uj.im[k * n + kp];
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
            const w = gc * c1 * c2;
            const a = u1.re[m1 * n1 + m1p], b = u1.im[m1 * n1 + m1p];
            const cc = u2.re[m2 * n2 + m2p], dd = u2.im[m2 * n2 + m2p];
            // role J: conj(u) * (u1 u2)
            const pr = a * cc - b * dd, pi = a * dd + b * cc;
            gr[t.J].r[k * n + kp] += w * pr;
            gi[t.J].r[k * n + kp] += w * pi;
            // role J1: conj(uJ) * u2 ; dB/dRe = Re h, dB/dIm = -Im h
            const hr = cur * cc + cui * dd, hi = cur * dd - cui * cc;
            gr[t.J1].r[m1 * n1 + m1p] += w * hr;
            gi[t.J1].r[m1 * n1 + m1p] -= w * hi;
            // role J2: conj(uJ) * u1
            const yr = cur * a + cui * b, yi = cur * b - cui * a;
            gr[t.J2].r[m2 * n2 + m2p] += w * yr;
            gi[t.J2].r[m2 * n2 + m2p] -= w * yi;
          }
        }
      }
    }
  }
};

/**
 * chemflag adjoint: adds g dB/d(u) to per-element gradient tables. The
 * conjugated slot J uses element gEj, the holomorphic slots J1, J2 use gE1,
 * gE2 (each an array of per-J real/imaginary gradient matrices). Same signs
 * as adjointBispectrum.
 */
export const adjointComponentChem = (
  t: Triple, u1: Cmat, u2: Cmat, uj: Cmat, g: number,
  gE1: Cmat[], gE2: Cmat[], gEj: Cmat[],
): void => {
  const n1 = t.J1 + 1, n2 = t.J2 + 1, n = t.J + 1;
  for (let k = 0; k < n; k++) {
    for (let kp = 0; kp < n; kp++) {
      const cur = uj.re[k * n + kp], cui = uj.im[k * n + kp];
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
          const w = g * c1 * c2;
          const a = u1.re[m1 * n1 + m1p], b = u1.im[m1 * n1 + m1p];
          const cc = u2.re[m2 * n2 + m2p], dd = u2.im[m2 * n2 + m2p];
          gEj[t.J].re[k * n + kp] += w * (a * cc - b * dd);
          gEj[t.J].im[k * n + kp] += w * (a * dd + b * cc);
          const hr = cur * cc + cui * dd, hi = cur * dd - cui * cc;
          gE1[t.J1].re[m1 * n1 + m1p] += w * hr;
          gE1[t.J1].im[m1 * n1 + m1p] -= w * hi;
          const yr = cur * a + cui * b, yi = cur * b - cui * a;
          gE2[t.J2].re[m2 * n2 + m2p] += w * yr;
          gE2[t.J2].im[m2 * n2 + m2p] -= w * yi;
        }
      }
    }
  }
};


/**
 * Cayley-Klein Wigner tables for every J <= tj, with optional derivatives with
 * respect to the four real parameters p = (Re a, Im a, Re b, Im b).
 *
 * Same matrices as wignerU (orthonormal monomial basis). The polynomial
 * P_{k,J} = L1^k L2^(J-k) (L1 = a x - conj(b) y, L2 = b x + conj(a) y) is built by
 * the recursion P_{0,J} = P_{0,J-1} L2 and P_{k,J} = P_{k-1,J-1} L1, each step
 * one multiplication by a linear form on flat arrays; the derivative chain
 * uses the constant derivative forms of L1 and L2. Output U^J is row-major
 * [kp*(J+1)+k], as in wignerU.
 */
export class WignerTables {
  readonly tj: number;
  /** Offset of the polynomial block of each J (size (J+1)^2 per J). */
  private readonly off: Int32Array;
  /** Normalization sqrt(kp!(J-kp)!)/sqrt(k!(J-k)!), per J, row-major [kp*(J+1)+k]. */
  private readonly norm: Float64Array[] = [];
  private readonly pr: Float64Array;
  private readonly pi: Float64Array;
  private readonly dpr: Float64Array[];
  private readonly dpi: Float64Array[];
  /** Outputs: U^J (value) and dU^J/dp_q (derivative), per J. */
  readonly ur: Float64Array[] = [];
  readonly ui: Float64Array[] = [];
  readonly dur: Float64Array[][] = [];
  readonly dui: Float64Array[][] = [];

  constructor(tj: number) {
    this.tj = tj;
    this.off = new Int32Array(tj + 2);
    for (let J = 0; J <= tj; J++) this.off[J + 1] = this.off[J] + (J + 1) * (J + 1);
    const S = this.off[tj + 1];
    this.pr = new Float64Array(S);
    this.pi = new Float64Array(S);
    this.dpr = [0, 1, 2, 3].map(() => new Float64Array(S));
    this.dpi = [0, 1, 2, 3].map(() => new Float64Array(S));
    for (let J = 0; J <= tj; J++) {
      const n = J + 1;
      const nm = new Float64Array(n * n);
      for (let kp = 0; kp <= J; kp++) {
        for (let k = 0; k <= J; k++) {
          nm[kp * n + k] = Math.sqrt(factorial(kp) * factorial(J - kp)) / Math.sqrt(factorial(k) * factorial(J - k));
        }
      }
      this.norm.push(nm);
      this.ur.push(new Float64Array(n * n));
      this.ui.push(new Float64Array(n * n));
      this.dur.push([0, 1, 2, 3].map(() => new Float64Array(n * n)));
      this.dui.push([0, 1, 2, 3].map(() => new Float64Array(n * n)));
    }
  }

  /** Fills the outputs for the Cayley-Klein parameters (a, b); derivatives only if deriv. */
  compute(ar: number, ai: number, br: number, bi: number, deriv: boolean): void {
    const { pr, pi, dpr, dpi, off } = this;
    // linear forms [y-coefficient, x-coefficient] as (re, im)
    const L1y = [-br, bi], L1x = [ar, ai];
    const L2y = [ar, -ai], L2x = [br, bi];
    pr[0] = 1; pi[0] = 0;
    if (deriv) for (let p = 0; p < 4; p++) { dpr[p][0] = 0; dpi[p][0] = 0; }
    for (let J = 1; J <= this.tj; J++) {
      const oJ = off[J], oP = off[J - 1], nJ = J + 1, nP = J;
      for (let k = 0; k <= J; k++) {
        const dst = oJ + k * nJ;
        const src = k === 0 ? oP : oP + (k - 1) * nP;
        // base form F and its constant derivative forms
        const fy = k === 0 ? L2y : L1y;
        const fx = k === 0 ? L2x : L1x;
        for (let i = 0; i < nJ; i++) {
          let vr = 0, vi = 0;
          if (i < nP) {
            const a = pr[src + i], b = pi[src + i];
            vr += fy[0] * a - fy[1] * b;
            vi += fy[0] * b + fy[1] * a;
          }
          if (i >= 1) {
            const a = pr[src + i - 1], b = pi[src + i - 1];
            vr += fx[0] * a - fx[1] * b;
            vi += fx[0] * b + fx[1] * a;
          }
          pr[dst + i] = vr;
          pi[dst + i] = vi;
        }
        if (!deriv) continue;
        for (let p = 0; p < 4; p++) {
          // constant derivative form of the base factor (complex [re, im] pairs)
          const dfy = k === 0 ? DL2Y[p] : DL1Y[p];
          const dfx = k === 0 ? DL2X[p] : DL1X[p];
          const Dr = dpr[p], Di = dpi[p];
          for (let i = 0; i < nJ; i++) {
            // d(new) = dold * F + old * dF
            let vr = 0, vi = 0;
            if (i < nP) {
              const a = Dr[src + i], b = Di[src + i];
              vr += fy[0] * a - fy[1] * b;
              vi += fy[0] * b + fy[1] * a;
              const c = pr[src + i], d = pi[src + i];
              vr += dfy[0] * c - dfy[1] * d;
              vi += dfy[0] * d + dfy[1] * c;
            }
            if (i >= 1) {
              const a = Dr[src + i - 1], b = Di[src + i - 1];
              vr += fx[0] * a - fx[1] * b;
              vi += fx[0] * b + fx[1] * a;
              const c = pr[src + i - 1], d = pi[src + i - 1];
              vr += dfx[0] * c - dfx[1] * d;
              vi += dfx[0] * d + dfx[1] * c;
            }
            Dr[dst + i] = vr;
            Di[dst + i] = vi;
          }
        }
      }
    }
    for (let J = 0; J <= this.tj; J++) {
      const n = J + 1, oJ = off[J], nm = this.norm[J];
      const ur = this.ur[J], ui = this.ui[J];
      for (let k = 0; k <= J; k++) {
        for (let kp = 0; kp < n; kp++) {
          const s = nm[kp * n + k];
          const src = oJ + k * n + kp;
          ur[kp * n + k] = pr[src] * s;
          ui[kp * n + k] = pi[src] * s;
          if (deriv) {
            for (let p = 0; p < 4; p++) {
              this.dur[J][p][kp * n + k] = dpr[p][src] * s;
              this.dui[J][p][kp * n + k] = dpi[p][src] * s;
            }
          }
        }
      }
    }
  }
}

/**
 * compute snad/atom and snav/atom (docs.lammps.org/compute_sna_atom.html).
 *
 * "Compute *snad/atom* calculates the derivative of the bispectrum components
 * summed separately for each LAMMPS atom type:" -sum_{i' in I} dB^{i'}_{j1,j2,j}/dr_i;
 * "The sum is over all atoms *i'* of atom type *I*". "Compute *snav/atom* calculates
 * the virial contribution due to the derivatives:" -r_i (x) sum_{i' in I} dB^{i'}/dr_i.
 *
 * Layout (doc, Output info): "Compute *snad/atom* evaluates a per-atom array.
 * The columns are arranged into *ntypes* blocks, listed in order of atom type I.
 * Each block contains three sub-blocks corresponding to the *x*, *y*, and *z*
 * components of the atom position." "Compute *snav/atom* ... Each block contains
 * six sub-blocks corresponding to the *xx*, *yy*, *zz*, *yz*, *xz*, and *xy*
 * components". "For computes *snad/atom* and *snav/atom* each set of K(K+1)/2
 * additional columns is inserted directly after each of sub-block of linear
 * terms i.e. linear and quadratic terms are contiguous."
 *
 * snav (measured with native LAMMPS, black box, periodic and non-periodic cases):
 * the virial of atom o uses the image positions x_j of the neighbour entries
 * that are images of o, i.e. snav_o = sum_{i' in I} sum_{j: owner(j)=o} -x_j (x) G_j
 * plus the self term x_o (x) sum_j G_{o->j} for o in I, with G_j = dB^{i'}/dd_j.
 * For a non-periodic cluster this equals x_o (x) snad_o. The Voigt off-diagonal
 * components use the pair order (z,y), (z,x), (y,x) for yz, xz, xy (measured).
 *
 * Derivation used for snad: the sum over i' of type I runs over owned atoms i'
 * (the central atoms) and their neighbour entries j (ghost images are folded
 * to their owner). With d_j = x_j - x_i', the term -dB^{i'}/dx_o is
 * +sum_j G_j for o = i' (translation invariance) and -G_j for o = owner(j),
 * where G_j = dB^{i'}/dd_j. Quadratic terms use dQ = B_k dB_l + B_l dB_k
 * (Q_kk = B_k^2/2). The Cayley-Klein parameters and f_c follow the sna/atom
 * conventions (positive radial normalization; f_c = 1 for r < rmin0).
 */
export class ComputeSnaDeriv extends ComputeSnaAtom {
  override readonly style: string;
  private readonly mode: 'snad' | 'snav';

  constructor(sys: System, id: string, group: string, args: string[], mode: 'snad' | 'snav') {
    super(sys, id, group, args);
    this.mode = mode;
    this.style = `${mode}/atom`;
    if (this.chemflag) throw new StyleError(`compute ${id} (${mode}/atom): keyword 'chem' is not implemented in this engine`);
    const K = this.nbComps;
    const blk = K + (this.quadraticflag ? (K * (K + 1)) / 2 : 0);
    const nt = this.radius.length;
    this.sizePeratomCols = mode === 'snad' ? nt * 3 * blk : nt * 6 * blk;
  }

  protected override computePeratom(): void {
    this.sys.forces();
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const cutmax = this.maxCutoff();
    if (nb.cutghost < cutmax - 1e-12) {
      throw new StyleError(`compute ${this.id} (${this.style}): cutoff up to ${cutmax} exceeds the ghost cutoff ${nb.cutghost} (set a pair style with a larger cutoff)`);
    }
    const n = s.n, nall = nb.nall, xa = nb.xall, ta = nb.typeall, owner = nb.owner;
    const tj = this.twojmax, triples = this.triples, K = this.nbComps;
    const quad = this.quadraticflag;
    const Q = quad ? (K * (K + 1)) / 2 : 0;
    const blk = K + Q;
    const nt = this.radius.length;
    const ncolSnad = nt * 3 * blk;
    const snad = new Float64Array(n * ncolSnad);
    // snav: pair virial with the image positions of the neighbours (see the class header)
    const pairsV: [number, number][] = [[0, 0], [1, 1], [2, 2], [2, 1], [2, 0], [1, 0]];
    const ncolSnav = nt * 6 * blk;
    const vir = this.mode === 'snav' ? new Float64Array(n * ncolSnav) : null;
    // B0 and normalizations (same conventions as ComputeSnaAtom.finish)
    const b0 = new Float64Array(K);
    {
      const id: Cmat[] = [];
      for (let J = 0; J <= tj; J++) {
        const m = J + 1;
        const re = new Float64Array(m * m), im = new Float64Array(m * m);
        for (let k = 0; k < m; k++) re[k * m + k] = 1;
        id.push({ re, im });
      }
      rawBispectrum(triples, id, b0);
      if (!this.bzeroflag) b0.fill(0);
    }
    const norm = new Float64Array(K);
    for (let c = 0; c < K; c++) norm[c] = this.bnormflag ? triples[c].J + 1 : 1;

    // per-J offsets of the flat Wigner tables
    const offJ = new Int32Array(tj + 2);
    for (let J = 0; J <= tj; J++) offJ[J + 1] = offJ[J] + (J + 1) * (J + 1);
    const S = offJ[tj + 1];
    const T = new WignerTables(tj);
    const u: Cmat[] = [], gr: Grad[] = [], gi: Grad[] = [];
    for (let J = 0; J <= tj; J++) {
      const nn = (J + 1) * (J + 1);
      u.push({ re: new Float64Array(nn), im: new Float64Array(nn) });
      gr.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
      gi.push({ r: new Float64Array(nn), i: new Float64Array(nn) });
    }
    const cap = Math.max(nall, 1);
    const nbJ = new Int32Array(cap);
    const nDx = new Float64Array(cap), nDy = new Float64Array(cap), nDz = new Float64Array(cap);
    const nR = new Float64Array(cap), nTh = new Float64Array(cap), nRc = new Float64Array(cap);
    const nSc = new Float64Array(cap), nDsc = new Float64Array(cap);
    // derivative tables per neighbour: WR/WI[a][(offJ[J]+q)*3 + m]
    let WR = new Float64Array(0), WI = new Float64Array(0);
    // Jacobian: G[k][a][m], dQ for quadratic pairs
    let G = new Float64Array(0);
    const raw = new Float64Array(K), Bf = new Float64Array(K), gbuf = new Float64Array(K);
    const dQ = new Float64Array(Q * 3);
    const rfac0 = this.rfac0, rmin0 = this.rmin0;

    for (let ip = 0; ip < n; ip++) {
      if (!(s.mask[ip] & this.groupBit)) continue;
      const tI = ta[ip] - 1;
      const xi = xa[3 * ip], yi = xa[3 * ip + 1], zi = xa[3 * ip + 2];
      // neighbours inside their pair cutoff
      let m = 0;
      for (let k = 0; k < nall; k++) {
        if (k === ip) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Rii = this.rcutfac * (this.radius[tI] + this.radius[ta[k] - 1]);
        if (!(r < Rii) || r === 0) continue;
        nbJ[m] = k; nDx[m] = dx; nDy[m] = dy; nDz[m] = dz; nR[m] = r; nRc[m] = Rii;
        const span = Rii - rmin0;
        nTh[m] = (rfac0 * Math.PI * (r - rmin0)) / span;
        const inside = this.switchflag && r >= rmin0;
        const fc = inside ? 0.5 * (Math.cos((Math.PI * (r - rmin0)) / span) + 1) : 1;
        const dfc = inside ? -0.5 * Math.sin((Math.PI * (r - rmin0)) / span) * (Math.PI / span) : 0;
        const w = this.weight[ta[k] - 1];
        nSc[m] = fc * w; nDsc[m] = dfc * w;
        m++;
      }
      if (WR.length < m * 3 * S) {
        WR = new Float64Array(m * 3 * S);
        WI = new Float64Array(m * 3 * S);
        G = new Float64Array(K * m * 3);
      }
      // pass 1: u matrices, and derivative tables of every neighbour
      for (let J = 0; J <= tj; J++) {
        u[J].re.fill(0); u[J].im.fill(0);
        for (let q = 0; q < J + 1; q++) u[J].re[q * (J + 1) + q] = 1;
      }
      for (let a = 0; a < m; a++) {
        const r = nR[a];
        const sn = Math.sin(nTh[a]), cs = Math.cos(nTh[a]);
        const sg = sn < 0 ? -1 : 1;
        const Sx = sg * sn, Cx = sg * cs;
        const g = Sx / r;
        const dx = nDx[a], dy = nDy[a], dz = nDz[a];
        const nm = [dx / r, dy / r, dz / r];
        const thp = (rfac0 * Math.PI) / (nRc[a] - rmin0);
        const gp = (Cx * thp * r - Sx) / (r * r);
        const dp = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
        const vz = [0, 0, 1], vy = [0, 1, 0], vx = [1, 0, 0];
        for (let mm = 0; mm < 3; mm++) {
          dp[0][mm] = -Sx * thp * nm[mm];
          dp[1][mm] = gp * nm[mm] * dz + g * vz[mm];
          dp[2][mm] = gp * nm[mm] * dy + g * vy[mm];
          dp[3][mm] = gp * nm[mm] * dx + g * vx[mm];
        }
        const ar = Cx, ai = g * dz, br = g * dy, bi = g * dx;
        const sc = nSc[a], dsc = nDsc[a];
        T.compute(ar, ai, br, bi, true);
        for (let J = 0; J <= tj; J++) {
          const nn = (J + 1) * (J + 1);
          const Ur = T.ur[J], Ui = T.ui[J];
          const Dr = T.dur[J], Di = T.dui[J];
          for (let q = 0; q < nn; q++) {
            u[J].re[q] += sc * Ur[q];
            u[J].im[q] += sc * Ui[q];
            const base = a * 3 * S + (offJ[J] + q) * 3;
            for (let mm = 0; mm < 3; mm++) {
              let dre = dsc * nm[mm] * Ur[q], dim = dsc * nm[mm] * Ui[q];
              for (let pp = 0; pp < 4; pp++) {
                dre += sc * Dr[pp][q] * dp[pp][mm];
                dim += sc * Di[pp][q] * dp[pp][mm];
              }
              WR[base + mm] = dre;
              WI[base + mm] = dim;
            }
          }
        }
      }
      rawBispectrum(triples, u, raw);
      for (let c = 0; c < K; c++) Bf[c] = (raw[c] - b0[c]) / norm[c];

      // Jacobian of every bispectrum component with respect to every neighbour displacement
      for (let k = 0; k < K; k++) {
        gbuf.fill(0);
        gbuf[k] = 1 / norm[k];
        for (let J = 0; J <= tj; J++) { gr[J].r.fill(0); gi[J].r.fill(0); }
        adjointBispectrum(triples, u, gbuf, gr, gi);
        for (let a = 0; a < m; a++) {
          for (let mm = 0; mm < 3; mm++) {
            let acc = 0;
            for (let J = 0; J <= tj; J++) {
              const nn = (J + 1) * (J + 1);
              const GR = gr[J].r, GI = gi[J].r;
              for (let q = 0; q < nn; q++) {
                const gR = GR[q], gI = GI[q];
                if (gR === 0 && gI === 0) continue;
                const idx = a * 3 * S + (offJ[J] + q) * 3 + mm;
                acc += gR * WR[idx] + gI * WI[idx];
              }
            }
            G[(k * m + a) * 3 + mm] = acc;
          }
        }
      }

      // accumulate into snad: -dB/dx_o for every owner o of a neighbour, +sum_j for o = i'
      for (let a = 0; a < m; a++) {
        const o = owner[nbJ[a]];
        // quadratic derivatives for this neighbour
        if (quad) {
          let q = 0;
          for (let kk = 0; kk < K; kk++) {
            for (let ll = kk; ll < K; ll++) {
              for (let mm = 0; mm < 3; mm++) {
                const gk = G[(kk * m + a) * 3 + mm], gl = G[(ll * m + a) * 3 + mm];
                dQ[q * 3 + mm] = kk === ll ? Bf[kk] * gk : Bf[kk] * gl + Bf[ll] * gk;
              }
              q++;
            }
          }
        }
        if (vir) {
          const jx = xa[3 * nbJ[a]], jy = xa[3 * nbJ[a] + 1], jz = xa[3 * nbJ[a] + 2];
          const xj = [jx, jy, jz];
          const xi3 = [xi, yi, zi];
          for (let c = 0; c < 6; c++) {
            const [pa, pb] = pairsV[c];
            const colBase = (tI * 6 + c) * blk;
            for (let k = 0; k < K; k++) {
              const g = G[(k * m + a) * 3 + pb];
              vir[o * ncolSnav + colBase + k] -= xj[pa] * g;
              vir[ip * ncolSnav + colBase + k] += xi3[pa] * g;
            }
            for (let q = 0; q < Q; q++) {
              const g = dQ[q * 3 + pb];
              vir[o * ncolSnav + colBase + K + q] -= xj[pa] * g;
              vir[ip * ncolSnav + colBase + K + q] += xi3[pa] * g;
            }
          }
        }
        for (let mm = 0; mm < 3; mm++) {
          const colBase = tI * 3 * blk + mm * blk;
          for (let k = 0; k < K; k++) {
            const g = G[(k * m + a) * 3 + mm];
            snad[o * ncolSnad + colBase + k] -= g;
            snad[ip * ncolSnad + colBase + k] += g;
          }
          for (let q = 0; q < Q; q++) {
            const g = dQ[q * 3 + mm];
            snad[o * ncolSnad + colBase + K + q] -= g;
            snad[ip * ncolSnad + colBase + K + q] += g;
          }
        }
      }
    }

    // output rows of the group atoms only
    if (this.mode === 'snad') {
      this.arrayAtom = new Float64Array(n * ncolSnad);
      for (let i = 0; i < n; i++) {
        if (!(s.mask[i] & this.groupBit)) continue;
        for (let c = 0; c < ncolSnad; c++) this.arrayAtom[i * ncolSnad + c] = snad[i * ncolSnad + c];
      }
      return;
    }
    // snav: Voigt order xx yy zz yz xz xy (vir was accumulated above)
    this.arrayAtom = new Float64Array(n * ncolSnav);
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      for (let c = 0; c < ncolSnav; c++) this.arrayAtom[i * ncolSnav + c] = vir![i * ncolSnav + c];
    }
  }
}
