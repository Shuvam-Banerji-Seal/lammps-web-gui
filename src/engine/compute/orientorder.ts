import { Compute } from './compute';
import { StyleError } from '../force/types';
import type { System } from '../system';
import { parseNum, parseInt_ } from '../force/util';

/*
 * compute ID group-ID orientorder/atom keyword values ...
 * — docs.lammps.org/compute_orientorder_atom.html
 *
 *   compute ID group-ID orientorder/atom keyword values ...
 *
 *   keyword = *cutoff* or *nnn* or *degrees* or *wl* or *wl/hat* or *components* or *chunksize*
 *     *cutoff* value = distance cutoff
 *     *nnn* value = number of nearest neighbors
 *     *degrees* values = nlvalues, l1, l2,...
 *     *wl* value = *yes* or *no*
 *     *wl/hat* value = *yes* or *no*
 *     *components* value = ldegree
 *     *chunksize* value = number of atoms in each pass
 *
 * "Define a computation that calculates a set of bond-orientational
 * order parameters :math:`Q_\ell` for each atom in a group. These order
 * parameters were introduced by :ref:`Steinhardt et al. <Steinhardt>` as a
 * way to characterize the local orientational order in atomic structures.
 * For each atom, :math:`Q_\ell` is a real number defined as follows:"
 *
 *   \bar{Y}_{\ell m} = & \frac{1}{nnn}\sum_{j = 1}^{nnn} Y_{\ell m}\bigl( \theta( \mathbf{r}_{ij} ), \phi( \mathbf{r}_{ij} ) \bigr) \\
 *   Q_\ell  = & \sqrt{\frac{4 \pi}{2 \ell  + 1} \sum_{m = -\ell }^{m = \ell } \bar{Y}_{\ell m} \bar{Y}^*_{\ell m}}
 *
 * "The summation is over the *nnn* nearest neighbors of the central atom.
 * The angles :math:`\theta` and :math:`\phi` are the standard spherical polar
 * angles defining the direction of the bond vector :math:`r_{ij}`. The phase
 * and sign of :math:`Y_{\ell m}` follow the standard conventions, so that
 * :math:`\mathrm{sign}(Y_{\ell\ell}(0,0)) = (-1)^\ell`."
 * For the FCC crystal with nnn = 12,
 *
 *   Q_4 = \sqrt{\frac{7}{192}} \approx 0.19094
 *
 * "The optional keyword *cutoff* defines the distance cutoff
 * used when searching for neighbors. The default value, also
 * the maximum allowable value, is the cutoff specified
 * by the pair style."
 *
 * "The optional keyword *nnn* defines the number of nearest
 * neighbors used to calculate :math:`Q_\ell`. The default value is 12.
 * If the value is NULL, then all neighbors up to the
 * specified distance cutoff are used."
 *
 * "The optional keyword *wl* will output the third-order invariants
 * :math:`W_\ell` (see Eq. 1.4 in :ref:`Steinhardt <Steinhardt>`) for the same
 * degrees as for the :math:`Q_\ell` parameters. For the FCC crystal with
 * *nnn* = 12,"
 *
 *   W_4 = -\sqrt{\frac{14}{143}} \left(\frac{49}{4096}\right) \pi^{-3/2} \approx -0.0006722136
 *
 * "The optional keyword *wl/hat* will output the normalized third-order
 * invariants :math:`\hat{W}_\ell` (see Eq. 2.2 in
 * :ref:`Steinhardt <Steinhardt>`) for the same degrees as for the
 * :math:`Q_\ell` parameters. For the FCC crystal with *nnn* =12,"
 *
 *   \hat{W}_4 = -\frac{7}{3} \sqrt{\frac{2}{429}} \approx -0.159317
 * "The optional keyword *components* will output the components of the
 * *normalized* complex vector
 * :math:`\hat{Y}_{\ell m} = \bar{Y}_{\ell m}/|\bar{Y}_{\ell m}|`
 * of degree *ldegree*\, which must be included in the list of order parameters to
 * be computed. This option can be used in conjunction with compute coord_atom
 * to calculate the ten Wolde's criterion to identify crystal-like particles"
 * (ten Wolde et al., J. Chem. Phys. 104, 9932 (1996), whose normalized
 * vector divides each component by the vector norm sqrt(sum_m |Ybar_lm|^2)).
 *
 * "The optional keyword *chunksize* is only applicable when using the
 * the KOKKOS package and is ignored otherwise."
 *
 * "The value of :math:`Q_\ell` is set to zero for atoms not in the
 * specified compute group, as well as for atoms that have less than
 * *nnn* neighbors within the distance cutoff, unless *nnn* is NULL."
 *
 * Output info: "This compute calculates a per-atom array with *nlvalues*
 * columns, giving the :math:`Q_\ell` values for each atom, which are real
 * numbers in the range :math:`0 \le Q_\ell \le 1`."
 *
 * "In summary, the per-atom array will contain *nlvalues* columns, followed by
 * an additional *nlvalues* columns if *wl* is set to yes, followed by
 * an additional *nlvalues* columns if *wl/hat* is set to yes, followed
 * by an additional 2\*(2\* *ldegree*\ +1) columns if the *components*
 * keyword is set." With components, "the real and imaginary parts of each
 * component of *normalized* :math:`\hat{Y}_{\ell m}` will be added to the
 * output array in the following order:" Re(Yhat_{-m}), Im(Yhat_{-m}),
 * Re(Yhat_{-m+1}), Im(Yhat_{-m+1}), ..., Re(Yhat_m), Im(Yhat_m).
 *
 * "The neighbor list needed to compute this quantity is constructed each
 * time the calculation is performed (i.e., each time a snapshot of atoms is
 * dumped)."
 *
 * Default: "The option defaults are *cutoff* = pair style cutoff, *nnn* = 12,
 * *degrees* = 5 4 6 8 10 12 (i.e., :math:`Q_4`, :math:`Q_6`, :math:`Q_8`,
 * :math:`Q_{10}`, and :math:`Q_{12}`), *wl* = no, *wl/hat* = no,
 * *components* off, and *chunksize* = 16384"
 *
 * Implementation notes. Y_lm use the Condon-Shortley phase (textbook
 * associated-Legendre recurrence; Y_{l,-m} = (-1)^m conj(Y_{l,m})). The
 * third-order invariant is computed from the same Ybar averages as Q_l:
 *   Winv_l = sum_{m1+m2+m3=0} (l l l; m1 m2 m3) Ybar_{m1} Ybar_{m2} Ybar_{m3},
 * with the Wigner 3j symbols from Racah's sum formula. Because
 * conj(Winv_l) = (-1)^l Winv_l, the invariant is real for even l (all degrees
 * the documentation gives values for) and purely imaginary for odd l; the
 * real part is stored, which is exact for even degrees. The wl column outputs
 * Winv_l / sqrt(2l+1) and the wl/hat column Winv_l / (sum_m |Ybar_lm|^2)^{3/2}:
 * the oracle case w5orient_q (native LAMMPS 2 Sep 2026) measures
 * wl = Winv/sqrt(2l+1) exactly (divisors 1/3, 1/sqrt(13), 1/sqrt(17) for
 * l = 4, 6, 8 on every atom), while wl/hat and the doc's FCC example values
 * (W_4 = -0.0006722136... = Winv without the factor, What_4 = -0.159317...)
 * pin the formulas above; What_6 = -0.0026260/sqrt(...) literature values
 * reproduce as well. Neighbors: a direct loop over
 * owned + ghost atoms (nb.xall, the full periodic environment) within the
 * cutoff, taking the nnn nearest when nnn is set — the doc note about
 * special_bonds removing 1-2/1-3/1-4 pairs from LAMMPS's list does not apply
 * to this engine's direct loop, which always sees every periodic image pair.
 */

/** Grow-and-cache factorials as doubles (sizes here stay far below 170!). */
const FACT: number[] = [1];
const fact = (k: number): number => {
  while (FACT.length <= k) FACT.push(FACT[FACT.length - 1] * FACT.length);
  return FACT[k];
};

/**
 * Wigner 3j symbol (l l l; m1 m2 m3) from Racah's sum formula (textbook):
 *   (l l l; m1 m2 m3) = (-1)^(-m3) sqrt(Delta) sqrt(prod (l+/-mi)!)
 *                       * sum_k (-1)^k / [k!(l-k)!(l-m1-k)!(l+m2-k)!(l+m1+k)!(l-m2+k)!]
 * with Delta = (l!)^3/(3l+1)!, k over all integers keeping every factorial
 * argument non-negative.
 */
const w3jLLL = (l: number, m1: number, m2: number, m3: number): number => {
  if (m1 + m2 + m3 !== 0) return 0;
  if (Math.abs(m1) > l || Math.abs(m2) > l || Math.abs(m3) > l) return 0;
  const kmin = Math.max(0, -m1, m2);
  const kmax = Math.min(l, l - m1, l + m2);
  if (kmax < kmin) return 0;
  let sum = 0;
  for (let k = kmin; k <= kmax; k++) {
    sum += (k & 1 ? -1 : 1) / (fact(k) * fact(l - k) * fact(l - m1 - k) * fact(l + m2 - k) * fact(m1 + k) * fact(k - m2));
  }
  let pref = (Math.abs(m3) & 1 ? -1 : 1) * Math.sqrt(fact(l) ** 3 / fact(3 * l + 1));
  pref *= Math.sqrt(fact(l + m1) * fact(l - m1));
  pref *= Math.sqrt(fact(l + m2) * fact(l - m2));
  pref *= Math.sqrt(fact(l + m3) * fact(l - m3));
  return pref * sum;
};

export class ComputeOrientorderAtom extends Compute {
  readonly style = 'orientorder/atom';
  peratomFlag = true;

  private readonly degrees: number[];
  /** null = all neighbors up to the cutoff; else the nnn nearest. */
  private readonly nnn: number | null;
  /** User cutoff keyword, or null for the pair style cutoff. */
  private readonly cutoffArg: number | null;
  private readonly wl: boolean;
  private readonly wlHat: boolean;
  private readonly componentsDeg: number | null;

  private readonly maxL: number;
  /** N_lm = sqrt((2l+1)/(4pi) (l-m)!/(l+m)!) per degree, m = 0..l. */
  private readonly norms: Float64Array[] = [];
  /** Nonzero 3j (l,l,l;m1,m2,m3) per degree: flat m1,m2,m3 (offset l) + coef. */
  private readonly tripIdx: Int32Array[] = [];
  private readonly tripCoef: Float64Array[] = [];
  /** Odd double factorials (2m-1)!! for the Legendre recurrence. */
  private readonly oddf: Float64Array;
  /** Associated Legendre table P_m^l(cos theta), m = 0..maxL, l = m..maxL. */
  private readonly ptab: Float64Array;
  /** Raw (unnormalized) Ybar sums per degree. */
  private readonly yre: Float64Array[] = [];
  private readonly yim: Float64Array[] = [];
  /** Neighbor scratch: squared distances, bond vectors, selection order. */
  private r2a = new Float64Array(0);
  private dxa = new Float64Array(0);
  private dya = new Float64Array(0);
  private dza = new Float64Array(0);
  private order = new Int32Array(0);

  constructor(sys: System, id: string, group: string, args: string[]) {
    super(sys, id, group, args);
    let nnn: number | null = 12;
    let cutoffArg: number | null = null;
    let wl = false;
    let wlHat = false;
    let componentsDeg: number | null = null;
    let degrees: number[] | null = null;
    for (let k = 0; k < args.length; k++) {
      const kw = args[k];
      if (kw === 'cutoff') {
        const v = parseNum(args[++k], `compute ${id} (orientorder/atom) cutoff`);
        if (!(v > 0)) throw new StyleError(`compute ${id} (orientorder/atom): cutoff must be > 0 (got ${args[k]})`);
        cutoffArg = v;
        continue;
      }
      if (kw === 'nnn') {
        const w = args[++k];
        if (w === undefined) throw new StyleError(`compute ${id} (orientorder/atom): nnn needs a value or NULL`);
        if (w === 'NULL') nnn = null;
        else {
          const v = parseInt_(w, `compute ${id} (orientorder/atom) nnn`);
          if (v < 1) throw new StyleError(`compute ${id} (orientorder/atom): nnn must be a positive integer or NULL (got ${w})`);
          nnn = v;
        }
        continue;
      }
      if (kw === 'degrees') {
        const nlv = parseInt_(args[++k], `compute ${id} (orientorder/atom) degrees nlvalues`);
        if (nlv < 1) throw new StyleError(`compute ${id} (orientorder/atom): degrees nlvalues must be >= 1 (got ${args[k]})`);
        const list: number[] = [];
        for (let t = 0; t < nlv; t++) {
          const w = args[++k];
          if (w === undefined) throw new StyleError(`compute ${id} (orientorder/atom): degrees needs ${nlv} degrees after nlvalues, got ${t}`);
          const v = parseInt_(w, `compute ${id} (orientorder/atom) degrees l${t + 1}`);
          if (v < 0) throw new StyleError(`compute ${id} (orientorder/atom): degrees must be non-negative integers (got ${w})`);
          list.push(v);
        }
        degrees = list;
        continue;
      }
      if (kw === 'wl' || kw === 'wl/hat') {
        const w = args[++k];
        if (w !== 'yes' && w !== 'no') throw new StyleError(`compute ${id} (orientorder/atom): ${kw} must be yes or no (got '${w}')`);
        if (kw === 'wl') wl = w === 'yes';
        else wlHat = w === 'yes';
        continue;
      }
      if (kw === 'components') {
        const v = parseInt_(args[++k], `compute ${id} (orientorder/atom) components ldegree`);
        if (v < 0) throw new StyleError(`compute ${id} (orientorder/atom): components ldegree must be a non-negative integer (got ${args[k]})`);
        componentsDeg = v;
        continue;
      }
      if (kw === 'chunksize') {
        const v = parseInt_(args[++k], `compute ${id} (orientorder/atom) chunksize`);
        if (v < 1) throw new StyleError(`compute ${id} (orientorder/atom): chunksize must be a positive integer (got ${args[k]})`);
        continue; // KOKKOS-only pass size, ignored outside the KOKKOS package
      }
      throw new StyleError(`compute ${id} (orientorder/atom): unknown keyword '${kw}' (use cutoff, nnn, degrees, wl, wl/hat, components, chunksize)`);
    }
    this.degrees = degrees ?? [4, 6, 8, 10, 12];
    this.nnn = nnn;
    this.cutoffArg = cutoffArg;
    this.wl = wl;
    this.wlHat = wlHat;
    this.componentsDeg = componentsDeg;
    if (componentsDeg !== null && !this.degrees.includes(componentsDeg)) {
      throw new StyleError(`compute ${id} (orientorder/atom): components degree ${componentsDeg} must be included in the list of order parameters (degrees ${this.degrees.join(' ')})`);
    }
    let maxL = 0;
    for (const l of this.degrees) if (l > maxL) maxL = l;
    this.maxL = maxL;
    const nlv = this.degrees.length;
    let wCols = 0;
    let hatCols = 0;
    let compCols = 0;
    if (wl) wCols = nlv;
    if (wlHat) hatCols = nlv;
    if (componentsDeg !== null) compCols = 2 * (2 * componentsDeg + 1);
    this.sizePeratomCols = nlv + wCols + hatCols + compCols;
    // per-degree constants
    const L = maxL;
    this.ptab = new Float64Array((L + 1) * (L + 1));
    this.oddf = new Float64Array(L + 1);
    this.oddf[0] = 1;
    for (let m = 1; m <= L; m++) this.oddf[m] = this.oddf[m - 1] * (2 * m - 1);
    for (const l of this.degrees) {
      const nm = new Float64Array(l + 1);
      for (let m = 0; m <= l; m++) nm[m] = Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * fact(l - m) / fact(l + m));
      this.norms.push(nm);
      const idx: number[] = [];
      const cf: number[] = [];
      for (let m1 = -l; m1 <= l; m1++) {
        for (let m2 = -l; m2 <= l; m2++) {
          const m3 = -(m1 + m2);
          if (Math.abs(m3) > l) continue;
          const c = w3jLLL(l, m1, m2, m3);
          if (c === 0) continue;
          idx.push(m1 + l, m2 + l, m3 + l);
          cf.push(c);
        }
      }
      this.tripIdx.push(Int32Array.from(idx));
      this.tripCoef = [...this.tripCoef, Float64Array.from(cf)];
      this.yre.push(new Float64Array(2 * l + 1));
      this.yim.push(new Float64Array(2 * l + 1));
    }
  }

  /** Effective distance cutoff: keyword value or the pair style cutoff (the maximum allowable value). */
  private resolveCutoff(): number {
    const pair = this.sys.ff.pair;
    let maxCut = 0;
    if (pair) {
      const cut = pair.cut;
      for (let k = 0; k < cut.length; k++) if (cut[k] > maxCut) maxCut = cut[k];
    }
    const cut = this.cutoffArg ?? maxCut;
    if (!(cut > 0)) {
      throw new StyleError(`compute ${this.id} (orientorder/atom): no pair style cutoff is defined; define a pair style or use the cutoff keyword`);
    }
    if (cut > maxCut + 1e-9) {
      throw new StyleError(`compute ${this.id} (orientorder/atom): cutoff ${cut} exceeds the pair style cutoff ${maxCut} (the maximum allowable value)`);
    }
    return cut;
  }

  /** Wigner-3j weighted real part of the triple product sum, from raw Y sums. */
  private wRaw(d: number): number {
    const idx = this.tripIdx[d];
    const cf = this.tripCoef[d];
    const re = this.yre[d];
    const im = this.yim[d];
    let acc = 0;
    for (let t = 0; t < cf.length; t++) {
      const a = re[idx[3 * t]], b = im[idx[3 * t]];
      const c = re[idx[3 * t + 1]], e = im[idx[3 * t + 1]];
      const f = re[idx[3 * t + 2]], g = im[idx[3 * t + 2]];
      acc += cf[t] * ((a * c - b * e) * f - (a * e + b * c) * g);
    }
    return acc;
  }

  /** Fills the associated-Legendre table P_m^l(cos theta) for m = 0..maxL. */
  private fillLegendre(cost: number, sint: number): void {
    const L = this.maxL;
    const P = this.ptab;
    const oddf = this.oddf;
    let sm = 1;
    for (let m = 0; m <= L; m++) {
      const pmm = (m & 1 ? -oddf[m] : oddf[m]) * sm;
      sm *= sint;
      const row = m * (L + 1);
      P[row + m] = pmm;
      if (m + 1 <= L) P[row + m + 1] = cost * (2 * m + 1) * pmm;
      for (let l = m + 2; l <= L; l++) {
        P[row + l] = ((2 * l - 1) * cost * P[row + l - 1] - (l + m - 1) * P[row + l - 2]) / (l - m);
      }
    }
  }

  protected computePeratom(): void {
    // ensures ghosts and neighbor structures exist for the current state
    this.sys.forces();
    const sys = this.sys;
    const s = sys.state;
    const nb = sys.nb;
    const cut = this.resolveCutoff();
    const nb2 = nb;
    if (nb2.cutghost > 0 && nb2.cutghost < cut + nb2.skin - 1e-9) {
      throw new StyleError(`compute ${this.id} (orientorder/atom): cutoff ${cut} exceeds the ghost cutoff ${nb2.cutghost} (force cutoff + skin); a larger cutoff needs comm_modify cutoff`);
    }
    const n = s.n;
    const nall = nb.nall;
    const xa = nb.xall;
    const cols = this.sizePeratomCols;
    const out = (this.arrayAtom = new Float64Array(cols * n));
    const cut2 = cut * cut;
    const nlv = this.degrees.length;
    const sel = this.nnn ?? 0;
    // grow scratch to hold every owned+ghost candidate
    if (this.r2a.length < nall) {
      const cap = nall + 64;
      this.r2a = new Float64Array(cap);
      this.dxa = new Float64Array(cap);
      this.dya = new Float64Array(cap);
      this.dza = new Float64Array(cap);
      this.order = new Int32Array(cap);
    }
    const r2a = this.r2a, dxa = this.dxa, dya = this.dya, dza = this.dza, order = this.order;
    const degrees = this.degrees;
    for (let i = 0; i < n; i++) {
      if (!(s.mask[i] & this.groupBit)) continue;
      const xi = xa[3 * i], yi = xa[3 * i + 1], zi = xa[3 * i + 2];
      let cnt = 0;
      for (let k = 0; k < nall; k++) {
        if (k === i) continue;
        const dx = xa[3 * k] - xi, dy = xa[3 * k + 1] - yi, dz = xa[3 * k + 2] - zi;
        const r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cut2) continue;
        r2a[cnt] = r2; dxa[cnt] = dx; dya[cnt] = dy; dza[cnt] = dz;
        cnt++;
      }
      if (cnt === 0) continue;
      if (sel > 0) {
        if (cnt < sel) continue; // fewer than nnn neighbors within the cutoff
        for (let t = 0; t < cnt; t++) order[t] = t;
        for (let t = 0; t < sel; t++) {
          let best = t;
          for (let u = t + 1; u < cnt; u++) if (r2a[order[u]] < r2a[order[best]]) best = u;
          const tmp = order[t]; order[t] = order[best]; order[best] = tmp;
        }
      } else {
        for (let t = 0; t < cnt; t++) order[t] = t;
      }
      const nsel = sel > 0 ? sel : cnt;
      const inv = 1 / nsel;
      for (let d = 0; d < nlv; d++) {
        this.yre[d].fill(0);
        this.yim[d].fill(0);
      }
      for (let t = 0; t < nsel; t++) {
        const k = order[t];
        const dx = dxa[k], dy = dya[k], dz = dza[k];
        const r = Math.sqrt(r2a[k]);
        let cost = dz / r;
        if (cost > 1) cost = 1; else if (cost < -1) cost = -1;
        const sint = Math.sqrt(Math.max(0, 1 - cost * cost));
        const phi = Math.atan2(dy, dx);
        this.fillLegendre(cost, sint);
        const P = this.ptab;
        const L = this.maxL;
        for (let d = 0; d < nlv; d++) {
          const l = degrees[d];
          const nm = this.norms[d];
          const re = this.yre[d];
          const im = this.yim[d];
          for (let m = 0; m <= l; m++) {
            const A = nm[m] * P[m * (L + 1) + l];
            if (A === 0) continue;
            const c = Math.cos(m * phi);
            const sn = Math.sin(m * phi);
            re[m + l] += A * c;
            im[m + l] += A * sn;
            if (m > 0) {
              const sgn = m & 1 ? -1 : 1;
              re[l - m] += sgn * A * c;
              im[l - m] -= sgn * A * sn;
            }
          }
        }
      }
      const row = cols * i;
      const inv3 = inv * inv * inv;
      const wOff = nlv;
      const hatOff = nlv + (this.wl ? nlv : 0);
      const compOff = hatOff + (this.wlHat ? nlv : 0);
      for (let d = 0; d < nlv; d++) {
        const l = degrees[d];
        const re = this.yre[d];
        const im = this.yim[d];
        let norm2 = 0;
        for (let t = 0; t < 2 * l + 1; t++) norm2 += re[t] * re[t] + im[t] * im[t];
        // Q_l = sqrt(4 pi/(2l+1) sum |Ybar|^2), Ybar = raw/nsel
        out[row + d] = Math.sqrt(((4 * Math.PI) / (2 * l + 1)) * norm2) * inv;
        const wr = this.wRaw(d);
        if (this.wl) out[row + wOff + d] = (wr * inv3) / Math.sqrt(2 * l + 1);
        if (this.wlHat) out[row + hatOff + d] = norm2 > 0 ? wr / Math.pow(norm2, 1.5) : 0;
        if (this.componentsDeg === l && norm2 > 0) {
          const sc = 1 / Math.sqrt(norm2);
          for (let m = 0; m <= 2 * l; m++) {
            out[row + compOff + 2 * m] = re[m] * sc;
            out[row + compOff + 2 * m + 1] = im[m] * sc;
          }
        }
      }
    }
  }
}
