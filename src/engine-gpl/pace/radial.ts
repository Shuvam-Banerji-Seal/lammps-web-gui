/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use: ML-PACE/ace-evaluator/ace_radial.cpp and ace_radial.h of
 * lammps-user-pace v.2025.12.4.p1 (the sources checked against LAMMPS's CMake SHA256). The LAMMPS package that uses them is
 * src/ML-PACE/pair_pace.cpp at commit b332f28. Authors of the library (from its headers): Yury Lysogorskiy, Cas van der
 * Oord, Anton Bochkarev, Sarath Menon, Matteo Rinaldi, Thomas Hammerschmidt, Matous Mrovec, Aidan Thompson, Gabor Csanyi,
 * Christoph Ortner, Ralf Drautz (ICAMS, Ruhr-University Bochum). The library is dual GPL-2 / GPL-3; it is used here under
 * GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Radial basis functions g_k(r), R_nl(r) and the hard-core repulsion, their Chebyshev recursion, and the cubic spline
 * look-up tables (SplineInterpolator) that the evaluator reads. The arithmetic follows the C++ operation order; the C
 * library calls are the exact ports in ../libm (pow, exp, sin/cos as glibc computes them).
 */
import { exp as libmExp } from '../libm/exp';
import { pow as libmPow } from '../libm/pow';
import { sin as libmSin, cos as libmCos } from '../libm/sincos';

/** pi as written in ace_radial.cpp (const DOUBLE_TYPE pi = 3.14159265358979323846264338327950288419). */
export const PI = 3.14159265358979323846264338327950288419;

/** ACEZBL::K is unused by the density inner cutoff; only the radial basis and the core repulsion are needed here. */

/** cutoff_func_poly (ace_radial.cpp): the polynomial inner cutoff, descending from r_in-delta_in to r_in. */
export const cutoffFuncPoly = (r: number, rIn: number, deltaIn: number): [number, number] => {
  if (r <= rIn - deltaIn) return [1, 0];
  if (r >= rIn) return [0, 0];
  const x = 1 - 2 * (1 + (r - rIn) / deltaIn);
  const fc = 0.5 + 7.5 / 2. * (x / 4. - libmPow(x, 3) / 6. + libmPow(x, 5) / 20.);
  const dfc = -7.5 / deltaIn * (0.25 - x * x / 2.0 + libmPow(x, 4) / 4.);
  return [fc, dfc];
};

/**
 * SplineInterpolator (ace_radial.cpp): cubic spline look-up table over [0, cutoff] with ntot bins. setup() calls func(r)
 * for every node (func writes the values and derivatives into the arrays passed to setup), as setupSplines does.
 */
export class SplineInterpolator {
  cutoff = 0;
  deltaSplineBins = 0.001;
  ntot = 10000;
  nlut = 10000;
  invrscalelookup = 1;
  rscalelookup = 1;
  numOfFunctions = 0;
  values = new Float64Array(0);
  derivatives = new Float64Array(0);
  /** lookupTable(n, func_id, idx) at ((n * numOfFunctions) + func_id) * 4 + idx */
  lookup = new Float64Array(0);

  setup(numOfFunctions: number, func: (r: number) => void, vals: Float64Array, dvals: Float64Array,
    deltaSplineBins: number, cutoff: number): void {
    this.deltaSplineBins = deltaSplineBins;
    if (deltaSplineBins <= 0) throw new Error('deltaSplineBins should be positive');
    this.cutoff = cutoff;
    this.ntot = Math.trunc(cutoff / deltaSplineBins);
    this.numOfFunctions = numOfFunctions;
    this.values = new Float64Array(numOfFunctions);
    this.derivatives = new Float64Array(numOfFunctions);
    const f1g = new Float64Array(numOfFunctions);
    const f1gd1 = new Float64Array(numOfFunctions);
    this.nlut = this.ntot;
    this.rscalelookup = this.nlut / cutoff;
    this.invrscalelookup = 1.0 / this.rscalelookup;
    this.lookup = new Float64Array((this.ntot + 1) * numOfFunctions * 4);
    if (numOfFunctions === 0) return;
    const c = [0, 0, 0, 0];
    for (let n = this.nlut; n >= 1; n--) {
      const r = this.invrscalelookup * n;
      func(r);
      for (let id = 0; id < numOfFunctions; id++) {
        const f0 = vals[id];
        const f1 = f1g[id];
        const f0d1 = dvals[id] * this.invrscalelookup;
        const f1d1 = f1gd1[id];
        c[0] = f0;
        c[1] = f0d1;
        c[2] = 3.0 * (f1 - f0) - f1d1 - 2.0 * f0d1;
        c[3] = -2.0 * (f1 - f0) + f1d1 + f0d1;
        const base = (n * numOfFunctions + id) * 4;
        for (let idx = 0; idx <= 3; idx++) this.lookup[base + idx] = c[idx];
        f1g[id] = c[0];
        f1gd1[id] = c[1];
      }
    }
  }

  /** calcSplines(r): fills values and derivatives for all functions (no second derivatives are needed here). */
  calc(r: number): void {
    const x = r * this.rscalelookup;
    const nl = Math.floor(x);
    if (nl <= 0) throw new Error('Encountered very small distance. Stopping.');
    const nf = this.numOfFunctions;
    if (nl < this.nlut) {
      const wl = x - nl;
      const wl2 = wl * wl;
      const wl3 = wl2 * wl;
      const w2l1 = 2.0 * wl;
      const w3l2 = 3.0 * wl2;
      for (let id = 0; id < nf; id++) {
        const b = (nl * nf + id) * 4;
        const c0 = this.lookup[b], c1 = this.lookup[b + 1], c2 = this.lookup[b + 2], c3 = this.lookup[b + 3];
        this.values[id] = c0 + c1 * wl + c2 * wl2 + c3 * wl3;
        this.derivatives[id] = (c1 + c2 * w2l1 + c3 * w3l2) * this.rscalelookup;
      }
    } else {
      this.values.fill(0);
      this.derivatives.fill(0);
    }
  }
}

/** Per pair (mu_i, mu_j) splines and the look-up radial arrays of ACERadialFunctions. */
export class ACERadialFunctions {
  nelements: number;
  nradbase: number;
  lmax: number;
  nradial: number;
  deltaSplineBins: number;
  /** radbasename of each pair (one name in the .ace file, applied to every pair) */
  radbasename: string;
  /** "density" (the default of the .ace loader; no inner cutoff) */
  readonly innerCutoffType = 'density';
  elements: string[] = [];

  gr: Float64Array;
  dgr: Float64Array;
  /** fr(n, l) at n * (lmax + 1) + l */
  fr: Float64Array;
  dfr: Float64Array;
  cr = 0;
  dcr = 0;

  private cheb: Float64Array;
  private dcheb: Float64Array;
  private cheb2: Float64Array;

  /** index of the pair (mu_i, mu_j) = mu_i * nelements + mu_j */
  lambda: Float64Array;
  cut: Float64Array;
  dcut: Float64Array;
  cutIn: Float64Array;
  dcutIn: Float64Array;
  prehc: Float64Array;
  lambdahc: Float64Array;
  /** crad(mu_i, mu_j, n, l, k) at ((((mu_i * nel + mu_j) * nradial + n) * (lmax + 1) + l) * nradbase + k) */
  crad: Float64Array;

  splinesGk: SplineInterpolator[] = [];
  splinesRnl: SplineInterpolator[] = [];
  splinesHc: SplineInterpolator[] = [];

  constructor(nradb: number, lmax: number, nradial: number, deltaSplineBins: number, nelements: number, radbasename: string) {
    this.nelements = nelements;
    this.nradbase = nradb;
    this.lmax = lmax;
    this.nradial = nradial;
    this.deltaSplineBins = deltaSplineBins;
    this.radbasename = radbasename;
    const np = nelements * nelements;
    this.gr = new Float64Array(nradb);
    this.dgr = new Float64Array(nradb);
    this.fr = new Float64Array(nradial * (lmax + 1));
    this.dfr = new Float64Array(nradial * (lmax + 1));
    this.cheb = new Float64Array(nradb + 1);
    this.dcheb = new Float64Array(nradb + 1);
    this.cheb2 = new Float64Array(nradb + 1);
    this.lambda = new Float64Array(np).fill(1);
    this.cut = new Float64Array(np).fill(1);
    this.dcut = new Float64Array(np).fill(1);
    this.cutIn = new Float64Array(np).fill(0);
    this.dcutIn = new Float64Array(np).fill(1e-5);
    this.prehc = new Float64Array(np);
    this.lambdahc = new Float64Array(np).fill(1);
    this.crad = new Float64Array(np * nradial * (lmax + 1) * nradb);
  }

  cradIndex(pair: number, n: number, l: number, k: number): number {
    return ((pair * this.nradial + n) * (this.lmax + 1) + l) * this.nradbase + k;
  }

  /** calcCheb (ace_radial.cpp): Chebyshev polynomials of the first and second kind, and the derivative of the first. */
  private calcCheb(n: number, x: number): void {
    const twox = 2.0 * x;
    const cheb = this.cheb, cheb2 = this.cheb2, dcheb = this.dcheb;
    cheb[0] = 1.;
    dcheb[0] = 0.;
    cheb2[0] = 1.;
    if (this.nradbase >= 1) {
      cheb[1] = x;
      cheb2[1] = twox;
    }
    for (let m = 1; m <= n - 1; m++) {
      cheb[m + 1] = twox * cheb[m] - cheb[m - 1];
      cheb2[m + 1] = twox * cheb2[m] - cheb2[m - 1];
    }
    for (let m = 1; m <= n; m++) dcheb[m] = m * cheb2[m - 1];
  }

  /** radbase (ace_radial.cpp), inner cutoff of the "density" type (no multiplication). */
  private radbase(lam: number, cut: number, dcut: number, name: string, r: number, cutIn: number, dcutIn: number): void {
    const gr = this.gr, dgr = this.dgr;
    if (r <= cutIn - dcutIn || r >= cut) {
      gr.fill(0);
      dgr.fill(0);
      return;
    }
    if (name === 'ChebPow') this.chebPow(lam, cut, dcut, r);
    else if (name === 'ChebExpCos') this.chebExpCos(lam, cut, dcut, r);
    else if (name === 'ChebLinear') this.chebLinear(lam, cut, dcut, r);
    else throw new Error('Unknown radial basis function name: ' + name);
  }

  /** ChebExpCos: the exponential-scaled Chebyshev basis with a cosine envelope (ace_radial.cpp chebExpCos). */
  private chebExpCos(lam: number, cut: number, dcut: number, r: number): void {
    const gr = this.gr, dgr = this.dgr, n0 = this.nradbase;
    const y1 = libmExp(-lam * r / cut);
    const y2 = libmExp(-lam);
    let x = 1.0 - 2.0 * ((y1 - y2) / (1 - y2));
    let dx = 2 * (lam / cut) * (y1 / (1 - y2));
    this.calcCheb(n0, x);
    gr[0] = this.cheb[0];
    dgr[0] = this.dcheb[0] * dx;
    for (let n = 2; n <= n0; n++) {
      gr[n - 1] = 0.5 - 0.5 * this.cheb[n - 1];
      dgr[n - 1] = -0.5 * this.dcheb[n - 1] * dx;
    }
    const arg = PI * r / cut;
    const env = 0.5 * (1.0 + libmCos(arg));
    const denv = -0.5 * libmSin(arg) * PI / cut;
    for (let n = 0; n < n0; n++) {
      dgr[n] = gr[n] * denv + dgr[n] * env;
      gr[n] = gr[n] * env;
    }
    dx = cut - dcut;
    if (r > dx) {
      const arg2 = PI * (r - dx) / dcut;
      const fcut = 0.5 * (1.0 + libmCos(arg2));
      const dfcut = -0.5 * libmSin(arg2) * PI / dcut;
      for (let n = 0; n < n0; n++) {
        dgr[n] = gr[n] * dfcut + dgr[n] * fcut;
        gr[n] = gr[n] * fcut;
      }
    }
  }

  /** ChebPow: x = 2 (1 - (1 - r/rcut)^lam) - 1 (ace_radial.cpp chebPow). */
  private chebPow(lam: number, cut: number, _dcut: number, r: number): void {
    const gr = this.gr, dgr = this.dgr, n0 = this.nradbase;
    let y = 1.0 - r / cut;
    let dy = libmPow(y, lam - 1.0);
    y = dy * y;
    dy = -lam / cut * dy;
    const x = 2.0 * (1.0 - y) - 1.0;
    const dx = -2.0 * dy;
    this.calcCheb(n0, x);
    for (let n = 1; n <= n0; n++) {
      gr[n - 1] = 0.5 - 0.5 * this.cheb[n];
      dgr[n - 1] = -0.5 * this.dcheb[n] * dx;
    }
  }

  /** ChebLinear: x = 1 - r/rcut (ace_radial.cpp chebLinear). */
  private chebLinear(_lam: number, cut: number, _dcut: number, r: number): void {
    const gr = this.gr, dgr = this.dgr, n0 = this.nradbase;
    const x = 1.0 - r / cut;
    const dx = -1 / cut;
    this.calcCheb(n0, x);
    for (let n = 1; n <= n0; n++) {
      gr[n - 1] = 0.5 - 0.5 * this.cheb[n];
      dgr[n - 1] = -0.5 * this.dcheb[n] * dx;
    }
  }

  /** radfunc: R_nl = sum_k crad(mu_i, mu_j, n, l, k) g_k(r) (ace_radial.cpp radfunc). */
  private radfunc(pair: number): void {
    for (let n = 0; n < this.nradial; n++) {
      for (let l = 0; l <= this.lmax; l++) {
        let frval = 0.0;
        let dfrval = 0.0;
        for (let k = 0; k < this.nradbase; k++) {
          const c = this.crad[this.cradIndex(pair, n, l, k)];
          frval += c * this.gr[k];
          dfrval += c * this.dgr[k];
        }
        this.fr[n * (this.lmax + 1) + l] = frval;
        this.dfr[n * (this.lmax + 1) + l] = dfrval;
      }
    }
  }

  private allRadfunc(pair: number, mui: number, muj: number, r: number): void {
    this.radbase(this.lambda[pair], this.cut[pair], this.dcut[pair], this.radbasename, r,
      this.cutIn[pair], this.dcutIn[pair]);
    this.radfunc(pair);
    void mui; void muj;
  }

  /** radcore (density type): the hard-core repulsion pre exp(-lambda r^2) / r with a cosine envelope. */
  private radcore(r: number, pre: number, lambda: number, cutoff: number, out: Float64Array): void {
    pre = Math.abs(pre);
    lambda = Math.abs(lambda);
    const r2 = r * r;
    const lr2 = lambda * r2;
    if (lr2 < 50.0) {
      const y = libmExp(-lr2);
      let cr = pre * y / r;
      let dcr = -pre * y * (2.0 * lr2 + 1.0) / r2;
      const x0 = r / cutoff;
      const env = 0.5 * (1.0 + libmCos(PI * x0));
      const denv = -0.5 * libmSin(PI * x0) * PI / cutoff;
      dcr = cr * denv + dcr * env;
      cr = cr * env;
      out[0] = cr;
      out[1] = dcr;
    } else {
      out[0] = 0.0;
      out[1] = 0.0;
    }
  }

  /** setuplookupRadspline: the splines of every pair (ace_radial.cpp setuplookupRadspline, density branch). */
  setuplookupRadspline(): void {
    const nel = this.nelements;
    this.splinesGk = [];
    this.splinesRnl = [];
    this.splinesHc = [];
    for (let ei = 0; ei < nel; ei++) {
      for (let ej = 0; ej < nel; ej++) {
        const pair = ei * nel + ej;
        const lam = this.lambda[pair];
        const rCut = this.cut[pair];
        const drCut = this.dcut[pair];
        const rIn = this.cutIn[pair];
        const deltaIn = this.dcutIn[pair];
        const gk = new SplineInterpolator();
        gk.setup(this.nradbase, (r) => this.radbase(lam, rCut, drCut, this.radbasename, r, rIn, deltaIn),
          this.gr, this.dgr, this.deltaSplineBins, rCut);
        const rnl = new SplineInterpolator();
        rnl.setup(this.nradial * (this.lmax + 1), (r) => this.allRadfunc(pair, ei, ej, r),
          this.fr, this.dfr, this.deltaSplineBins, rCut);
        const hc = new SplineInterpolator();
        const pre = this.prehc[pair];
        const lamhc = this.lambdahc[pair];
        const crc = new Float64Array(1), dcrc = new Float64Array(1);
        const tmp = new Float64Array(2);
        hc.setup(1, (r) => {
          this.radcore(r, pre, lamhc, rCut, tmp);
          crc[0] = tmp[0];
          dcrc[0] = tmp[1];
        }, crc, dcrc, this.deltaSplineBins, rCut);
        this.splinesGk.push(gk);
        this.splinesRnl.push(rnl);
        this.splinesHc.push(hc);
      }
    }
  }

  /**
   * evaluate (ace_radial.cpp): g_k, R_nl and the core repulsion of one distance from the look-up tables, into
   * gr, dgr, fr, dfr, cr and dcr.
   */
  evaluate(r: number, muI: number, muJ: number): void {
    const pair = muI * this.nelements + muJ;
    const sgk = this.splinesGk[pair];
    sgk.calc(r);
    // gr / dgr are the spline's outputs (values and derivatives of the first nradbase functions)
    this.gr.set(sgk.values.subarray(0, this.nradbase));
    this.dgr.set(sgk.derivatives.subarray(0, this.nradbase));
    const srnl = this.splinesRnl[pair];
    srnl.calc(r);
    this.fr.set(srnl.values);
    this.dfr.set(srnl.derivatives);
    const shc = this.splinesHc[pair];
    shc.calc(r);
    this.cr = shc.values[0];
    this.dcr = shc.derivatives[0];
  }
}
