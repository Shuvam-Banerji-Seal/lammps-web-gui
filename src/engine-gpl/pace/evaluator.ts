/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use: ML-PACE/ace-evaluator/ace_recursive.cpp
 * (ACERecursiveEvaluator::compute_atom with the recursive and the product (non-recursive, Julia-style) branches) and
 * ace_complex.h (ACEComplex arithmetic), plus ace_abstract_basis.cpp (Fexp) of lammps-user-pace v.2025.12.4.p1 (checked
 * against LAMMPS's CMake SHA256). The LAMMPS package that calls it is src/ML-PACE/pair_pace.cpp at commit b332f28. Library
 * authors (from its headers): Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz and co-authors (ICAMS,
 * Ruhr-University Bochum). Used here under GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * compute_atom gives the atomic energy of atom i and, for every neighbour in the list, the force on that neighbour from
 * the atomic energy (neighbours_forces). The loops, the accumulation order and the complex arithmetic follow the C++ code.
 * ACEComplex is kept as two numbers: complex * real scales both parts, complex += real adds to the real part only.
 */
import type { CTildeBasis } from './cbasis';
import { ACEDAG, acejlformat, type ProductForm } from './dag';
import { pow as libmPow } from '../libm/pow';
import { exp as libmExp } from '../libm/exp';
import { innerCutoff } from './cbasis';

/** Fexp (ace_abstract_basis.cpp): the FinnisSinclair embedding F(rho) = sign (omega am + lambda g a). */
const fexp = (x: number, m: number): [number, number] => {
  const w = 1.e6;
  const eps = 1e-10;
  const lambda = libmPow(1.0 / w, m - 1.0);
  if (Math.abs(x) > eps) {
    const a = Math.abs(x);
    const am = libmPow(a, m);
    const w3x3 = libmPow(w * a, 3);
    const sign = (x < 0 || Object.is(x, -0)) ? -1 : 1;
    const g = w3x3 > 30.0 ? 0.0 : libmExp(-w3x3);
    const omg = 1.0 - g;
    const F = sign * (omg * am + lambda * g * a);
    const dg = -3.0 * w * w * w * a * a * g;
    const DF = m * libmPow(a, m - 1.0) * omg - am * dg + lambda * dg * a + lambda * g;
    return [F, DF];
  }
  return [lambda * x, lambda];
};

/** FexpShiftedScaled (ace_abstract_basis.cpp), the FinnisSinclairShiftedScaled type. */
const fexpShifted = (rho: number, mexp: number): [number, number] => {
  const eps = 1e-10;
  if (Math.abs(mexp - 1.0) < eps) return [rho, 1];
  const a = Math.abs(rho);
  const exprho = libmExp(-a);
  const nx = 1. / mexp;
  const xoff = libmPow(nx, (nx / (1.0 - nx))) * exprho;
  const yoff = libmPow(nx, (1 / (1.0 - nx))) * exprho;
  const sign = (rho < 0 || Object.is(rho, -0)) ? -1 : 1;
  const F = sign * (libmPow(xoff + a, mexp) - yoff);
  const DF = yoff + mexp * (-xoff + 1.0) * libmPow(xoff + a, mexp - 1.);
  return [F, DF];
};

/** The per-species computational data the evaluator reads. */
interface SpeciesData {
  product: ProductForm;
  dag: ACEDAG;
}

/** Y00 of the library (ace_spherical_cart.h). */
const Y00 = 1;

/** The ACE evaluator of pair_style pace (ACERecursiveEvaluator with the set_recursive flag). */
export class ACEPaceEvaluator {
  /** e_atom: energy of the current atom, including the core repulsion and E0 */
  eAtom = 0;
  /** ace_fcut: the embedding cutoff of the current atom */
  aceFcut = 1;
  /** neighbours_forces(jj, alpha) of the last compute_atom call, jnum * 3 */
  neighboursForces = new Float64Array(0);

  private readonly lmax: number;
  private readonly nm: number;
  private readonly nradb: number;
  private readonly nradi: number;
  private readonly nel: number;
  private readonly species: SpeciesData[] = [];
  /** element_type_mapping: LAMMPS type -> species (-1 for NULL) */
  private readonly typeMap: Int32Array;

  // per-atom arrays reused between calls
  private readonly Are: Float64Array;
  private readonly Aim: Float64Array;
  private readonly Arank1: Float64Array;
  private readonly weightsRe: Float64Array;
  private readonly weightsIm: Float64Array;
  private readonly weightsRank1: Float64Array;
  private readonly rhos: Float64Array;
  private readonly dFdrho: Float64Array;
  private cap = 0;
  private dgCache = new Float64Array(0);
  private drCache = new Float64Array(0);
  private rCache = new Float64Array(0);
  private yRe = new Float64Array(0);
  private yIm = new Float64Array(0);
  private dyCache = new Float64Array(0);
  private dcrCache = new Float64Array(0);

  constructor(readonly basis: CTildeBasis, readonly recursive: boolean, typeMap: Int32Array) {
    this.lmax = basis.lmax;
    this.nm = 2 * basis.lmax + 1;
    this.nradb = basis.nradbase;
    this.nradi = basis.nradmax;
    this.nel = basis.nelements;
    this.typeMap = typeMap;
    const ndens = basis.ndensitymax;
    for (let mu = 0; mu < this.nel; mu++) {
      const product = acejlformat(basis, mu);
      const dag = ACEDAG.build(product.Aspec, product.num1, product.AAspec, product.orders, product.coeffs,
        product.num2, product.maxorder, ndens);
      this.species.push({ product, dag });
    }
    const nA = this.nel * this.nradi * (this.lmax + 1) * this.nm;
    this.Are = new Float64Array(nA);
    this.Aim = new Float64Array(nA);
    this.Arank1 = new Float64Array(this.nel * this.nradb);
    this.weightsRe = new Float64Array(nA);
    this.weightsIm = new Float64Array(nA);
    this.weightsRank1 = new Float64Array(this.nel * this.nradb);
    this.rhos = new Float64Array(ndens);
    this.dFdrho = new Float64Array(ndens);
  }

  /** index into A / weights: (mu, n (0-based), l, m) with m in -l..l stored at m + l */
  private aIdx(mu: number, n: number, l: number, m: number): number {
    return ((mu * this.nradi + n) * (this.lmax + 1) + l) * this.nm + (m + this.lmax);
  }

  /** resize_neighbours_cache: the per-neighbour caches for jnum neighbours */
  private ensureCache(jnum: number): void {
    if (jnum <= this.cap) return;
    this.cap = jnum;
    const L1 = this.lmax + 1;
    this.dgCache = new Float64Array(jnum * this.nradb);
    this.drCache = new Float64Array(jnum * this.nradi * L1);
    this.rCache = new Float64Array(jnum * this.nradi * L1);
    this.yRe = new Float64Array(jnum * L1 * L1);
    this.yIm = new Float64Array(jnum * L1 * L1);
    this.dyCache = new Float64Array(jnum * L1 * L1 * 6);
    this.dcrCache = new Float64Array(jnum);
  }

  /**
   * compute_atom(i, x, type, jnum, jlist): x holds 3 coordinates per atom (owned and ghost), type the LAMMPS types,
   * jlist the neighbour indices of atom i. Fills eAtom, aceFcut and neighboursForces.
   */
  computeAtom(i: number, x: Float64Array, type: Int32Array, jnum: number, jlist: Int32Array, jlistOff: number): void {
    this.ensureCache(jnum);
    const basis = this.basis, rad = basis.radial, sph = basis.sph;
    const L1 = this.lmax + 1, nm = this.nm;
    const nradb = this.nradb, nradi = this.nradi;
    const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
    const muI = this.typeMap[type[i]];
    const totalRank1 = basis.totalBasisSizeRank1[muI];
    const rank1 = basis.basisRank1[muI];
    const ndensity = basis.embedding[muI].ndensity;
    const sp = this.species[muI];
    const product = sp.product;
    const dag = sp.dag;
    const num1 = dag.num1;

    this.Are.fill(0);
    this.Aim.fill(0);
    this.Arank1.fill(0);
    this.rhos.fill(0);
    this.dFdrho.fill(0);
    if (this.neighboursForces.length < 3 * jnum) this.neighboursForces = new Float64Array(3 * jnum);
    this.neighboursForces.fill(0, 0, 3 * jnum);
    this.weightsRe.fill(0);
    this.weightsIm.fill(0);
    this.weightsRank1.fill(0);

    // neighbours inside the cutoff, in list order
    const rNorms = new Float64Array(jnum);
    const invRNorms = new Float64Array(jnum);
    const rhats = new Float64Array(3 * jnum);
    const elements = new Int32Array(jnum);
    const nbrIndex = new Int32Array(jnum);
    let jnumActual = 0;
    for (let jj = 0; jj < jnum; jj++) {
      const j = jlist[jlistOff + jj];
      const xn = x[3 * j] - xtmp;
      const yn = x[3 * j + 1] - ytmp;
      const zn = x[3 * j + 2] - ztmp;
      const muJ = this.typeMap[type[j]];
      const currentCutoff = rad.cut[muI * this.nel + muJ];
      const rXyz = Math.sqrt(xn * xn + yn * yn + zn * zn);
      if (rXyz >= currentCutoff) continue;
      const invR = 1 / rXyz;
      rNorms[jnumActual] = rXyz;
      invRNorms[jnumActual] = invR;
      rhats[3 * jnumActual] = xn * invR;
      rhats[3 * jnumActual + 1] = yn * invR;
      rhats[3 * jnumActual + 2] = zn * invR;
      elements[jnumActual] = muJ;
      nbrIndex[jnumActual] = jj;
      jnumActual++;
    }

    let rhoCore = 0;
    for (let jj = 0; jj < jnumActual; jj++) {
      const rNorm = rNorms[jj];
      const muJ = elements[jj];
      const rh0 = rhats[3 * jj], rh1 = rhats[3 * jj + 1], rh2 = rhats[3 * jj + 2];
      rad.evaluate(rNorm, muI, muJ);
      sph.computeYlm(rh0, rh1, rh2, this.lmax);
      const gr = rad.gr, dgr = rad.dgr, fr = rad.fr, dfr = rad.dfr;
      for (let n = 0; n < nradb; n++) {
        const GR = gr[n];
        this.dgCache[jj * nradb + n] = dgr[n];
        this.Arank1[muJ * nradb + n] += GR * Y00;
      }
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= this.lmax; l++) {
          const R = fr[n * L1 + l];
          const rc = (jj * nradi + n) * L1 + l;
          this.drCache[rc] = dfr[n * L1 + l];
          this.rCache[rc] = R;
          for (let m = 0; m <= l; m++) {
            const yr = sph.ylmRe[l * L1 + m], yi = sph.ylmIm[l * L1 + m];
            const a = this.aIdx(muJ, n, l, m);
            this.Are[a] += yr * R;
            this.Aim[a] += yi * R;
            const yc = (jj * L1 + l) * L1 + m;
            this.yRe[yc] = yr;
            this.yIm[yc] = yi;
            const d = sph.dylm;
            for (let q = 0; q < 6; q++) this.dyCache[yc * 6 + q] = d[(l * L1 + m) * 6 + q];
          }
        }
      }
      rhoCore += rad.cr;
      this.dcrCache[jj] = rad.dcr;
    }

    // m < 0 by conjugation (A(l, -m) = conj(A(l, m)) * (-1)^m)
    for (let mu = 0; mu < this.nel; mu++) {
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= this.lmax; l++) {
          for (let m = 1; m <= l; m++) {
            const factor = m % 2 === 0 ? 1 : -1;
            const src = this.aIdx(mu, n, l, m), dst = this.aIdx(mu, n, l, -m);
            this.Are[dst] = this.Are[src] * factor;
            this.Aim[dst] = (-this.Aim[src]) * factor;
          }
        }
      }
    }

    // rank-1 functions
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      const Acur = this.Arank1[func.mus[0] * nradb + (func.ns[0] - 1)];
      for (let p = 0; p < ndensity; p++) this.rhos[p] += func.ctildes[p] * Acur;
    }

    // the one-particle A values of the graph: AAbuf(idx) = A(Aspec(idx))
    const AAre = dag.AAbufRe, AAim = dag.AAbufIm;
    const nodes = dag.nodes, dcoef = dag.coeffs;
    const asp = product.Aspec;
    for (let idx = 0; idx < num1; idx++) {
      const a = this.aIdx(asp[4 * idx], asp[4 * idx + 1] - 1, asp[4 * idx + 2], asp[4 * idx + 3]);
      AAre[idx] = this.Are[a];
      AAim[idx] = this.Aim[a];
    }
    const rhos = this.rhos;
    if (this.recursive) {
      // forward pass: interior products (saved in the buffer) and leaves
      let idxNodes = 0;
      let idxCoefs = 0;
      const num2Int = dag.num2Int, num2Leaf = dag.num2Leaf;
      for (let idx = num1; idx < num1 + num2Int; idx++) {
        const i1 = nodes[idxNodes++];
        const i2 = nodes[idxNodes++];
        const cRe = AAre[i1] * AAre[i2] - AAim[i1] * AAim[i2];
        const cIm = AAre[i1] * AAim[i2] + AAim[i1] * AAre[i2];
        AAre[idx] = cRe;
        AAim[idx] = cIm;
        for (let p = 0; p < ndensity; p++, idxCoefs++) rhos[p] += cRe * dcoef[idxCoefs];
      }
      for (let q = 0; q < num2Leaf; q++) {
        const i1 = nodes[idxNodes++];
        const i2 = nodes[idxNodes++];
        const cRe = AAre[i1] * AAre[i2] - AAim[i1] * AAim[i2];
        for (let p = 0; p < ndensity; p++, idxCoefs++) rhos[p] += cRe * dcoef[idxCoefs];
      }
    } else {
      // product form: AA = prod of the A's of each product, coefficients on its real part
      const flat = product.AAspecFlat, orders = product.orders, coeffs = product.coeffs;
      let idxSpec = 0;
      let idxCoefs = 0;
      for (let iAA = 0; iAA < product.num2; iAA++) {
        let cRe = 1, cIm = 0;
        const order = orders[iAA];
        for (let r = 0; r < order; r++, idxSpec++) {
          const k = flat[idxSpec];
          const nRe = cRe * AAre[k] - cIm * AAim[k];
          const nIm = cRe * AAim[k] + cIm * AAre[k];
          cRe = nRe;
          cIm = nIm;
        }
        for (let p = 0; p < ndensity; p++, idxCoefs++) rhos[p] += cRe * coeffs[idxCoefs];
      }
    }

    // embedding energy and the core cutoff
    const emb = basis.embedding[muI];
    const rhoCut = emb.rhoCoreCutoff, drhoCut = emb.drhoCoreCutoff;
    let evdwl = 0;
    const dF = this.dFdrho;
    for (let p = 0; p < ndensity; p++) {
      const wpre = emb.FS[p * 2 + 0];
      const mexp = emb.FS[p * 2 + 1];
      if (emb.npoti !== 'FinnisSinclair' && emb.npoti !== 'FinnisSinclairShiftedScaled') {
        throw new Error(`pair_style pace: embedding-function ${emb.npoti} is not supported`);
      }
      const [F, DF] = emb.npoti === 'FinnisSinclair' ? fexp(rhos[p], mexp) : fexpShifted(rhos[p], mexp);
      evdwl += F * wpre;
      dF[p] = DF * wpre;
    }
    const [fcut, dfcut] = innerCutoff(rhoCore, rhoCut, drhoCut);
    const evdwlCut0 = evdwl * fcut + rhoCore;
    const dFdrhoCore = evdwl * dfcut + 1;
    for (let p = 0; p < ndensity; p++) dF[p] *= fcut;
    const evdwlCut = evdwlCut0 + basis.E0vals[muI];

    // weights (rank 1, then the graph backward pass)
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      for (let p = 0; p < ndensity; p++) {
        this.weightsRank1[func.mus[0] * nradb + (func.ns[0] - 1)] += dF[p] * func.ctildes[p];
      }
    }
    const wRe = dag.wRe, wIm = dag.wIm;
    if (this.recursive) {
      const num2Int = dag.num2Int, num2Leaf = dag.num2Leaf;
      wRe.fill(0);
      wIm.fill(0);
      let idxNodes = 2 * (num2Int + num2Leaf) - 1;
      let idxCoefs = ndensity * (num2Int + num2Leaf) - 1;
      for (let idx = num1 + num2Int + num2Leaf - 1; idx >= num1; idx--) {
        const i2 = nodes[idxNodes--];
        const i1 = nodes[idxNodes--];
        const a1Re = AAre[i1], a1Im = AAim[i1];
        const a2Re = AAre[i2], a2Im = AAim[i2];
        let cRe = wRe[idx], cIm = wIm[idx];
        for (let p = ndensity - 1; p >= 0; p--, idxCoefs--) cRe += dF[p] * dcoef[idxCoefs];
        // w(i1) += wcur * AA2 ; w(i2) += wcur * AA1
        wRe[i1] += cRe * a2Re - cIm * a2Im;
        wIm[i1] += cRe * a2Im + cIm * a2Re;
        wRe[i2] += cRe * a1Re - cIm * a1Im;
        wIm[i2] += cRe * a1Im + cIm * a1Re;
      }
    } else {
      wRe.fill(0);
      wIm.fill(0);
      const flat = product.AAspecFlat, orders = product.orders, coeffs = product.coeffs;
      const spec = new Int32Array(product.maxorder + 1);
      const Acache = new Float64Array(2 * (product.maxorder + 1));
      const dARe = new Float64Array(product.maxorder + 1), dAIm = new Float64Array(product.maxorder + 1);
      let idxSpec = 0;
      let idxCoefs = 0;
      for (let iAA = 0; iAA < product.num2; iAA++) {
        const order = orders[iAA];
        let thRe = 0, thIm = 0;
        for (let p = 0; p < ndensity; p++, idxCoefs++) thRe += dF[p] * coeffs[idxCoefs];
        dARe[0] = 1.0;
        dAIm[0] = 0.0;
        let AAfRe = 1.0, AAfIm = 0.0;
        for (let t = 0; t < order - 1; t++, idxSpec++) {
          spec[t] = flat[idxSpec];
          Acache[2 * t] = AAre[spec[t]];
          Acache[2 * t + 1] = AAim[spec[t]];
          const nRe = AAfRe * Acache[2 * t] - AAfIm * Acache[2 * t + 1];
          const nIm = AAfRe * Acache[2 * t + 1] + AAfIm * Acache[2 * t];
          AAfRe = nRe;
          AAfIm = nIm;
          dARe[t + 1] = AAfRe;
          dAIm[t + 1] = AAfIm;
        }
        spec[order - 1] = flat[idxSpec];
        idxSpec++;
        Acache[2 * (order - 1)] = AAre[spec[order - 1]];
        Acache[2 * (order - 1) + 1] = AAim[spec[order - 1]];
        let AAbRe = 1.0, AAbIm = 0.0;
        for (let t = order - 1; t >= 1; t--) {
          const nRe = AAbRe * Acache[2 * t] - AAbIm * Acache[2 * t + 1];
          const nIm = AAbRe * Acache[2 * t + 1] + AAbIm * Acache[2 * t];
          AAbRe = nRe;
          AAbIm = nIm;
          // dA(t-1) *= AAb
          const oRe = dARe[t - 1], oIm = dAIm[t - 1];
          dARe[t - 1] = oRe * AAbRe - oIm * AAbIm;
          dAIm[t - 1] = oRe * AAbIm + oIm * AAbRe;
          // w(spec(t)) += theta * dA(t)
          const s = spec[t];
          wRe[s] += thRe * dARe[t] - thIm * dAIm[t];
          wIm[s] += thRe * dAIm[t] + thIm * dARe[t];
        }
        const s0 = spec[0];
        wRe[s0] += thRe * dARe[0] - thIm * dAIm[0];
        wIm[s0] += thRe * dAIm[0] + thIm * dARe[0];
      }
    }

    // write the graph weights back into the weights of (mu, n, l, m)
    for (let idx = 0; idx < num1; idx++) {
      const mu = asp[4 * idx], n = asp[4 * idx + 1] - 1, l = asp[4 * idx + 2], m = asp[4 * idx + 3];
      if (m >= 0) {
        const a = this.aIdx(mu, n, l, m);
        this.weightsRe[a] += wRe[idx];
        this.weightsIm[a] += wIm[idx];
      } else {
        const factor = (m % 2 === 0 ? 1 : -1);
        const a = this.aIdx(mu, n, l, -m);
        this.weightsRe[a] += wRe[idx] * factor;
        this.weightsIm[a] += (-wIm[idx]) * factor;
      }
    }

    // forces on the neighbours
    for (let jj = 0; jj < jnumActual; jj++) {
      const muJ = elements[jj];
      const rh0 = rhats[3 * jj], rh1 = rhats[3 * jj + 1], rh2 = rhats[3 * jj + 2];
      const invR = invRNorms[jj];
      let f0 = 0, f1 = 0, f2 = 0;
      for (let n = 0; n < nradb; n++) {
        const wr = this.weightsRank1[muJ * nradb + n];
        if (wr === 0) continue;
        let DGR = this.dgCache[jj * nradb + n] * Y00;
        DGR *= wr;
        f0 += DGR * rh0;
        f1 += DGR * rh1;
        f2 += DGR * rh2;
      }
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= this.lmax; l++) {
          const rc = (jj * nradi + n) * L1 + l;
          const Ro = this.rCache[rc] * invR;
          const DR = this.drCache[rc];
          for (let m = 0; m <= l; m++) {
            const a = this.aIdx(muJ, n, l, m);
            const wRe2 = this.weightsRe[a], wIm2 = this.weightsIm[a];
            if (wRe2 === 0 && wIm2 === 0) continue;
            const yc = (jj * L1 + l) * L1 + m;
            const yr = this.yRe[yc], yi = this.yIm[yc];
            // Y_DR = Y * DR (complex * real)
            const ydRe = yr * DR, ydIm = yi * DR;
            const d = this.dyCache;
            const o = yc * 6;
            // grad_k = Y_DR * r_hat_k + DY_k * R_over_r
            const g0Re = ydRe * rh0 + d[o + 0] * Ro, g0Im = ydIm * rh0 + d[o + 1] * Ro;
            const g1Re = ydRe * rh1 + d[o + 2] * Ro, g1Im = ydIm * rh1 + d[o + 3] * Ro;
            const g2Re = ydRe * rh2 + d[o + 4] * Ro, g2Im = ydIm * rh2 + d[o + 5] * Ro;
            f0 += wRe2 * g0Re - wIm2 * g0Im;
            f1 += wRe2 * g1Re - wIm2 * g1Im;
            f2 += wRe2 * g2Re - wIm2 * g2Im;
          }
        }
      }
      const DCR = this.dcrCache[jj];
      f0 += dFdrhoCore * DCR * rh0;
      f1 += dFdrhoCore * DCR * rh1;
      f2 += dFdrhoCore * DCR * rh2;
      const nb = nbrIndex[jj];
      this.neighboursForces[3 * nb] = f0;
      this.neighboursForces[3 * nb + 1] = f1;
      this.neighboursForces[3 * nb + 2] = f2;
    }
    this.eAtom = evdwlCut;
    this.aceFcut = fcut;
  }
}
