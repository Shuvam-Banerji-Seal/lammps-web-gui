/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28) and the ML-PACE library (ACE evaluator), GPL-2.0 use:
 * ML-PACE/ace-evaluator/ace_evaluator.cpp (ACECTildeEvaluator::compute_atom, built with EXTRA_C_PROJECTIONS and
 * COMPUTE_B_GRAD as the library's CMakeLists.txt sets them) and ace_complex.h of lammps-user-pace v.2025.12.4.p1 (checked
 * against LAMMPS's CMake SHA256). The LAMMPS command that calls it is src/ML-PACE/compute_pace.cpp at commit b332f28.
 * Library authors (from its headers): Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz and co-authors (ICAMS,
 * Ruhr-University Bochum). Used here under GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * The parts of compute_atom that compute pace reads: the C-tilde projections of atom i (the 0th-density value of each
 * basis function, rank 1 and higher rank) and their gradients with respect to each neighbour position
 * (neighbours_dB). The atomic base A, the products over the ms combinations, weights_rank1_dB and weights_dB follow the
 * C++ loops and operation order; the energy, the embedding and the forces of compute_atom are not needed here.
 */
import type { CTildeBasis } from './cbasis';

/** Y00 of the library (ace_spherical_cart.h): 1, not 1/sqrt(4 pi). */
const Y00 = 1;

export class AceProjections {
  private readonly lmax: number;
  private readonly nm: number;
  private readonly nradb: number;
  private readonly nradi: number;
  private readonly nel: number;
  private readonly Are: Float64Array;
  private readonly Aim: Float64Array;
  private readonly Arank1: Float64Array;
  /** projections of the last computeAtom: rank-1 functions first, then the higher ranks (species of atom i). */
  projections = new Float64Array(0);
  /** neighbours_dB(func, jj, k) at (func * jnum + jj) * 3 + k, for the jnum entries of the neighbour list given. */
  neighboursDB = new Float64Array(0);
  private jnumLast = 0;
  private wdbRe = new Float64Array(0);
  private wdbIm = new Float64Array(0);

  constructor(readonly basis: CTildeBasis, private readonly typeMap: Int32Array) {
    this.lmax = basis.lmax;
    this.nm = 2 * basis.lmax + 1;
    this.nradb = basis.nradbase;
    this.nradi = basis.nradmax;
    this.nel = basis.nelements;
    const nA = this.nel * this.nradi * (this.lmax + 1) * this.nm;
    this.Are = new Float64Array(nA);
    this.Aim = new Float64Array(nA);
    this.Arank1 = new Float64Array(this.nel * this.nradb);
  }

  /** The stride of neighboursDB per function (the jnum of the last computeAtom). */
  get jnum(): number {
    return this.jnumLast;
  }

  private aIdx(mu: number, n: number, l: number, m: number): number {
    return ((mu * this.nradi + n) * (this.lmax + 1) + l) * this.nm + (m + this.lmax);
  }

  /** compute_atom(i, x, type, jnum, jlist) with compute_projections and compute_b_grad set. */
  computeAtom(i: number, x: Float64Array, type: Int32Array, jnum: number, jlist: Int32Array): void {
    const basis = this.basis, rad = basis.radial, sph = basis.sph;
    const L1 = this.lmax + 1, lmax = this.lmax;
    const nradb = this.nradb, nradi = this.nradi;
    const muI = this.typeMap[type[i]];
    const totalRank1 = basis.totalBasisSizeRank1[muI];
    const totalBasis = basis.totalBasisSize[muI];
    const rank1 = basis.basisRank1[muI];
    const funcs = basis.basis[muI];
    const ndensity = basis.embedding[muI].ndensity;
    const nfunc = totalRank1 + totalBasis;
    this.projections = new Float64Array(nfunc);
    this.neighboursDB = new Float64Array(nfunc * jnum * 3);
    this.jnumLast = jnum;
    this.Are.fill(0);
    this.Aim.fill(0);
    this.Arank1.fill(0);

    // neighbours inside the cutoff of (mu_i, mu_j), in list order
    const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
    const invRNorms = new Float64Array(jnum);
    const rNorms = new Float64Array(jnum);
    const rhats = new Float64Array(3 * jnum);
    const elements = new Int32Array(jnum);
    const nbrIndex = new Int32Array(jnum);
    let jnumActual = 0;
    for (let jj = 0; jj < jnum; jj++) {
      const j = jlist[jj];
      const xn = x[3 * j] - xtmp, yn = x[3 * j + 1] - ytmp, zn = x[3 * j + 2] - ztmp;
      const muJ = this.typeMap[type[j]];
      const rXyz = Math.sqrt(xn * xn + yn * yn + zn * zn);
      if (rXyz >= rad.cut[muI * this.nel + muJ]) continue;
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

    // ALGORITHM 1: the atomic base A and the per-neighbour caches
    const dg = new Float64Array(jnumActual * nradb);
    const rc = new Float64Array(jnumActual * nradi * L1);
    const drc = new Float64Array(jnumActual * nradi * L1);
    const yRe = new Float64Array(jnumActual * L1 * L1), yIm = new Float64Array(jnumActual * L1 * L1);
    const dy = new Float64Array(jnumActual * L1 * L1 * 6);
    for (let jj = 0; jj < jnumActual; jj++) {
      const muJ = elements[jj];
      rad.evaluate(rNorms[jj], muI, muJ);
      sph.computeYlm(rhats[3 * jj], rhats[3 * jj + 1], rhats[3 * jj + 2], lmax);
      for (let n = 0; n < nradb; n++) {
        dg[jj * nradb + n] = rad.dgr[n];
        this.Arank1[muJ * nradb + n] += rad.gr[n] * Y00;
      }
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= lmax; l++) {
          const R = rad.fr[n * L1 + l];
          const c = (jj * nradi + n) * L1 + l;
          drc[c] = rad.dfr[n * L1 + l];
          rc[c] = R;
          for (let m = 0; m <= l; m++) {
            const yr = sph.ylmRe[l * L1 + m], yi = sph.ylmIm[l * L1 + m];
            const a = this.aIdx(muJ, n, l, m);
            this.Are[a] += R * yr;
            this.Aim[a] += R * yi;
            const yc = (jj * L1 + l) * L1 + m;
            yRe[yc] = yr;
            yIm[yc] = yi;
            for (let q = 0; q < 6; q++) dy[yc * 6 + q] = sph.dylm[(l * L1 + m) * 6 + q];
          }
        }
      }
    }
    // A(l, -m) = conj(A(l, m)) (-1)^m
    for (let mu = 0; mu < this.nel; mu++) {
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= lmax; l++) {
          for (let m = 1; m <= l; m++) {
            const factor = m % 2 === 0 ? 1 : -1;
            const s = this.aIdx(mu, n, l, m), d = this.aIdx(mu, n, l, -m);
            this.Are[d] = this.Are[s] * factor;
            this.Aim[d] = -this.Aim[s] * factor;
          }
        }
      }
    }

    // ALGORITHM 2: the projections; rank 1, then the products of the higher ranks with their dB
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      this.projections[f] += func.ctildes[0] * this.Arank1[func.mus[0] * nradb + (func.ns[0] - 1)];
    }
    // weights_dB(func, mu, n, l, m) of the higher ranks
    const nA = this.nel * nradi * L1 * this.nm;
    if (this.wdbRe.length < totalBasis * nA) {
      this.wdbRe = new Float64Array(totalBasis * nA);
      this.wdbIm = new Float64Array(totalBasis * nA);
    }
    const wdbRe = this.wdbRe, wdbIm = this.wdbIm;
    wdbRe.fill(0, 0, totalBasis * nA);
    wdbIm.fill(0, 0, totalBasis * nA);
    const fwdRe = new Float64Array(basis.rankmax + 1), fwdIm = new Float64Array(basis.rankmax + 1);
    const bwdRe = new Float64Array(basis.rankmax + 1), bwdIm = new Float64Array(basis.rankmax + 1);
    const acRe = new Float64Array(basis.rankmax), acIm = new Float64Array(basis.rankmax);
    for (let f = 0; f < totalBasis; f++) {
      const func = funcs[f];
      const rank = func.rank, r = rank - 1;
      const mus = func.mus, ns = func.ns, ls = func.ls;
      for (let msInd = 0; msInd < func.numMsCombs; msInd++) {
        const off = msInd * rank;
        fwdRe[0] = 1; fwdIm[0] = 0;
        bwdRe[r] = 1; bwdIm[r] = 0;
        for (let t = 0; t < rank; t++) {
          const a = this.aIdx(mus[t], ns[t] - 1, ls[t], func.msCombs[off + t]);
          acRe[t] = this.Are[a]; acIm[t] = this.Aim[a];
          fwdRe[t + 1] = fwdRe[t] * acRe[t] - fwdIm[t] * acIm[t];
          fwdIm[t + 1] = fwdRe[t] * acIm[t] + fwdIm[t] * acRe[t];
        }
        const Bre = fwdRe[rank];
        for (let t = r; t >= 1; t--) {
          bwdRe[t - 1] = bwdRe[t] * acRe[t] - bwdIm[t] * acIm[t];
          bwdIm[t - 1] = bwdRe[t] * acIm[t] + bwdIm[t] * acRe[t];
        }
        // projections(total_basis_size_rank1 + func_ind) += B.real_part_product(ctildes[ms_ind * ndensity])
        this.projections[totalRank1 + f] += Bre * func.ctildes[msInd * ndensity];
        // ALGORITHM 3 (dB part): theta_dB = 0.5 ctilde of the 0th density
        const thetaDB = func.ctildes[msInd * ndensity + 0] * 0.5;
        for (let t = 0; t < rank; t++) {
          const dbRe = fwdRe[t] * bwdRe[t] - fwdIm[t] * bwdIm[t];
          const dbIm = fwdRe[t] * bwdIm[t] + fwdIm[t] * bwdRe[t];
          const mt = func.msCombs[off + t];
          const factor = mt % 2 === 0 ? 1 : -1;
          const p = f * nA + this.aIdx(mus[t], ns[t] - 1, ls[t], mt);
          wdbRe[p] += thetaDB * dbRe;
          wdbIm[p] += thetaDB * dbIm;
          const q = f * nA + this.aIdx(mus[t], ns[t] - 1, ls[t], -mt);
          wdbRe[q] += (thetaDB * dbRe) * factor;
          wdbIm[q] += (thetaDB * -dbIm) * factor;
        }
      }
    }
    // weights_rank1_dB(func, mu, n) += ctildes[0]
    const w1 = new Float64Array(totalRank1 * this.nel * nradb);
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      w1[(f * this.nel + func.mus[0]) * nradb + (func.ns[0] - 1)] += func.ctildes[0];
    }
    // the nonzero weights_dB(func, mu, n, l, m >= 0) entries of each function and neighbour species, in the C++ loop order
    // (n, l, m): the radial cache offset n (L1) + l, the Y cache offset l L1 + m and the weight (doubled for m > 0)
    const nel = this.nel;
    const start = new Int32Array(totalBasis * nel + 1);
    const eR: number[] = [], eY: number[] = [], eWr: number[] = [], eWi: number[] = [];
    for (let f = 0; f < totalBasis; f++) {
      for (let mu = 0; mu < nel; mu++) {
        start[f * nel + mu] = eR.length;
        for (let n = 0; n < nradi; n++) {
          for (let l = 0; l <= lmax; l++) {
            for (let m = 0; m <= l; m++) {
              const a = f * nA + this.aIdx(mu, n, l, m);
              let wr = wdbRe[a], wi = wdbIm[a];
              if (wr === 0 && wi === 0) continue;
              if (m > 0) { wr *= 2; wi *= 2; }
              eR.push(n * L1 + l);
              eY.push(l * L1 + m);
              eWr.push(wr);
              eWi.push(wi);
            }
          }
        }
      }
    }
    start[totalBasis * nel] = eR.length;
    const entR = Int32Array.from(eR), entY = Int32Array.from(eY);
    const entWr = Float64Array.from(eWr), entWi = Float64Array.from(eWi);

    // the gradients for each neighbour
    const nDB = this.neighboursDB;
    for (let jj = 0; jj < jnumActual; jj++) {
      const muJ = elements[jj];
      const rh0 = rhats[3 * jj], rh1 = rhats[3 * jj + 1], rh2 = rhats[3 * jj + 2];
      const invR = invRNorms[jj];
      const nb = nbrIndex[jj];
      for (let f = 0; f < totalRank1; f++) {
        const n = rank1[f].ns[0] - 1;
        let DGR = dg[jj * nradb + n] * Y00;
        DGR *= w1[(f * nel + muJ) * nradb + n];
        const o = (f * jnum + nb) * 3;
        nDB[o] += DGR * rh0;
        nDB[o + 1] += DGR * rh1;
        nDB[o + 2] += DGR * rh2;
      }
      // rank > 1: grad_phi_nlm = Y DR r_hat + DY R / r, and the real part of w_dB times it
      const rBase = jj * nradi * L1, yBase = jj * L1 * L1;
      for (let f = 0; f < totalBasis; f++) {
        const k1 = start[f * nel + muJ + 1];
        const o = ((totalRank1 + f) * jnum + nb) * 3;
        let a0 = nDB[o], a1 = nDB[o + 1], a2 = nDB[o + 2];
        for (let k = start[f * nel + muJ]; k < k1; k++) {
          const c = rBase + entR[k];
          const Ro = rc[c] * invR;
          const DR = drc[c];
          const yc = yBase + entY[k];
          const ydRe = yRe[yc] * DR, ydIm = yIm[yc] * DR;
          const d = yc * 6;
          const wr = entWr[k], wi = entWi[k];
          a0 += wr * (ydRe * rh0 + dy[d] * Ro) - wi * (ydIm * rh0 + dy[d + 1] * Ro);
          a1 += wr * (ydRe * rh1 + dy[d + 2] * Ro) - wi * (ydIm * rh1 + dy[d + 3] * Ro);
          a2 += wr * (ydRe * rh2 + dy[d + 4] * Ro) - wi * (ydIm * rh2 + dy[d + 5] * Ro);
        }
        nDB[o] = a0;
        nDB[o + 1] = a1;
        nDB[o + 2] = a2;
      }
    }
  }
}
