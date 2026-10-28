/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the ML-PACE library v.2025.12.4.p1 it builds
 * for src/ML-PACE (with EXTRA_C_PROJECTIONS, as its CMakeLists.txt defines), GPL-2.0 use: ace/ace_b_evaluator.cpp
 * (ACEBEvaluator::init, resize_neighbours_cache, compute_atom with the projections, dE_dc and the extrapolation grade,
 * load_active_set, validate_ASI_shape, validate_ASI_square_shape, resize_projections). Library authors (from its
 * headers): Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz and co-authors. This file is GPL-2.0
 * (src/engine-gpl/LICENSE).
 *
 * The B-basis evaluator pair_style pace/extrapolation uses on the steps that need the extrapolation grade: the atomic
 * energy and the neighbour forces from the B-basis functions directly, the B projections of the atom and the grade
 * gamma = max_i |sum_k v_k (A^-1)_ki| with v the projections (linear active set) or dE/dc (non-linear). ACEComplex is
 * kept as two numbers, with its operations in the C++ order.
 */
import { StyleError } from '../../engine/force/types';
import type { BBasis } from './bbasis';
import { innerCutoff } from './cbasis';
import { fexp, fexpShifted } from './evaluator';
import { npzLoad } from './npz';

/** Y00 of the library (ace_spherical_cart.h). */
const Y00 = 1;

export class ACEBEvaluator {
  eAtom = 0;
  aceFcut = 1;
  /** neighbours_forces(jj, alpha) of the last compute_atom call */
  neighboursForces = new Float64Array(0);
  /** max_gamma_grade: the grade of the last atom whose species has an active set */
  maxGammaGrade = 0;
  isLinearExtrapolationGrade = true;
  /** compute_projections: set by the pair style before each compute_atom on a grade step */
  computeProjections = false;
  projections = new Float64Array(0);
  dEdc = new Float64Array(0);
  /** A_active_set_inv by species: the transposed inverse of the active set, dim x dim row-major */
  private readonly activeSetInv = new Map<number, { dim0: number; dim1: number; m: Float64Array }>();

  private readonly lmax: number;
  private readonly L1: number;
  private readonly nm: number;
  private readonly nradb: number;
  private readonly nradi: number;
  private readonly nel: number;
  private readonly Are: Float64Array;
  private readonly Aim: Float64Array;
  private readonly Arank1: Float64Array;
  private readonly wRe: Float64Array;
  private readonly wIm: Float64Array;
  private readonly wRank1: Float64Array;
  private readonly rhos: Float64Array;
  private readonly dFdrho: Float64Array;
  private dBRe = new Float64Array(0);
  private dBIm = new Float64Array(0);
  private cap = 0;
  private dgCache = new Float64Array(0);
  private drCache = new Float64Array(0);
  private rCache = new Float64Array(0);
  private yRe = new Float64Array(0);
  private yIm = new Float64Array(0);
  private dyCache = new Float64Array(0);
  private dcrCache = new Float64Array(0);

  constructor(readonly basis: BBasis, private readonly typeMap: Int32Array) {
    this.lmax = basis.lmax;
    this.L1 = basis.lmax + 1;
    this.nm = 2 * basis.lmax + 1;
    this.nradb = basis.nradbase;
    this.nradi = basis.nradmax;
    this.nel = basis.nelements;
    const nA = this.nel * this.nradi * this.L1 * this.nm;
    this.Are = new Float64Array(nA);
    this.Aim = new Float64Array(nA);
    this.wRe = new Float64Array(nA);
    this.wIm = new Float64Array(nA);
    this.Arank1 = new Float64Array(this.nel * this.nradb);
    this.wRank1 = new Float64Array(this.nel * this.nradb);
    this.rhos = new Float64Array(basis.ndensitymax);
    this.dFdrho = new Float64Array(basis.ndensitymax);
    // max_dB_array_size: the largest sum of rank * num_ms_combs over the functions of one species
    let maxDB = 0;
    for (let mu = 0; mu < this.nel; mu++) {
      let s = 0;
      for (const f of basis.basis[mu]) s += f.rank * f.numMsCombs;
      if (s > maxDB) maxDB = s;
    }
    this.dBRe = new Float64Array(maxDB);
    this.dBIm = new Float64Array(maxDB);
  }

  private aIdx(mu: number, n: number, l: number, m: number): number {
    return ((mu * this.nradi + n) * this.L1 + l) * this.nm + (m + this.lmax);
  }

  /** resize_neighbours_cache */
  private ensureCache(jnum: number): void {
    if (jnum <= this.cap) return;
    this.cap = jnum;
    const L1 = this.L1;
    this.dgCache = new Float64Array(jnum * this.nradb);
    this.drCache = new Float64Array(jnum * this.nradi * L1);
    this.rCache = new Float64Array(jnum * this.nradi * L1);
    this.yRe = new Float64Array(jnum * L1 * L1);
    this.yIm = new Float64Array(jnum * L1 * L1);
    this.dyCache = new Float64Array(jnum * L1 * L1 * 6);
    this.dcrCache = new Float64Array(jnum);
  }

  /** compute_atom(i, x, type, jnum, jlist) of ACEBEvaluator (forces always computed: compute_energy_only is false). */
  computeAtom(i: number, x: Float64Array, type: Int32Array, jnum: number, jlist: Int32Array): void {
    this.ensureCache(jnum);
    const basis = this.basis, rad = basis.radial, sph = basis.sph;
    const L1 = this.L1, lmax = this.lmax, nradb = this.nradb, nradi = this.nradi;
    const muI = this.typeMap[type[i]];
    const totalRank1 = basis.totalBasisSizeRank1[muI];
    const total = basis.totalBasisSize[muI];
    const rank1 = basis.basisRank1[muI];
    const funcs = basis.basis[muI];
    const ndensity = basis.embedding[muI].ndensity;
    const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];

    this.dBRe.fill(0);
    this.dBIm.fill(0);
    this.Are.fill(0);
    this.Aim.fill(0);
    this.Arank1.fill(0);
    this.rhos.fill(0);
    this.dFdrho.fill(0);
    if (this.neighboursForces.length < 3 * jnum) this.neighboursForces = new Float64Array(3 * jnum);
    this.neighboursForces.fill(0, 0, 3 * jnum);
    this.wRe.fill(0);
    this.wIm.fill(0);
    this.wRank1.fill(0);
    const nproj = totalRank1 + total;
    if (this.computeProjections) {
      this.projections = new Float64Array(nproj);
      this.dEdc = new Float64Array(nproj * basis.ndensitymax);
    }

    // neighbours inside the cutoff, in list order
    const rNorms = new Float64Array(jnum), invRNorms = new Float64Array(jnum);
    const rhats = new Float64Array(3 * jnum);
    const elements = new Int32Array(jnum), nbrIndex = new Int32Array(jnum);
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

    // the A's: radial times spherical harmonics summed over the neighbours
    let rhoCore = 0;
    for (let jj = 0; jj < jnumActual; jj++) {
      const muJ = elements[jj];
      rad.evaluate(rNorms[jj], muI, muJ);
      sph.computeYlm(rhats[3 * jj], rhats[3 * jj + 1], rhats[3 * jj + 2], lmax);
      const gr = rad.gr, dgr = rad.dgr, fr = rad.fr, dfr = rad.dfr;
      for (let n = 0; n < nradb; n++) {
        this.dgCache[jj * nradb + n] = dgr[n];
        this.Arank1[muJ * nradb + n] += gr[n] * Y00;
      }
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= lmax; l++) {
          const R = fr[n * L1 + l];
          const rc = (jj * nradi + n) * L1 + l;
          this.drCache[rc] = dfr[n * L1 + l];
          this.rCache[rc] = R;
          for (let m = 0; m <= l; m++) {
            const yr = sph.ylmRe[l * L1 + m], yi = sph.ylmIm[l * L1 + m];
            const a = this.aIdx(muJ, n, l, m);
            this.Are[a] += R * yr;
            this.Aim[a] += R * yi;
            const yc = (jj * L1 + l) * L1 + m;
            this.yRe[yc] = yr;
            this.yIm[yc] = yi;
            for (let q = 0; q < 6; q++) this.dyCache[yc * 6 + q] = sph.dylm[(l * L1 + m) * 6 + q];
          }
        }
      }
      rhoCore += rad.cr;
      this.dcrCache[jj] = rad.dcr;
    }
    for (let mu = 0; mu < this.nel; mu++) {
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= lmax; l++) {
          for (let m = 1; m <= l; m++) {
            const factor = m % 2 === 0 ? 1 : -1;
            const src = this.aIdx(mu, n, l, m), dst = this.aIdx(mu, n, l, -m);
            this.Are[dst] = this.Are[src] * factor;
            this.Aim[dst] = (-this.Aim[src]) * factor;
          }
        }
      }
    }

    // densities (and projections): rank 1, then the products of the higher ranks
    const rhos = this.rhos, proj = this.projections;
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      const Acur = this.Arank1[func.mus[0] * nradb + (func.ns[0] - 1)];
      if (this.computeProjections) proj[f] += Acur;
      for (let p = 0; p < ndensity; p++) rhos[p] += func.coeff[p] * Acur;
    }
    const maxRank = basis.rankmax + 1;
    const fwRe = new Float64Array(maxRank), fwIm = new Float64Array(maxRank);
    const bwRe = new Float64Array(maxRank), bwIm = new Float64Array(maxRank);
    const cRe = new Float64Array(maxRank), cIm = new Float64Array(maxRank);
    let msT = 0;
    for (let fi = 0; fi < total; fi++) {
      const func = funcs[fi];
      const rank = func.rank, r = rank - 1;
      const mus = func.mus, ns = func.ns, ls = func.ls;
      for (let k = 0; k < func.numMsCombs; k++) {
        const o = k * rank;
        fwRe[0] = 1; fwIm[0] = 0;
        bwRe[r] = 1; bwIm[r] = 0;
        for (let t = 0; t < rank; t++) {
          const a = this.aIdx(mus[t], ns[t] - 1, ls[t], func.msCombs[o + t]);
          cRe[t] = this.Are[a];
          cIm[t] = this.Aim[a];
          fwRe[t + 1] = fwRe[t] * cRe[t] - fwIm[t] * cIm[t];
          fwIm[t + 1] = fwRe[t] * cIm[t] + fwIm[t] * cRe[t];
        }
        const BRe = fwRe[rank];
        for (let t = r; t >= 1; t--) {
          bwRe[t - 1] = bwRe[t] * cRe[t] - bwIm[t] * cIm[t];
          bwIm[t - 1] = bwRe[t] * cIm[t] + bwIm[t] * cRe[t];
        }
        for (let t = 0; t < rank; t++, msT++) {
          this.dBRe[msT] = fwRe[t] * bwRe[t] - fwIm[t] * bwIm[t];
          this.dBIm[msT] = fwRe[t] * bwIm[t] + fwIm[t] * bwRe[t];
        }
        if (this.computeProjections) proj[totalRank1 + fi] += BRe * func.genCgs[k];
        for (let p = 0; p < ndensity; p++) rhos[p] += BRe * (func.genCgs[k] * func.coeff[p]);
      }
    }

    // FS_values_and_derivatives
    const emb = basis.embedding[muI];
    let evdwl = 0;
    const dF = this.dFdrho;
    for (let p = 0; p < ndensity; p++) {
      const wpre = emb.FS[p * 2], mexp = emb.FS[p * 2 + 1];
      if (emb.npoti !== 'FinnisSinclair' && emb.npoti !== 'FinnisSinclairShiftedScaled') {
        throw new StyleError(`pair_style pace/extrapolation: embedding-function ${emb.npoti} is not supported`);
      }
      const [F, DF] = emb.npoti === 'FinnisSinclair' ? fexp(rhos[p], mexp) : fexpShifted(rhos[p], mexp);
      evdwl += F * wpre;
      dF[p] = DF * wpre;
    }
    if (this.computeProjections && !this.isLinearExtrapolationGrade) {
      let q = 0;
      for (let p = 0; p < ndensity; p++) for (let k = 0; k < nproj; k++, q++) this.dEdc[q] = dF[p] * proj[k];
    }
    const [fcut, dfcut] = innerCutoff(rhoCore, emb.rhoCoreCutoff, emb.drhoCoreCutoff);
    const evdwlCut = evdwl * fcut + rhoCore;
    const dFdrhoCore = evdwl * dfcut + 1;
    for (let p = 0; p < ndensity; p++) dF[p] *= fcut;

    // weights
    for (let f = 0; f < totalRank1; f++) {
      const func = rank1[f];
      for (let p = 0; p < ndensity; p++) this.wRank1[func.mus[0] * nradb + (func.ns[0] - 1)] += dF[p] * func.coeff[p];
    }
    msT = 0;
    for (let fi = 0; fi < total; fi++) {
      const func = funcs[fi];
      const rank = func.rank;
      for (let k = 0; k < func.numMsCombs; k++) {
        let theta = 0;
        for (let p = 0; p < ndensity; p++) theta += dF[p] * func.genCgs[k] * func.coeff[p];
        theta *= 0.5;
        for (let t = 0; t < rank; t++, msT++) {
          const mt = func.msCombs[k * rank + t];
          const factor = mt % 2 === 0 ? 1 : -1;
          const dRe = this.dBRe[msT], dIm = this.dBIm[msT];
          const ns1 = func.ns[t] - 1;
          const a = this.aIdx(func.mus[t], ns1, func.ls[t], mt);
          this.wRe[a] += theta * dRe;
          this.wIm[a] += theta * dIm;
          const b = this.aIdx(func.mus[t], ns1, func.ls[t], -mt);
          this.wRe[b] += (theta * dRe) * factor;
          this.wIm[b] += (theta * -dIm) * factor;
        }
      }
    }

    // forces on the neighbours
    for (let jj = 0; jj < jnumActual; jj++) {
      const muJ = elements[jj];
      const rh0 = rhats[3 * jj], rh1 = rhats[3 * jj + 1], rh2 = rhats[3 * jj + 2];
      const invR = invRNorms[jj];
      let f0 = 0, f1 = 0, f2 = 0;
      for (let n = 0; n < nradb; n++) {
        const wr = this.wRank1[muJ * nradb + n];
        if (wr === 0) continue;
        let DGR = this.dgCache[jj * nradb + n] * Y00;
        DGR *= wr;
        f0 += DGR * rh0;
        f1 += DGR * rh1;
        f2 += DGR * rh2;
      }
      for (let n = 0; n < nradi; n++) {
        for (let l = 0; l <= lmax; l++) {
          const rc = (jj * nradi + n) * L1 + l;
          const Ro = this.rCache[rc] * invR;
          const DR = this.drCache[rc];
          for (let m = 0; m <= l; m++) {
            const a = this.aIdx(muJ, n, l, m);
            let wr = this.wRe[a], wi = this.wIm[a];
            if (wr === 0 && wi === 0) continue;
            if (m > 0) { wr *= 2; wi *= 2; }
            const yc = (jj * L1 + l) * L1 + m;
            const ydRe = this.yRe[yc] * DR, ydIm = this.yIm[yc] * DR;
            const d = this.dyCache, o = yc * 6;
            const g0Re = ydRe * rh0 + d[o] * Ro, g0Im = ydIm * rh0 + d[o + 1] * Ro;
            const g1Re = ydRe * rh1 + d[o + 2] * Ro, g1Im = ydIm * rh1 + d[o + 3] * Ro;
            const g2Re = ydRe * rh2 + d[o + 4] * Ro, g2Im = ydIm * rh2 + d[o + 5] * Ro;
            f0 += wr * g0Re - wi * g0Im;
            f1 += wr * g1Re - wi * g1Im;
            f2 += wr * g2Re - wi * g2Im;
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

    // the extrapolation grade
    if (this.computeProjections) {
      const asi = this.activeSetInv.get(muI);
      if (asi) {
        const v = this.isLinearExtrapolationGrade ? this.projections : this.dEdc;
        const kmax = Math.min(v.length, asi.dim1);
        let gammaMax = 0;
        for (let r = 0; r < asi.dim0; r++) {
          let g = 0;
          for (let k = 0; k < kmax; k++) g += v[k] * asi.m[r * asi.dim1 + k];
          if (Math.abs(g) > gammaMax) gammaMax = Math.abs(g);
        }
        this.maxGammaGrade = gammaMax;
      }
    }
  }

  /** load_active_set(asi_filename) with the linear or non-linear type decided from the matrix size. */
  loadActiveSet(bytes: Uint8Array, filename: string): void {
    const basis = this.basis;
    const npz = npzLoad(bytes);
    const names = [...npz.keys()].sort();
    if (npz.size !== basis.nelements) {
      throw new StyleError(`Number of species types in ASI \`${filename}\` (${npz.size})not equal to number of species in ACEBBassiSet (${basis.nelements})`);
    }
    const species = (name: string): number => {
      const st = basis.elementsToIndex.get(name);
      if (st === undefined) throw new StyleError(`Active Set Inverted: species ${name} is not in the B-basis`);
      return st;
    };
    const square = (st: number, shape: number[]) => {
      if (shape[0] !== shape[1]) {
        throw new StyleError(`Active Set Inverted for element \`${basis.elementsName[st]}\`:should be square matrix, but has shape (${shape[0]}, ${shape[1]})`);
      }
    };
    const flags: boolean[] = [];
    for (const name of names) {
      const st = species(name);
      const shape = npz.get(name)!.shape;
      square(st, shape);
      const nfun = basis.totalBasisSizeRank1[st] + basis.totalBasisSize[st];
      const nd = basis.embedding[st].ndensity;
      if (shape[0] === nfun) flags[st] = true;
      else if (shape[0] === nfun * nd) flags[st] = false;
      else {
        throw new StyleError(`Active Set Inverted for element \`${name}\`:expected size ${nfun} (linear) or ${nfun * nd} (nonlinear) , but has size ${shape[0]}`);
      }
    }
    for (let st = 1; st < basis.nelements; st++) {
      if (flags[st] !== flags[0]) {
        throw new StyleError('Active Set Inverted: could not determine extrapolation type (linear or non-linear) automatically, because it differs for different elements');
      }
    }
    this.isLinearExtrapolationGrade = flags[0];
    for (const name of names) {
      const st = species(name);
      const arr = npz.get(name)!;
      const shape = arr.shape;
      square(st, shape);
      const nfun = basis.totalBasisSizeRank1[st] + basis.totalBasisSize[st];
      const expected = this.isLinearExtrapolationGrade ? nfun : nfun * basis.embedding[st].ndensity;
      if (expected !== shape[0]) {
        throw new StyleError(`Active Set Inverted for element \`${name}\`:expected shape: (${expected}, ${expected}) , but has shape (${shape[0]}, ${shape[1]})`);
      }
      // Array2D::set_flatten_vector (row-major), then the transpose
      const d0 = shape[0], d1 = shape[1];
      const m = new Float64Array(d1 * d0);
      for (let r = 0; r < d0; r++) for (let c = 0; c < d1; c++) m[c * d0 + r] = arr.data[r * d1 + c] ?? 0;
      this.activeSetInv.set(st, { dim0: d1, dim1: d0, m });
    }
  }
}
