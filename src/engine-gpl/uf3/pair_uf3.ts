/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28), GPL-2.0.
 * Source: src/ML-UF3/pair_uf3.cpp and src/ML-UF3/pair_uf3.h (class PairUF3: settings, coeff, init_style, init_one,
 * create_bsplines, the starting-index searches, create_cached_constants_2b / _3b, compute), with the file reader in
 * potfile.ts and the basis constants in bspline.ts. Authors credited in the C++ headers: Ajinkya Hire, Hendrik Krass,
 * Matthias Rupp and Richard Hennig. Copyright (2003) Sandia Corporation (GNU General Public License). This file is
 * GPL-2.0 (src/engine-gpl/LICENSE).
 *
 * Native summation, kept as is: the 2-body term loops over the FULL neighbour list, so each ordered pair (i, j)
 * adds its energy and its force to both atoms; the 3-body term loops over the short list of the 2-body neighbours
 * of i with rij <= cut_3b_list, one unordered pair (j, k) per centre i. Global virial by the fdotr sum (the native
 * vflag_fdotr path); per-atom energy split as ev_tally_xyz (half to each atom) and ev_tally3 (a third to each atom).
 * Per-atom stress is not ported and is refused.
 */
import { Pair, StyleError, type PairCompute, type StyleContext } from '../../engine/force/types';
import { NEIGHMASK } from '../../engine/neighbor';
import { uf3Basis2Constants, uf3Basis3Constants } from './bspline';
import { idx3, key3, readUF3Potential, type UF3Potential } from './potfile';

const THIRD = 1 / 3;

export class PairUF3 extends Pair {
  readonly name: string = 'uf3';
  manybody = true;
  needsFull = true;
  needsHalf = false;
  virialFdotr = true;

  private nbody: 2 | 3 = 3;
  private pot: UF3Potential | null = null;
  /** Element index (native map[]) of each atom type 1..ntypes. */
  private mapOf: number[] = [];
  private elementOfType: string[] = [];
  private elements: string[] = [];
  /** Built by initStyle from the potential. */
  private uniform2 = false;
  private uniform3 = false;
  private cached2: Float64Array[] = [];
  private deri2: Float64Array[] = [];
  private cached3: Float64Array[] = [];
  private deri3: Float64Array[] = [];
  private coefDerIJ: Float64Array[] = [];
  private coefDerIK: Float64Array[] = [];
  private coefDerJK: Float64Array[] = [];
  /** Stride of the coefficient cubes (the largest 2-body or 3-body coefficient count). */
  private cM = 0;
  /** Coefficient cubes per key, restrided to cM (native n3b_coeff_array). */
  private coeff3S: Float64Array[] = [];
  private neighshort = new Int32Array(0);

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 1) throw new StyleError('Invalid number of arguments for pair_style uf3  Are you using a 2-body or 2 & 3-body UF potential?');
    if (args[0] === '2') this.nbody = 2;
    else if (args[0] === '3') this.nbody = 3;
    else throw new StyleError(`Pair style uf3 not (yet) implemented for ${args[0]}-body terms`);
  }

  coeff(args: string[], ctx: StyleContext): void {
    if (this.ntypes === 0) throw new StyleError('pair_coeff needs the simulation box (create_box) first');
    if (args.length !== 3 + this.ntypes) throw new StyleError('Invalid number of arguments uf3 in pair coeffs.');
    if (args[0] !== '*' || args[1] !== '*') throw new StyleError('the first 2 arguments of pair_coeff for style uf3 must be * *');
    const elems = args.slice(3);
    if (elems.includes('NULL')) throw new StyleError('pair style uf3: NULL atom types are not supported');
    this.elements = [...new Set(elems)];
    this.elementOfType = [''].concat(elems);
    this.mapOf = [-1].concat(elems.map((e) => this.elements.indexOf(e)));
    this.pot = readUF3Potential(ctx.readFile(args[2]), this.nbody, this.ntypes, this.elementOfType, this.mapOf, this.elements, args[2]);
  }

  override initStyle(ctx: StyleContext): void {
    const pot = this.pot;
    if (!pot) throw new StyleError('pair style uf3 needs a pair_coeff command with the unified UF3 potential file');
    if (this.tail) throw new StyleError('pair style uf3 does not support pair_modify tail yes');
    this.noTermLog(ctx, 'shift');
    const N = this.ntypes, S = N + 1;
    for (let i = 1; i <= N; i++) {
      for (let j = 1; j <= N; j++) {
        if (pot.setflag2[i * S + j] !== 1) {
          throw new StyleError(`UF3: Not all 2-body UF potentials are set, missing potential for ${i}-${j} interaction`);
        }
      }
    }
    const spacingType = pot.pair2[S + 1]?.spacingType ?? 0;
    if (pot.nbody3) {
      for (let i = 1; i <= N; i++) {
        for (let j = 1; j <= N; j++) {
          for (let k = 1; k <= N; k++) {
            if (pot.setflag3[idx3(N, i, j, k)] !== 1) {
              throw new StyleError(`UF3: Not all 3-body UF potentials are set, missing potential for ${i}-${j}-${k} interaction`);
            }
            if (pot.spacingType3[idx3(N, i, j, k)] !== spacingType) {
              throw new StyleError(`UF3: In the current version the knot spacing type, for all interactions needs to be same. For ${i}-${j}-${k} i.e. ${this.elementOfType[i]}-${this.elementOfType[j]}-${this.elementOfType[k]} interaction expected${spacingType}, but found ${pot.spacingType3[idx3(N, i, j, k)]}`);
            }
          }
        }
      }
    }
    this.uniform2 = spacingType === 0;
    this.uniform3 = spacingType === 0;
    this.createCaches(pot);
  }

  initOne(i: number, j: number): number {
    if (!this.pot || this.pot.setflag2[i * (this.ntypes + 1) + j] !== 1) throw new StyleError('All pair coeffs are not set');
    return this.pot.cut2[i * (this.ntypes + 1) + j];
  }

  /** create_cached_constants_2b and _3b: the basis constants of every knot span and the coefficient derivatives. */
  private createCaches(pot: UF3Potential): void {
    const N = this.ntypes, S = N + 1;
    const M = pot.maxC2 > pot.maxC3 ? pot.maxC2 : pot.maxC3;
    this.cM = M;
    this.cached2 = [];
    this.deri2 = [];
    for (let i = 1; i <= N; i++) {
      for (let j = 1; j <= N; j++) {
        const p = i * S + j;
        const e = pot.pair2[p];
        const cached = new Float64Array(M * 16);
        const deri = new Float64Array(M * 9);
        if (e) {
          for (let l = 0; l < e.nCoeff; l++) {
            cached.set(uf3Basis3Constants(e.knots, l, e.coeff[l]), l * 16);
          }
          const nk = e.nKnots;
          const kd = e.knots.subarray(1, nk - 1);
          for (let l = 0; l < e.nCoeff - 1; l++) {
            const dntemp = 3 / (e.knots[l + 4] - e.knots[l + 1]);
            const cd = (e.coeff[l + 1] - e.coeff[l]) * dntemp;
            deri.set(uf3Basis2Constants(kd, l, cd), l * 9);
          }
        }
        this.cached2[p] = cached;
        this.deri2[p] = deri;
      }
    }
    if (!pot.nbody3) return;
    const nKeys = N * N * N;
    this.cached3 = [];
    this.deri3 = [];
    this.coefDerIJ = [];
    this.coefDerIK = [];
    this.coefDerJK = [];
    this.coeff3S = [];
    for (let key = 0; key < nKeys; key++) {
      const kJK = pot.knots3[key * 3], kIK = pot.knots3[key * 3 + 1], kIJ = pot.knots3[key * 3 + 2];
      const nJK = pot.nKnots3[key * 3], nIK = pot.nKnots3[key * 3 + 1], nIJ = pot.nKnots3[key * 3 + 2];
      const d1 = pot.sizeC3[key * 3], d2 = pot.sizeC3[key * 3 + 1], d3 = pot.sizeC3[key * 3 + 2];
      const M3 = pot.maxC3;
      const c = new Float64Array(M * M * M);
      for (let a = 0; a < d1; a++) {
        for (let b = 0; b < d2; b++) {
          for (let q = 0; q < d3; q++) c[(a * M + b) * M + q] = pot.coeff3[key][(a * M3 + b) * M3 + q];
        }
      }
      this.coeff3S[key] = c;
      const cached = new Float64Array(3 * M * 16);
      const deri = new Float64Array(3 * M * 9);
      // value bases: dim 0 = ij (from the ij knots), 1 = ik, 2 = jk
      for (let l = 0; l < nIJ - 4; l++) cached.set(uf3Basis3Constants(kIJ, l, 1), (0 * M + l) * 16);
      for (let l = 0; l < nIK - 4; l++) cached.set(uf3Basis3Constants(kIK, l, 1), (1 * M + l) * 16);
      for (let l = 0; l < nJK - 4; l++) cached.set(uf3Basis3Constants(kJK, l, 1), (2 * M + l) * 16);
      // coefficient derivatives
      const dJK = new Float64Array(M * M * M), dIK = new Float64Array(M * M * M), dIJ = new Float64Array(M * M * M);
      const at = (a: number, b: number, cc: number) => (a * M + b) * M + cc;
      for (let l = 0; l < d1; l++) {
        for (let m = 0; m < d2; m++) {
          for (let n = 0; n < d3 - 1; n++) {
            const dntemp = 3 / (kJK[n + 4] - kJK[n + 1]);
            dJK[at(l, m, n)] = (c[at(l, m, n + 1)] - c[at(l, m, n)]) * dntemp;
          }
        }
      }
      for (let l = 0; l < d1; l++) {
        for (let m = 0; m < d2 - 1; m++) {
          const dntemp = 3 / (kIK[m + 4] - kIK[m + 1]);
          for (let n = 0; n < d3; n++) {
            dIK[at(l, m, n)] = (c[at(l, m + 1, n)] - c[at(l, m, n)]) * dntemp;
          }
        }
      }
      for (let l = 0; l < d1 - 1; l++) {
        const dntemp = 3 / (kIJ[l + 4] - kIJ[l + 1]);
        for (let m = 0; m < d2; m++) {
          for (let n = 0; n < d3; n++) {
            dIJ[at(l, m, n)] = (c[at(l + 1, m, n)] - c[at(l, m, n)]) * dntemp;
          }
        }
      }
      // derivative bases of the knots without their first and last entry
      const sJK = kJK.subarray(1, nJK - 1), sIK = kIK.subarray(1, nIK - 1), sIJ = kIJ.subarray(1, nIJ - 1);
      for (let l = 0; l < d1 - 1; l++) deri.set(uf3Basis2Constants(sIJ, l, 1), (0 * M + l) * 9);
      for (let l = 0; l < d2 - 1; l++) deri.set(uf3Basis2Constants(sIK, l, 1), (1 * M + l) * 9);
      for (let l = 0; l < d3 - 1; l++) deri.set(uf3Basis2Constants(sJK, l, 1), (2 * M + l) * 9);
      this.cached3[key] = cached;
      this.deri3[key] = deri;
      this.coefDerIJ[key] = dIJ;
      this.coefDerIK[key] = dIK;
      this.coefDerJK[key] = dJK;
    }
  }

  /** get_starting_index_*_2b: the knot span of r (native index, -1 when not found). */
  private span2(p: number, r: number): number {
    const e = this.pot!.pair2[p]!;
    if (this.uniform2) return 3 + Math.trunc((r - e.knots[0]) / e.spacing);
    for (let l = 3; l < e.nKnots - 1; l++) {
      if (e.knots[l] <= r && r < e.knots[l + 1]) return l;
    }
    return -1;
  }

  /** get_starting_index_*_3b: the span of r on knot dimension dim (0 jk, 1 ik, 2 ij) of key. */
  private span3(i: number, j: number, k: number, r: number, dim: number): number {
    const pot = this.pot!;
    const N = this.ntypes;
    const key = key3(N, i, j, k);
    const kn = pot.knots3[key * 3 + dim];
    if (this.uniform3) return 3 + Math.trunc((r - kn[0]) / pot.spacing3[idx3(N, i, j, k) * 3 + dim]);
    const n = pot.nKnots3[key * 3 + dim];
    for (let l = 3; l < n - 1; l++) {
      if (kn[l] <= r && r < kn[l + 1]) return l;
    }
    return -1;
  }

  compute(pc: PairCompute): void {
    const list = pc.full;
    if (!list) throw new Error('pair style uf3 needs a full neighbor list');
    if (pc.vatom) throw new StyleError('pair style uf3: per-atom stress (compute stress/atom) is not ported');
    const pot = this.pot;
    if (!pot) throw new Error('pair style uf3: no coefficients');
    const { x, f, type } = pc;
    const nlocal = pc.nlocal;
    const nb = list.neighbors;
    const N = this.ntypes, S = N + 1;
    const cutsq = this.cutsq;
    const three = pot.nbody3;
    const M = this.cM;
    const eatom = pc.eatom;
    const acc = pc.acc;
    for (let i = 0; i < nlocal; i++) {
      const itype = type[i];
      const xtmp = x[3 * i], ytmp = x[3 * i + 1], ztmp = x[3 * i + 2];
      const k0 = list.firstneigh[i], k1 = k0 + list.numneigh[i];
      let numshort = 0;
      for (let kk0 = k0; kk0 < k1; kk0++) {
        const j = nb[kk0] & NEIGHMASK;
        const delx = xtmp - x[3 * j], dely = ytmp - x[3 * j + 1], delz = ztmp - x[3 * j + 2];
        const rsq = delx * delx + dely * dely + delz * delz;
        const jtype = type[j];
        if (!(rsq < cutsq[itype * S + jtype])) continue;
        const rij = Math.sqrt(rsq);
        if (three && rij <= pot.cutList3[itype * S + jtype]) {
          if (numshort >= this.neighshort.length) {
            const grown = new Int32Array(Math.max(20, this.neighshort.length * 2));
            grown.set(this.neighshort);
            this.neighshort = grown;
          }
          this.neighshort[numshort++] = j;
        }
        // ---- 2-body term (native compute, first block) ----
        const p = itype * S + jtype;
        const ks = this.span2(p, rij);
        const dd = this.deri2[p];
        const cc = this.cached2[p];
        let force2 = dd[(ks - 1) * 9 + 0];
        force2 += rij * dd[(ks - 1) * 9 + 1];
        force2 += rsq * dd[(ks - 1) * 9 + 2];
        force2 += dd[(ks - 2) * 9 + 3];
        force2 += rij * dd[(ks - 2) * 9 + 4];
        force2 += rsq * dd[(ks - 2) * 9 + 5];
        force2 += dd[(ks - 3) * 9 + 6];
        force2 += rij * dd[(ks - 3) * 9 + 7];
        force2 += rsq * dd[(ks - 3) * 9 + 8];
        const fpair = (-1 * force2) / rij;
        const fx = delx * fpair, fy = dely * fpair, fz = delz * fpair;
        f[3 * i] += fx; f[3 * i + 1] += fy; f[3 * i + 2] += fz;
        f[3 * j] -= fx; f[3 * j + 1] -= fy; f[3 * j + 2] -= fz;
        const rth = rsq * rij;
        let e2 = cc[ks * 16 + 0];
        e2 += rij * cc[ks * 16 + 1];
        e2 += rsq * cc[ks * 16 + 2];
        e2 += rth * cc[ks * 16 + 3];
        e2 += cc[(ks - 1) * 16 + 4];
        e2 += rij * cc[(ks - 1) * 16 + 5];
        e2 += rsq * cc[(ks - 1) * 16 + 6];
        e2 += rth * cc[(ks - 1) * 16 + 7];
        e2 += cc[(ks - 2) * 16 + 8];
        e2 += rij * cc[(ks - 2) * 16 + 9];
        e2 += rsq * cc[(ks - 2) * 16 + 10];
        e2 += rth * cc[(ks - 2) * 16 + 11];
        e2 += cc[(ks - 3) * 16 + 12];
        e2 += rij * cc[(ks - 3) * 16 + 13];
        e2 += rsq * cc[(ks - 3) * 16 + 14];
        e2 += rth * cc[(ks - 3) * 16 + 15];
        acc.evdwl += e2;
        if (eatom) {
          eatom[i] += 0.5 * e2;
          eatom[j] += 0.5 * e2;
        }
      }
      if (!three) continue;
      // ---- 3-body term: unordered pairs (j, k) of the short list (native compute, second block) ----
      const xi = xtmp, yi = ytmp, zi = ztmp;
      const ns = this.neighshort;
      for (let jj = 0; jj < numshort - 1; jj++) {
        const j = ns[jj];
        const jtype = type[j];
        const dji0 = x[3 * j] - xi, dji1 = x[3 * j + 1] - yi, dji2 = x[3 * j + 2] - zi;
        const rijSq = dji0 * dji0 + dji1 * dji1 + dji2 * dji2;
        const rij = Math.sqrt(rijSq);
        for (let kk = jj + 1; kk < numshort; kk++) {
          const k = ns[kk];
          const ktype = type[k];
          const dki0 = x[3 * k] - xi, dki1 = x[3 * k + 1] - yi, dki2 = x[3 * k + 2] - zi;
          const rikSq = dki0 * dki0 + dki1 * dki1 + dki2 * dki2;
          const rik = Math.sqrt(rikSq);
          const ijk = idx3(N, itype, jtype, ktype);
          const ikj = idx3(N, itype, ktype, jtype);
          if (!(rij <= pot.cut3[ijk] && rik <= pot.cut3[ikj] && rij >= pot.minCut3[ijk * 3 + 2] && rik >= pot.minCut3[ijk * 3 + 1])) continue;
          const dkj0 = x[3 * k] - x[3 * j], dkj1 = x[3 * k + 1] - x[3 * j + 1], dkj2 = x[3 * k + 2] - x[3 * j + 2];
          const rjkSq = dkj0 * dkj0 + dkj1 * dkj1 + dkj2 * dkj2;
          const rjk = Math.sqrt(rjkSq);
          if (!(rjk >= pot.minCut3[ijk * 3])) continue;
          const rijTh = rij * rijSq, rikTh = rik * rikSq, rjkTh = rjk * rjkSq;
          const key = key3(N, itype, jtype, ktype);
          const cc = this.cached3[key], dd = this.deri3[key];
          const ksIJ = this.span3(itype, jtype, ktype, rij, 2);
          const ksIK = this.span3(itype, jtype, ktype, rik, 1);
          const ksJK = this.span3(itype, jtype, ktype, rjk, 0);
          // value bases: [0..3] of each dim, from the knot spans
          const bIJ = [0, 0, 0, 0], bIK = [0, 0, 0, 0], bJK = [0, 0, 0, 0];
          this.basisValue(cc, 0, ksIJ, rij, rijSq, rijTh, bIJ);
          this.basisValue(cc, 1, ksIK, rik, rikSq, rikTh, bIK);
          this.basisValue(cc, 2, ksJK, rjk, rjkSq, rjkTh, bJK);
          const bIJd = [0, 0, 0], bIKd = [0, 0, 0], bJKd = [0, 0, 0];
          this.basisDeriv(dd, 0, ksIJ, rij, rijSq, bIJd);
          this.basisDeriv(dd, 1, ksIK, rik, rikSq, bIKd);
          this.basisDeriv(dd, 2, ksJK, rjk, rjkSq, bJKd);
          const tev = [0, 0, 0, 0];
          const iIJ = ksIJ - 3, iIK = ksIK - 3, iJK = ksJK - 3;
          const dIJ = this.coefDerIJ[key], dIK = this.coefDerIK[key], dJK = this.coefDerJK[key];
          const cf = this.coeff3S[key];
          for (let l = 0; l < 3; l++) {
            const bl = bIJd[l];
            for (let m = 0; m < 4; m++) {
              const factor = bl * bIK[m];
              const base = ((iIJ + l) * M + (iIK + m)) * M + iJK;
              const t0 = dIJ[base] * bJK[0];
              const t1 = dIJ[base + 1] * bJK[1];
              const t2 = dIJ[base + 2] * bJK[2];
              const t3 = dIJ[base + 3] * bJK[3];
              tev[1] += factor * (t0 + t1 + t2 + t3);
            }
          }
          for (let l = 0; l < 4; l++) {
            const bl = bIJ[l];
            for (let m = 0; m < 3; m++) {
              const factor = bl * bIKd[m];
              const base = ((iIJ + l) * M + (iIK + m)) * M + iJK;
              const t0 = dIK[base] * bJK[0];
              const t1 = dIK[base + 1] * bJK[1];
              const t2 = dIK[base + 2] * bJK[2];
              const t3 = dIK[base + 3] * bJK[3];
              tev[2] += factor * (t0 + t1 + t2 + t3);
            }
          }
          for (let l = 0; l < 4; l++) {
            const bl = bIJ[l];
            for (let m = 0; m < 4; m++) {
              const factor = bl * bIK[m];
              const base = ((iIJ + l) * M + (iIK + m)) * M + iJK;
              const t0 = dJK[base] * bJKd[0];
              const t1 = dJK[base + 1] * bJKd[1];
              const t2 = dJK[base + 2] * bJKd[2];
              tev[3] += factor * (t0 + t1 + t2);
            }
          }
          for (let l = 0; l < 4; l++) {
            const bl = bIJ[l];
            for (let m = 0; m < 4; m++) {
              const factor = bl * bIK[m];
              const base = ((iIJ + l) * M + (iIK + m)) * M + iJK;
              const t0 = cf[base] * bJK[0];
              const t1 = cf[base + 1] * bJK[1];
              const t2 = cf[base + 2] * bJK[2];
              const t3 = cf[base + 3] * bJK[3];
              tev[0] += factor * (t0 + t1 + t2 + t3);
            }
          }
          const evdwl = tev[0];
          const fijS = tev[1], fikS = tev[2], fjkS = tev[3];
          const fij = [0, 0, 0], fik = [0, 0, 0], fjk = [0, 0, 0];
          for (let d = 0; d < 3; d++) {
            const dji = d === 0 ? dji0 : d === 1 ? dji1 : dji2;
            const dki = d === 0 ? dki0 : d === 1 ? dki1 : dki2;
            const dkj = d === 0 ? dkj0 : d === 1 ? dkj1 : dkj2;
            fij[d] = fijS * (dji / rij);
            fik[d] = fikS * (dki / rik);
            fjk[d] = fjkS * (dkj / rjk);
          }
          // forces: F_i = f_ij + f_ik on i; F_j = -f_ij + f_jk on j; F_k = -f_ik - f_jk on k
          const Fi0 = fij[0] + fik[0], Fi1 = fij[1] + fik[1], Fi2 = fij[2] + fik[2];
          f[3 * i] += Fi0; f[3 * i + 1] += Fi1; f[3 * i + 2] += Fi2;
          const Fj0 = -fij[0] + fjk[0], Fj1 = -fij[1] + fjk[1], Fj2 = -fij[2] + fjk[2];
          f[3 * j] += Fj0; f[3 * j + 1] += Fj1; f[3 * j + 2] += Fj2;
          const Fk0 = -fik[0] + -fjk[0], Fk1 = -fik[1] + -fjk[1], Fk2 = -fik[2] + -fjk[2];
          f[3 * k] += Fk0; f[3 * k + 1] += Fk1; f[3 * k + 2] += Fk2;
          acc.evdwl += evdwl;
          if (eatom) {
            const e3 = THIRD * evdwl;
            eatom[i] += e3; eatom[j] += e3; eatom[k] += e3;
          }
        }
      }
    }
  }

  /** The four value bases of one dimension: span ks, distance r. Out in b[0..3] (native basis_ij / basis_ik / basis_jk). */
  private basisValue(cc: Float64Array, dim: number, ks: number, r: number, r2: number, r3: number, b: number[]): void {
    const M = this.cM;
    const at = (l: number, base: number) => cc[(dim * M + l) * 16 + base];
    b[0] = at(ks - 3, 12) + r * at(ks - 3, 13) + r2 * at(ks - 3, 14) + r3 * at(ks - 3, 15);
    b[1] = at(ks - 2, 8) + r * at(ks - 2, 9) + r2 * at(ks - 2, 10) + r3 * at(ks - 2, 11);
    b[2] = at(ks - 1, 4) + r * at(ks - 1, 5) + r2 * at(ks - 1, 6) + r3 * at(ks - 1, 7);
    b[3] = at(ks, 0) + r * at(ks, 1) + r2 * at(ks, 2) + r3 * at(ks, 3);
  }

  /** The three derivative bases of one dimension. */
  private basisDeriv(dd: Float64Array, dim: number, ks: number, r: number, r2: number, b: number[]): void {
    const M = this.cM;
    const at = (l: number, base: number) => dd[(dim * M + l) * 9 + base];
    b[0] = at(ks - 3, 6) + r * at(ks - 3, 7) + r2 * at(ks - 3, 8);
    b[1] = at(ks - 2, 3) + r * at(ks - 2, 4) + r2 * at(ks - 2, 5);
    b[2] = at(ks - 1, 0) + r * at(ks - 1, 1) + r2 * at(ks - 1, 2);
  }
}
