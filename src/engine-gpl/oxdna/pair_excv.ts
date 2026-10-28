/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna_excv.cpp and
 * pair_oxdna_excv.h (class PairOxdnaExcv, pair_style oxdna/excv): the oxDNA excluded volume between the
 * backbone (bk) and base (bs) sites of two nucleotides, with the tetramer-dependent cutoffs and LJ tables of
 * the 3'-5' neighbours. Original authors (from the headers): Oliver Henrich (University of Strathclyde,
 * Glasgow) and Kierran Falloon (University of Strathclyde, Glasgow); described by Ouldridge, Louis and Doye
 * (J Chem Phys 2011). LAMMPS is Copyright (2003) Sandia Corporation, distributed under the GNU General Public
 * License, version 2 (GPL-2.0); this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 *
 * The C++ computes the local frames of its ghost atoms from the fix OXDNA/LRF (forward communication); here the
 * frame of every ghost is the frame of the owned atom it images. The pair list is the default (half) list.
 */
import { Pair, StyleError, typeBounds, type AdaptAccess, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_5P, OXDNA_3P, isEllipsoid } from '../../engine/atoms';
import { parseNum } from '../../engine/force/util';
import { type OxdnaConstants, F3, backboneOxdna1, baseSiteOxdna1, lrfOf, oxdnaConstants, potentialLines } from './common';
import { pow as libmPow } from '../libm/pow';
import { NativePairOrder } from '../neighbor/nativeOrder';
import { oxdnaTableAccess } from './adapt';

/** The three site kinds of the excluded volume: bk-bk, bk-bs (both orders) and bs-bs. */
interface SiteTable {
  eps: Float64Array;
  sigma: Float64Array;
  cutAst: Float64Array;
  cutsqAst: Float64Array;
  b: Float64Array;
  cutC: Float64Array;
  cutsqC: Float64Array;
  lj1: Float64Array;
  lj2: Float64Array;
}

const newTable = (n: number): SiteTable => ({
  eps: new Float64Array(n), sigma: new Float64Array(n), cutAst: new Float64Array(n), cutsqAst: new Float64Array(n),
  b: new Float64Array(n), cutC: new Float64Array(n), cutsqC: new Float64Array(n), lj1: new Float64Array(n), lj2: new Float64Array(n),
});

/**
 * pair_oxdna_excv.cpp coeff(): b and the cutoff cut_c follow from sigma and cut_ast by the formulas written
 * there (the tail of the LJ at cut_ast is continued by a quadratic that reaches zero at cut_c).
 */
export const derived = (sigma: number, cutAst: number): { b: number; cutC: number } => {
  const x = sigma / cutAst;
  const b = 4.0 / sigma
    * (6.0 * libmPow(x, 7) - 12.0 * libmPow(x, 13))
    * 4.0 / sigma * (6.0 * libmPow(x, 7) - 12.0 * libmPow(x, 13))
    / 4.0 / (4.0 * (libmPow(x, 12) - libmPow(x, 6)));
  const cutC = cutAst
    - 2.0 * 4.0 * (libmPow(x, 12) - libmPow(x, 6))
    / (4.0 / sigma * (6.0 * libmPow(x, 7) - 12.0 * libmPow(x, 13)));
  return { b, cutC };
};

export class PairOxdnaExcv extends Pair {
  readonly name: string = 'oxdna/excv';
  /** backbone site rule of the nucleotide model (the oxDNA2 subclass overrides it) */
  protected backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna1(c, e1, e2, e3, out);
  }
  // the global virial is the dot product of positions and forces (pair_oxdna_excv.cpp: virial_fdotr_compute)
  virialFdotr = true;
  /**
   * neighbor.cpp (special_flag): when the pair style matches the pattern "^ox.*" followed by "excv", the
   * special flags of 1-2, 1-3 and 1-4 neighbours become 2 (kept in the list), so the bonded pairs stay in the
   * neighbour list with their special bits. Only the bkbk term multiplies by the special factor.
   */
  keepExcluded = true;
  /** base site rule of the nucleotide model (the oxDNA3 subclass takes the base type into account) */
  protected baseSiteOf: (c: OxdnaConstants, type: number, e1: ArrayLike<number>, out: number[]) => void = baseSiteOxdna1;
  protected nt = 0;
  protected set: Uint8Array = new Uint8Array(0);
  protected bk!: SiteTable;
  protected bs!: SiteTable;
  protected bb!: SiteTable;
  /** tetramer tables (pair_oxdna_excv.cpp sigma4_bsbs, cut4..., lj14, lj24, b4), indexed [i][j][k][l] */
  protected t4!: { sigma: Float64Array; cutAst: Float64Array; cutsqAst: Float64Array; cutC: Float64Array; cutsqC: Float64Array; b: Float64Array; lj14: Float64Array; lj24: Float64Array };
  protected tEps = new Float64Array(0);
  private readonly natPairs = new NativePairOrder();

  /**
   * fix adapt's names (pair_oxdna_excv.cpp extract): epsilon, sigma, cut_ast, b and cut_c of the bkbk, bkbs and bsbs
   * tables, sigma4_bkbs being sigma_bkbs there. Written at [i][j] only, as init_one does not mirror. The force loop
   * reads epsilon, b and cut_c as they are then (epsilon_bsbs in the tetramer terms too), the squared cutoffs and LJ
   * factors stay those of pair_coeff, and init_one returns cut_bkbk_c as the cutoff of the next init.
   */
  override adaptParam(name: string): AdaptAccess | null {
    const nt = this.nt;
    if (name === 'epsilon_bsbs') {
      const eps = this.bb.eps, tEps = this.tEps;
      return { dim: 2, get: (i, j) => eps[i * nt + j], set: (i, j, v) => { eps[i * nt + j] = v; tEps[i * nt + j] = v; } };
    }
    if (name === 'sigma4_bkbs') return oxdnaTableAccess(this.bs.sigma, nt);
    const m = /^(epsilon|sigma|cut|b)_(bkbk|bkbs|bsbs)(_ast|_c)?$/.exec(name);
    if (!m || (m[1] === 'cut') !== (m[3] !== undefined)) return null;
    const t = m[2] === 'bkbk' ? this.bk : m[2] === 'bkbs' ? this.bs : this.bb;
    const field = m[1] === 'cut' ? (m[3] === '_ast' ? t.cutAst : t.cutC) : m[1] === 'epsilon' ? t.eps : m[1] === 'sigma' ? t.sigma : t.b;
    return oxdnaTableAccess(field, nt);
  }

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length !== 0) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n = this.nt * this.nt;
    this.set = new Uint8Array(n);
    this.bk = newTable(n);
    this.bs = newTable(n);
    this.bb = newTable(n);
    const n4 = libmPow(this.nt, 4);
    this.t4 = {
      sigma: new Float64Array(n4), cutAst: new Float64Array(n4), cutsqAst: new Float64Array(n4), cutC: new Float64Array(n4),
      cutsqC: new Float64Array(n4), b: new Float64Array(n4), lj14: new Float64Array(n4), lj24: new Float64Array(n4),
    };
    this.tEps = new Float64Array(n);
  }

  /** Fills one table entry (i, j) from the sigma, cut_ast and epsilon of a site kind. */
  protected fill(t: SiteTable, idx: number, eps: number, sigma: number, cutAst: number, b: number, cutC: number): void {
    t.eps[idx] = eps; t.sigma[idx] = sigma; t.cutAst[idx] = cutAst; t.b[idx] = b; t.cutC[idx] = cutC;
    t.lj1[idx] = 4.0 * eps * libmPow(sigma, 12.0);
    t.lj2[idx] = 4.0 * eps * libmPow(sigma, 6.0);
    t.cutsqAst[idx] = cutAst * cutAst;
    t.cutsqC[idx] = cutC * cutC;
  }

  /**
   * pair_oxdna_excv.cpp coeff(): "pair_coeff I J epsilon sigma cut_ast epsilon sigma cut_ast epsilon sigma cut_ast"
   * (bkbk, bkbs, bsbs), or the potential file form "pair_coeff I J file" (the line whose words are I, J, "excv").
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 3 && args.length !== 11) throw new StyleError('Incorrect args for pair coefficients in oxdna/excv');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    typeBounds(args[1], this.ntypes);
    let v: number[];
    if (args.length === 11) {
      v = args.slice(2).map((w, k) => parseNum(w, `oxdna/excv coefficient ${k + 1}`));
    } else {
      const text = ctx?.readFile(args[2], potentialFile('oxdna potential')) ?? '';
      let found: number[] | null = null;
      for (const w of potentialLines(text, [args[0], args[1]], 'excv')) {
        if (w.length < 3) continue;
        if (w[2] === 'excv') {
          if (w.length < 12) throw new StyleError(`Problem parsing oxdna potential file: excv needs 9 values in ${args[2]}`);
          found = w.slice(3, 12).map((x) => parseNum(x, 'excv'));
          break;
        }
      }
      if (!found) throw new StyleError(`No corresponding excv potential found in file ${args[2]} for pair type ${args[0]} ${args[1]}`);
      v = found;
    }
    this.fillSites(ilo, ihi, v);
    const [, , , , , , , sigBb, cutBbAst] = v;
    const nt = this.nt;
    const d3 = derived(sigBb, cutBbAst);
    // tetramer tables: entries [i][j][k][l] with i and l from 0 (terminal type) and j, k in the range
    for (let i = 0; i <= ihi; i++) {
      for (let j = ilo; j <= ihi; j++) {
        for (let k = ilo; k <= ihi; k++) {
          for (let l = 0; l <= ihi; l++) {
            const q = ((i * nt + j) * nt + k) * nt + l;
            const t = this.t4;
            t.sigma[q] = sigBb;
            t.cutAst[q] = cutBbAst;
            t.b[q] = d3.b;
            t.cutC[q] = d3.cutC;
            t.cutsqAst[q] = cutBbAst * cutBbAst;
            t.cutsqC[q] = d3.cutC * d3.cutC;
            t.lj14[q] = 4.0 * this.tEps[j * nt + k] * libmPow(t.sigma[q], 12.0);
            t.lj24[q] = 4.0 * this.tEps[j * nt + k] * libmPow(t.sigma[q], 6.0);
          }
        }
      }
    }
  }

  /**
   * The bkbk, bkbs and bsbs site tables of the range ilo..ihi from the nine values (epsilon, sigma, cut_ast of each
   * kind): every (i, j) of the range gets the same values (pair_oxdna_excv.cpp coeff()).
   */
  protected fillSites(ilo: number, ihi: number, v: number[]): void {
    const [epsBk, sigBk, cutBkAst, epsBs, sigBs, cutBsAst, epsBb, sigBb, cutBbAst] = v;
    const nt = this.nt;
    const d1 = derived(sigBk, cutBkAst), d2 = derived(sigBs, cutBsAst), d3 = derived(sigBb, cutBbAst);
    for (let i = ilo; i <= ihi; i++) {
      for (let j = ilo; j <= ihi; j++) {
        const idx = i * nt + j;
        this.fill(this.bk, idx, epsBk, sigBk, cutBkAst, d1.b, d1.cutC);
        this.fill(this.bs, idx, epsBs, sigBs, cutBsAst, d2.b, d2.cutC);
        this.fill(this.bb, idx, epsBb, sigBb, cutBbAst, d3.b, d3.cutC);
        this.tEps[idx] = epsBb;
        this.set[idx] = 1;
      }
    }
  }

  init(ctx: StyleContext): void {
    // pair_oxdna_excv.cpp init_one(): "Offset not supported in oxdna"
    if (this.shift) throw new StyleError('Offset not supported in oxdna');
    super.init(ctx);
  }

  /** pair_oxdna_excv.cpp init_one(): "Coefficient mixing not defined in oxdna"; returns cut_bkbk_c. */
  initOne(i: number, j: number): number {
    if (!this.set[i * this.nt + j]) throw new StyleError('Coefficient mixing not defined in oxdna');
    return this.bk.cutC[i * this.nt + j];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    if (!s.custom.has(OXDNA_5P) || !s.quat || !s.torque) throw new StyleError('pair_style oxdna/excv requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nlocal = pc.nlocal;
    const nt = this.nt;
    const constants = oxdnaConstants(s.units.style);
    const id5p = s.custom.get(OXDNA_5P)!.data, id3p = s.custom.get(OXDNA_3P)!.data;
    const special = pc.specialLJ;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    // local reference frames of owned atoms, copied to ghosts (fix OXDNA/LRF and its forward communication)
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    // site offsets of every atom (owned and ghost): backbone and base
    const rbk = new Float64Array(3 * nall), rbs = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      const e = L.subarray(9 * k, 9 * k + 3);
      this.backbone(constants, e, L.subarray(9 * k + 3, 9 * k + 6), L.subarray(9 * k + 6, 9 * k + 9), tmp); rbk[3 * k] = tmp[0]; rbk[3 * k + 1] = tmp[1]; rbk[3 * k + 2] = tmp[2];
      this.baseSiteOf(constants, type[k], e, tmp); rbs[3 * k] = tmp[0]; rbs[3 * k + 1] = tmp[1]; rbs[3 * k + 2] = tmp[2];
    }
    // identities by atom ID for the tetramer lookup; tag of a ghost is the tag of its owner
    const tagOf = (k: number): number => s.id[owner[k]];
    const typeOfTag = new Map<number, number>();
    for (let k = 0; k < nlocal; k++) typeOfTag.set(s.id[k], type[k]);
    const partner = (id: number): number => (id ? (typeOfTag.get(id) ?? 0) : 0);
    const fp = [0];
    let evdwl = 0;

    const tally = (a: number, b: number, e: number, fx: number, fy: number, fz: number, dx: number, dy: number, dz: number): void => {
      evdwl += e;
      if (eatom) { eatom[a] += 0.5 * e; eatom[b] += 0.5 * e; }
      if (vatom) {
        const w = [dx * fx, dy * fy, dz * fz, dx * fy, dx * fz, dy * fz];
        for (let c = 0; c < 6; c++) { vatom[6 * a + c] += 0.5 * w[c]; vatom[6 * b + c] += 0.5 * w[c]; }
      }
    };
    // force on a along delr (fpair * delr), the torques cross(ra, delf) on a and -cross(rb, delf) on b
    const apply = (a: number, b: number, dr: number[], fpair: number, ra: ArrayLike<number>, rb: ArrayLike<number>, delf: number[]): void => {
      delf[0] = dr[0] * fpair; delf[1] = dr[1] * fpair; delf[2] = dr[2] * fpair;
      f[3 * a] += delf[0]; f[3 * a + 1] += delf[1]; f[3 * a + 2] += delf[2];
      tq[3 * a] += ra[1] * delf[2] - ra[2] * delf[1];
      tq[3 * a + 1] += ra[2] * delf[0] - ra[0] * delf[2];
      tq[3 * a + 2] += ra[0] * delf[1] - ra[1] * delf[0];
      f[3 * b] -= delf[0]; f[3 * b + 1] -= delf[1]; f[3 * b + 2] -= delf[2];
      tq[3 * b] -= rb[1] * delf[2] - rb[2] * delf[1];
      tq[3 * b + 1] -= rb[2] * delf[0] - rb[0] * delf[2];
      tq[3 * b + 2] -= rb[0] * delf[1] - rb[1] * delf[0];
    };
    const delf = [0, 0, 0];
    const dbk = [0, 0, 0], dbs = [0, 0, 0], dsk = [0, 0, 0], dss = [0, 0, 0];
    const rtBk = [0, 0, 0], rtBs = [0, 0, 0];

    // the pairs in native's half-list order and orientation (which atom of a pair is a): the [atype][btype] entries
    // differ from [btype][atype] once fix adapt has written one of them
    const pairs = this.natPairs.pairs(s, pc.nb, pc.geom, list);
    let a = -1, atype = 0;
    let ra_bk = rbk.subarray(0, 0), ra_bs = rbs.subarray(0, 0);
    for (let p = 0; p < pairs.length; p += 2) {
      if (pairs[p] !== a) {
        a = pairs[p];
        atype = type[a];
        ra_bk = rbk.subarray(3 * a, 3 * a + 3);
        ra_bs = rbs.subarray(3 * a, 3 * a + 3);
        for (let d = 0; d < 3; d++) { rtBk[d] = x[3 * a + d] + ra_bk[d]; rtBs[d] = x[3 * a + d] + ra_bs[d]; }
      }
      {
        const jj = pairs[p + 1];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const btype = type[b];
        const rb_bk = rbk.subarray(3 * b, 3 * b + 3), rb_bs = rbs.subarray(3 * b, 3 * b + 3);
        for (let d = 0; d < 3; d++) {
          dbk[d] = rtBk[d] - (x[3 * b + d] + rb_bk[d]);
          dbs[d] = rtBk[d] - (x[3 * b + d] + rb_bs[d]);
          dsk[d] = rtBs[d] - (x[3 * b + d] + rb_bk[d]);
          dss[d] = rtBs[d] - (x[3 * b + d] + rb_bs[d]);
        }
        const rsqBk = dbk[0] * dbk[0] + dbk[1] * dbk[1] + dbk[2] * dbk[2];
        const rsqBkBs = dbs[0] * dbs[0] + dbs[1] * dbs[1] + dbs[2] * dbs[2];
        const rsqBsBk = dsk[0] * dsk[0] + dsk[1] * dsk[1] + dsk[2] * dsk[2];
        const rsqBsBs = dss[0] * dss[0] + dss[1] * dss[1] + dss[2] * dss[2];
        const ij = atype * nt + btype;
        // rel = x_a - x_b for the per-atom and per-atom virial tallies
        const rx = x[3 * a] - x[3 * b], ry = x[3 * a + 1] - x[3 * b + 1], rz = x[3 * a + 2] - x[3 * b + 2];

        if (rsqBk < this.bk.cutsqC[ij]) {
          let e = F3(rsqBk, this.bk.cutsqAst[ij], this.bk.cutC[ij], this.bk.lj1[ij], this.bk.lj2[ij], this.bk.eps[ij], this.bk.b[ij], fp);
          let fpair = fp[0];
          fpair *= factorLJ;
          e *= factorLJ;
          tally(a, b, e, dbk[0] * fpair, dbk[1] * fpair, dbk[2] * fpair, rx, ry, rz);
          apply(a, b, dbk, fpair, ra_bk, rb_bk, delf);
        }
        if (rsqBkBs < this.bs.cutsqC[ij]) {
          const e = F3(rsqBkBs, this.bs.cutsqAst[ij], this.bs.cutC[ij], this.bs.lj1[ij], this.bs.lj2[ij], this.bs.eps[ij], this.bs.b[ij], fp);
          const fpair = fp[0];
          tally(a, b, e, dbs[0] * fpair, dbs[1] * fpair, dbs[2] * fpair, rx, ry, rz);
          apply(a, b, dbs, fpair, ra_bk, rb_bs, delf);
        }
        if (rsqBsBk < this.bs.cutsqC[ij]) {
          const e = F3(rsqBsBk, this.bs.cutsqAst[ij], this.bs.cutC[ij], this.bs.lj1[ij], this.bs.lj2[ij], this.bs.eps[ij], this.bs.b[ij], fp);
          const fpair = fp[0];
          tally(a, b, e, dsk[0] * fpair, dsk[1] * fpair, dsk[2] * fpair, rx, ry, rz);
          apply(a, b, dsk, fpair, ra_bs, rb_bk, delf);
        }
        // bs-bs: the tetramer tables apply when a and b are 3'-5' neighbours in either order
        let eBsBs = 0;
        let fBsBs = 0;
        const ta = tagOf(a), tb = tagOf(b);
        if (ta === id3p[b] && tb === id5p[a]) {
          const t3 = partner(id3p[a]), t5 = partner(id5p[b]);
          const q = ((t3 * nt + atype) * nt + btype) * nt + t5;
          if (rsqBsBs < this.t4.cutsqC[q]) {
            eBsBs = F3(rsqBsBs, this.t4.cutsqAst[q], this.t4.cutC[q], this.t4.lj14[q], this.t4.lj24[q], this.tEps[atype * nt + btype], this.t4.b[q], fp);
            fBsBs = fp[0];
          }
        } else if (ta === id5p[b] && tb === id3p[a]) {
          const t3 = partner(id3p[b]), t5 = partner(id5p[a]);
          const q = ((t3 * nt + btype) * nt + atype) * nt + t5;
          if (rsqBsBs < this.t4.cutsqC[q]) {
            eBsBs = F3(rsqBsBs, this.t4.cutsqAst[q], this.t4.cutC[q], this.t4.lj14[q], this.t4.lj24[q], this.tEps[btype * nt + atype], this.t4.b[q], fp);
            fBsBs = fp[0];
          }
        } else if (rsqBsBs < this.bb.cutsqC[ij]) {
          eBsBs = F3(rsqBsBs, this.bb.cutsqAst[ij], this.bb.cutC[ij], this.bb.lj1[ij], this.bb.lj2[ij], this.bb.eps[ij], this.bb.b[ij], fp);
          fBsBs = fp[0];
        }
        if (eBsBs !== 0) {
          tally(a, b, eBsBs, dss[0] * fBsBs, dss[1] * fBsBs, dss[2] * fBsBs, rx, ry, rz);
          apply(a, b, dss, fBsBs, ra_bs, rb_bs, delf);
        }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}
