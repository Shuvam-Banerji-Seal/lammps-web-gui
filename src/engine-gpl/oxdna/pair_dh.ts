/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): src/CG-DNA/pair_oxdna2_dh.cpp and
 * pair_oxdna2_dh.h (class PairOxdna2Dh, pair_style oxdna2/dh: the Debye-Huckel electrostatics between the
 * backbone sites of the oxDNA2 model), and src/CG-DNA/pair_oxrna2_dh.cpp and pair_oxrna2_dh.h (class
 * PairOxrna2Dh, pair_style oxrna2/dh: the same term with the oxRNA2 backbone site). Original authors (from the
 * headers): Oliver Henrich (University of Strathclyde, Glasgow) and others of the CG-DNA package; the screening
 * and prefactors are those of Snodin et al. (J Chem Phys 142, 234901, 2015). LAMMPS is Copyright (2003) Sandia
 * Corporation, distributed under the GNU General Public License, version 2 (GPL-2.0); this file is GPL-2.0 too
 * (src/engine-gpl/LICENSE).
 *
 * The per-atom effective charge qeff is an internal per-atom property of atom_style oxdna. pair_coeff sets it
 * on every atom (1, or 0.5 at strand ends with "half_charged_ends yes"), as pair_oxdna2_dh.cpp coeff() does.
 */
import { Pair, StyleError, typeBounds, type PairCompute, type StyleContext, potentialFile } from '../../engine/force/types';
import { NEIGHMASK, SBBITS } from '../../engine/neighbor';
import { OXDNA_3P, OXDNA_5P, OXDNA_QEFF, isEllipsoid } from '../../engine/atoms';
import { parseNum, logical as readLogical } from '../../engine/force/util';
import { type OxdnaConstants, backboneOxdna2, backboneOxrna2Dh, lrfOf, oxdnaConstants, potentialLines } from './common';
import { exp as libmExp } from '../libm/exp';

/** utils::logical of LAMMPS for the half_charged_ends keyword. */
const logical = (w: string): boolean => readLogical(w);

type Tab = Float64Array;

export class PairOxdna2Dh extends Pair {
  readonly name: string = 'oxdna2/dh';
  virialFdotr = true;
  /** the backbone site rule (the oxRNA2 subclass overrides it) */
  protected backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxdna2(c, e1, e2, e3, out);
  }
  private nt = 0;
  private kappa!: Tab; private qpf!: Tab; private bDh!: Tab; private cutAst!: Tab; private cutC!: Tab;
  private set!: Uint8Array;

  settings(args: string[], _ctx: StyleContext): void {
    if (args.length) throw new StyleError('Illegal pair_style command');
  }

  allocate(ntypes: number): void {
    super.allocate(ntypes);
    this.nt = ntypes + 1;
    const n = this.nt * this.nt;
    this.kappa = new Float64Array(n); this.qpf = new Float64Array(n); this.bDh = new Float64Array(n);
    this.cutAst = new Float64Array(n); this.cutC = new Float64Array(n);
    this.set = new Uint8Array(n);
  }

  /**
   * pair_oxdna2_dh.cpp coeff(): "pair_coeff I J T rhos_salt qeff [half_charged_ends yes|no]", or with a potential
   * file in place of qeff ("I J dh qeff" lines). lambda = C1 sqrt(T / 0.1 / rhos); kappa = 1 / lambda;
   * qeff_pf = C2 qeff^2; cut_ast = 3 lambda; cut_c = cut_ast (cut_ast + 3 lambda) / (cut_ast + lambda).
   */
  coeff(args: string[], ctx?: StyleContext): void {
    if (args.length !== 5 && args.length !== 7) throw new StyleError('Incorrect args for pair coefficients in oxdna2/dh');
    const [ilo, ihi] = typeBounds(args[0], this.ntypes);
    const [jlo, jhi] = typeBounds(args[1], this.ntypes);
    const T = parseNum(args[2], 'T');
    const rhos = parseNum(args[3], 'rhos');
    let qeffOne: number;
    if (/^[a-zA-Z0-9_]*\.cgdna$/.test(args[4])) {
      const text = ctx?.readFile(args[4], potentialFile('oxdna potential')) ?? '';
      let found: number | null = null;
      for (const w of potentialLines(text, [args[0], args[1]], 'dh')) {
        if (w.length < 4) continue;
        if (w[2] === 'dh') { found = parseNum(w[3], 'qeff'); break; }
      }
      if (found === null) throw new StyleError(`No corresponding dh potential found in file ${args[4]} for pair type ${args[0]} ${args[1]}`);
      qeffOne = found;
    } else {
      qeffOne = parseNum(args[4], 'qeff');
    }
    // pair_oxdna2_dh.cpp: "half_charged_ends" is read only when it is the 6th word; the flag is reset per call
    let halfCharged = false;
    if (args.length === 7) {
      if (args[5] !== 'half_charged_ends') throw new StyleError(`pair_coeff oxdna2/dh: unknown keyword '${args[5]}' (expected half_charged_ends)`);
      halfCharged = logical(args[6]);
    }
    const c = oxdnaConstants(ctx?.s?.units.style ?? 'lj');
    const lambda = c.lambda_dh_one_prefactor * Math.sqrt(T / 0.1 / rhos);
    const kappaOne = 1.0 / lambda;
    const qpfOne = c.qeff_dh_pf_one_prefactor * qeffOne * qeffOne;
    const cutAstOne = 3.0 * lambda;
    const bOne = (libmExp(-cutAstOne / lambda) * qpfOne * (cutAstOne + lambda) * (cutAstOne + lambda))
      / (4.0 * cutAstOne * cutAstOne * cutAstOne * lambda * lambda);
    const cutCOne = cutAstOne * (cutAstOne + 3.0 * lambda) / (cutAstOne + lambda);
    const nt = this.nt;
    for (let i = ilo; i <= ihi; i++) {
      for (let j = Math.max(jlo, i); j <= jhi; j++) {
        const q = i * nt + j;
        this.kappa[q] = kappaOne; this.qpf[q] = qpfOne; this.bDh[q] = bOne;
        this.cutAst[q] = cutAstOne; this.cutC[q] = cutCOne;
        this.set[q] = 1;
      }
    }
    // pair_oxdna2_dh.cpp coeff(): qeff of every atom is set here (1, or 0.5 at strand ends)
    const s = ctx?.s;
    if (!s) throw new StyleError('pair_coeff oxdna2/dh needs the atoms (read_data or create_atoms first)');
    const qeff = s.custom.get(OXDNA_QEFF);
    const id3p = s.custom.get(OXDNA_3P), id5p = s.custom.get(OXDNA_5P);
    if (!qeff || !id3p || !id5p) throw new StyleError('pair_style oxdna2/dh requires atom_style hybrid bond ellipsoid oxdna');
    for (let k = 0; k < s.n; k++) {
      qeff.data[k] = 1.0;
      if (halfCharged && (id3p.data[k] === 0 || id5p.data[k] === 0)) qeff.data[k] = 0.5;
    }
  }

  init(ctx: StyleContext): void {
    if (this.shift) throw new StyleError('Offset not supported in oxDNA');
    super.init(ctx);
  }

  /** pair_oxdna2_dh.cpp init_one(): symmetric copies, returns cut_dh_c[i][j]. */
  initOne(i: number, j: number): number {
    const nt = this.nt;
    if (!this.set[i * nt + j]) throw new StyleError('Coefficient mixing not defined in oxDNA');
    const q = i * nt + j, r = j * nt + i;
    this.kappa[r] = this.kappa[q]; this.qpf[r] = this.qpf[q]; this.bDh[r] = this.bDh[q];
    this.cutAst[r] = this.cutAst[q]; this.cutC[r] = this.cutC[q];
    return this.cutC[q];
  }

  compute(pc: PairCompute): void {
    const s = pc.s;
    const qeffArr = s.custom.get(OXDNA_QEFF);
    if (!qeffArr || !s.quat || !s.torque) throw new StyleError('pair_style oxdna2/dh requires atom_style hybrid bond ellipsoid oxdna');
    const list = pc.half!;
    const { x, f, type } = pc;
    const owner = pc.nb.owner;
    const nall = pc.nall;
    const nt = this.nt;
    const constants = oxdnaConstants(s.units.style);
    const special = pc.specialLJ;
    const qeff = qeffArr.data;
    const tq = new Float64Array(3 * nall);
    const eatom = pc.eatom, vatom = pc.vatom;
    const L = new Float64Array(9 * nall);
    for (let k = 0; k < nall; k++) {
      const o = owner[k];
      if (isEllipsoid(s, o)) lrfOf(s.quat, 4 * o, L, 9 * k);
    }
    const rs = new Float64Array(3 * nall);
    const tmp = [0, 0, 0];
    for (let k = 0; k < nall; k++) {
      this.backbone(constants, L.subarray(9 * k, 9 * k + 3), L.subarray(9 * k + 3, 9 * k + 6), L.subarray(9 * k + 6, 9 * k + 9), tmp);
      rs[3 * k] = tmp[0]; rs[3 * k + 1] = tmp[1]; rs[3 * k + 2] = tmp[2];
    }
    let evdwl = 0;
    const delr = [0, 0, 0], delf = [0, 0, 0], delta = [0, 0, 0], deltb = [0, 0, 0];
    const rtmp = [0, 0, 0];
    const cross = (u: readonly number[], v: readonly number[], out: number[]) => {
      out[0] = u[1] * v[2] - u[2] * v[1]; out[1] = u[2] * v[0] - u[0] * v[2]; out[2] = u[0] * v[1] - u[1] * v[0];
    };
    for (let a = 0; a < list.inum; a++) {
      const atype = type[a];
      for (let d = 0; d < 3; d++) rtmp[d] = x[3 * a + d] + rs[3 * a + d];
      const jstart = list.firstneigh[a], jend = jstart + list.numneigh[a];
      for (let kk = jstart; kk < jend; kk++) {
        const jj = list.neighbors[kk];
        const factorLJ = special[jj >>> SBBITS];
        const b = jj & NEIGHMASK;
        const btype = type[b];
        for (let d = 0; d < 3; d++) delr[d] = rtmp[d] - x[3 * b + d] - rs[3 * b + d];
        const rsq = delr[0] * delr[0] + delr[1] * delr[1] + delr[2] * delr[2];
        const ij = atype * nt + btype;
        if (rsq > this.cutC[ij] * this.cutC[ij]) continue;
        const r = Math.sqrt(rsq);
        const rinv = 1.0 / r;
        const qa = qeff[a], qb = qeff[owner[b]];
        const kap = this.kappa[ij], qpf = this.qpf[ij];
        let fpair: number, e: number;
        if (r <= this.cutAst[ij]) {
          fpair = qa * qb * qpf * libmExp(-kap * r) * (kap + rinv) * rinv * rinv;
          e = qa * qb * qpf * libmExp(-kap * r) * rinv;
        } else {
          fpair = 2.0 * qa * qb * this.bDh[ij] * (this.cutC[ij] - r) * rinv;
          e = qa * qb * this.bDh[ij] * (r - this.cutC[ij]) * (r - this.cutC[ij]);
        }
        fpair *= factorLJ;
        e *= factorLJ;
        delf[0] = delr[0] * fpair; delf[1] = delr[1] * fpair; delf[2] = delr[2] * fpair;
        for (let d = 0; d < 3; d++) f[3 * a + d] += delf[d];
        cross([rs[3 * a], rs[3 * a + 1], rs[3 * a + 2]], delf, delta);
        for (let d = 0; d < 3; d++) tq[3 * a + d] += delta[d];
        for (let d = 0; d < 3; d++) f[3 * b + d] -= delf[d];
        cross([rs[3 * b], rs[3 * b + 1], rs[3 * b + 2]], delf, deltb);
        for (let d = 0; d < 3; d++) tq[3 * b + d] -= deltb[d];
        evdwl += e;
        if (eatom) { eatom[a] += 0.5 * e; eatom[b] += 0.5 * e; }
        if (vatom) {
          const rx = x[3 * a] - x[3 * b], ry = x[3 * a + 1] - x[3 * b + 1], rz = x[3 * a + 2] - x[3 * b + 2];
          const w = [rx * delf[0], ry * delf[1], rz * delf[2], rx * delf[1], rx * delf[2], ry * delf[2]];
          for (let c = 0; c < 6; c++) { vatom[6 * a + c] += 0.5 * w[c]; vatom[6 * b + c] += 0.5 * w[c]; }
        }
      }
    }
    pc.nb.reverseSum(tq, 3, s.torque);
    pc.acc.evdwl += evdwl;
  }
}

/** pair_style oxrna2/dh (PairOxrna2Dh): the oxRNA2 backbone site. */
export class PairOxrna2Dh extends PairOxdna2Dh {
  override readonly name: string = 'oxrna2/dh';
  protected override backbone(c: OxdnaConstants, e1: ArrayLike<number>, e2: ArrayLike<number>, e3: ArrayLike<number>, out: number[]): void {
    backboneOxrna2Dh(c, e1, e2, e3, out);
  }
}
